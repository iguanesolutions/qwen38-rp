package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"net/url"
	"path"

	"github.com/hekmon/httplog/v3"
)

// rewriteRequestURL rewrites request URL and Host header for proxying to backend
func rewriteRequestURL(req *http.Request, target *url.URL) {
	req.URL.Scheme = target.Scheme
	req.URL.Host = target.Host
	req.Host = target.Host
	req.URL.Path = path.Join(target.Path, req.URL.Path)
	req.URL.RawPath = ""
	targetQuery := target.RawQuery
	if targetQuery == "" || req.URL.RawQuery == "" {
		req.URL.RawQuery = targetQuery + req.URL.RawQuery
	} else {
		req.URL.RawQuery = targetQuery + "&" + req.URL.RawQuery
	}
}

// maxRequestBodySize is the maximum allowed request body size (100 MB).
// This is a DoS safeguard, not a functional limit — conversation histories
// with base64-encoded images can legitimately be very large.
const maxRequestBodySize = 100 << 20 // 100 MB

// maxSSEEventSize is the maximum allowed size for a single buffered SSE event (10 MB).
// This prevents unbounded memory growth if the backend sends a malformed stream
// without proper event delimiters.
const maxSSEEventSize = 10 << 20 // 10 MB

// hopByHopHeaders lists headers that must not be forwarded by proxies (RFC 7230 §6.1)
var hopByHopHeaders = []string{
	"Connection",
	"Keep-Alive",
	"Proxy-Authenticate",
	"Proxy-Authorization",
	"Te",
	"Trailer",
	"Transfer-Encoding",
	"Upgrade",
}

// stripHopByHopHeaders removes hop-by-hop headers from the request before proxying
func stripHopByHopHeaders(req *http.Request) {
	for _, h := range hopByHopHeaders {
		req.Header.Del(h)
	}
}

// applySamplingParams applies the official sampling defaults of a Qwen mode according to policy:
// off never touches the request, fill only sets the keys the client omitted, and enforce always
// overwrites them. The applied values are logged at DEBUG level, never the request body.
func applySamplingParams(data map[string]any, samplingParams map[string]any, mode string,
	policy SamplingPolicy, logger *slog.Logger) {
	if policy == SamplingPolicyOff {
		logger.Debug("sampling parameters disabled by policy", slog.String("mode", mode))
		return
	}
	attrs := make([]any, 0, 2+len(samplingParams))
	attrs = append(attrs,
		slog.String("mode", mode),
		slog.String("policy", string(policy)),
	)
	for k, v := range samplingParams {
		oldValue, exists := data[k]
		if exists && policy != SamplingPolicyEnforce {
			attrs = append(attrs, slog.Group(k,
				slog.Any("old_value", oldValue),
				slog.Any("new_value", oldValue),
			))
			continue
		}
		data[k] = v
		attrs = append(attrs, slog.Group(k,
			slog.Any("old_value", oldValue),
			slog.Any("new_value", v),
		))
	}
	logger.Debug("applied sampling parameters", attrs...)
}

// copyHeaders copies response headers from backend response to the client ResponseWriter,
// stripping hop-by-hop headers that must not be forwarded by proxies.
func copyHeaders(w http.ResponseWriter, resp *http.Response) {
	for _, h := range hopByHopHeaders {
		resp.Header.Del(h)
	}
	for header, values := range resp.Header {
		for _, value := range values {
			w.Header().Add(header, value)
		}
	}
}

// httpError writes an OpenAI-compatible JSON error response
func httpError(ctx context.Context, w http.ResponseWriter, statusCode int) {
	reqID := ctx.Value(httplog.ReqIDKey)
	message := fmt.Sprintf("%s - check qwen38-rp logs for more details (request id #%v)",
		http.StatusText(statusCode),
		reqID,
	)
	writeErrorResponse(w, statusCode, message)
}

// httpErrorWithDetail writes an OpenAI-compatible JSON error response with an explicit
// client-facing detail message, for errors the client can act upon.
func httpErrorWithDetail(ctx context.Context, w http.ResponseWriter, statusCode int, detail string) {
	reqID := ctx.Value(httplog.ReqIDKey)
	message := fmt.Sprintf("%s - %s (request id #%v)",
		http.StatusText(statusCode),
		detail,
		reqID,
	)
	writeErrorResponse(w, statusCode, message)
}

func writeErrorResponse(w http.ResponseWriter, statusCode int, message string) {
	errResp := map[string]any{
		"error": map[string]any{
			"message": message,
			"type":    http.StatusText(statusCode),
			"code":    statusCode,
		},
	}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(statusCode)
	if err := json.NewEncoder(w).Encode(errResp); err != nil {
		logger.Error("failed to write error response", slog.Any("error", err))
	}
}

// readBodyStatusCode returns the appropriate HTTP status code for a body read error.
// Returns 413 for MaxBytesError, 500 for everything else.
func readBodyStatusCode(err error) int {
	var maxBytesErr *http.MaxBytesError
	if errors.As(err, &maxBytesErr) {
		return http.StatusRequestEntityTooLarge
	}
	return http.StatusInternalServerError
}

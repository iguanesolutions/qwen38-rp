package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"io"
	"log/slog"
	"maps"
	"net/http"
	"net/url"
	"syscall"

	"github.com/hekmon/httplog/v3"
)

func tokenize(httpCli *http.Client, target *url.URL,
	cfg Config, profiles map[string]ModelProfile) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		logger := logger.With(httplog.GetReqIDSLogAttr(r.Context()))
		ctx := r.Context()
		servedModel := cfg.ServedModelName

		// Read request body
		// Full body materialization is required to rewrite JSON fields in-place.
		// This is acceptable for Qwen 3.8's text-only outputs.
		r.Body = http.MaxBytesReader(w, r.Body, maxRequestBodySize)
		requestBody, err := io.ReadAll(r.Body)
		if err != nil {
			logger.Error("failed to read body", slog.String("error", err.Error()))
			httpError(ctx, w, readBodyStatusCode(err))
			return
		}

		// Parse as generic map to read and modify fields
		var reqData map[string]any
		if err := json.Unmarshal(requestBody, &reqData); err != nil {
			logger.Error("failed to parse body as JSON", slog.String("error", err.Error()))
			httpError(ctx, w, http.StatusBadRequest)
			return
		}

		modelName, _ := reqData["model"].(string)

		// Act based on the model field
		switch modelName {
		case "":
			// by default vllm accept a empty model name as it serves only one model
			logger.Debug("tokenize request received without a model name, accept it anyway and set the actual served model name",
				slog.String("served_model", servedModel),
			)
			reqData["model"] = servedModel
		default:
			profile, valid := profiles[modelName]
			if !valid {
				logger.Error("tokenize request received with an invalid model name",
					slog.String("requested_model", modelName),
					slog.String("served_model", servedModel),
				)
				httpError(ctx, w, http.StatusBadRequest)
				return
			}

			logger.Debug("tokenize request received with a valid virtual model name",
				slog.String("virtual_model", modelName),
				slog.String("served_model", servedModel),
			)

			// Resolve what the rendered prompt will contain, so the injected kwargs and
			// the partial-count caveat agree with the chat path.
			var (
				mode             string // resolved auto mode
				think            bool
				preserveThinking bool
			)
			if profile.Auto {
				// The auto model is validated like on the chat path: a request the chat
				// path refuses to serve must not get a token count either.
				var err error
				if mode, err = resolveAutoMode(reqData); err != nil {
					logger.Error("invalid chat_template_kwargs", slog.String("error", err.Error()))
					httpError(ctx, w, http.StatusBadRequest)
					return
				}
				if detail := autoModeRejection(mode, cfg.NoInstruct); detail != "" {
					logger.Warn("auto model resolved to instruct mode but instruct is disabled on this deployment",
						slog.String("virtual_model", modelName),
					)
					httpErrorWithDetail(ctx, w, http.StatusBadRequest, detail)
					return
				}
				think = mode == modeThinking
				preserveThinking = cfg.AutoPreserveThinking
			} else {
				think = profile.Think
				preserveThinking = profile.PreserveThinking
			}

			// Apply the virtual model contract: inject chat_template_kwargs so the
			// rendered template matches what the chat completions path would produce.
			kwargsMap, err := chatTemplateKwargs(reqData)
			if err != nil {
				logger.Error("invalid chat_template_kwargs", slog.String("error", err.Error()))
				httpError(ctx, w, http.StatusBadRequest)
				return
			}
			if profile.Auto {
				// preserve_thinking is pinned exactly like on the chat path.
				maps.Copy(kwargsMap, autoChatTemplateKwargs(cfg))
				// enable_thinking, on the other hand, must be pinned here and nowhere
				// else: vLLM only derives enable_thinking from reasoning_effort in
				// build_chat_params, which /tokenize never runs. Leaving it out would
				// render a thinking prompt for an instruct request and over-count.
				kwargsMap["enable_thinking"] = think
			} else {
				kwargsMap["enable_thinking"] = think
				kwargsMap["preserve_thinking"] = preserveThinking
			}

			// vLLM's chat path also forwards reasoning_effort to the template, which
			// renders an effort directive into the prompt (the default rendering is the
			// xhigh one). /tokenize only sees it through chat_template_kwargs, so pin it
			// for parity — but never inject a value the template rejects, or /tokenize
			// would fail where the chat path succeeds.
			if effort, ok := templateEffort(profile, reqData); ok && think {
				kwargsMap["reasoning_effort"] = effort
			}

			// vLLM's /tokenize ignores message.reasoning_content while the chat template
			// re-renders it as thinking history. Mirror it into the field /tokenize does
			// render, so preserved thinking is counted.
			if think && preserveThinking {
				mirrorReasoningContent(reqData["messages"])
			}

			reqData["model"] = servedModel
		}

		// Marshal the modified request body
		if requestBody, err = json.Marshal(reqData); err != nil {
			logger.Error("failed to marshal modified request body", slog.String("error", err.Error()))
			httpError(ctx, w, http.StatusInternalServerError)
			return
		}

		// Create a new request with the modified body
		outreq := r.Clone(ctx)
		rewriteRequestURL(outreq, target)
		stripHopByHopHeaders(outreq)
		outreq.Body = io.NopCloser(bytes.NewReader(requestBody))
		outreq.ContentLength = int64(len(requestBody))
		outreq.RequestURI = ""
		outResp, err := httpCli.Do(outreq)
		if err != nil {
			logger.Error("failed to send upstream request", slog.Any("error", err))
			switch {
			case errors.Is(err, syscall.ECONNREFUSED):
				httpError(ctx, w, http.StatusBadGateway)
			default:
				httpError(ctx, w, http.StatusInternalServerError)
			}
			return
		}
		defer outResp.Body.Close()
		modifiedRequests.Add(1)

		// Copy response as is
		copyHeaders(w, outResp)
		w.WriteHeader(outResp.StatusCode)
		if _, err = io.Copy(w, outResp.Body); err != nil {
			logger.Error("failed to write response", slog.String("error", err.Error()))
		}
	}
}

// templateEfforts are the reasoning_effort values the Qwen 3.8 chat template accepts.
// Other values (high, minimal, max, ...) are normalized by vLLM through
// --reasoning-effort-map before the template sees them; injecting them as a template
// kwarg makes /tokenize answer 400 where the chat path succeeds.
var templateEfforts = map[string]struct{}{
	"low": {}, "medium": {}, "xhigh": {}, "none": {},
}

// templateEffort returns the reasoning_effort the chat path renders into the prompt:
// a pre-configured profile pins it, otherwise the client decides. The second return
// value is false when the value is not one the template accepts.
func templateEffort(profile ModelProfile, data map[string]any) (string, bool) {
	effort := profile.Effort
	if effort == "" {
		effort, _ = data["reasoning_effort"].(string)
	}
	if _, ok := templateEfforts[effort]; !ok {
		return "", false
	}
	return effort, true
}

// mirrorReasoningContent mirrors message.reasoning_content into message.reasoning, the
// field vLLM's /tokenize renders as thinking history. reasoning_content is kept as-is, and
// an existing reasoning field is never overwritten, so mirroring cannot double-count.
func mirrorReasoningContent(rawMessages any) {
	messages, ok := rawMessages.([]any)
	if !ok {
		return
	}
	for _, rawMessage := range messages {
		message, ok := rawMessage.(map[string]any)
		if !ok {
			continue
		}
		if _, exists := message["reasoning"]; exists {
			continue
		}
		if reasoning, ok := message["reasoning_content"].(string); ok && reasoning != "" {
			message["reasoning"] = reasoning
		}
	}
}

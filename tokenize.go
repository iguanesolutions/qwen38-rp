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

			// The auto model is validated like on the chat path: the mode itself is not
			// needed to render the prompt (preserve_thinking is mode-independent), but a
			// request the chat path refuses to serve must not get a token count either.
			if profile.Auto {
				mode, err := resolveAutoMode(reqData)
				if err != nil {
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
				// Mirror the chat path exactly: pin preserve_thinking (otherwise the
				// template default of true would re-render historical thinking and
				// overcount tokens), but never enable_thinking, which the backend
				// derives from reasoning_effort.
				maps.Copy(kwargsMap, autoChatTemplateKwargs(cfg))
			} else {
				kwargsMap["enable_thinking"] = profile.Think
				kwargsMap["preserve_thinking"] = profile.PreserveThinking
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

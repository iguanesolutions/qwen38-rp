package main

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"testing"
	"time"

	"github.com/hekmon/httplog/v3"
)

// These tests need a live vLLM backend and are skipped unless one is configured:
//
//	QWEN38RP_TEST_TARGET_URL=http://127.0.0.1:8000 go test -run Integration -v .
//
// QWEN38RP_TEST_SERVED_MODEL overrides the model name (by default it is discovered from
// the backend's /v1/models). QWEN38RP_TEST_API_KEY sets the bearer token; VLLM_API_KEY is
// used as a fallback, so `set -a; . ./.backend.env; set +a` is enough.
//
// They assert count equality between /v1/chat/completions (usage.prompt_tokens) and
// /tokenize (count), never absolute numbers, so they stay portable across tokenizers and
// model revisions.

// bearerTransport injects an Authorization header into outgoing backend calls.
type bearerTransport struct {
	key  string
	base http.RoundTripper
}

func (t bearerTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	if t.key == "" {
		return t.base.RoundTrip(req)
	}
	req = req.Clone(req.Context())
	req.Header.Set("Authorization", "Bearer "+t.key)
	return t.base.RoundTrip(req)
}

func integrationAPIKey() string {
	if key := os.Getenv("QWEN38RP_TEST_API_KEY"); key != "" {
		return key
	}
	return os.Getenv("VLLM_API_KEY")
}

func integrationTarget(t *testing.T) (*url.URL, string, *http.Client) {
	t.Helper()
	rawTarget := os.Getenv("QWEN38RP_TEST_TARGET_URL")
	if rawTarget == "" {
		t.Skip("QWEN38RP_TEST_TARGET_URL is not set, skipping live backend integration test")
	}
	target, err := url.Parse(rawTarget)
	if err != nil {
		t.Fatalf("parse QWEN38RP_TEST_TARGET_URL: %v", err)
	}
	client := &http.Client{
		Timeout:   2 * time.Minute,
		Transport: bearerTransport{key: integrationAPIKey(), base: http.DefaultTransport},
	}

	servedModel := os.Getenv("QWEN38RP_TEST_SERVED_MODEL")
	if servedModel == "" {
		servedModel = integrationDiscoverModel(t, client, target)
	}
	return target, servedModel, client
}

// integrationDiscoverModel returns the first model id served by the backend.
func integrationDiscoverModel(t *testing.T, client *http.Client, target *url.URL) string {
	t.Helper()
	resp, err := client.Get(target.JoinPath("/v1/models").String())
	if err != nil {
		t.Fatalf("fetch /v1/models from backend: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("fetch /v1/models from backend: status %d", resp.StatusCode)
	}
	var modelsResp struct {
		Data []struct {
			ID string `json:"id"`
		} `json:"data"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&modelsResp); err != nil {
		t.Fatalf("decode /v1/models response: %v", err)
	}
	if len(modelsResp.Data) == 0 {
		t.Fatal("backend serves no model")
	}
	return modelsResp.Data[0].ID
}

// integrationPost sends a JSON request through a proxy handler and returns the recorder.
func integrationPost(t *testing.T, handler http.HandlerFunc, path string, request map[string]any) *httptest.ResponseRecorder {
	t.Helper()
	payload, err := json.Marshal(request)
	if err != nil {
		t.Fatalf("marshal request: %v", err)
	}
	req := httptest.NewRequest(http.MethodPost, path, bytes.NewReader(payload))
	req.Header.Set("Content-Type", "application/json")
	req = req.WithContext(context.WithValue(req.Context(), httplog.ReqIDKey, uint64(1)))
	if key := integrationAPIKey(); key != "" {
		// The handlers forward incoming headers, so the backend sees the bearer token.
		req.Header.Set("Authorization", "Bearer "+key)
	}
	rec := httptest.NewRecorder()
	handler(rec, req)
	return rec
}

// integrationChatPromptTokens returns usage.prompt_tokens for a non-streaming chat call.
func integrationChatPromptTokens(t *testing.T, handler http.HandlerFunc, request map[string]any) int {
	t.Helper()
	rec := integrationPost(t, handler, "/v1/chat/completions", request)
	if rec.Code != http.StatusOK {
		t.Fatalf("chat completion: status %d, body %s", rec.Code, rec.Body.String())
	}
	var chatResp struct {
		Usage struct {
			PromptTokens int `json:"prompt_tokens"`
		} `json:"usage"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &chatResp); err != nil {
		t.Fatalf("decode chat response %q: %v", rec.Body.String(), err)
	}
	if chatResp.Usage.PromptTokens == 0 {
		t.Fatalf("chat response carries no usage.prompt_tokens: %s", rec.Body.String())
	}
	return chatResp.Usage.PromptTokens
}

// integrationTokenizeCount returns the count field of a /tokenize response.
func integrationTokenizeCount(t *testing.T, handler http.HandlerFunc, request map[string]any) int {
	t.Helper()
	rec := integrationPost(t, handler, "/tokenize", request)
	if rec.Code != http.StatusOK {
		t.Fatalf("tokenize: status %d, body %s", rec.Code, rec.Body.String())
	}
	var tokenizeResp struct {
		Count int `json:"count"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &tokenizeResp); err != nil {
		t.Fatalf("decode tokenize response %q: %v", rec.Body.String(), err)
	}
	return tokenizeResp.Count
}

func integrationConfig(servedModel string, autoPreserveThinking bool) Config {
	cfg := autoTestConfig("fill")
	cfg.ServedModelName = servedModel
	cfg.AutoPreserveThinking = autoPreserveThinking
	return cfg
}

// TestIntegrationTokenizeMatchesChat asserts count equality against a live backend.
func TestIntegrationTokenizeMatchesChat(t *testing.T) {
	target, servedModel, client := integrationTarget(t)

	oneTurn := []any{map[string]any{"role": "user", "content": "Explain what a prefix cache is in one sentence."}}
	thinkingHistory := []any{
		map[string]any{"role": "user", "content": "Explain what a prefix cache is in one sentence."},
		map[string]any{
			"role":              "assistant",
			"content":           "It reuses previously computed KV entries for a shared prompt prefix.",
			"reasoning_content": "The user wants one sentence. Focus on KV reuse across requests.",
		},
		map[string]any{"role": "user", "content": "And what breaks it?"},
	}

	tests := []struct {
		name                 string
		autoPreserveThinking bool
		messages             []any
		effort               string
	}{
		{
			name:     "one turn, effort none",
			messages: oneTurn,
			effort:   "none",
		},
		{
			name:     "one turn, thinking",
			messages: oneTurn,
		},
		{
			name:     "one turn, effort low renders the effort directive",
			messages: oneTurn,
			effort:   "low",
		},
		{
			name:     "one turn, effort medium renders the effort directive",
			messages: oneTurn,
			effort:   "medium",
		},
		{
			name:                 "thinking history, preserve thinking",
			autoPreserveThinking: true,
			messages:             thinkingHistory,
			effort:               "low",
		},
		{
			name:     "thinking history, preserve disabled",
			messages: thinkingHistory,
			effort:   "low",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := integrationConfig(servedModel, tt.autoPreserveThinking)
			profiles := buildModelProfiles(cfg.VirtualModelPrefix, false, false, true)

			request := map[string]any{
				"model":    cfg.VirtualModelPrefix + "-auto",
				"messages": tt.messages,
				// Keep generation cost near zero; only prompt_tokens is asserted.
				"max_tokens": 1,
			}
			if tt.effort != "" {
				request["reasoning_effort"] = tt.effort
			}

			chatTokens := integrationChatPromptTokens(t, transform(client, target, cfg, profiles), request)
			tokenizeTokens := integrationTokenizeCount(t, tokenize(client, target, cfg, profiles), request)
			if chatTokens != tokenizeTokens {
				t.Errorf("count mismatch: chat prompt_tokens=%d, tokenize count=%d", chatTokens, tokenizeTokens)
			}
		})
	}
}

// TestIntegrationTokenizePreserveIsLowerThanPreserveThinking documents case 4: with
// preserve_thinking disabled the preserved-history prompt is strictly smaller.
func TestIntegrationTokenizePreserveIsLowerThanPreserveThinking(t *testing.T) {
	target, servedModel, client := integrationTarget(t)

	request := map[string]any{
		"model": servedModel,
		"messages": []any{
			map[string]any{"role": "user", "content": "Explain what a prefix cache is in one sentence."},
			map[string]any{
				"role":              "assistant",
				"content":           "It reuses previously computed KV entries for a shared prompt prefix.",
				"reasoning_content": "The user wants one sentence. Focus on KV reuse across requests.",
			},
			map[string]any{"role": "user", "content": "And what breaks it?"},
		},
		"max_tokens": 1,
	}

	preserveCfg := integrationConfig(servedModel, true)
	noPreserveCfg := integrationConfig(servedModel, false)
	profiles := buildModelProfiles(preserveCfg.VirtualModelPrefix, false, false, true)

	autoRequest := map[string]any{
		"model":            preserveCfg.VirtualModelPrefix + "-auto",
		"messages":         request["messages"],
		"max_tokens":       1,
		"reasoning_effort": "low",
	}
	preserveTokens := integrationChatPromptTokens(t, transform(client, target, preserveCfg, profiles), autoRequest)
	noPreserveTokens := integrationChatPromptTokens(t, transform(client, target, noPreserveCfg, profiles), autoRequest)

	if preserveTokens <= noPreserveTokens {
		t.Errorf("preserve_thinking=true should render more tokens: preserve=%d, no-preserve=%d",
			preserveTokens, noPreserveTokens)
	}
}

// TestIntegrationTokenizePromptModeIsForwardedAsIs covers case 5: raw prompt tokenization
// is forwarded untouched, so the proxy count equals the backend's own count.
func TestIntegrationTokenizePromptModeIsForwardedAsIs(t *testing.T) {
	target, servedModel, client := integrationTarget(t)
	cfg := integrationConfig(servedModel, true)
	profiles := buildModelProfiles(cfg.VirtualModelPrefix, false, false, true)

	request := map[string]any{
		"model":  cfg.VirtualModelPrefix + "-auto",
		"prompt": "raw prompt tokenization must not apply the chat template",
	}
	proxyCount := integrationTokenizeCount(t, tokenize(client, target, cfg, profiles), request)

	// Same request straight to the backend, with the served model name.
	directRequest := map[string]any{"model": servedModel, "prompt": request["prompt"]}
	directCount := integrationTokenizeCount(t, passthrough(client, target), directRequest)

	if proxyCount != directCount {
		t.Errorf("prompt-mode count changed by the proxy: proxy=%d backend=%d", proxyCount, directCount)
	}
}

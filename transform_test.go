package main

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"sync"
	"testing"

	"github.com/hekmon/httplog/v3"
)

// thinkingDefaults / instructDefaults mirror the sampling sets as seen after a JSON
// round-trip (all numbers decoded as float64).
var (
	thinkingDefaults = map[string]any{
		"temperature": 1.0, "top_p": 0.95, "top_k": 20.0, "min_p": 0.0,
		"presence_penalty": 0.0, "repetition_penalty": 1.0,
	}
	instructDefaults = map[string]any{
		"temperature": 0.7, "top_p": 0.80, "top_k": 20.0, "min_p": 0.0,
		"presence_penalty": 1.5, "repetition_penalty": 1.0,
	}
)

// bodyCapture stores the request body received by the fake vLLM backend.
type bodyCapture struct {
	mu   sync.Mutex
	body []byte
}

func (c *bodyCapture) set(body []byte) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.body = body
}

func (c *bodyCapture) get() []byte {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.body
}

// roundTripFunc adapts a function to http.RoundTripper, so the fake backend needs no listener.
type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(req *http.Request) (*http.Response, error) {
	return f(req)
}

// testHarness fakes the vLLM backend and captures the proxified request body.
type testHarness struct {
	target  *url.URL
	client  *http.Client
	capture *bodyCapture
}

func newTestHarness(t *testing.T, backendResponse string) *testHarness {
	t.Helper()
	capture := &bodyCapture{}
	client := &http.Client{Transport: roundTripFunc(func(req *http.Request) (*http.Response, error) {
		var buf bytes.Buffer
		if _, err := buf.ReadFrom(req.Body); err != nil {
			return nil, err
		}
		capture.set(buf.Bytes())
		return &http.Response{
			StatusCode: http.StatusOK,
			Header:     http.Header{"Content-Type": []string{"application/json"}},
			Body:       io.NopCloser(strings.NewReader(backendResponse)),
			Request:    req,
		}, nil
	})}
	target, err := url.Parse("http://vllm.test")
	if err != nil {
		t.Fatalf("parse backend URL: %v", err)
	}
	return &testHarness{target: target, client: client, capture: capture}
}

// post sends a JSON request through handler and returns the recorder plus the request
// body the backend received (nil when the handler did not reach the backend).
func (h *testHarness) post(t *testing.T, handler http.HandlerFunc, path string, request map[string]any) (*httptest.ResponseRecorder, map[string]any) {
	t.Helper()
	payload, err := json.Marshal(request)
	if err != nil {
		t.Fatalf("marshal request: %v", err)
	}
	req := httptest.NewRequest(http.MethodPost, path, bytes.NewReader(payload))
	// The handlers expect the request ID set by the httplog middleware.
	req = req.WithContext(context.WithValue(req.Context(), httplog.ReqIDKey, uint64(1)))
	rec := httptest.NewRecorder()
	handler(rec, req)
	sent := h.capture.get()
	if sent == nil {
		return rec, nil
	}
	var body map[string]any
	if err := json.Unmarshal(sent, &body); err != nil {
		t.Fatalf("unmarshal proxified body %q: %v", string(sent), err)
	}
	return rec, body
}

func autoTestConfig(policy string) Config {
	return Config{
		ServedModelName:    "Qwen/Qwen3.8-27B",
		VirtualModelPrefix: "qwen38",
		SamplingParams:     policy,
		AutoSamplingParams: policy,
	}
}

// assertEnableThinkingUntouched checks that auto mode neither injects enable_thinking
// nor rewrites a client-provided one (the core rule of the auto model).
func assertEnableThinkingUntouched(t *testing.T, request, sent map[string]any) {
	t.Helper()
	requestKwargs, _ := request["chat_template_kwargs"].(map[string]any)
	want, wantPresent := requestKwargs["enable_thinking"]
	sentKwargs, _ := sent["chat_template_kwargs"].(map[string]any)
	got, gotPresent := sentKwargs["enable_thinking"]
	if wantPresent != gotPresent || got != want {
		t.Errorf("enable_thinking must be left untouched: client sent %v (present=%v), backend got %v (present=%v)",
			want, wantPresent, got, gotPresent)
	}
}

func TestTransformAutoModelMatrix(t *testing.T) {
	tests := []struct {
		name                 string
		policy               string
		noInstruct           bool
		autoPreserveThinking bool
		request              map[string]any
		wantStatus           int
		// wantSampling is asserted key by key against the proxified body.
		wantSampling map[string]any
		// wantAbsent lists sampling keys the proxified body must not contain.
		wantAbsent []string
		// wantErrorDetail is matched against the error message of a 400 response.
		wantErrorDetail string
	}{
		{
			name:         "1 - no effort resolves to thinking",
			policy:       "fill",
			request:      map[string]any{"model": "qwen38-auto", "messages": []any{}},
			wantSampling: thinkingDefaults,
		},
		{
			name:         "2 - effort none resolves to instruct",
			policy:       "fill",
			request:      map[string]any{"model": "qwen38-auto", "messages": []any{}, "reasoning_effort": "none"},
			wantSampling: instructDefaults,
		},
		{
			name:         "3 - effort low resolves to thinking",
			policy:       "fill",
			request:      map[string]any{"model": "qwen38-auto", "messages": []any{}, "reasoning_effort": "low"},
			wantSampling: thinkingDefaults,
		},
		{
			name:   "4 - enable_thinking true wins over effort none",
			policy: "fill",
			request: map[string]any{
				"model": "qwen38-auto", "messages": []any{}, "reasoning_effort": "none",
				"chat_template_kwargs": map[string]any{"enable_thinking": true},
			},
			wantSampling: thinkingDefaults,
		},
		{
			name:   "5 - enable_thinking false wins over absent effort",
			policy: "fill",
			request: map[string]any{
				"model": "qwen38-auto", "messages": []any{},
				"chat_template_kwargs": map[string]any{"enable_thinking": false},
			},
			wantSampling: instructDefaults,
		},
		{
			name:   "5b - enable_thinking false without effort none",
			policy: "fill",
			request: map[string]any{
				"model": "qwen38-auto", "messages": []any{}, "reasoning_effort": "xhigh",
				"chat_template_kwargs": map[string]any{"enable_thinking": false},
			},
			wantSampling: instructDefaults,
		},
		{
			name:   "6 - enforce overwrites client sampling",
			policy: "enforce",
			request: map[string]any{
				"model": "qwen38-auto", "messages": []any{}, "reasoning_effort": "none",
				"temperature": 0.2, "top_p": 0.1,
			},
			wantSampling: instructDefaults,
		},
		{
			name:       "7 - off leaves sampling untouched",
			policy:     "off",
			request:    map[string]any{"model": "qwen38-auto", "messages": []any{}, "reasoning_effort": "none", "temperature": 0.2},
			wantAbsent: []string{"top_p", "top_k", "min_p", "presence_penalty", "repetition_penalty"},
			wantSampling: map[string]any{
				"temperature": 0.2,
			},
		},
		{
			name:       "8 - no instruct deployment rejects instruct mode",
			policy:     "fill",
			noInstruct: true,
			request: map[string]any{
				"model": "qwen38-auto", "messages": []any{}, "reasoning_effort": "none",
			},
			wantStatus:      http.StatusBadRequest,
			wantErrorDetail: "no instruct mode on this deployment",
		},
		{
			name:       "8b - no instruct deployment rejects enable_thinking false",
			policy:     "fill",
			noInstruct: true,
			request: map[string]any{
				"model": "qwen38-auto", "messages": []any{},
				"chat_template_kwargs": map[string]any{"enable_thinking": false},
			},
			wantStatus:      http.StatusBadRequest,
			wantErrorDetail: "no instruct mode on this deployment",
		},
		{
			name:         "8c - no instruct deployment keeps thinking mode",
			policy:       "fill",
			noInstruct:   true,
			request:      map[string]any{"model": "qwen38-auto", "messages": []any{}},
			wantSampling: thinkingDefaults,
		},
		{
			name:   "9 - fill preserves client temperature",
			policy: "fill",
			request: map[string]any{
				"model": "qwen38-auto", "messages": []any{}, "temperature": 0.2,
			},
			wantSampling: map[string]any{
				"temperature": 0.2, "top_p": 0.95, "top_k": 20.0, "min_p": 0.0,
				"presence_penalty": 0.0, "repetition_penalty": 1.0,
			},
		},
		{
			name:                 "12 - auto preserve thinking pins preserve_thinking",
			policy:               "fill",
			autoPreserveThinking: true,
			request:              map[string]any{"model": "qwen38-auto", "messages": []any{}},
			wantSampling:         thinkingDefaults,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := autoTestConfig(tt.policy)
			cfg.NoInstruct = tt.noInstruct
			cfg.AutoPreserveThinking = tt.autoPreserveThinking
			profiles := buildModelProfiles(cfg.VirtualModelPrefix, false, cfg.NoInstruct, true)
			h := newTestHarness(t, `{"model":"Qwen/Qwen3.8-27B","choices":[]}`)

			rec, sent := h.post(t, transform(h.client, h.target, cfg, profiles), "/v1/chat/completions", tt.request)

			wantStatus := tt.wantStatus
			if wantStatus == 0 {
				wantStatus = http.StatusOK
			}
			if rec.Code != wantStatus {
				t.Fatalf("status: want %d, got %d (body %s)", wantStatus, rec.Code, rec.Body.String())
			}
			if tt.wantErrorDetail != "" {
				if !strings.Contains(rec.Body.String(), tt.wantErrorDetail) {
					t.Fatalf("error message should contain %q, got %s", tt.wantErrorDetail, rec.Body.String())
				}
				if sent != nil {
					t.Errorf("rejected request should not reach the backend, got %v", sent)
				}
				return
			}

			if sent == nil {
				t.Fatal("request did not reach the backend")
			}
			if got := sent["model"]; got != cfg.ServedModelName {
				t.Errorf("model: want %q, got %v", cfg.ServedModelName, got)
			}
			for k, want := range tt.wantSampling {
				if got := sent[k]; got != want {
					t.Errorf("%s: want %v, got %v", k, want, got)
				}
			}
			for _, k := range tt.wantAbsent {
				if got, exists := sent[k]; exists {
					t.Errorf("%s: want absent, got %v", k, got)
				}
			}
			// Hard requirement: the auto model never writes enable_thinking.
			assertEnableThinkingUntouched(t, tt.request, sent)
			kwargs, _ := sent["chat_template_kwargs"].(map[string]any)
			if want := tt.autoPreserveThinking; kwargs["preserve_thinking"] != want {
				t.Errorf("preserve_thinking: want %v, got %v", want, kwargs["preserve_thinking"])
			}
		})
	}
}

// TestTransformAutoNeverInjectsEnableThinking is a regression guard for the core rule:
// whatever the client sends (or does not send), auto mode must leave enable_thinking alone.
func TestTransformAutoNeverInjectsEnableThinking(t *testing.T) {
	cfg := autoTestConfig("fill")
	profiles := buildModelProfiles(cfg.VirtualModelPrefix, false, false, true)

	requests := []map[string]any{
		{"model": "qwen38-auto"},
		{"model": "qwen38-auto", "reasoning_effort": "none"},
		{"model": "qwen38-auto", "reasoning_effort": "xhigh"},
		{"model": "qwen38-auto", "reasoning_effort": "none", "chat_template_kwargs": map[string]any{"enable_thinking": true}},
		{"model": "qwen38-auto", "chat_template_kwargs": map[string]any{"enable_thinking": false}},
		{"model": "qwen38-auto", "chat_template_kwargs": map[string]any{"enable_thinking": "yes"}},
		{"model": "qwen38-auto", "chat_template_kwargs": map[string]any{"other_kwarg": 1}},
	}

	for _, request := range requests {
		h := newTestHarness(t, `{"model":"Qwen/Qwen3.8-27B","choices":[]}`)
		_, sent := h.post(t, transform(h.client, h.target, cfg, profiles), "/v1/chat/completions", request)
		if sent == nil {
			t.Fatalf("request %v did not reach the backend", request)
		}
		assertEnableThinkingUntouched(t, request, sent)
	}
}

func TestTransformAutoPreservesReasoningEffort(t *testing.T) {
	cfg := autoTestConfig("fill")
	profiles := buildModelProfiles(cfg.VirtualModelPrefix, false, false, true)
	h := newTestHarness(t, `{"model":"Qwen/Qwen3.8-27B","choices":[]}`)

	_, sent := h.post(t, transform(h.client, h.target, cfg, profiles), "/v1/chat/completions",
		map[string]any{"model": "qwen38-auto", "reasoning_effort": "none"})
	if sent["reasoning_effort"] != "none" {
		t.Errorf("reasoning_effort must be forwarded untouched, got %v", sent["reasoning_effort"])
	}
}

func TestTransformAutoInvalidKwargs(t *testing.T) {
	cfg := autoTestConfig("fill")
	profiles := buildModelProfiles(cfg.VirtualModelPrefix, false, false, true)
	h := newTestHarness(t, `{"model":"Qwen/Qwen3.8-27B","choices":[]}`)

	rec, sent := h.post(t, transform(h.client, h.target, cfg, profiles), "/v1/chat/completions",
		map[string]any{"model": "qwen38-auto", "chat_template_kwargs": "not-a-map"})
	if rec.Code != http.StatusBadRequest {
		t.Errorf("status: want %d, got %d", http.StatusBadRequest, rec.Code)
	}
	if sent != nil {
		t.Errorf("invalid request should not reach the backend, got %v", sent)
	}
}

// TestTransformThinkingModelUnchanged is the regression guard for fixed models.
func TestTransformThinkingModelUnchanged(t *testing.T) {
	cfg := autoTestConfig("fill")
	profiles := buildModelProfiles(cfg.VirtualModelPrefix, true, false, true)
	h := newTestHarness(t, `{"model":"Qwen/Qwen3.8-27B","choices":[]}`)

	rec, sent := h.post(t, transform(h.client, h.target, cfg, profiles), "/v1/chat/completions",
		map[string]any{"model": "qwen38-thinking", "reasoning_effort": "high"})
	if rec.Code != http.StatusOK {
		t.Fatalf("status: want 200, got %d", rec.Code)
	}
	kwargs, _ := sent["chat_template_kwargs"].(map[string]any)
	if kwargs["enable_thinking"] != true {
		t.Errorf("enable_thinking: want true, got %v", kwargs["enable_thinking"])
	}
	if kwargs["preserve_thinking"] != false {
		t.Errorf("preserve_thinking: want false, got %v", kwargs["preserve_thinking"])
	}
	if sent["reasoning_effort"] != "high" {
		t.Errorf("reasoning_effort: want high, got %v", sent["reasoning_effort"])
	}
	for k, want := range thinkingDefaults {
		if got := sent[k]; got != want {
			t.Errorf("%s: want %v, got %v", k, want, got)
		}
	}

	// Extended models keep enforcing their effort.
	h2 := newTestHarness(t, `{"model":"Qwen/Qwen3.8-27B","choices":[]}`)
	_, sent = h2.post(t, transform(h2.client, h2.target, cfg, profiles), "/v1/chat/completions",
		map[string]any{"model": "qwen38-thinking-medium", "reasoning_effort": "low"})
	if sent["reasoning_effort"] != "medium" {
		t.Errorf("reasoning_effort: want medium, got %v", sent["reasoning_effort"])
	}
}

// TestLegacyCompletionsAutoModel covers edge case 3: /v1/completions swaps the name only.
func TestLegacyCompletionsAutoModel(t *testing.T) {
	cfg := autoTestConfig("fill")
	profiles := buildModelProfiles(cfg.VirtualModelPrefix, false, false, true)
	h := newTestHarness(t, `{"model":"Qwen/Qwen3.8-27B","choices":[]}`)

	rec, sent := h.post(t, legacyCompletions(h.client, h.target, cfg.ServedModelName, profiles), "/v1/completions",
		map[string]any{"model": "qwen38-auto", "prompt": "hi", "reasoning_effort": "none"})
	if rec.Code != http.StatusOK {
		t.Fatalf("status: want 200, got %d", rec.Code)
	}
	if sent["model"] != cfg.ServedModelName {
		t.Errorf("model: want %q, got %v", cfg.ServedModelName, sent["model"])
	}
	for k := range thinkingDefaults {
		if got, exists := sent[k]; exists {
			t.Errorf("%s: legacy completions must not set sampling params, got %v", k, got)
		}
	}
	if _, exists := sent["chat_template_kwargs"]; exists {
		t.Errorf("legacy completions must not inject chat_template_kwargs, got %v", sent["chat_template_kwargs"])
	}
	if sent["reasoning_effort"] != "none" {
		t.Errorf("reasoning_effort: want none, got %v", sent["reasoning_effort"])
	}
}

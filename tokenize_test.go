package main

import (
	"net/http"
	"reflect"
	"strings"
	"testing"
)

// TestTokenizeAutoModel covers edge case 4: the auto model is renamed and pins
// preserve_thinking like the chat path does, but never enable_thinking, so the backend
// keeps deriving the mode from reasoning_effort.
func TestTokenizeAutoModel(t *testing.T) {
	tests := []struct {
		name                 string
		autoPreserveThinking bool
		request              map[string]any
		wantEnableThinking   any
		wantPreserveThinking any
	}{
		{
			name:                 "auto pins preserve_thinking false",
			request:              map[string]any{"model": "qwen38-auto", "messages": []any{}},
			wantPreserveThinking: false,
		},
		{
			name:                 "auto pins preserve_thinking true",
			autoPreserveThinking: true,
			request:              map[string]any{"model": "qwen38-auto", "messages": []any{}},
			wantPreserveThinking: true,
		},
		{
			name:                 "auto with effort none pins preserve_thinking false",
			request:              map[string]any{"model": "qwen38-auto", "messages": []any{}, "reasoning_effort": "none"},
			wantPreserveThinking: false,
		},
		{
			name:                 "thinking model still injects both kwargs",
			request:              map[string]any{"model": "qwen38-thinking", "messages": []any{}},
			wantEnableThinking:   true,
			wantPreserveThinking: false,
		},
		{
			name:                 "instruct model still injects both kwargs",
			request:              map[string]any{"model": "qwen38-instruct", "messages": []any{}},
			wantEnableThinking:   false,
			wantPreserveThinking: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := autoTestConfig("fill")
			cfg.AutoPreserveThinking = tt.autoPreserveThinking
			profiles := buildModelProfiles(cfg.VirtualModelPrefix, false, false, true)
			h := newTestHarness(t, `{"count":1,"max_model_len":1}`)

			rec, sent := h.post(t, tokenize(h.client, h.target, cfg, profiles), "/tokenize", tt.request)
			if rec.Code != http.StatusOK {
				t.Fatalf("status: want 200, got %d (body %s)", rec.Code, rec.Body.String())
			}
			if sent["model"] != cfg.ServedModelName {
				t.Errorf("model: want %q, got %v", cfg.ServedModelName, sent["model"])
			}
			kwargs, ok := sent["chat_template_kwargs"].(map[string]any)
			if !ok {
				t.Fatalf("chat_template_kwargs: want a map, got %v", sent["chat_template_kwargs"])
			}
			gotEnableThinking, hasEnableThinking := kwargs["enable_thinking"]
			if tt.wantEnableThinking == nil {
				if hasEnableThinking {
					t.Errorf("enable_thinking: want absent, got %v", gotEnableThinking)
				}
			} else if gotEnableThinking != tt.wantEnableThinking {
				t.Errorf("enable_thinking: want %v, got %v", tt.wantEnableThinking, gotEnableThinking)
			}
			if kwargs["preserve_thinking"] != tt.wantPreserveThinking {
				t.Errorf("preserve_thinking: want %v, got %v", tt.wantPreserveThinking, kwargs["preserve_thinking"])
			}
		})
	}
}

// TestTokenizeAutoPreservesOtherClientKwargs makes sure pre-existing kwargs survive for
// auto, and that a client-provided preserve_thinking is overridden like on the chat path.
func TestTokenizeAutoPreservesOtherClientKwargs(t *testing.T) {
	cfg := autoTestConfig("fill")
	profiles := buildModelProfiles(cfg.VirtualModelPrefix, false, false, true)
	h := newTestHarness(t, `{"count":1,"max_model_len":1}`)

	_, sent := h.post(t, tokenize(h.client, h.target, cfg, profiles), "/tokenize",
		map[string]any{
			"model":                "qwen38-auto",
			"messages":             []any{},
			"chat_template_kwargs": map[string]any{"other": "value", "preserve_thinking": !cfg.AutoPreserveThinking},
		})
	want := map[string]any{"other": "value", "preserve_thinking": cfg.AutoPreserveThinking}
	if !reflect.DeepEqual(sent["chat_template_kwargs"], want) {
		t.Errorf("chat_template_kwargs: want %v, got %v", want, sent["chat_template_kwargs"])
	}
}

// TestTokenizeAutoNoInstructRejectsInstructMode pins the decision for edge case 6: the
// tokenize path rejects what the chat path would reject, so a client can never obtain a
// token budget for a request the deployment cannot serve.
func TestTokenizeAutoNoInstructRejectsInstructMode(t *testing.T) {
	cfg := autoTestConfig("fill")
	cfg.NoInstruct = true
	profiles := buildModelProfiles(cfg.VirtualModelPrefix, false, true, true)
	h := newTestHarness(t, `{"count":1,"max_model_len":1}`)

	tests := []struct {
		name       string
		request    map[string]any
		wantStatus int
	}{
		{
			name:       "effort none is rejected",
			request:    map[string]any{"model": "qwen38-auto", "messages": []any{}, "reasoning_effort": "none"},
			wantStatus: http.StatusBadRequest,
		},
		{
			name: "enable_thinking false is rejected",
			request: map[string]any{
				"model": "qwen38-auto", "messages": []any{},
				"chat_template_kwargs": map[string]any{"enable_thinking": false},
			},
			wantStatus: http.StatusBadRequest,
		},
		{
			name:       "thinking mode is accepted",
			request:    map[string]any{"model": "qwen38-auto", "messages": []any{}},
			wantStatus: http.StatusOK,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			rec, sent := h.post(t, tokenize(h.client, h.target, cfg, profiles), "/tokenize", tt.request)
			if rec.Code != tt.wantStatus {
				t.Fatalf("status: want %d, got %d (body %s)", tt.wantStatus, rec.Code, rec.Body.String())
			}
			if tt.wantStatus != http.StatusOK {
				if sent != nil {
					t.Errorf("rejected request should not reach the backend, got %v", sent)
				}
				if !strings.Contains(rec.Body.String(), "no instruct mode on this deployment") {
					t.Errorf("error message should explain the rejection, got %s", rec.Body.String())
				}
			}
		})
	}
}

// autoChatKwargsParityRequests returns request shapes used across both endpoints.
var autoChatKwargsParityRequests = map[string]map[string]any{
	"multi-turn with thinking history, effort low": {
		"model": "qwen38-auto",
		"messages": []any{
			map[string]any{"role": "user", "content": "first question"},
			map[string]any{"role": "assistant", "content": "first answer", "reasoning_content": "prior thinking"},
			map[string]any{"role": "user", "content": "second question"},
		},
		"reasoning_effort": "low",
	},
	"multi-turn with thinking history, effort none": {
		"model": "qwen38-auto",
		"messages": []any{
			map[string]any{"role": "user", "content": "first question"},
			map[string]any{"role": "assistant", "content": "first answer", "reasoning_content": "prior thinking"},
			map[string]any{"role": "user", "content": "second question"},
		},
		"reasoning_effort": "none",
	},
	"client kwargs alongside auto": {
		"model":                "qwen38-auto",
		"messages":             []any{},
		"chat_template_kwargs": map[string]any{"other": "value"},
	},
}

// TestTokenizeChatKwargsParity is the regression guard for the token budget contract:
// for the same input, the chat transform and /tokenize must produce identical
// chat_template_kwargs and messages. Identical template inputs mean vLLM renders the
// same prompt on both paths, so a /tokenize count matches what the chat request sends.
func TestTokenizeChatKwargsParity(t *testing.T) {
	for name, request := range autoChatKwargsParityRequests {
		for _, autoPreserveThinking := range []bool{false, true} {
			cfg := autoTestConfig("fill")
			cfg.AutoPreserveThinking = autoPreserveThinking
			profiles := buildModelProfiles(cfg.VirtualModelPrefix, false, false, true)

			chatHarness := newTestHarness(t, `{"model":"Qwen/Qwen3.8-27B","choices":[]}`)
			_, chatSent := chatHarness.post(t, transform(chatHarness.client, chatHarness.target, cfg, profiles),
				"/v1/chat/completions", request)
			tokenizeHarness := newTestHarness(t, `{"count":1,"max_model_len":1}`)
			_, tokenizeSent := tokenizeHarness.post(t, tokenize(tokenizeHarness.client, tokenizeHarness.target, cfg, profiles),
				"/tokenize", request)

			if chatSent == nil || tokenizeSent == nil {
				t.Fatalf("%s (preserve=%v): request did not reach the backend", name, autoPreserveThinking)
			}
			if !reflect.DeepEqual(chatSent["chat_template_kwargs"], tokenizeSent["chat_template_kwargs"]) {
				t.Errorf("%s (preserve=%v): chat kwargs %v differ from tokenize kwargs %v",
					name, autoPreserveThinking, chatSent["chat_template_kwargs"], tokenizeSent["chat_template_kwargs"])
			}
			if !reflect.DeepEqual(chatSent["messages"], tokenizeSent["messages"]) {
				t.Errorf("%s (preserve=%v): messages differ between chat and tokenize", name, autoPreserveThinking)
			}
			for _, sent := range []map[string]any{chatSent, tokenizeSent} {
				kwargs, _ := sent["chat_template_kwargs"].(map[string]any)
				if _, injected := kwargs["enable_thinking"]; injected {
					t.Errorf("%s (preserve=%v): enable_thinking must never be injected for auto, got %v",
						name, autoPreserveThinking, kwargs)
				}
			}
		}
	}
}

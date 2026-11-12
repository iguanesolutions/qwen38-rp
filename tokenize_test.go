package main

import (
	"net/http"
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
			name:                 "auto without effort pins preserve_thinking false",
			request:              map[string]any{"model": "qwen38-auto", "messages": []any{}},
			wantPreserveThinking: false,
		},
		{
			name:                 "auto with effort none pins preserve_thinking false",
			request:              map[string]any{"model": "qwen38-auto", "messages": []any{}, "reasoning_effort": "none"},
			wantPreserveThinking: false,
		},
		{
			name:                 "auto honors auto-preserve-thinking",
			autoPreserveThinking: true,
			request:              map[string]any{"model": "qwen38-auto", "messages": []any{}},
			wantPreserveThinking: true,
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

			rec, sent := h.post(t, tokenize(h.client, h.target, cfg.ServedModelName, profiles, cfg.AutoPreserveThinking), "/tokenize", tt.request)
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

// TestTokenizeAutoPreservesOtherClientKwargs makes sure pre-existing kwargs survive for auto.
func TestTokenizeAutoPreservesOtherClientKwargs(t *testing.T) {
	cfg := autoTestConfig("fill")
	profiles := buildModelProfiles(cfg.VirtualModelPrefix, false, false, true)
	h := newTestHarness(t, `{"count":1,"max_model_len":1}`)

	_, sent := h.post(t, tokenize(h.client, h.target, cfg.ServedModelName, profiles, cfg.AutoPreserveThinking), "/tokenize",
		map[string]any{
			"model":    "qwen38-auto",
			"messages": []any{},
			// A client-provided preserve_thinking must be overridden, exactly like the
			// chat path does, so token counts stay consistent.
			"chat_template_kwargs": map[string]any{"other": "value", "preserve_thinking": true},
		})
	kwargs, ok := sent["chat_template_kwargs"].(map[string]any)
	if !ok {
		t.Fatalf("chat_template_kwargs: want a map, got %v", sent["chat_template_kwargs"])
	}
	if kwargs["other"] != "value" {
		t.Errorf("other kwarg: want value, got %v", kwargs["other"])
	}
	if kwargs["preserve_thinking"] != cfg.AutoPreserveThinking {
		t.Errorf("preserve_thinking: want %v, got %v", cfg.AutoPreserveThinking, kwargs["preserve_thinking"])
	}
	if _, injected := kwargs["enable_thinking"]; injected {
		t.Errorf("auto model must not inject enable_thinking, got %v", kwargs)
	}
}

// TestTokenizeMatchesChatPreserveThinking is the regression guard for the token budget
// contract: /tokenize must render the same preserve_thinking as the chat path, or
// multi-turn token counts would not match what the backend actually renders.
func TestTokenizeMatchesChatPreserveThinking(t *testing.T) {
	for _, autoPreserveThinking := range []bool{false, true} {
		cfg := autoTestConfig("fill")
		cfg.AutoPreserveThinking = autoPreserveThinking
		profiles := buildModelProfiles(cfg.VirtualModelPrefix, false, false, true)
		request := map[string]any{"model": "qwen38-auto", "messages": []any{}, "reasoning_effort": "none"}

		chatHarness := newTestHarness(t, `{"model":"Qwen/Qwen3.8-27B","choices":[]}`)
		_, chatSent := chatHarness.post(t, transform(chatHarness.client, chatHarness.target, cfg, profiles),
			"/v1/chat/completions", request)
		tokenizeHarness := newTestHarness(t, `{"count":1,"max_model_len":1}`)
		_, tokenizeSent := tokenizeHarness.post(t, tokenize(tokenizeHarness.client, tokenizeHarness.target, cfg.ServedModelName, profiles, cfg.AutoPreserveThinking),
			"/tokenize", request)

		chatKwargs, _ := chatSent["chat_template_kwargs"].(map[string]any)
		tokenizeKwargs, _ := tokenizeSent["chat_template_kwargs"].(map[string]any)
		if chatKwargs["preserve_thinking"] != tokenizeKwargs["preserve_thinking"] {
			t.Errorf("auto-preserve-thinking=%v: preserve_thinking differs between chat (%v) and tokenize (%v)",
				autoPreserveThinking, chatKwargs["preserve_thinking"], tokenizeKwargs["preserve_thinking"])
		}
		if _, injected := tokenizeKwargs["enable_thinking"]; injected {
			t.Errorf("tokenize must not inject enable_thinking for auto, got %v", tokenizeKwargs)
		}
		if _, injected := chatKwargs["enable_thinking"]; injected {
			t.Errorf("chat must not inject enable_thinking for auto, got %v", chatKwargs)
		}
	}
}

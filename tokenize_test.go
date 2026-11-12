package main

import (
	"net/http"
	"testing"
)

// TestTokenizeAutoModel covers edge case 4: the auto model is renamed but gets no
// chat_template_kwargs, so the backend derives enable_thinking from reasoning_effort.
func TestTokenizeAutoModel(t *testing.T) {
	cfg := autoTestConfig("fill")
	profiles := buildModelProfiles(cfg.VirtualModelPrefix, false, false, true)

	tests := []struct {
		name         string
		request      map[string]any
		wantKwargs   bool
		wantThinking any
	}{
		{
			name:    "auto without effort injects no kwargs",
			request: map[string]any{"model": "qwen38-auto", "messages": []any{}},
		},
		{
			name:    "auto with effort none injects no kwargs",
			request: map[string]any{"model": "qwen38-auto", "messages": []any{}, "reasoning_effort": "none"},
		},
		{
			name:         "thinking model still injects kwargs",
			request:      map[string]any{"model": "qwen38-thinking", "messages": []any{}},
			wantKwargs:   true,
			wantThinking: true,
		},
		{
			name:         "instruct model still injects kwargs",
			request:      map[string]any{"model": "qwen38-instruct", "messages": []any{}},
			wantKwargs:   true,
			wantThinking: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			h := newTestHarness(t, `{"count":1,"max_model_len":1}`)
			rec, sent := h.post(t, tokenize(h.client, h.target, cfg.ServedModelName, profiles), "/tokenize", tt.request)
			if rec.Code != http.StatusOK {
				t.Fatalf("status: want 200, got %d (body %s)", rec.Code, rec.Body.String())
			}
			if sent["model"] != cfg.ServedModelName {
				t.Errorf("model: want %q, got %v", cfg.ServedModelName, sent["model"])
			}
			if tt.wantKwargs {
				kwargs, ok := sent["chat_template_kwargs"].(map[string]any)
				if !ok {
					t.Fatalf("chat_template_kwargs: want a map, got %v", sent["chat_template_kwargs"])
				}
				if kwargs["enable_thinking"] != tt.wantThinking {
					t.Errorf("enable_thinking: want %v, got %v", tt.wantThinking, kwargs["enable_thinking"])
				}
				return
			}
			if kwargs, exists := sent["chat_template_kwargs"]; exists {
				t.Errorf("auto model must not inject chat_template_kwargs, got %v", kwargs)
			}
		})
	}
}

// TestTokenizePreservesClientKwargs makes sure pre-existing kwargs survive for auto.
func TestTokenizePreservesClientKwargs(t *testing.T) {
	cfg := autoTestConfig("fill")
	profiles := buildModelProfiles(cfg.VirtualModelPrefix, false, false, true)
	h := newTestHarness(t, `{"count":1,"max_model_len":1}`)

	_, sent := h.post(t, tokenize(h.client, h.target, cfg.ServedModelName, profiles), "/tokenize",
		map[string]any{
			"model":                "qwen38-auto",
			"messages":             []any{},
			"chat_template_kwargs": map[string]any{"other": "value"},
		})
	kwargs, ok := sent["chat_template_kwargs"].(map[string]any)
	if !ok {
		t.Fatalf("chat_template_kwargs: want a map, got %v", sent["chat_template_kwargs"])
	}
	if kwargs["other"] != "value" {
		t.Errorf("other kwarg: want value, got %v", kwargs["other"])
	}
	if _, injected := kwargs["enable_thinking"]; injected {
		t.Errorf("auto model must not inject enable_thinking, got %v", kwargs)
	}
}

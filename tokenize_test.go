package main

import (
	"maps"
	"net/http"
	"reflect"
	"strings"
	"testing"
)

// TestTokenizeAutoModel covers the auto model on /tokenize: the name is swapped, the mode
// the backend would have derived from reasoning_effort is pinned explicitly (because
// /tokenize does not run vLLM's build_chat_params derivation), and preserve_thinking is
// pinned like on the chat path.
func TestTokenizeAutoModel(t *testing.T) {
	tests := []struct {
		name                 string
		autoPreserveThinking bool
		request              map[string]any
		wantEnableThinking   any
		wantPreserveThinking any
	}{
		{
			name:                 "auto without effort pins thinking mode",
			request:              map[string]any{"model": "qwen38-auto", "messages": []any{}},
			wantEnableThinking:   true,
			wantPreserveThinking: false,
		},
		{
			name:                 "auto with effort none pins instruct mode",
			request:              map[string]any{"model": "qwen38-auto", "messages": []any{}, "reasoning_effort": "none"},
			wantEnableThinking:   false,
			wantPreserveThinking: false,
		},
		{
			name:                 "auto honors auto-preserve-thinking",
			autoPreserveThinking: true,
			request:              map[string]any{"model": "qwen38-auto", "messages": []any{}},
			wantEnableThinking:   true,
			wantPreserveThinking: true,
		},
		{
			name: "auto honors client enable_thinking over effort",
			request: map[string]any{
				"model": "qwen38-auto", "messages": []any{}, "reasoning_effort": "none",
				"chat_template_kwargs": map[string]any{"enable_thinking": true},
			},
			wantEnableThinking:   true,
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
			if gotEnableThinking, hasEnableThinking := kwargs["enable_thinking"]; !hasEnableThinking {
				t.Errorf("enable_thinking: want %v, got absent", tt.wantEnableThinking)
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
	want := map[string]any{
		"other":             "value",
		"preserve_thinking": cfg.AutoPreserveThinking,
		"enable_thinking":   true,
	}
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

// TestTokenizePartialHeader was replaced: vLLM's /tokenize can be made exact, see
// TestTokenizeMirrorsReasoningHistory and TestTokenizePinsTemplateEffort.

// TestTokenizeMirrorsReasoningHistory covers cause 2: vLLM's /tokenize ignores
// message.reasoning_content, so the proxy mirrors it into the field it does render.
func TestTokenizeMirrorsReasoningHistory(t *testing.T) {
	tests := []struct {
		name                 string
		autoPreserveThinking bool
		request              map[string]any
		wantMirrored         bool
	}{
		{
			name:                 "preserve thinking mirrors reasoning_content",
			autoPreserveThinking: true,
			request: map[string]any{"model": "qwen38-auto", "messages": []any{
				map[string]any{"role": "user", "content": "q"},
				map[string]any{"role": "assistant", "content": "a", "reasoning_content": "thinking"},
				map[string]any{"role": "user", "content": "q2"},
			}},
			wantMirrored: true,
		},
		{
			name: "preserve disabled does not mirror",
			request: map[string]any{"model": "qwen38-auto", "messages": []any{
				map[string]any{"role": "assistant", "content": "a", "reasoning_content": "thinking"},
			}},
		},
		{
			name:                 "instruct mode does not mirror",
			autoPreserveThinking: true,
			request: map[string]any{"model": "qwen38-auto", "messages": []any{
				map[string]any{"role": "assistant", "content": "a", "reasoning_content": "thinking"},
			}, "reasoning_effort": "none"},
		},
		{
			name:                 "thinking-preserve model mirrors",
			autoPreserveThinking: false,
			request: map[string]any{"model": "qwen38-thinking-preserve", "messages": []any{
				map[string]any{"role": "assistant", "content": "a", "reasoning_content": "thinking"},
			}},
			wantMirrored: true,
		},
		{
			name: "plain thinking model does not mirror",
			request: map[string]any{"model": "qwen38-thinking", "messages": []any{
				map[string]any{"role": "assistant", "content": "a", "reasoning_content": "thinking"},
			}},
		},
		{
			name:                 "existing reasoning field is kept",
			autoPreserveThinking: true,
			request: map[string]any{"model": "qwen38-auto", "messages": []any{
				map[string]any{"role": "assistant", "content": "a", "reasoning": "original", "reasoning_content": "other"},
			}},
			wantMirrored: true,
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
			// Locate the assistant message in both request and forwarded body.
			findAssistant := func(messages any) map[string]any {
				list, _ := messages.([]any)
				for _, raw := range list {
					message, _ := raw.(map[string]any)
					if _, ok := message["reasoning_content"]; ok {
						return message
					}
					if _, ok := message["reasoning"]; ok {
						return message
					}
				}
				return nil
			}
			wantMessage := findAssistant(tt.request["messages"])
			gotMessage := findAssistant(sent["messages"])
			if wantMessage == nil || gotMessage == nil {
				t.Fatalf("assistant message not found: request=%v forwarded=%v", tt.request["messages"], sent["messages"])
			}
			mirrored, hasReasoning := gotMessage["reasoning"]
			if tt.wantMirrored != hasReasoning {
				t.Fatalf("reasoning present: want %v, got %v (%v)", tt.wantMirrored, hasReasoning, gotMessage)
			}
			if !tt.wantMirrored {
				return
			}
			if original, exists := wantMessage["reasoning"]; exists {
				if mirrored != original {
					t.Errorf("existing reasoning must be preserved: want %v, got %v", original, mirrored)
				}
				return
			}
			if want := wantMessage["reasoning_content"]; mirrored != want {
				t.Errorf("reasoning: want %v, got %v", want, mirrored)
			}
			if _, kept := gotMessage["reasoning_content"]; !kept {
				t.Error("reasoning_content should be kept alongside the mirror")
			}
		})
	}
}

// TestTokenizePinsTemplateEffort covers the effort directive: the chat path renders it,
// /tokenize only sees it through chat_template_kwargs.
func TestTokenizePinsTemplateEffort(t *testing.T) {
	tests := []struct {
		name            string
		model           string
		clientEffort    string
		wantEffort      any // nil = must be absent
		wantEnableThink bool
	}{
		{name: "no effort", model: "qwen38-auto", wantEnableThink: true},
		{name: "low", model: "qwen38-auto", clientEffort: "low", wantEffort: "low", wantEnableThink: true},
		{name: "medium", model: "qwen38-auto", clientEffort: "medium", wantEffort: "medium", wantEnableThink: true},
		{name: "xhigh", model: "qwen38-auto", clientEffort: "xhigh", wantEffort: "xhigh", wantEnableThink: true},
		{name: "none resolves to instruct, no directive", model: "qwen38-auto", clientEffort: "none", wantEnableThink: false},
		{name: "unknown effort is never injected", model: "qwen38-auto", clientEffort: "minimal", wantEnableThink: true},
		{name: "high is left to the backend map", model: "qwen38-auto", clientEffort: "high", wantEnableThink: true},
		{name: "pre-configured model pins its own effort", model: "qwen38-thinking-medium", clientEffort: "low", wantEffort: "medium", wantEnableThink: true},
		{name: "base thinking model forwards the client effort", model: "qwen38-thinking", clientEffort: "low", wantEffort: "low", wantEnableThink: true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := autoTestConfig("fill")
			profiles := buildModelProfiles(cfg.VirtualModelPrefix, true, false, true)
			h := newTestHarness(t, `{"count":1,"max_model_len":1}`)

			request := map[string]any{"model": tt.model, "messages": []any{}}
			if tt.clientEffort != "" {
				request["reasoning_effort"] = tt.clientEffort
			}
			rec, sent := h.post(t, tokenize(h.client, h.target, cfg, profiles), "/tokenize", request)
			if rec.Code != http.StatusOK {
				t.Fatalf("status: want 200, got %d (body %s)", rec.Code, rec.Body.String())
			}
			kwargs, _ := sent["chat_template_kwargs"].(map[string]any)
			gotEffort, hasEffort := kwargs["reasoning_effort"]
			if tt.wantEffort == nil {
				if hasEffort {
					t.Errorf("reasoning_effort kwarg: want absent, got %v", gotEffort)
				}
			} else if gotEffort != tt.wantEffort {
				t.Errorf("reasoning_effort kwarg: want %v, got %v", tt.wantEffort, gotEffort)
			}
			if kwargs["enable_thinking"] != tt.wantEnableThink {
				t.Errorf("enable_thinking: want %v, got %v", tt.wantEnableThink, kwargs["enable_thinking"])
			}
		})
	}
}

// autoParityRequests are request shapes exercised against both endpoints.
var autoParityRequests = map[string]map[string]any{
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
	"client enable_thinking true with effort none": {
		"model":                "qwen38-auto",
		"messages":             []any{},
		"reasoning_effort":     "none",
		"chat_template_kwargs": map[string]any{"enable_thinking": true},
	},
	"client kwargs alongside auto": {
		"model":                "qwen38-auto",
		"messages":             []any{},
		"chat_template_kwargs": map[string]any{"other": "value"},
	},
}

// TestTokenizeMatchesChatRenderedPrompt is the guard for the token budget contract.
//
// The invariant is not "identical kwargs" — the chat path deliberately omits
// enable_thinking (vLLM derives it from reasoning_effort) while /tokenize must pin it
// (vLLM does not derive it there). The invariant is "identical rendered prompt": the same
// messages, the same preserve_thinking, and an enable_thinking value that reflects the
// mode the chat path's backend will derive.
func TestTokenizeMatchesChatRenderedPrompt(t *testing.T) {
	for name, request := range autoParityRequests {
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

			wantMode, err := resolveAutoMode(request)
			if err != nil {
				t.Fatalf("%s: resolving mode: %v", name, err)
			}

			chatKwargs, _ := chatSent["chat_template_kwargs"].(map[string]any)
			tokenizeKwargs, _ := tokenizeSent["chat_template_kwargs"].(map[string]any)

			// The chat path leaves enable_thinking to the backend: it never adds one and never
			// rewrites a client-supplied value.
			assertEnableThinkingUntouched(t, request, chatSent)
			// /tokenize must pin exactly the mode the chat path's backend derives.
			if got, want := tokenizeKwargs["enable_thinking"], wantMode == modeThinking; got != want {
				t.Errorf("%s (preserve=%v): tokenize enable_thinking: want %v, got %v", name, autoPreserveThinking, want, got)
			}
			if chatKwargs["preserve_thinking"] != tokenizeKwargs["preserve_thinking"] {
				t.Errorf("%s (preserve=%v): preserve_thinking differs between chat (%v) and tokenize (%v)",
					name, autoPreserveThinking, chatKwargs["preserve_thinking"], tokenizeKwargs["preserve_thinking"])
			}
			// Same messages, except that /tokenize mirrors reasoning_content into the
			// reasoning field it renders (chat renders reasoning_content directly).
			chatMessages, _ := chatSent["messages"].([]any)
			tokenizeMessages, _ := tokenizeSent["messages"].([]any)
			if len(chatMessages) != len(tokenizeMessages) {
				t.Fatalf("%s (preserve=%v): message count differs: chat=%d tokenize=%d",
					name, autoPreserveThinking, len(chatMessages), len(tokenizeMessages))
			}
			wantMirror := wantMode == modeThinking && autoPreserveThinking
			for i := range chatMessages {
				chatMessage, _ := chatMessages[i].(map[string]any)
				tokenizeMessage, _ := tokenizeMessages[i].(map[string]any)
				mirrored, hasMirror := tokenizeMessage["reasoning"]
				withoutMirror := maps.Clone(tokenizeMessage)
				delete(withoutMirror, "reasoning")
				if !reflect.DeepEqual(chatMessage, withoutMirror) {
					t.Errorf("%s (preserve=%v): message %d differs beyond the mirror: chat=%v tokenize=%v",
						name, autoPreserveThinking, i, chatMessage, withoutMirror)
				}
				if !hasMirror {
					continue
				}
				if !wantMirror {
					t.Errorf("%s (preserve=%v): unexpected reasoning mirror in message %d", name, autoPreserveThinking, i)
				}
				if mirrored != tokenizeMessage["reasoning_content"] {
					t.Errorf("%s (preserve=%v): mirrored reasoning must equal reasoning_content in message %d", name, autoPreserveThinking, i)
				}
			}
		}
	}
}

package main

import (
	"bytes"
	"encoding/json"
	"io"
	"log/slog"
	"testing"
)

func testLogger() *slog.Logger {
	return slog.New(slog.NewTextHandler(io.Discard, nil))
}

func TestBuildModelProfiles(t *testing.T) {
	tests := []struct {
		name           string
		prefix         string
		enableExtended bool
		noInstruct     bool
		wantModels     []string
	}{
		{
			name:           "default",
			prefix:         "qwen38",
			enableExtended: false,
			noInstruct:     false,
			wantModels:     []string{"qwen38-instruct", "qwen38-thinking", "qwen38-thinking-preserve"},
		},
		{
			name:           "extended",
			prefix:         "qwen38",
			enableExtended: true,
			noInstruct:     false,
			wantModels: []string{
				"qwen38-instruct", "qwen38-thinking", "qwen38-thinking-preserve",
				"qwen38-thinking-low", "qwen38-thinking-medium", "qwen38-thinking-xhigh",
				"qwen38-thinking-preserve-low", "qwen38-thinking-preserve-medium", "qwen38-thinking-preserve-xhigh",
			},
		},
		{
			name:           "no instruct",
			prefix:         "qwen38",
			enableExtended: false,
			noInstruct:     true,
			wantModels:     []string{"qwen38-thinking", "qwen38-thinking-preserve"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := buildModelProfiles(tt.prefix, tt.enableExtended, tt.noInstruct)
			if len(got) != len(tt.wantModels) {
				t.Fatalf("expected %d models, got %d", len(tt.wantModels), len(got))
			}
			for _, want := range tt.wantModels {
				if _, ok := got[want]; !ok {
					t.Errorf("expected model %q not found", want)
				}
			}
		})
	}
}

func TestBuildModelProfilesContent(t *testing.T) {
	profiles := buildModelProfiles("qwen38", false, false)

	instruct, ok := profiles["qwen38-instruct"]
	if !ok {
		t.Fatal("instruct profile not found")
	}
	if instruct.Think {
		t.Error("instruct should have Think=false")
	}
	if instruct.PreserveThinking {
		t.Error("instruct should have PreserveThinking=false")
	}
	if instruct.Effort != "" {
		t.Error("instruct should have empty Effort")
	}

	thinking, ok := profiles["qwen38-thinking"]
	if !ok {
		t.Fatal("thinking profile not found")
	}
	if !thinking.Think {
		t.Error("thinking should have Think=true")
	}
	if thinking.PreserveThinking {
		t.Error("thinking should have PreserveThinking=false")
	}

	extended := buildModelProfiles("qwen38", true, false)
	low, ok := extended["qwen38-thinking-low"]
	if !ok {
		t.Fatal("thinking-low profile not found")
	}
	if low.Effort != "low" {
		t.Errorf("expected effort low, got %q", low.Effort)
	}
}

func TestFixModelNameInResponse(t *testing.T) {
	logger := testLogger()

	tests := []struct {
		name         string
		input        string
		virtualModel string
		wantSame     bool
		wantContains string
	}{
		{
			name:         "replaces model name",
			input:        `{"model":"Qwen/Qwen3.8-27B","choices":[]}`,
			virtualModel: "qwen38-thinking",
			wantContains: `"model":"qwen38-thinking"`,
		},
		{
			name:         "no model field",
			input:        `{"choices":[]}`,
			virtualModel: "qwen38-thinking",
			wantSame:     true,
		},
		{
			name:         "invalid json",
			input:        `{not json`,
			virtualModel: "qwen38-thinking",
			wantSame:     true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := fixModelNameInResponse([]byte(tt.input), tt.virtualModel, logger)
			if tt.wantSame {
				if string(got) != tt.input {
					t.Errorf("expected original input, got %s", string(got))
				}
				return
			}
			if !bytes.Contains(got, []byte(tt.wantContains)) {
				t.Errorf("expected output to contain %q, got %s", tt.wantContains, string(got))
			}
		})
	}
}

func TestFixReasoningContentBug(t *testing.T) {
	logger := testLogger()

	tests := []struct {
		name        string
		input       map[string]any
		wantFixed   bool
		wantContent string
	}{
		{
			name: "moves reasoning_content to content",
			input: map[string]any{
				"choices": []any{
					map[string]any{
						"message": map[string]any{
							"content":           "",
							"reasoning_content": "some reasoning",
						},
					},
				},
			},
			wantFixed:   true,
			wantContent: "some reasoning",
		},
		{
			name: "moves reasoning to content",
			input: map[string]any{
				"choices": []any{
					map[string]any{
						"message": map[string]any{
							"content":   "",
							"reasoning": "some reasoning",
						},
					},
				},
			},
			wantFixed:   true,
			wantContent: "some reasoning",
		},
		{
			name: "does not modify when content exists",
			input: map[string]any{
				"choices": []any{
					map[string]any{
						"message": map[string]any{
							"content":           "existing content",
							"reasoning_content": "some reasoning",
						},
					},
				},
			},
			wantFixed:   false,
			wantContent: "existing content",
		},
		{
			name:      "no choices",
			input:     map[string]any{},
			wantFixed: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Deep copy via JSON to avoid mutating test table data
			b, _ := json.Marshal(tt.input)
			var inputCopy map[string]any
			json.Unmarshal(b, &inputCopy)

			got := fixReasoningContentBug(inputCopy, logger)
			if got != tt.wantFixed {
				t.Errorf("expected fixed=%v, got %v", tt.wantFixed, got)
			}
			if tt.wantContent != "" {
				choices, ok := inputCopy["choices"].([]any)
				if !ok || len(choices) == 0 {
					t.Fatal("expected choices after fix")
				}
				msg := choices[0].(map[string]any)["message"].(map[string]any)
				content := msg["content"].(string)
				if content != tt.wantContent {
					t.Errorf("expected content %q, got %q", tt.wantContent, content)
				}
			}
		})
	}
}

func TestFixModelNameInSSE(t *testing.T) {
	logger := testLogger()

	tests := []struct {
		name         string
		event        string
		virtualModel string
		wantSame     bool
		wantContains string
	}{
		{
			name:         "replaces model in data line",
			event:        "data: {\"model\":\"backend-model\",\"choices\":[]}\n\n",
			virtualModel: "qwen38-thinking",
			wantContains: `"model":"qwen38-thinking"`,
		},
		{
			name:         "ignores DONE",
			event:        "data: [DONE]\n\n",
			virtualModel: "qwen38-thinking",
			wantSame:     true,
		},
		{
			name:         "ignores empty data",
			event:        "data: \n\n",
			virtualModel: "qwen38-thinking",
			wantSame:     true,
		},
		{
			name:         "preserves non-data lines",
			event:        "id: 123\ndata: {\"model\":\"backend\"}\n\n",
			virtualModel: "qwen38-virt",
			wantContains: "id: 123",
		},
		{
			name:         "invalid json in data line",
			event:        "data: {not json}\n\n",
			virtualModel: "qwen38-thinking",
			wantSame:     true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := fixModelNameInSSE([]byte(tt.event), tt.virtualModel, logger)
			if tt.wantSame {
				if string(got) != tt.event {
					t.Errorf("expected unchanged event, got %q", string(got))
				}
				return
			}
			if !bytes.Contains(got, []byte(tt.wantContains)) {
				t.Errorf("expected output to contain %q, got %q", tt.wantContains, string(got))
			}
		})
	}
}

package main

import (
	"errors"
	"net/http"
	"testing"
)

func TestApplySamplingParams(t *testing.T) {
	logger := testLogger()
	defaults := map[string]any{"temperature": 1.0, "top_p": 0.95}

	tests := []struct {
		name     string
		data     map[string]any
		policy   SamplingPolicy
		wantTemp any
		wantTopP any
	}{
		{
			name:     "sets missing params",
			data:     map[string]any{},
			policy:   SamplingPolicyFill,
			wantTemp: 1.0,
			wantTopP: 0.95,
		},
		{
			name:     "preserves existing when filling",
			data:     map[string]any{"temperature": 0.5},
			policy:   SamplingPolicyFill,
			wantTemp: 0.5,
			wantTopP: 0.95,
		},
		{
			name:     "overrides existing when enforcing",
			data:     map[string]any{"temperature": 0.5},
			policy:   SamplingPolicyEnforce,
			wantTemp: 1.0,
			wantTopP: 0.95,
		},
		{
			name:     "adds nothing when off",
			data:     map[string]any{"temperature": 0.5},
			policy:   SamplingPolicyOff,
			wantTemp: 0.5,
			wantTopP: nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			data := make(map[string]any)
			for k, v := range tt.data {
				data[k] = v
			}
			applySamplingParams(data, defaults, modeThinking, tt.policy, logger)
			if data["temperature"] != tt.wantTemp {
				t.Errorf("temperature: want %v, got %v", tt.wantTemp, data["temperature"])
			}
			if data["top_p"] != tt.wantTopP {
				t.Errorf("top_p: want %v, got %v", tt.wantTopP, data["top_p"])
			}
		})
	}
}

func TestReadBodyStatusCode(t *testing.T) {
	if readBodyStatusCode(errors.New("random")) != http.StatusInternalServerError {
		t.Error("expected 500 for random error")
	}
	if readBodyStatusCode(&http.MaxBytesError{Limit: 100}) != http.StatusRequestEntityTooLarge {
		t.Error("expected 413 for MaxBytesError")
	}
}

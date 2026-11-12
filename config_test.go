package main

import (
	"log/slog"
	"testing"
)

func TestConfigValidate(t *testing.T) {
	valid := Config{
		Listen:             "0.0.0.0",
		Port:               9000,
		Target:             "http://localhost:8000",
		LogLevel:           "INFO",
		ServedModelName:    "Qwen/Qwen3.8-27B",
		VirtualModelPrefix: "qwen38",
	}

	tests := []struct {
		name    string
		cfg     Config
		wantErr bool
	}{
		{"valid", valid, false},
		{"empty listen", func() Config { c := valid; c.Listen = ""; return c }(), true},
		{"port zero", func() Config { c := valid; c.Port = 0; return c }(), true},
		{"port too high", func() Config { c := valid; c.Port = 70000; return c }(), true},
		{"empty target", func() Config { c := valid; c.Target = ""; return c }(), true},
		{"empty log level", func() Config { c := valid; c.LogLevel = ""; return c }(), true},
		{"empty model", func() Config { c := valid; c.ServedModelName = ""; return c }(), true},
		{"empty prefix", func() Config { c := valid; c.VirtualModelPrefix = ""; return c }(), true},
		{"empty sampling params defaults to fill", func() Config { c := valid; c.SamplingParams = ""; return c }(), false},
		{"valid enforce", func() Config { c := valid; c.SamplingParams = "enforce"; return c }(), false},
		{"unknown sampling params", func() Config { c := valid; c.SamplingParams = "always"; return c }(), true},
		{"unknown auto sampling params", func() Config {
			c := valid
			c.SamplingParams = "off"
			c.AutoSamplingParams = "sometimes"
			return c
		}(), true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.cfg.Validate()
			if tt.wantErr && err == nil {
				t.Error("expected error, got nil")
			}
			if !tt.wantErr && err != nil {
				t.Errorf("unexpected error: %v", err)
			}
		})
	}
}

func TestParseSamplingPolicy(t *testing.T) {
	tests := []struct {
		input   string
		want    SamplingPolicy
		wantErr bool
	}{
		{"off", SamplingPolicyOff, false},
		{"fill", SamplingPolicyFill, false},
		{"enforce", SamplingPolicyEnforce, false},
		{"ENFORCE", SamplingPolicyEnforce, false},
		{" fill ", SamplingPolicyFill, false},
		{"", "", true},
		{"always", "", true},
	}

	for _, tt := range tests {
		t.Run(tt.input, func(t *testing.T) {
			got, err := ParseSamplingPolicy(tt.input)
			if tt.wantErr {
				if err == nil {
					t.Fatalf("expected error for %q, got %q", tt.input, got)
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error for %q: %v", tt.input, err)
			}
			if got != tt.want {
				t.Errorf("ParseSamplingPolicy(%q) = %q, want %q", tt.input, got, tt.want)
			}
		})
	}
}

func TestResolveSamplingParams(t *testing.T) {
	tests := []struct {
		name             string
		samplingRaw      string
		samplingExplicit bool
		legacyEnforce    bool
		want             SamplingPolicy
		wantErr          bool
	}{
		{"defaults to fill", "fill", false, false, SamplingPolicyFill, false},
		{"legacy false maps to fill", "fill", false, false, SamplingPolicyFill, false},
		{"legacy true maps to enforce", "fill", false, true, SamplingPolicyEnforce, false},
		{"explicit wins over legacy", "off", true, true, SamplingPolicyOff, false},
		{"explicit empty defaults to fill", "", true, false, SamplingPolicyFill, false},
		{"explicit invalid", "sometimes", true, false, "", true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := resolveSamplingParams(tt.samplingRaw, tt.samplingExplicit, tt.legacyEnforce)
			if tt.wantErr {
				if err == nil {
					t.Fatalf("expected error, got %q", got)
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if got != tt.want {
				t.Errorf("got %q, want %q", got, tt.want)
			}
		})
	}
}

func TestResolvedSamplingPolicies(t *testing.T) {
	tests := []struct {
		name       string
		global     string
		auto       string
		wantGlobal SamplingPolicy
		wantAuto   SamplingPolicy
	}{
		{"both empty", "", "", SamplingPolicyFill, SamplingPolicyFill},
		{"auto inherits global", "enforce", "", SamplingPolicyEnforce, SamplingPolicyEnforce},
		{"auto overrides global", "enforce", "off", SamplingPolicyEnforce, SamplingPolicyOff},
		{"auto fills when global off", "off", "", SamplingPolicyOff, SamplingPolicyOff},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := Config{SamplingParams: tt.global, AutoSamplingParams: tt.auto}
			global, auto, err := cfg.resolvedSamplingPolicies()
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if global != tt.wantGlobal {
				t.Errorf("global: got %q, want %q", global, tt.wantGlobal)
			}
			if auto != tt.wantAuto {
				t.Errorf("auto: got %q, want %q", auto, tt.wantAuto)
			}
		})
	}
}

func TestParseLogLevel(t *testing.T) {
	tests := []struct {
		input string
		want  slog.Level
	}{
		{"COMPLETE", COMPLETE},
		{"DEBUG", slog.LevelDebug},
		{"INFO", slog.LevelInfo},
		{"WARN", slog.LevelWarn},
		{"ERROR", slog.LevelError},
		{"unknown", slog.LevelInfo},
		{"info", slog.LevelInfo},
	}

	for _, tt := range tests {
		t.Run(tt.input, func(t *testing.T) {
			got := parseLogLevel(tt.input)
			if got != tt.want {
				t.Errorf("parseLogLevel(%q) = %v, want %v", tt.input, got, tt.want)
			}
		})
	}
}

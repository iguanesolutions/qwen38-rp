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

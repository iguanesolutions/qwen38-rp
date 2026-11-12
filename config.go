package main

import (
	"errors"
	"flag"
	"fmt"
	"log"
	"log/slog"
	"os"
	"strconv"
	"strings"
)

// COMPLETE is a log level more verbose than DEBUG for complete request/response dumps
const COMPLETE = slog.LevelDebug - 4
const COMPLETE_LEVEL = "COMPLETE"

// SamplingPolicy controls how the proxy applies a mode's official sampling defaults.
type SamplingPolicy string

const (
	// SamplingPolicyOff never modifies the client's sampling parameters.
	SamplingPolicyOff SamplingPolicy = "off"
	// SamplingPolicyFill only sets the sampling parameters the client omitted.
	SamplingPolicyFill SamplingPolicy = "fill"
	// SamplingPolicyEnforce always overwrites the client-provided sampling parameters.
	SamplingPolicyEnforce SamplingPolicy = "enforce"
)

// ParseSamplingPolicy parses and validates a sampling policy name.
func ParseSamplingPolicy(raw string) (SamplingPolicy, error) {
	policy := SamplingPolicy(strings.ToLower(strings.TrimSpace(raw)))
	switch policy {
	case SamplingPolicyOff, SamplingPolicyFill, SamplingPolicyEnforce:
		return policy, nil
	default:
		return "", fmt.Errorf("invalid sampling params %q: must be one of %q, %q, %q",
			raw, SamplingPolicyOff, SamplingPolicyFill, SamplingPolicyEnforce,
		)
	}
}

type Config struct {
	Listen               string
	Port                 int
	Target               string
	LogLevel             string
	ServedModelName      string
	EnableExtendedModels bool
	EnableAutoModel      bool
	SamplingParams       string
	AutoSamplingParams   string
	AutoPreserveThinking bool
	VirtualModelPrefix   string
	NoInstruct           bool
}

func (c Config) Validate() error {
	if c.Listen == "" {
		return errors.New("listen address cannot be empty")
	}
	if c.Port < 1 || c.Port > 65535 {
		return errors.New("port must be between 1 and 65535")
	}
	if c.Target == "" {
		return errors.New("target cannot be empty")
	}
	if c.LogLevel == "" {
		return errors.New("log level cannot be empty")
	}
	if c.ServedModelName == "" {
		return errors.New("served model name cannot be empty")
	}
	if c.VirtualModelPrefix == "" {
		return errors.New("virtual model prefix cannot be empty")
	}
	if _, _, err := c.resolvedSamplingPolicies(); err != nil {
		return err
	}
	return nil
}

// resolvedSamplingPolicies returns the effective global and auto model sampling policies.
// An empty global policy means the default (fill); an empty auto policy means "same as global".
func (c Config) resolvedSamplingPolicies() (global, auto SamplingPolicy, err error) {
	globalRaw := c.SamplingParams
	if strings.TrimSpace(globalRaw) == "" {
		globalRaw = string(SamplingPolicyFill)
	}
	autoRaw := c.AutoSamplingParams
	if strings.TrimSpace(autoRaw) == "" {
		autoRaw = globalRaw
	}
	if global, err = ParseSamplingPolicy(globalRaw); err != nil {
		return "", "", fmt.Errorf("sampling params: %w", err)
	}
	if auto, err = ParseSamplingPolicy(autoRaw); err != nil {
		return "", "", fmt.Errorf("auto sampling params: %w", err)
	}
	return global, auto, nil
}

// samplingPolicy returns the global sampling policy. Invalid values are rejected by
// Validate at startup; this falls back to the default policy for safety.
func (c Config) samplingPolicy() SamplingPolicy {
	global, _, err := c.resolvedSamplingPolicies()
	if err != nil {
		return SamplingPolicyFill
	}
	return global
}

// autoSamplingPolicy returns the sampling policy used by the auto virtual model.
// Invalid values are rejected by Validate at startup; this falls back to the default
// policy for safety.
func (c Config) autoSamplingPolicy() SamplingPolicy {
	_, auto, err := c.resolvedSamplingPolicies()
	if err != nil {
		return SamplingPolicyFill
	}
	return auto
}

// resolveSamplingParams resolves the global sampling policy, mapping the deprecated
// -enforce-sampling-params option onto it (true = enforce, false = fill).
// The dedicated -sampling-params option always wins when explicitly set.
func resolveSamplingParams(samplingRaw string, samplingExplicit bool, legacyEnforce bool) (SamplingPolicy, error) {
	if !samplingExplicit {
		if legacyEnforce {
			return SamplingPolicyEnforce, nil
		}
		return SamplingPolicyFill, nil
	}
	if strings.TrimSpace(samplingRaw) == "" {
		return SamplingPolicyFill, nil
	}
	return ParseSamplingPolicy(samplingRaw)
}

func LoadConfig() (Config, error) {
	var cfg Config

	listen := flag.String("listen", "0.0.0.0", "IP address to listen on")
	port := flag.Int("port", 9000, "Port to listen on")
	target := flag.String("target", "http://127.0.0.1:8000", "Backend target, default is for a local vLLM")
	loglevel := flag.String("loglevel", slog.LevelInfo.String(), "Log level (COMPLETE, DEBUG, INFO, WARN, ERROR)")
	servedModel := flag.String("served-model", "", "Name of the served model")
	enableExtended := flag.Bool("enable-extended-models", false, "Enable extended pre-configured virtual models (low/medium/xhigh)")
	enableAuto := flag.Bool("enable-auto-model", false, "Enable the auto virtual model, which derives the Qwen mode from reasoning_effort")
	samplingParams := flag.String("sampling-params", string(SamplingPolicyFill), "Sampling parameters policy: off, fill, or enforce")
	autoSamplingParams := flag.String("auto-sampling-params", "", "Sampling parameters policy for the auto model (defaults to -sampling-params)")
	autoPreserveThinking := flag.Bool("auto-preserve-thinking", false, "Preserve historical thinking content in the auto model")
	enforceSampling := flag.Bool("enforce-sampling-params", false, "Deprecated: enforce sampling parameters, overriding client-provided values (use -sampling-params=enforce)")
	virtualModelPrefix := flag.String("virtual-model-prefix", "qwen38", "Prefix for virtual model names exposed by the proxy")
	noInstruct := flag.Bool("no-instruct", false, "Disable the instruct virtual model (for models that do not support instruct mode)")

	flag.Parse()

	// Detect explicitly set options so the deprecated -enforce-sampling-params option
	// can only act as a fallback.
	explicitFlags := map[string]bool{}
	flag.Visit(func(f *flag.Flag) {
		explicitFlags[f.Name] = true
	})
	_, samplingEnvSet := os.LookupEnv("QWEN38RP_SAMPLING_PARAMS")
	_, legacyEnvSet := os.LookupEnv("QWEN38RP_ENFORCE_SAMPLING_PARAMS")
	samplingExplicit := explicitFlags["sampling-params"] || samplingEnvSet
	legacyExplicit := explicitFlags["enforce-sampling-params"] || legacyEnvSet

	cfg.Listen = getEnvOrFlag(*listen, "QWEN38RP_LISTEN")
	cfg.Target = getEnvOrFlag(*target, "QWEN38RP_TARGET")
	cfg.LogLevel = getEnvOrFlag(*loglevel, "QWEN38RP_LOGLEVEL")
	cfg.ServedModelName = getEnvOrFlag(*servedModel, "QWEN38RP_SERVED_MODEL_NAME")

	var err error
	cfg.Port, err = getEnvOrFlagInt(*port, "QWEN38RP_PORT")
	if err != nil {
		return cfg, err
	}
	cfg.EnableExtendedModels, err = getEnvOrFlagBool(*enableExtended, "QWEN38RP_ENABLE_EXTENDED_MODELS")
	if err != nil {
		return cfg, err
	}
	cfg.EnableAutoModel, err = getEnvOrFlagBool(*enableAuto, "QWEN38RP_ENABLE_AUTO_MODEL")
	if err != nil {
		return cfg, err
	}
	legacyEnforce, err := getEnvOrFlagBool(*enforceSampling, "QWEN38RP_ENFORCE_SAMPLING_PARAMS")
	if err != nil {
		return cfg, err
	}
	if samplingExplicit && legacyExplicit {
		log.Printf("WARN both -sampling-params/QWEN38RP_SAMPLING_PARAMS and -enforce-sampling-params/QWEN38RP_ENFORCE_SAMPLING_PARAMS are set, the former wins")
	}
	globalPolicy, err := resolveSamplingParams(
		getEnvOrFlag(*samplingParams, "QWEN38RP_SAMPLING_PARAMS"), samplingExplicit, legacyEnforce,
	)
	if err != nil {
		return cfg, err
	}
	cfg.SamplingParams = string(globalPolicy)
	cfg.AutoSamplingParams = strings.TrimSpace(getEnvOrFlag(*autoSamplingParams, "QWEN38RP_AUTO_SAMPLING_PARAMS"))
	if cfg.AutoSamplingParams == "" {
		cfg.AutoSamplingParams = cfg.SamplingParams
	}
	_, autoPolicy, err := cfg.resolvedSamplingPolicies()
	if err != nil {
		return cfg, err
	}
	cfg.AutoSamplingParams = string(autoPolicy)
	cfg.AutoPreserveThinking, err = getEnvOrFlagBool(*autoPreserveThinking, "QWEN38RP_AUTO_PRESERVE_THINKING")
	if err != nil {
		return cfg, err
	}
	cfg.VirtualModelPrefix = getEnvOrFlag(*virtualModelPrefix, "QWEN38RP_VIRTUAL_MODEL_PREFIX")
	cfg.NoInstruct, err = getEnvOrFlagBool(*noInstruct, "QWEN38RP_NO_INSTRUCT")
	if err != nil {
		return cfg, err
	}

	return cfg, cfg.Validate()
}

func getEnvOrFlag(flagVal string, envName string) string {
	if envVal, exists := os.LookupEnv(envName); exists {
		return envVal
	}
	return flagVal
}

func getEnvOrFlagInt(flagVal int, envName string) (int, error) {
	if envVal, exists := os.LookupEnv(envName); exists {
		intVal, err := strconv.Atoi(envVal)
		if err != nil {
			return 0, fmt.Errorf("invalid value for %s=%q: %w", envName, envVal, err)
		}
		return intVal, nil
	}
	return flagVal, nil
}

func getEnvOrFlagBool(flagVal bool, envName string) (bool, error) {
	if envVal, exists := os.LookupEnv(envName); exists {
		boolVal, err := strconv.ParseBool(envVal)
		if err != nil {
			return false, fmt.Errorf("invalid value for %s=%q: %w", envName, envVal, err)
		}
		return boolVal, nil
	}
	return flagVal, nil
}

// parseLogLevel parses a log level string, including the COMPLETE level
func parseLogLevel(levelStr string) slog.Level {
	switch strings.ToUpper(levelStr) {
	case COMPLETE_LEVEL:
		return COMPLETE
	case "DEBUG":
		return slog.LevelDebug
	case "INFO":
		return slog.LevelInfo
	case "WARN":
		return slog.LevelWarn
	case "ERROR":
		return slog.LevelError
	default:
		return slog.LevelInfo
	}
}

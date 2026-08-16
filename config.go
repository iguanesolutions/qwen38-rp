package main

import (
	"errors"
	"flag"
	"fmt"
	"log/slog"
	"os"
	"strconv"
	"strings"
)

// COMPLETE is a log level more verbose than DEBUG for complete request/response dumps
const COMPLETE = slog.LevelDebug - 4
const COMPLETE_LEVEL = "COMPLETE"

type Config struct {
	Listen                string
	Port                  int
	Target                string
	LogLevel              string
	ServedModelName       string
	EnableExtendedModels  bool
	EnforceSamplingParams bool
	VirtualModelPrefix    string
	NoInstruct            bool
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
	return nil
}

func LoadConfig() (Config, error) {
	var cfg Config

	listen := flag.String("listen", "0.0.0.0", "IP address to listen on")
	port := flag.Int("port", 9000, "Port to listen on")
	target := flag.String("target", "http://127.0.0.1:8000", "Backend target, default is for a local vLLM")
	loglevel := flag.String("loglevel", slog.LevelInfo.String(), "Log level (COMPLETE, DEBUG, INFO, WARN, ERROR)")
	servedModel := flag.String("served-model", "", "Name of the served model")
	enableExtended := flag.Bool("enable-extended-models", false, "Enable extended pre-configured virtual models (low/medium/xhigh)")
	enforceSampling := flag.Bool("enforce-sampling-params", false, "Enforce sampling parameters, overriding client-provided values")
	virtualModelPrefix := flag.String("virtual-model-prefix", "qwen38", "Prefix for virtual model names exposed by the proxy")
	noInstruct := flag.Bool("no-instruct", false, "Disable the instruct virtual model (for models that do not support instruct mode)")

	flag.Parse()

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
	cfg.EnforceSamplingParams, err = getEnvOrFlagBool(*enforceSampling, "QWEN38RP_ENFORCE_SAMPLING_PARAMS")
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

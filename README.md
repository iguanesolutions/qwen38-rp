# Qwen 3.8 Reverse Proxy

Use every official Qwen 3.8 mode — **without changing your OpenAI clients**.

Qwen 3.8 natively supports several runtime modes (instruct, thinking, preserve thinking) and three reasoning effort levels. Activating them requires sending vendor-specific parameters such as `chat_template_kwargs` and `reasoning_effort`, which most standard OpenAI clients do not expose.

**qwen38-rp** is a lightweight HTTP reverse proxy designed specifically to sit **in front of a vLLM backend serving Qwen 3.8**. It exposes each official Qwen 3.8 mode as a distinct virtual model name. Your client simply picks the model — the proxy automatically injects the correct `chat_template_kwargs`, sampling parameters, and reasoning effort required by the backend.

## Core Functionality

This proxy's primary purpose is to:

1. **Expose Qwen 3.8's official modes as virtual model names**, so no client-side change is required:
   - **3 base models** (always exposed):
     - `qwen38-instruct` — Native instruct mode (no reasoning)
     - `qwen38-thinking` — Native thinking mode, `reasoning_effort` controllable by the client
     - `qwen38-thinking-preserve` — Native thinking mode with historical thinking preservation, `reasoning_effort` controllable by the client
   - **6 optional pre-configured models** (enabled via `-enable-extended-models`):
     - `qwen38-thinking-low`, `qwen38-thinking-medium`, `qwen38-thinking-xhigh`
     - `qwen38-thinking-preserve-low`, `qwen38-thinking-preserve-medium`, `qwen38-thinking-preserve-xhigh`
2. **Apply the official Qwen 3.8 sampling defaults** automatically for the selected mode:
   - **Thinking modes**: `temperature=1.0`, `top_p=0.95`, `top_k=20`, `min_p=0.0`, `presence_penalty=0.0`, `repetition_penalty=1.0`
   - **Instruct mode**: `temperature=0.7`, `top_p=0.80`, `top_k=20`, `min_p=0.0`, `presence_penalty=1.5`, `repetition_penalty=1.0`
3. **Activate the native thinking behavior** by injecting the `chat_template_kwargs` that Qwen 3.8 expects at runtime:
   - `enable_thinking=true` for thinking modes
   - `enable_thinking=false` for instruct modes
   - `preserve_thinking=true` for preserve-thinking modes
4. **Enforce the official reasoning effort** on pre-configured models:
   - When a pre-configured model is called (e.g., `qwen38-thinking-low`), the proxy **always overrides** any client-provided `reasoning_effort` with the official value.
   - For base thinking models (`qwen38-thinking`, `qwen38-thinking-preserve`), the proxy **does not touch** `reasoning_effort` if absent — the backend applies its own default.
5. **Rewrite the model name** to the actual backend model name (e.g., `Qwen/Qwen3.8-27B`) before forwarding to vLLM
6. **Fix vLLM response bugs** where non-thinking, non-streaming responses incorrectly place content in `reasoning_content` or `reasoning` fields instead of `content`
7. **Enrich `/v1/models` endpoint** by fetching backend models and exposing virtual models with the same metadata (permissions, max_model_len, etc.)
8. **Provide a `/tokenize` endpoint** that replaces virtual model names with the backend model name and injects the matching `chat_template_kwargs` before forwarding to vLLM's `/tokenize`

## Installation

Requirements: Go 1.24.2 or later

```bash
go build -o qwen38-rp .
```

## Usage

```bash
./qwen38-rp \
  -target "http://127.0.0.1:8000" \
  -served-model "Qwen/Qwen3.8-27B" \
  -enable-extended-models
```

Or using environment variables:

```bash
export QWEN38RP_TARGET="http://127.0.0.1:8000"
export QWEN38RP_SERVED_MODEL_NAME="Qwen/Qwen3.8-27B"
export QWEN38RP_ENABLE_EXTENDED_MODELS="true"
./qwen38-rp
```

## Configuration

Configure the proxy using command-line flags or environment variables:

| Flag | Environment Variable | Default | Description |
|------|---------------------|---------|-------------|
| `-listen` | `QWEN38RP_LISTEN` | `0.0.0.0` | IP address to listen on |
| `-port` | `QWEN38RP_PORT` | `9000` | Port to listen on |
| `-target` | `QWEN38RP_TARGET` | `http://127.0.0.1:8000` | Backend target URL |
| `-loglevel` | `QWEN38RP_LOGLEVEL` | `INFO` | Log level (COMPLETE, DEBUG, INFO, WARN, ERROR) |
| `-served-model` | `QWEN38RP_SERVED_MODEL_NAME` | (required) | Backend model name to use in outgoing requests |
| `-enable-extended-models` | `QWEN38RP_ENABLE_EXTENDED_MODELS` | `false` | Expose the 6 pre-configured models (low/medium/xhigh) |
| `-enforce-sampling-params` | `QWEN38RP_ENFORCE_SAMPLING_PARAMS` | `false` | Enforce sampling parameters, overriding client-provided values |

### Virtual Model Behavior

#### Base Models (always available)

| Model | `enable_thinking` | `preserve_thinking` | `reasoning_effort` | Sampling |
|-------|-------------------|---------------------|--------------------|----------|
| `qwen38-instruct` | `false` | — | — | Instruct |
| `qwen38-thinking` | `true` | `false` | Client-controlled | Thinking |
| `qwen38-thinking-preserve` | `true` | `true` | Client-controlled | Thinking |

#### Extended Models (requires `-enable-extended-models`)

| Model | `enable_thinking` | `preserve_thinking` | `reasoning_effort` |
|-------|-------------------|---------------------|--------------------|
| `qwen38-thinking-low` | `true` | `false` | `low` (immutable) |
| `qwen38-thinking-medium` | `true` | `false` | `medium` (immutable) |
| `qwen38-thinking-xhigh` | `true` | `false` | `xhigh` (immutable) |
| `qwen38-thinking-preserve-low` | `true` | `true` | `low` (immutable) |
| `qwen38-thinking-preserve-medium` | `true` | `true` | `medium` (immutable) |
| `qwen38-thinking-preserve-xhigh` | `true` | `true` | `xhigh` (immutable) |

**Golden rule:** pre-configured models are an immutable contract. The proxy always overrides `reasoning_effort` when they are called. This allows clients that do not expose `reasoning_effort` to access predefined reasoning profiles by simply selecting the appropriate model.

### Enforce Sampling Parameters

By default, the proxy only sets sampling parameters if they are not already present in the request. When `-enforce-sampling-params` is enabled, the proxy will **always override** client-provided sampling parameters with the predefined values for the detected mode.

## Request Routing

- **`GET /v1/models`**: Enriched (fetches backend models, validates served model, exposes virtual models)
- **`POST /v1/responses`**: Returns HTTP 501 Not Implemented by design (vLLM's Responses API doesn't support `chat_template_kwargs` needed to configure thinking mode and `preserve_thinking`, which defaults to `true` in Qwen 3.8)
- **`POST /v1/chat/completions`**: Transformed (sampling params + thinking mode + reasoning effort applied)
- **`POST /v1/completions`**: Model name validated and swapped (no sampling params or thinking mode — raw prompt completions bypass the chat template)
- **`POST /tokenize`**: Replaces virtual model names with backend model name, injects matching `chat_template_kwargs`, and forwards to vLLM's `/tokenize`
- **All other paths**: Passed through unchanged to the backend

## OpenAI SDK Examples

### Base model (client controls reasoning_effort)

When using a base model, the proxy injects `chat_template_kwargs` automatically. You only need to choose the model and optionally set `reasoning_effort`:

```python
completion = client.chat.completions.create(
    model="qwen38-thinking-preserve",
    messages=messages,
    reasoning_effort="xhigh",  # client-controlled for base models
    stream=True,
    stream_options={"include_usage": True},
)
```

### Pre-configured model (reasoning_effort is enforced by the proxy)

For clients that expose a limited or mismatched set of `reasoning_effort` levels, use a pre-configured model. The proxy sets the correct effort automatically — any value sent by the client is ignored:

```python
completion = client.chat.completions.create(
    model="qwen38-thinking-medium",  # reasoning_effort="medium" is automatic
    messages=messages,
    stream=True,
    stream_options={"include_usage": True},
)
```

## Responses API

The `/v1/responses` endpoint returns HTTP 501 Not Implemented by design. vLLM's Responses API endpoint does not support `chat_template_kwargs`, which is required to control Qwen's thinking mode (`enable_thinking`) and `preserve_thinking`. In Qwen 3.8, `preserve_thinking` defaults to `true`, so without `chat_template_kwargs` the backend would remain in preserve-thinking mode regardless of the virtual model selected. The proxy cannot apply the predefined profiles that adjust sampling parameters and thinking behavior. Use the Chat Completions API (`/v1/chat/completions`) instead.

### vLLM Backend Requirements

For full functionality with thinking mode and tool calls using the Chat Completions API, the vLLM backend should be started with the following flags:

```bash
--reasoning-parser=qwen3                                  # Required for thinking/reasoning mode
--enable-auto-tool-choice --tool-call-parser=qwen3_coder  # Required for tool/function calls
```

## Tokenize API

The proxy provides a `/tokenize` endpoint that forwards tokenization requests to vLLM's `/tokenize`. The proxy replaces virtual model names with the backend served model name, injects `chat_template_kwargs` according to the virtual model profile (so that the chat template produces the same text as it would for chat completions), then forwards the request. Two modes:

- **`{"prompt": "..."}`** — raw text tokenization, forwarded as-is. No chat template is applied.
- **`{"messages": [...], "tools": [...]}`** — vLLM applies the model's chat template (`apply_chat_template`) then tokenizes the result. Messages and tools must be in Chat Completions API format (same as transformers `apply_chat_template`): a list of dictionaries with `role` and `content` keys.

## Health Check

- **`GET /health`**: Returns `{"status":"healthy"}` for Docker health checks

## Log Levels

The proxy supports the following log levels:

| Level | Description |
|-------|-------------|
| `COMPLETE` | Most verbose - includes full HTTP request/response dumps |
| `DEBUG` | Debug information including parameter application details |
| `INFO` | General operational information |
| `WARN` | Warning messages |
| `ERROR` | Error messages only |

When set to `COMPLETE`, the proxy will log full HTTP request and response bodies, which is useful for debugging but very verbose.

⚠️ **Privacy Warning**: LLM requests often contain sensitive or personal data (conversation history, personal information, confidential content). The `COMPLETE` log level will expose all this data in plaintext. Only enable it in secure, non-production environments or ensure logs are properly secured and retained temporarily.

## systemd Integration

The proxy includes native systemd support for production deployments:

- **Type**: `notify` - The proxy signals readiness to systemd automatically
- **Status Updates**: Sends periodic status updates to systemd showing processed request counts
- **Graceful Shutdown**: Properly signals systemd when stopping
- **Journald Logging**: Structured logging output is compatible with journald

Example systemd unit file:

```ini
[Unit]
Description=Qwen 3.8 Reverse Proxy
After=network.target

[Service]
Type=notify
User=qwen38-rp
Group=qwen38-rp
ExecStart=/usr/local/bin/qwen38-rp -served-model "Qwen/Qwen3.8-27B" -enable-extended-models
Restart=on-failure
Environment=QWEN38RP_LOGLEVEL=INFO

[Install]
WantedBy=multi-user.target
```

⚠️ **Security Best Practice**: Always run the proxy under a dedicated, unprivileged user account (e.g., `qwen38-rp`). Never run as root. Create the user with:
```bash
sudo useradd --system --no-create-home --shell /usr/sbin/nologin qwen38-rp
sudo chown qwen38-rp:qwen38-rp /usr/local/bin/qwen38-rp
```

## Graceful Shutdown

The server supports graceful shutdown with a 3-minute timeout to allow in-flight requests to complete. Send `SIGINT` or `SIGTERM` to initiate shutdown. When running under systemd, the proxy will automatically signal the service manager when ready and during shutdown.

## License

MIT License - see [LICENSE](LICENSE) file for details.

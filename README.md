# Qwen 3.8 Reverse Proxy

Use every official Qwen 3.8 mode — **without changing your OpenAI clients**.

Qwen 3.8 natively supports several runtime modes (instruct, thinking, preserve thinking) and three reasoning effort levels. Activating them requires sending vendor-specific parameters such as `chat_template_kwargs`, which most standard OpenAI clients do not expose.

**qwen38-rp** is a lightweight HTTP reverse proxy designed specifically to sit **in front of a vLLM backend serving Qwen 3.8**. It exposes each official Qwen 3.8 mode as a distinct virtual model name. Your client simply picks the model — the proxy automatically injects the correct `chat_template_kwargs`, sampling parameters, and reasoning effort required by the backend.

## Scope

This proxy is designed for Qwen 3.8's Chat Completions and Legacy Completions APIs.

## Architecture & Design Philosophy

This proxy is a **vLLM enhancement layer**. It sits directly in front of a vLLM instance to inject Qwen 3.8-specific parameters, but otherwise preserves native vLLM behavior. It does not terminate TLS, authenticate clients, or rate limit, but it passes all headers—including `Authorization`—through to vLLM untouched. It shares vLLM's exposure boundary: do not expose this proxy to any network you would not expose vLLM itself to.

- **No artificial timeouts:** LLM inference duration is inherently unpredictable. The proxy does not impose global request or response timeouts. Client disconnects cancel the request context, which propagates upstream immediately.
- **Full body materialization:** Request bodies are always read entirely into memory to allow JSON rewriting. Non-streaming responses are also read in full to apply model-name fixes and vLLM bug workarounds. Streaming responses are processed incrementally (event by event) and are not held in memory.
- **Single-target only:** The proxy forwards to one vLLM backend. It is not a load balancer.

## Core Functionality

This proxy's primary purpose is to:

1. **Expose Qwen 3.8's official modes as virtual model names**, so no client-side change is required:
   - **Auto model** (enabled via `-enable-auto-model`):
     - `qwen38-auto` — derives the mode (and its sampling defaults) from the request, for clients that can only express `reasoning_effort`
   - **Up to 3 base models** (instruct can be disabled via `-no-instruct`):
     - `qwen38-instruct` — Native instruct mode (no reasoning)
     - `qwen38-thinking` — Native thinking mode, `reasoning_effort` controllable by the client
     - `qwen38-thinking-preserve` — Native thinking mode with historical thinking preservation, `reasoning_effort` controllable by the client
   - **6 optional pre-configured models** (enabled via `-enable-extended-models`):
     - `qwen38-thinking-low`, `qwen38-thinking-medium`, `qwen38-thinking-xhigh`
     - `qwen38-thinking-preserve-low`, `qwen38-thinking-preserve-medium`, `qwen38-thinking-preserve-xhigh`
   - The prefix (`qwen38`) is customizable via `-virtual-model-prefix` to differentiate deployments when running multiple Qwen 3.8 variants.
2. **Apply the official Qwen 3.8 sampling defaults** automatically for the selected mode:
   - **Thinking modes**: `temperature=1.0`, `top_p=0.95`, `top_k=20`, `min_p=0.0`, `presence_penalty=0.0`, `repetition_penalty=1.0`
   - **Instruct mode**: `temperature=0.7`, `top_p=0.80`, `top_k=20`, `min_p=0.0`, `presence_penalty=1.5`, `repetition_penalty=1.0`
3. **Activate the native thinking behavior** by injecting the `chat_template_kwargs` that Qwen 3.8 expects at runtime:
   - `enable_thinking=true` for thinking modes
   - `enable_thinking=false` for instruct modes
   - `preserve_thinking=true` for preserve-thinking modes
   - The auto model is the exception: it never writes `enable_thinking` (vLLM derives it from `reasoning_effort`) and only pins `preserve_thinking`.
4. **Enforce the official reasoning effort** on pre-configured models:
   - When a pre-configured model is called (e.g., `qwen38-thinking-low`), the proxy **always overrides** any client-provided `reasoning_effort` with the official value.
   - For base thinking models (`qwen38-thinking`, `qwen38-thinking-preserve`), the proxy **does not touch** `reasoning_effort` — if the client sends a value it is passed through unchanged; if it is absent the backend applies its own default.
5. **Rewrite the model name** to the actual backend model name (e.g., `Qwen/Qwen3.8-27B`) before forwarding to vLLM
6. **Fix vLLM response bugs** where non-thinking, non-streaming responses incorrectly place content in `reasoning_content` or `reasoning` fields instead of `content`
7. **Enrich `/v1/models` endpoint** by fetching backend models and exposing virtual models with the same metadata (permissions, max_model_len, etc.)
8. **Provide a `/tokenize` endpoint** that replaces virtual model names with the backend model name and injects the matching `chat_template_kwargs` before forwarding to vLLM's `/tokenize` (for the auto model, only `preserve_thinking` is pinned, so the rendered template matches the chat request)

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
  -virtual-model-prefix "qwen38-27B" \
  -enable-auto-model \
  -enable-extended-models
```

For the 2.4T-A95B variant (max model):

```bash
./qwen38-rp \
  -target "http://127.0.0.1:8000" \
  -served-model "Qwen/Qwen3.8-2.4T-A95B" \
  -virtual-model-prefix "qwen38-2.4T" \
  -no-instruct \
  -enable-extended-models
```

> ⚠️ **Max model (2.4T-A95B) strongly recommends keeping thinking enabled.** Disabling thinking on this variant is not supported and produces poor results. Always use `-no-instruct` when proxying the max model so that clients cannot accidentally select an instruct (non-thinking) profile.

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
| `-enable-auto-model` | `QWEN38RP_ENABLE_AUTO_MODEL` | `false` | Expose the mode-deriving `<prefix>-auto` model |
| `-sampling-params` | `QWEN38RP_SAMPLING_PARAMS` | `fill` | Sampling policy for every virtual model: `off`, `fill`, or `enforce` |
| `-auto-sampling-params` | `QWEN38RP_AUTO_SAMPLING_PARAMS` | (same as `-sampling-params`) | Sampling policy for `<prefix>-auto` only |
| `-auto-preserve-thinking` | `QWEN38RP_AUTO_PRESERVE_THINKING` | `true` | Pin `preserve_thinking` on `<prefix>-auto` (set to `false` for compact context) |
| `-enforce-sampling-params` | `QWEN38RP_ENFORCE_SAMPLING_PARAMS` | `false` | **Deprecated**: `true` maps to `-sampling-params=enforce`, `false` to `fill` |
| `-virtual-model-prefix` | `QWEN38RP_VIRTUAL_MODEL_PREFIX` | `qwen38` | Prefix for virtual model names (e.g. `qwen38-27B`) |
| `-no-instruct` | `QWEN38RP_NO_INSTRUCT` | `false` | Disable the instruct virtual model. **Required for the 2.4T-A95B max model**, which does not support instruct mode and recommends keeping thinking enabled. |

### Virtual Model Layers

All virtual models are named `<prefix>-<suffix>` and come in three layers:

| Layer | Models | Mode | `reasoning_effort` |
|-------|--------|------|--------------------|
| auto | `<prefix>-auto` | derived per request | read to select the mode, forwarded untouched |
| fixed | `<prefix>-instruct`, `<prefix>-thinking`, `<prefix>-thinking-preserve` | pinned | passed through (thinking), ignored (instruct) |
| optional fixed | `<prefix>-thinking[-preserve]-{low,medium,xhigh}` | pinned | pinned (immutable) |

### Virtual Model Behavior

#### Auto Model (requires `-enable-auto-model`)

`<prefix>-auto` targets clients that can express `reasoning_effort` but cannot send `chat_template_kwargs` (typically OpenAI SDK users). The proxy derives the Qwen mode from the request and applies the matching sampling defaults.

Resolution order:

1. `chat_template_kwargs.enable_thinking`, when present and boolean, wins: `true` → thinking, `false` → instruct. A non-boolean value is ignored.
2. Otherwise, `reasoning_effort: "none"` → instruct mode.
3. Otherwise → thinking mode.

| Resolved mode | `enable_thinking` | `preserve_thinking` | Sampling |
|---------------|-------------------|---------------------|----------|
| thinking (default, or `effort` ≠ `none`) | **never written by the proxy** | `true` by default | Thinking |
| instruct (`effort: none`, or `enable_thinking: false`) | **never written by the proxy** | `true` by default | Instruct |

In auto mode the proxy **must not** write `enable_thinking`: vLLM derives it itself (`enable_thinking = reasoning_effort != "none"`), and injecting it re-creates the `thinking + none → 400` trap. The proxy only fixes sampling parameters.

`preserve_thinking` is pinned to `true` by default, for three reasons:

1. **It is the vendor default.** The Qwen 3.8 template treats an undefined `preserve_thinking` as `true`. A model named *auto* should adapt sampling, not silently change prompt semantics, otherwise it behaves differently from raw vLLM in a way clients cannot anticipate.
2. **It keeps the prefix cache warm.** With `true`, turn N+1 contains turn N's assistant message byte-identically (thinking included), so the backend reuses the cached prefix and only pre-fills the new user turn. With `false` the prompt diverges at the start of the last assistant message, forcing a re-prefill of the assistant content *plus* the new user turn on **every** follow-up.
3. **It does not duplicate an existing option.** `<prefix>-thinking` already means interleaved (thinking stripped between turns) and `<prefix>-thinking-preserve` means preserved; `-auto` is raw vLLM behaviour plus adapted sampling.

Pinning it is legal whatever the mode, because vLLM's contradiction guard only inspects `enable_thinking`. Pass `-auto-preserve-thinking=false` (or `QWEN38RP_AUTO_PRESERVE_THINKING=false`) when compact context matters more than cache reuse.

`/tokenize` pins the same `preserve_thinking` value, from the same helper the chat path uses, so counting tokens on a multi-turn conversation gives the same result as what the chat request actually renders.

Edge cases:

- **`-no-instruct` + a request resolving to instruct mode** (`reasoning_effort: "none"` or `enable_thinking: false`) → HTTP 400 with an explicit message. Instruct sampling is never applied silently on a deployment without instruct support.
- **Contradictory request** (`enable_thinking: true` + `reasoning_effort: "none"`) → forwarded untouched, so vLLM's own template validation returns the error. The proxy never changes these semantics.
- **`/v1/completions`** → same as any other virtual model: the name is validated and swapped, nothing else is touched (no chat template is applied there).
- **`/tokenize`** → the name is swapped and the same `chat_template_kwargs` are pinned as on the chat path (`preserve_thinking` only, never `enable_thinking`), so token counts match the rendered prompt. A request the chat path would reject is rejected here too: with `-no-instruct`, an auto request resolving to instruct mode returns HTTP 400 rather than a token budget for an unservable request. The check is mode-based, so it also applies to raw `prompt` tokenization.

#### Base Models

The instruct model is omitted when `-no-instruct` is set. This is **required for the 2.4T-A95B max model**, which does not support instruct mode and strongly recommends keeping thinking enabled at all times.

| Model | `enable_thinking` | `preserve_thinking` | `reasoning_effort` | Sampling |
|-------|-------------------|---------------------|--------------------|----------|
| `<prefix>-instruct` | `false` | — | — | Instruct |
| `<prefix>-thinking` | `true` | `false` | Client-controlled | Thinking |
| `<prefix>-thinking-preserve` | `true` | `true` | Client-controlled | Thinking |

#### Extended Models (requires `-enable-extended-models`)

| Model | `enable_thinking` | `preserve_thinking` | `reasoning_effort` |
|-------|-------------------|---------------------|--------------------|
| `<prefix>-thinking-low` | `true` | `false` | `low` (immutable) |
| `<prefix>-thinking-medium` | `true` | `false` | `medium` (immutable) |
| `<prefix>-thinking-xhigh` | `true` | `false` | `xhigh` (immutable) |
| `<prefix>-thinking-preserve-low` | `true` | `true` | `low` (immutable) |
| `<prefix>-thinking-preserve-medium` | `true` | `true` | `medium` (immutable) |
| `<prefix>-thinking-preserve-xhigh` | `true` | `true` | `xhigh` (immutable) |

**Golden rule:** extended models are an immutable contract. The proxy always overrides `reasoning_effort` when they are called. This allows clients that do not expose `reasoning_effort` to access predefined reasoning profiles by simply selecting the appropriate model.

### Sampling Parameters Policy

`-sampling-params` selects how the official sampling defaults of the detected mode are applied:

| Value | Behaviour |
|-------|-----------|
| `off` | never modify sampling parameters |
| `fill` | set only the keys the client omitted (default) |
| `enforce` | always overwrite the client's values |

`-auto-sampling-params` overrides the policy for `<prefix>-auto` only, and defaults to the global policy when unset.

The deprecated `-enforce-sampling-params` / `QWEN38RP_ENFORCE_SAMPLING_PARAMS` option is still honoured: `true` maps to `enforce`, `false` to `fill`. When both options are set, `-sampling-params` wins and a single warning is logged at startup.

## Request Routing

- **`GET /v1/models`**: Enriched (fetches backend models, validates served model, exposes virtual models)
- **`POST /v1/responses`**: Returns HTTP 501 Not Implemented by design (vLLM's Responses API doesn't support `chat_template_kwargs` needed to configure thinking mode and `preserve_thinking`, which defaults to `true` in Qwen 3.8)
- **`POST /v1/chat/completions`**: Transformed (sampling params + thinking mode + reasoning effort applied; `<prefix>-auto` additionally derives the mode from `reasoning_effort`)
- **`POST /v1/completions`**: Model name validated and swapped (no sampling params or thinking mode — raw prompt completions bypass the chat template)
- **`POST /tokenize`**: Replaces virtual model names with backend model name, injects matching `chat_template_kwargs` (for `<prefix>-auto`, `preserve_thinking` only), and forwards to vLLM's `/tokenize`
- **All other paths**: Passed through unchanged to the backend

## OpenAI SDK Examples

### Auto model (mode derived from reasoning_effort)

With `-enable-auto-model`, a plain OpenAI client gets the right mode and the right sampling defaults without any vendor-specific parameter:

```python
completion = client.chat.completions.create(
    model="qwen38-auto",
    messages=messages,
    reasoning_effort="none",  # instruct mode + instruct sampling defaults
    stream=True,
    stream_options={"include_usage": True},
)
```

Omit `reasoning_effort` (or set any other supported level) to get thinking mode with thinking sampling defaults.

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
ExecStart=/usr/local/bin/qwen38-rp -served-model "Qwen/Qwen3.8-27B" -virtual-model-prefix "qwen38-27B" -enable-extended-models
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

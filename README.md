# mr_kyutai

Drop-in replacement for `mr_eleven_stream` using Kyutai's streaming TTS (via the `moshi` Python package).

## Enable realtime incremental input

Realtime incremental input is **enabled by default**.

```bash
export MR_KYUTAI_REALTIME_STREAM=0   # to disable
```

### Environment variables

- `MR_KYUTAI_HF_REPO` (default: `kyutai/tts-1.6b-en_fr`)
- `MR_KYUTAI_VOICE_REPO` (default: `kyutai/tts-voices`)
- `MR_KYUTAI_VOICE` (default: `expresso/ex03-ex01_happy_001_channel1_334s.wav`)
- `MR_KYUTAI_DEVICE` (default: `cuda`, falls back to `cpu` if unavailable; **local inference only**)
- `MR_KYUTAI_REALTIME_STREAM` (default: enabled). Set to `0` to disable the partial-command realtime pipeline.
- `KYUTAI_REMOTE` (optional): if set, **do not run TTS locally**.
  - Default remote mode is **moshi-server WebSockets** (recommended):
    - `host:port` (interpreted as `ws://host:port`)
    - `ws://host:port`
    - `wss://host:port`
  - Legacy TCP mode (included reference server): prefix with `tcp://`:
    - `tcp://host:port`
    - (If port omitted in TCP mode, defaults to `8765`.)
- `KYUTAI_API_KEY` (default: `public_token`) for moshi-server auth (header `kyutai-api-key`).

## Notes

- Output to SIP is **ulaw 8 kHz**.
- Kyutai generates 24 kHz PCM internally; we resample to 8 kHz and mu-law encode before sending to SIP.

## Remote server

This plugin includes a small reference remote server that speaks a simple framed-TCP protocol:

1. Client streams JSON control frames: `{"op":"start", ...}`, then repeated `{"op":"text","text":"..."}`, then `{"op":"finish"}`
2. Server streams back audio frames as raw **ulaw 8 kHz** bytes (usually 20ms/160B chunks), then an `end` frame.

### Run the remote server (GPU box)

```bash
export MR_KYUTAI_DEVICE=cuda
export MR_KYUTAI_HF_REPO=kyutai/tts-1.6b-en_fr
export MR_KYUTAI_VOICE=expresso/ex03-ex01_happy_001_channel1_334s.wav

python -m mr_kyutai.remote_server --host 0.0.0.0 --port 8765
```

### Use it from MindRoot host

```bash
export KYUTAI_REMOTE=tcp://10.0.0.23:8765
```

### Alternative: official Kyutai production server

Kyutai also provides a production-grade server (`moshi-server`) that exposes streaming TTS over WebSockets
(see https://github.com/kyutai-labs/delayed-streams-modeling and the `config-tts.toml` / `moshi-server worker` docs).
This plugin's `KYUTAI_REMOTE` supports both the included TCP server and moshi-server (`ws://` / `wss://`) modes.

---

## Batched TTS Server

A Python WebSocket + TCP server that wraps Kyutai's `TTSService` (from `tts.py`, ported from the Rust `moshi-server`) for **concurrent batched inference**. Multiple client connections share a single GPU inference loop, with each connection assigned to a batch slot.

This is the recommended server for production use with multiple concurrent sessions.

### Features

- **WebSocket protocol** (msgpack binary) — matches the official `moshi-server` Rust implementation
- **Legacy TCP protocol** — compatible with the framed J/A/E/X protocol from `remote_server.py`
- **Batched inference** — up to `batch_size` concurrent sessions on one GPU
- **TextTokenizer** — Python port of the Rust `tts_preprocess.rs` (normalization, BOS, `<break>` tags)

### Install

The batched server requires the `moshi` package (for the TTS model) plus a few extras:

```bash
# Install mr_kyutai with local inference dependencies
pip install -e ".[local]"

# Or install dependencies individually:
pip install moshi>=0.2.11 torch numpy websockets msgpack
```

> **Note:** `moshi` and `torch` are only needed on the GPU server running the batched server. MindRoot hosts connecting as clients only need `websockets` and `msgpack`.

### Run

```bash
# Basic usage (defaults: ws://0.0.0.0:8765, batch_size=8)
python -m mr_kyutai.batched_tts_server

# Custom options
python -m mr_kyutai.batched_tts_server \
    --host 0.0.0.0 \
    --ws-port 8765 \
    --tcp-port 8766 \
    --batch-size 8 \
    --log-level INFO
```

### Environment variables

| Variable | Default | Description |
|---|---|---|
| `MR_KYUTAI_HF_REPO` | `kyutai/tts-1.6b-en_fr` | HuggingFace model repo |
| `MR_KYUTAI_DEVICE` | `cuda` | Device (`cuda` or `cpu`) |
| `MR_KYUTAI_BATCH_SIZE` | `8` | Max concurrent sessions |
| `MR_KYUTAI_WS_PORT` | `8765` | WebSocket listen port |
| `MR_KYUTAI_AUTH_TOKEN` | `public_token` | Auth token for API key header |
| `MR_KYUTAI_VOICE` | `expresso/ex03-ex01_happy_001_channel1_334s.wav` | Default voice |
| `MR_KYUTAI_VOICE_FOLDER` | `hf-snapshot://kyutai/tts-voices/**/*.safetensors` | Voice folder path |
| `MR_KYUTAI_DEFAULT_VOICE` | `unmute-prod-website/default_voice.wav` | Default voice name |
| `MR_KYUTAI_NQ` | `24` | Number of codebooks |
| `MR_KYUTAI_TEXT_BOS_TOKEN` | `1` | BOS token ID |
| `LOG_LEVEL` | `INFO` | Logging level |

### Connect from MindRoot

Set `KYUTAI_REMOTE` to point to the WebSocket server:

```bash
# WebSocket mode (recommended for batched server)
export KYUTAI_REMOTE=ws://10.0.0.23:8765

# Or with auth token
export KYUTAI_REMOTE=ws://10.0.0.23:8765
export KYUTAI_API_KEY=your_token
```

For the legacy TCP protocol, use `tcp://`:

```bash
export KYUTAI_REMOTE=tcp://10.0.0.23:8766
```

### WebSocket protocol (msgpack)

Matches the official `moshi-server` Rust implementation:

**Client → Server** (msgpack binary frames):

| Field | Description |
|---|---|
| `{type: "Text", text: "word"}` | Send a word token for synthesis |
| `{type: "Eos"}` | Signal end of input |
| `{type: "Voice", embeddings: [...], shape: [...]}` | Set custom voice embeddings |

**Server → Client** (msgpack binary frames):

| Field | Description |
|---|---|
| `{type: "Ready"}` | Connection accepted, slot allocated |
| `{type: "Audio", pcm: [float32...]}` | 1920 float32 samples @ 24 kHz per frame |
| `{type: "Error", message: "..."}` | Error occurred |

Voice can also be selected via query params: `?voice=name&format=PcmMessagePack`

Auth via `kyutai-api-key` header or `auth_id` query param.

### Legacy TCP protocol

Same framed J/A/E/X protocol as `remote_server.py`:

- Client sends `J` frames with JSON: `{"op":"start"}`, `{"op":"text","text":"..."}`, `{"op":"finish"}`
- Server responds with `A` frames (ulaw 8 kHz audio), `E` (end), `X` (error)

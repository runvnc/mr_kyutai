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

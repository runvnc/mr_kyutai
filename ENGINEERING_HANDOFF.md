# Engineering Handoff: mr_kyutai + Smart Turn v3 + Cohere Transcribe

**Date:** 2026-05-08
**Session:** Debugging and fixing smart_turn_v3 STT + mr_kyutai TTS integration on Nebius H200

## Project Locations

| Component | Path |
|---|---|
| mr_sip (STT/SIP) | `/xfiles/update_plugins/mr_sip` |
| mr_kyutai (TTS plugin) | `/xfiles/plugins_ah/mr_kyutai` |
| Container/deploy | `/files/upd6/mr_verification_dashboard/containers/vllm_qwen35_35b_a3b` |
| Moshi repo (cloned) | `/tmp/moshi-rust` (may need re-clone) |
| Kyutai batched TTS source | `/xfiles/plugins_ah/mr_kyutai/tts.py` (copied from moshi-server) |

## Current Architecture

```
[SIP Call] -> [mr_sip/sip_client_v2] -> [smart_turn_v3_stt] -> [Silero VAD + Smart Turn ONNX + Cohere Transcribe]
                                                                         |
                                                                    [MindRoot Agent]
                                                                         |
[SIP Audio Out] <- [AudioPacer] <- [mr_kyutai plugin] <- [Kyutai remote_server.py (TCP)]
```

### Container Stack (H200, 141GB VRAM)
- **LLM:** Qwen3.6-27B INT4 AutoRound (~19GB) on vLLM, port 8000
- **TTS:** Kyutai TTS 1.8B via `remote_server.py` (TCP, port 8765) (~7GB)
- **STT:** Cohere Transcribe 2B via `cohere_transcribe_server.py` (HTTP, port 8881) (~2GB)
- **MindRoot:** Voice agent platform, port 8010
- **Available VRAM:** ~103GB unused

## Fixes Applied This Session

### 1. Smart Turn v3 STT (`mr_sip`, commits a072a02, e99178f)

**Problem A:** Turn detection fired after only 100ms of speech (false positives).
- Pre-roll buffer (2304 bytes) exceeded the minimum check (1024 bytes)
- Model saw 7.9s silence padding + 0.1s speech → "turn complete" at prob=0.976

**Fix:** Added `SMART_TURN_MIN_SPEECH_MS` env var (default 500ms). Poll loop skips inference until enough speech elapsed.

**Problem B:** Transcription never completed - self-cancellation bug.
- `_stop_polling()` called `self._poll_task.cancel()` from INSIDE the poll task
- This killed the `await run_in_executor(transcribe)` call via CancelledError
- Logs showed `[TRANSCRIBE] Starting...` then `[SMART_TURN] Poll loop cancelled` with no result

**Fix:** `_stop_polling()` now only sets `_poll_active = False`. The loop exits naturally after transcription completes. `finally` block cleans up `_poll_task = None`.

**Problem C:** Smart Turn ONNX loaded on CPU only (`providers=['CPUExecutionProvider']`).
- The MindRoot venv has `onnxruntime-gpu` installed but CUDA provider not loading
- Needs investigation - may need `onnxruntime-gpu` reinstall or CUDA path fix

### 2. mr_kyutai Plugin (commits df5eaff through 70868dc)

**Problem A:** Plugin pipe never registered.
- `__init__.py` only imported `mod.py`, never `realtime_stream.py`
- The `@pipe(name="partial_command")` decorator never ran

**Fix:** Added `from . import realtime_stream` in `__init__.py`, wrapped in try/except for standalone server mode.

**Problem B:** Standalone server crashed on import.
- `python -m mr_kyutai.remote_server` imports `__init__.py` which imports `mod.py` which needs MindRoot
- MindRoot not available in standalone server process

**Fix:** try/except ImportError in `__init__.py`.

**Problem C:** `previous_text` not reset between speak commands.
- First speak: `previous_text='Hi, my name is Katie.'`
- Second speak starts fresh: `text='The'` (shorter) → no delta sent
- TTS thread connected but never received text → server timeout

**Fix:** Reset `previous_text` when text_len=0 (new command start) or when new text isn't a prefix of previous.

**Problem D:** Session never finished - no flush of remaining text.
- `partial_command` pipe feeds deltas, but nobody called `session.finish()` when speak completed
- Remaining buffered text never flushed, TTS thread never got `_END`
- No `post_command` pipe exists in MindRoot

**Fix:** `speak()` command in `mod.py` now calls `session.finish()` when executed (after all partials).

**Problem E:** "is already streaming!" on second session.
- Shared global model's streaming state not cleaned up between sessions
- `streaming_forever()` enters context that was never exited

**Fix:** 
- `_KyutaiGen.cleanup()` calls `__exit__` on streaming context
- `_force_reset_streaming()` walks all sub-modules clearing `_streaming_state`
- Global `_model_lock` serializes sessions

### 3. Deploy/Dockerfile
- Added `REQUIRE_DEEPGRAM=false` to Dockerfile env vars
- Added conditional `REQUIRE_DEEPGRAM` in `deploy_unified.py` based on STT_PROVIDER

## NEXT STEPS (Priority Order)

### 1. 🔴 Replace remote_server.py with Batched TTS Server

**Discovery:** Kyutai's production TTS server (`moshi-server/tts.py`, 582 lines) implements **batched inference** with `batch_size=32`, `exec_mask` for selective processing, and per-slot `ClientState`. This is what Unmute uses. Even the Rust server uses a mutex for TTS - batching is the only way to get concurrency.

**Files copied to mr_kyutai:**
- `tts.py` - Batched TTSService (the key file)
- `voice.py` - Voice management utilities  
- `batched_asr.py` - Batched ASR (reference only)

**Architecture for new server:**
```python
# Pseudocode for batched_tts_server.py
service = TTSService(batch_size=32, ...)  # One shared instance

# WebSocket/TCP handler assigns connection to a free batch slot
# Main loop calls service.step() at ~12.5Hz (80ms per frame)
# Each step processes ALL active slots in one forward pass
# PCM output routed back to respective connections
```

**Key API of TTSService.step():**
```python
service.step(
    updates=[(slot_idx, token_list, voice_name_or_embedding), ...],
    pcm_out=np.ndarray,      # (batch_size, 1920) float32 24kHz
    flags_out=np.ndarray,    # (batch_size,) int32 bitmask
    code_out=np.ndarray,     # (batch_size, 33) int32
)
```

**Token protocol:**
- `[-1, tok1, tok2, ...]` = reset slot + push first word tokens
- `[tok1, tok2, ...]` = push word tokens to existing slot
- `[-2]` = mark slot as complete (flush remaining audio)
- Padding token = insert pause

**The client (realtime_stream.py) needs to:**
1. Tokenize text words using the model's SentencePiece tokenizer
2. Send token IDs (not raw text) to the server
3. Receive PCM audio frames back

**Important:** The current `remote_server.py` accepts raw text and tokenizes server-side. The batched server expects pre-tokenized token IDs. The new server wrapper needs to handle tokenization.

### 2. 🟡 Replace cohere_transcribe_server.py with nano-cohere-transcribe

**Repo:** https://github.com/Deep-unlearning/nano-cohere-transcribe

**Benefits:**
- 1.5-3.6x faster than transformers (CUDA graph decoder, KV cache)
- Built-in batching (batch_size=64 on A100)
- 6 dependencies (no transformers needed)
- 791x RTFx short-form vs 530x for transformers
- Same WER (10.82%)

**Integration approach:**
- Clone repo, adapt as HTTP server (same `/transcribe` endpoint)
- Add request batching: queue incoming requests, batch process every N ms
- Drop-in replacement for current `cohere_transcribe_server.py`

### 3. 🟢 Smart Turn v3 ONNX on CUDA

Currently loading on CPU only. Need to verify `onnxruntime-gpu` is properly installed in the MindRoot venv and CUDA providers are available. May need:
```bash
/app/.venv/bin/pip install onnxruntime-gpu
```

### 4. 🟢 Kyutai TTS Quantization

No quantized versions exist. Kyutai suggests `n_q=24` instead of 32 for speed/quality tradeoff. The batched server's `Config` class already supports this via the `n_q` parameter.

## Key Files Reference

### mr_kyutai Plugin Structure
```
src/mr_kyutai/
  __init__.py          - Guarded imports (MindRoot vs standalone)
  mod.py               - speak() command, calls session.finish()
  realtime_stream.py   - partial_command pipe, RealtimeSpeakSession, TCP/WS client
  audio_pacer.py       - AudioPacer for SIP timing
  remote_server.py     - Current single-session TCP server (TO BE REPLACED)
```

### Batched TTS Reference (from moshi-server)
```
tts.py                 - TTSService with batch_size, exec_mask, ClientState
voice.py               - Voice loading/management
batched_asr.py         - Batched ASR reference
```

### Container Config
```
Dockerfile             - Multi-stage build, all services
start.sh               - Assembles supervisord.conf based on TTS_BACKEND
supervisord_base.conf  - LLM (vLLM) config
deploy_unified.py      - Multi-cloud deployment (RunPod/Nebius)
```

## Environment Variables

| Var | Default | Description |
|---|---|---|
| `TTS_BACKEND` | `kyutai` | TTS backend selection |
| `KYUTAI_REMOTE` | `tcp://localhost:8765` | Remote TTS server URL |
| `MR_KYUTAI_REALTIME_STREAM` | `1` | Enable realtime streaming pipe |
| `STT_PROVIDER` | `smart_turn_v3` | STT provider selection |
| `SMART_TURN_MIN_SPEECH_MS` | `500` | Min speech before accepting turn |
| `SMART_TURN_THRESHOLD` | `0.5` | Turn detection probability threshold |
| `SMART_TURN_POLL_MS` | `80` | Polling interval |
| `COHERE_TRANSCRIBE_URL` | `http://localhost:8881` | Cohere server URL |
| `REQUIRE_DEEPGRAM` | `false` | Require Deepgram API key |

## Git Repos & Recent Commits

| Repo | Latest Commit | Description |
|---|---|---|
| runvnc/mr_sip | e99178f | Fix self-cancellation + min speech |
| runvnc/mr_kyutai | 70868dc (+ d4c42e6) | All plugin fixes + streaming cleanup |

## Testing Notes

- Testing on **Nebius** H200 instance
- SIP calls via mr_sip to external phone numbers
- Kyutai TTS audio IS working (heard response on call)
- Smart Turn v3 IS detecting turns and transcribing (saw "Sing." transcription)
- Main remaining issue: TTS only single-session (batched server needed)
- Debug logs: `/workspace/kyutai.err`, `/tmp/smart_turn_v3_stt.log`

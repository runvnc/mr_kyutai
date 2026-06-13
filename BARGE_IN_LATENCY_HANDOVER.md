# Engineering Handover: Eliminate Barge-In Latency from Kyutai Audio Drain

## Context

In the `mr_sip` Smart Turn v3 code path, barge-in detection and LLM response generation happen immediately at the eager EOT trigger. However, the **outgoing Kyutai TTS audio** for the new response can be delayed because the previous `speak()` command is still waiting inside `session.finish()` for the `AudioPacer` to drain the prior response's audio. That `speak()` command holds a per-`log_id` asyncio lock, and the new response's `partial_command` pipe blocks on the same lock.

This document describes the exact problem, the relevant code paths, and a concrete implementation plan so the work can be picked up in a clean session.

---

## Relevant files

| File | Purpose |
|------|---------|
| `/xfiles/plugins_ah/mr_kyutai/src/mr_kyutai/mod.py` | `speak()` command entry point; acquires the serial lock and calls `finish()`. |
| `/xfiles/plugins_ah/mr_kyutai/src/mr_kyutai/realtime_stream.py` | `RealtimeSpeakSession`, `handle_speak_partial()` partial_command pipe, `on_interrupt()` hook, session lifecycle. |
| `/xfiles/plugins_ah/mr_kyutai/src/mr_kyutai/audio_pacer.py` | `AudioPacer` used to pace ulaw chunks to SIP. |
| `/xfiles/update_plugins/mr_sip/src/mr_sip/sip_client_v2.py` | Barge-in handler `_handle_turn_resumed()`; calls `sip_halt_audio()` and cancels the AI response. |
| `/xfiles/update_plugins/mr_sip/src/mr_sip/services_v2.py` | `sip_halt_audio()`, `sip_clear_audio_queue()`, `sip_resume_audio()` services. |
| `/xfiles/upd5/mr_eleven_stream/src/mr_eleven_stream/mod.py` | Reference implementation: rejects concurrent `speak()` instead of blocking. |

---

## Current behavior

### 1. Eager EOT triggers agent processing immediately

In `smart_turn_v3_stt.py`:

```python
async def _emit_eager_eot(self, prob, silence_at_end_ms):
    ...
    result = STTResult(text=text, is_final=False, is_eager_eot=True, ...)
    self._emit_partial(result)
```

In `sip_client_v2.py`:

```python
def _on_partial_result(self, result: STTResult):
    if result.is_eager_eot:
        self.draft_response_active = True
        self.last_eager_eot_text = result.text
        if self.on_utterance_callback:
            self._schedule_coroutine(
                self._call_utterance_callback(result.text, utterance_num, ..., is_eager=True)
            )
```

The callback in `services_v2.py` calls:

```python
await service_manager.cancel_and_wait(ctx.log_id, ctx.username)
await service_manager.backend_user_message(message=text)
await service_manager.send_message_to_agent(session_id=ctx.log_id, message=text, context=ctx)
```

So **LLM processing starts immediately at eager EOT**; it is not blocked by the Kyutai drain.

### 2. Kyutai `speak()` blocks until prior audio drains

In `mr_kyutai/src/mr_kyutai/mod.py`:

```python
lock = get_speak_serial_lock(log_id)
await lock.acquire()          # <-- blocks if another speak() is still running
...
await s.finish()              # <-- waits for AudioPacer drain
await cleanup_session(log_id)
...
finally:
    if acquired_lock and lock.locked():
        lock.release()
```

`RealtimeSpeakSession.finish()` in `realtime_stream.py`:

```python
async def finish(self):
    self._text_queue.put(_END)
    if self._tts_thread and self._tts_thread.is_alive():
        await loop.run_in_executor(None, lambda: self._tts_thread.join(timeout=60.0))
    if self._audio_task:
        await asyncio.wait_for(self._audio_task, timeout=60.0)
```

The audio task in `_process_audio()` waits here:

```python
self._pacer.mark_finished()
if not self._pacer.interrupted:
    await self._pacer.wait_until_done()
```

### 3. New partials also block on the same lock

In `realtime_stream.py` `handle_speak_partial()`:

```python
serial_lock = get_speak_serial_lock(log_id)
if serial_lock.locked():
    ...
    await serial_lock.acquire()   # <-- waits for prior speak() to finish
    ...
```

### 4. Barge-in path

In `sip_client_v2.py`:

```python
def _handle_turn_resumed(self):
    self._schedule_coroutine(self._halt_audio_output())
    self.last_eager_eot_text = ''
    if self.draft_response_active:
        self._schedule_coroutine(self._cancel_ai_response())
    self.draft_response_active = False
```

MindRoot core also calls the `on_interrupt` hook in `mr_kyutai`:

```python
@hook()
async def on_interrupt(context=None):
    log_id = getattr(context, \"log_id\", None)
    if has_active_session(log_id):
        await cleanup_session(log_id)
```

`cleanup_session()` calls `session.cancel()`, but the `speak()` command is still stuck in `finish()` holding the lock.

---

## Root cause

The serial lock is held for the entire duration of `finish()`, which waits for the `AudioPacer` to drain. After barge-in, the new response's `partial_command` pipe cannot start a fresh Kyutai session until the old `speak()` finally releases the lock.

---

## Implementation plan

### Step 1: Make `AudioPacer.wait_until_done()` return immediately when interrupted

**File:** `/xfiles/plugins_ah/mr_kyutai/src/mr_kyutai/audio_pacer.py`

- In `wait_until_done()`, check `self._interrupted` and return without awaiting `pacer_task` if interrupted.
- Ensure `_pace_loop()` breaks immediately when `_interrupted` is set.

### Step 2: Make `RealtimeSpeakSession.finish()` interruptible

**File:** `/xfiles/plugins_ah/mr_kyutai/src/mr_kyutai/realtime_stream.py`

- Add an `asyncio.Event` named `_drain_complete_event` to `RealtimeSpeakSession`.
- Set it in `_process_audio()` when the audio loop exits (normal, finished, or interrupted).
- In `finish()`, wait on `_drain_complete_event` with a short timeout and check a `_cancelled` / `_interrupted` flag; return immediately if cancelled.
- In `cancel()`, set the flag and set `_drain_complete_event` so any waiter is unblocked.

### Step 3: Release the speak serial lock immediately on interrupt

**File:** `/xfiles/plugins_ah/mr_kyutai/src/mr_kyutai/mod.py`

- The existing `finally` block releases the lock, but only after `finish()` returns.
- By making `finish()` return quickly on interrupt (Step 2), the `finally` runs promptly.
- Optionally track the active lock holder per `log_id` so `on_interrupt()` can force-release if needed.

### Step 4: Let `partial_command` skip the serial lock when barge-in is pending

**File:** `/xfiles/plugins_ah/mr_kyutai/src/mr_kyutai/realtime_stream.py`

- Add a per-`log_id` barge-in flag/event (e.g. `_barge_in_flags: Dict[str, asyncio.Event]`).
- `on_interrupt()` sets the flag.
- In `handle_speak_partial()`, if the flag is set:
  - Do **not** wait on `serial_lock`.
  - Force-cleanup the existing session (fire-and-forget or with a short timeout).
  - Clear the flag and start a fresh session immediately.
- If no barge-in is pending, keep the existing wait behavior for normal consecutive sentences.

### Step 5: Ensure `on_interrupt` is synchronous-fast

**File:** `/xfiles/plugins_ah/mr_kyutai/src/mr_kyutai/realtime_stream.py`

- Set flags and cancel tasks immediately.
- Move session cleanup to a background `asyncio.create_task()` if it could await anything slow.
- New partials should not wait for old cleanup to finish.

### Step 6: Discard buffered audio on interrupt

- `session.cancel()` should clear the pacer buffer and stop the pacer.
- Any audio already queued in `sip_audio_out_chunk` is cleared by `mr_sip`'s `sip_halt_audio()` / `sip_clear_audio_queue()`.
- Do **not** attempt to drain a tail; that reintroduces the delay.

### Step 7: Add per-`log_id` interrupt state tracking

- Use a module-level dict of `asyncio.Event` objects keyed by `log_id`.
- Set on interrupt, checked/cleared by `handle_speak_partial()`.
- This is more reliable than inferring interrupt state from lock state alone.

### Step 8: Prevent lock double-release or orphaned locks

- Keep the existing `acquired_lock` boolean in `speak()`.
- If `on_interrupt()` force-releases a lock, mark the session/command as interrupted so the original `finally` does not double-release.
- Consider using a small wrapper that tracks ownership.

### Step 9: Test scenarios

1. **Normal turn:** user speaks, Smart Turn final, agent responds, audio plays fully. No regression.
2. **Barge-in during TTS:** user speaks while AI is speaking. Current audio halts, new response audio starts with minimal gap.
3. **Back-to-back sentences:** agent emits `[speak, speak]` in one turn. Second sentence still waits for first to drain (desired).
4. **Barge-in at response start:** interrupt before any audio generated. New response starts immediately.
5. **Barge-in during final drain:** `finish()` is waiting on pacer; interrupt unblocks it.

### Step 10: Optional metrics cleanup

- `sip_client_v2.start_tts_response()` logs `TTS_RESPONSE_START utterance=0` because it reads `self.stt._utterance_count` at the wrong time. Consider fixing separately for clean E2E latency logs.

---

## Key design constraint

Preserve the existing behavior that **consecutive sentences in the same AI turn do not overlap**. The serial lock should still serialize normal `[speak, speak]` sequences. It should only be bypassed when a barge-in has explicitly occurred.

---

## Suggested verification metrics

- Time from `EAGER_EOT_CALLBACK` / `UTTERANCE_CALLBACK` to `TTS_RESPONSE_START` for the new response.
- Time from `on_interrupt` to first audio chunk of the new response.
- No \"Speech already in progress\" errors or lock leaks.
- No overlapping audio from two responses.

---

## Related commits for reference

- `mr_sip` commit `0848c08` \u2014 Add Smart Turn v3 eager endpointing.
- `mr_sip` commit `c5af2c8` \u2014 Add e2e latency profiling events for v2/smart_turn_v3 + kyutai streaming path.
- `mr_kyutai` commit `f752c9e` \u2014 Fix Kyutai speak serialization and barge-in cleanup.
- `mr_kyutai` commit `5cb36ec` \u2014 speak in progress bug.
- `mr_eleven_stream` commit `1ba755f` \u2014 speak in progress bug (reference: rejects concurrent speak instead of blocking).

---

## Notes

- Do not modify code in this session; this document is for a follow-up implementation pass.
- The problem is specifically in the Kyutai realtime streaming path (`MR_KYUTAI_REALTIME_STREAM=1`).
- The Smart Turn eager/final logic itself is correct and should not be changed.

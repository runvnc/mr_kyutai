from typing import Optional, Dict, Any
import logging

print("Loading Kyutai")

from lib.providers.commands import command

@command()
async def speak(
    text: str,
    voice_id: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None
) -> None:
    """Convert text to speech using Kyutai streaming TTS.

    In realtime mode (MR_KYUTAI_REALTIME_STREAM=1), the actual audio
    generation is handled by the partial_command pipe which intercepts
    incremental speak() calls as the LLM streams. This command exists
    so it can be assigned to agents.

    When this command is called, all partial_command events have already
    been processed, so we call finish() on the session to flush remaining
    text and complete the TTS generation.

    We also feed any remaining text that partial_command may not have delivered yet.
    """
    log_id = None
    if context:
        log_id = getattr(context, "log_id", None)

    acquired_lock = False
    lock = None

    if log_id:
        try:
            from .realtime_stream import (
                _realtime_sessions,
                RealtimeSpeakSession,
                cleanup_session,
                is_realtime_streaming_enabled,
                get_speak_serial_lock,
                _klog,
            )

            # In non-realtime mode this plugin currently has no separate fallback
            # implementation here; keep the existing no-op behavior.
            if not is_realtime_streaming_enabled():
                return None

            # Serialize speak commands per conversation. Do NOT reject a second
            # speak; normal turns may be [speak, speak, speak]. Later speak()
            # calls wait here until the previous speak's TTS generation and SIP
            # AudioPacer drain have completed.
            lock = get_speak_serial_lock(log_id)
            await lock.acquire()
            acquired_lock = True
            s = _realtime_sessions.get(log_id)
            if s is None:
                # Important completion-barrier case:
                # If the final command executes before the partial_command pipe
                # has created a session, the old code returned immediately, so a
                # following hangup() could cut off all speech. Create the session
                # here, feed the final text, and wait for finish().
                if text and text.strip():
                    _klog(f"speak() command: no active session, creating one for final text log_id={log_id}")
                    s = RealtimeSpeakSession(context=context)
                    s._e2e_utterance_num = 0
                    _realtime_sessions[log_id] = s
                    await s.start()
                    await s.feed_text_delta(text)
                    s.previous_text = text
            elif not s.is_active:
                # Stale inactive session object: restart it and use this speak()
                # call as the authoritative final text.
                if text and text.strip():
                    _klog(f"speak() command: restarting inactive session for final text log_id={log_id}")
                    await s.start()
                    await s.feed_text_delta(text)
                    s.previous_text = text
            else:
                # Existing realtime partial session: feed any remaining text that
                # partial_command has not delivered yet, then finish and block
                # until the TTS thread + audio pacer drain.
                if text and len(text) > len(s.previous_text):
                    remaining = text[len(s.previous_text):]
                    if remaining.strip():
                        _klog(f"speak() command: feeding remaining text: {repr(remaining[:80])}")
                        await s.feed_text_delta(remaining)
                        s.previous_text = text
                elif text and not text.startswith(s.previous_text):
                    _klog(f"speak() command: text mismatch, feeding full text: {repr(text[:80])}")
                    await s.feed_text_delta(text)
                    s.previous_text = text

            if s is not None and s.is_active:
                _klog(f"speak() command: finishing session for log_id={log_id}")
                await s.finish()
                _klog(f"speak() command: session finished for log_id={log_id}")
                await cleanup_session(log_id, session=s)
        except ImportError:
            pass
        except Exception as e:
            logging.getLogger(__name__).exception(f"speak() finish error: {e}")
            # Do not silently swallow TTS transport/server failures.
            raise
        finally:
            if acquired_lock and lock is not None and lock.locked():
                lock.release()

print("OK")

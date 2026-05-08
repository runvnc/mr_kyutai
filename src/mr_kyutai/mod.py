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

    if log_id:
        try:
            from .realtime_stream import _realtime_sessions, _klog
            s = _realtime_sessions.get(log_id)
            if s is not None and s.is_active:
                # Feed any remaining text that partial_command hasn't delivered yet.
                # This handles the race where speak() runs before all partials arrive.
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
                _klog(f"speak() command: finishing session for log_id={log_id}")
                await s.finish()
                _klog(f"speak() command: session finished for log_id={log_id}")
        except ImportError:
            pass
        except Exception as e:
            logging.getLogger(__name__).warning(f"speak() finish error: {e}")

print("OK")

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
    """
    log_id = None
    if context:
        log_id = getattr(context, "log_id", None)

    if log_id:
        try:
            from .realtime_stream import _realtime_sessions, _klog
            s = _realtime_sessions.get(log_id)
            if s is not None and s.is_active:
                _klog(f"speak() command: finishing session for log_id={log_id}")
                await s.finish()
                _klog(f"speak() command: session finished for log_id={log_id}")
        except ImportError:
            pass
        except Exception as e:
            logging.getLogger(__name__).warning(f"speak() finish error: {e}")

print("OK")

from __future__ import annotations

import os
import asyncio
import logging
from typing import Dict, Any, Optional

print("Loading Kyutai")

from lib.providers.commands import command
from lib.providers.services import service, service_manager

from .audio_pacer import AudioPacer
from .realtime_stream import (
    RealtimeSpeakSession,
    _realtime_sessions,
    is_realtime_streaming_enabled,
)

logger = logging.getLogger(__name__)

# Default config
DEFAULT_VOICE = os.environ.get('MR_KYUTAI_VOICE', 'expresso/ex03-ex01_happy_001_channel1_334s.wav')

# Per-session speak locks (same pattern as mr_qwen3tts/mr_pocket_tts)
_active_speak_locks: Dict[str, asyncio.Lock] = {}

@command()
async def speak(
    text: str,
    voice_id: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None
) -> None:
    """Convert text to speech using Kyutai streaming TTS.

    In realtime streaming mode (MR_KYUTAI_REALTIME_STREAM=1), this command
    is handled by the partial_command pipe in realtime_stream.py instead.
    This fallback implementation sends the full text at once.
    """
    log_id = None
    if context and hasattr(context, 'log_id'):
        log_id = context.log_id
    sip_response_started = False

    logger.info(f"speak() CALLED text='{text[:60]}...' log_id={log_id}")

    try:
        if log_id:
            if log_id not in _active_speak_locks:
                _active_speak_locks[log_id] = asyncio.Lock()
            lock = _active_speak_locks[log_id]
            if lock.locked():
                logger.warning(f"speak() already running for log_id {log_id}")
                return "ERROR: Speech already in progress"
            await lock.acquire()

        # In realtime mode, the partial_command pipe handles incremental
        # streaming. This fallback just sends the full text via the session.
        if is_realtime_streaming_enabled() and log_id and log_id in _realtime_sessions:
            sess = _realtime_sessions[log_id]
            # Wait for any existing audio to finish, then send full text
            await sess.finish()
            sess2 = RealtimeSpeakSession(context=context)
            _realtime_sessions[log_id] = sess2
            await sess2.start()
            await sess2.feed_text_delta(text)
            await sess2.finish()
        else:
            # Non-realtime path: just log, actual streaming handled by pipe
            logger.info(f"speak() text queued for non-realtime processing")

        return None

    except Exception as e:
        logger.exception(f"Error in speak command: {e}")
        return f"Error: {e}"
    finally:
        if log_id and log_id in _active_speak_locks:
            lock = _active_speak_locks[log_id]
            if lock.locked():
                lock.release()

print("OK")

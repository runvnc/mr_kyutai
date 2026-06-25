import asyncio
import time
import logging
from collections import deque
from typing import Callable, Optional, Any

logger = logging.getLogger(__name__)


class AudioPacer:
    """Pace ulaw_8k audio chunks to real-time.

    Assumes ulaw @ 8kHz => 1 byte per sample.
    Callback signature: await on_audio_chunk(chunk, timestamp=?, context=?)
    """

    def __init__(self, sample_rate: int = 8000):
        self.sample_rate = sample_rate
        self.buffer = deque()
        self.pacer_task: Optional[asyncio.Task] = None
        self.on_audio_chunk: Optional[Callable] = None
        self.context: Any = None
        self._running = False

        self.start_time: Optional[float] = None
        self.bytes_sent = 0
        self.audio_start_time: Optional[float] = None
        self._finished_adding = False
        self._interrupted = False
        self._resume = False

    async def add_chunk(self, audio_bytes: bytes):
        if self._running and not self._interrupted:
            self.buffer.append(audio_bytes)
            # NOTE: do NOT anchor audio_start_time here. Buffering happens as
            # soon as the TTS engine produces audio, which on a cold/first call
            # can be well before the pace loop actually starts sending (engine
            # warmup, event-loop contention). Anchoring at buffer time makes the
            # recording place the whole turn ~that-delay too early (turn looks
            # "instant" while the caller really waited ~1s). The pace loop sets
            # audio_start_time at the moment the first frame is actually sent.

    @property
    def interrupted(self) -> bool:
        return self._interrupted

    def _set_interrupted(self):
        self._interrupted = True

    def interrupt(self):
        """Interrupt pacing immediately and discard any buffered audio."""
        self._interrupted = True
        self._running = False
        self._finished_adding = True
        self.buffer.clear()

    def mark_finished(self):
        self._finished_adding = True

    async def clear(self):
        self.buffer.clear()
        self.audio_start_time = None
        self.bytes_sent = 0
        self.start_time = time.perf_counter()
        self._interrupted = False
        self._finished_adding = False
        logger.debug("AudioPacer cleared and reset")

    async def start_pacing(self, on_audio_chunk: Callable, context: Any):
        self.on_audio_chunk = on_audio_chunk
        self.context = context
        self._running = True
        self._finished_adding = False
        self._interrupted = False

        self.start_time = time.perf_counter()
        self.bytes_sent = 0
        self.audio_start_time = None
        self._resume = False

        self.pacer_task = asyncio.create_task(self._pace_loop())

    async def _pace_loop(self):
        while self._running:
            if self._interrupted:
                break
            if len(self.buffer) > 0:
                chunk = self.buffer.popleft()

                # Re-anchor the timing clock to wall-clock time whenever audio
                # resumes after the buffer went idle (or on the very first
                # chunk). Without this, chunk_timestamp only advances by the
                # amount of audio actually sent, so any wall-clock pause between
                # utterances (e.g. LLM think latency) is collapsed -- making the
                # recording show outgoing speech starting instantly after the
                # caller stops, and causing a burst-to-catch-up on RTP output.
                if self.audio_start_time is None or self._resume:
                    self.audio_start_time = time.perf_counter()
                    self.bytes_sent = 0
                    self._resume = False

                if self.audio_start_time:
                    chunk_timestamp = self.audio_start_time + (self.bytes_sent / self.sample_rate)
                else:
                    chunk_timestamp = None

                try:
                    result = await self.on_audio_chunk(chunk, timestamp=chunk_timestamp, context=self.context)
                    if result is False:
                        logger.debug("AudioPacer: callback requested stop")
                        self._set_interrupted()
                        break
                except Exception as e:
                    logger.error(f"AudioPacer: error in callback: {e}")
                    break

                self.bytes_sent += len(chunk)

                base_time = self.audio_start_time if self.audio_start_time else self.start_time
                target_time = base_time + (self.bytes_sent / self.sample_rate)
                current_time = time.perf_counter()
                sleep_duration = target_time - current_time
                if sleep_duration > 0:
                    await asyncio.sleep(sleep_duration)
                    if self._interrupted:
                        break
            else:
                if self._finished_adding:
                    break
                self._resume = True
                await asyncio.sleep(0.005)

        logger.debug(f"AudioPacer: finished, sent {self.bytes_sent} bytes")

    async def stop(self):
        self._running = False
        if self.pacer_task:
            self.pacer_task.cancel()
            try:
                await self.pacer_task
            except asyncio.CancelledError:
                pass
        self.buffer.clear()

    async def wait_until_done(self):
        if self._interrupted:
            return
        if self.pacer_task:
            try:
                await self.pacer_task
            except asyncio.CancelledError:
                pass

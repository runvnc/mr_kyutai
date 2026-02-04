from __future__ import annotations

"""Realtime streaming TTS using Kyutai (moshi) with incremental text input.

Drop-in-ish analogue of mr_eleven_stream/realtime_stream.py:
- a partial_command pipe intercepts incremental speak(text=...) updates
- we diff to get deltas
- we buffer deltas into word-complete chunks
- we stream audio out to SIP as ulaw_8k

Env:
  MR_KYUTAI_REALTIME_STREAM=1
  MR_KYUTAI_HF_REPO=kyutai/tts-1.6b-en_fr
  MR_KYUTAI_VOICE_REPO=kyutai/tts-voices
  MR_KYUTAI_VOICE=expresso/ex03-ex01_happy_001_channel1_334s.wav
  MR_KYUTAI_DEVICE=cuda|cpu      (only used for local inference mode)
  KYUTAI_REMOTE=host:port        (default: moshi-server over WebSockets)
  KYUTAI_REMOTE=ws://host:port   (moshi-server over WebSockets)
  KYUTAI_REMOTE=wss://host:port  (moshi-server over WebSockets, TLS)
  KYUTAI_REMOTE=tcp://host:port  (explicit: use the included framed-TCP reference server)
  KYUTAI_API_KEY=public_token    (optional; moshi-server auth token if using ws/wss)
"""

import os
import asyncio
import logging
import threading
import queue
import re
import audioop
import socket
import json
import struct
from urllib.parse import urlparse, urlencode, parse_qsl, urlunparse

from typing import Dict, Any, Optional, Iterator

import numpy as np

from lib.pipelines.pipe import pipe
from lib.providers.services import service_manager

from .audio_pacer import AudioPacer

logger = logging.getLogger(__name__)

_END = object()

def _require_local_tts_deps():
    """
    Local Kyutai inference deps (moshi + torch) are optional.
    Import them lazily so remote (moshi-server) mode can run without heavy deps.
    """
    global torch, CheckpointInfo, TTSModel, ConditionAttributes, script_to_entries, dropout_all_conditions, LMGen
    try:
        import torch as _torch  # type: ignore
        from moshi.models.loaders import CheckpointInfo as _CheckpointInfo  # type: ignore
        from moshi.models.tts import (  # type: ignore
            TTSModel as _TTSModel,
            ConditionAttributes as _ConditionAttributes,
            script_to_entries as _script_to_entries,
        )
        from moshi.conditioners import dropout_all_conditions as _dropout_all_conditions  # type: ignore
        from moshi.models.lm import LMGen as _LMGen  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Local Kyutai inference requested, but dependencies are missing. "
            "Install mr_kyutai with the 'local' extra (moshi + torch), or set KYUTAI_REMOTE to use moshi-server."
        ) from e

    torch = _torch
    CheckpointInfo = _CheckpointInfo
    TTSModel = _TTSModel
    ConditionAttributes = _ConditionAttributes
    script_to_entries = _script_to_entries
    dropout_all_conditions = _dropout_all_conditions
    LMGen = _LMGen

def _is_remote_enabled() -> bool:
    return bool((os.environ.get("KYUTAI_REMOTE") or "").strip())

def _is_tcp_remote(remote: str) -> bool:
    return (remote or "").strip().lower().startswith("tcp://")

def _parse_kyutai_remote(val: str) -> tuple[str, int]:
    """Parse KYUTAI_REMOTE as host:port or tcp://host:port (port optional)."""
    s = (val or "").strip()
    if not s:
        raise ValueError("empty KYUTAI_REMOTE")
    if s.startswith("tcp://"):
        s = s[len("tcp://") :]
    if "/" in s:
        s = s.split("/", 1)[0]
    if ":" in s:
        host, port_s = s.rsplit(":", 1)
        host = host.strip()
        port = int(port_s.strip())
        return host, port
    return s, 8765


def _send_frame(sock: socket.socket, frame_type: bytes, payload: bytes) -> None:
    """Frame format: 1 byte type + 4-byte big-endian length + payload."""
    if not isinstance(frame_type, (bytes, bytearray)) or len(frame_type) != 1:
        raise ValueError("frame_type must be 1 byte")
    header = frame_type + struct.pack(">I", len(payload))
    sock.sendall(header + payload)


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("socket closed")
        buf.extend(chunk)
    return bytes(buf)


def _recv_frame(sock: socket.socket) -> tuple[bytes, bytes]:
    header = _recv_exact(sock, 5)
    ftype = header[:1]
    (n,) = struct.unpack(">I", header[1:])
    payload = _recv_exact(sock, n) if n else b""
    return ftype, payload

def is_realtime_streaming_enabled() -> bool:
    # Default ON: this plugin is meant to be a drop-in realtime streaming TTS path.
    # Set MR_KYUTAI_REALTIME_STREAM=0/false/off to disable.
    val = os.environ.get("MR_KYUTAI_REALTIME_STREAM", "1").lower()
    return val in ("1", "true", "yes", "on")


def _get_device() -> str:
    _require_local_tts_deps()
    dev = os.environ.get("MR_KYUTAI_DEVICE", "cuda")
    if dev.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return dev


# --- Global model cache ---
_tts_model: Optional[TTSModel] = None
_tts_model_lock = threading.Lock()


def get_tts_model() -> TTSModel:
    """Load Kyutai TTS model once per process."""
    _require_local_tts_deps()
    global _tts_model
    if _tts_model is not None:
        return _tts_model
    with _tts_model_lock:
        if _tts_model is not None:
            return _tts_model
        hf_repo = os.environ.get("MR_KYUTAI_HF_REPO", "kyutai/tts-1.6b-en_fr")
        device = _get_device()
        logger.info(f"mr_kyutai: loading TTS model from {hf_repo} on {device}...")
        ckpt = CheckpointInfo.from_hf_repo(hf_repo)
        _tts_model = TTSModel.from_checkpoint_info(ckpt, n_q=32, temp=0.6, device=device)
        logger.info("mr_kyutai: model loaded")
        return _tts_model


def _make_null(all_attributes):
    _require_local_tts_deps()
    return dropout_all_conditions(all_attributes)


class _KyutaiGen:
    """Incremental generator adapted from scripts/tts_pytorch_streaming.py."""

    def __init__(self, tts_model: TTSModel, attributes: list[ConditionAttributes], on_frame=None):
        _require_local_tts_deps()
        self.tts_model = tts_model
        self.on_frame = on_frame

        self.offset = 0
        self.state = self.tts_model.machine.new_state([])

        attrs = attributes
        if tts_model.cfg_coef != 1.0:
            if tts_model.valid_cfg_conditionings:
                raise ValueError(
                    "This model does not support direct CFG; pass cfg_coef to make_condition_attributes instead."
                )
            nulled = _make_null(attrs)
            attrs = list(attrs) + nulled

        assert tts_model.lm.condition_provider is not None
        prepared = tts_model.lm.condition_provider.prepare(attrs)
        condition_tensors = tts_model.lm.condition_provider(prepared)

        def _on_text_logits_hook(text_logits):
            if tts_model.padding_bonus:
                text_logits[..., tts_model.machine.token_ids.pad] += tts_model.padding_bonus
            return text_logits

        def _on_audio_hook(audio_tokens):
            audio_offset = tts_model.lm.audio_offset
            delays = tts_model.lm.delays
            for q in range(audio_tokens.shape[1]):
                delay = delays[q + audio_offset]
                if self.offset < delay + tts_model.delay_steps:
                    audio_tokens[:, q] = tts_model.machine.token_ids.zero

        def _on_text_hook(text_tokens):
            tokens = text_tokens.tolist()
            out_tokens = []
            for token in tokens:
                out_token, _ = tts_model.machine.process(self.offset, self.state, token)
                out_tokens.append(out_token)
            text_tokens[:] = torch.tensor(out_tokens, dtype=torch.long, device=text_tokens.device)

        tts_model.lm.dep_q = tts_model.n_q
        self.lm_gen = LMGen(
            tts_model.lm,
            temp=tts_model.temp,
            temp_text=tts_model.temp,
            cfg_coef=tts_model.cfg_coef,
            condition_tensors=condition_tensors,
            on_text_logits_hook=_on_text_logits_hook,
            on_text_hook=_on_text_hook,
            on_audio_hook=_on_audio_hook,
            cfg_is_masked_until=None,
            cfg_is_no_text=True,
        )
        self.lm_gen.streaming_forever(1)

    def append_entry(self, entry):
        self.state.entries.append(entry)

    def process(self):
        while len(self.state.entries) > self.tts_model.machine.second_stream_ahead:
            self._step()

    def process_last(self):
        while len(self.state.entries) > 0 or self.state.end_step is not None:
            self._step()
        additional_steps = self.tts_model.delay_steps + max(self.tts_model.lm.delays) + 8
        for _ in range(additional_steps):
            self._step()

    def _step(self):
        missing = self.tts_model.lm.n_q - self.tts_model.lm.dep_q
        input_tokens = torch.full(
            (1, missing, 1),
            self.tts_model.machine.token_ids.zero,
            dtype=torch.long,
            device=self.tts_model.lm.device,
        )
        frame = self.lm_gen.step(input_tokens)
        self.offset += 1
        if frame is not None and self.on_frame is not None:
            self.on_frame(frame)


def _prepare_script_piece(model: TTSModel, script_piece: str, first_turn: bool):
    _require_local_tts_deps()
    multi_speaker = first_turn and model.multi_speaker
    return script_to_entries(
        model.tokenizer,
        model.machine.token_ids,
        model.mimi.frame_rate,
        [script_piece],
        multi_speaker=multi_speaker,
        padding_between=1,
    )


class RealtimeSpeakSession:
    """Realtime incremental-input Kyutai TTS session.

    Async side:
      - feed_text_delta(delta) (deltas from partial_command diff)
      - finish()

    Thread side:
      - turns buffered words into generation steps
      - converts Mimi 24k PCM -> 8k ulaw chunks
      - pushes ulaw chunks into audio queue

    Audio output:
      - if sip_audio_out_chunk is available, we pace and send to SIP
    """

    def __init__(self, context: Any):
        self.context = context
        self.is_active = False
        self.is_finished = False
        self.previous_text = ""

        self._text_queue: queue.Queue = queue.Queue()
        self._audio_queue: queue.Queue = queue.Queue()

        self._tts_thread: Optional[threading.Thread] = None
        self._audio_task: Optional[asyncio.Task] = None
        self._pacer: Optional[AudioPacer] = None

        # remote mode
        self._remote_sock: Optional[socket.socket] = None

        self._buffer = ""  # partial-word buffer

    def _split_word_complete(self, delta: str) -> list[str]:
        """Accumulate delta, return list of word-complete chunks to synthesize.

        Strategy: only emit up to last whitespace boundary.
        Remaining partial word stays buffered.
        """
        if not delta:
            return []
        self._buffer += delta

        # If we have any whitespace, we can safely emit up to the last whitespace.
        m = re.search(r"\s+(?!.*\s)", self._buffer)  # last whitespace match
        if not m:
            return []
        cut = m.end()  # include the whitespace
        emit = self._buffer[:cut]
        self._buffer = self._buffer[cut:]
        return [emit]

    def _flush_buffer(self) -> Optional[str]:
        s = self._buffer.strip()
        self._buffer = ""
        return s if s else None

    def _get_effective_voice_rel(self) -> str:
        voice_rel = os.environ.get(
            "MR_KYUTAI_VOICE", "expresso/ex03-ex01_happy_001_channel1_334s.wav"
        )
        try:
            agent_data = asyncio.run(service_manager.get_agent_data(self.context.agent_name))
            persona = agent_data.get("persona", {})
            persona_voice = persona.get("kyutai_voice") or persona.get("voice_id")
            if isinstance(persona_voice, str) and persona_voice:
                voice_rel = persona_voice
        except Exception:
            pass
        return voice_rel

    def _build_moshi_ws_uri(self, base: str, voice_rel: str) -> str:
        """
        Build moshi-server TTS streaming WebSocket URI.

        Matches Kyutai's scripts/tts_rust_server.py:
          ws://HOST:PORT/api/tts_streaming?voice=...&format=PcmMessagePack
        """
        b = (base or "").strip()
        if not b:
            raise ValueError("empty KYUTAI_REMOTE")
        # Default to moshi-server WebSocket mode when the user provides host:port without a scheme.
        if "://" not in b:
            b = "ws://" + b

        parsed = urlparse(b)
        if parsed.scheme not in ("ws", "wss"):
            raise ValueError(f"expected ws:// or wss://, got: {base!r}")

        path = parsed.path or ""
        if not path or path == "/":
            path = "/api/tts_streaming"
        elif not path.endswith("/api/tts_streaming") and "/api/tts_streaming" not in path:
            # If user gave a base URL like ws://host:port, append the expected path.
            path = path.rstrip("/") + "/api/tts_streaming"

        # Merge existing query params with required ones.
        q = dict(parse_qsl(parsed.query, keep_blank_values=True))
        q["voice"] = voice_rel
        q["format"] = q.get("format") or "PcmMessagePack"

        return urlunparse((parsed.scheme, parsed.netloc, path, parsed.params, urlencode(q), parsed.fragment))

    def _run_remote_tts_ws(self):
        """
        Remote TTS mode (moshi-server): stream Text/Eos over WebSockets using msgpack,
        receive Audio messages containing 24kHz float32 PCM, convert to ulaw8k chunks.
        """
        remote = (os.environ.get("KYUTAI_REMOTE") or "").strip()
        voice_rel = self._get_effective_voice_rel()
        uri = self._build_moshi_ws_uri(remote, voice_rel)

        api_key = (os.environ.get("KYUTAI_API_KEY") or os.environ.get("KYUTAI_REMOTE_API_KEY") or "public_token").strip()
        headers = {"kyutai-api-key": api_key} if api_key else None

        try:
            import msgpack  # type: ignore
            import websockets  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "KYUTAI_REMOTE is ws://..., but dependencies are missing. Install 'websockets' and 'msgpack' (or add them to mr_kyutai deps)."
            ) from e

        ratecv_state = None
        src_sr = 24000

        async def _ws_main():
            nonlocal ratecv_state
            async with websockets.connect(uri, additional_headers=headers) as websocket:
                loop = asyncio.get_running_loop()

                async def rx_loop():
                    try:
                        async for message_bytes in websocket:
                            msg = msgpack.unpackb(message_bytes)
                            if not isinstance(msg, dict):
                                continue
                            if msg.get("type") != "Audio":
                                continue
                            pcm = np.array(msg.get("pcm", []), dtype=np.float32)
                            if pcm.size == 0:
                                continue

                            # float32 [-1,1] -> s16le bytes
                            pcm = np.clip(pcm, -1.0, 1.0)
                            pcm16 = (pcm * 32767.0).astype(np.int16).tobytes()

                            # resample 24k -> 8k (stateful)
                            pcm8k, ratecv_state = audioop.ratecv(
                                pcm16,
                                2,      # width
                                1,      # channels
                                src_sr,
                                8000,
                                ratecv_state,
                            )
                            ulaw = audioop.lin2ulaw(pcm8k, 2)

                            # 20ms ulaw frames = 160 bytes at 8kHz
                            chunk_size = 160
                            for i in range(0, len(ulaw), chunk_size):
                                chunk = ulaw[i : i + chunk_size]
                                if chunk:
                                    self._audio_queue.put(bytes(chunk))
                    except Exception as e:
                        logger.exception(f"mr_kyutai moshi-server rx error: {e}")
                    finally:
                        self._audio_queue.put(_END)

                async def tx_loop():
                    try:
                        while True:
                            item = await loop.run_in_executor(None, self._text_queue.get)
                            if item is _END:
                                leftover = self._flush_buffer()
                                if leftover:
                                    for word in leftover.split():
                                        await websocket.send(msgpack.packb({"type": "Text", "text": word}))
                                await websocket.send(msgpack.packb({"type": "Eos"}))
                                break

                            if not isinstance(item, str) or not item.strip():
                                continue
                            # Follow Kyutai's example: send per-word Text messages.
                            for word in item.split():
                                await websocket.send(msgpack.packb({"type": "Text", "text": word}))
                    except Exception as e:
                        logger.exception(f"mr_kyutai moshi-server tx error: {e}")
                    finally:
                        try:
                            await websocket.close()
                        except Exception:
                            pass

                await asyncio.gather(rx_loop(), tx_loop())

        # Run an event loop inside this worker thread.
        asyncio.run(_ws_main())

    def _run_remote_tts(self):
        """Remote TTS mode: stream text chunks to remote server, receive ulaw8k chunks back."""
        remote = (os.environ.get("KYUTAI_REMOTE") or "").strip()
        if not _is_tcp_remote(remote):
            logger.info("mr_kyutai: using moshi-server WebSocket mode (set KYUTAI_REMOTE=tcp://... for legacy TCP server)")
            self._run_remote_tts_ws()
            return
        host, port = _parse_kyutai_remote(remote)

        sock = socket.create_connection((host, port), timeout=15.0)
        sock.settimeout(30.0)
        self._remote_sock = sock

        voice_rel = self._get_effective_voice_rel()

        start_msg = {
            "op": "start",
            "voice": voice_rel,
            "sample_rate": 8000,
            "codec": "ulaw",
            "chunk_bytes": 160,
        }
        _send_frame(sock, b"J", json.dumps(start_msg).encode("utf-8"))

        # Receiver thread: read audio frames from server and push to _audio_queue.
        def _rx():
            try:
                while True:
                    ftype, payload = _recv_frame(sock)
                    if ftype == b"A":
                        if payload:
                            # server SHOULD already chunk at 160B, but we tolerate larger frames
                            chunk_size = 160
                            for i in range(0, len(payload), chunk_size):
                                chunk = payload[i : i + chunk_size]
                                if chunk:
                                    self._audio_queue.put(chunk)
                    elif ftype == b"E":
                        break
                    elif ftype == b"X":
                        # error message
                        try:
                            msg = payload.decode("utf-8", errors="replace")
                        except Exception:
                            msg = repr(payload)
                        logger.error(f"mr_kyutai remote server error: {msg}")
                        break
                    else:
                        logger.warning(f"mr_kyutai remote: unknown frame type {ftype!r}")
            except Exception as e:
                logger.exception(f"mr_kyutai remote rx error: {e}")
            finally:
                self._audio_queue.put(_END)

        rx_thread = threading.Thread(target=_rx, daemon=True)
        rx_thread.start()

        # TX loop in this (tts) thread
        try:
            while True:
                item = self._text_queue.get()
                if item is _END:
                    leftover = self._flush_buffer()
                    if leftover:
                        _send_frame(
                            sock,
                            b"J",
                            json.dumps({"op": "text", "text": leftover}).encode("utf-8"),
                        )
                    _send_frame(sock, b"J", json.dumps({"op": "finish"}).encode("utf-8"))
                    break

                if not isinstance(item, str) or not item.strip():
                    continue
                _send_frame(sock, b"J", json.dumps({"op": "text", "text": item}).encode("utf-8"))
        except Exception as e:
            logger.exception(f"mr_kyutai remote tx error: {e}")
        finally:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except Exception:
                pass
            try:
                sock.close()
            except Exception:
                pass
            self._remote_sock = None

    def _run_tts_thread(self):
        try:
            if _is_remote_enabled():
                logger.info("mr_kyutai: KYUTAI_REMOTE set, using remote TTS")
                self._run_remote_tts()
                return

            tts_model = get_tts_model()

            voice_repo = os.environ.get("MR_KYUTAI_VOICE_REPO", "kyutai/tts-voices")
            voice_rel = self._get_effective_voice_rel()

            # Kyutai voice conditioning: for multi_speaker, pass list of voices, else []
            if tts_model.multi_speaker:
                # Note: moshi (PyTorch) get_voice_path() does not reliably accept voice_repo kwarg.
                # It typically fetches from the model's default voice repo.
                voice_path = tts_model.get_voice_path(voice_rel)
                voices = [voice_path]
            else:
                voices = []

            # cfg_coef goes in conditioning for CFG-distilled models.
            cond = tts_model.make_condition_attributes(voices, cfg_coef=2.0)

            # audio conversion state
            src_sr = int(getattr(tts_model.mimi, "sample_rate", 24000))
            ratecv_state = None

            def on_frame(frame: torch.Tensor):
                nonlocal ratecv_state
                if (frame == -1).any():
                    return
                # decode Mimi audio tokens -> PCM float [-1,1] @ 24k
                pcm = tts_model.mimi.decode(frame[:, 1:, :]).detach().cpu().numpy()
                pcm = np.clip(pcm[0, 0], -1.0, 1.0)

                # float -> s16le bytes
                pcm16 = (pcm * 32767.0).astype(np.int16).tobytes()

                # resample to 8k
                pcm8k, ratecv_state = audioop.ratecv(
                    pcm16,  # fragment
                    2,      # width
                    1,      # channels
                    src_sr,
                    8000,
                    ratecv_state,
                )

                # ulaw encode
                ulaw = audioop.lin2ulaw(pcm8k, 2)

                # chunk for SIP pacing: 20ms = 160 bytes
                chunk_size = 160
                for i in range(0, len(ulaw), chunk_size):
                    chunk = ulaw[i : i + chunk_size]
                    if chunk:
                        self._audio_queue.put(chunk)

            gen = _KyutaiGen(tts_model, [cond], on_frame=on_frame)

            first_turn = True
            with tts_model.mimi.streaming(1):
                while True:
                    item = self._text_queue.get()
                    if item is _END:
                        # flush any partial word
                        leftover = self._flush_buffer()
                        if leftover:
                            entries = _prepare_script_piece(tts_model, leftover, first_turn)
                            first_turn = False
                            for e in entries:
                                gen.append_entry(e)
                                gen.process()
                        gen.process_last()
                        break

                    if not isinstance(item, str) or not item.strip():
                        continue

                    entries = _prepare_script_piece(tts_model, item, first_turn)
                    first_turn = False
                    for e in entries:
                        gen.append_entry(e)
                        gen.process()

        except Exception as e:
            logger.exception(f"mr_kyutai realtime TTS thread error: {e}")
        finally:
            self._audio_queue.put(_END)

    async def _process_audio(self):
        try:
            sip_available = service_manager.functions.get("sip_audio_out_chunk") is not None

            if sip_available:
                self._pacer = AudioPacer(sample_rate=8000)

                async def send_to_sip(chunk, timestamp=None, context=None):
                    return await service_manager.sip_audio_out_chunk(
                        chunk, timestamp=timestamp, context=context
                    )

                await self._pacer.start_pacing(send_to_sip, self.context)

            while True:
                try:
                    audio_chunk = self._audio_queue.get_nowait()
                except queue.Empty:
                    await asyncio.sleep(0.01)
                    continue

                if audio_chunk is _END:
                    break

                if isinstance(audio_chunk, (bytes, bytearray)) and audio_chunk:
                    if sip_available and self._pacer is not None:
                        await self._pacer.add_chunk(bytes(audio_chunk))
                        if self._pacer.interrupted:
                            break

            if sip_available and self._pacer is not None:
                self._pacer.mark_finished()
                if not self._pacer.interrupted:
                    await self._pacer.wait_until_done()
                await self._pacer.stop()

        except Exception as e:
            logger.exception(f"mr_kyutai audio processor error: {e}")

    async def start(self):
        if self.is_active:
            return
        self.is_active = True
        self.is_finished = False
        self.previous_text = ""
        self._buffer = ""

        self._tts_thread = threading.Thread(target=self._run_tts_thread, daemon=True)
        self._tts_thread.start()
        self._audio_task = asyncio.create_task(self._process_audio())

    async def feed_text_delta(self, delta: str):
        if not self.is_active or self.is_finished:
            return
        # Buffer until word boundary.
        chunks = self._split_word_complete(delta)
        for ch in chunks:
            self._text_queue.put(ch)

    async def finish(self):
        if not self.is_active:
            return
        self.is_finished = True
        self._text_queue.put(_END)
        if self._tts_thread and self._tts_thread.is_alive():
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, lambda: self._tts_thread.join(timeout=60.0))
        if self._audio_task:
            try:
                await asyncio.wait_for(self._audio_task, timeout=60.0)
            except Exception:
                pass
        self.is_active = False

    async def cancel(self):
        self.is_finished = True
        self.is_active = False
        try:
            if self._remote_sock is not None:
                try:
                    self._remote_sock.shutdown(socket.SHUT_RDWR)
                except Exception:
                    pass
                self._remote_sock.close()
        except Exception:
            pass
        try:
            self._text_queue.put(_END)
        except Exception:
            pass
        if self._pacer:
            await self._pacer.stop()
        if self._audio_task:
            self._audio_task.cancel()


_realtime_sessions: Dict[str, RealtimeSpeakSession] = {}


def get_session(log_id: str) -> Optional[RealtimeSpeakSession]:
    return _realtime_sessions.get(log_id)


def has_active_session(log_id: str) -> bool:
    s = _realtime_sessions.get(log_id)
    return s is not None and s.is_active


async def cleanup_session(log_id: str):
    if log_id in _realtime_sessions:
        s = _realtime_sessions[log_id]
        if s.is_active:
            await s.cancel()
        del _realtime_sessions[log_id]


@pipe(name="partial_command", priority=10)
async def handle_speak_partial(data: dict, context=None) -> dict:
    """Intercept partial speak(text=...) and stream deltas into Kyutai realtime session."""
    if not is_realtime_streaming_enabled():
        return data

    if data.get("command") != "speak":
        return data

    log_id = getattr(context, "log_id", None) if context else None
    if not log_id:
        return data

    params = data.get("params", {})
    new_text = params.get("text", "") or ""
    if not new_text:
        return data

    if log_id not in _realtime_sessions:
        s = RealtimeSpeakSession(context=context)
        _realtime_sessions[log_id] = s
        await s.start()

    s = _realtime_sessions[log_id]

    # Prefix-diff (same as mr_eleven_stream)
    if len(new_text) > len(s.previous_text):
        delta = new_text[len(s.previous_text) :]
        if delta:
            await s.feed_text_delta(delta)
            s.previous_text = new_text

    return data

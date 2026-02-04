"""
Remote Kyutai TTS streaming server for mr_kyutai.

Protocol (framed TCP):
  Frame = 1 byte type + 4-byte big-endian length + payload

Client -> Server:
  'J' JSON frames:
    {"op":"start","voice":"...","sample_rate":8000,"codec":"ulaw","chunk_bytes":160}
    {"op":"text","text":"hello "}
    {"op":"finish"}

Server -> Client:
  'A' raw ulaw8k bytes (usually 160 bytes per frame = 20ms)
  'E' end
  'X' utf-8 error message

Run:
  python -m mr_kyutai.remote_server --host 0.0.0.0 --port 8765

Env:
  MR_KYUTAI_HF_REPO=kyutai/tts-1.6b-en_fr
  MR_KYUTAI_DEVICE=cuda|cpu
  MR_KYUTAI_VOICE_REPO=kyutai/tts-voices   (best-effort; moshi may ignore)
  MR_KYUTAI_VOICE=expresso/ex03-ex01_happy_001_channel1_334s.wav
"""

import argparse
import json
import logging
import os
import queue
import socket
import socketserver
import struct
import threading
import audioop
from typing import Optional

import numpy as np
import torch

from moshi.models.loaders import CheckpointInfo
from moshi.models.tts import (
    TTSModel,
    ConditionAttributes,
    script_to_entries,
)
from moshi.conditioners import dropout_all_conditions
from moshi.models.lm import LMGen

logger = logging.getLogger(__name__)

_END = object()


def _get_device() -> str:
    dev = os.environ.get("MR_KYUTAI_DEVICE", "cuda")
    if dev.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return dev


_tts_model: Optional[TTSModel] = None
_tts_model_lock = threading.Lock()


def get_tts_model() -> TTSModel:
    global _tts_model
    if _tts_model is not None:
        return _tts_model
    with _tts_model_lock:
        if _tts_model is not None:
            return _tts_model
        hf_repo = os.environ.get("MR_KYUTAI_HF_REPO", "kyutai/tts-1.6b-en_fr")
        device = _get_device()
        logger.info(f"mr_kyutai.remote_server: loading TTS model from {hf_repo} on {device}...")
        ckpt = CheckpointInfo.from_hf_repo(hf_repo)
        _tts_model = TTSModel.from_checkpoint_info(ckpt, n_q=32, temp=0.6, device=device)
        logger.info("mr_kyutai.remote_server: model loaded")
        return _tts_model


def _make_null(all_attributes):
    return dropout_all_conditions(all_attributes)


class _KyutaiGen:
    """Incremental generator adapted from kyutai-labs/delayed-streams-modeling scripts/tts_pytorch_streaming.py."""

    def __init__(self, tts_model: TTSModel, attributes: list[ConditionAttributes], on_frame=None):
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
    multi_speaker = first_turn and model.multi_speaker
    return script_to_entries(
        model.tokenizer,
        model.machine.token_ids,
        model.mimi.frame_rate,
        [script_piece],
        multi_speaker=multi_speaker,
        padding_between=1,
    )


def _send_frame(sock: socket.socket, frame_type: bytes, payload: bytes) -> None:
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


class _Session:
    def __init__(self, sock: socket.socket):
        self.sock = sock
        self.send_lock = threading.Lock()
        self.text_q: queue.Queue = queue.Queue()
        self.started = False

        self.first_turn = True
        self.tts_model: Optional[TTSModel] = None
        self.gen: Optional[_KyutaiGen] = None

        self.chunk_bytes = 160
        self.src_sr = 24000
        self.ratecv_state = None

    def send_audio(self, ulaw_bytes: bytes):
        if not ulaw_bytes:
            return
        with self.send_lock:
            _send_frame(self.sock, b"A", ulaw_bytes)

    def send_end(self):
        with self.send_lock:
            _send_frame(self.sock, b"E", b"")

    def send_error(self, msg: str):
        payload = (msg or "error").encode("utf-8", errors="replace")
        with self.send_lock:
            _send_frame(self.sock, b"X", payload)

    def _make_condition(self, voice_rel: str) -> ConditionAttributes:
        assert self.tts_model is not None
        tts_model = self.tts_model

        # best-effort; moshi's get_voice_path may ignore voice_repo kwarg in some versions
        _ = os.environ.get("MR_KYUTAI_VOICE_REPO", "kyutai/tts-voices")

        if tts_model.multi_speaker:
            voice_path = tts_model.get_voice_path(voice_rel)
            voices = [voice_path]
        else:
            voices = []
        return tts_model.make_condition_attributes(voices, cfg_coef=2.0)

    def _init_model(self, voice_rel: str):
        self.tts_model = get_tts_model()
        self.src_sr = int(getattr(self.tts_model.mimi, "sample_rate", 24000))

        def on_frame(frame: torch.Tensor):
            if (frame == -1).any():
                return
            assert self.tts_model is not None
            tts_model = self.tts_model

            pcm = tts_model.mimi.decode(frame[:, 1:, :]).detach().cpu().numpy()
            pcm = np.clip(pcm[0, 0], -1.0, 1.0)
            pcm16 = (pcm * 32767.0).astype(np.int16).tobytes()

            pcm8k, self.ratecv_state = audioop.ratecv(
                pcm16,
                2,  # width
                1,  # channels
                self.src_sr,
                8000,
                self.ratecv_state,
            )
            ulaw = audioop.lin2ulaw(pcm8k, 2)

            for i in range(0, len(ulaw), self.chunk_bytes):
                ch = ulaw[i : i + self.chunk_bytes]
                if ch:
                    self.send_audio(ch)

        cond = self._make_condition(voice_rel)
        self.gen = _KyutaiGen(self.tts_model, [cond], on_frame=on_frame)

    def run(self):
        # RX thread: JSON frames -> queue text chunks.
        def rx():
            try:
                while True:
                    ftype, payload = _recv_frame(self.sock)
                    if ftype != b"J":
                        raise ValueError(f"expected JSON frame 'J', got {ftype!r}")
                    msg = json.loads(payload.decode("utf-8"))
                    op = msg.get("op")
                    if op == "start":
                        if self.started:
                            continue
                        voice_rel = msg.get("voice") or os.environ.get(
                            "MR_KYUTAI_VOICE", "expresso/ex03-ex01_happy_001_channel1_334s.wav"
                        )
                        self.chunk_bytes = int(msg.get("chunk_bytes") or 160)
                        codec = msg.get("codec") or "ulaw"
                        sr = int(msg.get("sample_rate") or 8000)
                        if codec != "ulaw" or sr != 8000:
                            raise ValueError("only codec=ulaw sample_rate=8000 supported by this server")
                        self._init_model(voice_rel)
                        self.started = True
                    elif op == "text":
                        text = msg.get("text") or ""
                        if text:
                            self.text_q.put(text)
                    elif op == "finish":
                        self.text_q.put(_END)
                        break
                    else:
                        raise ValueError(f"unknown op: {op!r}")
            except Exception as e:
                logger.exception(f"remote_server rx error: {e}")
                try:
                    self.text_q.put(_END)
                except Exception:
                    pass
                try:
                    self.send_error(str(e))
                except Exception:
                    pass

        rx_thread = threading.Thread(target=rx, daemon=True)
        rx_thread.start()

        try:
            # Wait until started or ended
            while not self.started and rx_thread.is_alive():
                try:
                    item = self.text_q.get(timeout=0.05)
                except queue.Empty:
                    continue
                if item is _END:
                    return

            if not self.started:
                return

            assert self.tts_model is not None and self.gen is not None
            with self.tts_model.mimi.streaming(1):
                while True:
                    item = self.text_q.get()
                    if item is _END:
                        self.gen.process_last()
                        break
                    if not isinstance(item, str) or not item.strip():
                        continue
                    entries = _prepare_script_piece(self.tts_model, item, self.first_turn)
                    self.first_turn = False
                    for e in entries:
                        self.gen.append_entry(e)
                        self.gen.process()

            self.send_end()

        except Exception as e:
            logger.exception(f"remote_server session error: {e}")
            try:
                self.send_error(str(e))
            except Exception:
                pass
        finally:
            try:
                self.sock.shutdown(socket.SHUT_RDWR)
            except Exception:
                pass
            try:
                self.sock.close()
            except Exception:
                pass


class KyutaiTTSHandler(socketserver.BaseRequestHandler):
    def handle(self):
        sock: socket.socket = self.request
        sock.settimeout(60.0)
        logger.info(f"remote_server: connection from {self.client_address}")
        _Session(sock).run()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--log-level", default=os.environ.get("LOG_LEVEL", "INFO"))
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    logger.info(f"mr_kyutai.remote_server: listening on {args.host}:{args.port}")

    # Load model at startup (avoids first-connection latency spikes)
    get_tts_model()

    class _Server(socketserver.ThreadingTCPServer):
        allow_reuse_address = True
        daemon_threads = True

    with _Server((args.host, args.port), KyutaiTTSHandler) as srv:
        srv.serve_forever()


if __name__ == "__main__":
    main()

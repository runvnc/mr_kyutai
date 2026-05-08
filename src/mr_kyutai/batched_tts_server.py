#!/usr/bin/env python3
"""Batched Kyutai TTS WebSocket server.

Wraps Kyutai's TTSService (from moshi-server/tts.py) in a Python WebSocket
server that handles multiple concurrent connections via batched inference.

Protocol (matches moshi-server Rust implementation):
  Client -> Server (msgpack binary):
    {type: "Text", text: "word"}
    {type: "Eos"}
    {type: "Voice", embeddings: [...], shape: [...]}

  Server -> Client (msgpack binary):
    {type: "Audio", pcm: [float32...]}   # 1920 samples @ 24kHz per frame
    {type: "Ready"}
    {type: "Error", message: "..."}

  Voice selection via query params: ?voice=name&format=PcmMessagePack
  Auth via kyutai-api-key header or auth_id query param.

Also supports the legacy framed TCP protocol from remote_server.py:
  Client -> Server: 'J' frames with JSON {op: start/text/finish}
  Server -> Client: 'A' ulaw8k frames, 'E' end, 'X' error

Usage:
  python -m mr_kyutai.batched_tts_server --port 8765 --batch-size 8

Env:
  MR_KYUTAI_HF_REPO       Model repo (default: kyutai/tts-1.6b-en_fr)
  MR_KYUTAI_DEVICE         cuda or cpu
  MR_KYUTAI_BATCH_SIZE     Batch size (default: 8)
  MR_KYUTAI_WS_PORT        WebSocket port (default: 8765)
  MR_KYUTAI_AUTH_TOKEN     Auth token (default: public_token)
"""

import argparse
import asyncio
import audioop
import json
import logging
import os
import queue
import re
import struct
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
from urllib.parse import parse_qs, urlparse

import numpy as np

logger = logging.getLogger(__name__)

# Flags from TTSService step() - must match tts.py MaskFlags
MASK_HAS_PCM = 1
MASK_IS_EOS = 2
MASK_WORD_FINISHED = 4
MASK_AR_STEP = 8
MASK_MISSING_WORDS = 16

FRAME_SIZE = 1920  # 24kHz * 80ms

# ---------------------------------------------------------------------------
# Text tokenizer (port of Rust tts_preprocess.rs)
# ---------------------------------------------------------------------------

_BREAK_RE = re.compile(r'<break\s+time="([0-9.]+)s"\s*/>')


def _normalize_text(text: str) -> str:
    return text.replace("\u2019", "'").replace("\u2013", "").replace(":", " ").replace("(", "").replace(")", "")


class TextTokenizer:
    """Tokenizes text into word-token lists for TTSService.step().

    Mirrors the Rust tts_preprocess::Tokenizer.
    """

    def __init__(self, sp_model, text_bos_token: int = 1, pad_id: int = 3):
        self.sp = sp_model
        self.text_bos_token = text_bos_token
        self.pad_id = pad_id
        self.inserted_bos = False

    def reset(self):
        self.inserted_bos = False

    def tokenize_word(self, word: str) -> list[int]:
        """Tokenize a single word, prepending BOS on first call."""
        if not word:
            return []
        tokens = self.sp.encode(word)  # returns list[int]
        if not self.inserted_bos:
            self.inserted_bos = True
            tokens.insert(0, self.text_bos_token)
        return tokens

    def preprocess(self, text: str) -> list[tuple[str, list[int]]]:
        """Tokenize text into (word, tokens) pairs.

        Handles <break time="Ns"/> tags.
        """
        results = []
        last = 0
        for m in _BREAK_RE.finditer(text):
            if m.start() > last:
                chunk = text[last:m.start()]
                results.extend(self._tokenize_chunk(chunk))
            secs = float(m.group(1))
            if secs > 0:
                npad = max(int(min(secs, 10.0) * 12.5), 1)
                results.append((f'<break time="{secs:.2f}s">', [self.pad_id] * npad))
            last = m.end()
        if last < len(text):
            chunk = text[last:].strip()
            if chunk:
                results.extend(self._tokenize_chunk(chunk))
        return results

    def _tokenize_chunk(self, text: str) -> list[tuple[str, list[int]]]:
        text = _normalize_text(text)
        results = []
        for word in text.split():
            if not word:
                continue
            tokens = self.tokenize_word(word)
            if tokens:
                results.append((word, tokens))
        return results


# ---------------------------------------------------------------------------
# Slot / connection state
# ---------------------------------------------------------------------------

@dataclass
class SlotState:
    """Per-batch-slot state tracking a single client connection."""
    slot_idx: int
    voice: Optional[str] = None
    voice_embeddings: Optional[np.ndarray] = None
    sent_init: bool = False
    steps: int = 0
    prev_word_steps: int = 0
    words: list = field(default_factory=list)

    # Thread-safe queues for communication
    in_queue: queue.Queue = field(default_factory=queue.Queue)  # words/eos from WS handler
    out_queue: asyncio.Queue = field(default=None)  # PCM/events to WS handler
    loop_ref: asyncio.AbstractEventLoop = field(default=None)  # event loop for out_queue

    active: bool = True
    tokenizer: TextTokenizer = field(default=None)

    def on_end_of_word(self):
        """Called when WORD_FINISHED flag is set."""
        if self.prev_word_steps > 0 and self.words:
            word = self.words.pop(0) if self.words else ""
            start_s = self.prev_word_steps / 12.5
            stop_s = self.steps / 12.5
            # Could send word timestamps to client here
        self.prev_word_steps = self.steps


class _Eos:
    pass


class _VoiceUpdate:
    def __init__(self, embeddings, shape):
        self.embeddings = embeddings
        self.shape = shape


# ---------------------------------------------------------------------------
# Batched TTS Server
# ---------------------------------------------------------------------------

class BatchedTTSServer:
    def __init__(self, batch_size: int = 8, config_override: Optional[dict] = None):
        self.batch_size = batch_size
        self.config_override = config_override or {}
        self.service = None
        self.sp_model = None  # sentencepiece model
        self.text_bos_token = int(os.environ.get('MR_KYUTAI_TEXT_BOS_TOKEN', '1'))
        self.slots: list[Optional[SlotState]] = [None] * batch_size
        self.slots_lock = threading.Lock()

        self._running = False
        self._loop_thread: Optional[threading.Thread] = None

        self.auth_token = os.environ.get('MR_KYUTAI_AUTH_TOKEN', 'public_token')

        self.pad_id = 3  # updated after model load

    def allocate_slot(self, voice: Optional[str], loop: asyncio.AbstractEventLoop) -> Optional[SlotState]:
        """Allocate a free batch slot for a new connection."""
        with self.slots_lock:
            for i, slot in enumerate(self.slots):
                if slot is None:
                    tokenizer = TextTokenizer(self.sp_model, self.text_bos_token, self.pad_id)
                    slot = SlotState(
                        slot_idx=i,
                        voice=voice,
                        tokenizer=tokenizer,
                        out_queue=asyncio.Queue(),
                        loop_ref=loop,
                    )
                    self.slots[i] = slot
                    logger.info(f"Allocated slot {i}, voice={voice}")
                    return slot
        return None

    def release_slot(self, slot_idx: int):
        """Release a batch slot."""
        with self.slots_lock:
            if self.slots[slot_idx] is not None:
                logger.info(f"Released slot {slot_idx}")
                self.slots[slot_idx] = None

    def start_inference_loop(self):
        """Start the main inference loop in a background thread."""
        self._running = True
        self._loop_thread = threading.Thread(target=self._inference_loop, daemon=True)
        self._loop_thread.start()
        logger.info("Inference loop started")

    def stop(self):
        self._running = False
        if self._loop_thread:
            self._loop_thread.join(timeout=10)

    def _send_to_slot(self, slot: SlotState, msg: Any):
        """Thread-safe send to a slot's async out_queue."""
        if slot.loop_ref and slot.out_queue is not None:
            slot.loop_ref.call_soon_threadsafe(slot.out_queue.put_nowait, msg)

    def _inference_loop(self):
        """Main inference loop - runs service.step() continuously."""
        service = self.service
        bs = self.batch_size

        pcm_out = np.zeros((bs, FRAME_SIZE), dtype=np.float32)
        flags_out = np.zeros(bs, dtype=np.int32)
        code_out = np.zeros((bs, 33), dtype=np.int32)

        logger.info("Inference loop running")

        while self._running:
            # Pre-process: collect updates from all active slots
            updates = []
            slot_snapshot = []

            with self.slots_lock:
                for i in range(bs):
                    slot = self.slots[i]
                    slot_snapshot.append(slot)
                    if slot is None or not slot.active:
                        continue

                    # Try to get next message from this slot
                    try:
                        msg = slot.in_queue.get_nowait()
                    except queue.Empty:
                        continue

                    if isinstance(msg, _Eos):
                        if slot.sent_init:
                            updates.append((i, [-2], None))
                        else:
                            # Never sent anything, just release
                            self.slots[i] = None
                            slot_snapshot[i] = None
                    elif isinstance(msg, _VoiceUpdate):
                        emb = np.array(msg.embeddings, dtype=np.float32).reshape(msg.shape)
                        slot.voice_embeddings = emb
                    elif isinstance(msg, tuple):
                        # (word, tokens) from tokenizer
                        word, tokens = msg
                        slot.words.append(word)
                        t = list(tokens)
                        if not slot.sent_init:
                            t.insert(0, -1)  # reset signal
                            slot.sent_init = True
                            # Pass voice on first update
                            voice = None
                            if slot.voice_embeddings is not None:
                                voice = slot.voice_embeddings
                                slot.voice_embeddings = None
                            elif slot.voice:
                                voice = slot.voice
                            updates.append((i, t, voice))
                        else:
                            updates.append((i, t, None))

            # Check if any slots are active at all
            any_active = any(s is not None for s in slot_snapshot)
            if not any_active:
                time.sleep(0.002)  # Sleep briefly when idle
                continue

            # Step
            try:
                service.step(updates, pcm_out=pcm_out, flags_out=flags_out, code_out=code_out)
            except Exception as e:
                logger.exception(f"service.step() error: {e}")
                time.sleep(0.01)
                continue

            # Post-process: distribute results
            for i in range(bs):
                slot = slot_snapshot[i]
                if slot is None or not slot.active:
                    continue
                if not slot.sent_init:
                    continue

                mask = int(flags_out[i])

                if mask & MASK_AR_STEP:
                    slot.steps += 1

                if mask & MASK_WORD_FINISHED:
                    if slot.prev_word_steps > 0:
                        slot.on_end_of_word()
                    slot.prev_word_steps = slot.steps

                if mask & MASK_HAS_PCM:
                    pcm = pcm_out[i].tolist()  # 1920 float32 samples @ 24kHz
                    self._send_to_slot(slot, ('audio', pcm))

                if mask & MASK_IS_EOS:
                    if slot.words:
                        slot.on_end_of_word()
                    logger.info(f"Slot {i} TTS finished after {slot.steps} steps")
                    self._send_to_slot(slot, ('eos', None))
                    with self.slots_lock:
                        self.slots[i] = None

    # -------------------------------------------------------------------
    # WebSocket handler (msgpack protocol, matches moshi-server)
    # -------------------------------------------------------------------

    async def handle_websocket(self, websocket):
        """Handle a single WebSocket connection."""
        import msgpack

        # Parse query params for voice and format
        path = getattr(websocket, 'path', '') or ''
        if hasattr(websocket, 'request') and hasattr(websocket.request, 'path'):
            path = websocket.request.path or ''
        params = parse_qs(urlparse(path).query)
        voice = (params.get('voice', [None]) or [None])[0]
        if not voice:
            voice = os.environ.get(
                'MR_KYUTAI_VOICE', 'expresso/ex03-ex01_happy_001_channel1_334s.wav'
            )

        loop = asyncio.get_event_loop()
        slot = self.allocate_slot(voice, loop)
        if slot is None:
            err = msgpack.packb({'type': 'Error', 'message': 'no free channels'})
            await websocket.send(err)
            await websocket.close()
            return

        try:
            # Send Ready
            await websocket.send(msgpack.packb({'type': 'Ready'}))

            # Receive and send tasks
            async def recv_loop():
                try:
                    async for raw in websocket:
                        msg = msgpack.unpackb(raw, raw=False)
                        if not isinstance(msg, dict):
                            continue
                        mtype = msg.get('type', '')
                        if mtype == 'Text':
                            text = msg.get('text', '')
                            if text:
                                pairs = slot.tokenizer.preprocess(text)
                                for word, tokens in pairs:
                                    slot.in_queue.put((word, tokens))
                        elif mtype == 'Eos':
                            slot.in_queue.put(_Eos())
                            break
                        elif mtype == 'Voice':
                            emb = msg.get('embeddings', [])
                            shape = msg.get('shape', [])
                            slot.in_queue.put(_VoiceUpdate(emb, shape))
                except Exception as e:
                    logger.debug(f"Slot {slot.slot_idx} recv error: {e}")
                finally:
                    slot.in_queue.put(_Eos())

            async def send_loop():
                try:
                    while True:
                        msg = await slot.out_queue.get()
                        if msg is None:
                            break
                        mtype, data = msg
                        if mtype == 'audio':
                            await websocket.send(msgpack.packb({'type': 'Audio', 'pcm': data}))
                        elif mtype == 'eos':
                            break
                        elif mtype == 'error':
                            await websocket.send(msgpack.packb({'type': 'Error', 'message': data}))
                            break
                except Exception as e:
                    logger.debug(f"Slot {slot.slot_idx} send error: {e}")

            await asyncio.gather(recv_loop(), send_loop())

        except Exception as e:
            logger.exception(f"WebSocket handler error for slot {slot.slot_idx}: {e}")
        finally:
            self.release_slot(slot.slot_idx)
            try:
                await websocket.close()
            except Exception:
                pass

    # -------------------------------------------------------------------
    # Legacy TCP handler (framed protocol, matches remote_server.py)
    # -------------------------------------------------------------------

    async def handle_tcp_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        """Handle a legacy TCP connection (framed J/A/E/X protocol)."""
        addr = writer.get_extra_info('peername')
        logger.info(f"TCP connection from {addr}")

        loop = asyncio.get_event_loop()
        slot = None
        ratecv_state = None
        src_sr = 24000
        chunk_bytes = 160

        async def recv_frame():
            header = await reader.readexactly(5)
            ftype = header[:1]
            n = struct.unpack('>I', header[1:])[0]
            payload = await reader.readexactly(n) if n else b''
            return ftype, payload

        def send_frame(ftype: bytes, payload: bytes):
            header = ftype + struct.pack('>I', len(payload))
            writer.write(header + payload)

        try:
            while True:
                ftype, payload = await recv_frame()
                if ftype != b'J':
                    raise ValueError(f"expected J frame, got {ftype!r}")
                msg = json.loads(payload.decode('utf-8'))
                op = msg.get('op')

                if op == 'start':
                    voice = msg.get('voice') or os.environ.get(
                        'MR_KYUTAI_VOICE', 'expresso/ex03-ex01_happy_001_channel1_334s.wav'
                    )
                    chunk_bytes = int(msg.get('chunk_bytes', 160))
                    slot = self.allocate_slot(voice, loop)
                    if slot is None:
                        send_frame(b'X', b'no free channels')
                        await writer.drain()
                        return

                    # Start send loop for this TCP connection
                    async def tcp_send_loop():
                        nonlocal ratecv_state
                        try:
                            while True:
                                out = await slot.out_queue.get()
                                if out is None:
                                    break
                                mtype, data = out
                                if mtype == 'audio':
                                    # Convert 24kHz float32 PCM -> 8kHz ulaw
                                    pcm = np.array(data, dtype=np.float32)
                                    pcm = np.clip(pcm, -1.0, 1.0)
                                    pcm16 = (pcm * 32767.0).astype(np.int16).tobytes()
                                    pcm8k, ratecv_state = audioop.ratecv(
                                        pcm16, 2, 1, src_sr, 8000, ratecv_state
                                    )
                                    ulaw = audioop.lin2ulaw(pcm8k, 2)
                                    for j in range(0, len(ulaw), chunk_bytes):
                                        ch = ulaw[j:j + chunk_bytes]
                                        if ch:
                                            send_frame(b'A', ch)
                                    await writer.drain()
                                elif mtype == 'eos':
                                    send_frame(b'E', b'')
                                    await writer.drain()
                                    break
                                elif mtype == 'error':
                                    send_frame(b'X', (data or 'error').encode('utf-8'))
                                    await writer.drain()
                                    break
                        except Exception as e:
                            logger.debug(f"TCP send error: {e}")

                    send_task = asyncio.create_task(tcp_send_loop())

                elif op == 'text' and slot:
                    text = msg.get('text', '')
                    if text:
                        pairs = slot.tokenizer.preprocess(text)
                        for word, tokens in pairs:
                            slot.in_queue.put((word, tokens))

                elif op == 'finish' and slot:
                    slot.in_queue.put(_Eos())
                    # Wait for send loop to complete
                    if send_task:
                        await send_task
                    break

        except (asyncio.IncompleteReadError, ConnectionError) as e:
            logger.debug(f"TCP connection closed: {e}")
            if slot:
                slot.in_queue.put(_Eos())
        except Exception as e:
            logger.exception(f"TCP handler error: {e}")
            if slot:
                slot.in_queue.put(_Eos())
                try:
                    send_frame(b'X', str(e).encode('utf-8'))
                    await writer.drain()
                except Exception:
                    pass
        finally:
            if slot:
                self.release_slot(slot.slot_idx)
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# tts_service.py shim - imports init() from the copied tts.py
# ---------------------------------------------------------------------------

def _create_tts_init():
    """Create the init function that wraps tts.py's init()."""
    import importlib.util

    # Search for tts.py in likely locations:
    # 1) mr_kyutai repo root (e.g. /xfiles/plugins_ah/mr_kyutai/tts.py)
    # 2) Next to this file
    # 3) Parent dirs
    candidates = [
        Path(__file__).resolve().parent.parent.parent / 'tts.py',  # repo root (src/mr_kyutai -> src -> mr_kyutai_repo)
        Path(__file__).resolve().parent / 'tts.py',
        Path(__file__).resolve().parent.parent / 'tts.py',
    ]
    tts_py_path = None
    for candidate in candidates:
        if candidate.exists():
            tts_py_path = candidate
            break

    if tts_py_path is None:
        raise FileNotFoundError(
            f"Cannot find tts.py (TTSService). Searched: {[str(c) for c in candidates]}"
        )

    logger.info(f"Loading TTSService from {tts_py_path}")
    spec = importlib.util.spec_from_file_location('mr_kyutai._tts_service', str(tts_py_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.init


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

async def serve(host: str = '0.0.0.0', ws_port: int = 8765, tcp_port: int = 0,
                batch_size: int = 8, config_override: Optional[dict] = None):
    """Start the batched TTS server."""
    import websockets

    server = BatchedTTSServer(batch_size=batch_size, config_override=config_override)

    # Initialize in a thread to not block the event loop
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, lambda: _init_server(server))

    server.start_inference_loop()

    # Start WebSocket server
    ws_server = await websockets.serve(
        server.handle_websocket, host, ws_port,
        max_size=10 * 1024 * 1024,  # 10MB max message
    )
    logger.info(f"WebSocket server listening on ws://{host}:{ws_port}")

    # Optionally start TCP server for legacy protocol
    tcp_server = None
    if tcp_port > 0:
        tcp_server = await asyncio.start_server(
            server.handle_tcp_client, host, tcp_port
        )
        logger.info(f"TCP server listening on {host}:{tcp_port}")

    try:
        await asyncio.Future()  # run forever
    finally:
        server.stop()
        ws_server.close()
        if tcp_server:
            tcp_server.close()


def _init_server(server: BatchedTTSServer):
    """Initialize the server (loads model, creates TTSService)."""
    tts_init = _create_tts_init()

    config = {
        'hf_repo': os.environ.get('MR_KYUTAI_HF_REPO', 'kyutai/tts-1.6b-en_fr'),
        'device': os.environ.get('MR_KYUTAI_DEVICE', 'cuda'),
        'n_q': int(os.environ.get('MR_KYUTAI_NQ', '24')),
        'voice_folder': os.environ.get(
            'MR_KYUTAI_VOICE_FOLDER',
            'hf-snapshot://kyutai/tts-voices/**/*.safetensors'
        ),
        'default_voice': os.environ.get(
            'MR_KYUTAI_DEFAULT_VOICE',
            'unmute-prod-website/default_voice.wav'
        ),
    }
    config.update(server.config_override)

    # Use the init() from tts.py
    server.service = tts_init(
        batch_size=server.batch_size,
        config_override=config,
    )

    # Get tokenizer from model
    server.sp_model = server.service.tts_model.tokenizer
    try:
        pid = server.sp_model.pad_id()
        if pid is not None and pid >= 0:
            server.pad_id = pid
    except Exception:
        pass

    logger.info(f"Server initialized: batch_size={server.batch_size}, "
                f"pad_id={server.pad_id}, bos={server.text_bos_token}")


def main():
    parser = argparse.ArgumentParser(description='Batched Kyutai TTS Server')
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--ws-port', type=int,
                        default=int(os.environ.get('MR_KYUTAI_WS_PORT', '8765')))
    parser.add_argument('--tcp-port', type=int, default=0,
                        help='Legacy TCP port (0=disabled)')
    parser.add_argument('--batch-size', type=int,
                        default=int(os.environ.get('MR_KYUTAI_BATCH_SIZE', '8')))
    parser.add_argument('--log-level', default=os.environ.get('LOG_LEVEL', 'INFO'))
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format='[%(asctime)s] %(levelname)s %(name)s: %(message)s',
    )

    asyncio.run(serve(
        host=args.host,
        ws_port=args.ws_port,
        tcp_port=args.tcp_port,
        batch_size=args.batch_size,
    ))


if __name__ == '__main__':
    main()

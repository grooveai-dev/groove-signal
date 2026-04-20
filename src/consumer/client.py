"""Groove Decentralized Inference - Consumer Client.

The consumer holds a single websocket to the signal service (or legacy
relay). All work is sent in ENVELOPEs addressed by stream_id (returned in
PIPELINE_CONFIG) and target_node_id (the node the consumer wants to reach).
Responses come back via the same socket — a single receive task fans them
out to per-seq_pos futures. The consumer never learns or contacts a node
directly.

M3: --signal flag connects to the Groove signal service over TLS. The signal
service scores and matches nodes, then brokers the connection.
"""

import argparse
import asyncio
import json
import logging
import sys
import time
from typing import AsyncGenerator, Optional

import msgpack
import numpy as np
import torch
import websockets

logger = logging.getLogger("consumer")

from src.common.protocol import (
    ACTIVATIONS,
    ENVELOPE,
    ERROR,
    LOGITS,
    PIPELINE_CONFIG,
    PROTOCOL_VERSION,
    VERIFY_RESULT,
    decode_message,
    encode_message,
    make_activations,
    make_envelope,
    make_session_init,
    new_session_id,
)
from src.common.tensor_transfer import deserialize_tensor, serialize_tensor


def _logits_from_response(response: dict) -> np.ndarray:
    blob = response["logits_bytes"]
    tensor = deserialize_tensor(blob)
    return tensor.cpu().float().numpy()


def _last_token_logits(logits: np.ndarray) -> np.ndarray:
    while logits.ndim > 1:
        logits = logits[-1]
    return logits


class InferenceClient:
    def __init__(
        self,
        relay_host: str = "localhost",
        relay_port: int = 8770,
        json_mode: bool = False,
        use_tls: bool = False,
        signal_mode: bool = False,
    ):
        self.relay_host = relay_host
        self.relay_port = relay_port
        self.use_tls = use_tls
        self.signal_mode = signal_mode
        self.relay_ws = None
        self.session_id: Optional[str] = None
        self.stream_id: Optional[str] = None
        self.pipeline: list[dict] = []
        self.tokenizer = None
        self.model_name: Optional[str] = None
        self._waiters: dict[int, asyncio.Future] = {}
        self._recv_task: Optional[asyncio.Task] = None
        self._closed = False
        self.envelope_count: int = 0
        self.json_mode = json_mode
        self.tokens_generated: int = 0

    def emit_event(self, event: dict) -> None:
        """Print one JSON event line to stdout. No-op when json_mode is off."""
        if not self.json_mode:
            return
        print(json.dumps(event), flush=True)

    async def connect(self) -> None:
        if self.relay_host.startswith("wss://") or self.relay_host.startswith("ws://"):
            uri = self.relay_host
        elif self.signal_mode:
            uri = f"wss://{self.relay_host}"
        else:
            scheme = "wss" if self.use_tls else "ws"
            uri = f"{scheme}://{self.relay_host}:{self.relay_port}"
        self.relay_ws = await websockets.connect(uri, max_size=10 * 1024 * 1024)
        if self.signal_mode:
            self.emit_event({
                "type": "signal_connected",
                "signal": self.relay_host,
            })

    async def start_session(
        self, model_name: str, config: Optional[dict] = None
    ) -> str:
        from transformers import AutoTokenizer

        self.model_name = model_name
        self.session_id = new_session_id()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=False,
        )

        init = make_session_init(self.session_id, model_name, 0, -1, config or {})
        await self.relay_ws.send(encode_message(init))

        raw = await self.relay_ws.recv()
        if isinstance(raw, str):
            raw = raw.encode("utf-8")
        response = decode_message(raw)

        if response["type"] == ERROR:
            msg = response.get("message", "")
            code = response.get("code")
            if code in ("COVERAGE_INCOMPLETE", "NO_NODES") or "coverage" in msg.lower():
                raise RuntimeError(
                    "Network does not have full model coverage yet. "
                    "Waiting for more nodes to join..."
                )
            raise RuntimeError(f"Session init failed: {msg}")
        if response["type"] != PIPELINE_CONFIG:
            raise RuntimeError(f"Unexpected response type: {response['type']}")

        self.pipeline = response["nodes"]
        self.stream_id = response.get("stream_id")
        if not self.stream_id:
            raise RuntimeError("Relay did not return a stream_id in PIPELINE_CONFIG")

        if self.signal_mode:
            self.emit_event({
                "type": "matched",
                "nodes": [
                    {
                        "node_id": n["node_id"],
                        "score": n.get("score"),
                        "device": n.get("device"),
                        "layers": [n.get("layer_start"), n.get("layer_end")],
                    }
                    for n in self.pipeline
                ],
            })
        self.emit_event({
            "type": "connected",
            "signal" if self.signal_mode else "relay": (
                self.relay_host if self.signal_mode
                else f"{self.relay_host}:{self.relay_port}"
            ),
            "session_id": self.session_id,
        })
        self.emit_event({
            "type": "pipeline",
            "nodes": [
                {
                    "node_id": n["node_id"],
                    "layers": [n.get("layer_start"), n.get("layer_end")],
                }
                for n in self.pipeline
            ],
        })

        self._recv_task = asyncio.create_task(self._receive_loop())
        return self.session_id

    async def _receive_loop(self) -> None:
        try:
            async for raw in self.relay_ws:
                if isinstance(raw, str):
                    raw = raw.encode("utf-8")
                try:
                    msg = decode_message(raw)
                except Exception:
                    logger.warning("failed to decode message from relay, skipping")
                    continue

                if msg["type"] == ENVELOPE:
                    try:
                        inner = msgpack.unpackb(
                            msg["payload"], raw=False,
                            max_str_len=10 * 1024 * 1024,
                            max_bin_len=10 * 1024 * 1024,
                            max_array_len=10_000,
                            max_map_len=1_000,
                        )
                    except Exception:
                        logger.warning("failed to decode envelope payload, skipping")
                        continue
                    if inner.get("type") == ERROR:
                        seq = inner.get("seq_pos")
                        err = RuntimeError(
                            f"Node error [{inner.get('code')}]: "
                            f"{inner.get('message', '')}"
                        )
                        if seq is not None:
                            fut = self._waiters.pop(seq, None)
                            if fut is not None and not fut.done():
                                fut.set_exception(err)
                        else:
                            for fut in list(self._waiters.values()):
                                if not fut.done():
                                    fut.set_exception(err)
                            self._waiters.clear()
                        continue
                    seq = inner.get("seq_pos")
                    fut = self._waiters.pop(seq, None)
                    if fut is not None and not fut.done():
                        fut.set_result(inner)
                elif msg["type"] == ERROR:
                    err = RuntimeError(
                        f"Relay error [{msg.get('code')}]: {msg.get('message', '')}"
                    )
                    for fut in list(self._waiters.values()):
                        if not fut.done():
                            fut.set_exception(err)
                    self._waiters.clear()
        except websockets.ConnectionClosed:
            pass
        except Exception:
            logger.exception("receive loop crashed")
        finally:
            err = ConnectionError("Relay connection closed")
            for fut in list(self._waiters.values()):
                if not fut.done():
                    fut.set_exception(err)
            self._waiters.clear()

    async def _send_to_node(
        self, node_id: str, inner: dict, timeout: float = 120.0,
    ) -> dict:
        if self.relay_ws is None or self.stream_id is None:
            raise RuntimeError("No active session — call connect()/start_session() first")
        if self._recv_task is not None and self._recv_task.done():
            exc = self._recv_task.exception() if not self._recv_task.cancelled() else None
            raise ConnectionError(
                f"Receive loop died before response: {exc or 'cancelled'}"
            )
        seq = inner["seq_pos"]
        if seq in self._waiters:
            raise RuntimeError(f"In-flight request already exists for seq_pos={seq}")
        loop = asyncio.get_event_loop()
        fut: asyncio.Future = loop.create_future()
        self._waiters[seq] = fut

        self.envelope_count += 1
        env = make_envelope(
            self.stream_id,
            msgpack.packb(inner, use_bin_type=True),
            target_node_id=node_id,
            envelope_count=self.envelope_count,
        )
        try:
            await self.relay_ws.send(encode_message(env))
        except Exception:
            self._waiters.pop(seq, None)
            raise

        try:
            return await asyncio.wait_for(fut, timeout=timeout)
        except asyncio.TimeoutError:
            self._waiters.pop(seq, None)
            raise RuntimeError(
                f"Pipeline node {node_id[:10]}… did not respond within "
                f"{timeout}s (seq_pos={seq})"
            )

    async def send_to_pipeline(self, message: dict) -> dict:
        """Forward a message through every node in pipeline order.

        The message flows node-by-node: each non-terminal node returns
        ACTIVATIONS that we hand off to the next node; we return the first
        terminal response (LOGITS, VERIFY_RESULT, or ERROR).
        """
        msg = message
        last_response: dict = {}
        for node in self.pipeline:
            response = await self._send_to_node(node["node_id"], msg)
            t = response.get("type")
            if t in (LOGITS, VERIFY_RESULT, ERROR):
                return response
            if t == ACTIVATIONS:
                msg = response
                continue
            return response
        return last_response

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 200,
        use_speculative: bool = False,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> AsyncGenerator[str, None]:
        if not self.session_id or not self.tokenizer:
            raise RuntimeError("No active session. Call start_session() first.")

        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")[0].tolist()

        if use_speculative:
            from src.consumer.speculative import SpeculativeDecoder
            from src.consumer.draft_model import DraftModel

            draft = DraftModel(device="cpu")
            await asyncio.to_thread(draft.load)
            decoder = SpeculativeDecoder(draft_model=draft, client=self, window_size=8)
            async for token_batch in decoder.full_generate(
                input_ids, max_tokens, temperature=temperature, top_p=top_p
            ):
                text = self.tokenizer.decode(token_batch, skip_special_tokens=True)
                self.tokens_generated += 1
                self.emit_event({
                    "type": "token",
                    "text": text,
                    "tokens_generated": self.tokens_generated,
                })
                yield text
        else:
            async for token_text in self._autoregressive_generate(
                input_ids, max_tokens, temperature, top_p
            ):
                self.tokens_generated += 1
                self.emit_event({
                    "type": "token",
                    "text": token_text,
                    "tokens_generated": self.tokens_generated,
                })
                yield token_text

    async def _autoregressive_generate(
        self,
        input_ids: list[int],
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> AsyncGenerator[str, None]:
        generated = list(input_ids)
        eos_id = self.tokenizer.eos_token_id

        prompt_tensor = torch.tensor(generated, dtype=torch.int64)
        msg = make_activations(
            self.session_id,
            seq_pos=0,
            hidden_states_bytes=serialize_tensor(prompt_tensor),
            shape=tuple(prompt_tensor.shape),
            dtype="int64",
        )
        msg["is_prompt"] = True

        response = await self.send_to_pipeline(msg)
        if response.get("type") == ERROR:
            raise RuntimeError(f"Pipeline error: {response.get('message', '')}")

        logits = _logits_from_response(response)
        next_token = self._sample_token(
            _last_token_logits(logits), temperature, top_p
        )
        generated.append(next_token)
        yield self.tokenizer.decode([next_token], skip_special_tokens=True)

        for _ in range(max_tokens - 1):
            if next_token == eos_id:
                break

            token_tensor = torch.tensor([next_token], dtype=torch.int64)
            msg = make_activations(
                self.session_id,
                seq_pos=len(generated) - 1,
                hidden_states_bytes=serialize_tensor(token_tensor),
                shape=tuple(token_tensor.shape),
                dtype="int64",
            )
            msg["is_prompt"] = False

            response = await self.send_to_pipeline(msg)
            if response.get("type") == ERROR:
                raise RuntimeError(f"Pipeline error: {response.get('message', '')}")

            logits = _logits_from_response(response)
            next_token = self._sample_token(
                _last_token_logits(logits), temperature, top_p
            )
            generated.append(next_token)
            yield self.tokenizer.decode([next_token], skip_special_tokens=True)

    def _sample_token(
        self, logits: np.ndarray, temperature: float, top_p: float
    ) -> int:
        if temperature <= 0:
            return int(np.argmax(logits))

        logits = logits.astype(np.float64)
        logits /= temperature
        logits -= np.max(logits)
        probs = np.exp(logits)
        probs /= np.sum(probs)

        sorted_indices = np.argsort(probs)[::-1]
        sorted_probs = probs[sorted_indices]
        cumulative = np.cumsum(sorted_probs)

        cutoff_idx = int(np.searchsorted(cumulative, top_p)) + 1
        top_indices = sorted_indices[:cutoff_idx]
        top_probs = probs[top_indices]
        top_probs /= np.sum(top_probs)

        return int(np.random.choice(top_indices, p=top_probs))

    async def close_session(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._recv_task is not None:
            self._recv_task.cancel()
            try:
                await self._recv_task
            except (asyncio.CancelledError, Exception):
                pass
            self._recv_task = None
        if self.relay_ws:
            try:
                await self.relay_ws.close()
            except Exception:
                pass
            self.relay_ws = None
        self.session_id = None
        self.stream_id = None
        self.pipeline = []


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Groove Decentralized Inference Client"
    )
    conn_group = parser.add_mutually_exclusive_group(required=True)
    conn_group.add_argument(
        "--signal", type=str,
        help="Signal service hostname (e.g. signal.groovedev.ai). Uses wss:// automatically.",
    )
    conn_group.add_argument(
        "--relay", type=str,
        help="Legacy relay host:port (use --signal for production)",
    )
    parser.add_argument(
        "--model", type=str, default="Qwen/Qwen2.5-7B", help="Model name"
    )
    parser.add_argument("--prompt", type=str, required=True, help="Input prompt")
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--speculative", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument(
        "--json", action="store_true",
        help="Emit structured JSON events (one per line) on stdout",
    )
    parser.add_argument("--tls", action="store_true", help="Use wss:// (automatic with --signal)")
    args = parser.parse_args()

    if args.signal:
        host = args.signal
        port = 443
        use_tls = True
        signal_mode = True
    else:
        host, port_str = args.relay.rsplit(":", 1)
        port = int(port_str)
        use_tls = args.tls
        signal_mode = False
    client = InferenceClient(
        relay_host=host, relay_port=port, json_mode=args.json,
        use_tls=use_tls, signal_mode=signal_mode,
    )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        stream=sys.stderr,
    )

    start_ms = time.time() * 1000.0

    connect_label = f"signal={args.signal}" if args.signal else f"relay={args.relay}"

    try:
        try:
            await client.connect()
        except (OSError, websockets.WebSocketException) as e:
            client.emit_event({
                "type": "error",
                "message": f"Failed to connect to {connect_label}: {e}",
                "code": "CONNECTION_FAILED",
            })
            logger.error("connect failed: %s", e)
            sys.exit(1)
        logger.info("connected to %s (protocol v%d)", connect_label, PROTOCOL_VERSION)

        session_id = await client.start_session(args.model)
        logger.info(
            "session=%s stream=%s pipeline=%d nodes",
            session_id, client.stream_id, len(client.pipeline),
        )

        if not args.json:
            sys.stdout.write("Output: ")
            sys.stdout.flush()

        async for text in client.generate(
            args.prompt,
            max_tokens=args.max_tokens,
            use_speculative=args.speculative,
            temperature=args.temperature,
            top_p=args.top_p,
        ):
            if not args.json:
                sys.stdout.write(text)
                sys.stdout.flush()

        if not args.json:
            sys.stdout.write("\n")
        client.emit_event({
            "type": "done",
            "tokens_generated": client.tokens_generated,
            "total_time_ms": int(time.time() * 1000.0 - start_ms),
        })
        logger.info("generation complete")
    except asyncio.TimeoutError as e:
        client.emit_event({
            "type": "error",
            "message": f"Operation timed out: {e}",
            "code": "TIMEOUT",
        })
        logger.error("inference timed out: %s", e)
        sys.exit(1)
    except RuntimeError as e:
        msg = str(e)
        low = msg.lower()
        if "coverage" in low:
            code = "COVERAGE_INCOMPLETE"
        elif "no nodes" in low or "no active compute" in low:
            code = "NO_NODES"
        elif "lost node" in low or "node_gone" in low or "pipeline" in low:
            code = "PIPELINE_BROKEN"
        elif "node error" in low or "no shard" in low:
            code = "NODE_ERROR"
        else:
            code = "SESSION_ERROR"
        client.emit_event({"type": "error", "message": msg, "code": code})
        logger.error("inference failed: %s", e)
        sys.exit(1)
    except ConnectionError as e:
        client.emit_event({
            "type": "error",
            "message": str(e),
            "code": "CONNECTION_FAILED",
        })
        logger.error("connection lost: %s", e)
        sys.exit(1)
    except Exception as e:
        client.emit_event({
            "type": "error",
            "message": str(e),
            "code": "SESSION_ERROR",
        })
        logger.error("inference failed: %s", e)
        sys.exit(1)
    finally:
        await client.close_session()


if __name__ == "__main__":
    asyncio.run(main())

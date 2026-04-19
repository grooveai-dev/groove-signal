"""M1 smoke-test consumer: opens a relay websocket, receives PIPELINE_CONFIG,
sends a single envelope to the first node, and asserts the reply envelope
carries the same payload.

This script does NOT load a tokenizer or model — it exists purely to
validate the v2 data plane (relay-mediated envelope routing).
"""

import argparse
import asyncio
import sys

import msgpack
import websockets

from src.common.protocol import (
    ENVELOPE,
    PIPELINE_CONFIG,
    decode_message,
    encode_message,
    make_envelope,
    make_session_init,
)


async def run(relay: str, model_name: str) -> int:
    uri = f"ws://{relay}"
    async with websockets.connect(uri, max_size=16 * 1024 * 1024) as ws:
        session_id = "smoke-sess-1"
        await ws.send(encode_message(
            make_session_init(session_id, model_name, 0, -1, {})
        ))

        raw = await asyncio.wait_for(ws.recv(), timeout=5.0)
        if isinstance(raw, str):
            raw = raw.encode("utf-8")
        cfg = decode_message(raw)

        if cfg.get("type") != PIPELINE_CONFIG:
            print(f"[smoke_consumer] FAIL: expected pipeline_config, got {cfg!r}", file=sys.stderr)
            return 1

        stream_id = cfg["stream_id"]
        nodes = cfg["nodes"]
        if not nodes:
            print("[smoke_consumer] FAIL: pipeline_config has no nodes", file=sys.stderr)
            return 1

        first_node = nodes[0]
        for field in ("host", "port"):
            if field in first_node:
                print(f"[smoke_consumer] FAIL: pipeline_config node carries forbidden field {field!r}", file=sys.stderr)
                return 1

        print(f"[smoke_consumer] pipeline_config: stream_id={stream_id} nodes={nodes}", flush=True)

        payload_obj = {"smoke": "ping", "n": 42}
        payload = msgpack.packb(payload_obj, use_bin_type=True)
        target = first_node["node_id"]
        await ws.send(encode_message(
            make_envelope(stream_id, payload, target_node_id=target)
        ))
        print(f"[smoke_consumer] sent envelope to {target}", flush=True)

        raw = await asyncio.wait_for(ws.recv(), timeout=5.0)
        if isinstance(raw, str):
            raw = raw.encode("utf-8")
        reply = decode_message(raw)

        if reply.get("type") != ENVELOPE:
            print(f"[smoke_consumer] FAIL: expected envelope reply, got {reply!r}", file=sys.stderr)
            return 1
        if reply.get("stream_id") != stream_id:
            print(f"[smoke_consumer] FAIL: stream_id mismatch ({reply.get('stream_id')} != {stream_id})", file=sys.stderr)
            return 1

        echoed = msgpack.unpackb(reply["payload"], raw=False)
        if echoed != payload_obj:
            print(f"[smoke_consumer] FAIL: payload mismatch ({echoed!r} != {payload_obj!r})", file=sys.stderr)
            return 1

        print(f"[smoke_consumer] PASS: envelope round-trip ok, echoed={echoed}", flush=True)
        return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="M1 smoke-test consumer")
    parser.add_argument("--relay", default="127.0.0.1:8770")
    parser.add_argument("--model", default="smoke-model")
    args = parser.parse_args()
    rc = asyncio.run(run(args.relay, args.model))
    sys.exit(rc)


if __name__ == "__main__":
    main()

"""Mock compute node for the Milestone 1 data-plane smoke test.

Speaks the v2 protocol: opens one outbound websocket to the relay, sends
REGISTER_NODE, waits for REGISTER_ACK, then echoes every inbound ENVELOPE
payload back in an outbound ENVELOPE on the same stream. No model load.
"""

import argparse
import asyncio
import signal
import sys

from websockets.asyncio.client import connect as ws_connect
from websockets.exceptions import ConnectionClosed

from src.common.protocol import (
    ENVELOPE,
    REGISTER_ACK,
    decode_message,
    encode_message,
    make_deregister,
    make_envelope,
    make_register_node,
)


async def run(relay: str, node_id: str, layer_start: int, layer_end: int) -> None:
    stop = asyncio.Event()

    def _request_stop() -> None:
        stop.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _request_stop)
        except NotImplementedError:
            pass

    async with ws_connect(f"ws://{relay}", max_size=16 * 1024 * 1024) as ws:
        await ws.send(encode_message(
            make_register_node(node_id, layer_start, layer_end, {"mock": True})
        ))
        raw = await asyncio.wait_for(ws.recv(), timeout=5.0)
        if isinstance(raw, str):
            raw = raw.encode("utf-8")
        ack = decode_message(raw)
        if ack.get("type") != REGISTER_ACK or not ack.get("accepted", False):
            print(f"[mock_node] register rejected: {ack}", file=sys.stderr)
            return
        print(f"[mock_node] registered as {node_id} layers=[{layer_start},{layer_end})", flush=True)

        async def _recv_loop() -> None:
            async for raw in ws:
                if isinstance(raw, str):
                    raw = raw.encode("utf-8")
                msg = decode_message(raw)
                if msg.get("type") != ENVELOPE:
                    continue
                stream_id = msg["stream_id"]
                payload = msg["payload"]
                print(f"[mock_node] echoing envelope stream={stream_id} bytes={len(payload)}", flush=True)
                await ws.send(encode_message(
                    make_envelope(stream_id, payload, target_node_id=None)
                ))

        recv_task = asyncio.create_task(_recv_loop())
        stop_task = asyncio.create_task(stop.wait())
        done, pending = await asyncio.wait(
            {recv_task, stop_task}, return_when=asyncio.FIRST_COMPLETED
        )
        for t in pending:
            t.cancel()
            try:
                await t
            except asyncio.CancelledError:
                pass

        try:
            await asyncio.wait_for(
                ws.send(encode_message(make_deregister(node_id, "smoke-test-shutdown"))),
                timeout=2.0,
            )
        except (ConnectionClosed, OSError, asyncio.TimeoutError):
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Mock relay-client node for M1 smoke test")
    parser.add_argument("--relay", default="127.0.0.1:8770")
    parser.add_argument("--node-id", default="mock-node-A")
    parser.add_argument("--layer-start", type=int, default=0)
    parser.add_argument("--layer-end", type=int, default=31)
    args = parser.parse_args()

    try:
        asyncio.run(run(args.relay, args.node_id, args.layer_start, args.layer_end))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

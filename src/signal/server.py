"""Groove Signal Server.

The signal service is the public-facing matchmaker for the Groove
decentralized inference network. It plays two roles simultaneously:

  1. Matchmaker: consumers query for the best nodes to run a given
     model; the signal ranks all registered nodes with a gaussian
     decay scoring algorithm (proximity, uptime, compute, load) and
     returns a ranked list.

  2. Relay (beta only): until STUN/TURN hole-punching lands, the signal
     forwards opaque envelopes between consumers and nodes. It never
     inspects payloads — routing is by stream_id / target_node_id only.

The signal is the ONLY component in the Groove stack that legitimately
binds to 0.0.0.0. Nodes bind to 127.0.0.1 and connect outbound; consumers
connect outbound; the signal is the rendezvous point.

Evolves from groove-network/src/relay/relay.py; most of the routing,
rate-limiting, and heartbeat infrastructure is structurally identical.
The additions are the SIGNAL_* message handling and the scoring-driven
pipeline assembly.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import secrets
import ssl
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from websockets.asyncio.server import serve, ServerConnection
from websockets.exceptions import ConnectionClosed

from src.common.protocol import (
    ASSIGN_LAYERS,
    ASSIGNMENT_ACK,
    AUTH_CHALLENGE,
    AUTH_RESPONSE,
    DEREGISTER,
    ENVELOPE,
    ERROR,
    HEARTBEAT,
    PIPELINE_CONFIG,
    PROTOCOL_VERSION,
    REBALANCE,
    REGISTER_NODE,
    SESSION_INIT,
    SIGNAL_DEREGISTER,
    SIGNAL_HEARTBEAT,
    SIGNAL_QUERY,
    SIGNAL_REGISTER,
    decode_message,
    encode_message,
    make_assign_layers,
    make_auth_challenge,
    make_error,
    make_pipeline_config,
    make_rebalance,
    make_register_ack,
    make_signal_ack,
    make_signal_match,
    normalize_capabilities,
)
from src.relay.scheduler import (
    MODEL_REGISTRY,
    assign_layers,
    calculate_rebalance,
    get_model_info,
    validate_coverage,
)
from src.node.identity import address_from_public_key, verify_signature
from src.signal.matcher import ConsumerMatcher
from src.signal.registry import NodeRecord, NodeRegistry
from src.signal.scoring import configure_geoip, estimate_location_from_ip


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
_RESERVED_LOG_FIELDS = frozenset({
    "args", "asctime", "created", "exc_info", "exc_text", "filename",
    "funcName", "levelname", "levelno", "lineno", "message", "module",
    "msecs", "msg", "name", "pathname", "process", "processName",
    "relativeCreated", "stack_info", "thread", "threadName", "taskName",
})


class JsonFormatter(logging.Formatter):
    """One JSON object per log record. No external deps."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for k, v in record.__dict__.items():
            if k in _RESERVED_LOG_FIELDS:
                continue
            payload[k] = v
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


logger = logging.getLogger("signal")

_NODE_ID_RE = re.compile(r'^0x[0-9a-fA-F]{40}$')


# ---------------------------------------------------------------------------
# Consumer stream state (mirrors relay's ConsumerEntry)
# ---------------------------------------------------------------------------
@dataclass
class ConsumerStream:
    stream_id: str
    ws: ServerConnection
    session_id: str
    pipeline: list[str]
    model_name: str = ""
    created_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    envelope_counts: dict[str, int] = field(default_factory=dict)
    last_envelope_count: dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Simple rate limiter (token bucket)
# ---------------------------------------------------------------------------
class _RateLimiter:
    __slots__ = ("_rate", "_burst", "_tokens", "_last")

    def __init__(self, rate: float = 100.0, burst: int = 200):
        self._rate = rate
        self._burst = float(burst)
        self._tokens = float(burst)
        self._last = time.monotonic()

    def allow(self) -> bool:
        now = time.monotonic()
        self._tokens = min(
            self._burst, self._tokens + (now - self._last) * self._rate
        )
        self._last = now
        if self._tokens >= 1.0:
            self._tokens -= 1.0
            return True
        return False


# ---------------------------------------------------------------------------
# SignalServer
# ---------------------------------------------------------------------------
class SignalServer:
    ASSIGNMENT_TIMEOUT_S = 600.0

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8770,
        http_port: int = 8771,
        tls_cert: str | None = None,
        tls_key: str | None = None,
        cors_origin: str | None = None,
        max_connections_per_ip: int = 20,
        max_nodes: int = 10_000,
        max_streams: int = 50_000,
        max_message_size: int = 10 * 1024 * 1024,
        session_idle_timeout: float = 300.0,
        scoring_weights: dict | None = None,
        signal_id: str | None = None,
        require_auth: bool = True,
    ) -> None:
        self.host = host
        self.port = port
        self.http_port = http_port
        self.tls_cert = tls_cert
        self.tls_key = tls_key
        self.cors_origin = cors_origin
        self.max_connections_per_ip = max_connections_per_ip
        self.max_nodes = max_nodes
        self.max_streams = max_streams
        self.max_message_size = max_message_size
        self.session_idle_timeout = session_idle_timeout
        self.scoring_weights = scoring_weights
        self.signal_id = signal_id or uuid.uuid4().hex

        self.require_auth = require_auth

        self.registry = NodeRegistry()
        self.matcher = ConsumerMatcher(self.registry, scorer_weights=scoring_weights)

        self.streams: dict[str, ConsumerStream] = {}
        self.heartbeat_timeout = 120.0
        self.start_time: float = time.time()
        self._sched_lock = asyncio.Lock()
        self._connections_per_ip: dict[str, int] = {}
        self._node_streams: dict[str, set[str]] = {}
        self._rate_limiters_per_ip: dict[str, _RateLimiter] = {}
        self._pending_teardowns: dict[str, asyncio.Task] = {}
        self.teardown_grace_period: float = 15.0

        dashboard_path = Path(__file__).parent / "dashboard.html"
        try:
            self._dashboard_html = dashboard_path.read_bytes()
        except FileNotFoundError:
            self._dashboard_html = b"dashboard not found"

    # ---- helpers -------------------------------------------------------
    def _active_assignments_for_model(
        self, model_name: str,
    ) -> dict[str, tuple[int, int]]:
        out: dict[str, tuple[int, int]] = {}
        for nid, record in self.registry.nodes.items():
            if (
                record.assignment_status == "active"
                and record.assigned_model == model_name
                and record.layer_start is not None
                and record.layer_end is not None
            ):
                out[nid] = (record.layer_start, record.layer_end)
        return out

    async def _safe_send(self, ws: ServerConnection, data: bytes) -> bool:
        try:
            await ws.send(data)
            return True
        except (ConnectionClosed, OSError) as e:
            logger.warning("ws send failed", extra={"err": str(e)})
            return False

    async def _send_error(
        self, ws: ServerConnection, session_id: str, code: str, message: str,
    ) -> None:
        try:
            await ws.send(encode_message(make_error(session_id, code, message)))
        except (ConnectionClosed, OSError):
            pass

    async def _teardown_streams_using_node(self, node_id: str) -> None:
        affected = [sid for sid, ce in self.streams.items() if node_id in ce.pipeline]
        for sid in affected:
            ce = self.streams.pop(sid, None)
            if ce is None:
                continue
            await self._send_error(
                ce.ws, ce.session_id, "NODE_GONE",
                f"Pipeline node {node_id} disconnected",
            )
            try:
                await ce.ws.close()
            except (ConnectionClosed, OSError):
                pass
            logger.info(
                "stream torn down",
                extra={"stream_id": sid, "reason": "node_gone", "node_id": node_id},
            )

    async def _delayed_stream_teardown(self, node_id: str, delay: float) -> None:
        try:
            await asyncio.sleep(delay)
            record = self.registry.get_node(node_id)
            if record is not None and record.assignment_status == "active":
                logger.info(
                    "node reconnected within grace period, skipping teardown",
                    extra={"node_id": node_id},
                )
                return
            await self._teardown_streams_using_node(node_id)
            self._node_streams.pop(node_id, None)
        except asyncio.CancelledError:
            pass
        finally:
            self._pending_teardowns.pop(node_id, None)

    async def _authenticate_node(
        self, ws: ServerConnection, node_id: str, public_key: str | None,
    ) -> bool:
        """Challenge-response authentication. Returns True on success."""
        if not self.require_auth:
            return True
        if not public_key:
            await self._send_error(ws, "", "AUTH_REQUIRED",
                                   "registration requires public_key for authentication")
            return False
        try:
            derived = address_from_public_key(public_key)
        except (ValueError, Exception):
            await self._send_error(ws, "", "AUTH_FAILED", "invalid public key")
            return False
        if derived.lower() != node_id.lower():
            await self._send_error(ws, "", "AUTH_FAILED",
                                   "public key does not match claimed node_id")
            return False
        nonce = secrets.token_hex(32)
        challenge = make_auth_challenge(node_id, nonce)
        if not await self._safe_send(ws, encode_message(challenge)):
            return False
        try:
            raw_auth = await asyncio.wait_for(ws.recv(), timeout=10.0)
        except (asyncio.TimeoutError, ConnectionClosed):
            return False
        if isinstance(raw_auth, str):
            raw_auth = raw_auth.encode("utf-8")
        try:
            auth_msg = decode_message(raw_auth)
        except Exception:
            await self._send_error(ws, "", "AUTH_FAILED", "invalid auth response")
            return False
        if auth_msg.get("type") != AUTH_RESPONSE:
            await self._send_error(ws, "", "AUTH_FAILED", "expected auth_response")
            return False
        signature = auth_msg.get("signature", "")
        if not verify_signature(nonce.encode("utf-8"), signature, public_key):
            await self._send_error(ws, "", "AUTH_FAILED",
                                   "signature verification failed")
            return False
        logger.info("node authenticated", extra={"node_id": node_id})
        return True

    # ---- node connection handler --------------------------------------
    async def _handle_node(self, ws: ServerConnection, first: dict) -> None:
        """Handle a node's persistent outbound connection.

        Accepts either REGISTER_NODE (legacy / relay-compatible path) or
        SIGNAL_REGISTER (M3-native, carries location + supported models).
        Both paths converge on the same NodeRecord in the registry.
        """
        msg_type = first.get("type")
        node_id = first.get("node_id")
        pv = first.get("protocol_version")
        if pv != PROTOCOL_VERSION:
            await self._send_error(
                ws, "", "BAD_PROTOCOL",
                f"Unsupported protocol version {pv!r}; expected {PROTOCOL_VERSION}",
            )
            return
        if not node_id:
            await self._send_error(ws, "", "BAD_HELLO", "missing node_id")
            return
        if not _NODE_ID_RE.match(node_id):
            await self._send_error(ws, "", "BAD_HELLO", "invalid node_id format")
            return
        if self.registry.node_count() >= self.max_nodes:
            await self._send_error(ws, "", "CAPACITY", "node limit reached")
            return

        if not await self._authenticate_node(ws, node_id, first.get("public_key")):
            try:
                await ws.close()
            except (ConnectionClosed, OSError):
                pass
            return

        caps = normalize_capabilities(first.get("capabilities"))
        models_supported = first.get("models_supported") or []

        # Always derive location from IP, never trust self-reported.
        ip = ws.remote_address[0] if ws.remote_address else ""
        location = estimate_location_from_ip(ip)

        # Cancel any pending delayed teardown for this node.
        pending = self._pending_teardowns.pop(node_id, None)
        if pending is not None:
            pending.cancel()
            logger.info(
                "cancelled pending teardown — node reconnected",
                extra={"node_id": node_id},
            )

        # Pre-existing record: preserve assignment state, then close stale socket.
        existing = self.registry.get_node(node_id)
        prev_assignment = None
        if existing is not None:
            if (existing.assignment_status == "active"
                    and existing.assigned_model
                    and existing.layer_start is not None):
                prev_assignment = {
                    "model": existing.assigned_model,
                    "layer_start": existing.layer_start,
                    "layer_end": existing.layer_end,
                }
            try:
                await existing.ws.close()
            except (ConnectionClosed, OSError):
                pass

        record = self.registry.register(
            node_id=node_id,
            ws=ws,
            capabilities=caps,
            location=location,
            models_supported=models_supported,
        )

        layer_start = first.get("layer_start")
        layer_end = first.get("layer_end")
        is_legacy_static = (
            msg_type == REGISTER_NODE
            and layer_start is not None and layer_end is not None
        )

        if is_legacy_static:
            record.layer_start = layer_start
            record.layer_end = layer_end
            record.assignment_status = "active"
            loaded = caps.get("models_loaded") or []
            if loaded:
                record.assigned_model = loaded[0]
            ack = make_register_ack(node_id, True, "ok")
        elif prev_assignment is not None:
            record.assigned_model = prev_assignment["model"]
            record.layer_start = prev_assignment["layer_start"]
            record.layer_end = prev_assignment["layer_end"]
            record.assignment_status = "active"
            if msg_type == SIGNAL_REGISTER:
                ack = make_signal_ack(
                    node_id, True, self.signal_id,
                    "reconnected; restored assignment",
                )
            else:
                ack = make_register_ack(node_id, True, "reconnected; restored assignment")
            logger.info(
                "restored previous assignment on reconnect",
                extra={
                    "node_id": node_id,
                    "model": prev_assignment["model"],
                    "layers": [prev_assignment["layer_start"], prev_assignment["layer_end"]],
                },
            )
        elif msg_type == SIGNAL_REGISTER:
            ack = make_signal_ack(
                node_id, True, self.signal_id,
                "registered; awaiting assignment",
            )
        else:  # dynamic REGISTER_NODE
            ack = make_register_ack(node_id, True, "registered; awaiting assignment")

        if not await self._safe_send(ws, encode_message(ack)):
            self.registry.deregister(node_id)
            return

        logger.info(
            "node registered",
            extra={
                "node_id": node_id,
                "signal_id": self.signal_id,
                "mode": msg_type,
                "location": location,
                "models_supported": models_supported,
                "capabilities": caps,
            },
        )

        if not is_legacy_static and prev_assignment is None:
            asyncio.create_task(self._dynamic_assign(record))

        await self._node_message_loop(ws, record)

    async def _node_message_loop(
        self, ws: ServerConnection, record: NodeRecord,
    ) -> None:
        node_id = record.node_id
        ip = ws.remote_address[0] if ws.remote_address else "unknown"
        limiter = self._rate_limiters_per_ip.setdefault(ip, _RateLimiter())
        try:
            async for raw in ws:
                if not limiter.allow():
                    logger.warning("node rate-limited", extra={"node_id": node_id})
                    continue
                if isinstance(raw, str):
                    raw = raw.encode("utf-8")
                try:
                    msg = decode_message(raw)
                except Exception:
                    logger.exception(
                        "decode failed on node frame",
                        extra={"node_id": node_id},
                    )
                    continue

                t = msg.get("type")
                if t in (HEARTBEAT, SIGNAL_HEARTBEAT):
                    hb_caps = msg.get("capabilities")
                    self.registry.update_heartbeat(
                        node_id,
                        capabilities=(normalize_capabilities(hb_caps)
                                      if isinstance(hb_caps, dict) else None),
                        active_sessions=msg.get("active_sessions"),
                        load=msg.get("load"),
                        status=msg.get("status"),
                    )
                elif t == ENVELOPE:
                    await self._forward_envelope_from_node(record, msg)
                elif t == ASSIGNMENT_ACK:
                    self._handle_assignment_ack(record, msg)
                elif t in (DEREGISTER, SIGNAL_DEREGISTER):
                    logger.info(
                        "node deregistered",
                        extra={"node_id": node_id, "reason": msg.get("reason", "")},
                    )
                    break
                else:
                    logger.warning(
                        "unexpected node frame type",
                        extra={"node_id": node_id, "type": t},
                    )
        except ConnectionClosed:
            pass
        finally:
            current = self.registry.get_node(node_id)
            if current is record:
                # Node did NOT reconnect yet — deregister immediately so it
                # isn't matched to new consumers, but give it a grace period
                # before tearing down existing consumer streams.
                self.registry.deregister(node_id)
                if record.assigned_model:
                    asyncio.create_task(
                        self._rebalance_after_leave(record.assigned_model),
                    )
                self._pending_teardowns[node_id] = asyncio.create_task(
                    self._delayed_stream_teardown(
                        node_id, self.teardown_grace_period,
                    ),
                )
                logger.info(
                    "node connection closed — grace period started",
                    extra={
                        "node_id": node_id,
                        "grace_s": self.teardown_grace_period,
                    },
                )
            else:
                # Node already reconnected with a fresh websocket — do NOT
                # tear down consumer streams or trigger rebalance.  The new
                # connection inherits the assignment and the streams remain
                # valid because _consumer_envelope_loop resolves the target
                # via registry.get_node() on every envelope.
                logger.info(
                    "stale node connection closed (node already reconnected)",
                    extra={"node_id": node_id},
                )

    async def _forward_envelope_from_node(
        self, record: NodeRecord, msg: dict,
    ) -> None:
        sid = msg.get("stream_id")
        ce = self.streams.get(sid)
        if ce is None:
            logger.warning(
                "envelope for unknown stream dropped",
                extra={"stream_id": sid, "from_node": record.node_id},
            )
            return
        if record.node_id not in ce.pipeline:
            logger.warning(
                "node not in stream pipeline, dropping envelope",
                extra={"stream_id": sid, "node_id": record.node_id},
            )
            return
        ce.envelope_counts[record.node_id] = (
            ce.envelope_counts.get(record.node_id, 0) + 1
        )
        ce.last_activity = time.time()
        ok = await self._safe_send(ce.ws, encode_message(msg))
        if not ok:
            self.streams.pop(sid, None)

    # ---- assignment / scheduling --------------------------------------
    def _handle_assignment_ack(self, record: NodeRecord, msg: dict) -> None:
        fut = record.pending_ack
        record.pending_ack = None
        accepted = bool(msg.get("accepted", False))
        log_extra = {
            "node_id": record.node_id,
            "model": msg.get("model_name"),
            "layer_start": msg.get("layer_start"),
            "layer_end": msg.get("layer_end"),
            "accepted": accepted,
            "reason": msg.get("reason", ""),
            "load_time_ms": msg.get("load_time_ms", 0),
        }
        if accepted:
            logger.info("assignment accepted", extra=log_extra)
        else:
            logger.warning("assignment rejected", extra=log_extra)
        if fut is not None and not fut.done():
            fut.set_result(msg)

    async def _send_assignment(
        self,
        record: NodeRecord,
        model_name: str,
        layer_start: int,
        layer_end: int,
        kind: str,
        reason: str = "",
    ) -> dict | None:
        live = self.registry.get_node(record.node_id)
        if live is not record:
            logger.info(
                "skipping assignment to stale record",
                extra={"node_id": record.node_id, "kind": kind},
            )
            return None

        try:
            info = get_model_info(model_name)
        except ValueError:
            logger.error(
                "send_assignment: unknown model",
                extra={"model": model_name},
            )
            return None

        loop = asyncio.get_event_loop()
        fut: asyncio.Future = loop.create_future()
        record.pending_ack = fut
        record.pending_range = (layer_start, layer_end)
        record.assignment_status = (
            "rebalancing" if kind == "rebalance" else "loading"
        )

        if kind == "rebalance":
            frame = make_rebalance(
                record.node_id, model_name, layer_start, layer_end, reason=reason,
            )
        else:
            frame = make_assign_layers(
                record.node_id, model_name, layer_start, layer_end,
                total_layers=info["total_layers"],
                hidden_size=info["hidden_size"],
            )

        if not await self._safe_send(record.ws, encode_message(frame)):
            record.pending_ack = None
            record.pending_range = None
            return None

        try:
            ack = await asyncio.wait_for(fut, timeout=self.ASSIGNMENT_TIMEOUT_S)
        except asyncio.TimeoutError:
            record.pending_ack = None
            record.pending_range = None
            logger.warning(
                "assignment ack timeout",
                extra={"node_id": record.node_id, "kind": kind, "model": model_name},
            )
            return None

        if ack.get("accepted"):
            record.assigned_model = model_name
            record.layer_start = layer_start
            record.layer_end = layer_end
            record.assignment_status = "active"
        else:
            record.pending_range = None
        return ack

    def _pick_model_for_new_node(self, record: NodeRecord) -> str:
        """Prefer a model the node already supports; else follow the network."""
        if record.models_supported:
            for m in record.models_supported:
                if m in MODEL_REGISTRY:
                    return m
        for other in self.registry.nodes.values():
            if other is record:
                continue
            if (other.assignment_status == "active"
                    and other.assigned_model in MODEL_REGISTRY):
                return other.assigned_model
        if MODEL_REGISTRY:
            return next(iter(MODEL_REGISTRY))
        return ""

    async def _dynamic_assign(self, record: NodeRecord) -> None:
        node_id = record.node_id
        async with self._sched_lock:
            live = self.registry.get_node(node_id)
            if live is None or live is not record:
                logger.info(
                    "node reconnected before assignment; skipping stale assign",
                    extra={"node_id": node_id},
                )
                return

            model_name = self._pick_model_for_new_node(record)
            if not model_name:
                logger.warning(
                    "no model in registry; cannot assign",
                    extra={"node_id": node_id},
                )
                return

            current = self._active_assignments_for_model(model_name)
            all_caps = [
                {"node_id": nid, **self.registry.nodes[nid].capabilities}
                for nid in current
                if nid in self.registry.nodes
            ]
            for nid, rec in self.registry.nodes.items():
                if rec is record or nid in current:
                    continue
                if rec.assignment_status in ("loading", "pending"):
                    all_caps.append({"node_id": nid, **rec.capabilities})
            all_caps.append(
                {"node_id": node_id, **record.capabilities},
            )
            try:
                new_assignments, affected = calculate_rebalance(
                    current, all_caps, model_name,
                )
            except ValueError as e:
                logger.warning(
                    "scheduler refused to assign",
                    extra={"node_id": node_id, "err": str(e)},
                )
                return
            await self._apply_assignments(
                model_name, new_assignments, affected, reason="join",
            )

    async def _apply_assignments(
        self,
        model_name: str,
        new_assignments: dict[str, tuple[int, int]],
        affected: list[str],
        reason: str,
    ) -> None:
        tasks = []
        for nid in affected:
            record = self.registry.get_node(nid)
            if record is None:
                continue
            new = new_assignments.get(nid)
            if new is None:
                continue
            ls, le = new
            kind = "assign" if record.assignment_status == "pending" else "rebalance"
            tasks.append(
                self._send_assignment(record, model_name, ls, le, kind, reason=reason),
            )
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _rebalance_after_leave(self, model_name: str) -> None:
        async with self._sched_lock:
            current = self._active_assignments_for_model(model_name)
            if not current:
                return
            try:
                new_assignments = assign_layers(
                    [
                        {"node_id": nid, **self.registry.nodes[nid].capabilities}
                        for nid in current
                        if nid in self.registry.nodes
                    ],
                    model_name,
                )
            except ValueError as e:
                logger.warning(
                    "post-leave rebalance impossible",
                    extra={"model": model_name, "err": str(e)},
                )
                return
            affected = [
                nid for nid, span in new_assignments.items()
                if current.get(nid) != span
            ]
            await self._apply_assignments(
                model_name, new_assignments, affected, reason="leave",
            )

    # ---- consumer handlers --------------------------------------------
    async def _handle_signal_query(
        self, ws: ServerConnection, msg: dict,
    ) -> None:
        """Consumer asked for the best nodes for a given model.

        Signal does NOT open a routing stream for SIGNAL_QUERY; it is a
        pure matchmaker request/response. Consumers that want to route
        inference traffic through the signal must follow up with a
        SESSION_INIT (relay-compatible path).
        """
        raw_sid = msg.get("session_id") or ""
        session_id = raw_sid if re.match(r'^[a-f0-9]{32}$', raw_sid) else uuid.uuid4().hex
        model_name = msg.get("model_name") or ""
        consumer_location = msg.get("consumer_location")
        requirements = msg.get("requirements") or {}
        top_n = min(max(1, int(msg.get("top_n") or 10)), 100)

        # If the consumer didn't send a location, try to geolocate.
        if not consumer_location:
            ip = ws.remote_address[0] if ws.remote_address else ""
            consumer_location = estimate_location_from_ip(ip)

        ranked = self.matcher.find_best_nodes(
            model_name=model_name,
            consumer_location=consumer_location,
            requirements=requirements,
            top_n=top_n,
        )
        nodes_payload = [
            {
                "node_id": n["node_id"],
                "score": round(float(n["score"]), 6),
                "device": (n.get("capabilities") or {}).get("device", "cpu"),
                "gpu_model": (n.get("capabilities") or {}).get("gpu_model", ""),
                "layers": (
                    [n["layer_start"], n["layer_end"]]
                    if n.get("layer_start") is not None
                    and n.get("layer_end") is not None
                    else []
                ),
                "capabilities": n.get("capabilities") or {},
            }
            for n in ranked
        ]
        reply = make_signal_match(session_id, nodes_payload)
        await self._safe_send(ws, encode_message(reply))
        logger.info(
            "signal query answered",
            extra={
                "session_id": session_id,
                "model": model_name,
                "matches": len(nodes_payload),
                "signal_id": self.signal_id,
            },
        )

    async def _handle_session_init(
        self, ws: ServerConnection, first: dict,
    ) -> None:
        """Consumer opened a relay session — assemble a scored pipeline."""
        raw_sid = first.get("session_id") or ""
        session_id = raw_sid if re.match(r'^[a-f0-9]{32}$', raw_sid) else uuid.uuid4().hex
        model_name = first.get("model_name", "")
        pv = first.get("protocol_version")
        if pv != PROTOCOL_VERSION:
            await self._send_error(
                ws, session_id, "BAD_PROTOCOL",
                f"Unsupported protocol version {pv!r}; expected {PROTOCOL_VERSION}",
            )
            return
        if len(self.streams) >= self.max_streams:
            await self._send_error(
                ws, session_id, "CAPACITY", "session limit reached",
            )
            return

        # Geolocate the consumer for scoring purposes.
        consumer_location = first.get("consumer_location")
        if not consumer_location:
            ip = ws.remote_address[0] if ws.remote_address else ""
            consumer_location = estimate_location_from_ip(ip)

        pipeline_nodes = self.matcher.assemble_pipeline(
            model_name=model_name,
            consumer_location=consumer_location,
            requirements=first.get("requirements") or {},
        )
        if not pipeline_nodes:
            await self._send_error(
                ws, session_id, "NO_NODES",
                "no suitable pipeline available",
            )
            return

        # Verify every pipeline node still has a live websocket before
        # committing the consumer to this pipeline.  A node may show as
        # "active" in the registry while its socket is already dead (e.g.
        # nginx dropped the connection between heartbeats).
        for pn in pipeline_nodes:
            nid = pn["node_id"]
            record = self.registry.get_node(nid)
            if record is None or record.assignment_status != "active":
                await self._send_error(
                    ws, session_id, "NO_NODES",
                    f"pipeline node {nid} is no longer available",
                )
                return
            # Probe the websocket — if the underlying transport is closed
            # the state will reflect it even before the next recv() fires.
            try:
                if record.ws.state.name != "OPEN":
                    logger.warning(
                        "pipeline node ws not OPEN, evicting",
                        extra={"node_id": nid, "ws_state": record.ws.state.name},
                    )
                    self.registry.deregister(nid)
                    await self._send_error(
                        ws, session_id, "NO_NODES",
                        f"pipeline node {nid} connection is stale",
                    )
                    return
            except Exception:
                pass  # If we can't check state, proceed optimistically.

            if (time.time() - record.last_heartbeat) > 30.0:
                logger.warning(
                    "pipeline node has stale heartbeat, evicting",
                    extra={"node_id": nid, "age_s": round(time.time() - record.last_heartbeat, 1)},
                )
                self.registry.deregister(nid)
                await self._send_error(
                    ws, session_id, "NO_NODES",
                    f"pipeline node {nid} has stale heartbeat",
                )
                return

        stream_id = uuid.uuid4().hex
        nodes_payload = [
            {
                "node_id": n["node_id"],
                "layer_start": n["layer_start"],
                "layer_end": n["layer_end"],
                "score": round(float(n.get("score", 0.0)), 6),
            }
            for n in pipeline_nodes
        ]
        cfg = make_pipeline_config(session_id, nodes_payload, stream_id=stream_id)
        pipeline_ids = [n["node_id"] for n in pipeline_nodes]
        ce = ConsumerStream(
            stream_id=stream_id,
            ws=ws,
            session_id=session_id,
            pipeline=pipeline_ids,
            model_name=model_name,
        )
        self.streams[stream_id] = ce
        for nid in pipeline_ids:
            self._node_streams.setdefault(nid, set()).add(stream_id)

        if not await self._safe_send(ws, encode_message(cfg)):
            self.streams.pop(stream_id, None)
            for nid in pipeline_ids:
                self._node_streams.get(nid, set()).discard(stream_id)
            return

        logger.info(
            "session opened",
            extra={
                "stream_id": stream_id,
                "session_id": session_id,
                "model": model_name,
                "pipeline": pipeline_ids,
                "signal_id": self.signal_id,
            },
        )

        await self._consumer_envelope_loop(ws, ce, pipeline_ids)

    async def _consumer_envelope_loop(
        self,
        ws: ServerConnection,
        ce: ConsumerStream,
        pipeline_ids: list[str],
    ) -> None:
        ip = ce.ws.remote_address[0] if ce.ws.remote_address else "unknown"
        limiter = self._rate_limiters_per_ip.setdefault(ip, _RateLimiter())
        try:
            async for raw in ws:
                if not limiter.allow():
                    continue
                if isinstance(raw, str):
                    raw = raw.encode("utf-8")
                try:
                    msg = decode_message(raw)
                except Exception:
                    logger.exception(
                        "decode failed on consumer frame",
                        extra={"stream_id": ce.stream_id},
                    )
                    continue

                if msg.get("type") != ENVELOPE:
                    await self._send_error(
                        ws, ce.session_id, "BAD_FRAME", "expected envelope",
                    )
                    continue

                target = msg.get("target_node_id")
                if not target:
                    await self._send_error(
                        ws, ce.session_id, "BAD_TARGET",
                        "envelope missing target_node_id",
                    )
                    continue
                if target not in ce.pipeline:
                    await self._send_error(
                        ws, ce.session_id, "BAD_TARGET",
                        "target node is not in this session's pipeline",
                    )
                    continue

                record = self.registry.get_node(target)
                if record is None or record.assignment_status != "active":
                    await self._send_error(
                        ws, ce.session_id, "NODE_UNAVAILABLE",
                        "target node is not available",
                    )
                    continue

                msg["stream_id"] = ce.stream_id

                count = msg.get("envelope_count", 0)
                last = ce.last_envelope_count.get(target, -1)
                if count <= last:
                    await self._send_error(
                        ws, ce.session_id, "REPLAY",
                        "envelope_count not monotonically increasing",
                    )
                    continue
                ce.last_envelope_count[target] = count

                ce.envelope_counts[target] = ce.envelope_counts.get(target, 0) + 1
                ce.last_activity = time.time()
                ok = await self._safe_send(record.ws, encode_message(msg))
                if not ok:
                    self.registry.deregister(target)
                    await self._send_error(
                        ws, ce.session_id, "NODE_UNAVAILABLE",
                        "send to target node failed",
                    )
        except ConnectionClosed:
            pass
        finally:
            self.streams.pop(ce.stream_id, None)
            for nid in pipeline_ids:
                self._node_streams.get(nid, set()).discard(ce.stream_id)
            logger.info(
                "session closed",
                extra={
                    "stream_id": ce.stream_id,
                    "session_id": ce.session_id,
                    "model": ce.model_name,
                    "envelope_counts": dict(ce.envelope_counts),
                    "duration_s": round(time.time() - ce.created_at, 3),
                },
            )

    # ---- dispatch ------------------------------------------------------
    async def handle_connection(self, ws: ServerConnection) -> None:
        remote = ws.remote_address
        ip = remote[0] if remote else "unknown"

        count = self._connections_per_ip.get(ip, 0)
        if count >= self.max_connections_per_ip:
            await self._send_error(
                ws, "", "RATE_LIMITED",
                "too many connections from this address",
            )
            try:
                await ws.close()
            except (ConnectionClosed, OSError):
                pass
            return
        self._connections_per_ip[ip] = count + 1

        try:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=30.0)
            except (ConnectionClosed, asyncio.TimeoutError):
                return
            if isinstance(raw, str):
                raw = raw.encode("utf-8")
            try:
                msg = decode_message(raw)
            except Exception:
                await self._send_error(
                    ws, "", "BAD_HELLO", "failed to decode first frame",
                )
                try:
                    await ws.close()
                except (ConnectionClosed, OSError):
                    pass
                return

            t = msg.get("type")
            if t in (REGISTER_NODE, SIGNAL_REGISTER):
                await self._handle_node(ws, msg)
            elif t == SIGNAL_QUERY:
                # One-shot matchmaker query; no persistent stream.
                await self._handle_signal_query(ws, msg)
                try:
                    await ws.close()
                except (ConnectionClosed, OSError):
                    pass
            elif t == SESSION_INIT:
                await self._handle_session_init(ws, msg)
            else:
                await self._send_error(
                    ws, msg.get("session_id", ""), "BAD_HELLO",
                    "first frame must be register_node, signal_register, "
                    "signal_query, or session_init",
                )
                try:
                    await ws.close()
                except (ConnectionClosed, OSError):
                    pass
        finally:
            self._connections_per_ip[ip] = max(
                0, self._connections_per_ip.get(ip, 1) - 1,
            )

    # ---- status --------------------------------------------------------
    def build_status(self) -> dict:
        nodes_payload = []
        active_node_count = 0
        for nid, record in self.registry.nodes.items():
            if record.layer_start is not None and record.layer_end is not None:
                layers = [record.layer_start, record.layer_end]
            else:
                layers = []
            if record.assignment_status == "active":
                active_node_count += 1
            redacted_id = nid[:8] + "..." if len(nid) > 8 else nid
            loc = record.location or {}
            caps = record.capabilities or {}
            nodes_payload.append({
                "node_id": redacted_id,
                "device": caps.get("device", "cpu"),
                "layers": layers,
                "status": record.assignment_status,
                "city": loc.get("city", ""),
                "country": loc.get("country", ""),
                "active_sessions": record.active_sessions,
                "ram_mb": caps.get("ram_mb", 0),
                "vram_mb": caps.get("vram_mb", 0),
                "gpu_model": caps.get("gpu_model", ""),
                "cpu_cores": caps.get("cpu_cores", 0),
                "bandwidth_mbps": caps.get("bandwidth_mbps", 0.0),
                "max_context_length": caps.get("max_context_length", 0),
                "load": record.load if record.load is not None else 0.0,
            })

        models_payload = []
        for model_name, info in MODEL_REGISTRY.items():
            total = info["total_layers"]
            assignments = self._active_assignments_for_model(model_name)
            covered = sum(end - start for (start, end) in assignments.values())
            covered = max(0, min(covered, total))
            available = bool(assignments) and validate_coverage(assignments, model_name)
            models_payload.append({
                "name": model_name,
                "total_layers": total,
                "covered_layers": covered,
                "available": available,
            })

        active_records = [r for r in self.registry.nodes.values() if r.assignment_status == "active"]
        all_count = len(self.registry.nodes)
        compute = {
            "total_ram_mb": sum((r.capabilities or {}).get("ram_mb", 0) for r in active_records),
            "total_vram_mb": sum((r.capabilities or {}).get("vram_mb", 0) for r in active_records),
            "total_cpu_cores": sum((r.capabilities or {}).get("cpu_cores", 0) for r in active_records),
            "total_bandwidth_mbps": sum((r.capabilities or {}).get("bandwidth_mbps", 0.0) for r in active_records),
            "active_nodes": len(active_records),
            "total_nodes": all_count,
            "avg_load": round(sum(r.load or 0.0 for r in active_records) / max(1, len(active_records)), 2),
        }

        return {
            "signal_id": self.signal_id,
            "merkle_root": self.registry.merkle_root(),
            "nodes": nodes_payload,
            "models": models_payload,
            "total_nodes": active_node_count,
            "active_sessions": len(self.streams),
            "uptime_seconds": int(time.time() - self.start_time),
            "compute": compute,
        }

    # ---- HTTP ----------------------------------------------------------
    async def _http_respond(
        self,
        writer: asyncio.StreamWriter,
        status_code: int,
        status_text: str,
        body: bytes,
        content_type: str = "text/plain; charset=utf-8",
    ) -> None:
        cors_headers = ""
        if self.cors_origin:
            cors_headers = (
                f"Access-Control-Allow-Origin: {self.cors_origin}\r\n"
                "Access-Control-Allow-Methods: GET, OPTIONS\r\n"
                "Access-Control-Allow-Headers: Content-Type\r\n"
            )
        headers = (
            f"HTTP/1.1 {status_code} {status_text}\r\n"
            f"Content-Type: {content_type}\r\n"
            f"Content-Length: {len(body)}\r\n"
            "Connection: close\r\n"
            f"{cors_headers}"
            "\r\n"
        )
        writer.write(headers.encode("ascii") + body)
        try:
            await writer.drain()
        except (ConnectionResetError, BrokenPipeError):
            pass

    async def _handle_http(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter,
    ) -> None:
        try:
            request_line = await asyncio.wait_for(
                reader.readline(), timeout=10.0,
            )
            if not request_line:
                return
            if len(request_line) > 8192:
                await self._http_respond(
                    writer, 414, "URI Too Long", b"uri too long",
                )
                return
            try:
                method, path, _ = request_line.decode(
                    "ascii", errors="replace",
                ).rstrip("\r\n").split(" ", 2)
            except ValueError:
                await self._http_respond(
                    writer, 400, "Bad Request", b"bad request",
                )
                return
            header_count = 0
            while header_count < 100:
                line = await asyncio.wait_for(reader.readline(), timeout=10.0)
                if not line or line in (b"\r\n", b"\n"):
                    break
                header_count += 1

            if method == "OPTIONS":
                await self._http_respond(writer, 204, "No Content", b"")
                return
            if method != "GET":
                await self._http_respond(
                    writer, 405, "Method Not Allowed", b"method not allowed",
                )
                return

            clean_path = path.split("?", 1)[0]
            if clean_path in ("/", ""):
                await self._http_respond(
                    writer, 200, "OK", self._dashboard_html,
                    content_type="text/html; charset=utf-8",
                )
                return
            if clean_path == "/status":
                body = json.dumps(self.build_status()).encode("utf-8")
                await self._http_respond(
                    writer, 200, "OK", body,
                    content_type="application/json",
                )
                return
            if clean_path == "/health":
                await self._http_respond(
                    writer, 200, "OK", b'{"status":"ok"}',
                    content_type="application/json",
                )
                return
            await self._http_respond(writer, 404, "Not Found", b"not found")
        except Exception:
            logger.exception("http handler error")
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except (ConnectionResetError, BrokenPipeError, OSError):
                pass

    # ---- monitors ------------------------------------------------------
    async def monitor_heartbeats(self) -> None:
        while True:
            now = time.time()
            stale_records = [
                (nid, record) for nid, record in self.registry.nodes.items()
                if (now - record.last_heartbeat) > self.heartbeat_timeout
            ]
            for nid, record in stale_records:
                logger.warning(
                    "node stale — no heartbeat",
                    extra={"node_id": nid, "timeout_s": self.heartbeat_timeout},
                )
                self.registry.deregister(nid)
                try:
                    await record.ws.close()
                except (ConnectionClosed, OSError):
                    pass
                if nid not in self._pending_teardowns:
                    self._pending_teardowns[nid] = asyncio.create_task(
                        self._delayed_stream_teardown(
                            nid, self.teardown_grace_period,
                        ),
                    )
            await asyncio.sleep(5.0)

    async def monitor_idle_sessions(self) -> None:
        while True:
            now = time.time()
            for sid, ce in list(self.streams.items()):
                idle = now - ce.last_activity
                if idle > self.session_idle_timeout:
                    logger.info(
                        "closing idle session",
                        extra={"stream_id": sid, "idle_s": round(idle, 1)},
                    )
                    self.streams.pop(sid, None)
                    try:
                        await ce.ws.close()
                    except (ConnectionClosed, OSError):
                        pass
            await asyncio.sleep(30.0)

    async def start(self) -> None:
        asyncio.create_task(self.monitor_heartbeats())
        asyncio.create_task(self.monitor_idle_sessions())
        logger.info(
            "signal starting",
            extra={
                "host": self.host,
                "port": self.port,
                "http_port": self.http_port,
                "signal_id": self.signal_id,
            },
        )

        ssl_ctx = None
        if self.tls_cert and self.tls_key:
            ssl_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            ssl_ctx.minimum_version = ssl.TLSVersion.TLSv1_2
            ssl_ctx.load_cert_chain(self.tls_cert, self.tls_key)
            logger.info("TLS enabled for WebSocket and HTTP")

        http_server = await asyncio.start_server(
            self._handle_http, self.host, self.http_port, ssl=ssl_ctx,
        )
        scheme = "https" if ssl_ctx else "http"
        logger.info(
            "status http listening",
            extra={
                "host": self.host,
                "http_port": self.http_port,
                "url": f"{scheme}://{self.host}:{self.http_port}/status",
            },
        )
        async with serve(
            self.handle_connection,
            self.host,
            self.port,
            max_size=self.max_message_size,
            ssl=ssl_ctx,
            ping_interval=20,
            ping_timeout=20,
        ), http_server:
            ws_scheme = "wss" if ssl_ctx else "ws"
            logger.info(
                "signal listening",
                extra={"host": self.host, "port": self.port, "scheme": ws_scheme},
            )
            await asyncio.Future()


def configure_logging(level: str) -> None:
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(getattr(logging, level))


async def main() -> None:
    parser = argparse.ArgumentParser(description="Groove Signal Server")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="Bind address — signal is public-facing, 0.0.0.0 is intentional")
    parser.add_argument("--port", type=int, default=8770)
    parser.add_argument("--http-port", type=int, default=8771,
                        help="Port for HTTP /status and /health endpoints")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--cert", default=None, help="Path to TLS certificate")
    parser.add_argument("--key", default=None, help="Path to TLS private key")
    parser.add_argument("--cors-origin", default="*")
    parser.add_argument("--max-connections-per-ip", type=int, default=20)
    parser.add_argument("--max-message-size", type=int, default=10 * 1024 * 1024)
    parser.add_argument(
        "--scoring-weights", default=None,
        help='JSON string with scoring weight overrides, '
             'e.g. \'{"proximity":0.5,"uptime":0.2,"compute":0.2,"load":0.1}\'',
    )
    parser.add_argument(
        "--geoip-db", default=None,
        help="Path to MaxMind GeoLite2-City .mmdb for IP geolocation",
    )
    parser.add_argument(
        "--signal-id", default=None,
        help="Stable identifier for this signal operator (auto-generated if omitted)",
    )
    parser.add_argument(
        "--no-auth", action="store_true",
        help="Disable challenge-response node authentication (dev only)",
    )
    args = parser.parse_args()
    configure_logging(args.log_level)

    weights = None
    if args.scoring_weights:
        try:
            weights = json.loads(args.scoring_weights)
        except json.JSONDecodeError as e:
            logger.error(
                "invalid --scoring-weights JSON",
                extra={"err": str(e)},
            )
            sys.exit(2)

    configure_geoip(args.geoip_db)

    signal = SignalServer(
        host=args.host,
        port=args.port,
        http_port=args.http_port,
        tls_cert=args.cert,
        tls_key=args.key,
        cors_origin=args.cors_origin,
        max_connections_per_ip=args.max_connections_per_ip,
        max_message_size=args.max_message_size,
        scoring_weights=weights,
        signal_id=args.signal_id,
        require_auth=not args.no_auth,
    )
    await signal.start()


if __name__ == "__main__":
    asyncio.run(main())

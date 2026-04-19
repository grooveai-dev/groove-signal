"""In-memory registry of connected nodes.

The signal service is the only public-facing Groove infrastructure, so
its node registry is the source of truth for "who is online right now".
Every query, heartbeat, assignment, and disconnect mutates this structure.

The registry is also cryptographically committable: each snapshot has a
Merkle root over the sorted set of node_ids, letting external observers
verify membership without trusting the signal operator (see M3 eclipse-
attack defense in ROADMAP.md).
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class NodeRecord:
    node_id: str
    ws: Any  # websockets ServerConnection, but we keep the type loose
    capabilities: dict
    location: dict | None
    models_supported: list[str]
    layer_start: int | None = None
    layer_end: int | None = None
    # 'pending' | 'loading' | 'active' | 'rebalancing'
    assignment_status: str = "pending"
    assigned_model: str = ""
    active_sessions: int = 0
    load: float = 0.0
    last_heartbeat: float = field(default_factory=time.time)
    registered_at: float = field(default_factory=time.time)
    # Disconnect count over a rolling 7-day window. Used by scoring.
    downtime_events: int = 0
    # Set while awaiting an ASSIGNMENT_ACK from the node.
    pending_ack: asyncio.Future | None = field(default=None, repr=False)
    pending_range: tuple[int, int] | None = None

    def to_public_dict(self) -> dict:
        """Redacted view suitable for scoring / public status.

        Deliberately excludes the websocket reference and the full wallet
        address (node_id is truncated to match relay.py's build_status
        convention).
        """
        return {
            "node_id": self.node_id,
            "capabilities": dict(self.capabilities),
            "location": dict(self.location) if self.location else None,
            "models_supported": list(self.models_supported),
            "layer_start": self.layer_start,
            "layer_end": self.layer_end,
            "assignment_status": self.assignment_status,
            "assigned_model": self.assigned_model,
            "active_sessions": self.active_sessions,
            "load": self.load,
            "downtime_events": self.downtime_events,
        }


class NodeRegistry:
    """Thread-safe-ish registry (single-event-loop asyncio model).

    All mutations happen on the signal server's event loop, so no locking
    is needed. If the registry is ever accessed from multiple loops or
    threads we'll need to revisit — but that's a pattern change, not a
    one-line fix, so we leave it explicit.
    """

    def __init__(self) -> None:
        self.nodes: dict[str, NodeRecord] = {}
        self._downtime_history: dict[str, int] = {}

    # ---- lifecycle -----------------------------------------------------
    def register(
        self,
        node_id: str,
        ws,
        capabilities: dict,
        location: dict | None,
        models_supported: list[str] | None = None,
    ) -> NodeRecord:
        """Add (or replace) a node record.

        If a record for this node_id already exists, it is replaced. The
        previous record's downtime counter is preserved — a flapping node
        should not be able to reset its score by reconnecting.
        """
        prev = self.nodes.get(node_id)
        downtime = (
            prev.downtime_events if prev is not None
            else self._downtime_history.get(node_id, 0)
        )
        record = NodeRecord(
            node_id=node_id,
            ws=ws,
            capabilities=dict(capabilities or {}),
            location=dict(location) if location else None,
            models_supported=list(models_supported or []),
            downtime_events=downtime,
        )
        self.nodes[node_id] = record
        return record

    def deregister(self, node_id: str) -> NodeRecord | None:
        """Remove a node and increment its disconnect counter."""
        record = self.nodes.pop(node_id, None)
        if record is not None:
            record.downtime_events += 1
            self._downtime_history[node_id] = record.downtime_events
        return record

    # ---- heartbeats ----------------------------------------------------
    def update_heartbeat(
        self,
        node_id: str,
        capabilities: dict | None = None,
        active_sessions: int | None = None,
        load: float | None = None,
        status: str | None = None,
    ) -> NodeRecord | None:
        record = self.nodes.get(node_id)
        if record is None:
            return None
        record.last_heartbeat = time.time()
        if capabilities is not None:
            record.capabilities = dict(capabilities)
        if active_sessions is not None:
            record.active_sessions = int(active_sessions)
        if load is not None:
            record.load = float(load)
        # status is accepted but not currently surfaced — future hook for
        # 'draining', 'maintenance', etc.
        _ = status
        return record

    # ---- queries -------------------------------------------------------
    def get_node(self, node_id: str) -> NodeRecord | None:
        return self.nodes.get(node_id)

    def get_active_nodes(self, model_name: str | None = None) -> list[dict]:
        """Return public-dict view of nodes eligible for consumer matching.

        A node is "active" when its assignment_status is 'active' OR it
        has declared support for the requested model but hasn't been
        assigned layers yet (the signal's scheduler will handle that).
        """
        out: list[dict] = []
        for record in self.nodes.values():
            if record.assignment_status not in ("active", "pending"):
                continue
            if model_name:
                models_ok = (
                    model_name in record.models_supported
                    or record.assigned_model == model_name
                    or not record.models_supported  # empty = any
                )
                if not models_ok:
                    continue
            out.append(record.to_public_dict())
        return out

    def node_count(self) -> int:
        return len(self.nodes)

    # ---- stale cleanup -------------------------------------------------
    def cleanup_stale(self, timeout_seconds: float = 120.0) -> list[str]:
        """Remove nodes that haven't heartbeat in `timeout_seconds`.

        Returns the list of removed node_ids so the caller can close
        websockets / fire rebalancing.
        """
        now = time.time()
        stale = [
            nid for nid, r in self.nodes.items()
            if (now - r.last_heartbeat) > timeout_seconds
        ]
        for nid in stale:
            self.deregister(nid)
        return stale

    # ---- cryptographic commitment --------------------------------------
    def merkle_root(self) -> str:
        """SHA256 Merkle root over sorted node_ids.

        Used by the signal service to publish cryptographic commitments to
        its node set (see M3 eclipse-attack defense). The Merkle tree is
        binary; odd leaves at each level are promoted unchanged to the
        next level. Empty registry returns the SHA256 of the empty byte
        string for a well-defined baseline.
        """
        ids = sorted(self.nodes.keys())
        if not ids:
            return hashlib.sha256(b"").hexdigest()
        leaves = [hashlib.sha256(nid.encode("utf-8")).digest() for nid in ids]
        while len(leaves) > 1:
            nxt: list[bytes] = []
            for i in range(0, len(leaves), 2):
                if i + 1 < len(leaves):
                    combined = leaves[i] + leaves[i + 1]
                else:
                    combined = leaves[i]
                nxt.append(hashlib.sha256(combined).digest())
            leaves = nxt
        return leaves[0].hex()

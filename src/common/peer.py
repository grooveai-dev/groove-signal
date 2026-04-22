"""WebRTC peer connection manager for direct P2P data channels.

Uses aiortc to establish encrypted data channels between pipeline peers,
bypassing the signal relay for inference data transfer.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from typing import Awaitable, Callable

from aiortc import (
    RTCConfiguration,
    RTCIceCandidate,
    RTCIceServer,
    RTCPeerConnection,
    RTCSessionDescription,
)

logger = logging.getLogger("peer")


def _default_rtc_config() -> RTCConfiguration:
    """STUN/TURN configuration for NAT traversal."""
    return RTCConfiguration(iceServers=[
        RTCIceServer(urls=["stun:stun.l.google.com:19302"]),
        RTCIceServer(urls=["stun:stun1.l.google.com:19302"]),
        RTCIceServer(
            urls=["turn:turn.groovedev.ai:3478"],
            username="groove",
            credential="<rotated-secret>",
        ),
    ])


@dataclass
class _PeerState:
    pc: RTCPeerConnection
    channel: object | None = None
    recv_queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    connected: bool = False


class PeerConnectionManager:
    """Manages WebRTC data channels between pipeline peers."""

    def __init__(
        self,
        session_id: str | None = None,
        rtc_config: RTCConfiguration | None = None,
    ):
        self._session_id = session_id or uuid.uuid4().hex
        self._config = rtc_config or _default_rtc_config()
        self._peers: dict[str, _PeerState] = {}
        self.on_ice_candidate: Callable[[str, dict], Awaitable[None]] | None = None

    def _setup_channel_events(self, remote_node_id: str, channel) -> None:
        state = self._peers[remote_node_id]

        @channel.on("open")
        def on_open():
            state.connected = True
            logger.info("data channel open", extra={"peer": remote_node_id})

        @channel.on("close")
        def on_close():
            state.connected = False
            logger.info("data channel closed", extra={"peer": remote_node_id})

        @channel.on("message")
        def on_message(data):
            state.recv_queue.put_nowait(data)

        if getattr(channel, "readyState", None) == "open":
            state.connected = True

    async def create_offer(self, remote_node_id: str) -> str:
        """Create RTCPeerConnection + data channel, return SDP offer."""
        pc = RTCPeerConnection(configuration=self._config)
        state = _PeerState(pc=pc)
        self._peers[remote_node_id] = state

        channel = pc.createDataChannel(
            f"groove-{self._session_id}",
            ordered=True,
            maxRetransmits=None,
        )
        state.channel = channel
        self._setup_channel_events(remote_node_id, channel)

        @pc.on("connectionstatechange")
        async def on_state():
            if pc.connectionState in ("failed", "closed"):
                state.connected = False

        offer = await pc.createOffer()
        await pc.setLocalDescription(offer)
        return pc.localDescription.sdp

    async def handle_offer(self, remote_node_id: str, sdp_offer: str) -> str:
        """Receive SDP offer, create answer, return SDP answer string."""
        pc = RTCPeerConnection(configuration=self._config)
        state = _PeerState(pc=pc)
        self._peers[remote_node_id] = state

        @pc.on("datachannel")
        def on_datachannel(channel):
            state.channel = channel
            self._setup_channel_events(remote_node_id, channel)

        @pc.on("connectionstatechange")
        async def on_state():
            if pc.connectionState in ("failed", "closed"):
                state.connected = False

        offer = RTCSessionDescription(sdp=sdp_offer, type="offer")
        await pc.setRemoteDescription(offer)
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        return pc.localDescription.sdp

    async def accept_answer(self, remote_node_id: str, sdp_answer: str) -> None:
        """Apply remote SDP answer to complete the handshake."""
        state = self._peers.get(remote_node_id)
        if state is None:
            raise ValueError(f"No pending connection for {remote_node_id}")
        answer = RTCSessionDescription(sdp=sdp_answer, type="answer")
        await state.pc.setRemoteDescription(answer)

    async def add_ice_candidate(self, remote_node_id: str, candidate: dict) -> None:
        """Add a trickled ICE candidate to the connection."""
        state = self._peers.get(remote_node_id)
        if state is None:
            raise ValueError(f"No connection for {remote_node_id}")
        candidate_str = candidate.get("candidate", "")
        if not candidate_str:
            return
        sdp_mid = candidate.get("sdpMid")
        sdp_m_line_index = candidate.get("sdpMLineIndex")
        ice = _parse_ice_candidate(candidate_str, sdp_mid, sdp_m_line_index)
        await state.pc.addIceCandidate(ice)

    async def send(self, remote_node_id: str, data: bytes) -> None:
        """Send binary data over the data channel."""
        state = self._peers.get(remote_node_id)
        if state is None or state.channel is None:
            raise ValueError(f"No data channel for {remote_node_id}")
        state.channel.send(data)

    async def recv(self, remote_node_id: str) -> bytes:
        """Receive binary data from a peer's data channel."""
        state = self._peers.get(remote_node_id)
        if state is None:
            raise ValueError(f"No connection for {remote_node_id}")
        return await state.recv_queue.get()

    def is_connected(self, remote_node_id: str) -> bool:
        """Check if the data channel is open."""
        state = self._peers.get(remote_node_id)
        if state is None:
            return False
        return state.connected

    async def close(self, remote_node_id: str) -> None:
        """Close a specific peer connection and clean up."""
        state = self._peers.pop(remote_node_id, None)
        if state is None:
            return
        state.connected = False
        await state.pc.close()

    async def close_all(self) -> None:
        """Close all peer connections."""
        for peer_id in list(self._peers.keys()):
            await self.close(peer_id)


def _parse_ice_candidate(
    candidate_str: str,
    sdp_mid: str | None,
    sdp_m_line_index: int | None,
) -> RTCIceCandidate:
    """Parse an ICE candidate SDP string into an RTCIceCandidate."""
    parts = candidate_str.split()
    foundation = parts[0].split(":", 1)[-1] if ":" in parts[0] else parts[0]
    component = int(parts[1])
    protocol = parts[2]
    priority = int(parts[3])
    ip = parts[4]
    port = int(parts[5])
    cand_type = parts[7]
    related_address = None
    related_port = None
    tcp_type = None
    i = 8
    while i < len(parts) - 1:
        if parts[i] == "raddr":
            related_address = parts[i + 1]
            i += 2
        elif parts[i] == "rport":
            related_port = int(parts[i + 1])
            i += 2
        elif parts[i] == "tcptype":
            tcp_type = parts[i + 1]
            i += 2
        else:
            i += 1
    return RTCIceCandidate(
        component=component,
        foundation=foundation,
        ip=ip,
        port=port,
        priority=priority,
        protocol=protocol,
        type=cand_type,
        relatedAddress=related_address,
        relatedPort=related_port,
        sdpMid=sdp_mid,
        sdpMLineIndex=sdp_m_line_index,
        tcpType=tcp_type,
    )

"""Tests for PeerConnectionManager WebRTC data channel management."""

import asyncio

import pytest

from aiortc import RTCConfiguration

from src.common.peer import PeerConnectionManager, _parse_ice_candidate


@pytest.fixture
def rtc_config():
    """Empty config for local loopback testing (no STUN/TURN needed)."""
    return RTCConfiguration(iceServers=[])


@pytest.mark.asyncio
async def test_create_offer(rtc_config):
    mgr = PeerConnectionManager(session_id="test-session", rtc_config=rtc_config)
    try:
        sdp = await mgr.create_offer("node-b")
        assert isinstance(sdp, str)
        assert "v=0" in sdp
    finally:
        await mgr.close_all()


@pytest.mark.asyncio
async def test_offer_answer_exchange(rtc_config):
    mgr_a = PeerConnectionManager(session_id="test-session", rtc_config=rtc_config)
    mgr_b = PeerConnectionManager(session_id="test-session", rtc_config=rtc_config)
    try:
        offer_sdp = await mgr_a.create_offer("node-b")
        answer_sdp = await mgr_b.handle_offer("node-a", offer_sdp)
        assert isinstance(answer_sdp, str)
        assert "v=0" in answer_sdp
        await mgr_a.accept_answer("node-b", answer_sdp)
    finally:
        await mgr_a.close_all()
        await mgr_b.close_all()


@pytest.mark.asyncio
async def test_send_recv(rtc_config):
    mgr_a = PeerConnectionManager(session_id="test-session", rtc_config=rtc_config)
    mgr_b = PeerConnectionManager(session_id="test-session", rtc_config=rtc_config)
    try:
        offer_sdp = await mgr_a.create_offer("node-b")
        answer_sdp = await mgr_b.handle_offer("node-a", offer_sdp)
        await mgr_a.accept_answer("node-b", answer_sdp)

        for _ in range(50):
            if mgr_a.is_connected("node-b") and mgr_b.is_connected("node-a"):
                break
            await asyncio.sleep(0.1)

        assert mgr_a.is_connected("node-b")
        assert mgr_b.is_connected("node-a")

        payload = b"hello from node-a"
        await mgr_a.send("node-b", payload)
        received = await asyncio.wait_for(mgr_b.recv("node-a"), timeout=5.0)
        assert received == payload

        payload_back = b"hello from node-b"
        await mgr_b.send("node-a", payload_back)
        received_back = await asyncio.wait_for(mgr_a.recv("node-b"), timeout=5.0)
        assert received_back == payload_back
    finally:
        await mgr_a.close_all()
        await mgr_b.close_all()


@pytest.mark.asyncio
async def test_is_connected_initially_false(rtc_config):
    mgr = PeerConnectionManager(session_id="test-session", rtc_config=rtc_config)
    assert mgr.is_connected("nonexistent") is False
    await mgr.close_all()


@pytest.mark.asyncio
async def test_is_connected_transitions(rtc_config):
    mgr_a = PeerConnectionManager(session_id="test-session", rtc_config=rtc_config)
    mgr_b = PeerConnectionManager(session_id="test-session", rtc_config=rtc_config)
    try:
        assert mgr_a.is_connected("node-b") is False

        offer_sdp = await mgr_a.create_offer("node-b")
        assert mgr_a.is_connected("node-b") is False

        answer_sdp = await mgr_b.handle_offer("node-a", offer_sdp)
        await mgr_a.accept_answer("node-b", answer_sdp)

        for _ in range(50):
            if mgr_a.is_connected("node-b"):
                break
            await asyncio.sleep(0.1)

        assert mgr_a.is_connected("node-b") is True

        await mgr_a.close("node-b")
        assert mgr_a.is_connected("node-b") is False
    finally:
        await mgr_a.close_all()
        await mgr_b.close_all()


@pytest.mark.asyncio
async def test_close_cleanup(rtc_config):
    mgr_a = PeerConnectionManager(session_id="test-session", rtc_config=rtc_config)
    mgr_b = PeerConnectionManager(session_id="test-session", rtc_config=rtc_config)
    try:
        offer_sdp = await mgr_a.create_offer("node-b")
        answer_sdp = await mgr_b.handle_offer("node-a", offer_sdp)
        await mgr_a.accept_answer("node-b", answer_sdp)

        for _ in range(50):
            if mgr_a.is_connected("node-b"):
                break
            await asyncio.sleep(0.1)

        await mgr_a.close("node-b")
        assert mgr_a.is_connected("node-b") is False
        assert "node-b" not in mgr_a._peers
    finally:
        await mgr_a.close_all()
        await mgr_b.close_all()


@pytest.mark.asyncio
async def test_close_all(rtc_config):
    mgr = PeerConnectionManager(session_id="test-session", rtc_config=rtc_config)
    try:
        await mgr.create_offer("node-a")
        await mgr.create_offer("node-b")
        assert len(mgr._peers) == 2
        await mgr.close_all()
        assert len(mgr._peers) == 0
    except Exception:
        await mgr.close_all()
        raise


@pytest.mark.asyncio
async def test_accept_answer_unknown_peer(rtc_config):
    mgr = PeerConnectionManager(session_id="test-session", rtc_config=rtc_config)
    try:
        with pytest.raises(ValueError, match="No pending connection"):
            await mgr.accept_answer("unknown", "fake-sdp")
    finally:
        await mgr.close_all()


@pytest.mark.asyncio
async def test_send_no_channel(rtc_config):
    mgr = PeerConnectionManager(session_id="test-session", rtc_config=rtc_config)
    with pytest.raises(ValueError, match="No data channel"):
        await mgr.send("unknown", b"data")


@pytest.mark.asyncio
async def test_recv_no_connection(rtc_config):
    mgr = PeerConnectionManager(session_id="test-session", rtc_config=rtc_config)
    with pytest.raises(ValueError, match="No connection"):
        await mgr.recv("unknown")


@pytest.mark.asyncio
async def test_ice_candidate_no_connection(rtc_config):
    mgr = PeerConnectionManager(session_id="test-session", rtc_config=rtc_config)
    with pytest.raises(ValueError, match="No connection"):
        await mgr.add_ice_candidate("unknown", {"candidate": "test"})


@pytest.mark.asyncio
async def test_add_ice_candidate_empty_string(rtc_config):
    mgr = PeerConnectionManager(session_id="test-session", rtc_config=rtc_config)
    try:
        await mgr.create_offer("node-b")
        await mgr.add_ice_candidate("node-b", {"candidate": ""})
    finally:
        await mgr.close_all()


def test_parse_ice_candidate_host():
    ice = _parse_ice_candidate(
        "candidate:842163049 1 udp 1677729535 192.168.1.100 56789 typ host",
        "0", 0,
    )
    assert ice.foundation == "842163049"
    assert ice.component == 1
    assert ice.protocol == "udp"
    assert ice.priority == 1677729535
    assert ice.ip == "192.168.1.100"
    assert ice.port == 56789
    assert ice.type == "host"
    assert ice.sdpMid == "0"
    assert ice.sdpMLineIndex == 0


def test_parse_ice_candidate_srflx():
    ice = _parse_ice_candidate(
        "candidate:1 1 udp 1694498815 203.0.113.5 45678 typ srflx raddr 10.0.0.1 rport 12345",
        "audio", 0,
    )
    assert ice.type == "srflx"
    assert ice.relatedAddress == "10.0.0.1"
    assert ice.relatedPort == 12345

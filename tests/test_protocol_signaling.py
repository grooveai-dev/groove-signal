"""Tests for WebRTC signaling protocol message types."""

import pytest

from src.common.protocol import (
    ALL_MESSAGE_TYPES,
    ICE_CANDIDATE,
    P2P_READY,
    SDP_ANSWER,
    SDP_OFFER,
    decode_message,
    encode_message,
    make_ice_candidate,
    make_p2p_ready,
    make_sdp_answer,
    make_sdp_offer,
)


def test_signaling_types_in_all():
    assert SDP_OFFER in ALL_MESSAGE_TYPES
    assert SDP_ANSWER in ALL_MESSAGE_TYPES
    assert ICE_CANDIDATE in ALL_MESSAGE_TYPES
    assert P2P_READY in ALL_MESSAGE_TYPES


def test_make_sdp_offer():
    msg = make_sdp_offer("sess-1", "0xaaa", "0xbbb", "v=0\r\n...")
    assert msg["type"] == SDP_OFFER
    assert msg["session_id"] == "sess-1"
    assert msg["from_node_id"] == "0xaaa"
    assert msg["to_node_id"] == "0xbbb"
    assert msg["sdp"] == "v=0\r\n..."


def test_make_sdp_answer():
    msg = make_sdp_answer("sess-1", "0xbbb", "0xaaa", "v=0\r\nanswer")
    assert msg["type"] == SDP_ANSWER
    assert msg["session_id"] == "sess-1"
    assert msg["from_node_id"] == "0xbbb"
    assert msg["to_node_id"] == "0xaaa"
    assert msg["sdp"] == "v=0\r\nanswer"


def test_make_ice_candidate():
    msg = make_ice_candidate(
        "sess-1", "0xaaa", "0xbbb",
        "candidate:842163049 1 udp 1677729535 192.168.1.1 56789 typ srflx",
        "0", 0,
    )
    assert msg["type"] == ICE_CANDIDATE
    assert msg["session_id"] == "sess-1"
    assert msg["from_node_id"] == "0xaaa"
    assert msg["to_node_id"] == "0xbbb"
    assert "candidate:" in msg["candidate"]
    assert msg["sdp_mid"] == "0"
    assert msg["sdp_m_line_index"] == 0


def test_make_p2p_ready():
    msg = make_p2p_ready("sess-1", "0xaaa", "0xbbb")
    assert msg["type"] == P2P_READY
    assert msg["session_id"] == "sess-1"
    assert msg["node_id"] == "0xaaa"
    assert msg["peer_node_id"] == "0xbbb"


def test_sdp_offer_roundtrip():
    msg = make_sdp_offer("sess-1", "0xaaa", "0xbbb", "v=0\r\noffer-sdp")
    encoded = encode_message(msg)
    decoded = decode_message(encoded)
    assert decoded["type"] == SDP_OFFER
    assert decoded["session_id"] == "sess-1"
    assert decoded["from_node_id"] == "0xaaa"
    assert decoded["to_node_id"] == "0xbbb"
    assert decoded["sdp"] == "v=0\r\noffer-sdp"


def test_sdp_answer_roundtrip():
    msg = make_sdp_answer("sess-1", "0xbbb", "0xaaa", "v=0\r\nanswer-sdp")
    encoded = encode_message(msg)
    decoded = decode_message(encoded)
    assert decoded["type"] == SDP_ANSWER
    assert decoded["sdp"] == "v=0\r\nanswer-sdp"


def test_ice_candidate_roundtrip():
    msg = make_ice_candidate(
        "sess-1", "0xaaa", "0xbbb",
        "candidate:1 1 udp 2130706431 10.0.0.1 12345 typ host",
        "audio", 0,
    )
    encoded = encode_message(msg)
    decoded = decode_message(encoded)
    assert decoded["type"] == ICE_CANDIDATE
    assert decoded["candidate"] == "candidate:1 1 udp 2130706431 10.0.0.1 12345 typ host"
    assert decoded["sdp_mid"] == "audio"
    assert decoded["sdp_m_line_index"] == 0


def test_p2p_ready_roundtrip():
    msg = make_p2p_ready("sess-1", "0xaaa", "0xbbb")
    encoded = encode_message(msg)
    decoded = decode_message(encoded)
    assert decoded["type"] == P2P_READY
    assert decoded["node_id"] == "0xaaa"
    assert decoded["peer_node_id"] == "0xbbb"


def test_sdp_offer_field_types():
    msg = make_sdp_offer("sess-1", "0xaaa", "0xbbb", "sdp-data")
    assert isinstance(msg["type"], str)
    assert isinstance(msg["session_id"], str)
    assert isinstance(msg["from_node_id"], str)
    assert isinstance(msg["to_node_id"], str)
    assert isinstance(msg["sdp"], str)


def test_ice_candidate_field_types():
    msg = make_ice_candidate("sess-1", "0xaaa", "0xbbb", "cand", "mid", 1)
    assert isinstance(msg["type"], str)
    assert isinstance(msg["session_id"], str)
    assert isinstance(msg["from_node_id"], str)
    assert isinstance(msg["to_node_id"], str)
    assert isinstance(msg["candidate"], str)
    assert isinstance(msg["sdp_mid"], str)
    assert isinstance(msg["sdp_m_line_index"], int)


def test_p2p_ready_field_types():
    msg = make_p2p_ready("sess-1", "0xaaa", "0xbbb")
    assert isinstance(msg["type"], str)
    assert isinstance(msg["session_id"], str)
    assert isinstance(msg["node_id"], str)
    assert isinstance(msg["peer_node_id"], str)

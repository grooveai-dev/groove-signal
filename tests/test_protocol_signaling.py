"""Tests for WebRTC signaling protocol message types and TURN credentials."""

import base64
import hashlib
import hmac
import time

import pytest

from src.common.protocol import (
    ALL_MESSAGE_TYPES,
    ICE_CANDIDATE,
    P2P_READY,
    PIPELINE_CONFIG,
    SDP_ANSWER,
    SDP_OFFER,
    decode_message,
    encode_message,
    make_ice_candidate,
    make_p2p_ready,
    make_pipeline_config,
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


def test_pipeline_config_without_turn():
    nodes = [{"node_id": "a", "layer_start": 0, "layer_end": 5}]
    msg = make_pipeline_config("sess-1", nodes, stream_id="stream-1")
    assert msg["type"] == PIPELINE_CONFIG
    assert msg["session_id"] == "sess-1"
    assert msg["stream_id"] == "stream-1"
    assert "turn_servers" not in msg


def test_pipeline_config_with_turn():
    nodes = [{"node_id": "a", "layer_start": 0, "layer_end": 5}]
    turn = [{"urls": ["turn:turn.example.com:3478"], "username": "u", "credential": "p"}]
    msg = make_pipeline_config("sess-1", nodes, stream_id="stream-1", turn_servers=turn)
    assert msg["type"] == PIPELINE_CONFIG
    assert msg["turn_servers"] == turn
    assert len(msg["turn_servers"]) == 1
    assert msg["turn_servers"][0]["urls"] == ["turn:turn.example.com:3478"]


def test_pipeline_config_with_turn_roundtrip():
    nodes = [{"node_id": "a", "layer_start": 0, "layer_end": 5}]
    turn = [{"urls": ["turn:turn.example.com:3478"], "username": "u", "credential": "p"}]
    msg = make_pipeline_config("sess-1", nodes, turn_servers=turn)
    encoded = encode_message(msg)
    decoded = decode_message(encoded)
    assert decoded["turn_servers"] == turn


# ---------------------------------------------------------------------------
# TURN credential generation tests (coturn HMAC shared-secret model).
# ---------------------------------------------------------------------------

def _generate_turn_credentials(secret: str, username: str, ttl: int = 86400) -> dict:
    """Standalone reimplementation for testing — mirrors RelayNode._generate_turn_credentials."""
    expiry = int(time.time()) + ttl
    turn_username = f"{expiry}:{username}"
    turn_password = base64.b64encode(
        hmac.new(secret.encode(), turn_username.encode(), hashlib.sha1).digest()
    ).decode()
    return {
        "urls": ["turn:turn.groovedev.ai:3478"],
        "username": turn_username,
        "credential": turn_password,
    }


def test_turn_credential_format():
    creds = _generate_turn_credentials("test-secret", "node-a")
    assert creds["urls"] == ["turn:turn.groovedev.ai:3478"]
    parts = creds["username"].split(":", 1)
    assert len(parts) == 2
    expiry = int(parts[0])
    assert expiry > time.time()
    assert parts[1] == "node-a"


def test_turn_credential_expiry_in_future():
    creds = _generate_turn_credentials("test-secret", "node-a", ttl=3600)
    expiry = int(creds["username"].split(":")[0])
    assert expiry > time.time()
    assert expiry <= time.time() + 3601


def test_turn_credential_hmac_matches_coturn():
    secret = "my-shared-secret"
    creds = _generate_turn_credentials(secret, "node-x")
    turn_username = creds["username"]
    expected = base64.b64encode(
        hmac.new(secret.encode(), turn_username.encode(), hashlib.sha1).digest()
    ).decode()
    assert creds["credential"] == expected


def test_turn_credential_different_secrets_differ():
    c1 = _generate_turn_credentials("secret-a", "node-a")
    c2 = _generate_turn_credentials("secret-b", "node-a")
    assert c1["credential"] != c2["credential"]

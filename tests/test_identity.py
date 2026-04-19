"""Tests for src/node/identity.py — keypair + Ethereum-style address."""

import json
import os

import pytest

from src.node.identity import (
    derive_node_id,
    generate_keypair,
    load_or_create_identity,
    sign_message,
    verify_signature,
)


def _is_hex(s: str) -> bool:
    if not s.startswith("0x"):
        return False
    try:
        int(s[2:], 16)
    except ValueError:
        return False
    return True


def test_generate_keypair_shape():
    kp = generate_keypair()
    for k in ("private_key", "public_key", "address", "created_at"):
        assert k in kp

    assert _is_hex(kp["private_key"])
    assert len(kp["private_key"]) == 2 + 64  # 32 bytes hex

    assert _is_hex(kp["public_key"])
    assert len(kp["public_key"]) == 2 + 130  # 65-byte uncompressed
    assert kp["public_key"].startswith("0x04")

    assert _is_hex(kp["address"])
    assert len(kp["address"]) == 2 + 40


def test_generate_keypair_is_random():
    a = generate_keypair()
    b = generate_keypair()
    assert a["private_key"] != b["private_key"]
    assert a["address"] != b["address"]


def test_load_or_create_creates_on_first_run(tmp_path):
    path = tmp_path / "node_key.json"
    assert not path.exists()

    ident = load_or_create_identity(str(path))

    assert path.exists()
    with path.open() as f:
        on_disk = json.load(f)
    assert on_disk == ident
    assert _is_hex(ident["address"])


def test_load_or_create_loads_existing(tmp_path):
    path = tmp_path / "node_key.json"
    first = load_or_create_identity(str(path))
    second = load_or_create_identity(str(path))
    assert first == second


def test_load_or_create_file_permissions(tmp_path):
    path = tmp_path / "node_key.json"
    load_or_create_identity(str(path))
    mode = os.stat(path).st_mode & 0o777
    assert mode == 0o600, f"expected 0o600, got {oct(mode)}"


def test_sign_verify_round_trip():
    kp = generate_keypair()
    msg = b"groove decentralized inference"
    sig = sign_message(msg, kp["private_key"])
    assert _is_hex(sig)
    assert verify_signature(msg, sig, kp["public_key"]) is True


def test_verify_rejects_tampered_message():
    kp = generate_keypair()
    sig = sign_message(b"original", kp["private_key"])
    assert verify_signature(b"tampered", sig, kp["public_key"]) is False


def test_verify_rejects_wrong_key():
    kp_a = generate_keypair()
    kp_b = generate_keypair()
    sig = sign_message(b"msg", kp_a["private_key"])
    assert verify_signature(b"msg", sig, kp_b["public_key"]) is False


def test_derive_node_id_format():
    kp = generate_keypair()
    nid = derive_node_id(kp["address"])
    assert nid.startswith("0x")
    assert len(nid[2:]) == 40
    assert all(c in "0123456789abcdef" for c in nid[2:])


def test_derive_node_id_lowercases():
    nid = derive_node_id("0xABCDEF0123456789ABCDEF0123456789ABCDEF01")
    assert nid == "0xabcdef0123456789abcdef0123456789abcdef01"


def test_derive_node_id_rejects_bad_input():
    with pytest.raises(ValueError):
        derive_node_id("not-an-address")
    with pytest.raises(ValueError):
        derive_node_id("0xZZ")


def test_address_is_deterministic_from_private_key(tmp_path):
    """Reloading the same key file yields the same address."""
    path = tmp_path / "node_key.json"
    a = load_or_create_identity(str(path))
    b = load_or_create_identity(str(path))
    assert derive_node_id(a["address"]) == derive_node_id(b["address"])

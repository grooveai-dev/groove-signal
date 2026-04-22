"""Minimal crypto identity helpers for signal server authentication.

Extracted from src/node/identity.py — only the two functions needed
by the signal server: address_from_public_key and verify_signature.
"""

from __future__ import annotations


def _keccak256(data: bytes) -> bytes:
    try:
        from Crypto.Hash import keccak

        h = keccak.new(digest_bits=256)
        h.update(data)
        return h.digest()
    except ImportError:
        pass
    try:
        import sha3

        h = sha3.keccak_256()
        h.update(data)
        return h.digest()
    except ImportError as e:
        raise ImportError(
            "Need pycryptodome or pysha3 for keccak256. "
            "Install with `pip install pycryptodome`."
        ) from e


def _load_backend():
    try:
        import eth_keys  # noqa: F401
        from eth_keys import keys as eth_keys_keys

        return "eth_keys", eth_keys_keys
    except ImportError:
        pass
    try:
        import coincurve

        return "coincurve", coincurve
    except ImportError:
        pass
    try:
        import ecdsa

        return "ecdsa", ecdsa
    except ImportError as e:
        raise ImportError(
            "Need eth-keys, coincurve, or ecdsa for secp256k1. "
            "Install with `pip install eth-keys`."
        ) from e


def _strip_0x(s: str) -> str:
    return s[2:] if s.startswith("0x") else s


def _address_from_pub(pub_uncompressed: bytes) -> str:
    if len(pub_uncompressed) == 65 and pub_uncompressed[0] == 0x04:
        xy = pub_uncompressed[1:]
    elif len(pub_uncompressed) == 64:
        xy = pub_uncompressed
    else:
        raise ValueError(
            f"expected 64- or 65-byte public key, got {len(pub_uncompressed)}"
        )
    digest = _keccak256(xy)
    return "0x" + digest[-20:].hex()


def address_from_public_key(public_key_hex: str) -> str:
    """Derive the Ethereum address from a hex-encoded public key."""
    pub = bytes.fromhex(_strip_0x(public_key_hex))
    return _address_from_pub(pub)


def verify_signature(
    message_bytes: bytes, signature_hex: str, public_key_hex: str
) -> bool:
    """Verify a signature produced by sign_message."""
    sig = bytes.fromhex(_strip_0x(signature_hex))
    pub = bytes.fromhex(_strip_0x(public_key_hex))
    digest = _keccak256(message_bytes)
    backend, mod = _load_backend()
    try:
        if backend == "eth_keys":
            sig_obj = mod.Signature(sig)
            xy = pub[1:] if len(pub) == 65 and pub[0] == 0x04 else pub
            pk = mod.PublicKey(xy)
            return sig_obj.verify_msg_hash(digest, pk)
        if backend == "coincurve":
            import coincurve  # type: ignore

            pk = coincurve.PublicKey(pub)
            return pk.verify(sig, digest, hasher=None)
        if backend == "ecdsa":
            xy = pub[1:] if len(pub) == 65 and pub[0] == 0x04 else pub
            vk = mod.VerifyingKey.from_string(xy, curve=mod.SECP256k1)
            return vk.verify_digest(sig, digest)
    except Exception:
        return False
    return False

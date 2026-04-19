"""Node cryptographic identity.

Each node is identified by a secp256k1 keypair whose Ethereum-style address
(keccak256 of the uncompressed public key, last 20 bytes) serves as the
node_id. The keypair is persisted to ~/.groove/node_key.json on first run
and reused on subsequent runs. This is the same identity scheme that will
be used for $GROOVE token payments on Base L2.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import os
import secrets
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_KEY_PATH = "~/.groove/node_key.json"
_SECP256K1_N = (
    0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
)


def _keccak256(data: bytes) -> bytes:
    """Keccak-256 (the pre-NIST variant used by Ethereum)."""
    try:
        from Crypto.Hash import keccak

        h = keccak.new(digest_bits=256)
        h.update(data)
        return h.digest()
    except ImportError:
        pass
    try:
        import sha3  # pysha3

        h = sha3.keccak_256()
        h.update(data)
        return h.digest()
    except ImportError as e:
        raise ImportError(
            "Need pycryptodome or pysha3 for keccak256. "
            "Install with `pip install pycryptodome`."
        ) from e


def _load_backend():
    """Return a tuple (backend_name, module). Prefers eth_keys."""
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


def _priv_to_pub_uncompressed(priv_bytes: bytes) -> bytes:
    """Return 65-byte uncompressed public key (0x04 || X || Y)."""
    backend, mod = _load_backend()
    if backend == "eth_keys":
        pk = mod.PrivateKey(priv_bytes)
        pub_xy = pk.public_key.to_bytes()  # 64 bytes, no 0x04 prefix
        return b"\x04" + pub_xy
    if backend == "coincurve":
        pk = mod.PrivateKey(priv_bytes)
        return pk.public_key.format(compressed=False)  # 65 bytes
    if backend == "ecdsa":
        sk = mod.SigningKey.from_string(priv_bytes, curve=mod.SECP256k1)
        vk = sk.get_verifying_key()
        return b"\x04" + vk.to_string()
    raise RuntimeError(f"unknown backend: {backend}")


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
    """Derive the Ethereum address from a hex-encoded public key.

    Accepts 0x-prefixed or bare hex, 64-byte or 65-byte (0x04-prefixed).
    """
    pub = bytes.fromhex(_strip_0x(public_key_hex))
    return _address_from_pub(pub)


def generate_keypair() -> dict:
    """Generate a fresh secp256k1 keypair and derive the Ethereum address.

    Returns a dict with hex-encoded private_key (32 bytes), public_key
    (65-byte uncompressed, 0x04-prefixed), and address (20-byte keccak256
    of the raw X||Y public key), plus an ISO-8601 created_at timestamp.
    """
    while True:
        priv = secrets.token_bytes(32)
        if 0 < int.from_bytes(priv, "big") < _SECP256K1_N:
            break
    pub = _priv_to_pub_uncompressed(priv)
    return {
        "private_key": "0x" + priv.hex(),
        "public_key": "0x" + pub.hex(),
        "address": _address_from_pub(pub),
        "created_at": _dt.datetime.now(_dt.timezone.utc).isoformat().replace("+00:00", "Z"),
    }


def _resolve(path: str | os.PathLike) -> Path:
    return Path(os.path.expanduser(str(path)))


def load_or_create_identity(key_path: str | os.PathLike = DEFAULT_KEY_PATH) -> dict:
    """Load the node identity from disk, or generate and persist one.

    The file is written with 0600 permissions since it contains a private
    key. If the directory does not exist it is created with 0700.
    """
    p = _resolve(key_path)
    if p.exists():
        try:
            with p.open("r") as f:
                ident = json.load(f)
            required = {"private_key", "public_key", "address", "created_at"}
            if not required.issubset(ident.keys()):
                raise ValueError(f"identity file missing fields: {required - ident.keys()}")
            logger.info("loaded node identity address=%s", ident["address"])
            return ident
        except (json.JSONDecodeError, ValueError) as e:
            logger.error("identity file %s unreadable: %s", p, e)
            raise

    ident = generate_keypair()
    p.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with p.open("w") as f:
        json.dump(ident, f, indent=2)
    try:
        os.chmod(p, 0o600)
    except OSError:
        logger.warning(
            "could not set 0600 permissions on key file %s — "
            "private key may be readable by other users",
            p,
        )
    logger.info("generated new node identity address=%s path=%s", ident["address"], p)
    return ident


def sign_message(message_bytes: bytes, private_key_hex: str) -> str:
    """Sign arbitrary bytes with the node's private key.

    Returns a hex-encoded signature (65 bytes for eth_keys/coincurve, 64
    for ecdsa). Prepended with 0x.
    """
    priv = bytes.fromhex(_strip_0x(private_key_hex))
    backend, mod = _load_backend()
    digest = _keccak256(message_bytes)
    if backend == "eth_keys":
        pk = mod.PrivateKey(priv)
        sig = pk.sign_msg_hash(digest)
        return "0x" + sig.to_bytes().hex()
    if backend == "coincurve":
        pk = mod.PrivateKey(priv)
        sig = pk.sign(digest, hasher=None)
        return "0x" + sig.hex()
    if backend == "ecdsa":
        sk = mod.SigningKey.from_string(priv, curve=mod.SECP256k1)
        return "0x" + sk.sign_digest(digest).hex()
    raise RuntimeError(f"unknown backend: {backend}")


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


def derive_node_id(address: str) -> str:
    """Return the canonical node_id for an Ethereum-style address."""
    addr = address.lower()
    if not addr.startswith("0x"):
        addr = "0x" + addr
    hex_part = addr[2:]
    if len(hex_part) != 40 or any(c not in "0123456789abcdef" for c in hex_part):
        raise ValueError(f"invalid Ethereum address: {address!r}")
    return addr

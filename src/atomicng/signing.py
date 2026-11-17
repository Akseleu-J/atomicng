"""Sign and verify overlays and writer checkpoints with detached Ed25519 signatures.

Publishers retain the private key. Consumers verify the file bytes with the
public key. The implementation uses the optional cryptography package.
"""
from __future__ import annotations

from pathlib import Path

from .exceptions import BackendError

try:
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
except ImportError:  # pragma: no cover - import when using, not when loading a packet
    Ed25519PrivateKey = None  # type: ignore[assignment]


def _require():
    if Ed25519PrivateKey is None:
        raise BackendError("Signing requires cryptography: pip install cryptography")
    return True


def generate_keypair() -> tuple[bytes, bytes]:
    """→ (private_pem, public_pem); The private key is only stored by the publisher.."""
    _require()
    key = Ed25519PrivateKey.generate()
    private_pem = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    public_pem = key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    return private_pem, public_pem


def sign_file(path: str, private_pem: bytes) -> str:
    """Signs bytes of the file with a detached signature, writes `<path>.sig`, returns hex signature."""
    _require()
    key = serialization.load_pem_private_key(private_pem, password=None)
    signature = key.sign(Path(path).read_bytes())
    Path(f"{path}.sig").write_bytes(signature)
    return signature.hex()


def verify_file(path: str, public_pem: bytes) -> bool:
    """Checking. `<path>.sig` Not signing a file is not a success.."""
    _require()
    sig_path = Path(f"{path}.sig")
    if not sig_path.exists():
        return False
    public_key = serialization.load_pem_public_key(public_pem)
    try:
        public_key.verify(sig_path.read_bytes(), Path(path).read_bytes())
    except InvalidSignature:
        return False
    return True

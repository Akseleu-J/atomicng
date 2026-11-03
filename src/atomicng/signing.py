"""Подпись артефактов (S1, P0): detached ed25519-подписи для .dng-оверлеев и writer-чекпоинтов.

Модель как у релизов: приватный ключ у издателя, публичный — у потребителей; `verify_file`
проверяет подпись байтов файла (манифест целостности внутри оверлея ловит случайную порчу,
подпись — подмену). Требует `cryptography` (опциональная зависимость).
"""
from __future__ import annotations

from pathlib import Path

from .exceptions import BackendError

try:
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
except ImportError:  # pragma: no cover — импорт при использовании, не при загрузке пакета
    Ed25519PrivateKey = None  # type: ignore[assignment]


def _require():
    if Ed25519PrivateKey is None:
        raise BackendError("signing требует cryptography: pip install cryptography")
    return True


def generate_keypair() -> tuple[bytes, bytes]:
    """→ (private_pem, public_pem); приватный ключ хранит только издатель."""
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
    """Подписывает байты файла detached-подписью, пишет `<path>.sig`, возвращает hex-подпись."""
    _require()
    key = serialization.load_pem_private_key(private_pem, password=None)
    signature = key.sign(Path(path).read_bytes())
    Path(f"{path}.sig").write_bytes(signature)
    return signature.hex()


def verify_file(path: str, public_pem: bytes) -> bool:
    """Проверяет `<path>.sig` против байтов файла. Отсутствие подписи — не успех."""
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

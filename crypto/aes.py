"""
AES-256-GCM Authenticated Encryption Module
============================================
Provides authenticated encryption / decryption using AES-256 in GCM mode.

Design decisions:
    - 96-bit nonce (NIST SP 800-38D recommended size for GCM).
    - Nonce is generated via os.urandom() per encryption call (CSPRNG).
    - Authentication tag is 128 bits (full-length, appended to ciphertext).
    - Wire format: ``salt ‖ nonce ‖ tag ‖ ciphertext``.
    - Decryption verifies the tag before releasing any plaintext.

Security notes:
    - GCM nonce reuse is catastrophic; each call generates a fresh nonce.
    - The caller is responsible for key management and salt storage.
    - No padding oracle exists because GCM is a stream cipher mode.
"""

from __future__ import annotations

import os
import struct
from dataclasses import dataclass
from typing import Optional, Tuple

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from crypto.kdf import DerivedKeyBundle, derive_key


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_NONCE_LEN = 12   # 96 bits — NIST-recommended for AES-GCM
_TAG_LEN = 16     # 128-bit authentication tag
_SALT_LEN_FIELD = 1  # 1 byte to encode salt length in wire format


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class EncryptedPayload:
    """
    Encapsulates the full output of an AES-GCM encryption operation.

    Attributes
    ----------
    ciphertext : bytes
        The raw AES-GCM ciphertext **including** the appended 128-bit tag.
    nonce : bytes
        The 96-bit nonce used for this encryption.
    salt : bytes
        The salt used in key derivation (stored so decryption can re-derive).
    kdf_algorithm : str
        Identifier of the KDF algorithm used.
    """
    ciphertext: bytes   # GCM ciphertext ‖ tag
    nonce: bytes
    salt: bytes
    kdf_algorithm: str

    # ---- Serialization to / from wire format ----
    def to_bytes(self) -> bytes:
        """
        Serialize to a self-contained byte string.

        Wire format:
            [1B salt_len][salt][12B nonce][ciphertext+tag]
        """
        if len(self.salt) > 255:
            raise ValueError("Salt length must fit in a single byte.")
        return (
            struct.pack("B", len(self.salt))
            + self.salt
            + self.nonce
            + self.ciphertext
        )

    @classmethod
    def from_bytes(cls, data: bytes, kdf_algorithm: str = "argon2id") -> "EncryptedPayload":
        """Deserialize from wire format produced by ``to_bytes``."""
        offset = 0
        (salt_len,) = struct.unpack_from("B", data, offset)
        offset += 1
        salt = data[offset:offset + salt_len]
        offset += salt_len
        nonce = data[offset:offset + _NONCE_LEN]
        offset += _NONCE_LEN
        ciphertext = data[offset:]
        return cls(
            ciphertext=ciphertext,
            nonce=nonce,
            salt=salt,
            kdf_algorithm=kdf_algorithm,
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def encrypt(
    plaintext: bytes,
    password: str | bytes,
    kdf_algorithm: str = "argon2id",
    associated_data: Optional[bytes] = None,
) -> EncryptedPayload:
    """
    Encrypt *plaintext* with AES-256-GCM using a password-derived key.

    Parameters
    ----------
    plaintext : bytes
        Data to encrypt.
    password : str or bytes
        Passphrase for key derivation.
    kdf_algorithm : str
        KDF to use (``"argon2id"`` or ``"pbkdf2"``).
    associated_data : bytes, optional
        Additional authenticated data (AAD) — authenticated but not encrypted.

    Returns
    -------
    EncryptedPayload
    """
    key_bundle: DerivedKeyBundle = derive_key(password, algorithm=kdf_algorithm)
    nonce = os.urandom(_NONCE_LEN)

    aesgcm = AESGCM(key_bundle.key)
    ciphertext_with_tag: bytes = aesgcm.encrypt(nonce, plaintext, associated_data)

    return EncryptedPayload(
        ciphertext=ciphertext_with_tag,
        nonce=nonce,
        salt=key_bundle.salt,
        kdf_algorithm=kdf_algorithm,
    )


def decrypt(
    payload: EncryptedPayload,
    password: str | bytes,
    associated_data: Optional[bytes] = None,
) -> bytes:
    """
    Decrypt an ``EncryptedPayload`` produced by :func:`encrypt`.

    Parameters
    ----------
    payload : EncryptedPayload
    password : str or bytes
    associated_data : bytes, optional

    Returns
    -------
    bytes
        The original plaintext.

    Raises
    ------
    cryptography.exceptions.InvalidTag
        If the authentication tag is invalid (tampered or wrong password).
    """
    key_bundle: DerivedKeyBundle = derive_key(
        password,
        algorithm=payload.kdf_algorithm,
        salt=payload.salt,
    )

    aesgcm = AESGCM(key_bundle.key)
    return aesgcm.decrypt(payload.nonce, payload.ciphertext, associated_data)


def encrypt_bytes(
    plaintext: bytes,
    password: str | bytes,
    kdf_algorithm: str = "argon2id",
) -> bytes:
    """Convenience: encrypt and return the self-contained wire-format bytes."""
    return encrypt(plaintext, password, kdf_algorithm).to_bytes()


def decrypt_bytes(
    data: bytes,
    password: str | bytes,
    kdf_algorithm: str = "argon2id",
) -> bytes:
    """Convenience: decrypt from wire-format bytes."""
    payload = EncryptedPayload.from_bytes(data, kdf_algorithm=kdf_algorithm)
    return decrypt(payload, password)

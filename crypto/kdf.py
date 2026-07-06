"""
Key Derivation Functions (KDF) Module
=====================================
Implements cryptographically secure key derivation using Argon2id and PBKDF2.

Argon2id is the recommended KDF for password hashing (RFC 9106), providing
resistance against both GPU-based and side-channel attacks. PBKDF2-HMAC-SHA256
is provided as a NIST SP 800-132 compliant fallback.

Security Notes:
    - Salt is generated via os.urandom() (CSPRNG) at 128 bits minimum.
    - Argon2id parameters follow OWASP recommendations (≥19 MiB, ≥2 iterations).
    - Derived keys are 256 bits for use with AES-256-GCM.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import struct
from dataclasses import dataclass, field
from typing import Literal, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Optional: argon2-cffi for Argon2id (graceful fallback to PBKDF2)
# ---------------------------------------------------------------------------
try:
    from argon2.low_level import Type as Argon2Type
    from argon2.low_level import hash_secret_raw

    _HAS_ARGON2 = True
except ImportError:  # pragma: no cover
    _HAS_ARGON2 = False


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Argon2Params:
    """Argon2id tuning parameters."""
    time_cost: int = 3
    memory_cost: int = 65_536  # KiB (64 MiB)
    parallelism: int = 4
    hash_len: int = 32          # 256-bit output
    salt_len: int = 16          # 128-bit salt


@dataclass(frozen=True)
class PBKDF2Params:
    """PBKDF2-HMAC-SHA256 tuning parameters."""
    iterations: int = 600_000   # OWASP 2023 recommendation
    hash_algorithm: str = "sha256"
    salt_len: int = 16
    hash_len: int = 32


@dataclass(frozen=True)
class DerivedKeyBundle:
    """Container for a derived key together with its derivation metadata."""
    key: bytes
    salt: bytes
    algorithm: str
    params: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.key) < 16:
            raise ValueError("Derived key must be at least 128 bits.")

    # Prevent accidental leakage in logs
    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"DerivedKeyBundle(algorithm={self.algorithm!r}, "
            f"key_len={len(self.key)}, salt_len={len(self.salt)})"
        )


# ---------------------------------------------------------------------------
# Entropy validation helpers
# ---------------------------------------------------------------------------
def _shannon_entropy_bytes(data: bytes) -> float:
    """Compute Shannon entropy (bits) of a byte sequence."""
    if not data:
        return 0.0
    counts = np.bincount(np.frombuffer(data, dtype=np.uint8), minlength=256)
    probs = counts[counts > 0] / len(data)
    return float(-np.sum(probs * np.log2(probs)))


def validate_key_entropy(key: bytes, *, min_bits: float = 3.5) -> bool:
    """
    Return True if the derived key has Shannon entropy ≥ *min_bits* per byte.

    A uniformly random 256-bit key has ≈ 8.0 bits/byte entropy in the limit
    of long sequences.  For short keys (32 bytes) the measured entropy per
    byte fluctuates considerably; we use a conservative threshold of 3.5
    bits/byte to flag pathological (constant or near-constant) outputs while
    avoiding false negatives on correctly derived keys.
    """
    return _shannon_entropy_bytes(key) >= min_bits


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def generate_salt(length: int = 16) -> bytes:
    """Generate a cryptographically secure random salt."""
    if length < 16:
        raise ValueError("Salt must be at least 128 bits (16 bytes).")
    return os.urandom(length)


def derive_key_argon2(
    password: str | bytes,
    salt: Optional[bytes] = None,
    params: Optional[Argon2Params] = None,
) -> DerivedKeyBundle:
    """
    Derive a 256-bit key using Argon2id.

    Parameters
    ----------
    password : str or bytes
        User-supplied passphrase.
    salt : bytes, optional
        If *None*, a fresh 128-bit salt is generated.
    params : Argon2Params, optional
        Tuning parameters (defaults to dataclass defaults).

    Returns
    -------
    DerivedKeyBundle

    Raises
    ------
    RuntimeError
        If argon2-cffi is not installed.
    """
    if not _HAS_ARGON2:
        raise RuntimeError(
            "argon2-cffi is required for Argon2id key derivation.  "
            "Install it via: pip install argon2-cffi"
        )

    params = params or Argon2Params()
    salt = salt or generate_salt(params.salt_len)
    secret = password.encode("utf-8") if isinstance(password, str) else password

    key = hash_secret_raw(
        secret=secret,
        salt=salt,
        time_cost=params.time_cost,
        memory_cost=params.memory_cost,
        parallelism=params.parallelism,
        hash_len=params.hash_len,
        type=Argon2Type.ID,
    )

    return DerivedKeyBundle(
        key=key,
        salt=salt,
        algorithm="argon2id",
        params={
            "time_cost": params.time_cost,
            "memory_cost": params.memory_cost,
            "parallelism": params.parallelism,
            "hash_len": params.hash_len,
        },
    )


def derive_key_pbkdf2(
    password: str | bytes,
    salt: Optional[bytes] = None,
    params: Optional[PBKDF2Params] = None,
) -> DerivedKeyBundle:
    """
    Derive a 256-bit key using PBKDF2-HMAC-SHA256.

    This implementation uses the stdlib ``hashlib.pbkdf2_hmac`` which delegates
    to OpenSSL and is therefore constant-time.

    Parameters
    ----------
    password : str or bytes
        User-supplied passphrase.
    salt : bytes, optional
        If *None*, a fresh 128-bit salt is generated.
    params : PBKDF2Params, optional
        Tuning parameters.

    Returns
    -------
    DerivedKeyBundle
    """
    params = params or PBKDF2Params()
    salt = salt or generate_salt(params.salt_len)
    secret = password.encode("utf-8") if isinstance(password, str) else password

    key = hashlib.pbkdf2_hmac(
        hash_name=params.hash_algorithm,
        password=secret,
        salt=salt,
        iterations=params.iterations,
        dklen=params.hash_len,
    )

    return DerivedKeyBundle(
        key=key,
        salt=salt,
        algorithm="pbkdf2-hmac-sha256",
        params={
            "iterations": params.iterations,
            "hash_algorithm": params.hash_algorithm,
            "hash_len": params.hash_len,
        },
    )


def derive_key(
    password: str | bytes,
    algorithm: Literal["argon2id", "pbkdf2"] = "argon2id",
    salt: Optional[bytes] = None,
    **kwargs,
) -> DerivedKeyBundle:
    """
    Unified key derivation interface.

    Parameters
    ----------
    password : str or bytes
    algorithm : {"argon2id", "pbkdf2"}
    salt : bytes, optional
    **kwargs
        Forwarded to the underlying param dataclass.

    Returns
    -------
    DerivedKeyBundle
    """
    if algorithm == "argon2id":
        params = Argon2Params(**{k: v for k, v in kwargs.items() if k in Argon2Params.__dataclass_fields__})
        return derive_key_argon2(password, salt=salt, params=params)
    elif algorithm == "pbkdf2":
        params = PBKDF2Params(**{k: v for k, v in kwargs.items() if k in PBKDF2Params.__dataclass_fields__})
        return derive_key_pbkdf2(password, salt=salt, params=params)
    else:
        raise ValueError(f"Unsupported KDF algorithm: {algorithm!r}")

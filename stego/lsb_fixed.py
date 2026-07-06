"""
Fixed LSB Steganography Module
===============================
Implements traditional Least Significant Bit (LSB) embedding with fixed
bit-depth (1-bit and 2-bit) and controlled embedding rates (bpp-based).

The module uses a payload header (64 bits) that stores the embedded payload
length to enable blind extraction (the receiver does not need to know the
payload size a priori).

Wire format embedded in the image:
    [64-bit payload_length_header][payload_bits]

All pixel manipulation is fully vectorized with NumPy for performance.
"""

from __future__ import annotations

import struct
from typing import Literal, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
HEADER_BITS = 64  # 64-bit unsigned integer for payload length (in bytes)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _bytes_to_bits(data: bytes) -> np.ndarray:
    """Convert a byte string to a 1-D NumPy array of bits (uint8 0/1)."""
    arr = np.frombuffer(data, dtype=np.uint8)
    return np.unpackbits(arr)


def _bits_to_bytes(bits: np.ndarray) -> bytes:
    """Convert a 1-D bit array back to bytes (pads to multiple of 8)."""
    # Pad to multiple of 8
    remainder = len(bits) % 8
    if remainder:
        bits = np.concatenate([bits, np.zeros(8 - remainder, dtype=np.uint8)])
    return np.packbits(bits).tobytes()


def _make_header(payload_len_bytes: int) -> np.ndarray:
    """Create a 64-bit header encoding the payload length in bytes."""
    header_bytes = struct.pack(">Q", payload_len_bytes)
    return _bytes_to_bits(header_bytes)


def _parse_header(header_bits: np.ndarray) -> int:
    """Decode the 64-bit header to recover the payload length in bytes."""
    header_bytes = _bits_to_bytes(header_bits[:HEADER_BITS])
    return struct.unpack(">Q", header_bytes[:8])[0]


def _embedding_capacity(image: np.ndarray, lsb_depth: int) -> int:
    """Total embeddable bits across all channels at *lsb_depth* bits/sample."""
    return int(np.prod(image.shape)) * lsb_depth


def _bpp_to_max_bits(image: np.ndarray, bpp: float) -> int:
    """
    Convert bits-per-pixel to a maximum bit budget.

    bpp is defined per pixel (not per sample), so for an H×W×C image the
    total budget is ``H * W * bpp``.
    """
    h, w = image.shape[:2]
    return int(h * w * bpp)


# ---------------------------------------------------------------------------
# Embedding (fixed depth)
# ---------------------------------------------------------------------------
def embed_fixed(
    cover: np.ndarray,
    payload: bytes,
    lsb_depth: Literal[1, 2] = 1,
    bpp: Optional[float] = None,
    seed: int = 42,
) -> np.ndarray:
    """
    Embed *payload* into *cover* using fixed-depth LSB replacement.

    Parameters
    ----------
    cover : np.ndarray
        Cover image, uint8, shape (H, W) or (H, W, C).
    payload : bytes
        Secret data to embed.
    lsb_depth : {1, 2}
        Number of least-significant bits to replace per sample.
    bpp : float, optional
        If given, limits the total embedding to this many bits per pixel.
        Extra capacity is unused.
    seed : int
        PRNG seed for pseudo-random pixel permutation (spread embedding).

    Returns
    -------
    np.ndarray
        Stego image (same shape and dtype as *cover*).

    Raises
    ------
    ValueError
        If the payload (plus header) exceeds available capacity.
    """
    if cover.dtype != np.uint8:
        raise TypeError("Cover image must be uint8.")
    if lsb_depth not in (1, 2):
        raise ValueError("lsb_depth must be 1 or 2.")

    stego = cover.copy()
    flat = stego.ravel()  # mutable view

    total_samples = flat.shape[0]
    capacity_bits = total_samples * lsb_depth

    # Optionally clamp capacity by bpp
    if bpp is not None:
        capacity_bits = min(capacity_bits, _bpp_to_max_bits(cover, bpp))

    # Build bit stream: header + payload
    header_bits = _make_header(len(payload))
    payload_bits = _bytes_to_bits(payload)
    stream = np.concatenate([header_bits, payload_bits]).astype(np.uint8)

    if len(stream) > capacity_bits:
        raise ValueError(
            f"Payload ({len(stream)} bits) exceeds capacity "
            f"({capacity_bits} bits) at {lsb_depth}-bit depth."
        )

    # Pseudo-random sample ordering for spread embedding
    rng = np.random.RandomState(seed)
    indices = rng.permutation(total_samples)

    if lsb_depth == 1:
        # Clear LSB of selected samples, then set
        n = len(stream)
        sel = indices[:n]
        flat[sel] = (flat[sel] & 0xFE) | stream
    elif lsb_depth == 2:
        # Pack pairs of bits into the 2 LSBs
        # Ensure stream length is even
        if len(stream) % 2:
            stream = np.append(stream, np.uint8(0))
        n_samples = len(stream) // 2
        sel = indices[:n_samples]
        two_bit_values = (stream[0::2] << 1) | stream[1::2]
        flat[sel] = (flat[sel] & 0xFC) | two_bit_values

    return stego


# ---------------------------------------------------------------------------
# Extraction (fixed depth)
# ---------------------------------------------------------------------------
def extract_fixed(
    stego: np.ndarray,
    lsb_depth: Literal[1, 2] = 1,
    seed: int = 42,
) -> bytes:
    """
    Extract an embedded payload from *stego* using fixed-depth LSB.

    Parameters
    ----------
    stego : np.ndarray
        Stego image, uint8.
    lsb_depth : {1, 2}
    seed : int
        Must match the seed used during embedding.

    Returns
    -------
    bytes
        The extracted secret payload.
    """
    if stego.dtype != np.uint8:
        raise TypeError("Stego image must be uint8.")

    flat = stego.ravel()
    total_samples = flat.shape[0]

    rng = np.random.RandomState(seed)
    indices = rng.permutation(total_samples)

    if lsb_depth == 1:
        # Read header first
        header_sel = indices[:HEADER_BITS]
        header_bits = flat[header_sel] & 1
        payload_len = _parse_header(header_bits.astype(np.uint8))

        total_bits = HEADER_BITS + payload_len * 8
        sel = indices[:total_bits]
        all_bits = (flat[sel] & 1).astype(np.uint8)
    elif lsb_depth == 2:
        # Each sample holds 2 bits
        header_samples = (HEADER_BITS + 1) // 2
        header_sel = indices[:header_samples]
        two_bits = flat[header_sel] & 0x03
        header_bits = np.zeros(header_samples * 2, dtype=np.uint8)
        header_bits[0::2] = (two_bits >> 1) & 1
        header_bits[1::2] = two_bits & 1
        payload_len = _parse_header(header_bits[:HEADER_BITS].astype(np.uint8))

        total_bits = HEADER_BITS + payload_len * 8
        total_samples_needed = (total_bits + 1) // 2
        sel = indices[:total_samples_needed]
        two_bits_all = flat[sel] & 0x03
        all_bits = np.zeros(total_samples_needed * 2, dtype=np.uint8)
        all_bits[0::2] = (two_bits_all >> 1) & 1
        all_bits[1::2] = two_bits_all & 1
    else:
        raise ValueError("lsb_depth must be 1 or 2.")

    # Slice header off, keep payload bits
    payload_bits = all_bits[HEADER_BITS:HEADER_BITS + payload_len * 8]
    return _bits_to_bytes(payload_bits)[:payload_len]


# ---------------------------------------------------------------------------
# Capacity query
# ---------------------------------------------------------------------------
def capacity_bytes(
    image: np.ndarray,
    lsb_depth: Literal[1, 2] = 1,
    bpp: Optional[float] = None,
) -> int:
    """Return usable payload capacity in bytes (excluding header)."""
    cap_bits = _embedding_capacity(image, lsb_depth)
    if bpp is not None:
        cap_bits = min(cap_bits, _bpp_to_max_bits(image, bpp))
    return max(0, (cap_bits - HEADER_BITS) // 8)

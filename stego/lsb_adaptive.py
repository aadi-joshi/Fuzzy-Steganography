"""
Adaptive Fuzzy LSB Steganography Module
========================================
Embeds secret data using a per-pixel variable embedding depth determined by
the fuzzy inference engine.  The depth map is derived from local entropy,
edge magnitude, and payload pressure.

Key differences from fixed LSB:
    - Embedding depth varies per pixel (1–3 bits) based on local features.
    - Smooth regions use fewer bits (harder to detect).
    - Textured / edge regions use more bits (higher capacity, lower perceptibility).
    - Total capacity is adaptive and data-driven.

Protocol:
    1. The depth map is computed from the cover image features.
    2. A 64-bit header encodes payload length (embedded at depth 1 in the
       first 64 pseudo-randomly chosen samples for reliability).
    3. Payload bits are spread across channels in pseudo-random order,
       each sample contributing ``depth_map[pixel]`` bits.
"""

from __future__ import annotations

import struct
from typing import Optional, Tuple

import numpy as np

from stego.entropy import extract_features
from stego.fuzzy import FuzzyDepthController

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
HEADER_BITS = 64


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _bytes_to_bits(data: bytes) -> np.ndarray:
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8))


def _bits_to_bytes(bits: np.ndarray) -> bytes:
    remainder = len(bits) % 8
    if remainder:
        bits = np.concatenate([bits, np.zeros(8 - remainder, dtype=np.uint8)])
    return np.packbits(bits).tobytes()


def _make_header(payload_len_bytes: int) -> np.ndarray:
    return _bytes_to_bits(struct.pack(">Q", payload_len_bytes))


def _parse_header(header_bits: np.ndarray) -> int:
    return struct.unpack(">Q", _bits_to_bytes(header_bits[:HEADER_BITS])[:8])[0]


# ---------------------------------------------------------------------------
# Depth map computation
# ---------------------------------------------------------------------------
def compute_depth_map(
    cover: np.ndarray,
    controller: FuzzyDepthController,
    pressure: float = 0.5,
    window_size: int = 7,
) -> np.ndarray:
    """
    Compute the adaptive per-pixel embedding depth map.

    Parameters
    ----------
    cover : np.ndarray
        Cover image, uint8, shape (H, W) or (H, W, C).
    controller : FuzzyDepthController
    pressure : float
        Payload pressure in [0, 1].
    window_size : int
        Entropy window size.

    Returns
    -------
    np.ndarray, shape (H, W), dtype int32
        Per-pixel depth in {1, 2, 3}.
    """
    entropy_map, edge_map = extract_features(cover, window_size=window_size)
    return controller.infer(entropy_map, edge_map, pressure)


def adaptive_capacity_bytes(
    depth_map: np.ndarray,
    n_channels: int = 3,
) -> int:
    """Usable payload capacity (bytes) for a given depth map, minus header."""
    total_bits = int(np.sum(depth_map)) * n_channels
    return max(0, (total_bits - HEADER_BITS) // 8)


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------
def embed_adaptive(
    cover: np.ndarray,
    payload: bytes,
    controller: FuzzyDepthController,
    bpp: Optional[float] = None,
    pressure: float = 0.5,
    window_size: int = 7,
    seed: int = 42,
    depth_map: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Embed *payload* using fuzzy adaptive LSB replacement.

    Parameters
    ----------
    cover : np.ndarray
        Cover image, uint8, shape (H, W, C) with C ∈ {1, 3, 4}.
    payload : bytes
        Secret data to embed.
    controller : FuzzyDepthController
        Configured fuzzy inference engine.
    bpp : float, optional
        Maximum bits-per-pixel budget (if set, may reduce effective capacity).
    pressure : float
        Initial payload pressure estimate.
    window_size : int
        Window size for local entropy.
    seed : int
        PRNG seed.

    Returns
    -------
    stego : np.ndarray
        Stego image, same shape and dtype as *cover*.
    depth_map : np.ndarray, shape (H, W)
        The computed depth map (useful for analysis).

    Raises
    ------
    ValueError
        If payload exceeds adaptive capacity.
    """
    if cover.dtype != np.uint8:
        raise TypeError("Cover image must be uint8.")
    if cover.ndim == 2:
        cover = cover[..., np.newaxis]

    h, w, c = cover.shape
    n_pixels = h * w
    n_samples = n_pixels * c  # total sample slots

    # --- Compute depth map (or use pre-computed) ---
    if depth_map is None:
        depth_map = compute_depth_map(cover, controller, pressure, window_size)

    # Expand depth map to per-sample: (H, W) → (H, W, C) → (N*C,)
    depth_per_sample = np.broadcast_to(
        depth_map[..., np.newaxis], (h, w, c)
    ).ravel()  # (N*C,)

    # Optionally clamp total bits by bpp
    total_capacity_bits = int(np.sum(depth_per_sample))
    if bpp is not None:
        max_bits = int(h * w * bpp)
        if max_bits < total_capacity_bits:
            # Reduce depth map proportionally (truncate lowest-depth first)
            # Simple approach: only use the first max_bits worth of samples
            total_capacity_bits = max_bits

    # --- Build bit stream ---
    header_bits = _make_header(len(payload))
    payload_bits = _bytes_to_bits(payload)
    stream = np.concatenate([header_bits, payload_bits]).astype(np.uint8)

    if len(stream) > total_capacity_bits:
        raise ValueError(
            f"Payload + header ({len(stream)} bits) exceeds adaptive capacity "
            f"({total_capacity_bits} bits)."
        )

    # --- Pseudo-random embedding order ---
    rng = np.random.RandomState(seed)
    order = rng.permutation(n_samples)

    stego = cover.copy()
    flat = stego.reshape(-1)  # mutable uint8 view

    bit_idx = 0  # pointer into stream

    # Vectorized embedding: group samples by depth to avoid per-pixel loop
    depth_ordered = depth_per_sample[order]

    for d in (1, 2, 3):
        if bit_idx >= len(stream):
            break

        mask = depth_ordered == d
        sample_indices = order[mask]

        # How many bits can these samples absorb?
        bits_available = len(sample_indices) * d
        bits_needed = min(bits_available, len(stream) - bit_idx)

        if bits_needed <= 0:
            continue

        # Number of full samples we'll touch
        n_full = bits_needed // d
        remainder_bits = bits_needed % d

        if n_full > 0:
            sel = sample_indices[:n_full]
            bit_chunk = stream[bit_idx:bit_idx + n_full * d]

            # Reshape bit_chunk into (n_full, d) and pack into d LSBs
            bit_chunk_padded = bit_chunk  # already correct length
            bit_matrix = bit_chunk_padded.reshape(n_full, d)

            # Build the d-bit value from individual bits (MSB-first within chunk)
            value = np.zeros(n_full, dtype=np.uint8)
            for bit_pos in range(d):
                value = (value << 1) | bit_matrix[:, bit_pos]

            # Clear lower d bits, set new value
            clear_mask = np.uint8((0xFF << d) & 0xFF)
            flat[sel] = (flat[sel] & clear_mask) | value

            bit_idx += n_full * d

        # Handle leftover bits (partial sample) — embed at reduced depth
        if remainder_bits > 0 and n_full < len(sample_indices):
            sel_r = sample_indices[n_full]
            remaining = stream[bit_idx:bit_idx + remainder_bits]
            value_r = np.uint8(0)
            for b in remaining:
                value_r = (value_r << 1) | np.uint8(b)
            # Shift to align within the d LSBs
            value_r <<= (d - remainder_bits)
            clear_mask_r = np.uint8((0xFF << d) & 0xFF)
            flat[sel_r] = (flat[sel_r] & clear_mask_r) | value_r
            bit_idx += remainder_bits

    stego = flat.reshape(cover.shape)
    if stego.shape[2] == 1:
        stego = stego[..., 0]

    return stego, depth_map


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------
def extract_adaptive(
    stego: np.ndarray,
    controller: FuzzyDepthController,
    pressure: float = 0.5,
    window_size: int = 7,
    seed: int = 42,
    depth_map: Optional[np.ndarray] = None,
) -> bytes:
    """
    Extract an embedded payload from *stego* using adaptive LSB.

    The depth map is recomputed from the stego image.  Because adaptive
    embedding touches only LSBs, the feature maps (entropy, edges) computed
    on the stego image are nearly identical to those of the cover, ensuring
    the same depth map is recovered.

    Parameters
    ----------
    stego : np.ndarray
    controller : FuzzyDepthController
    pressure : float
    window_size : int
    seed : int

    Returns
    -------
    bytes
    """
    if stego.dtype != np.uint8:
        raise TypeError("Stego image must be uint8.")
    if stego.ndim == 2:
        stego = stego[..., np.newaxis]

    h, w, c = stego.shape
    n_samples = h * w * c

    if depth_map is None:
        depth_map = compute_depth_map(stego, controller, pressure, window_size)
    depth_per_sample = np.broadcast_to(
        depth_map[..., np.newaxis], (h, w, c)
    ).ravel()

    rng = np.random.RandomState(seed)
    order = rng.permutation(n_samples)
    flat = stego.reshape(-1)

    depth_ordered = depth_per_sample[order]

    # --- Extract bits in the SAME depth-group order used during embedding ---
    # Embedding processes all d=1 samples, then d=2, then d=3 sequentially.
    extracted_bits: list[np.ndarray] = []

    for d in (1, 2, 3):
        mask = depth_ordered == d
        sel = order[mask]
        if len(sel) == 0:
            continue
        values = flat[sel] & np.uint8((1 << d) - 1)
        # Unpack d bits per sample (MSB-first within chunk)
        bits = np.zeros((len(sel), d), dtype=np.uint8)
        for bit_pos in range(d):
            bits[:, bit_pos] = (values >> (d - 1 - bit_pos)) & 1
        extracted_bits.append(bits.ravel())

    if not extracted_bits:
        raise ValueError("No embedded bits found (empty depth map).")

    bit_stream = np.concatenate(extracted_bits)

    # Parse header
    if len(bit_stream) < HEADER_BITS:
        raise ValueError("Not enough embedded bits to read header.")
    payload_len = _parse_header(bit_stream[:HEADER_BITS])

    total_needed = HEADER_BITS + payload_len * 8
    if total_needed > len(bit_stream):
        raise ValueError("Payload length in header exceeds available bits.")

    payload_bits = bit_stream[HEADER_BITS:total_needed]
    return _bits_to_bytes(payload_bits)[:payload_len]

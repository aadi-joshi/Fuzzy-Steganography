"""
Fuzzy Adaptive Embedding Depth Controller
==========================================
A Mamdani-type fuzzy inference system that determines the optimal LSB
embedding depth (1–3 bits) for each pixel based on three linguistic inputs:

1. **Local entropy** — texture complexity in the pixel neighbourhood.
2. **Sobel edge magnitude** — gradient strength (edge regions tolerate more change).
3. **Payload pressure** — ratio of remaining payload to remaining capacity.

Membership functions are trapezoidal, defined by four corner points (a, b, c, d).
Defuzzification uses the centroid method by default.

The entire inference is **fully vectorized** over the spatial dimensions of the
input feature maps — no Python-level pixel loops.

Mathematical formulation
------------------------
Let  x = (e, g, p)  represent the input vector at a pixel, where:
    - e ∈ [0, 8]   local Shannon entropy
    - g ∈ [0, 1]   normalised Sobel magnitude
    - p ∈ [0, 1]   payload pressure

Each input is fuzzified using trapezoidal MFs  μ_{term}(x; a, b, c, d):

    μ(x; a, b, c, d) = max(min((x−a)/(b−a), 1, (d−x)/(d−c)), 0)

Rule activation uses the *minimum* (Mamdani AND) t-norm across antecedents.
Output aggregation uses the *maximum* (Mamdani OR) s-norm across rules.
Defuzzification uses the centroid:

    d* = ∫ y · μ_agg(y) dy  /  ∫ μ_agg(y) dy

The output d* ∈ [1, 3] is the continuous embedding depth, rounded to the
nearest integer for actual bit manipulation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml


# ---------------------------------------------------------------------------
# Trapezoidal membership function (vectorized)
# ---------------------------------------------------------------------------
def trapmf(x: np.ndarray, params: Tuple[float, float, float, float]) -> np.ndarray:
    """
    Evaluate the trapezoidal membership function μ(x; a, b, c, d).

    Parameters
    ----------
    x : np.ndarray
        Input array (any shape).
    params : (a, b, c, d)
        Four corner points with a ≤ b ≤ c ≤ d.

    Returns
    -------
    np.ndarray
        Membership degrees in [0, 1], same shape as *x*.
    """
    a, b, c, d = params
    x = np.asarray(x, dtype=np.float64)

    # Rising edge
    if b > a:
        rise = (x - a) / (b - a)
    else:
        rise = np.where(x >= a, 1.0, 0.0)

    # Falling edge
    if d > c:
        fall = (d - x) / (d - c)
    else:
        fall = np.where(x <= d, 1.0, 0.0)

    return np.clip(np.minimum(np.minimum(rise, 1.0), fall), 0.0, 1.0)


# ---------------------------------------------------------------------------
# Rule base
# ---------------------------------------------------------------------------
# Each rule is a tuple: (entropy_term, edge_term, pressure_term) → depth_term
# The rule base encodes domain knowledge: smooth low-edge regions with low
# pressure → shallow embedding; textured edges under high pressure → deep.
#
# 27 rules (3×3×3 full combinatorial):
DEFAULT_RULE_BASE: List[Tuple[str, str, str, str]] = [
    # entropy  | edge     | pressure | → depth
    ("low",      "weak",     "low",      "shallow"),
    ("low",      "weak",     "medium",   "shallow"),
    ("low",      "weak",     "high",     "moderate"),
    ("low",      "moderate", "low",      "shallow"),
    ("low",      "moderate", "medium",   "moderate"),
    ("low",      "moderate", "high",     "moderate"),
    ("low",      "strong",   "low",      "moderate"),
    ("low",      "strong",   "medium",   "moderate"),
    ("low",      "strong",   "high",     "deep"),
    ("medium",   "weak",     "low",      "shallow"),
    ("medium",   "weak",     "medium",   "moderate"),
    ("medium",   "weak",     "high",     "moderate"),
    ("medium",   "moderate", "low",      "moderate"),
    ("medium",   "moderate", "medium",   "moderate"),
    ("medium",   "moderate", "high",     "deep"),
    ("medium",   "strong",   "low",      "moderate"),
    ("medium",   "strong",   "medium",   "deep"),
    ("medium",   "strong",   "high",     "deep"),
    ("high",     "weak",     "low",      "moderate"),
    ("high",     "weak",     "medium",   "moderate"),
    ("high",     "weak",     "high",     "deep"),
    ("high",     "moderate", "low",      "moderate"),
    ("high",     "moderate", "medium",   "deep"),
    ("high",     "moderate", "high",     "deep"),
    ("high",     "strong",   "low",      "deep"),
    ("high",     "strong",   "medium",   "deep"),
    ("high",     "strong",   "high",     "deep"),
]


# ---------------------------------------------------------------------------
# Fuzzy Inference Engine
# ---------------------------------------------------------------------------
@dataclass
class FuzzyDepthController:
    """
    Mamdani fuzzy inference system for adaptive LSB embedding depth.

    Parameters
    ----------
    entropy_mf : dict
        ``{term_name: (a, b, c, d)}`` for entropy input.
    edge_mf : dict
        ``{term_name: (a, b, c, d)}`` for edge input.
    pressure_mf : dict
        ``{term_name: (a, b, c, d)}`` for pressure input.
    depth_mf : dict
        ``{term_name: (a, b, c, d)}`` for depth output.
    rules : list of (str, str, str, str)
    defuzz_method : str
        ``"centroid"`` (default) or ``"bisector"``.
    depth_resolution : int
        Number of discrete samples in the output universe for defuzzification.
    """

    entropy_mf: Dict[str, Tuple[float, float, float, float]]
    edge_mf: Dict[str, Tuple[float, float, float, float]]
    pressure_mf: Dict[str, Tuple[float, float, float, float]]
    depth_mf: Dict[str, Tuple[float, float, float, float]]
    rules: List[Tuple[str, str, str, str]] = field(default_factory=lambda: list(DEFAULT_RULE_BASE))
    defuzz_method: str = "centroid"
    depth_resolution: int = 100

    # Precomputed output universe
    _depth_universe: np.ndarray = field(init=False, repr=False)
    _depth_mf_values: Dict[str, np.ndarray] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        # Build the discrete output universe [1, 3]
        lo = min(p[0] for p in self.depth_mf.values())
        hi = max(p[3] for p in self.depth_mf.values())
        self._depth_universe = np.linspace(lo, hi, self.depth_resolution)
        self._depth_mf_values = {
            term: trapmf(self._depth_universe, params)
            for term, params in self.depth_mf.items()
        }

    # -----------------------------------------------------------------------
    # Core inference (vectorized over H×W pixels)
    # -----------------------------------------------------------------------
    def infer(
        self,
        entropy_map: np.ndarray,
        edge_map: np.ndarray,
        pressure: float,
    ) -> np.ndarray:
        """
        Compute the optimal embedding depth for every pixel.

        Parameters
        ----------
        entropy_map : np.ndarray, shape (H, W)
            Local Shannon entropy values.
        edge_map : np.ndarray, shape (H, W)
            Normalised Sobel edge magnitudes.
        pressure : float
            Scalar payload pressure in [0, 1].

        Returns
        -------
        np.ndarray, shape (H, W), dtype int
            Per-pixel embedding depth in {1, 2, 3}.
        """
        h, w = entropy_map.shape

        # --- Step 1: Fuzzify all inputs ---
        # Entropy: dict[term] → (H, W)
        ent_mu = {
            term: trapmf(entropy_map, params)
            for term, params in self.entropy_mf.items()
        }
        # Edge: dict[term] → (H, W)
        edg_mu = {
            term: trapmf(edge_map, params)
            for term, params in self.edge_mf.items()
        }
        # Pressure: dict[term] → scalar
        p_arr = np.array([pressure])
        prs_mu = {
            term: float(trapmf(p_arr, params)[0])
            for term, params in self.pressure_mf.items()
        }

        # --- Step 2: Evaluate rules (vectorized) ---
        # For each output term, accumulate the max rule activation (OR)
        # across all rules that fire into that term.
        # agg[term] has shape (H, W) — peak activation for that output term.
        agg: Dict[str, np.ndarray] = {
            term: np.zeros((h, w), dtype=np.float64)
            for term in self.depth_mf
        }

        for ent_term, edg_term, prs_term, depth_term in self.rules:
            # Rule activation: AND (minimum) of antecedent membership degrees
            alpha = np.minimum(ent_mu[ent_term], edg_mu[edg_term])  # (H, W)
            alpha = np.minimum(alpha, prs_mu[prs_term])              # (H, W)
            # OR aggregation (maximum) into the consequent term
            np.maximum(agg[depth_term], alpha, out=agg[depth_term])

        # --- Step 3: Defuzzification (centroid) ---
        return self._defuzzify(agg)

    # -----------------------------------------------------------------------
    # Defuzzification
    # -----------------------------------------------------------------------
    def _defuzzify(self, agg: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Centroid defuzzification.

        For each pixel we compute:
            d* = Σ_i  y_i · μ_agg(y_i)  /  Σ_i μ_agg(y_i)

        where μ_agg is the aggregated output fuzzy set (maximum of all
        clipped consequent MFs).

        The computation is vectorized: we broadcast the (H, W) activation
        maps against the (R,) output universe to produce (H, W, R) tensors,
        then reduce over R.
        """
        h, w = next(iter(agg.values())).shape
        R = self.depth_resolution
        y = self._depth_universe  # (R,)

        # Build aggregated output MF for every pixel: shape (H, W, R)
        mu_agg = np.zeros((h, w, R), dtype=np.float64)
        for term, alpha_map in agg.items():
            # Clip the output MF by the rule activation: min(alpha, MF(y))
            # alpha_map: (H, W) → (H, W, 1);  mf_values: (R,) → (1, 1, R)
            clipped = np.minimum(
                alpha_map[..., np.newaxis],
                self._depth_mf_values[term][np.newaxis, np.newaxis, :],
            )
            np.maximum(mu_agg, clipped, out=mu_agg)

        # Centroid
        numerator = np.sum(mu_agg * y[np.newaxis, np.newaxis, :], axis=2)
        denominator = np.sum(mu_agg, axis=2)

        # Where denominator is 0, default to depth 1 (conservative)
        depth_continuous = np.where(
            denominator > 1e-12,
            numerator / denominator,
            1.0,
        )

        # Round to nearest integer in {1, 2, 3}
        return np.clip(np.rint(depth_continuous).astype(np.int32), 1, 3)

    # -----------------------------------------------------------------------
    # Adaptive capacity
    # -----------------------------------------------------------------------
    def adaptive_capacity_bits(
        self,
        depth_map: np.ndarray,
        n_channels: int = 3,
    ) -> int:
        """
        Total embedding capacity (bits) given a per-pixel depth map.

        For a colour image each pixel contributes depth × n_channels bits.
        """
        return int(np.sum(depth_map) * n_channels)

    # -----------------------------------------------------------------------
    # Factory from config dict / YAML
    # -----------------------------------------------------------------------
    @classmethod
    def from_config(cls, cfg: dict) -> "FuzzyDepthController":
        """
        Instantiate from a parsed config dictionary (fuzzy subsection).

        Expected keys mirror ``config.yaml`` → ``stego.fuzzy``.
        """
        def _to_tuple(lst):
            return tuple(float(v) for v in lst)

        entropy_mf = {k: _to_tuple(v) for k, v in cfg["entropy_mf"].items()}
        edge_mf = {k: _to_tuple(v) for k, v in cfg["edge_mf"].items()}
        pressure_mf = {k: _to_tuple(v) for k, v in cfg["pressure_mf"].items()}
        depth_mf = {k: _to_tuple(v) for k, v in cfg["depth_mf"].items()}
        defuzz = cfg.get("defuzzification", "centroid")

        return cls(
            entropy_mf=entropy_mf,
            edge_mf=edge_mf,
            pressure_mf=pressure_mf,
            depth_mf=depth_mf,
            defuzz_method=defuzz,
        )

    @classmethod
    def from_yaml(cls, path: str) -> "FuzzyDepthController":
        """Load from a YAML config file."""
        with open(path, "r") as fh:
            cfg = yaml.safe_load(fh)
        return cls.from_config(cfg["stego"]["fuzzy"])

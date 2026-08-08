"""
Result types for geoinference.

Structured dataclasses that carry estimates, standard errors,
confidence intervals, and diagnostics from the inference pipeline.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class SEResult:
    """Standard error estimates from multiple methods.

    The same three methods are reported for every estimand, so a field means the
    same thing wherever you find it. ``recommended`` holds the one the design
    selects, or the one ``se_method`` asked for.

    There is deliberately no ``wild`` field. The wild cluster bootstrap is a
    percentile-t procedure: it bootstraps the *t statistic* to get an interval
    and studentises by the analytic cluster-robust SE, so it produces no standard
    error of its own. Putting one here would mean repeating ``cluster`` under a
    second name. It appears in :class:`CIResult` instead, where it belongs.
    """

    naive: float
    cluster: float
    bootstrap: float | None  # pairs cluster bootstrap: sd of the resampled estimates
    recommended: float
    method_used: str  # "naive", "cluster", "bootstrap"

    @property
    def ratio_cluster_to_naive(self) -> float:
        """Cluster SE / naive SE.  Near 1 ⇒ independence holds."""
        return self.cluster / self.naive if self.naive > 0 else float("nan")


@dataclass
class CIResult:
    """Confidence intervals from multiple methods."""

    normal: tuple[float, float]  # est ± z * se_recommended
    t: tuple[float, float]  # est ± t_{G-1} * se_recommended
    bootstrap: tuple[float, float] | None  # pairs cluster bootstrap, percentile
    wild: tuple[float, float] | None  # wild cluster bootstrap, percentile-t
    recommended: tuple[float, float]
    level: float  # e.g. 0.95


@dataclass
class Diagnostics:
    """Design and data quality diagnostics."""

    n_obs: int
    n_positive_frames: int
    n_empty_frames: int
    empty_frame_rate: float

    n_clusters: int
    cluster_sizes: np.ndarray
    cluster_size_mean: float
    cluster_size_cv: float

    icc: float
    deff: float
    n_eff: float

    # Kish effective number of itineraries, (sum n_c)^2 / sum n_c^2. Equal to
    # n_clusters when every itinerary is the same size, and far below it when a
    # few dominate. This is the number that predicts whether *any* interval here
    # can be trusted -- see `estimate`'s warning and the table in its docstring.
    n_clusters_eff: float

    se_ratio_cluster_to_naive: float  # for the photo-mean

    ratio_bias_approx: float  # O(1/N) bias estimate for ratio

    # Within-itinerary dependence diagnostics (NaN unless coords/time supplied).
    # Spatial axis (great-circle meters):
    morans_i_space: float = float("nan")
    morans_i_space_p: float = float("nan")
    variogram_range_m: float = float("nan")
    spatial_corr_ratio: float = float("nan")  # (sill - nugget) / sill
    n_eff_space: float = float("nan")
    # Temporal axis (seconds):
    morans_i_time: float = float("nan")
    morans_i_time_p: float = float("nan")
    variogram_range_s: float = float("nan")
    temporal_corr_ratio: float = float("nan")
    n_eff_time: float = float("nan")
    # Shared: mean within-itinerary vs between-itinerary semivariance ratio.
    within_between_ratio: float = float("nan")


@dataclass
class InferenceResult:
    """Full inference output from ``estimate()``.

    Carries point estimates, SEs, CIs, and diagnostics for both
    the ratio estimand and the photo-level mean.
    """

    # Point estimates
    ratio: float
    photo_mean: float

    # Standard errors
    ratio_se: SEResult
    photo_mean_se: SEResult

    # Confidence intervals
    ratio_ci: CIResult
    photo_mean_ci: CIResult

    # Diagnostics
    diagnostics: Diagnostics

    # Design metadata
    design_name: str
    n_obs: int
    n_clusters: int

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            "=" * 60,
            "geoinference: Inference Result",
            "=" * 60,
            f"Design: {self.design_name}",
            f"Observations: {self.n_obs} ({self.diagnostics.n_positive_frames} with h>0, "
            f"{self.diagnostics.n_empty_frames} empty)",
            f"Clusters: {self.n_clusters}",
            "",
            "── Ratio estimand (people-weighted) ──",
            f"  Estimate:  {self.ratio:.4f}",
            f"  SE:        {self.ratio_se.recommended:.4f}  ({self.ratio_se.method_used})",
            f"  95% CI:    [{self.ratio_ci.recommended[0]:.4f}, "
            f"{self.ratio_ci.recommended[1]:.4f}]",
            "",
            "── Photo-level mean (location-weighted) ──",
            f"  Estimate:  {self.photo_mean:.4f}",
            f"  SE:        {self.photo_mean_se.recommended:.4f}  "
            f"({self.photo_mean_se.method_used})",
            f"  95% CI:    [{self.photo_mean_ci.recommended[0]:.4f}, "
            f"{self.photo_mean_ci.recommended[1]:.4f}]",
            "",
            "── Diagnostics ──",
            f"  ICC:                {self.diagnostics.icc:.4f}",
            f"  Design effect:      {self.diagnostics.deff:.2f}",
            f"  Effective N:        {self.diagnostics.n_eff:.1f}",
            f"  SE ratio (cl/nv):   {self.diagnostics.se_ratio_cluster_to_naive:.3f}",
            f"  Ratio bias O(1/N):  {self.diagnostics.ratio_bias_approx:.6f}",
        ]
        lines += self._dependence_lines()
        lines.append("=" * 60)
        return "\n".join(lines)

    def _dependence_lines(self) -> list[str]:
        """Optional within-itinerary dependence block (only if computed)."""
        d = self.diagnostics
        out: list[str] = []
        if not np.isnan(d.within_between_ratio):
            out.append(f"  Within/between semivar: {d.within_between_ratio:.3f}")
        if not np.isnan(d.n_eff_space):
            out.append(
                f"  Spatial:  range={d.variogram_range_m:,.0f} m  "
                f"I={d.morans_i_space:.3f} (p={d.morans_i_space_p:.3f})  "
                f"n_eff={d.n_eff_space:.1f}"
            )
        if not np.isnan(d.n_eff_time):
            out.append(
                f"  Temporal: range={d.variogram_range_s:,.0f} s  "
                f"I={d.morans_i_time:.3f} (p={d.morans_i_time_p:.3f})  "
                f"n_eff={d.n_eff_time:.1f}"
            )
        if out:
            out = ["", "── Within-itinerary dependence ──", *out]
        return out

    def __repr__(self) -> str:
        return (
            f"InferenceResult(ratio={self.ratio:.4f}, photo_mean={self.photo_mean:.4f}, "
            f"n={self.n_obs}, G={self.n_clusters}, design='{self.design_name}')"
        )

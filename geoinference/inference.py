"""Inference engine for geoinference.

The ``estimate()`` function is the main entry point. It takes annotated
frame data plus a design object and returns point estimates, standard
errors, confidence intervals, and diagnostics.
"""

import warnings
from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from . import spatial
from .designs import Design, PointDesign
from .types import CIResult, Diagnostics, InferenceResult, SEResult

# Defaults for the three numbers that change what `estimate` does. They are
# exposed as arguments rather than kept here as constants because each is a
# judgement call, and a judgement compiled into a module constant is one the
# caller cannot disagree with. These names are the defaults; the decisions belong
# to whoever is doing the analysis.

# Pairwise dependence diagnostics are O(n^2); subsample above this many frames.
DEFAULT_MAX_DEPENDENCE_POINTS = 2500

# Above this coefficient of variation in itinerary size, warn that no interval is
# reliable. Measured rather than conventional. Coverage of a nominal 95% t
# interval at ICC 0.31, 400 replicates per cell, 3-sigma band [0.917, 0.983]:
#
#     G=5  CV 0.00  eff 5.0   0.943      G=20 CV 0.56  eff 15.2  0.927
#     G=10 CV 0.00  eff 10.0  0.932      G=20 CV 0.79  eff 12.5  0.905  out
#     G=20 CV 0.00  eff 20.0  0.955      G=20 CV 0.98  eff 10.6  0.897  out
#
# Note what this rules out: a *balanced* design covers correctly at five
# itineraries, so the number of itineraries is not the predictor and a threshold
# on it -- effective or nominal -- cries wolf on the commonest case. Unequal
# itinerary sizes are what break it, at every G tested, and no choice of
# se_method repairs it. The cut sits between the 0.56 that covers and the 0.79
# that does not.
DEFAULT_WARN_ABOVE_CLUSTER_SIZE_CV = 0.7

# Below this many itineraries, prefer the t_{G-1} interval over the normal for
# cluster-robust standard errors. Pre-existing behaviour, previously the literal
# 30 inside `_build_ci`.
DEFAULT_T_INTERVAL_BELOW_CLUSTERS = 30


# ─── Point estimators ────────────────────────────────────────────────


def _ratio_estimator(w: np.ndarray, h: np.ndarray) -> float:
    """R_hat = sum(w) / sum(h)."""
    total_h = h.sum()
    if total_h == 0:
        return float("nan")
    return float(w.sum() / total_h)


def _photo_mean_estimator(w: np.ndarray, h: np.ndarray) -> float:
    """theta_hat = mean(w_i / h_i) for h_i > 0."""
    mask = h > 0
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(w[mask] / h[mask]))


def _ratio_bias_approx(w: np.ndarray, h: np.ndarray) -> float:
    """O(1/N) bias approximation for the ratio estimator."""
    n = len(w)
    if n < 2 or h.sum() == 0:
        return float("nan")
    r = w.sum() / h.sum()
    eh = np.mean(h)
    vh = np.var(h, ddof=1)
    cwh = np.cov(w, h, ddof=1)[0, 1]
    return float((r * vh - cwh) / (n * eh**2))


# ─── Variance estimators ─────────────────────────────────────────────


def _naive_se_ratio(w: np.ndarray, h: np.ndarray) -> float:
    """Naive SE for ratio estimator (delta method, iid assumption)."""
    n = len(w)
    if n < 2 or h.sum() == 0:
        return float("nan")
    r = w.sum() / h.sum()
    e = w - r * h
    v = np.sum(e**2) / (n - 1) / h.sum() ** 2 * n
    return float(np.sqrt(v))


def _naive_se_mean(w: np.ndarray, h: np.ndarray) -> float:
    """Naive SE for photo-level mean (iid assumption)."""
    mask = h > 0
    if mask.sum() < 2:
        return float("nan")
    p = w[mask] / h[mask]
    return float(np.std(p, ddof=1) / np.sqrt(len(p)))


def _cluster_robust_se_ratio(w: np.ndarray, h: np.ndarray, labels: np.ndarray) -> float:
    """Linearization-based cluster-robust SE for ratio estimator."""
    r = _ratio_estimator(w, h)
    if np.isnan(r) or h.sum() == 0:
        return float("nan")

    total_h = h.sum()
    e = w - r * h

    unique = np.unique(labels)
    g = len(unique)
    if g < 2:
        return float("nan")

    e_g = np.array([e[labels == c].sum() for c in unique])
    e_bar = e_g.mean()

    v = (g / (g - 1)) * np.sum((e_g - e_bar) ** 2) / total_h**2
    return float(np.sqrt(v))


def _cluster_robust_se_mean(w: np.ndarray, h: np.ndarray, labels: np.ndarray) -> float:
    """Cluster-robust SE for photo-level mean."""
    mask = h > 0
    if mask.sum() == 0:
        return float("nan")

    p = np.full(len(w), np.nan)
    p[mask] = w[mask] / h[mask]
    theta = float(np.nanmean(p[mask]))
    m = int(mask.sum())

    unique = np.unique(labels)
    g = len(unique)
    if g < 2:
        return float("nan")

    s_g = np.array(
        [
            np.sum(p[(labels == c) & mask] - theta)
            if np.any((labels == c) & mask)
            else 0.0
            for c in unique
        ]
    )

    v = (g / (g - 1)) * np.sum(s_g**2) / m**2
    return float(np.sqrt(v))


def _wild_ci_from_scores(
    estimate_value: float,
    scores: np.ndarray,
    cluster_normalizers: np.ndarray,
    reps: int,
    ci_level: float,
    seed: int,
) -> tuple[float, float, float]:
    """Percentile-t interval from per-cluster influence scores.

    Every estimand in this module is asymptotically a sum of cluster-level
    scores over a normalizer -- the ratio's are ``sum(w - r*h)`` over
    ``sum(h)``, the photo mean's are ``sum(p_i - theta)`` over the number of
    positive frames. Given those, the wild cluster bootstrap is the same
    procedure regardless of which estimand produced them, which is why it lives
    here once rather than twice.

    Args:
        estimate_value: The point estimate being bracketed.
        scores: One summed influence contribution per cluster.
        cluster_normalizers: Each cluster's own share of the normalizer --
            ``sum(h)`` within the cluster for the ratio, the count of positive
            frames for the photo mean. Their total puts the scores on the
            estimate's scale.
        reps: Rademacher draws.
        ci_level: Coverage the interval claims.
        seed: Seed for the sign draws.

    Returns:
        tuple of float: ``(cluster_robust_se, ci_lo, ci_hi)``, NaN where the
        design cannot support an interval.
    """
    g = scores.size
    normalizer = float(cluster_normalizers.sum())
    if g < 2 or normalizer == 0 or not np.isfinite(estimate_value):
        return float("nan"), float("nan"), float("nan")

    centered = scores - scores.mean()
    v_obs = (g / (g - 1)) * np.sum(centered**2) / normalizer**2
    se = float(np.sqrt(v_obs))
    if se == 0:
        return se, estimate_value, estimate_value

    rng = np.random.default_rng(seed)
    t_star = np.empty(reps)
    valid = 0
    for _ in range(reps):
        weights = rng.choice(np.array([-1.0, 1.0]), size=g)
        drawn = weights * centered
        delta = drawn.sum() / normalizer
        # Residual scores at the bootstrap estimate, which is the observed one
        # shifted by delta. Cluster c's score moves by delta times *its own*
        # normalizer, not by the average -- `drawn - drawn.mean()` subtracts
        # delta * normalizer / g from every cluster, which is the same thing
        # only when every cluster is the same size. Measured wild-interval
        # coverage at nominal 0.95, 400 replicates, as itinerary sizes spread:
        #
        #     size CV     0.0    0.6    1.0    1.5
        #     mean-cent.  0.948  0.927  0.897  0.890
        #     this        0.948  0.940  0.915  0.927
        #
        # Identical when balanced, as the algebra says it must be, and worth
        # up to four points once sizes vary.
        drawn_c = drawn - delta * cluster_normalizers
        v_b = (g / (g - 1)) * np.sum(drawn_c**2) / normalizer**2
        if v_b > 0:
            t_star[valid] = delta / np.sqrt(v_b)
            valid += 1

    if valid < reps * 0.5:
        return se, float("nan"), float("nan")

    t_star = t_star[:valid]
    alpha = 1 - ci_level
    q_lo = float(np.percentile(t_star, 100 * (alpha / 2)))
    q_hi = float(np.percentile(t_star, 100 * (1 - alpha / 2)))
    # Percentile-t: invert t = (est_hat - est) / se.
    return se, estimate_value - se * q_hi, estimate_value - se * q_lo


def _mean_cluster_scores(
    w: np.ndarray, h: np.ndarray, labels: np.ndarray
) -> tuple[float, np.ndarray, np.ndarray]:
    """Photo-mean point estimate, per-cluster scores and normalizer.

    Mirrors :func:`_cluster_robust_se_mean` exactly, so the wild interval and the
    analytic SE are built from the same quantities.

    Args:
        w: Women counts per frame.
        h: People counts per frame.
        labels: Cluster label per frame.

    Returns:
        tuple: ``(theta, scores, counts)`` where counts holds each cluster's
        number of positive frames; scores is empty when undefined.
    """
    mask = h > 0
    if mask.sum() == 0:
        return float("nan"), np.array([]), np.array([])

    p = np.full(len(w), np.nan)
    p[mask] = w[mask] / h[mask]
    theta = float(np.nanmean(p[mask]))

    unique = np.unique(labels)
    scores = np.array(
        [
            np.sum(p[(labels == c) & mask] - theta)
            if np.any((labels == c) & mask)
            else 0.0
            for c in unique
        ]
    )
    counts = np.array([np.sum((labels == c) & mask) for c in unique], dtype=float)
    return theta, scores, counts


def wild_cluster_bootstrap_ci(
    w: np.ndarray,
    h: np.ndarray,
    labels: np.ndarray,
    reps: int = 999,
    ci_level: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Wild cluster bootstrap (percentile-t) CI for the ratio estimator.

    The pairs/cluster bootstrap and the analytic cluster-robust t both
    under-cover when the number of itineraries G is small. The wild cluster
    bootstrap (Cameron, Gelbach & Miller 2008) perturbs each cluster's
    linearized score by a Rademacher (+/-1) weight and studentizes, which
    restores near-nominal coverage with few clusters.

    Args:
        w: Per-observation women counts.
        h: Per-observation people counts.
        labels: Cluster labels.
        reps: Bootstrap replicates.
        ci_level: Coverage the interval claims.
        seed: Source of the Rademacher draws.

    Returns:
        (cluster_robust_se, ci_lo, ci_hi). NaNs if G < 2.
    """
    r = _ratio_estimator(w, h)
    total_h = float(h.sum())
    if np.isnan(r) or total_h == 0:
        return float("nan"), float("nan"), float("nan")

    unique = np.unique(labels)
    e = w - r * h
    scores = np.array([e[labels == c].sum() for c in unique])
    cluster_h = np.array([h[labels == c].sum() for c in unique], dtype=float)
    return _wild_ci_from_scores(r, scores, cluster_h, reps, ci_level, seed)


def wild_cluster_bootstrap_ci_mean(
    w: np.ndarray,
    h: np.ndarray,
    labels: np.ndarray,
    reps: int = 999,
    ci_level: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Wild cluster bootstrap (percentile-t) CI for the photo-level mean.

    The companion to :func:`wild_cluster_bootstrap_ci`. Both estimands get the
    same interval procedure, so ``ratio_ci.wild`` and ``photo_mean_ci.wild`` mean
    the same thing rather than one being a percentile-t and the other a raw
    percentile.

    Args:
        w: Women counts per frame.
        h: People counts per frame.
        labels: Itinerary label per frame.
        reps: Rademacher draws.
        ci_level: Coverage the interval claims.
        seed: Seed for the sign draws.

    Returns:
        tuple of float: ``(cluster_robust_se, ci_lo, ci_hi)``. NaNs if G < 2.
    """
    theta, scores, counts = _mean_cluster_scores(w, h, labels)
    return _wild_ci_from_scores(theta, scores, counts, reps, ci_level, seed)


def _cluster_bootstrap(
    w: np.ndarray,
    h: np.ndarray,
    labels: np.ndarray,
    estimator_fn: Callable[[np.ndarray, np.ndarray], float],
    reps: int = 2000,
    ci_level: float = 0.95,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float]:
    """Cluster bootstrap: resample clusters, return (se, ci_lo, ci_hi).

    Args:
        w: Per-observation weights.
        h: Per-observation counts.
        labels: Cluster labels.
        estimator_fn: Maps ``(w, h)`` to the statistic.
        reps: Bootstrap replicates.
        ci_level: Coverage the interval claims. This parameter did not exist and
            the percentiles were hardcoded at 2.5 and 97.5, so asking
            :func:`estimate` for a 90% interval returned a 95% one here while
            the wild bootstrap beside it honoured the request -- two intervals
            at different levels in one result object, and nothing said so.
        rng: Source of randomness.

    Returns:
        tuple: Standard error, lower bound, upper bound.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    unique = np.unique(labels)
    g = len(unique)

    cluster_data = {}
    for c in unique:
        idx = labels == c
        cluster_data[c] = (w[idx], h[idx])

    estimates = np.empty(reps)
    for b in range(reps):
        resampled = rng.choice(unique, size=g, replace=True)
        w_b = np.concatenate([cluster_data[c][0] for c in resampled])
        h_b = np.concatenate([cluster_data[c][1] for c in resampled])
        estimates[b] = estimator_fn(w_b, h_b)

    valid = estimates[~np.isnan(estimates)]
    if len(valid) < reps * 0.5:
        return float("nan"), float("nan"), float("nan")

    se = float(np.std(valid, ddof=1))
    alpha = 1 - ci_level
    lo = float(np.percentile(valid, 100 * alpha / 2))
    hi = float(np.percentile(valid, 100 * (1 - alpha / 2)))
    return se, lo, hi


# ─── ICC and design effect ───────────────────────────────────────────


def _compute_icc(values: np.ndarray, labels: np.ndarray) -> float:
    """Intraclass correlation within clusters."""
    unique = np.unique(labels)
    g = len(unique)
    n = len(values)

    if g < 2 or n < 3:
        return float("nan")

    grand_mean = values.mean()
    ssb = sum(
        np.sum(labels == c) * (values[labels == c].mean() - grand_mean) ** 2
        for c in unique
    )
    ssw = sum(
        np.sum((values[labels == c] - values[labels == c].mean()) ** 2) for c in unique
    )

    msb = ssb / (g - 1)
    msw = ssw / (n - g) if n > g else 0.0

    m = np.array([np.sum(labels == c) for c in unique], dtype=float)
    m0 = (n - np.sum(m**2) / n) / (g - 1)

    denom = msb + (m0 - 1) * msw
    if denom == 0:
        return 0.0

    icc = (msb - msw) / denom
    return float(max(icc, 0.0))


# ─── Within-itinerary dependence diagnostics ─────────────────────────


def _require_columns(data: pd.DataFrame, cols: list[str]) -> None:
    """Raise ValueError listing any of ``cols`` absent from ``data``."""
    missing = [c for c in cols if c not in data.columns]
    if missing:
        raise ValueError(
            f"Columns not found in data: {missing}. "
            f"Available columns: {list(data.columns)}"
        )


def _axis_diagnostics(
    values: np.ndarray, dist: np.ndarray, seed: int
) -> dict[str, float]:
    """Variogram range, correlation ratio, effective N, and Moran's I for one axis."""
    lags, gamma, counts = spatial.empirical_variogram(values, dist)
    c0, c1, rng_ = spatial.fit_variogram(lags, gamma, counts)
    corr_ratio = (c1 - c0) / c1 if np.isfinite(c1) and c1 > 0 else float("nan")
    n_eff = spatial.effective_n(values, dist, c0, c1, rng_)
    if np.isfinite(rng_) and rng_ > 0:
        cutoff = rng_
    else:
        pos = dist[dist > 0]
        cutoff = float(np.median(pos)) if pos.size else 0.0
    mi, mp = spatial.morans_i(values, dist, cutoff, seed=seed)
    return {
        "range": rng_,
        "corr_ratio": corr_ratio,
        "n_eff": n_eff,
        "morans_i": mi,
        "morans_i_p": mp,
    }


def _dependence_diagnostics(
    data: pd.DataFrame,
    mask: np.ndarray,
    p_obs: np.ndarray,
    labels_pos: np.ndarray,
    lon_var: str | None,
    lat_var: str | None,
    time_var: str | None,
    seed: int,
    max_dependence_points: int = DEFAULT_MAX_DEPENDENCE_POINTS,
) -> dict[str, float]:
    """Spatial/temporal within-itinerary dependence on the h>0 frames."""
    out: dict[str, float] = {}
    n = len(p_obs)
    if n < 3:
        return out

    idx = np.arange(n)
    if n > max_dependence_points:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(n, size=max_dependence_points, replace=False))
        warnings.warn(
            f"Dependence diagnostics subsampled to {max_dependence_points} of "
            f"{n} positive frames (pairwise cost is O(n^2)).",
            stacklevel=2,
        )

    vals = p_obs[idx]
    sub_labels = labels_pos[idx]

    if len(np.unique(sub_labels)) >= 2:
        wb = spatial.within_between_contrast(vals, sub_labels)
        out["within_between_ratio"] = wb["ratio"]

    if lon_var is not None and lat_var is not None:
        _require_columns(data, [lon_var, lat_var])
        lon = data[lon_var].to_numpy(dtype=float)[mask][idx]
        lat = data[lat_var].to_numpy(dtype=float)[mask][idx]
        sp = _axis_diagnostics(vals, spatial.haversine_matrix(lon, lat), seed)
        out["variogram_range_m"] = sp["range"]
        out["spatial_corr_ratio"] = sp["corr_ratio"]
        out["n_eff_space"] = sp["n_eff"]
        out["morans_i_space"] = sp["morans_i"]
        out["morans_i_space_p"] = sp["morans_i_p"]

    if time_var is not None:
        _require_columns(data, [time_var])
        ts = data[time_var].to_numpy()[mask][idx]
        tp = _axis_diagnostics(vals, spatial.time_gap_matrix(ts), seed)
        out["variogram_range_s"] = tp["range"]
        out["temporal_corr_ratio"] = tp["corr_ratio"]
        out["n_eff_time"] = tp["n_eff"]
        out["morans_i_time"] = tp["morans_i"]
        out["morans_i_time_p"] = tp["morans_i_p"]

    return out


# ─── Confidence interval construction ────────────────────────────────


def _build_ci(
    est: float,
    se: float,
    g: int,
    level: float = 0.95,
    boot_ci: tuple[float, float] | None = None,
    wild_ci: tuple[float, float] | None = None,
    recommended_method: str = "naive",
    t_interval_below_clusters: int = DEFAULT_T_INTERVAL_BELOW_CLUSTERS,
    ci_method: str | None = None,
) -> CIResult:
    """Construct every interval, and mark the one asked for.

    Args:
        est: Point estimate.
        se: Standard error to build the normal and t intervals from.
        g: Number of clusters.
        level: Coverage the intervals claim.
        boot_ci: Pairs cluster bootstrap percentile interval, if computed.
        wild_ci: Wild cluster bootstrap percentile-t interval, if computed.
        recommended_method: The SE method the design selected, used only when
            ``ci_method`` is None.
        t_interval_below_clusters: Prefer t over normal below this many clusters.
        ci_method: Explicit choice of ``"normal"``, ``"t"``, ``"bootstrap"`` or
            ``"wild"``. None follows the design.

    Returns:
        CIResult: All four intervals plus the selected one.

    Raises:
        ValueError: If ``ci_method`` names an interval that does not exist.
    """
    alpha = 1 - level
    z = -sp_stats.norm.ppf(alpha / 2)

    normal_ci = (est - z * se, est + z * se)

    if g >= 2:
        t_cv = -sp_stats.t.ppf(alpha / 2, df=g - 1)
        t_ci = (est - t_cv * se, est + t_cv * se)
    else:
        t_ci = (float("nan"), float("nan"))

    available = {"normal": normal_ci, "t": t_ci, "bootstrap": boot_ci, "wild": wild_ci}

    if ci_method is not None:
        if ci_method not in available:
            raise ValueError(
                f"ci_method must be one of {sorted(available)}, got {ci_method!r}"
            )
        chosen = available[ci_method]
        if chosen is None:
            raise ValueError(
                f"ci_method={ci_method!r} was requested but that interval was not "
                "computed; pass bootstrap=True to enable the bootstrap intervals."
            )
        recommended = chosen
    elif recommended_method == "cluster" and g < t_interval_below_clusters:
        recommended = t_ci
    elif boot_ci is not None and recommended_method == "bootstrap":
        recommended = boot_ci
    else:
        recommended = normal_ci

    return CIResult(
        normal=normal_ci,
        t=t_ci,
        bootstrap=boot_ci,
        wild=wild_ci,
        recommended=recommended,
        level=level,
    )


# ─── Main entry point ────────────────────────────────────────────────


def estimate(
    data: pd.DataFrame,
    women_var: str = "n_women",
    people_var: str = "n_people",
    design: Design | None = None,
    ci_level: float = 0.95,
    bootstrap: bool = True,
    bootstrap_reps: int = 2000,
    seed: int = 42,
    lon_var: str | None = None,
    lat_var: str | None = None,
    time_var: str | None = None,
    se_method: str | None = None,
    ci_method: str | None = None,
    warn_above_cluster_size_cv: float | None = DEFAULT_WARN_ABOVE_CLUSTER_SIZE_CV,
    t_interval_below_clusters: int = DEFAULT_T_INTERVAL_BELOW_CLUSTERS,
    max_dependence_points: int = DEFAULT_MAX_DEPENDENCE_POINTS,
) -> InferenceResult:
    """Estimate population gender ratio with correct standard errors.

    This is the main inference function. It takes annotated frame-level
    data, a design specification, and produces point estimates, standard
    errors, confidence intervals, and diagnostics for two estimands:

    - **Ratio** (people-weighted): sum(women) / sum(people)
    - **Photo-level mean** (location-weighted): mean(w_i / h_i) for h_i > 0

    The design object determines which SE estimator is primary. Under
    SRS with post-hoc itinerary clustering, the naive SE is recommended
    (observations are approximately independent). Under walk designs,
    the between-walk SE is recommended. Under PPS/GRTS, the cluster-robust
    (Horvitz-Thompson linearization) SE is recommended.

    Args:
        data: DataFrame with one row per annotated frame.
        women_var: Column with women count per frame.
        people_var: Column with total people count per frame.
        design: Design object (PointDesign or WalkDesign). If None,
            defaults to PointDesign(sampling="srs") with no clustering.
        ci_level: Confidence level for intervals (default 0.95).
        bootstrap: Whether to compute cluster bootstrap CIs.
        bootstrap_reps: Number of bootstrap replications.
        seed: Random seed for bootstrap and the dependence diagnostics.
        lon_var: Column with longitude (decimal degrees). If given together
            with ``lat_var``, the spatial within-itinerary dependence
            diagnostics (variogram range, Moran's I, effective spatial N) are
            computed on the h>0 frames.
        lat_var: Column with latitude (decimal degrees). See ``lon_var``.
        time_var: Column with a per-frame timestamp (datetime or epoch
            seconds). If given, the temporal within-itinerary dependence
            diagnostics are computed. Independent of the spatial axis: supply
            either, both, or neither.
        se_method: Which standard error ``recommended`` holds: ``"naive"``,
            ``"cluster"`` or ``"bootstrap"``. None lets the design choose, which
            means cluster-robust whenever a cluster variable was given.

            *How to reason about it.* Use ``"cluster"`` unless you can argue that
            frames within an itinerary are independent -- check
            ``diagnostics.se_ratio_cluster_to_naive``, which sits near 1 when
            they are and at 2-3 when they are not. ``"naive"`` is right only for
            a design with no clustering at all. ``"bootstrap"`` assumes less
            about the shape of the sampling distribution but needs itineraries to
            resample, so it is the weakest of the three when you have few. There
            is deliberately no ``"wild"``: that method produces an interval, not
            a standard error. See ``ci_method``.
        ci_method: Which interval ``recommended`` holds: ``"normal"``, ``"t"``,
            ``"bootstrap"`` or ``"wild"``. None follows the design, which prefers
            ``"t"`` for clustered data below ``t_interval_below_clusters``
            itineraries.

            *How to reason about it.* The number of itineraries decides this.
            Coverage of a nominal 95% interval, balanced design at ICC 0.31 over
            400 replicates::

                G          5      8     12     20     40
                normal     0.868  0.912 0.927  0.940  0.940
                t          0.943  0.950 0.943  0.955  0.943
                bootstrap  0.815  0.887 0.920  0.932  0.932
                wild       0.932  0.953 0.940  0.955  0.943

            ``"t"`` and ``"wild"`` hold up throughout; ``"normal"`` and
            ``"bootstrap"`` under-cover below about a dozen itineraries. Choose
            ``"wild"`` for an interval that does not lean on the t approximation,
            and ``"normal"`` only when itineraries are plentiful.
        warn_above_cluster_size_cv: Warn when itinerary sizes vary more than this
            (coefficient of variation). None silences it;
            ``diagnostics.cluster_size_cv`` and ``diagnostics.n_clusters_eff``
            are reported either way, so silencing the opinion does not cost you
            the numbers.

            *How to reason about it.* This is the knob that says "no interval
            here is trustworthy", so it is worth knowing what drives it. Coverage
            of a nominal 95% t interval at ICC 0.31 over 400 replicates was
            0.943 / 0.932 / 0.955 for *balanced* designs of 5, 10 and 20
            itineraries, then 0.927 / 0.905 / 0.897 at 20 itineraries as the size
            CV rose through 0.56 / 0.79 / 0.98. A balanced design covers
            correctly with five itineraries, so the count is not the predictor --
            the imbalance is, and the remedy is more even itinerary lengths
            rather than more itineraries. Raise it if you accept the coverage
            loss; lower it to be told sooner.
        t_interval_below_clusters: Below this many itineraries, prefer t_{G-1}
            over the normal for cluster-robust SEs.

            *How to reason about it.* The t correction widens the interval by
            t_{G-1}/1.96 -- 15% at ten itineraries, 2% at sixty -- so above a few
            dozen the choice makes no practical difference and the threshold is
            arbitrary. Below ten it matters, and the table under ``ci_method`` is
            the evidence: the normal interval covers 0.868 at five itineraries
            where t covers 0.943.
        max_dependence_points: Subsample the spatial and temporal dependence
            diagnostics to this many frames.

            *How to reason about it.* Cost, not correctness. The pairwise
            distance matrix is O(n^2), so ten thousand frames is a hundred
            million pairs. Raise it to compute the diagnostics on everything if
            you can wait; the estimates and intervals never depend on it.
            Subsampling warns when it happens, so you will know it did.

    Returns:
        InferenceResult with estimates, SEs, CIs, and diagnostics. The spatial
        and temporal dependence fields of ``diagnostics`` are NaN unless the
        corresponding coordinate/time columns are supplied.

    Raises:
        ValueError: If a named column is absent, or an argument is outside
            the range the estimator accepts.

    Example:
        >>> from geoinference import PointDesign, estimate
        >>> design = PointDesign(sampling="srs", cluster_var="itinerary_id")
        >>> result = estimate(df, "n_women", "n_people", design=design)
        >>> print(result.summary())
    """
    if design is None:
        design = PointDesign(sampling="srs")

    # ── Extract arrays ────────────────────────────────────────────
    w = data[women_var].to_numpy(dtype=float)
    h = data[people_var].to_numpy(dtype=float)
    n = len(w)

    # Cluster labels
    if design.has_clusters:
        cvar = design.cluster_var
        if cvar not in data.columns:
            raise ValueError(
                f"Cluster variable '{cvar}' not found in data. "
                f"Available columns: {list(data.columns)}"
            )
        labels = data[cvar].to_numpy()
    else:
        # Each observation is its own cluster
        labels = np.arange(n)

    unique_clusters = np.unique(labels)
    g = len(unique_clusters)

    # ── Point estimates ───────────────────────────────────────────
    ratio = _ratio_estimator(w, h)
    photo_mean = _photo_mean_estimator(w, h)

    # ── Standard errors ───────────────────────────────────────────
    se_ratio_naive = _naive_se_ratio(w, h)
    se_ratio_cluster = _cluster_robust_se_ratio(w, h, labels)
    se_mean_naive = _naive_se_mean(w, h)
    se_mean_cluster = _cluster_robust_se_mean(w, h, labels)

    rng = np.random.default_rng(seed)
    se_ratio_boot, boot_ratio_lo, boot_ratio_hi = (None, None, None)
    se_mean_boot, boot_mean_lo, boot_mean_hi = (None, None, None)
    wild_ratio_ci: tuple[float, float] | None = None
    wild_mean_ci: tuple[float, float] | None = None

    if bootstrap and g >= 3:
        # Both estimands get both bootstraps, so a field means the same thing
        # wherever you find it. The pairs bootstrap supplies a genuine standard
        # error -- the spread of the resampled estimates -- and a percentile
        # interval. The wild cluster bootstrap supplies only an interval: it is a
        # percentile-t procedure, studentising by the analytic cluster-robust SE,
        # so its "SE" would just be `cluster` under a second name.
        #
        # Prefer the wild interval when you have few itineraries. Coverage of a
        # nominal 95% interval, balanced, ICC 0.31, 400 replicates per cell:
        #
        #     G      5      8     12     20     40
        #     pairs  0.815  0.887 0.920  0.932  0.932
        #     wild   0.932  0.953 0.940  0.955  0.943
        #
        # The pairs bootstrap resamples G itineraries with replacement from G, so
        # only about 63% appear in a given draw and each replicate carries less
        # between-itinerary variation than the design does; nor is it
        # studentised, so it gets no small-sample refinement.
        se_ratio_boot, boot_ratio_lo, boot_ratio_hi = _cluster_bootstrap(
            w,
            h,
            labels,
            _ratio_estimator,
            reps=bootstrap_reps,
            ci_level=ci_level,
            rng=rng,
        )
        se_mean_boot, boot_mean_lo, boot_mean_hi = _cluster_bootstrap(
            w,
            h,
            labels,
            _photo_mean_estimator,
            reps=bootstrap_reps,
            ci_level=ci_level,
            rng=rng,
        )

        _, wild_r_lo, wild_r_hi = wild_cluster_bootstrap_ci(
            w, h, labels, reps=bootstrap_reps, ci_level=ci_level, seed=seed
        )
        _, wild_m_lo, wild_m_hi = wild_cluster_bootstrap_ci_mean(
            w, h, labels, reps=bootstrap_reps, ci_level=ci_level, seed=seed
        )
        if not np.isnan(wild_r_lo):
            wild_ratio_ci = (wild_r_lo, wild_r_hi)
        if not np.isnan(wild_m_lo):
            wild_mean_ci = (wild_m_lo, wild_m_hi)

    # ── Select recommended SE ─────────────────────────────────────
    rec_method = se_method if se_method is not None else design.recommended_se_method
    if rec_method not in ("naive", "cluster", "bootstrap"):
        raise ValueError(
            "se_method must be 'naive', 'cluster' or 'bootstrap', got "
            f"{se_method!r}. The wild cluster bootstrap produces an interval "
            "rather than a standard error; ask for it with ci_method='wild'."
        )

    def _pick_se(
        naive: float, cluster: float, boot: float | None, method: str
    ) -> SEResult:
        if method == "naive":
            recommended = naive
        elif method == "cluster":
            recommended = cluster
        elif method == "bootstrap" and boot is not None:
            recommended = boot
        else:
            recommended = naive
        return SEResult(
            naive=naive,
            cluster=cluster,
            bootstrap=boot,
            recommended=recommended,
            method_used=method,
        )

    ratio_se = _pick_se(se_ratio_naive, se_ratio_cluster, se_ratio_boot, rec_method)
    mean_se = _pick_se(se_mean_naive, se_mean_cluster, se_mean_boot, rec_method)

    # ── Confidence intervals ──────────────────────────────────────
    boot_ratio_ci = (
        (boot_ratio_lo, boot_ratio_hi)
        if boot_ratio_lo is not None and boot_ratio_hi is not None
        else None
    )
    boot_mean_ci = (
        (boot_mean_lo, boot_mean_hi)
        if boot_mean_lo is not None and boot_mean_hi is not None
        else None
    )

    ratio_ci = _build_ci(
        ratio,
        ratio_se.recommended,
        g,
        ci_level,
        boot_ci=boot_ratio_ci,
        wild_ci=wild_ratio_ci,
        recommended_method=rec_method,
        t_interval_below_clusters=t_interval_below_clusters,
        ci_method=ci_method,
    )
    mean_ci = _build_ci(
        photo_mean,
        mean_se.recommended,
        g,
        ci_level,
        boot_ci=boot_mean_ci,
        wild_ci=wild_mean_ci,
        recommended_method=rec_method,
        t_interval_below_clusters=t_interval_below_clusters,
        ci_method=ci_method,
    )

    # ── Diagnostics ───────────────────────────────────────────────
    mask = h > 0
    n_positive = int(mask.sum())
    n_empty = int((~mask).sum())

    p_obs = w[mask] / h[mask] if n_positive > 0 else np.array([])
    lab_pos = labels[mask] if n_positive > 0 else np.array([])

    icc = _compute_icc(p_obs, lab_pos) if n_positive > 2 else float("nan")
    cluster_sizes = np.array([np.sum(labels == c) for c in unique_clusters])
    m_bar = float(cluster_sizes.mean()) if len(cluster_sizes) > 0 else 0.0
    deff = 1 + (m_bar - 1) * icc if not np.isnan(icc) else float("nan")
    n_eff = n_positive / deff if deff > 0 and not np.isnan(deff) else float(n_positive)

    # Kish effective number of itineraries. Every cluster-robust interval here
    # estimates between-itinerary variance from this many units, not from
    # `n_clusters`, and unequal itinerary sizes drive the two far apart: 20
    # itineraries where one holds 68% of the frames behave like 2.1.
    n_clusters_eff = (
        float(cluster_sizes.sum() ** 2 / np.sum(cluster_sizes.astype(float) ** 2))
        if cluster_sizes.size > 0 and cluster_sizes.sum() > 0
        else 0.0
    )
    size_cv = float(cluster_sizes.std() / cluster_sizes.mean()) if m_bar > 0 else 0.0
    if (
        warn_above_cluster_size_cv is not None
        and g >= 2
        and size_cv > warn_above_cluster_size_cv
    ):
        warnings.warn(
            f"Itinerary sizes vary a lot: coefficient of variation {size_cv:.2f} "
            f"across {g} itineraries, leaving {n_clusters_eff:.1f} effective. In "
            "simulation every interval this function reports -- normal, t, and "
            "both bootstraps -- under-covered at this much imbalance, and no "
            "choice of se_method repaired it. Treat the interval as indicative. "
            "A balanced design covers correctly even with five itineraries, so "
            "the fix is more even itinerary lengths rather than more itineraries.",
            UserWarning,
            stacklevel=2,
        )

    se_ratio_cn = (
        se_mean_cluster / se_mean_naive
        if se_mean_naive > 0 and not np.isnan(se_mean_cluster)
        else float("nan")
    )

    dep: dict[str, float] = {}
    if n_positive > 2 and (lon_var is not None or time_var is not None):
        dep = _dependence_diagnostics(
            data,
            mask,
            p_obs,
            lab_pos,
            lon_var,
            lat_var,
            time_var,
            seed,
            max_dependence_points=max_dependence_points,
        )

    def _dep(key: str) -> float:
        return dep.get(key, float("nan"))

    diagnostics = Diagnostics(
        n_obs=n,
        n_positive_frames=n_positive,
        n_empty_frames=n_empty,
        empty_frame_rate=n_empty / n if n > 0 else 0.0,
        n_clusters=g,
        cluster_sizes=cluster_sizes,
        cluster_size_mean=m_bar,
        cluster_size_cv=(
            float(cluster_sizes.std() / cluster_sizes.mean()) if m_bar > 0 else 0.0
        ),
        icc=icc,
        deff=deff,
        n_eff=n_eff,
        n_clusters_eff=n_clusters_eff,
        se_ratio_cluster_to_naive=se_ratio_cn,
        ratio_bias_approx=_ratio_bias_approx(w, h),
        morans_i_space=_dep("morans_i_space"),
        morans_i_space_p=_dep("morans_i_space_p"),
        variogram_range_m=_dep("variogram_range_m"),
        spatial_corr_ratio=_dep("spatial_corr_ratio"),
        n_eff_space=_dep("n_eff_space"),
        morans_i_time=_dep("morans_i_time"),
        morans_i_time_p=_dep("morans_i_time_p"),
        variogram_range_s=_dep("variogram_range_s"),
        temporal_corr_ratio=_dep("temporal_corr_ratio"),
        n_eff_time=_dep("n_eff_time"),
        within_between_ratio=_dep("within_between_ratio"),
    )

    return InferenceResult(
        ratio=ratio,
        photo_mean=photo_mean,
        ratio_se=ratio_se,
        photo_mean_se=mean_se,
        ratio_ci=ratio_ci,
        photo_mean_ci=mean_ci,
        diagnostics=diagnostics,
        design_name=design.name,
        n_obs=n,
        n_clusters=g,
    )

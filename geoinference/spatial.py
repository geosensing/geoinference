"""Spatial and temporal dependence diagnostics for geoinference.

These functions measure how much within-itinerary correlation there is and
how fast it decays along an axis — geographic distance, time gap, or any
other pairwise-distance matrix. They are *diagnostics*: they describe the
dependence structure (and how much information itineraries cost), they do not
replace the standard-error estimators in ``inference.py``.

The core is axis-agnostic. ``empirical_variogram``, ``morans_i``, and
``effective_n`` all take a precomputed n×n distance matrix, so the same code
serves the spatial axis (``haversine_matrix``), the temporal axis
(``time_gap_matrix``), and any future axis (e.g. same/different enumerator).

References:
    Griffith, D.A. (2005). Effective geographic sample size in the presence
        of spatial autocorrelation.
    Watson, P.A. (2021). A note on the variogram-based effective sample size.
        J. Applied Statistics.
"""

import numpy as np
import pandas as pd
from scipy import optimize

# Mean Earth radius (meters), for great-circle distances.
_EARTH_RADIUS_M = 6_371_000.0


# ─── Pairwise distance matrices (the "axes") ─────────────────────────


def haversine_matrix(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """Pairwise great-circle distances in meters between lon/lat points.

    Args:
        lon: Longitudes in decimal degrees.
        lat: Latitudes in decimal degrees.

    Returns:
        n×n symmetric matrix of distances in meters (zero diagonal).
    """
    lon_r = np.radians(np.asarray(lon, dtype=float))
    lat_r = np.radians(np.asarray(lat, dtype=float))
    dlon = lon_r[:, None] - lon_r[None, :]
    dlat = lat_r[:, None] - lat_r[None, :]
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat_r)[:, None] * np.cos(lat_r)[None, :] * np.sin(dlon / 2.0) ** 2
    )
    a = np.clip(a, 0.0, 1.0)
    out: np.ndarray = 2.0 * _EARTH_RADIUS_M * np.arcsin(np.sqrt(a))
    return out


def _to_epoch_seconds(timestamps: np.ndarray) -> np.ndarray:
    """Convert timestamps to float epoch seconds.

    Accepts numeric epoch seconds (returned as-is) or anything pandas can
    parse to datetimes (datetime64, Timestamps, ISO strings).
    """
    ts = np.asarray(timestamps)
    if np.issubdtype(ts.dtype, np.number):
        return ts.astype(float)
    # Force nanosecond resolution before the integer cast: to_numpy() can return
    # datetime64 at us/ms/s depending on the source, which would mis-scale gaps.
    ns = pd.to_datetime(ts).to_numpy().astype("datetime64[ns]").astype("int64")
    out: np.ndarray = ns.astype(float) / 1e9
    return out


def time_gap_matrix(timestamps: np.ndarray) -> np.ndarray:
    """Pairwise absolute time differences in seconds.

    Args:
        timestamps: datetime64 / Timestamps / ISO strings, or numeric epoch
            seconds.

    Returns:
        n×n symmetric matrix of absolute time gaps in seconds (zero diagonal).
    """
    secs = _to_epoch_seconds(timestamps)
    out: np.ndarray = np.abs(secs[:, None] - secs[None, :])
    return out


# ─── Empirical variogram and exponential fit ─────────────────────────


def empirical_variogram(
    values: np.ndarray,
    dist: np.ndarray,
    n_bins: int = 15,
    max_dist: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Classical (Matheron) empirical semivariogram over distance bins.

    Semivariance per bin is the mean of ``0.5 * (z_i - z_j)**2`` over unique
    pairs whose distance falls in the bin.

    Args:
        values: Length-n outcome array.
        dist: n×n pairwise-distance matrix (any axis).
        n_bins: Number of distance bins.
        max_dist: Maximum distance to include. Defaults to half the maximum
            pairwise distance (the conventional cutoff).

    Returns:
        (lag_centers, semivariance, pair_counts) for non-empty bins only.
    """
    values = np.asarray(values, dtype=float)
    n = len(values)
    iu, ju = np.triu_indices(n, k=1)
    d = dist[iu, ju]
    sq = 0.5 * (values[iu] - values[ju]) ** 2

    if max_dist is None:
        max_dist = float(d.max()) / 2.0 if d.size else 0.0

    keep = (d > 0) & (d <= max_dist)
    d = d[keep]
    sq = sq[keep]
    if d.size == 0:
        empty = np.array([])
        return empty, empty, empty

    edges = np.linspace(0.0, max_dist, n_bins + 1)
    idx = np.clip(np.digitize(d, edges) - 1, 0, n_bins - 1)

    lags = np.full(n_bins, np.nan)
    gamma = np.full(n_bins, np.nan)
    counts = np.zeros(n_bins)
    for b in range(n_bins):
        sel = idx == b
        c = int(sel.sum())
        counts[b] = c
        if c > 0:
            lags[b] = float(d[sel].mean())
            gamma[b] = float(sq[sel].mean())

    nonempty = counts > 0
    return lags[nonempty], gamma[nonempty], counts[nonempty]


def _exponential_model(
    h: np.ndarray, nugget: float, partial_sill: float, rng_: float
) -> np.ndarray:
    """Exponential variogram: nugget + partial_sill * (1 - exp(-h/range))."""
    return nugget + partial_sill * (1.0 - np.exp(-h / rng_))


def fit_variogram(
    lags: np.ndarray,
    gamma: np.ndarray,
    counts: np.ndarray,
    model: str = "exponential",
) -> tuple[float, float, float]:
    """Weighted least-squares fit of an exponential variogram.

    The fitted correlation function is ``rho(h) = ((C1-C0)/C1) * exp(-h/r)``,
    so ``r`` is the e-folding scale (correlation falls to 1/e at h = r); the
    conventional "effective range" is ~3r.

    Args:
        lags: Bin-center distances from ``empirical_variogram``.
        gamma: Semivariances from ``empirical_variogram``.
        counts: Pair counts per bin (used as fit weights).
        model: Only ``"exponential"`` is supported.

    Returns:
        (nugget C0, sill C1, range r). NaNs if the fit cannot be performed.

    Raises:
        ValueError: If ``model`` names anything but ``"exponential"``.
    """
    if model != "exponential":
        raise ValueError(f"Unsupported variogram model: {model!r}")

    lags = np.asarray(lags, dtype=float)
    gamma = np.asarray(gamma, dtype=float)
    counts = np.asarray(counts, dtype=float)
    nan = (float("nan"), float("nan"), float("nan"))
    if lags.size < 3:
        return nan

    gmax = float(np.nanmax(gamma))
    lmax = float(np.nanmax(lags))
    if not np.isfinite(gmax) or gmax <= 0 or lmax <= 0:
        return nan

    p0 = [0.0, gmax, lmax / 3.0]
    bounds = ([0.0, 0.0, 1e-9], [gmax + 1e-12, 5.0 * gmax + 1e-9, 10.0 * lmax])
    sigma = 1.0 / np.sqrt(np.maximum(counts, 1.0))
    try:
        popt, _ = optimize.curve_fit(
            _exponential_model,
            lags,
            gamma,
            p0=p0,
            sigma=sigma,
            bounds=bounds,
            maxfev=10_000,
        )
    except (RuntimeError, ValueError):
        return nan

    nugget = float(popt[0])
    sill = float(popt[0] + popt[1])  # C1 = nugget + partial sill
    rng_ = float(popt[2])
    return nugget, sill, rng_


# ─── Moran's I and effective sample size ──────────────────────────────


def morans_i(
    values: np.ndarray,
    dist: np.ndarray,
    cutoff: float,
    n_perm: int = 999,
    seed: int = 0,
) -> tuple[float, float]:
    """Global Moran's I with binary distance-cutoff weights.

    Weight ``w_ij = 1`` if ``0 < dist_ij <= cutoff`` else 0. The p-value is a
    two-sided permutation test on the observed statistic.

    Args:
        values: Length-n outcome array.
        dist: n×n pairwise-distance matrix.
        cutoff: Neighbor distance threshold (same units as ``dist``).
        n_perm: Number of permutations for the p-value.
        seed: RNG seed for the permutation test.

    Returns:
        (I, p_value). NaNs if undefined (e.g. no neighbor pairs).
    """
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n < 3 or not np.isfinite(cutoff) or cutoff <= 0:
        return float("nan"), float("nan")

    w = ((dist > 0) & (dist <= cutoff)).astype(float)
    np.fill_diagonal(w, 0.0)
    s0 = w.sum()
    if s0 == 0:
        return float("nan"), float("nan")

    z = values - values.mean()
    denom = float(np.sum(z**2))
    if denom == 0:
        return float("nan"), float("nan")

    def _stat(zv: np.ndarray) -> float:
        return float((n / s0) * float(zv @ w @ zv) / denom)

    obs = _stat(z)

    rng = np.random.default_rng(seed)
    ge = 1  # +1 for the observed value (standard permutation p-value)
    for _ in range(n_perm):
        if abs(_stat(rng.permutation(z))) >= abs(obs):
            ge += 1
    return float(obs), float(ge / (n_perm + 1))


def effective_n(
    values: np.ndarray,
    dist: np.ndarray,
    nugget: float,
    sill: float,
    range_: float,
) -> float:
    """Variogram-based effective sample size under autocorrelation.

    ``n_eff = n / (1 + (1/n) * sum_{i!=j} rho(d_ij))`` with the exponential
    correlation ``rho(d) = ((sill - nugget)/sill) * exp(-d/range_)`` (Griffith
    2005; Watson 2021). Reduces to ``n`` when there is no autocorrelation.

    Args:
        values: The observations, used only for their count.
        dist: Condensed pairwise distance matrix over those observations.
        nugget: Variogram nugget C0.
        sill: Variogram sill C1.
        range_: Variogram e-folding range r.

    Returns:
        Effective sample size (<= n), or n if the fit was degenerate.
    """
    n = len(values)
    if (
        n < 2
        or not np.isfinite(sill)
        or sill <= 0
        or not np.isfinite(range_)
        or range_ <= 0
    ):
        return float(n)

    corr_ratio = (sill - nugget) / sill
    if not np.isfinite(corr_ratio) or corr_ratio <= 0:
        return float(n)

    rho = corr_ratio * np.exp(-dist / range_)
    np.fill_diagonal(rho, 0.0)
    deff = 1.0 + float(rho.sum()) / n
    if deff <= 0:
        return float(n)
    return float(n / deff)


def within_between_contrast(values: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """Mean semivariance of same-cluster vs different-cluster pairs.

    This is the "does the prior point predict the next one, versus a point
    elsewhere?" contrast. ``ratio = within / between`` near 1 means
    itineraries look like representative subsamples (dispersed); ``ratio``
    well below 1 means within-itinerary pairs are much more alike (compact
    routes), which inflates the design effect.

    Args:
        values: One value per observation.
        labels: Cluster label per observation, aligned to ``values``.

    Returns:
        ``{"within", "between", "ratio"}`` (semivariances; NaN where undefined).
    """
    values = np.asarray(values, dtype=float)
    labels = np.asarray(labels)
    n = len(values)
    nan = {"within": float("nan"), "between": float("nan"), "ratio": float("nan")}
    if n < 2:
        return nan

    iu, ju = np.triu_indices(n, k=1)
    sq = 0.5 * (values[iu] - values[ju]) ** 2
    same = labels[iu] == labels[ju]

    within = float(sq[same].mean()) if np.any(same) else float("nan")
    between = float(sq[~same].mean()) if np.any(~same) else float("nan")
    ratio = (
        within / between
        if np.isfinite(between) and between > 0 and np.isfinite(within)
        else float("nan")
    )
    return {"within": within, "between": between, "ratio": ratio}

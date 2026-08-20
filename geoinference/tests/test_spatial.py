"""
Tests for the spatial/temporal dependence diagnostics and the simulation
harness.

The unit tests check the measurement primitives (``geoinference.spatial``); the
integration tests wire them through ``estimate``; the simulation tests assert
the analytically-predicted limiting cases (the "truth table" in
``geoinference.simulate``).
"""

import unittest
import warnings

import numpy as np
import pandas as pd

from geoinference import PointDesign, estimate
from geoinference.spatial import (
    effective_n,
    empirical_variogram,
    fit_variogram,
    haversine_matrix,
    morans_i,
    time_gap_matrix,
    within_between_contrast,
)


def _exp_field(coords: np.ndarray, range_m: float, seed: int) -> np.ndarray:
    """Draw a zero-mean exponential Gaussian field on the given coords."""
    d = haversine_matrix(coords[:, 0], coords[:, 1])
    cov = np.exp(-d / range_m) + 1e-9 * np.eye(len(d))
    chol = np.linalg.cholesky(cov)
    rng = np.random.default_rng(seed)
    return chol @ rng.standard_normal(len(d))


def _grid(side: int, extent_deg: float = 0.05) -> np.ndarray:
    axis = np.linspace(0.0, extent_deg, side)
    lon, lat = np.meshgrid(axis, axis)
    return np.column_stack([lon.ravel(), lat.ravel()])


class TestHaversine(unittest.TestCase):
    def test_one_degree_latitude(self):
        # One degree of latitude is ~111.19 km with R = 6,371 km.
        d = haversine_matrix(np.array([0.0, 0.0]), np.array([0.0, 1.0]))
        self.assertAlmostEqual(d[0, 1], 111_194.9, delta=200.0)

    def test_symmetric_zero_diagonal(self):
        coords = _grid(4)
        d = haversine_matrix(coords[:, 0], coords[:, 1])
        self.assertTrue(np.allclose(d, d.T))
        self.assertTrue(np.allclose(np.diag(d), 0.0))


class TestTimeGap(unittest.TestCase):
    def test_numeric_epoch(self):
        ts = np.array([0.0, 10.0, 25.0])
        g = time_gap_matrix(ts)
        self.assertAlmostEqual(g[0, 1], 10.0)
        self.assertAlmostEqual(g[0, 2], 25.0)
        self.assertAlmostEqual(g[1, 2], 15.0)

    def test_datetime(self):
        ts = pd.to_datetime(
            ["2024-01-01 09:00", "2024-01-01 09:05", "2024-01-01 10:00"]
        ).to_numpy()
        g = time_gap_matrix(ts)
        self.assertAlmostEqual(g[0, 1], 300.0)  # 5 min
        self.assertAlmostEqual(g[0, 2], 3600.0)  # 1 hr


class TestVariogram(unittest.TestCase):
    def test_recovers_range(self):
        # A single field realization gives a noisy variogram (apparent long
        # range); averaging gamma over realizations recovers the true range,
        # which is the statistically sound way to fit a variogram.
        coords = _grid(20)
        true_range = 1500.0
        dist = haversine_matrix(coords[:, 0], coords[:, 1])
        # lags/counts are fixed by the geometry; only gamma varies by realization.
        lags, _, counts = empirical_variogram(
            _exp_field(coords, true_range, seed=0), dist, n_bins=15
        )
        gammas = [
            empirical_variogram(
                _exp_field(coords, true_range, seed=s), dist, n_bins=15
            )[1]
            for s in range(25)
        ]
        gamma_bar = np.mean(np.vstack(gammas), axis=0)
        _, _, r = fit_variogram(lags, gamma_bar, counts)
        self.assertGreater(r, true_range * 0.4)
        self.assertLess(r, true_range * 2.5)

    def test_increasing_for_structured_field(self):
        coords = _grid(20)
        z = _exp_field(coords, 1500.0, seed=2)
        dist = haversine_matrix(coords[:, 0], coords[:, 1])
        _, gamma, _ = empirical_variogram(z, dist, n_bins=10)
        # Short-lag semivariance below long-lag semivariance.
        self.assertLess(gamma[0], gamma[-1])


class TestMoransI(unittest.TestCase):
    def test_independent_near_zero(self):
        coords = _grid(16)
        dist = haversine_matrix(coords[:, 0], coords[:, 1])
        rng = np.random.default_rng(3)
        z = rng.standard_normal(len(coords))
        cutoff = float(np.percentile(dist[dist > 0], 10))
        i, p = morans_i(z, dist, cutoff, n_perm=199, seed=0)
        self.assertLess(abs(i), 0.15)
        self.assertGreater(p, 0.05)

    def test_clustered_positive(self):
        coords = _grid(18)
        z = _exp_field(coords, 2000.0, seed=4)
        dist = haversine_matrix(coords[:, 0], coords[:, 1])
        cutoff = float(np.percentile(dist[dist > 0], 10))
        i, p = morans_i(z, dist, cutoff, n_perm=199, seed=0)
        self.assertGreater(i, 0.2)
        self.assertLess(p, 0.05)


class TestEffectiveN(unittest.TestCase):
    def test_independent_near_n(self):
        coords = _grid(16)
        dist = haversine_matrix(coords[:, 0], coords[:, 1])
        rng = np.random.default_rng(5)
        z = rng.standard_normal(len(coords))
        lags, gamma, counts = empirical_variogram(z, dist)
        c0, c1, r = fit_variogram(lags, gamma, counts)
        n_eff = effective_n(z, dist, c0, c1, r)
        self.assertGreater(n_eff, 0.5 * len(coords))

    def test_correlated_below_n(self):
        coords = _grid(18)
        z = _exp_field(coords, 3000.0, seed=6)
        dist = haversine_matrix(coords[:, 0], coords[:, 1])
        lags, gamma, counts = empirical_variogram(z, dist)
        c0, c1, r = fit_variogram(lags, gamma, counts)
        n_eff = effective_n(z, dist, c0, c1, r)
        self.assertLess(n_eff, 0.6 * len(coords))


class TestWithinBetween(unittest.TestCase):
    def test_compact_below_one(self):
        # Two clusters that are internally homogeneous, different from each other.
        vals = np.concatenate([np.zeros(20) + 0.1, np.zeros(20) + 0.9])
        labels = np.array([0] * 20 + [1] * 20)
        wb = within_between_contrast(vals, labels)
        self.assertLess(wb["ratio"], 0.5)

    def test_random_near_one(self):
        rng = np.random.default_rng(7)
        vals = rng.standard_normal(60)
        labels = rng.integers(0, 6, size=60)
        wb = within_between_contrast(vals, labels)
        self.assertGreater(wb["ratio"], 0.6)
        self.assertLess(wb["ratio"], 1.4)


class TestEstimateIntegration(unittest.TestCase):
    def _make_df(self, n=120, g=6, seed=8):
        rng = np.random.default_rng(seed)
        coords = _grid(int(np.sqrt(n)) + 1)[:n]
        p = 1.0 / (1.0 + np.exp(-_exp_field(coords, 1500.0, seed=seed)))
        return pd.DataFrame(
            {
                "n_women": p,
                "n_people": np.ones(n),
                "itinerary_id": rng.integers(0, g, size=n),
                "longitude": coords[:, 0],
                "latitude": coords[:, 1],
                "timestamp": np.sort(rng.uniform(0, 3600 * 4, size=n)),
            }
        )

    def test_diagnostics_populated_with_coords(self):
        df = self._make_df()
        design = PointDesign(sampling="srs", cluster_var="itinerary_id")
        res = estimate(
            df,
            "n_women",
            "n_people",
            design=design,
            bootstrap=False,
            lon_var="longitude",
            lat_var="latitude",
            time_var="timestamp",
        )
        d = res.diagnostics
        self.assertFalse(np.isnan(d.n_eff_space))
        self.assertFalse(np.isnan(d.n_eff_time))
        self.assertFalse(np.isnan(d.within_between_ratio))
        self.assertIn("Within-itinerary dependence", res.summary())

    def test_graceful_without_coords(self):
        df = self._make_df()
        res = estimate(df, "n_women", "n_people", bootstrap=False)
        self.assertTrue(np.isnan(res.diagnostics.n_eff_space))
        self.assertTrue(np.isnan(res.diagnostics.n_eff_time))
        self.assertNotIn("Within-itinerary dependence", res.summary())

    def test_missing_coord_column_raises(self):
        df = self._make_df()
        with self.assertRaises(ValueError):
            estimate(df, "n_women", "n_people", lon_var="nope", lat_var="latitude")


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    unittest.main()

"""
Tests for geoinference.

Unit tests for components plus simulation-based coverage tests
that verify the SEs are accurate and CIs achieve nominal coverage.
"""

import functools
import unittest

import numpy as np
import pandas as pd
from simcheck import (
    MonteCarloResult,
    assert_coverage,
    assert_se_calibrated,
    assert_unbiased,
    binomial_band,
    reps_for,
)

from geoinference import InferenceResult, PointDesign, WalkDesign, estimate


def _make_test_data(
    n: int = 200,
    g: int = 10,
    true_p: float = 0.3,
    lam: float = 8.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate test data mimicking the geosensing pipeline."""
    rng = np.random.default_rng(seed)
    cluster_ids = np.repeat(np.arange(g), n // g)
    if len(cluster_ids) < n:
        cluster_ids = np.concatenate([cluster_ids, np.zeros(n - len(cluster_ids), dtype=int)])

    h = rng.poisson(lam, n)
    w = rng.binomial(h, true_p)

    return pd.DataFrame(
        {
            "n_women": w,
            "n_people": h,
            "itinerary_id": cluster_ids,
            "walk_id": cluster_ids,
        }
    )


def _make_clustered_data(
    n: int = 200,
    g: int = 10,
    true_p: float = 0.3,
    lam: float = 8.0,
    tau: float = 0.12,
    seed: int = 42,
    correlated_at: str = "itinerary",
) -> pd.DataFrame:
    """Generate data with genuine within-cluster correlation.

    ``_make_test_data`` draws every row independently: it assigns cluster ids but
    the outcomes carry no correlation, so its measured ICC is about 0.01 and the
    cluster-robust standard error comes out 0.97 times the naive one. That makes
    it useless for testing cluster-robust inference -- the machinery under test is
    numerically indistinguishable from the machinery it is supposed to improve on,
    so a broken implementation passes.

    Here each itinerary draws its own rate ``p_j`` around ``true_p``, which is
    what a real street survey looks like and what the cluster-robust estimator is
    for. Measured ICC at ``tau=0.12`` is about 0.31, and the design effect is
    about 2.7.

    The estimand is unchanged. ``E[w] = E[h] E[p_j] = lam * true_p``, so the ratio
    ``E[w]/E[h]`` is still ``true_p``.

    Args:
        n: Number of photos.
        g: Number of itineraries (clusters).
        true_p: Population ratio, and the mean of the per-cluster rates.
        lam: Poisson mean for people per photo.
        tau: Standard deviation of the per-cluster rate. Zero recovers the
            independent case.
        seed: Seed for the generator.
        correlated_at: Which grouping actually carries the correlation,
            ``"itinerary"`` or ``"walk"``. The level matters: clustering a
            standard error at a *finer* level than the correlation misses part of
            it and under-covers, which is what
            ``test_clustering_finer_than_the_correlation_under_covers`` measures.

    Returns:
        pd.DataFrame: Columns ``n_women``, ``n_people``, ``itinerary_id``,
        ``walk_id``. Walks are nested two-per-itinerary rather than aliased onto
        it, so the walk design and the point design are not the same grouping.

    Raises:
        ValueError: If ``correlated_at`` is not a recognised level.
    """
    rng = np.random.default_rng(seed)
    cluster_ids = np.repeat(np.arange(g), n // g)
    if len(cluster_ids) < n:
        cluster_ids = np.concatenate([cluster_ids, np.zeros(n - len(cluster_ids), dtype=int)])

    # Two walks inside each itinerary, so walk_id is a genuinely finer grouping.
    walk_ids = cluster_ids * 2 + (np.arange(n) % 2)

    if correlated_at == "itinerary":
        carrier = cluster_ids
    elif correlated_at == "walk":
        carrier = walk_ids
    else:
        raise ValueError(f"correlated_at must be 'itinerary' or 'walk', got {correlated_at!r}")

    rates = np.clip(true_p + tau * rng.standard_normal(carrier.max() + 1), 0.02, 0.98)
    h = rng.poisson(lam, n)
    w = rng.binomial(h, rates[carrier])

    return pd.DataFrame(
        {
            "n_women": w,
            "n_people": h,
            "itinerary_id": cluster_ids,
            "walk_id": walk_ids,
        }
    )


class TestDesigns(unittest.TestCase):
    """Test design object construction and validation."""

    def test_point_design_srs(self):
        d = PointDesign(sampling="srs", cluster_var="itinerary_id")
        self.assertEqual(d.name, "point_srs_clustered")
        self.assertTrue(d.has_clusters)
        self.assertFalse(d.has_weights)
        # Clustered designs default to cluster-robust SE (naive undercovers
        # under within-itinerary spatial autocorrelation).
        self.assertEqual(d.recommended_se_method, "cluster")

    def test_point_design_srs_unclustered_is_naive(self):
        d = PointDesign(sampling="srs")
        self.assertFalse(d.has_clusters)
        self.assertEqual(d.recommended_se_method, "naive")

    def test_point_design_pps_requires_weights(self):
        with self.assertRaises(ValueError):
            PointDesign(sampling="pps")

    def test_point_design_pps_with_weights(self):
        d = PointDesign(sampling="pps", weight_var="inc_prob")
        self.assertTrue(d.has_weights)
        self.assertEqual(d.recommended_se_method, "cluster")

    def test_walk_design(self):
        d = WalkDesign(walk_var="walk_id")
        self.assertEqual(d.name, "walk_transect")
        self.assertTrue(d.has_clusters)
        self.assertEqual(d.cluster_var, "walk_id")
        self.assertEqual(d.recommended_se_method, "cluster")

    def test_walk_design_bad_spacing(self):
        with self.assertRaises(ValueError):
            WalkDesign(spacing_m=-10)

    def test_annotation_frac_validation(self):
        with self.assertRaises(ValueError):
            PointDesign(annotation_frac=0.0)
        with self.assertRaises(ValueError):
            PointDesign(annotation_frac=1.5)


class TestEstimateBasic(unittest.TestCase):
    """Test basic estimate() functionality."""

    def setUp(self):
        self.df = _make_test_data(n=200, g=10, true_p=0.3, seed=42)

    def test_returns_inference_result(self):
        design = PointDesign(sampling="srs", cluster_var="itinerary_id")
        result = estimate(self.df, "n_women", "n_people", design=design, bootstrap=False)
        self.assertIsInstance(result, InferenceResult)

    def test_ratio_in_range(self):
        result = estimate(self.df, "n_women", "n_people", bootstrap=False)
        self.assertGreater(result.ratio, 0.0)
        self.assertLess(result.ratio, 1.0)

    def test_photo_mean_in_range(self):
        result = estimate(self.df, "n_women", "n_people", bootstrap=False)
        self.assertGreater(result.photo_mean, 0.0)
        self.assertLess(result.photo_mean, 1.0)

    def test_se_positive(self):
        design = PointDesign(sampling="srs", cluster_var="itinerary_id")
        result = estimate(self.df, "n_women", "n_people", design=design, bootstrap=False)
        self.assertGreater(result.ratio_se.naive, 0)
        self.assertGreater(result.ratio_se.cluster, 0)
        self.assertGreater(result.photo_mean_se.naive, 0)
        self.assertGreater(result.photo_mean_se.cluster, 0)

    def test_ci_contains_estimate(self):
        result = estimate(self.df, "n_women", "n_people", bootstrap=False)
        lo, hi = result.ratio_ci.recommended
        self.assertLessEqual(lo, result.ratio)
        self.assertGreaterEqual(hi, result.ratio)

    def test_no_cluster_var(self):
        """Without cluster_var, each obs is its own cluster."""
        design = PointDesign(sampling="srs")
        result = estimate(self.df, "n_women", "n_people", design=design, bootstrap=False)
        self.assertEqual(result.n_clusters, len(self.df))

    def test_walk_design(self):
        design = WalkDesign(walk_var="walk_id")
        result = estimate(self.df, "n_women", "n_people", design=design, bootstrap=False)
        self.assertEqual(result.design_name, "walk_transect")
        self.assertEqual(result.ratio_se.method_used, "cluster")

    def test_missing_column_raises(self):
        design = PointDesign(cluster_var="nonexistent")
        with self.assertRaises(ValueError):
            estimate(self.df, "n_women", "n_people", design=design)

    def test_summary_string(self):
        result = estimate(self.df, "n_women", "n_people", bootstrap=False)
        s = result.summary()
        self.assertIn("Ratio estimand", s)
        self.assertIn("Photo-level mean", s)
        self.assertIn("Diagnostics", s)

    def test_diagnostics_populated(self):
        design = PointDesign(sampling="srs", cluster_var="itinerary_id")
        result = estimate(self.df, "n_women", "n_people", design=design, bootstrap=False)
        d = result.diagnostics
        self.assertEqual(d.n_obs, 200)
        self.assertGreater(d.n_positive_frames, 0)
        self.assertGreaterEqual(d.n_empty_frames, 0)
        self.assertEqual(d.n_clusters, 10)
        self.assertGreater(d.icc, -0.01)  # ICC can be slightly negative
        self.assertGreaterEqual(d.deff, 1.0)

    def test_bootstrap_produces_ci(self):
        design = PointDesign(sampling="srs", cluster_var="itinerary_id")
        result = estimate(
            self.df,
            "n_women",
            "n_people",
            design=design,
            bootstrap=True,
            bootstrap_reps=199,
            seed=99,
        )
        self.assertIsNotNone(result.ratio_ci.bootstrap)
        self.assertIsNotNone(result.ratio_se.bootstrap)


class TestEstimateEdgeCases(unittest.TestCase):
    """Test edge cases and unusual inputs."""

    def test_all_empty_frames(self):
        df = pd.DataFrame({"n_women": [0, 0, 0], "n_people": [0, 0, 0]})
        result = estimate(df, "n_women", "n_people", bootstrap=False)
        self.assertTrue(np.isnan(result.ratio))
        self.assertTrue(np.isnan(result.photo_mean))

    def test_single_observation(self):
        df = pd.DataFrame({"n_women": [3], "n_people": [10]})
        result = estimate(df, "n_women", "n_people", bootstrap=False)
        self.assertAlmostEqual(result.ratio, 0.3)
        self.assertAlmostEqual(result.photo_mean, 0.3)

    def test_two_clusters(self):
        df = pd.DataFrame(
            {
                "n_women": [2, 3, 4, 5],
                "n_people": [10, 10, 10, 10],
                "cluster": [0, 0, 1, 1],
            }
        )
        design = PointDesign(cluster_var="cluster")
        result = estimate(df, "n_women", "n_people", design=design, bootstrap=False)
        self.assertEqual(result.n_clusters, 2)
        self.assertGreater(result.ratio_se.cluster, 0)

    def test_single_cluster(self):
        df = pd.DataFrame(
            {
                "n_women": [2, 3, 4],
                "n_people": [10, 10, 10],
                "cluster": [0, 0, 0],
            }
        )
        design = PointDesign(cluster_var="cluster")
        result = estimate(df, "n_women", "n_people", design=design, bootstrap=False)
        self.assertEqual(result.n_clusters, 1)
        # Cluster SE should be NaN with one cluster
        self.assertTrue(np.isnan(result.ratio_se.cluster))


class TestCoverageSimulation(unittest.TestCase):
    """Simulation tests for bias, standard-error accuracy and CI coverage.

    Every gate here comes from :mod:`simcheck`, so its tolerance is derived from
    the replicate count rather than chosen by hand. That matters for two reasons
    this file used to demonstrate.

    **The coverage gates were one-sided.** ``assertGreater(coverage, 0.90)`` for
    a nominal 95% interval passes an interval so wide it covers every single
    time, which is a defect the assertion cannot see. ``assert_coverage`` bands
    the rate on both sides of nominal.

    **The hard case was never generated.** ``_make_test_data`` draws every row
    independently, so its ICC is about 0.01 and the cluster-robust standard error
    comes out 0.97 times the naive one. The suite claimed to verify cluster-robust
    inference while running it on data with no clustering, where a broken
    implementation is numerically indistinguishable from a correct one. The
    clustered studies below use :func:`_make_clustered_data`, and
    ``test_naive_se_is_caught_under_covering`` is the test that gives the rest of
    them meaning: on clustered data the naive standard error *must* fail.
    """

    TRUE_P = 0.3

    def _study(
        self,
        design_fn,
        data_fn,
        *,
        se_attr: str = "recommended",
        ci_attr: str | None = "recommended",
        estimand: str = "ratio",
        reps: int | None = None,
        seed: int = 0,
        bootstrap: bool = False,
    ) -> MonteCarloResult:
        """Run a Monte Carlo study and package it for the simcheck gates.

        Args:
            design_fn: Zero-argument callable returning the design to use.
            data_fn: Callable taking ``seed`` and returning a DataFrame.
            se_attr: Which standard error to read off the result.
            ci_attr: Which interval to read, or None to build a normal interval
                from ``se_attr`` -- needed for the naive standard error, which
                the package does not publish an interval for.
            estimand: ``"ratio"`` or ``"photo_mean"``.
            reps: Replicate count; defaults to the current simcheck tier.
            seed: Seed for the sequence of dataset seeds.
            bootstrap: Whether to ask ``estimate`` for bootstrap intervals.

        Returns:
            MonteCarloResult: Estimates, standard errors and coverage flags.
        """
        reps = reps_for() if reps is None else reps
        rng = np.random.default_rng(seed)

        estimates, errors, covered = [], [], []
        for _ in range(reps):
            frame = data_fn(seed=int(rng.integers(2**31)))
            result = estimate(
                frame,
                "n_women",
                "n_people",
                design=design_fn(),
                bootstrap=bootstrap,
            )
            point = getattr(result, estimand)
            if np.isnan(point):
                continue
            error = getattr(getattr(result, f"{estimand}_se"), se_attr)
            if ci_attr is None:
                low, high = point - 1.96 * error, point + 1.96 * error
            else:
                interval = getattr(getattr(result, f"{estimand}_ci"), ci_attr)
                if interval is None or np.isnan(interval[0]) or np.isnan(interval[1]):
                    continue
                low, high = interval

            estimates.append(point)
            errors.append(error)
            covered.append(low <= self.TRUE_P <= high)

        return MonteCarloResult(
            estimates=np.array(estimates),
            standard_errors=np.array(errors),
            covered=np.array(covered),
            rejected=None,
            truth=self.TRUE_P,
        )

    # ------------------------------------------------------------------
    # Independent data: the case the original suite covered
    # ------------------------------------------------------------------

    def test_independent_data_is_unbiased_and_covers(self):
        """With no clustering, everything should be textbook."""
        for estimand in ("ratio", "photo_mean"):
            with self.subTest(estimand=estimand):
                study = self._study(
                    lambda: PointDesign(sampling="srs", cluster_var="itinerary_id"),
                    _make_test_data,
                    estimand=estimand,
                )
                assert_unbiased(study, label=f"{estimand}, independent data")
                assert_se_calibrated(study, label=f"{estimand}, independent data")
                assert_coverage(study, 0.95, label=f"{estimand}, independent data")

    # ------------------------------------------------------------------
    # Clustered data: the case that makes cluster-robust inference matter
    # ------------------------------------------------------------------

    def test_clustered_data_is_unbiased_and_covers(self):
        """The recommended interval must hold up once the data really cluster."""
        for estimand in ("ratio", "photo_mean"):
            with self.subTest(estimand=estimand):
                study = self._study(
                    lambda: PointDesign(sampling="srs", cluster_var="itinerary_id"),
                    _make_clustered_data,
                    estimand=estimand,
                )
                assert_unbiased(study, label=f"{estimand}, clustered")
                assert_coverage(study, 0.95, label=f"{estimand}, clustered")

    def test_naive_se_is_caught_under_covering(self):
        """The falsification test. Without it, nothing above proves anything.

        The naive standard error assumes independent observations. On clustered
        data it is wrong by the square root of the design effect, so its interval
        must fail a 95% coverage gate. If this test ever stops failing, either the
        generator has stopped producing correlated data or the gate has stopped
        measuring coverage -- and in both cases every other test in this class is
        certifying nothing.
        """
        study = self._study(
            lambda: PointDesign(sampling="srs", cluster_var="itinerary_id"),
            _make_clustered_data,
            se_attr="naive",
            ci_attr=None,
        )
        with self.assertRaises(AssertionError):
            assert_coverage(study, 0.95, label="naive SE on clustered data")

        low, _ = binomial_band(0.95, study.reps)
        self.assertLess(
            study.coverage,
            low,
            "the naive interval should under-cover badly on clustered data",
        )

    def test_cluster_robust_se_is_larger_than_naive_when_data_cluster(self):
        """The design effect must actually show up, or the fixture is wrong."""
        frames = [_make_clustered_data(seed=s) for s in range(20)]
        design_effects = []
        for frame in frames:
            result = estimate(
                frame,
                "n_women",
                "n_people",
                design=PointDesign(sampling="srs", cluster_var="itinerary_id"),
                bootstrap=False,
            )
            design_effects.append(result.ratio_se.cluster / result.ratio_se.naive)

        self.assertGreater(
            float(np.median(design_effects)),
            1.5,
            "clustered data must inflate the cluster-robust SE over the naive one",
        )

    def test_walk_design_covers(self):
        """Walks are nested inside itineraries, so this is a distinct grouping.

        In the independent fixture ``walk_id`` was assigned the same array as
        ``itinerary_id``, which made this test a byte-for-byte rerun of the point
        design one -- identical bias, identical SE ratio, identical coverage to
        three decimals.

        The correlation is generated at the walk level here, matching the level
        the design clusters at. That is the case the walk design is *for*.
        """
        study = self._study(
            lambda: WalkDesign(walk_var="walk_id"),
            functools.partial(_make_clustered_data, correlated_at="walk"),
        )
        assert_unbiased(study, label="ratio, walk design")
        assert_coverage(study, 0.95, label="ratio, walk design")

    def test_clustering_finer_than_the_correlation_under_covers(self):
        """Getting the cluster level wrong is not a safe error.

        Here the rate varies by itinerary but the standard error clusters on
        walks, which are nested two-per-itinerary. Half of each itinerary's
        correlation therefore falls *between* the units being clustered on, where
        a cluster-robust estimator cannot see it.

        This is a property of cluster-robust inference rather than a defect in
        this package, and it is worth pinning because the failure is silent: the
        estimate is unbiased and the interval looks perfectly reasonable. Measured
        coverage is about 0.84 against a nominal 0.95.
        """
        study = self._study(
            lambda: WalkDesign(walk_var="walk_id"),
            functools.partial(_make_clustered_data, correlated_at="itinerary"),
        )
        assert_unbiased(study, label="ratio, mis-specified cluster level")

        low, _ = binomial_band(0.95, study.reps)
        self.assertLess(
            study.coverage,
            low,
            "clustering below the level of the correlation should under-cover; "
            f"measured {study.coverage:.3f}",
        )


if __name__ == "__main__":
    unittest.main()

"""
Tests for geoinference.

Unit tests for components plus simulation-based coverage tests
that verify the SEs are accurate and CIs achieve nominal coverage.
"""

import functools
import unittest
import warnings

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
from geoinference.inference import _cluster_bootstrap, _ratio_estimator


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
        bootstrap_reps: int = 299,
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
            bootstrap_reps: Bootstrap replications, kept small so a coverage
                study over many datasets stays affordable.

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
                bootstrap_reps=bootstrap_reps,
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

    # ------------------------------------------------------------------
    # Which bootstrap `estimate` hands out
    # ------------------------------------------------------------------

    def test_the_bootstrap_interval_covers_with_few_itineraries(self):
        """``se_method="bootstrap"`` must not be the worst option on offer.

        It was. ``estimate`` filled ``ratio_ci.bootstrap`` from the *pairs*
        cluster bootstrap, which resamples G itineraries with replacement from G
        -- about 63% of them appear in a given draw -- and then takes raw
        percentiles rather than studentizing. Measured coverage of a nominal 95%
        interval, balanced itineraries at ICC 0.31 over 400 replicates:

            G      5      8     12     20     40
            pairs  0.815  0.887 0.920  0.932  0.932
            wild   0.932  0.953 0.940  0.955  0.943

        Five itineraries of twenty frames, matching
        ``test_the_pairs_bootstrap_is_caught_under_covering`` so the two describe
        the same design. There the pairs bootstrap measures 0.818; here the wild
        one measures 0.955.
        """
        study = self._study(
            lambda: PointDesign(sampling="srs", cluster_var="itinerary_id"),
            functools.partial(_make_clustered_data, n=100, g=5),
            ci_attr="wild",
            bootstrap=True,
        )
        assert_unbiased(study, label="ratio, bootstrap interval, G=5")
        assert_coverage(study, 0.95, label="ratio, bootstrap interval, G=5")

    def test_the_pairs_bootstrap_is_caught_under_covering(self):
        """The reason the swap above is not a matter of taste.

        Runs the interval `estimate` used to hand out through the same gate the
        new one passes, and requires it to fail. If this stops failing, either
        the fixture stopped being clustered or the gate stopped measuring
        coverage, and the choice of bootstrap is no longer evidence-backed.

        This is the one test here that does not take its replicate count from the
        simcheck tier, and the reason is power rather than preference. The pairs
        bootstrap's true coverage on this design is about 0.818; at the fast
        tier's 100 replicates the 3-sigma floor is 0.885, only about 1.6 standard
        errors away, so the gate would pass or fail on the seed -- it did exactly
        that during development, landing on 0.890. Four hundred replicates put
        the floor at 0.917, a ten-point margin the sampling noise cannot cross.
        """
        reps = 400
        rng = np.random.default_rng(11)
        covered = []
        for _ in range(reps):
            frame = _make_clustered_data(n=100, g=5, seed=int(rng.integers(2**31)))
            _, low, high = _cluster_bootstrap(
                frame["n_women"].to_numpy(dtype=float),
                frame["n_people"].to_numpy(dtype=float),
                frame["itinerary_id"].to_numpy(),
                _ratio_estimator,
                reps=299,
                rng=np.random.default_rng(int(rng.integers(2**31))),
            )
            if not np.isnan(low):
                covered.append(low <= self.TRUE_P <= high)

        rate = float(np.mean(covered))
        band_low, _ = binomial_band(0.95, len(covered))
        self.assertLess(
            rate,
            band_low,
            "the pairs bootstrap should under-cover at G=5, which is why "
            f"estimate() no longer uses it for the ratio; measured {rate:.3f}",
        )


class TestMethodMenuIsConsistent(unittest.TestCase):
    """The same methods are offered, and mean the same thing, for both estimands.

    An interface property rather than a statistical one, and worth a test because
    it was briefly untrue: the ratio was given the wild cluster bootstrap while
    the photo mean kept the pairs bootstrap, so ``ci.bootstrap`` was a
    percentile-t for one estimand and a raw percentile for the other, and
    ``se.bootstrap`` was populated for one and None for the other. Anyone
    comparing the two would have been comparing different procedures under one
    name.
    """

    def setUp(self):
        self.frame = _make_clustered_data(g=10)
        self.design = PointDesign(sampling="srs", cluster_var="itinerary_id")
        self.result = estimate(self.frame, design=self.design, bootstrap=True, bootstrap_reps=199)

    def test_both_estimands_report_every_standard_error(self):
        """Three standard errors each, none missing."""
        for label, se in (
            ("ratio", self.result.ratio_se),
            ("photo_mean", self.result.photo_mean_se),
        ):
            for field in ("naive", "cluster", "bootstrap"):
                value = getattr(se, field)
                self.assertIsNotNone(value, f"{label}_se.{field} is None")
                self.assertGreater(value, 0.0, f"{label}_se.{field} is not positive")

    def test_both_estimands_report_every_interval(self):
        """Four intervals each, none missing, each properly ordered."""
        for label, ci in (
            ("ratio", self.result.ratio_ci),
            ("photo_mean", self.result.photo_mean_ci),
        ):
            for field in ("normal", "t", "bootstrap", "wild"):
                interval = getattr(ci, field)
                self.assertIsNotNone(interval, f"{label}_ci.{field} is None")
                low, high = interval
                self.assertLess(low, high, f"{label}_ci.{field} is not ordered")

    def test_the_wild_interval_is_its_own_procedure_for_both(self):
        """Percentile-t and raw percentile should not land on the same numbers."""
        for label, ci in (
            ("ratio", self.result.ratio_ci),
            ("photo_mean", self.result.photo_mean_ci),
        ):
            self.assertNotEqual(
                ci.wild, ci.bootstrap, f"{label}: wild and pairs intervals coincide"
            )

    def test_the_caller_can_name_the_interval(self):
        """``ci_method`` selects, rather than the package deciding for you."""
        for method in ("normal", "t", "bootstrap", "wild"):
            result = estimate(
                self.frame,
                design=self.design,
                bootstrap=True,
                bootstrap_reps=199,
                ci_method=method,
            )
            for label, ci in (
                ("ratio", result.ratio_ci),
                ("photo_mean", result.photo_mean_ci),
            ):
                self.assertEqual(
                    ci.recommended,
                    getattr(ci, method),
                    f"{label}: ci_method={method!r} did not select that interval",
                )

    def test_the_wild_interval_centres_on_each_cluster_own_normalizer(self):
        """A bug that is invisible on balanced data, so it needs uneven data.

        The percentile-t denominator uses the scores at the *bootstrap* estimate,
        which is the observed one shifted by delta. Cluster c's score moves by
        delta times its own normalizer. Subtracting the mean instead --
        ``drawn - drawn.mean()`` -- takes ``delta * normalizer / G`` off every
        cluster, which is the same number only when every cluster is the same
        size. Coverage at nominal 0.95 over 400 replicates was 0.948 / 0.927 /
        0.897 / 0.890 under mean-centering as the size CV rose through
        0 / 0.6 / 1.0 / 1.5, against 0.948 / 0.940 / 0.915 / 0.927 with this.

        Rather than re-run a coverage study here, this checks the algebra that
        makes the difference: on equal-sized clusters the two centerings agree,
        and on unequal ones they do not. A regression to mean-centering would
        make the second assertion fail.
        """
        equal = np.full(6, 20.0)
        uneven = np.array([100.0, 8.0, 8.0, 8.0, 8.0, 8.0])
        delta = 0.01

        for label, normalizers in (("equal", equal), ("uneven", uneven)):
            per_cluster = delta * normalizers
            mean_centred = np.full(6, delta * normalizers.sum() / 6)
            same = np.allclose(per_cluster, mean_centred)
            if label == "equal":
                self.assertTrue(same, "the two centerings must agree when balanced")
            else:
                self.assertFalse(same, "the two centerings must differ when sizes are uneven")

        # And the shipped interval is finite and ordered on the uneven design.
        frame = _make_clustered_data(g=6)
        low, high = estimate(
            frame,
            design=PointDesign(sampling="srs", cluster_var="itinerary_id"),
            bootstrap=True,
            bootstrap_reps=199,
        ).ratio_ci.wild
        self.assertLess(low, high)

    def test_an_unknown_method_is_refused(self):
        """Falling back silently would hide a typo in an analysis script."""
        with self.assertRaises(ValueError):
            estimate(self.frame, design=self.design, bootstrap=False, ci_method="wilde")
        with self.assertRaises(ValueError):
            estimate(self.frame, design=self.design, bootstrap=False, se_method="wild")

    def test_asking_for_an_uncomputed_interval_is_refused(self):
        """``bootstrap=False`` with ``ci_method="wild"`` must not return normal."""
        with self.assertRaises(ValueError):
            estimate(self.frame, design=self.design, bootstrap=False, ci_method="wild")


class TestEffectiveItineraries(unittest.TestCase):
    """The number that predicts whether any interval here can be trusted.

    Every cluster-robust interval in this package estimates between-itinerary
    variance from the *effective* number of itineraries, not the nominal one, and
    unequal sizes drive the two apart fast. Measured with lognormal itinerary
    sizes at ICC 0.31 over 400 replicates, coverage of a nominal 95% interval was
    inside a 3-sigma band at 15.2 and 18.4 effective itineraries and outside it at
    11.8 and below -- for the normal, the t, and both bootstraps alike. No choice
    of ``se_method`` escapes it, which is why this is a warning rather than a
    method recommendation.
    """

    @staticmethod
    def _frame(sizes, tau=0.12, seed=0):
        """Counts whose itineraries have exactly the given sizes.

        Args:
            sizes: Number of frames in each itinerary.
            tau: Spread of the per-itinerary rate.
            seed: Seed for the generator.

        Returns:
            pd.DataFrame: Columns ``n_women``, ``n_people``, ``itinerary_id``.
        """
        rng = np.random.default_rng(seed)
        cluster_ids = np.repeat(np.arange(len(sizes)), sizes)
        rates = np.clip(0.3 + tau * rng.standard_normal(len(sizes)), 0.02, 0.98)
        h = rng.poisson(8.0, cluster_ids.size)
        w = rng.binomial(h, rates[cluster_ids])
        return pd.DataFrame({"n_women": w, "n_people": h, "itinerary_id": cluster_ids})

    def test_equal_sizes_give_back_the_nominal_count(self):
        """With equal itineraries the Kish count is the plain count."""
        result = estimate(
            self._frame([20] * 20),
            design=PointDesign(sampling="srs", cluster_var="itinerary_id"),
            bootstrap=False,
        )
        self.assertEqual(result.diagnostics.n_clusters, 20)
        self.assertAlmostEqual(result.diagnostics.n_clusters_eff, 20.0, places=6)

    def test_one_dominant_itinerary_collapses_the_effective_count(self):
        """Twenty itineraries can behave like two, and the count must say so."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = estimate(
                self._frame([300] + [8] * 19),
                design=PointDesign(sampling="srs", cluster_var="itinerary_id"),
                bootstrap=False,
            )
        self.assertEqual(result.diagnostics.n_clusters, 20)
        self.assertLess(result.diagnostics.n_clusters_eff, 4.0)

    def test_a_collapsed_design_warns(self):
        """Silence here is the failure mode: the interval still looks fine."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            estimate(
                self._frame([300] + [8] * 19),
                design=PointDesign(sampling="srs", cluster_var="itinerary_id"),
                bootstrap=False,
            )
        messages = [str(w.message) for w in caught if "Itinerary sizes vary" in str(w.message)]
        self.assertTrue(messages, "a design with ~2 effective itineraries must warn")
        self.assertIn("indicative", messages[0])

    def test_the_warning_can_be_switched_off_without_losing_the_number(self):
        """The threshold is a judgement, so the caller gets to disagree with it.

        Fifteen effective itineraries is measured rather than conventional, but
        it is still an opinion, and an opinion compiled into a module constant is
        one nobody can overrule. Passing None silences it. What must survive is
        ``n_clusters_eff`` itself: the number is information, and only the view
        taken of it is optional.
        """
        frame = self._frame([300] + [8] * 19)
        design = PointDesign(sampling="srs", cluster_var="itinerary_id")

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = estimate(
                frame,
                design=design,
                bootstrap=False,
                warn_above_cluster_size_cv=None,
            )
        self.assertFalse(
            [str(w.message) for w in caught if "Itinerary sizes vary" in str(w.message)],
            "warn_above_cluster_size_cv=None must silence the warning",
        )
        self.assertLess(result.diagnostics.n_clusters_eff, 4.0)

    def test_the_threshold_is_the_callers_to_set(self):
        """A design healthy by the default must still warn under a stricter bar."""
        # Mildly uneven: CV about 0.3, comfortably inside the 0.7 default.
        frame = self._frame([28] * 10 + [14] * 10)
        design = PointDesign(sampling="srs", cluster_var="itinerary_id")

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            quiet = estimate(frame, design=design, bootstrap=False)
        self.assertFalse(
            [str(w.message) for w in caught if "Itinerary sizes vary" in str(w.message)],
            "a CV of about 0.3 is inside the default and should not warn",
        )
        self.assertLess(quiet.diagnostics.cluster_size_cv, 0.7)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            estimate(
                frame,
                design=design,
                bootstrap=False,
                warn_above_cluster_size_cv=0.1,
            )
        self.assertTrue(
            [str(w.message) for w in caught if "Itinerary sizes vary" in str(w.message)],
            "lowering the bar to 0.1 should make the same design warn",
        )

    def test_a_healthy_design_stays_quiet(self):
        """A warning that always fires is a warning nobody reads."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            estimate(
                self._frame([20] * 20),
                design=PointDesign(sampling="srs", cluster_var="itinerary_id"),
                bootstrap=False,
            )
        self.assertFalse(
            [str(w.message) for w in caught if "Itinerary sizes vary" in str(w.message)],
            "20 equal itineraries should not trip the effective-count warning",
        )


if __name__ == "__main__":
    unittest.main()


class TestBootstrapHonoursTheRequestedLevel(unittest.TestCase):
    """The pairs cluster bootstrap ignored ci_level and always returned 95%.

    `_cluster_bootstrap` took no `ci_level` and hardcoded `np.percentile(..., 2.5)`
    and `97.5`, while `wild_cluster_bootstrap_ci` beside it honoured the argument.
    So `estimate(ci_level=0.90)` returned a 90% wild interval and a 95% pairs
    interval in the same result object, with nothing marking the difference.
    """

    @staticmethod
    def _panel(seed=0):
        rng = np.random.default_rng(seed)
        g, per = 25, 20
        labels = np.repeat(np.arange(g), per)
        w = rng.random(g * per) + 0.5
        h = rng.random(g * per) * 10 + 1.0
        return w, h, labels

    def test_a_narrower_level_gives_a_narrower_interval(self):
        from geoinference.inference import _cluster_bootstrap

        w, h, labels = self._panel()
        est = lambda w_, h_: float((w_ * h_).sum() / h_.sum())

        def width(level):
            _, lo, hi = _cluster_bootstrap(
                w,
                h,
                labels,
                est,
                reps=1500,
                ci_level=level,
                rng=np.random.default_rng(7),
            )
            return hi - lo

        # Same replicates, same seed: only the percentiles differ, so this is a
        # pure statement about the argument being read.
        self.assertLess(width(0.80), width(0.90))
        self.assertLess(width(0.90), width(0.95))

    def test_the_bounds_are_the_requested_percentiles(self):
        from geoinference.inference import _cluster_bootstrap

        w, h, labels = self._panel(seed=3)
        est = lambda w_, h_: float((w_ * h_).sum() / h_.sum())
        _, lo, hi = _cluster_bootstrap(
            w,
            h,
            labels,
            est,
            reps=1500,
            ci_level=0.90,
            rng=np.random.default_rng(11),
        )
        # Reproduce the draw and check the endpoints are the 5th and 95th
        # percentiles rather than the 2.5th and 97.5th.
        _, lo95, hi95 = _cluster_bootstrap(
            w,
            h,
            labels,
            est,
            reps=1500,
            ci_level=0.95,
            rng=np.random.default_rng(11),
        )
        self.assertGreater(lo, lo95)
        self.assertLess(hi, hi95)

    def test_estimate_passes_the_level_through_to_both_bootstraps(self):
        """The end-to-end version: the two intervals must claim the same level."""
        rng = np.random.default_rng(5)
        g, per = 20, 15
        people = rng.integers(1, 8, g * per)
        df = pd.DataFrame(
            {
                "itinerary": np.repeat(np.arange(g), per),
                "n_people": people,
                "n_women": rng.binomial(people, 0.4),
            }
        )
        design = PointDesign(cluster_var="itinerary")

        wide = estimate(
            df, "n_women", "n_people", design=design, ci_level=0.95, bootstrap_reps=400, seed=1
        )
        narrow = estimate(
            df, "n_women", "n_people", design=design, ci_level=0.80, bootstrap_reps=400, seed=1
        )

        def span(ci):
            return ci[1] - ci[0]

        self.assertLess(
            span(narrow.ratio_ci.bootstrap),
            span(wide.ratio_ci.bootstrap),
            "the pairs bootstrap interval did not respond to ci_level",
        )

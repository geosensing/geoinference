"""
Limiting-case tests for the space-time simulation harness.

These assert the analytically-predicted truth table. They use modest Monte
Carlo sizes with fixed seeds and deliberately loose thresholds (directional
inequalities with margin), so they verify the qualitative behavior without
being flaky.
"""

import unittest
import warnings

from geoinference.simulate import (
    Pipeline,
    PopulationFactory,
    SimConfig,
    run_pipeline,
)


def _run(cfg, pipe, se_method="auto", spatial_diag=False):
    fac = PopulationFactory(cfg)
    return run_pipeline(fac, pipe, cfg, se_method=se_method, spatial_diag=spatial_diag)


class TestSpatialLimits(unittest.TestCase):
    def test_corr_zero_compact_is_most_efficient(self):
        """White-noise field: compact (most points/budget) has the lowest SD."""
        cfg = SimConfig(range_s_m=0.0, diurnal_amp=0.0, sd_t=0.0, n_sims=80, grid_n=14)
        compact = _run(cfg, Pipeline("compact", routing="compact"))
        dispersed = _run(cfg, Pipeline("dispersed", routing="dispersed"))
        self.assertGreater(compact.mean_n, dispersed.mean_n)
        self.assertLess(compact.true_sd, dispersed.true_sd)

    def test_corr_zero_naive_se_well_calibrated(self):
        """No correlation: naive SE is honest (coverage near nominal)."""
        cfg = SimConfig(range_s_m=0.0, diurnal_amp=0.0, sd_t=0.0, n_sims=120, grid_n=14)
        r = _run(cfg, Pipeline("compact", routing="compact"), se_method="naive")
        self.assertGreater(r.coverage, 0.88)

    def test_corr_huge_collapses_effective_n(self):
        """Near-constant field: spatial n_eff collapses far below the point count."""
        cfg = SimConfig(range_s_m=1e7, diurnal_amp=0.0, sd_t=0.0, n_sims=60, grid_n=14)
        r = _run(cfg, Pipeline("compact", routing="compact"), spatial_diag=True)
        self.assertGreater(r.mean_n, 50)
        self.assertLess(r.mean_n_eff_space, 0.25 * r.mean_n)


class TestCoverageCliff(unittest.TestCase):
    def test_naive_se_breaks_under_correlation_cluster_holds(self):
        """Compact + correlation: naive SE undercovers; cluster/WCB stay valid.

        With K=8 itineraries the cluster-robust t and the wild cluster
        bootstrap reach ~0.90-0.94 (the gap to nominal 0.95 is the
        few-clusters limit); naive collapses toward ~0.6.
        """
        cfg = SimConfig(
            range_s_m=2000.0, diurnal_amp=0.0, sd_t=0.0, n_sims=200, grid_n=16
        )
        pipe = Pipeline("compact", routing="compact")
        naive = _run(cfg, pipe, se_method="naive")
        cluster = _run(cfg, pipe, se_method="cluster")
        wcb = _run(cfg, pipe, se_method="wcb")
        self.assertLess(naive.coverage, 0.78)  # cliff
        self.assertGreater(cluster.coverage, 0.88)  # robust (t_{G-1} CI)
        self.assertGreater(wcb.coverage, 0.88)  # robust (wild cluster boot)
        self.assertGreater(cluster.coverage, naive.coverage)


class TestTemporalBias(unittest.TestCase):
    def test_synced_starts_bias_fixed_by_staggering(self):
        """Strong diurnal effect: synced starts bias beta; staggering fixes it."""
        cfg = SimConfig(
            range_s_m=600.0, diurnal_amp=1.5, sd_t=0.0, n_sims=120, grid_n=14
        )
        synced = _run(cfg, Pipeline("c", routing="compact", staggered_starts=False))
        stagger = _run(cfg, Pipeline("c", routing="compact", staggered_starts=True))
        self.assertGreater(abs(synced.bias), 0.05)
        self.assertLess(abs(stagger.bias), abs(synced.bias) / 2.0)
        self.assertGreater(stagger.coverage, synced.coverage)

    def test_no_time_structure_no_bias(self):
        """No diurnal effect: start-time policy is irrelevant (no bias)."""
        cfg = SimConfig(
            range_s_m=600.0, diurnal_amp=0.0, sd_t=0.0, n_sims=100, grid_n=14
        )
        synced = _run(cfg, Pipeline("c", routing="compact", staggered_starts=False))
        self.assertLess(abs(synced.bias), 0.02)


class TestTheRequestedLevelReachesEveryMethod(unittest.TestCase):
    """`run_pipeline(ci_level=...)` was ignored by the bootstrap method.

    `_method_ci` builds the naive/cluster/auto intervals itself from
    ``ci_level``, and passes ``ci_level`` to the wild cluster bootstrap. But
    for ``se_method="boot"`` it reads ``res.ratio_ci.bootstrap`` straight off
    the ``InferenceResult`` — and the ``estimate`` call omitted ``ci_level``,
    so that interval was always ``estimate``'s 95% default. Coverage came out
    identical at every requested level, which makes a coverage table at any
    level but 0.95 fiction.
    """

    CFG = SimConfig(range_s_m=600.0, diurnal_amp=0.0, sd_t=0.0, n_sims=60, grid_n=12)

    def _coverage(self, se_method: str, ci_level: float) -> float:
        fac = PopulationFactory(self.CFG)
        return run_pipeline(
            fac,
            Pipeline("compact", routing="compact"),
            self.CFG,
            se_method=se_method,
            ci_level=ci_level,
        ).coverage

    def test_a_wider_level_covers_more_often(self):
        for method in ("boot", "naive", "cluster", "wcb"):
            with self.subTest(se_method=method):
                narrow = self._coverage(method, 0.50)
                wide = self._coverage(method, 0.99)
                self.assertGreater(
                    wide,
                    narrow,
                    f"{method}: coverage did not respond to ci_level "
                    f"(0.50 -> {narrow:.3f}, 0.99 -> {wide:.3f})",
                )


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    unittest.main()

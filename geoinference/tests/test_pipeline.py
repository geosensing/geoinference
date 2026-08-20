"""
Tests for the production / pipeline surface.

``estimate_from_file`` is tested directly (no optional deps). The real
geo_sampling -> allocator path is tested only when the ``pipeline`` extra is
installed; otherwise those tests skip cleanly.
"""

import importlib.util
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from geoinference import estimate_from_file, read_frames
from geoinference.simulate import PopulationFactory, SimConfig, evaluate_scene

_HAS_ALLOCATOR = importlib.util.find_spec("allocator") is not None
_DELHI = (
    Path(__file__).parent.parent.parent.parent
    / "allocator"
    / "examples"
    / "inputs"
    / "delhi-roads-1k.csv"
)


def _frames(n: int = 300, g: int = 12, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "n_women": rng.binomial(8, 0.3, n),
            "n_people": rng.poisson(8, n) + 1,
            "itinerary_id": rng.integers(0, g, n),
            "longitude": rng.uniform(77.0, 77.05, n),
            "latitude": rng.uniform(28.6, 28.65, n),
            "timestamp": np.sort(rng.uniform(0, 1e6, n)),
        }
    )


def _write(frame: pd.DataFrame, path: Path) -> Path:
    """Write a frame in whichever format the path's suffix names."""
    if path.suffix == ".parquet":
        frame.to_parquet(path, index=False)
    else:
        frame.to_csv(path, index=False)
    return path


class TestEstimateFromFile(unittest.TestCase):
    def test_full_columns(self):
        for name in ("frames.parquet", "frames.csv"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as d:
                p = _write(_frames(), Path(d) / name)
                res = estimate_from_file(p, bootstrap=False)
                self.assertGreater(res.ratio, 0.0)
                self.assertLess(res.ratio, 1.0)
                self.assertEqual(res.ratio_se.method_used, "cluster")
                self.assertFalse(np.isnan(res.diagnostics.n_eff_space))
                self.assertFalse(np.isnan(res.diagnostics.n_eff_time))

    def test_parquet_and_csv_agree(self):
        """The format must not move the estimate."""
        frame = _frames()
        with tempfile.TemporaryDirectory() as d:
            pq = estimate_from_file(
                _write(frame, Path(d) / "f.parquet"), bootstrap=False
            )
            csv = estimate_from_file(_write(frame, Path(d) / "f.csv"), bootstrap=False)
        self.assertAlmostEqual(pq.ratio, csv.ratio, places=12)
        self.assertAlmostEqual(
            pq.ratio_se.recommended, csv.ratio_se.recommended, places=12
        )

    def test_parquet_keeps_the_dtypes_csv_loses(self):
        """Counts stay integers and the timestamp stays a datetime."""
        frame = _frames(n=20).astype({"n_women": "int64", "n_people": "int64"})
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], unit="s")
        with tempfile.TemporaryDirectory() as d:
            pq = read_frames(_write(frame, Path(d) / "f.parquet"))
            csv = read_frames(_write(frame, Path(d) / "f.csv"))
        self.assertEqual(pq["n_women"].dtype, frame["n_women"].dtype)
        self.assertEqual(pq["timestamp"].dtype, frame["timestamp"].dtype)
        # CSV has no types: the timestamp comes back as text.
        self.assertNotEqual(csv["timestamp"].dtype, frame["timestamp"].dtype)

    def test_compressed_csv_is_read(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "frames.csv.gz"
            _frames(n=30).to_csv(p, index=False)
            res = estimate_from_file(p, bootstrap=False)
        self.assertGreater(res.ratio, 0.0)

    def test_an_unreadable_suffix_says_so(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "frames.xlsx"
            p.write_bytes(b"not a table")
            with self.assertRaises(ValueError) as ctx:
                estimate_from_file(p)
        self.assertIn("Cannot tell how to read", str(ctx.exception))

    def test_minimal_columns_no_optional(self):
        # Only the required counts present: optional vars are ignored, no error.
        with tempfile.TemporaryDirectory() as d:
            p = _write(
                pd.DataFrame({"n_women": [2, 3, 4], "n_people": [10, 10, 10]}),
                Path(d) / "frames.parquet",
            )
            res = estimate_from_file(p, bootstrap=False)
        self.assertAlmostEqual(res.ratio, 0.3)
        self.assertTrue(np.isnan(res.diagnostics.n_eff_space))

    def test_missing_required_raises(self):
        with tempfile.TemporaryDirectory() as d:
            p = _write(pd.DataFrame({"n_people": [10, 10]}), Path(d) / "frames.parquet")
            with self.assertRaises(ValueError):
                estimate_from_file(p)


@unittest.skipUnless(
    _HAS_ALLOCATOR and _DELHI.exists(),
    "pipeline extra (allocator) and the bundled Delhi roads CSV required",
)
class TestRealPipeline(unittest.TestCase):
    def setUp(self):
        warnings.simplefilter("ignore")

    def test_subsample_scene_and_validate(self):
        from geoinference.pipeline import points_from_roads, subsample_scene

        uni = points_from_roads(str(_DELHI), per_segment=4)
        self.assertEqual(len(uni), 4000)
        sample_idx, scene = subsample_scene(
            uni, n_sample=200, method="random_partition", n_itineraries=40, seed=1
        )
        self.assertEqual(len(scene), len(sample_idx))
        self.assertEqual(scene.n_itineraries, 40)
        self.assertGreater(scene.day_span, 0.0)

        cfg = SimConfig(range_s_m=800.0, diurnal_amp=0.0, sd_t=0.0, n_sims=40)
        factory = PopulationFactory(
            cfg, uni["longitude"].to_numpy(), uni["latitude"].to_numpy()
        )
        res = evaluate_scene(
            factory,
            sample_idx,
            scene.itinerary_id,
            scene.time_of_day_min,
            scene.timestamp_s,
            cfg,
            se_method="cluster",
            spatial_diag=False,
        )
        self.assertTrue(np.isfinite(res.bias))
        self.assertTrue(np.isfinite(res.coverage))
        self.assertGreater(res.coverage, 0.5)


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    unittest.main()

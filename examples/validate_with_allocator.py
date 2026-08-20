"""
Validate geoinference's standard errors on the REAL geo_sampling -> allocator
pipeline, then show the production estimate path.

What this does, end to end:

1. Take a city's road network (the bundled Delhi segments by default, or any
   roads CSV via ``--roads``) and densify it into a candidate universe.
2. Draw a probability sample of locations and route them into itineraries with
   the real ``allocator`` (this is the design you actually run in the field),
   spread over a multi-week operation.
3. Overlay a known space-time data-generating process and Monte-Carlo
   ``geoinference.estimate``: does each SE method's CI cover the true CITY mean?

Two scenarios make the design lessons concrete:

  A. Spatial only. Because geo_sampling selects points by SRS and the allocator
     only *partitions* them afterwards, the realized sample is SRS — so the
     naive SE is well-calibrated and the cluster-robust SE is conservative
     (it over-covers), even though within-itinerary spatial correlation is real.
     This is the Abadie-Athey-Imbens-Wooldridge point on real geometry: the
     sampling design, not within-cluster correlation, decides whether to cluster.

  B. Time-of-day. Add a diurnal pattern and compare synchronized vs staggered
     shift start times: synchronized starts bias beta toward the sampled
     time-of-day (more data does not help); staggering fixes it.

Run:
    python examples/validate_with_allocator.py
    python examples/validate_with_allocator.py --method random_partition --n-sims 300

Requires the pipeline extra:  pip install geoinference[pipeline]
(or: uv pip install -e ../allocator ../geo_sampling)
"""

import argparse

import numpy as np
import pandas as pd

from geoinference import PointDesign, estimate
from geoinference.pipeline import points_from_roads, subsample_scene
from geoinference.simulate import (
    PopulationFactory,
    SimConfig,
    evaluate_scene,
    results_table,
)

DEFAULT_ROADS = "../allocator/examples/inputs/delhi-roads-1k.csv"
SE_METHODS = ["naive", "cluster", "wcb"]


def _universe(args: argparse.Namespace):
    print(
        f"Building candidate universe from {args.roads} "
        f"(densify x{args.per_segment}) ..."
    )
    return points_from_roads(args.roads, per_segment=args.per_segment)


def _coverage_table(factory, sample_idx, scene, cfg, title):
    print(f"\n{title}")
    results = [
        evaluate_scene(
            factory,
            sample_idx,
            scene.itinerary_id,
            scene.time_of_day_min,
            scene.timestamp_s,
            cfg,
            se_method=m,
            spatial_diag=False,
            label=m,
        )
        for m in SE_METHODS
    ]
    print(results_table(results))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roads", default=DEFAULT_ROADS)
    parser.add_argument("--per-segment", type=int, default=8)
    parser.add_argument("--method", default="kmeans_tsp")
    parser.add_argument("--n-sample", type=int, default=400)
    parser.add_argument("--n-itineraries", type=int, default=80)
    parser.add_argument("--days", type=int, default=14)
    parser.add_argument("--n-sims", type=int, default=200)
    parser.add_argument("--range-m", type=float, default=800.0)
    args = parser.parse_args(argv)

    uni = _universe(args)
    lon = uni["longitude"].to_numpy()
    lat = uni["latitude"].to_numpy()
    print(f"Universe: {len(uni)} candidate locations")

    # --- Scenario A: spatial only -------------------------------------------
    sample_idx, scene = subsample_scene(
        uni,
        n_sample=args.n_sample,
        method=args.method,
        n_itineraries=args.n_itineraries,
        seed=1,
        stagger_starts=True,
        days=args.days,
    )
    frac = len(scene) / len(uni)
    print(
        f"Survey: {len(scene)} frames in {scene.n_itineraries} itineraries over "
        f"{scene.day_span:.1f} days  "
        f"(sampling fraction f={frac:.3f}, method={args.method})"
    )

    cfg_sp = SimConfig(
        range_s_m=args.range_m, diurnal_amp=0.0, sd_t=0.0, n_sims=args.n_sims
    )
    factory = PopulationFactory(cfg_sp, lon, lat)
    _coverage_table(
        factory,
        sample_idx,
        scene,
        cfg_sp,
        "[A] Spatial-only DGP — coverage of the city mean (nominal 0.95):",
    )
    print(
        "    Expect: naive well-calibrated (SRS selection); cluster/wcb conservative\n"
        "    (within-itinerary correlation is real but does not require clustering)."
    )

    # One full estimate showing the real-geometry diagnostics.
    rng = np.random.default_rng(7)
    pop = factory.draw(rng)
    p = pop.p_at(sample_idx, scene.time_of_day_min)
    frames = pd.DataFrame(
        {
            "n_women": p,
            "n_people": np.ones(len(scene)),
            "itinerary_id": scene.itinerary_id,
            "longitude": lon[sample_idx],
            "latitude": lat[sample_idx],
            "timestamp": scene.timestamp_s,
        }
    )
    res = estimate(
        frames,
        "n_women",
        "n_people",
        design=PointDesign(sampling="srs", cluster_var="itinerary_id"),
        bootstrap=False,
        lon_var="longitude",
        lat_var="latitude",
        time_var="timestamp",
    )
    print("\n  One realized survey, full diagnostics:")
    print(res.summary())

    # --- Scenario B: time-of-day bias ---------------------------------------
    cfg_t = SimConfig(
        range_s_m=args.range_m,
        diurnal_amp=1.2,
        range_t_min=60.0,
        sd_t=0.4,
        n_sims=args.n_sims,
    )
    factory_t = PopulationFactory(cfg_t, lon, lat)
    for stagger, tag in [(False, "synchronized starts"), (True, "staggered starts")]:
        idx_b, scene_b = subsample_scene(
            uni,
            n_sample=args.n_sample,
            method=args.method,
            n_itineraries=args.n_itineraries,
            seed=2,
            stagger_starts=stagger,
            days=args.days,
        )
        _coverage_table(
            factory_t,
            idx_b,
            scene_b,
            cfg_t,
            f"[B] Diurnal DGP, {tag} — watch bias & coverage:",
        )
    print(
        "    Expect: synchronized starts bias beta hard (low coverage); staggering\n"
        "    shrinks it sharply, with residual time-of-day imbalance fading as the\n"
        "    number of shifts grows. Time-of-day representativeness drives bias here,\n"
        "    not the SE method."
    )


if __name__ == "__main__":
    main()

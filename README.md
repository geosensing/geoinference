# geoinference

Design-based inference for spatially distributed observation surveys.

[![PyPI](https://img.shields.io/pypi/v/geoinference.svg)](https://pypi.org/project/geoinference/)
[![CI](https://github.com/geosensing/geoinference/actions/workflows/ci.yml/badge.svg)](https://github.com/geosensing/geoinference/actions/workflows/ci.yml)
[![Docs](https://github.com/geosensing/geoinference/actions/workflows/docs.yml/badge.svg)](https://geosensing.github.io/geoinference/)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)

## What this does

You collected street-level data using [geo-sampling](https://github.com/geosensing/geo-sampling) and [allocator](https://github.com/geosensing/allocator). You have a DataFrame of annotated frames with women counts and people counts. You want an unbiased estimate of the gender ratio with a correct standard error.

`geoinference` takes the annotated data plus a description of how it was collected (the *design*) and produces point estimates, standard errors, confidence intervals, and diagnostics — with the right SE estimator chosen automatically.

## Install

```bash
pip install geoinference
```

## Quick start

```python
from geoinference import PointDesign, WalkDesign, estimate

# Point-based pipeline (geo-sampling → allocator → annotate)
design = PointDesign(sampling="srs", cluster_var="itinerary_id")
result = estimate(df, "n_women", "n_people", design=design)
print(result.summary())

# Walk-based pipeline (random walks with GoPro)
design = WalkDesign(walk_var="walk_id")
result = estimate(df, "n_women", "n_people", design=design)
print(result.summary())
```

## Two estimands

**Ratio** (people-weighted): `sum(women) / sum(people)` — what fraction of observed *people* are women.

**Photo-level mean** (location-weighted): `mean(w_i / h_i)` for frames with `h_i > 0` — at a random *location*, what fraction of people present are women.

## Two designs

**PointDesign**: Discrete locations sampled by geo-sampling, grouped into itineraries by allocator. Clustered designs default to the **cluster-robust SE** (with a `t_{G-1}` CI for few itineraries) — the safe choice. When points are an SRS/probability sample and itineraries are only a *post-hoc* bundling (the usual case), the naive SE is also valid and a bit tighter, while the cluster SE is conservative; when the routing itself determines which points get observed, the cluster SE is required and the naive SE under-covers. Simulation (below) lets you check which regime you're in.

**WalkDesign**: Continuous random walks on the road network. Walks are independent by construction. The between-walk SE is the correct and only estimator.

## Within-itinerary dependence diagnostics

If the annotated frames carry coordinates and/or a timestamp, pass them and `estimate` reports how much within-itinerary correlation there is and how fast it decays — separately on the **spatial** (great-circle) and **temporal** (time-gap) axes:

```python
result = estimate(
    df,
    "n_women",
    "n_people",
    design=PointDesign(sampling="srs", cluster_var="itinerary_id"),
    lon_var="longitude",
    lat_var="latitude",
    time_var="timestamp",
)
```

You get an empirical variogram range, Moran's I (+ permutation p-value), a variogram-based **effective sample size** per axis (Griffith 2005; Watson 2021), and a within- vs between-itinerary semivariance ratio — so you can see whether the (stochastic) itinerary partition actually costs you information, and on which axis.

## Estimate from a file (production)

```python
from geoinference import estimate_from_file

result = estimate_from_file("frames.parquet")  # uses any optional columns present
print(result.summary())
```

Expected columns (all configurable): `n_women`, `n_people` (required), `itinerary_id`, `longitude`, `latitude`, `timestamp` (optional — diagnostics turn on when present). Or from the shell: `python -m geoinference.io estimate frames.parquet`.

The reader picks its format from the suffix: `.parquet`/`.pq`, or `.csv` (compressed or not). **Prefer Parquet.** CSV carries no types, so a count column with one missing value comes back as float and `timestamp` comes back as a string something downstream has to parse; Parquet gives back what the pipeline wrote.

## Validate a design before you trust the numbers

`geoinference.simulate` + `geoinference.pipeline` (install the `pipeline` extra) overlay a known space-time process on the **real** geo-sampling → allocator geometry and Monte-Carlo whether the CIs cover the truth:

```bash
pip install geoinference[pipeline]
python examples/validate_with_allocator.py
```

This reports bias, SE calibration, and coverage by SE method on a realistic survey (hundreds of itineraries over weeks), and surfaces failure modes — e.g. synchronized shift start-times biasing the estimate when the outcome varies by time of day.

## What you get back

```
============================================================
geoinference: Inference Result
============================================================
Design: point_srs_clustered
Observations: 200 (195 with h>0, 5 empty)
Clusters: 10

── Ratio estimand (people-weighted) ──
  Estimate:  0.2987
  SE:        0.0109  (cluster)
  95% CI:    [0.2773, 0.3201]

── Photo-level mean (location-weighted) ──
  Estimate:  0.3014
  SE:        0.0128  (cluster)
  95% CI:    [0.2764, 0.3265]

── Diagnostics ──
  ICC:                0.0312
  Design effect:      1.59
  Effective N:        123.0
  SE ratio (cl/nv):   1.142
  Ratio bias O(1/N):  -0.000089
============================================================
```

## License

MIT

# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- **`run_pipeline(ci_level=...)` and `evaluate_scene(ci_level=...)` were
  ignored by the pairs bootstrap.** `_method_ci` builds the naive/cluster/auto
  intervals itself from `ci_level` and forwards it to the wild cluster
  bootstrap, but for `se_method="boot"` it reads `res.ratio_ci.bootstrap`
  straight off the result — and the `estimate` call omitted `ci_level`, so that
  interval was always the 95% default. Measured coverage came out **identical
  at every requested level** (0.817 at 0.50, 0.80 and 0.99), which makes a
  bootstrap coverage table at any level but 0.95 fiction. It now tracks the
  request: 0.517 / 0.733 / 0.883. This is the same defect as the `ci_level` fix
  in #12, one layer up: #12 taught `_cluster_bootstrap` to honour the level,
  and the simulation layer then still failed to ask for it.
- Each replication of the Monte Carlo test harness now draws its own bootstrap
  seed. Every replication previously resampled with `estimate`'s default seed,
  so the bootstrap noise was common to all of them — not something a coverage
  study is entitled to assume. The seed comes from a separate stream so the
  data seeds, which the coverage thresholds are tuned against, do not move.
- **`estimate(max_dependence_points=N)` did nothing.** The argument was
  documented, `estimate_from_file` forwarded it, and `_dependence_diagnostics`
  reads it and subsamples above it — but the call inside `estimate` omitted it,
  so every run used the 2500 default. Asking for a cheaper diagnostic on a large
  frame set changed nothing. Regression tests assert the requested cap appears
  in the subsampling warning.

### Removed

- **`pipeline.sample_points` and the example's `--live` flag.** The function
  called `geo_sampling.sample_roads_for_region` and `geo_sampling.RoadSampler`.
  Neither exists: `geo_sampling` exports nothing at the top level, being a pair
  of CLI scripts (`geo_roads`, `sample_roads`). The function could never have
  run, and pyright said so as soon as the standard turned it on. Build a
  universe from a roads file with `points_from_roads` instead, which is what
  the example and the tests already do.

### Changed

- **`estimate_from_csv` is now `estimate_from_file`**, and the format comes from
  the suffix: `.parquet`/`.pq`, or `.csv`/`.tsv` with optional compression. A
  new `read_frames` exposes the reader on its own. Parquet is the format to
  prefer and what the docs now show: CSV carries no types, so a count column
  with one missing value returns as float and `timestamp` returns as a string
  that something downstream has to parse. A test asserts the two formats give
  the same estimate to twelve places, and another pins the dtype difference.
- **Adopted the shared `py-canon` project standard.** CI, docs, release and
  Dependabot workflows call `gojiplus/py-canon` reusable workflows instead of
  carrying their own copies, so a fix published there reaches this repo on its
  next run. `python-publish.yml` is superseded by `release.yml`.
- Linting is the fleet ruff configuration (line length 88, the standard rule
  selection, google docstring convention) and type checking is pyright rather
  than mypy, with pydoclint on docstrings. `make check` runs what CI runs.
- **The Python floor is 3.12**, up from 3.11, matching the fleet standard. The
  package has never been published, so nothing depended on 3.11.
- `__version__` is read from the installed metadata instead of being a second
  copy of the number in source.
- CI enforces a coverage floor of 80% (measured 84%).

### Added

- A changelog, a `CITATION.cff`, and a pre-commit configuration.
- The `>>>` examples in the package and `estimate` docstrings are real
  doctests now: they build their own frame and assert a value, so the docs
  build checks them. They previously referenced an undefined `df`.

[Unreleased]: https://github.com/geosensing/geoinference/commits/main

# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- **`estimate(max_dependence_points=N)` did nothing.** The argument was
  documented, `estimate_from_file` forwarded it, and `_dependence_diagnostics`
  reads it and subsamples above it — but the call inside `estimate` omitted it,
  so every run used the 2500 default. Asking for a cheaper diagnostic on a large
  frame set changed nothing. Regression tests assert the requested cap appears
  in the subsampling warning.

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

[Unreleased]: https://github.com/geosensing/geoinference/commits/main

"""Run geoinference on a real annotated-frames table — the production endpoint.

``estimate_from_file`` reads an annotated-frames table (one row per frame, as
produced downstream of the geosensing pipeline) and returns a full
``InferenceResult``. Coordinate/time columns are used only when present, so the
spatial/temporal dependence diagnostics turn on automatically when the data
carries ``longitude``/``latitude``/``timestamp``.

Parquet is the format to prefer. It carries its own types, so counts arrive as
integers and ``timestamp`` arrives as a datetime; CSV has no types, so a column
with one missing value comes back as float and every timestamp comes back as a
string that something downstream has to guess the format of. CSV is still read,
because the pipeline still emits it.

Expected columns (all configurable):

    n_women       count of women in the frame              (women_var)
    n_people      total people in the frame                (people_var)
    itinerary_id  the itinerary/route the frame belongs to (cluster_var)
    longitude     decimal degrees, optional                (lon_var)
    latitude      decimal degrees, optional                (lat_var)
    timestamp     datetime or epoch seconds, optional      (time_var)

CLI:  python -m geoinference.io estimate frames.parquet [--cluster-var itinerary_id]
"""

import argparse
from pathlib import Path

import pandas as pd

from .designs import PointDesign
from .inference import (
    DEFAULT_MAX_DEPENDENCE_POINTS,
    DEFAULT_T_INTERVAL_BELOW_CLUSTERS,
    DEFAULT_WARN_ABOVE_CLUSTER_SIZE_CV,
    estimate,
)
from .types import InferenceResult

PARQUET_SUFFIXES = {".parquet", ".pq"}
CSV_SUFFIXES = {".csv", ".tsv", ".txt"}
COMPRESSION_SUFFIXES = {".gz", ".bz2", ".zip", ".xz", ".zst"}


def read_frames(path: str | Path) -> pd.DataFrame:
    """Read an annotated-frames table, choosing the reader by suffix.

    Args:
        path: Path to a ``.parquet``/``.pq`` or ``.csv`` file, the latter
            optionally compressed.

    Returns:
        The table, with whatever types the file carries.

    Raises:
        ValueError: If the suffix names no format this reads.
    """
    suffixes = [s.lower() for s in Path(path).suffixes]
    if suffixes and suffixes[-1] in COMPRESSION_SUFFIXES:
        suffixes = suffixes[:-1]
    suffix = suffixes[-1] if suffixes else ""

    if suffix in PARQUET_SUFFIXES:
        return pd.read_parquet(path)
    if suffix in CSV_SUFFIXES:
        return pd.read_csv(path, sep="\t" if suffix == ".tsv" else ",")
    raise ValueError(
        f"Cannot tell how to read {path!r}: expected one of "
        f"{sorted(PARQUET_SUFFIXES | CSV_SUFFIXES)}, optionally compressed."
    )


def estimate_from_file(
    path: str | Path,
    women_var: str = "n_women",
    people_var: str = "n_people",
    cluster_var: str | None = "itinerary_id",
    lon_var: str | None = "longitude",
    lat_var: str | None = "latitude",
    time_var: str | None = "timestamp",
    sampling: str = "srs",
    ci_level: float = 0.95,
    bootstrap: bool = True,
    bootstrap_reps: int = 2000,
    seed: int = 42,
    warn_above_cluster_size_cv: float | None = DEFAULT_WARN_ABOVE_CLUSTER_SIZE_CV,
    t_interval_below_clusters: int = DEFAULT_T_INTERVAL_BELOW_CLUSTERS,
    max_dependence_points: int = DEFAULT_MAX_DEPENDENCE_POINTS,
) -> InferenceResult:
    """Estimate the population ratio from an annotated-frames table.

    The format is taken from the suffix: ``.parquet``/``.pq``, or ``.csv``
    (``.gz``/``.bz2``/``.zip``/``.xz`` compression included). Prefer Parquet,
    which keeps the column types the pipeline wrote.

    Optional columns (``cluster_var``, ``lon_var``, ``lat_var``, ``time_var``)
    are silently ignored when absent from the file, so the same call works on
    minimal and fully-attributed exports. ``women_var`` and ``people_var`` are
    required.

    Args:
        path: Path to the annotated-frames table.
        women_var: Required column with the count of women per frame.
        people_var: Required column with the total people per frame.
        cluster_var: Itinerary/cluster column (used iff present).
        lon_var: Longitude column; with lat_var, enables spatial diagnostics.
        lat_var: Latitude column; with lon_var, enables spatial diagnostics.
        time_var: Timestamp column; enables temporal diagnostics.
        sampling: PointDesign sampling scheme ("srs", "pps", "grts").
        ci_level: Coverage the intervals claim.
        bootstrap: Whether to compute cluster bootstrap intervals.
        bootstrap_reps: Number of bootstrap replications.
        seed: Random seed for the bootstrap and the dependence diagnostics.
        warn_above_cluster_size_cv: Warn above this cluster-size CV.
        t_interval_below_clusters: Use a t interval below this cluster count.
        max_dependence_points: Subsample the dependence diagnostics above this
            many positive frames.

    Returns:
        The ``InferenceResult`` from ``estimate``.

    Raises:
        ValueError: If the suffix names no format this reads, or a required
            count column is missing.
    """
    df = read_frames(path)
    for col in (women_var, people_var):
        if col not in df.columns:
            raise ValueError(
                f"Required column {col!r} not in {path} (have {list(df.columns)})"
            )

    def _present(col: str | None) -> str | None:
        return col if col is not None and col in df.columns else None

    design = PointDesign(sampling=sampling, cluster_var=_present(cluster_var))  # type: ignore[arg-type]
    return estimate(
        df,
        women_var,
        people_var,
        design=design,
        ci_level=ci_level,
        bootstrap=bootstrap,
        bootstrap_reps=bootstrap_reps,
        seed=seed,
        lon_var=_present(lon_var),
        lat_var=_present(lat_var),
        time_var=_present(time_var),
        warn_above_cluster_size_cv=warn_above_cluster_size_cv,
        t_interval_below_clusters=t_interval_below_clusters,
        max_dependence_points=max_dependence_points,
    )


def main(argv: list[str] | None = None) -> None:
    """CLI: ``python -m geoinference.io estimate frames.parquet``."""
    parser = argparse.ArgumentParser(prog="geoinference.io", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    est = sub.add_parser(
        "estimate", help="estimate from an annotated-frames Parquet or CSV file"
    )
    est.add_argument("path")
    est.add_argument("--women-var", default="n_women")
    est.add_argument("--people-var", default="n_people")
    est.add_argument("--cluster-var", default="itinerary_id")
    est.add_argument("--lon-var", default="longitude")
    est.add_argument("--lat-var", default="latitude")
    est.add_argument("--time-var", default="timestamp")
    est.add_argument("--sampling", default="srs")
    est.add_argument("--no-bootstrap", action="store_true")
    args = parser.parse_args(argv)

    result = estimate_from_file(
        args.path,
        women_var=args.women_var,
        people_var=args.people_var,
        cluster_var=args.cluster_var,
        lon_var=args.lon_var,
        lat_var=args.lat_var,
        time_var=args.time_var,
        sampling=args.sampling,
        bootstrap=not args.no_bootstrap,
    )
    print(result.summary())


if __name__ == "__main__":
    main()

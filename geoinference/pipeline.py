"""Bridge from the real geosensing pipeline to a geoinference simulation scene.

This turns the actual ``geo_sampling`` → ``allocator`` output into a fixed
"scene" — point coordinates, the itinerary partition, and per-frame visit
times spread over a multi-day field operation — that
``geoinference.simulate.evaluate_scene`` can validate a DGP against, and that
mirrors what the annotated frames look like in production.

``allocator`` and ``geo_sampling`` are optional; install them with
``pip install geoinference[pipeline]`` (or ``uv pip install -e ../allocator
../geo_sampling`` for local checkouts). They are imported lazily so core
geoinference keeps no heavy dependencies.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .spatial import haversine_matrix

_PIPELINE_HINT = "install the pipeline extra:  pip install geoinference[pipeline]"


@dataclass
class Scene:
    """A realized field operation: where, in which itinerary, and when.

    All arrays are per annotated frame and aligned by position.
    """

    longitude: np.ndarray
    latitude: np.ndarray
    itinerary_id: np.ndarray
    timestamp_s: np.ndarray  # absolute seconds across the whole operation
    time_of_day_min: np.ndarray  # minutes within the field "day" (drives diurnal DGP)

    def __len__(self) -> int:
        """Return the number of frames.

        Returns:
            One per longitude entry.
        """
        return len(self.longitude)

    def to_frame(self) -> pd.DataFrame:
        """Annotated-frame layout (minus the outcome columns)."""
        return pd.DataFrame(
            {
                "itinerary_id": self.itinerary_id,
                "longitude": self.longitude,
                "latitude": self.latitude,
                "timestamp": self.timestamp_s,
            }
        )

    @property
    def n_itineraries(self) -> int:
        """Count the distinct itineraries the frames belong to.

        Returns:
            The number of unique itinerary ids.
        """
        return int(np.unique(self.itinerary_id).size)

    @property
    def day_span(self) -> float:
        """Number of days the operation spans (from absolute timestamps)."""
        if len(self) == 0:
            return 0.0
        return float((self.timestamp_s.max() - self.timestamp_s.min()) / 86_400.0)


def points_from_roads(roads: pd.DataFrame | str, per_segment: int = 1) -> pd.DataFrame:
    """Point locations along road segments, from the road-segment schema.

    Accepts a DataFrame or a CSV path with the ``geo_sampling`` / ``allocator``
    columns ``start_lat, start_long, end_lat, end_long`` (and returns
    ``longitude``/``latitude``). If the frame already has ``longitude`` /
    ``latitude`` it is returned unchanged.

    ``per_segment`` interpolates that many evenly-spaced points along each
    segment (1 = midpoint). Densifying gives a larger candidate universe — and
    hence a small sampling fraction — for realistic validation.
    """
    df = pd.read_csv(roads) if isinstance(roads, str) else roads
    if {"longitude", "latitude"}.issubset(df.columns):
        return pd.DataFrame(
            {
                "longitude": df["longitude"].to_numpy(dtype=float),
                "latitude": df["latitude"].to_numpy(dtype=float),
            }
        )
    needed = {"start_lat", "start_long", "end_lat", "end_long"}
    if not needed.issubset(df.columns):
        raise ValueError(
            f"roads must have {sorted(needed)} or longitude/latitude; "
            f"got {list(df.columns)}"
        )
    s_lon = df["start_long"].to_numpy(dtype=float)
    s_lat = df["start_lat"].to_numpy(dtype=float)
    e_lon = df["end_long"].to_numpy(dtype=float)
    e_lat = df["end_lat"].to_numpy(dtype=float)
    # Fractions at segment-interior points (midpoint for per_segment == 1).
    fracs = (np.arange(per_segment) + 0.5) / per_segment
    lon = (s_lon[:, None] + fracs[None, :] * (e_lon - s_lon)[:, None]).ravel()
    lat = (s_lat[:, None] + fracs[None, :] * (e_lat - s_lat)[:, None]).ravel()
    return pd.DataFrame({"longitude": lon, "latitude": lat})


def build_itineraries(
    points: pd.DataFrame,
    method: str = "random_partition",
    n_itineraries: int | None = None,
    max_distance: float | None = None,
    seed: int | None = None,
) -> tuple[pd.DataFrame, list[list[int]]]:
    """Partition points into itineraries with ``allocator`` (offline haversine).

    Returns ``(data, routes)`` where ``data`` is the points frame plus an
    ``itinerary_id`` column and ``routes`` is the list of itineraries as
    point-index lists in visit order. Lazily imports ``allocator``.
    """
    try:
        from allocator import (  # pyright: ignore[reportMissingImports]
            create_itineraries,
        )
    except ImportError as exc:  # pragma: no cover - exercised only without extra
        raise ImportError(f"allocator not available; {_PIPELINE_HINT}") from exc

    result = create_itineraries(
        points,
        method=method,
        n_itineraries=n_itineraries,
        max_distance=max_distance,
        distance="haversine",
        seed=seed,
    )
    routes = [[int(i) for i in route] for route in result.itineraries]
    return result.data.reset_index(drop=True), routes


def assign_visit_times(
    points: pd.DataFrame,
    routes: list[list[int]],
    *,
    days: int = 14,
    shifts_per_day: int = 1,
    speed_m_per_min: float = 80.0,
    dwell_min: float = 2.0,
    day_minutes: float = 600.0,
    stagger_starts: bool = True,
    seed: int | None = None,
) -> Scene:
    """Spread itineraries over a multi-day operation and time every frame.

    Each itinerary is a single shift assigned to one (day, slot). Within a
    shift, visit times accumulate along the route as travel (haversine metres /
    ``speed_m_per_min``) plus a per-point ``dwell_min``. Shifts are spread
    round-robin across ``days × shifts_per_day`` so timestamps span the whole
    operation while each shift's frames stay close in time-of-day.

    Returns a ``Scene`` whose ``time_of_day_min`` drives the diurnal field and
    whose ``timestamp_s`` is the absolute time fed to ``estimate``'s temporal
    diagnostic.
    """
    lon = points["longitude"].to_numpy(dtype=float)
    lat = points["latitude"].to_numpy(dtype=float)
    n = len(lon)
    rng = np.random.default_rng(seed)

    itinerary_id = np.full(n, -1, dtype=int)
    time_of_day = np.full(n, np.nan)
    timestamp_s = np.full(n, np.nan)
    n_slots = max(1, days * shifts_per_day)
    slot_len_min = day_minutes / shifts_per_day

    for k, route in enumerate(routes):
        if not route:
            continue
        slot = k % n_slots
        day = slot // shifts_per_day
        within_day_slot = slot % shifts_per_day
        if stagger_starts:
            # Spread shift start-of-day uniformly → temporally representative.
            start_tod = float(rng.uniform(0.0, day_minutes))
        else:
            # All shifts start at the same clock time → synchronized.
            start_tod = within_day_slot * slot_len_min

        t = start_tod
        prev = route[0]
        for pos, pt in enumerate(route):
            if pos > 0:
                d_m = float(
                    haversine_matrix(
                        np.array([lon[prev], lon[pt]]), np.array([lat[prev], lat[pt]])
                    )[0, 1]
                )
                t += d_m / speed_m_per_min
            t += dwell_min
            itinerary_id[pt] = k
            time_of_day[pt] = t
            timestamp_s[pt] = (day * 86_400.0) + t * 60.0
            prev = pt

    # Any unrouted points (shouldn't happen) drop out of the scene.
    keep = itinerary_id >= 0
    return Scene(
        longitude=lon[keep],
        latitude=lat[keep],
        itinerary_id=itinerary_id[keep],
        timestamp_s=timestamp_s[keep],
        time_of_day_min=time_of_day[keep],
    )


def make_scene(
    roads: pd.DataFrame | str,
    method: str = "random_partition",
    n_itineraries: int = 200,
    seed: int | None = 0,
    **time_kwargs: object,
) -> Scene:
    """Convenience: roads CSV/frame → points → itineraries → timed ``Scene``."""
    points = points_from_roads(roads)
    data, routes = build_itineraries(
        points, method=method, n_itineraries=n_itineraries, seed=seed
    )
    return assign_visit_times(data, routes, seed=seed, **time_kwargs)  # type: ignore[arg-type]


def subsample_scene(
    universe: pd.DataFrame | str,
    n_sample: int,
    method: str = "kmeans_tsp",
    n_itineraries: int = 80,
    seed: int = 0,
    stagger_starts: bool = True,
    days: int = 14,
) -> tuple[np.ndarray, Scene]:
    """Sample a survey out of a city universe and route it into itineraries.

    Treats ``universe`` (all candidate road segments for a city) as the
    population, draws ``n_sample`` of them (SRS — a stand-in for live
    ``geo_sampling``), and routes the sample with the allocator. Returns
    ``(sample_idx, scene)`` where ``sample_idx`` indexes the universe (so the
    field can be drawn on the whole city and the sample scored against the city
    mean in ``geoinference.simulate.evaluate_scene``).
    """
    points = points_from_roads(universe)
    n_uni = len(points)
    rng = np.random.default_rng(seed)
    sample_idx = np.sort(rng.choice(n_uni, size=min(n_sample, n_uni), replace=False))
    pts = points.iloc[sample_idx].reset_index(drop=True)
    data, routes = build_itineraries(
        pts, method=method, n_itineraries=n_itineraries, seed=seed
    )
    scene = assign_visit_times(
        data, routes, days=days, stagger_starts=stagger_starts, seed=seed
    )
    if len(scene) != len(sample_idx):
        raise RuntimeError("allocator dropped points; sample/scene misaligned")
    return sample_idx, scene

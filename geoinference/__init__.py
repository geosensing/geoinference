"""geoinference: Design-based inference for spatially distributed observation surveys.

Takes annotated frame data from the geosensing pipeline (geo-sampling + allocator)
and produces correct point estimates, standard errors, and confidence intervals,
with the right SE estimator chosen automatically based on the collection design.

Quick start:
    >>> from geoinference import PointDesign, estimate
    >>> design = PointDesign(sampling="srs", cluster_var="itinerary_id")
    >>> result = estimate(df, "n_women", "n_people", design=design)
    >>> print(result.summary())
"""

from importlib.metadata import version

from .designs import Design, PointDesign, WalkDesign
from .inference import estimate
from .io import estimate_from_file, read_frames
from .types import CIResult, Diagnostics, InferenceResult, SEResult

__version__ = version("geoinference")

__all__ = [
    "CIResult",
    "Design",
    "Diagnostics",
    "InferenceResult",
    "PointDesign",
    "SEResult",
    "WalkDesign",
    "estimate",
    "estimate_from_file",
    "read_frames",
]

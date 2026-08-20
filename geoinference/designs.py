"""Design specifications for geoinference.

A design object encodes how the data was collected — the sampling
mechanism, the clustering structure, the weighting scheme — so that
the inference functions can choose the correct variance estimator.

The design is metadata about the data, not the data itself.
"""

from dataclasses import dataclass
from typing import Literal


@dataclass
class PointDesign:
    """Design for point-based spatial sampling (geo-sampling + allocator pipeline).

    The data consists of observations at discrete sampled locations,
    grouped into itineraries for field collection.

    Args:
        sampling: Sampling method used.
            - "srs": Simple random sampling (equal probability).
            - "grts": Generalized Random Tessellation Stratified.
            - "pps": Probability proportional to size.
        cluster_var: Column name identifying itineraries/clusters.
            If None, observations are treated as independent (no clustering).
        weight_var: Column name with inclusion probabilities or design weights.
            Required for "pps" and "grts" designs. Ignored for "srs".
        annotation_frac: Fraction of collected frames that were annotated.
            If < 1.0, inference accounts for the annotation subsampling.
            Can also be a column name for observation-level annotation probabilities.
        fpc: Finite population correction. If provided, the population size M
            from which N units were sampled. Reduces variance by factor (1 - N/M).
    """

    sampling: Literal["srs", "grts", "pps"] = "srs"
    cluster_var: str | None = None
    weight_var: str | None = None
    annotation_frac: float | str = 1.0
    fpc: int | None = None

    @property
    def name(self) -> str:
        """Return a slug naming the sampling scheme and what it carries.

        Returns:
            e.g. ``"point_srs_clustered"``.
        """
        parts = [f"point_{self.sampling}"]
        if self.cluster_var:
            parts.append("clustered")
        if self.weight_var:
            parts.append("weighted")
        return "_".join(parts)

    @property
    def has_clusters(self) -> bool:
        """Report whether observations are grouped into clusters.

        Returns:
            True when a cluster column was given.
        """
        return self.cluster_var is not None

    @property
    def has_weights(self) -> bool:
        """Report whether observations carry design weights.

        Returns:
            True when a weight column was given.
        """
        return self.weight_var is not None

    @property
    def recommended_se_method(self) -> str:
        """Which SE estimator the design recommends as primary.

        Any clustered design (an explicit ``cluster_var``, or PPS/GRTS) gets the
        cluster-robust SE. Simulation shows that when itineraries bundle nearby
        points and the outcome is spatially autocorrelated, the naive SE
        undercovers badly (95% CI coverage falling toward ~0.5 as correlation
        grows), while the cluster-robust SE — paired with the t_{G-1} CI used
        for few clusters — holds near nominal. The cost is mild conservatism
        under genuine independence, where the cluster SE ≈ the naive SE anyway.
        The naive SE is still reported (and is primary only when there is no
        clustering at all).
        """
        if self.cluster_var is not None or self.sampling in ("pps", "grts"):
            return "cluster"
        return "naive"

    def __post_init__(self) -> None:
        """Reject designs that cannot be estimated as specified.

        Raises:
            ValueError: If PPS/GRTS is chosen without weights, or
                ``annotation_frac`` falls outside (0, 1].
        """
        if self.sampling in ("pps", "grts") and self.weight_var is None:
            raise ValueError(
                f"Design '{self.sampling}' requires weight_var "
                f"(inclusion probabilities or design weights)."
            )
        if isinstance(self.annotation_frac, (int, float)) and not (
            0 < self.annotation_frac <= 1.0
        ):
            raise ValueError("annotation_frac must be in (0, 1].")


@dataclass
class WalkDesign:
    """Design for random-walk transect sampling.

    The data consists of observations along independent random walks
    on the road network. Each walk is an independent Markov chain
    realization, and inference uses between-walk variance.

    Args:
        walk_var: Column name identifying independent walks.
        spacing_m: Annotation spacing along the walk in meters.
            Used to compute effective sample size per walk
            (n_eff ≈ walk_length / (2 * correlation_length)).
            If None, all annotated frames are used without
            effective-N adjustment.
    """

    walk_var: str = "walk_id"
    spacing_m: float | None = None

    @property
    def name(self) -> str:
        """Return the design slug.

        Returns:
            Always ``"walk_transect"``.
        """
        return "walk_transect"

    @property
    def has_clusters(self) -> bool:
        """Report whether observations are grouped into clusters.

        Returns:
            Always True: the walk is the clustering unit.
        """
        return True

    @property
    def cluster_var(self) -> str:
        """Return the column that identifies a cluster.

        Returns:
            The walk column, since walks are the clusters.
        """
        return self.walk_var

    @property
    def has_weights(self) -> bool:
        """Report whether observations carry design weights.

        Returns:
            Always False: a walk transect is self-weighting by construction.
        """
        return False

    @property
    def recommended_se_method(self) -> str:
        """Between-walk SE is always the correct estimator for walk designs."""
        return "cluster"

    def __post_init__(self) -> None:
        """Reject a non-positive annotation spacing.

        Raises:
            ValueError: If ``spacing_m`` is given and not positive.
        """
        if self.spacing_m is not None and self.spacing_m <= 0:
            raise ValueError("spacing_m must be positive.")


# Union type for design dispatch
Design = PointDesign | WalkDesign

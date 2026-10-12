"""Component scores and the weighted confidence model.

Every analysis component produces a score in ``[-1, 1]`` that is already
oriented to the proposed trade: +1 strongly supports it, -1 strongly argues
against it, 0 is neutral. ``None`` means the component could not be
evaluated, and its weight is redistributed over the components that could.
"""

from dataclasses import dataclass, field

import numpy as np


COMPONENT_WEIGHTS = {
    "Signal quality": 0.25,
    "Trend alignment": 0.20,
    "Volume / accumulation": 0.15,
    "Intraday entry": 0.15,
    "Fundamentals": 0.15,
    "Historical edge": 0.10,
}


@dataclass
class ComponentResult:
    name: str
    score: float = None
    details: list = field(default_factory=list)
    metrics: dict = field(default_factory=dict)

    @property
    def available(self):
        return self.score is not None and np.isfinite(self.score)


def mean_score(values):
    """Average of the finite values, or None when there are none."""

    finite = [value for value in values if value is not None and np.isfinite(value)]
    if not finite:
        return None
    return float(np.mean(finite))


def weighted_score(pairs):
    """Weighted mean of ``(score, weight)`` pairs, skipping non-finite scores."""

    usable = [(score, weight) for score, weight in pairs if score is not None and np.isfinite(score)]
    total_weight = sum(weight for _, weight in usable)
    if total_weight <= 0:
        return None
    return float(sum(score * weight for score, weight in usable) / total_weight)


def combine_confidence(components, weights=None):
    """Return ``(confidence 0-100, share of total weight that was available)``."""

    weights = weights or COMPONENT_WEIGHTS
    pairs = [(component.score, weights.get(component.name, 0.0)) for component in components]
    combined = weighted_score(pairs)
    available_weight = sum(weight for score, weight in pairs if score is not None and np.isfinite(score))
    total_weight = sum(weights.get(component.name, 0.0) for component in components) or 1.0
    if combined is None:
        return None, 0.0
    return round(50.0 + 50.0 * combined, 1), available_weight / total_weight

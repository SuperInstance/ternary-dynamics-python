"""Phase transition detection for ternary agent dynamics.

Detects sudden shifts in population behavior by looking for large
changes in the rate-of-change (second derivative) of key metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .time_series import TimeSeries, Snapshot


@dataclass
class PhaseTransition:
    """A detected phase transition."""

    generation: int
    metric: str  # "fitness", "diversity", "avoidance"
    magnitude: float  # how big the shift was
    direction: str  # "up" or "down"


class PhaseTransitionDetector:
    """Detect phase transitions in a TimeSeries.

    A transition is flagged when the absolute second-derivative of a metric
    exceeds *threshold* between consecutive generations.
    """

    def __init__(self, threshold: float = 0.3, window: int = 3) -> None:
        self.threshold = threshold
        self.window = max(1, window)

    def detect(self, series: TimeSeries) -> list[PhaseTransition]:
        """Return all detected phase transitions, ordered by generation."""
        results: list[PhaseTransition] = []
        for metric in ("fitness", "diversity", "avoidance"):
            results.extend(self._detect_metric(series, metric))
        results.sort(key=lambda t: t.generation)
        return results

    def _detect_metric(self, series: TimeSeries, metric: str) -> list[PhaseTransition]:
        values = getattr(series, metric)
        if len(values) < 3:
            return []

        # First derivative
        d1 = [values[i + 1] - values[i] for i in range(len(values) - 1)]
        # Second derivative
        d2 = [abs(d1[i + 1] - d1[i]) for i in range(len(d1) - 1)]

        transitions: list[PhaseTransition] = []
        for idx, accel in enumerate(d2):
            if accel > self.threshold:
                gen = series[idx + 1].generation
                direction = "up" if d1[idx + 1] > d1[idx] else "down"
                transitions.append(
                    PhaseTransition(
                        generation=gen,
                        metric=metric,
                        magnitude=round(accel, 6),
                        direction=direction,
                    )
                )
        return transitions

    def detect_all(self, series_list: Sequence[TimeSeries]) -> list[list[PhaseTransition]]:
        """Detect transitions across multiple time series runs."""
        return [self.detect(s) for s in series_list]

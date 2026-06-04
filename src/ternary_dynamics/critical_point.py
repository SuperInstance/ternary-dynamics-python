"""Critical point detection for ternary agent dynamics.

Identifies fitness plateaus, diversity drops, and other inflection
points where the dynamics change character.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence

from .time_series import TimeSeries


class CriticalPointType(str, Enum):
    FITNESS_PLATEAU = "fitness_plateau"
    DIVERSITY_DROP = "diversity_drop"
    AVOIDANCE_SPIKE = "avoidance_spike"
    FITNESS_COLLAPSE = "fitness_collapse"


@dataclass
class CriticalPoint:
    """A detected critical point."""

    generation: int
    point_type: CriticalPointType
    severity: float  # 0..1 normalized severity
    description: str


class CriticalPointDetector:
    """Identify critical points in a TimeSeries.

    Detection strategies:
    - **Fitness plateau**: fitness derivative stays near zero for *plateau_window* generations.
    - **Diversity drop**: diversity falls by more than *diversity_threshold* in one step.
    - **Avoidance spike**: avoidance exceeds *avoidance_threshold*.
    - **Fitness collapse**: fitness drops by more than *collapse_threshold* in one step.
    """

    def __init__(
        self,
        plateau_window: int = 5,
        plateau_tolerance: float = 0.01,
        diversity_threshold: float = 0.3,
        avoidance_threshold: float = 0.8,
        collapse_threshold: float = 0.4,
    ) -> None:
        self.plateau_window = plateau_window
        self.plateau_tolerance = plateau_tolerance
        self.diversity_threshold = diversity_threshold
        self.avoidance_threshold = avoidance_threshold
        self.collapse_threshold = collapse_threshold

    def detect(self, series: TimeSeries) -> list[CriticalPoint]:
        """Return detected critical points ordered by generation."""
        points: list[CriticalPoint] = []

        if len(series) < 2:
            return points

        points.extend(self._find_fitness_plateaus(series))
        points.extend(self._find_diversity_drops(series))
        points.extend(self._find_avoidance_spikes(series))
        points.extend(self._find_fitness_collapses(series))

        points.sort(key=lambda p: p.generation)
        return points

    def _find_fitness_plateaus(self, series: TimeSeries) -> list[CriticalPoint]:
        derivs = series.fitness_derivative()
        points: list[CriticalPoint] = []
        w = self.plateau_window
        tol = self.plateau_tolerance

        for i in range(len(derivs) - w + 1):
            window = derivs[i : i + w]
            if all(abs(d) <= tol for d in window):
                gen = series[i + w // 2].generation
                # Severity: lower mean abs derivative → more severe plateau
                mean_abs = sum(abs(d) for d in window) / w
                severity = max(0.0, min(1.0, 1.0 - mean_abs / tol)) if tol > 0 else 1.0
                points.append(
                    CriticalPoint(
                        generation=gen,
                        point_type=CriticalPointType.FITNESS_PLATEAU,
                        severity=round(severity, 4),
                        description=f"Fitness plateau at generation {gen} (mean |d|={mean_abs:.4f})",
                    )
                )
        return points

    def _find_diversity_drops(self, series: TimeSeries) -> list[CriticalPoint]:
        vals = series.diversity
        points: list[CriticalPoint] = []
        for i in range(1, len(vals)):
            drop = vals[i - 1] - vals[i]
            if drop >= self.diversity_threshold:
                gen = series[i].generation
                severity = min(1.0, drop / 1.0)
                points.append(
                    CriticalPoint(
                        generation=gen,
                        point_type=CriticalPointType.DIVERSITY_DROP,
                        severity=round(severity, 4),
                        description=f"Diversity drop of {drop:.4f} at generation {gen}",
                    )
                )
        return points

    def _find_avoidance_spikes(self, series: TimeSeries) -> list[CriticalPoint]:
        vals = series.avoidance
        points: list[CriticalPoint] = []
        for i, val in enumerate(vals):
            if val >= self.avoidance_threshold:
                gen = series[i].generation
                severity = min(1.0, val / 1.0)
                points.append(
                    CriticalPoint(
                        generation=gen,
                        point_type=CriticalPointType.AVOIDANCE_SPIKE,
                        severity=round(severity, 4),
                        description=f"Avoidance spike of {val:.4f} at generation {gen}",
                    )
                )
        return points

    def _find_fitness_collapses(self, series: TimeSeries) -> list[CriticalPoint]:
        vals = series.fitness
        points: list[CriticalPoint] = []
        for i in range(1, len(vals)):
            drop = vals[i - 1] - vals[i]
            if drop >= self.collapse_threshold:
                gen = series[i].generation
                severity = min(1.0, drop / 1.0)
                points.append(
                    CriticalPoint(
                        generation=gen,
                        point_type=CriticalPointType.FITNESS_COLLAPSE,
                        severity=round(severity, 4),
                        description=f"Fitness collapse of {drop:.4f} at generation {gen}",
                    )
                )
        return points

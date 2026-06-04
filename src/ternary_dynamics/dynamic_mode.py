"""Dynamic mode classification for ternary agent dynamics.

Classifies the overall behavior of a time series as one of:
- **converging**: metrics settle toward fixed values
- **oscillating**: metrics show periodic behavior
- **chaotic**: irregular, unpredictable fluctuations
- **stable**: low variance, settled state
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Sequence

from .time_series import TimeSeries


class DynamicMode(str, Enum):
    CONVERGING = "converging"
    OSCILLATING = "oscillating"
    CHAOTIC = "chaotic"
    STABLE = "stable"


@dataclass
class ModeResult:
    """Classification result for a time series."""

    mode: DynamicMode
    confidence: float  # 0..1
    details: str


class DynamicModeClassifier:
    """Classify the dynamic mode of a TimeSeries.

    Uses a combination of:
    - Variance of the fitness derivative (volatility)
    - Sign-change rate of the derivative (oscillation proxy)
    - Trend magnitude (convergence proxy)
    """

    def __init__(
        self,
        stable_variance_threshold: float = 0.005,
        oscillation_sign_change_rate: float = 0.4,
        chaotic_variance_threshold: float = 0.1,
    ) -> None:
        self.stable_var = stable_variance_threshold
        self.osc_rate = oscillation_sign_change_rate
        self.chaotic_var = chaotic_variance_threshold

    def classify(self, series: TimeSeries) -> ModeResult:
        if len(series) < 3:
            return ModeResult(DynamicMode.STABLE, 1.0, "Too few data points; assuming stable")

        fitness = series.fitness
        derivs = [fitness[i + 1] - fitness[i] for i in range(len(fitness) - 1)]

        # Compute variance of derivatives
        var = self._variance(derivs)

        # Sign change rate
        sign_changes = sum(
            1 for i in range(1, len(derivs)) if derivs[i] * derivs[i - 1] < 0
        )
        sign_change_rate = sign_changes / max(1, len(derivs) - 1)

        # Trend (mean of derivatives)
        trend = sum(derivs) / len(derivs)

        # Classify
        if var < self.stable_var:
            return ModeResult(
                DynamicMode.STABLE,
                confidence=min(1.0, 1.0 - var / self.stable_var),
                details=f"Low variance ({var:.6f}); population is stable",
            )

        if sign_change_rate >= self.osc_rate and var < self.chaotic_var:
            return ModeResult(
                DynamicMode.OSCILLATING,
                confidence=min(1.0, sign_change_rate),
                details=f"Sign change rate {sign_change_rate:.2f}; oscillating behavior",
            )

        if var >= self.chaotic_var:
            return ModeResult(
                DynamicMode.CHAOTIC,
                confidence=min(1.0, var),
                details=f"High variance ({var:.4f}); chaotic dynamics",
            )

        # Remaining case: converging
        abs_trend = abs(trend)
        confidence = min(1.0, abs_trend / 0.1) if abs_trend > 0 else 0.5
        direction = "upward" if trend > 0 else "downward"
        return ModeResult(
            DynamicMode.CONVERGING,
            confidence=confidence,
            details=f"Trend {direction} ({trend:.4f}), variance {var:.4f}; converging",
        )

    def classify_all(self, series_list: Sequence[TimeSeries]) -> list[ModeResult]:
        return [self.classify(s) for s in series_list]

    @staticmethod
    def _variance(values: list[float]) -> float:
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        return sum((v - mean) ** 2 for v in values) / len(values)

"""Ternary agent dynamics — time series, phase transitions, critical points."""

from .time_series import TimeSeries, TimeSeriesRecorder
from .phase_transition import PhaseTransitionDetector, PhaseTransition
from .critical_point import CriticalPointDetector, CriticalPoint, CriticalPointType
from .dynamic_mode import DynamicModeClassifier, DynamicMode

__all__ = [
    "TimeSeries",
    "TimeSeriesRecorder",
    "PhaseTransitionDetector",
    "PhaseTransition",
    "CriticalPointDetector",
    "CriticalPoint",
    "CriticalPointType",
    "DynamicModeClassifier",
    "DynamicMode",
]

__version__ = "0.1.0"

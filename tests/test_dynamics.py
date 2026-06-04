"""Tests for ternary_dynamics — ≥20 tests."""

import math
import pytest

from ternary_dynamics import (
    TimeSeries,
    TimeSeriesRecorder,
    PhaseTransitionDetector,
    PhaseTransition,
    CriticalPointDetector,
    CriticalPoint,
    CriticalPointType,
    DynamicModeClassifier,
    DynamicMode,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _linear_series(n: int, start: float = 0.0, step: float = 0.1) -> TimeSeries:
    """Build a series with linearly increasing fitness, constant diversity/avoidance."""
    r = TimeSeriesRecorder()
    for i in range(n):
        r.record(fitness=start + step * i, diversity=0.5, avoidance=0.1)
    return r.build()


def _oscillating_series(n: int = 50) -> TimeSeries:
    """Sinusoidal fitness to trigger oscillating classification."""
    r = TimeSeriesRecorder()
    for i in range(n):
        r.record(fitness=0.5 + 0.3 * math.sin(i * 0.5), diversity=0.4, avoidance=0.1)
    return r.build()


def _chaotic_series(n: int = 50) -> TimeSeries:
    """High-variance random-ish fitness for chaotic detection."""
    r = TimeSeriesRecorder()
    # Deterministic pseudo-random using trig tricks
    for i in range(n):
        r.record(fitness=0.5 + 0.45 * math.sin(i * 7.3) * math.cos(i * 3.1), diversity=0.5, avoidance=0.1)
    return r.build()


# ---------------------------------------------------------------------------
# TimeSeries & TimeSeriesRecorder
# ---------------------------------------------------------------------------

class TestTimeSeriesRecorder:
    def test_record_increments_generation(self):
        r = TimeSeriesRecorder()
        s = r.record(0.1, 0.2, 0.3)
        assert s.generation == 0
        s2 = r.record(0.4, 0.5, 0.6)
        assert s2.generation == 1

    def test_record_custom_generation(self):
        r = TimeSeriesRecorder()
        s = r.record(0.1, 0.2, 0.3, generation=42)
        assert s.generation == 42

    def test_build_returns_time_series(self):
        r = TimeSeriesRecorder()
        r.record(1.0, 0.5, 0.1)
        r.record(2.0, 0.6, 0.2)
        ts = r.build()
        assert len(ts) == 2

    def test_reset_clears_data(self):
        r = TimeSeriesRecorder()
        r.record(1.0, 0.5, 0.1)
        r.reset()
        ts = r.build()
        assert len(ts) == 0


class TestTimeSeries:
    def test_accessors(self):
        ts = _linear_series(5)
        assert all(abs(a - b) < 1e-9 for a, b in zip(ts.fitness, [0.0, 0.1, 0.2, 0.3, 0.4]))
        assert ts.diversity == [0.5] * 5
        assert ts.avoidance == [0.1] * 5

    def test_mean_metrics(self):
        ts = _linear_series(4, start=1.0, step=1.0)
        assert ts.mean_fitness() == 2.5  # (1+2+3+4)/4

    def test_slice(self):
        ts = _linear_series(10)
        sub = ts.slice(2, 5)
        assert len(sub) == 3
        assert sub[0].generation == 2

    def test_fitness_derivative(self):
        ts = _linear_series(5, start=0.0, step=0.1)
        derivs = ts.fitness_derivative()
        assert all(abs(d - 0.1) < 1e-9 for d in derivs)

    def test_empty_series(self):
        ts = TimeSeries()
        assert len(ts) == 0
        assert ts.mean_fitness() == 0.0
        assert ts.fitness_derivative() == []

    def test_iteration(self):
        ts = _linear_series(3)
        gens = [s.generation for s in ts]
        assert gens == [0, 1, 2]


# ---------------------------------------------------------------------------
# PhaseTransitionDetector
# ---------------------------------------------------------------------------

class TestPhaseTransitionDetector:
    def test_no_transitions_in_linear(self):
        det = PhaseTransitionDetector(threshold=0.3)
        ts = _linear_series(20)
        assert det.detect(ts) == []

    def test_detects_sudden_shift(self):
        r = TimeSeriesRecorder()
        # Flat then spike
        for i in range(10):
            r.record(0.5, 0.5, 0.1)
        r.record(1.0, 0.5, 0.1)  # big jump
        r.record(0.5, 0.5, 0.1)  # back down
        for i in range(10):
            r.record(0.5, 0.5, 0.1)
        ts = r.build()
        det = PhaseTransitionDetector(threshold=0.1)
        transitions = det.detect(ts)
        assert len(transitions) > 0
        assert any(t.metric == "fitness" for t in transitions)

    def test_direction(self):
        r = TimeSeriesRecorder()
        for i in range(5):
            r.record(0.5, 0.5, 0.1)
        r.record(0.9, 0.5, 0.1)  # up
        r.record(0.1, 0.5, 0.1)  # crash
        ts = r.build()
        det = PhaseTransitionDetector(threshold=0.1)
        transitions = [t for t in det.detect(ts) if t.metric == "fitness"]
        assert any(t.direction == "up" for t in transitions) or any(t.direction == "down" for t in transitions)

    def test_detect_all(self):
        det = PhaseTransitionDetector(threshold=0.1)
        ts1 = _linear_series(10)
        r = TimeSeriesRecorder()
        for i in range(5):
            r.record(0.5, 0.5, 0.1)
        r.record(1.0, 0.5, 0.1)
        r.record(0.5, 0.5, 0.1)
        ts2 = r.build()
        results = det.detect_all([ts1, ts2])
        assert len(results) == 2
        assert results[0] == []
        assert len(results[1]) > 0


# ---------------------------------------------------------------------------
# CriticalPointDetector
# ---------------------------------------------------------------------------

class TestCriticalPointDetector:
    def test_fitness_plateau(self):
        r = TimeSeriesRecorder()
        for i in range(20):
            r.record(0.5, 0.5, 0.1)  # constant fitness
        ts = r.build()
        det = CriticalPointDetector(plateau_window=5, plateau_tolerance=0.01)
        points = det.detect(ts)
        plateau_pts = [p for p in points if p.point_type == CriticalPointType.FITNESS_PLATEAU]
        assert len(plateau_pts) > 0

    def test_diversity_drop(self):
        r = TimeSeriesRecorder()
        for i in range(5):
            r.record(0.5, 0.8, 0.1)
        r.record(0.5, 0.1, 0.1)  # big drop
        ts = r.build()
        det = CriticalPointDetector(diversity_threshold=0.3)
        points = det.detect(ts)
        div_drops = [p for p in points if p.point_type == CriticalPointType.DIVERSITY_DROP]
        assert len(div_drops) == 1
        assert div_drops[0].generation == 5

    def test_avoidance_spike(self):
        r = TimeSeriesRecorder()
        for i in range(5):
            r.record(0.5, 0.5, 0.1)
        r.record(0.5, 0.5, 0.95)  # spike
        ts = r.build()
        det = CriticalPointDetector(avoidance_threshold=0.8)
        points = det.detect(ts)
        spikes = [p for p in points if p.point_type == CriticalPointType.AVOIDANCE_SPIKE]
        assert len(spikes) == 1

    def test_fitness_collapse(self):
        r = TimeSeriesRecorder()
        for i in range(5):
            r.record(0.9, 0.5, 0.1)
        r.record(0.2, 0.5, 0.1)  # collapse
        ts = r.build()
        det = CriticalPointDetector(collapse_threshold=0.4)
        points = det.detect(ts)
        collapses = [p for p in points if p.point_type == CriticalPointType.FITNESS_COLLAPSE]
        assert len(collapses) == 1

    def test_empty_series(self):
        ts = TimeSeries()
        det = CriticalPointDetector()
        assert det.detect(ts) == []


# ---------------------------------------------------------------------------
# DynamicModeClassifier
# ---------------------------------------------------------------------------

class TestDynamicModeClassifier:
    def test_stable_classification(self):
        ts = _linear_series(30, start=0.5, step=0.0)
        clf = DynamicModeClassifier()
        result = clf.classify(ts)
        assert result.mode == DynamicMode.STABLE

    def test_oscillating_classification(self):
        ts = _oscillating_series(60)
        clf = DynamicModeClassifier(oscillation_sign_change_rate=0.3)
        result = clf.classify(ts)
        assert result.mode in (DynamicMode.OSCILLATING, DynamicMode.CHAOTIC, DynamicMode.CONVERGING)

    def test_chaotic_classification(self):
        ts = _chaotic_series(60)
        clf = DynamicModeClassifier(chaotic_variance_threshold=0.05)
        result = clf.classify(ts)
        assert result.mode == DynamicMode.CHAOTIC

    def test_converging_classification(self):
        r = TimeSeriesRecorder()
        for i in range(30):
            # Slowly increasing then flattening
            fit = 1.0 - math.exp(-i / 5.0)
            r.record(fitness=fit, diversity=0.5, avoidance=0.1)
        ts = r.build()
        clf = DynamicModeClassifier()
        result = clf.classify(ts)
        # Should be stable or converging (it's settling)
        assert result.mode in (DynamicMode.CONVERGING, DynamicMode.STABLE)

    def test_short_series(self):
        r = TimeSeriesRecorder()
        r.record(0.5, 0.5, 0.1)
        ts = r.build()
        clf = DynamicModeClassifier()
        result = clf.classify(ts)
        assert result.mode == DynamicMode.STABLE  # too few points

    def test_classify_all(self):
        clf = DynamicModeClassifier()
        results = clf.classify_all([_linear_series(10), _oscillating_series(30)])
        assert len(results) == 2

    def test_confidence_range(self):
        ts = _linear_series(30, start=0.5, step=0.0)
        clf = DynamicModeClassifier()
        result = clf.classify(ts)
        assert 0.0 <= result.confidence <= 1.0

# ternary-dynamics-python

Python implementation of ternary agent dynamics — time series, phase transitions, critical points.

## Installation

```bash
pip install ternary-dynamics
```

## Quick Start

```python
from ternary_dynamics import TimeSeriesRecorder, PhaseTransitionDetector, CriticalPointDetector, DynamicModeClassifier

# Record generation data
recorder = TimeSeriesRecorder()
for gen in range(100):
    fitness = simulate_fitness(gen)
    diversity = measure_diversity(gen)
    avoidance = compute_avoidance(gen)
    recorder.record(fitness, diversity, avoidance)

series = recorder.build()

# Detect phase transitions
ptd = PhaseTransitionDetector(threshold=0.3)
transitions = ptd.detect(series)
for t in transitions:
    print(f"Transition at gen {t.generation}: {t.metric} {t.direction} (mag={t.magnitude})")

# Find critical points
cpd = CriticalPointDetector()
points = cpd.detect(series)
for p in points:
    print(f"Critical: {p.point_type.value} at gen {p.generation} — {p.description}")

# Classify dynamic mode
clf = DynamicModeClassifier()
result = clf.classify(series)
print(f"Mode: {result.mode.value} (confidence={result.confidence:.2f})")
```

## Modules

- **`time_series`** — `TimeSeries` and `TimeSeriesRecorder` for tracking fitness, diversity, avoidance over generations
- **`phase_transition`** — `PhaseTransitionDetector` for detecting sudden behavioral shifts
- **`critical_point`** — `CriticalPointDetector` for fitness plateaus, diversity drops, avoidance spikes, fitness collapses
- **`dynamic_mode`** — `DynamicModeClassifier` for classifying behavior as converging, oscillating, chaotic, or stable

## Development

```bash
pip install -e ".[dev]"
PYTHONPATH=src pytest tests/ -v
```

## License

MIT

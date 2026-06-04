"""Time series tracking for ternary agent dynamics.

Tracks three core metrics over generations:
- **fitness**: population fitness level
- **diversity**: behavioral diversity within the population
- **avoidance**: avoidance behavior intensity
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence


@dataclass
class Snapshot:
    """A single generation's metrics."""

    generation: int
    fitness: float
    diversity: float
    avoidance: float


class TimeSeries:
    """Immutable view of recorded generation snapshots."""

    def __init__(self, snapshots: Sequence[Snapshot] | None = None) -> None:
        self._snapshots: list[Snapshot] = list(snapshots) if snapshots else []

    # --- accessors -----------------------------------------------------------

    @property
    def snapshots(self) -> list[Snapshot]:
        return list(self._snapshots)

    @property
    def generations(self) -> list[int]:
        return [s.generation for s in self._snapshots]

    @property
    def fitness(self) -> list[float]:
        return [s.fitness for s in self._snapshots]

    @property
    def diversity(self) -> list[float]:
        return [s.diversity for s in self._snapshots]

    @property
    def avoidance(self) -> list[float]:
        return [s.avoidance for s in self._snapshots]

    def __len__(self) -> int:
        return len(self._snapshots)

    def __getitem__(self, idx: int) -> Snapshot:
        return self._snapshots[idx]

    def __iter__(self):
        return iter(self._snapshots)

    def slice(self, start: int = 0, end: int | None = None) -> "TimeSeries":
        """Return a new TimeSeries for generations [start, end)."""
        return TimeSeries(self._snapshots[start:end])

    # --- statistics ----------------------------------------------------------

    def mean_fitness(self) -> float:
        if not self._snapshots:
            return 0.0
        return sum(s.fitness for s in self._snapshots) / len(self._snapshots)

    def mean_diversity(self) -> float:
        if not self._snapshots:
            return 0.0
        return sum(s.diversity for s in self._snapshots) / len(self._snapshots)

    def mean_avoidance(self) -> float:
        if not self._snapshots:
            return 0.0
        return sum(s.avoidance for s in self._snapshots) / len(self._snapshots)

    def fitness_derivative(self) -> list[float]:
        """First-order differences of fitness (rate of change per generation)."""
        vals = self.fitness
        return [vals[i + 1] - vals[i] for i in range(len(vals) - 1)]


class TimeSeriesRecorder:
    """Accumulates snapshots generation-by-generation."""

    def __init__(self) -> None:
        self._snapshots: list[Snapshot] = []
        self._generation: int = 0

    def record(self, fitness: float, diversity: float, avoidance: float, generation: int | None = None) -> Snapshot:
        gen = generation if generation is not None else self._generation
        snap = Snapshot(generation=gen, fitness=fitness, diversity=diversity, avoidance=avoidance)
        self._snapshots.append(snap)
        self._generation = gen + 1
        return snap

    def build(self) -> TimeSeries:
        """Freeze recorded data into an immutable TimeSeries."""
        return TimeSeries(self._snapshots)

    def reset(self) -> None:
        self._snapshots.clear()
        self._generation = 0

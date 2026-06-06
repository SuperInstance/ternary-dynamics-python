# ternary-dynamics-python

**Python port of ternary agent dynamics — time series tracking, phase transition detection, critical point identification, and dynamic mode classification for {-1, 0, +1} population systems.**

## Background

Complex systems — ecosystems, economies, neural networks, agent populations — exhibit rich temporal dynamics. Populations can be stable for long periods, then suddenly transition (a market crash, an ecosystem collapse, a phase transition in a spin glass). Understanding *when* and *why* these transitions occur is the central challenge of dynamical systems theory.

This crate provides tools for analyzing the temporal dynamics of ternary agent populations, where each agent holds a value in {-1, 0, +1}. The three metrics tracked over time are:

- **Fitness**: How well the population is performing (0.0 to 1.0). High fitness means the population has converged on a good solution.
- **Diversity**: How varied the population's behaviors are (0.0 to 1.0). High diversity means many different strategies are being explored.
- **Avoidance**: How strongly the population is avoiding negative outcomes (0.0 to 1.0). High avoidance suggests the population has learned to fear certain regions of the search space.

The crate implements four analytical capabilities:

1. **Time series recording**: A `TimeSeriesRecorder` that accumulates generation-by-generation snapshots, producing an immutable `TimeSeries` for analysis.

2. **Phase transition detection**: Identifies sudden shifts in population behavior by detecting large changes in the second derivative (acceleration) of key metrics. A phase transition is when the rate of change itself changes abruptly — the system shifts from "gradually improving" to "rapidly collapsing."

3. **Critical point identification**: Detects specific inflection points: fitness plateaus (stagnation), diversity drops (convergence), avoidance spikes (fear events), and fitness collapses (catastrophic failure).

4. **Dynamic mode classification**: Classifies the overall behavior of a population into four modes: **STABLE** (low variance, settled), **CONVERGING** (trending toward a fixed point), **OSCILLATING** (periodic behavior), or **CHAOTIC** (irregular, unpredictable).

## How It Works

**`TimeSeries`** and **`TimeSeriesRecorder`**:
- `TimeSeriesRecorder.record(fitness, diversity, avoidance)` appends a snapshot with an auto-incrementing generation counter.
- `TimeSeries` provides accessors: `.fitness`, `.diversity`, `.avoidance` (lists of floats), `.fitness_derivative()` (first differences), `.mean_fitness()`, `.slice(start, end)`.

**`PhaseTransitionDetector`**:
- For each metric (fitness, diversity, avoidance), computes the second derivative: `d²m/dt² = |d₁[i+1] - d₁[i]|` where `d₁` is the first derivative.
- A transition is flagged when `d²m/dt² > threshold` (default 0.3).
- Each transition has a direction ("up" if the derivative increases, "down" if it decreases) and a magnitude.

**`CriticalPointDetector`**:
- **Fitness plateau**: Scans for windows of `plateau_window` (default 5) consecutive generations where `|d(fitness)/dt| ≤ plateau_tolerance` (default 0.01). Severity = 1 - mean(|d|)/tolerance.
- **Diversity drop**: Single-generation diversity decrease ≥ `diversity_threshold` (default 0.3).
- **Avoidance spike**: Avoidance ≥ `avoidance_threshold` (default 0.8).
- **Fitness collapse**: Single-generation fitness decrease ≥ `collapse_threshold` (default 0.4).

**`DynamicModeClassifier`**:
- Computes the variance of fitness derivatives and the sign-change rate (how often the derivative switches between positive and negative).
- **STABLE**: variance < `stable_variance_threshold` (0.005).
- **OSCILLATING**: sign change rate ≥ `oscillation_sign_change_rate` (0.4) AND variance < `chaotic_variance_threshold`.
- **CHAOTIC**: variance ≥ `chaotic_variance_threshold` (0.1).
- **CONVERGING**: everything else (has a trend, moderate variance).

### Design Decisions

- **Immutable TimeSeries**: Once built, a `TimeSeries` cannot be modified. This prevents accidental data corruption during analysis and enables safe sharing across analysis routines.
- **Threshold-based detection**: All detectors use explicit thresholds with sensible defaults. This makes the analysis interpretable — you can explain *why* a critical point was detected by citing the specific threshold.
- **Python type hints throughout**: Full type annotations enable IDE autocompletion and static type checking with mypy.

## Experimental Results

All 26 tests pass. Key findings:

- **Time series recording**: Recording two snapshots produces a TimeSeries of length 2. Custom generation numbers override auto-increment. Reset clears all data.
- **Linear series**: A linearly increasing fitness series (step 0.1) has derivatives all equal to 0.1, confirmed to 6 decimal places.
- **Phase transitions in linear series**: No transitions detected (the second derivative is zero everywhere — constant rate of change).
- **Phase transition detection**: A flat series with a sudden spike (0.5 → 1.0 → 0.5) detects at least one transition with metric = "fitness". Both "up" and "down" directions are captured.
- **Fitness plateau**: 20 generations of constant fitness (all 0.5) produces at least one FITNESS_PLATEAU critical point.
- **Diversity drop**: A drop from 0.8 to 0.1 (Δ = 0.7) in one generation is detected as a DIVERSITY_DROP with severity 0.7.
- **Avoidance spike**: A spike to 0.95 is detected as an AVOIDANCE_SPIKE.
- **Fitness collapse**: A drop from 0.9 to 0.2 (Δ = 0.7) is detected as a FITNESS_COLLAPSE.
- **Dynamic mode classification**:
  - A constant series (step 0.0) → STABLE.
  - A sinusoidal series → OSCILLATING or CHAOTIC (depends on parameters).
  - A chaotic series (high-variance trigonometric mixing) → CHAOTIC.
  - An exponentially converging series → CONVERGING or STABLE.
  - A single-point series → STABLE (too few data points).
- **Confidence range**: All classification confidences are in [0.0, 1.0].
- **Batch processing**: `detect_all` and `classify_all` process multiple time series in one call.

## Impact

This library provides the analytical infrastructure for understanding ternary agent dynamics:

1. **Early warning for phase transitions**: The second-derivative detector can flag an impending transition *before* the population crashes, giving time for intervention.
2. **Objective stagnation detection**: Fitness plateaus are the bane of evolutionary algorithms. This detector tells you exactly when the population stops making progress.
3. **Regime identification**: Knowing whether the population is stable, oscillating, or chaotic determines the appropriate intervention strategy (exploit, explore, or randomize).

## Use Cases

1. **Training monitoring for ternary neural networks**: Track fitness (accuracy), diversity (weight distribution spread), and avoidance (loss spike avoidance) during training. Phase transition detection alerts when the network enters a qualitatively different learning regime.

2. **Evolutionary algorithm analysis**: Monitor population dynamics in a genetic algorithm with ternary genomes. Critical point detection identifies when diversity drops dangerously low (risk of premature convergence) or when fitness collapses (need to restart).

3. **Multi-agent system monitoring**: In a ternary agent simulation (agents hold opinion -1/0/+1), track how the population's sentiment evolves. Phase transitions detect sudden opinion shifts (viral spread, cascade failures).

4. **Reinforcement learning exploration tracking**: Track the exploration-exploitation balance via diversity. A diversity drop combined with a fitness plateau suggests the agent has stopped exploring — trigger an exploration bonus.

5. **Financial market regime detection**: Map market indicators to ternary (bullish/neutral/bearish). Track the population of signals over time. Dynamic mode classification tells you whether the market is trending (CONVERGING), ranging (OSCILLATING), or volatile (CHAOTIC).

## Open Questions

1. **Multi-scale analysis**: The current detectors operate at a single time scale. A wavelet-based approach could detect phase transitions that only appear at certain time scales (e.g., weekly oscillations within a monthly trend).

2. **Predictive detection**: The detectors identify transitions *after* they happen. Could a leading indicator (e.g., increasing variance before a collapse, "critical slowing down") provide advance warning?

3. **Correlation between metrics**: The current analysis treats fitness, diversity, and avoidance independently. In reality, they're coupled — diversity drops often precede fitness plateaus. A multivariate detector could capture these relationships.

## Connection to Oxide Stack

`ternary-dynamics-python` operates at the **cudaclaw** level as the monitoring and analysis layer. It consumes time series data produced by the **flux-core** computation engine, detects phase transitions and critical points, and reports them to the user. The Python implementation enables rapid prototyping and interactive analysis (Jupyter notebooks, matplotlib visualization). At the **pincher** level, the phase transition detector informs the scheduler about when to change computation strategy (e.g., switch from exploration to exploitation after detecting a fitness plateau).

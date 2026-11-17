"""
Clock / rhythm layer.

Responsibility: decide WHEN notes start. A clock returns the onset times
(in beats) at which the path is read to produce a note. It is separate
from the path simulator (the dynamics) and from the pitch mapper (what a
sampled value becomes), so rhythm and melody can be chosen independently.

Four clocks:

- UniformClock   one note every fixed step (the original quarter-note grid)
- PoissonClock   notes at random, with no memory
- HawkesClock    notes that trigger more notes: bursts and silences
- ProcessClock   a second stochastic process sets the note lengths
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np

from .paths import PathSimulator, time_grid


class Clock(ABC):
    """Base interface for all musical clocks."""

    @abstractmethod
    def get_sample_times(self, T: float, rng: np.random.Generator | None = None) -> np.ndarray:
        """
        Return the note onset times, sorted, without duplicates, within
        [0, T). `rng` is the source of randomness for stochastic clocks.
        """
        raise NotImplementedError


def _snap_to_grid(times: np.ndarray, grid: float | None, T: float) -> np.ndarray:
    """Snap times down onto a grid (dropping duplicates); grid=None leaves them free."""
    times = np.asarray(times, dtype=float)
    times = times[(times >= 0) & (times < T)]
    if grid is None:
        return np.unique(times)
    if grid <= 0:
        raise ValueError(f"grid must be > 0 (or None), got {grid}")
    snapped = np.unique(np.floor(times / grid + 1e-9)) * grid
    return snapped[snapped < T - 1e-9]


class UniformClock(Clock):
    """
    Fixed-subdivision clock: one note every `beat_duration` beats.

    Parameters
    ----------
    beat_duration : float
        Beats between successive notes (1.0 = quarter notes, 0.5 = eighths).
    """

    def __init__(self, beat_duration: float = 1.0):
        if beat_duration <= 0:
            raise ValueError(f"beat_duration must be > 0, got {beat_duration}")
        self.beat_duration = beat_duration

    def get_sample_times(self, T: float, rng: np.random.Generator | None = None) -> np.ndarray:
        num_notes = int(np.ceil(T / self.beat_duration - 1e-9))
        return np.arange(num_notes) * self.beat_duration


class PoissonClock(Clock):
    """
    Notes arriving as a Poisson process: each instant is as likely as any
    other to carry a note, independently of what came before. Irregular
    but evenly spread.

    Parameters
    ----------
    rate : float
        Average number of notes per beat (before snapping to the grid).
    grid : float or None
        Onsets are snapped onto multiples of `grid` beats (0.25 = sixteenth
        notes), at most one note per slot. None keeps the raw times.
    """

    def __init__(self, rate: float = 2.0, grid: float | None = 0.25):
        if rate <= 0:
            raise ValueError(f"rate must be > 0, got {rate}")
        self.rate = rate
        self.grid = grid

    def get_sample_times(self, T: float, rng: np.random.Generator | None = None) -> np.ndarray:
        rng = np.random.default_rng() if rng is None else rng
        count = rng.poisson(self.rate * T)
        return _snap_to_grid(rng.uniform(0.0, T, size=count), self.grid, T)


class HawkesClock(Clock):
    """
    Notes arriving as a self-exciting Hawkes process: every note briefly
    raises the chance of another one, so notes come in bursts separated
    by silences.

    The intensity (notes per beat) at time t is

        lambda(t) = base_rate + sum_i excitation * decay * exp(-decay * (t - t_i))

    summed over the past notes t_i. Each note triggers `excitation` more
    notes on average, so the long-run rate is base_rate / (1 - excitation).
    Simulated exactly with Ogata's thinning algorithm.

    Parameters
    ----------
    base_rate : float
        Notes per beat arriving on their own. Must be > 0.
    excitation : float
        Average number of notes triggered by each note, in [0, 1).
        0 is a Poisson clock; close to 1 gives long bursts.
    decay : float
        How fast the excitement fades, per beat. Higher = tighter bursts.
    grid : float or None
        Same as PoissonClock.
    """

    def __init__(
        self,
        base_rate: float = 0.5,
        excitation: float = 0.7,
        decay: float = 4.0,
        grid: float | None = 0.25,
    ):
        if base_rate <= 0:
            raise ValueError(f"base_rate must be > 0, got {base_rate}")
        if not 0.0 <= excitation < 1.0:
            raise ValueError(f"excitation must be in [0, 1), got {excitation}")
        if decay <= 0:
            raise ValueError(f"decay must be > 0, got {decay}")
        self.base_rate = base_rate
        self.excitation = excitation
        self.decay = decay
        self.grid = grid

    def get_sample_times(self, T: float, rng: np.random.Generator | None = None) -> np.ndarray:
        rng = np.random.default_rng() if rng is None else rng
        events = []
        t = 0.0
        excitement = 0.0  # the self-exciting part of the intensity, at time t
        while True:
            # The intensity only decays until the next note, so its current
            # value is an upper bound: propose from it, then accept or reject.
            bound = self.base_rate + excitement
            wait = rng.exponential(1.0 / bound)
            t += wait
            if t >= T:
                break
            excitement *= np.exp(-self.decay * wait)
            if rng.uniform() * bound <= self.base_rate + excitement:
                events.append(t)
                excitement += self.excitation * self.decay
        return _snap_to_grid(np.array(events), self.grid, T)


class ProcessClock(Clock):
    """
    Rhythm driven by a second stochastic process: the level of the
    process at a note's onset decides how long that note lasts.

    The process is simulated over the piece and rescaled to its own
    realized range. That range is cut into as many equal bands as there
    are durations: the lowest band gives the longest notes, the highest
    the shortest. A mean-reverting process hovers around a medium pace;
    a jump process switches pace abruptly.

    Parameters
    ----------
    process : PathSimulator
        The process driving the rhythm (independent of the melody's).
    durations : sequence of float
        Allowed note lengths in beats.
    dt : float
        Simulation step for the rhythm process.
    """

    def __init__(
        self,
        process: PathSimulator,
        durations: Sequence[float] = (0.25, 0.5, 1.0, 2.0),
        dt: float = 0.01,
    ):
        durations = sorted((float(d) for d in durations), reverse=True)
        if not durations or durations[-1] <= 0:
            raise ValueError("durations must be a non-empty sequence of positive numbers.")
        self.process = process
        self.durations = durations
        self.dt = dt

    def get_sample_times(self, T: float, rng: np.random.Generator | None = None) -> np.ndarray:
        rng = np.random.default_rng() if rng is None else rng
        times = time_grid(T, self.dt)
        values, _ = self.process.simulate_driven(times, rng.standard_normal(len(times) - 1), rng)

        low, high = values.min(), values.max()
        level = np.full_like(values, 0.5) if high == low else (values - low) / (high - low)

        onsets = []
        t = 0.0
        while t < T - 1e-9:
            onsets.append(t)
            index = min(int(round(t / T * (len(times) - 1))), len(times) - 1)
            band = min(int(level[index] * len(self.durations)), len(self.durations) - 1)
            t += self.durations[band]
        return np.array(onsets)


def note_durations(
    onsets: np.ndarray, T: float, gate: float = 1.0, max_duration: float | None = None
) -> np.ndarray:
    """
    Length of each note, in beats: the time until the next onset (or the
    end of the piece), optionally capped at `max_duration`, then
    multiplied by `gate` (1.0 = legato, 0.5 = staccato).
    """
    if not 0.0 < gate <= 1.0:
        raise ValueError(f"gate must be in (0, 1], got {gate}")
    gaps = np.diff(np.append(onsets, T))
    if max_duration is not None:
        gaps = np.minimum(gaps, max_duration)
    return gaps * gate


def sample_path(times: np.ndarray, values: np.ndarray, sample_times: np.ndarray) -> np.ndarray:
    """
    Read the path (times, values) at the given sample_times, taking the
    nearest point of the fine-grained path. Works with any clock.
    """
    sample_times = np.asarray(sample_times, dtype=float)
    indices = np.clip(np.searchsorted(times, sample_times, side="left"), 0, len(times) - 1)

    # searchsorted returns the point at or after each time; the one before may be closer
    left = np.clip(indices - 1, 0, len(times) - 1)
    choose_left = np.abs(times[left] - sample_times) < np.abs(times[indices] - sample_times)
    return values[np.where(choose_left, left, indices)]

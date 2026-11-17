import numpy as np
import pytest

import stochastic_music as sm


def rng(seed=0):
    return np.random.default_rng(seed)


def test_uniform_clock():
    np.testing.assert_allclose(sm.UniformClock(1.0).get_sample_times(4), [0, 1, 2, 3])
    assert len(sm.UniformClock(0.25).get_sample_times(4)) == 16


@pytest.mark.parametrize(
    "clock",
    [
        sm.UniformClock(0.5),
        sm.PoissonClock(2.0),
        sm.PoissonClock(2.0, grid=None),
        sm.HawkesClock(),
        sm.ProcessClock(sm.OrnsteinUhlenbeckProcess()),
    ],
    ids=lambda c: type(c).__name__,
)
def test_onsets_are_sorted_unique_and_in_range(clock):
    onsets = clock.get_sample_times(32, rng())
    assert len(onsets) > 0
    assert (np.diff(onsets) > 0).all()
    assert onsets[0] >= 0 and onsets[-1] < 32


def test_grid_snapping():
    onsets = sm.PoissonClock(3.0, grid=0.25).get_sample_times(64, rng())
    np.testing.assert_allclose(onsets / 0.25, np.round(onsets / 0.25))


def test_poisson_rate():
    onsets = sm.PoissonClock(1.0, grid=None).get_sample_times(5000, rng())
    assert len(onsets) / 5000 == pytest.approx(1.0, rel=0.05)


def test_hawkes_long_run_rate():
    clock = sm.HawkesClock(base_rate=0.5, excitation=0.5, decay=2.0, grid=None)
    onsets = clock.get_sample_times(5000, rng())
    assert len(onsets) / 5000 == pytest.approx(0.5 / (1 - 0.5), rel=0.1)


def test_hawkes_is_burstier_than_poisson():
    # Same average rate; self-excitation should make gaps more irregular.
    hawkes = sm.HawkesClock(base_rate=0.2, excitation=0.8, decay=4.0, grid=None)
    poisson = sm.PoissonClock(1.0, grid=None)
    cv = [
        np.std(g) / np.mean(g)
        for g in (np.diff(c.get_sample_times(3000, rng())) for c in (hawkes, poisson))
    ]
    assert cv[0] > 1.3 * cv[1]


def test_process_clock_uses_only_allowed_lengths():
    clock = sm.ProcessClock(sm.WienerProcess(), durations=[0.5, 1.0])
    gaps = np.diff(clock.get_sample_times(64, rng()))
    assert set(np.round(gaps, 9)) <= {0.5, 1.0}


def test_note_durations():
    onsets = np.array([0.0, 1.0, 3.0])
    np.testing.assert_allclose(sm.note_durations(onsets, 4.0), [1, 2, 1])
    np.testing.assert_allclose(sm.note_durations(onsets, 4.0, gate=0.5), [0.5, 1, 0.5])
    np.testing.assert_allclose(sm.note_durations(onsets, 4.0, max_duration=1.5), [1, 1.5, 1])


def test_sample_path_picks_nearest_point():
    times = np.linspace(0, 1, 11)
    values = times * 10
    np.testing.assert_allclose(sm.sample_path(times, values, [0.0, 0.31, 0.36, 1.0]), [0, 3, 4, 10])

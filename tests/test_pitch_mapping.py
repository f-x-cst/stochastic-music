import numpy as np
import pytest

import stochastic_music as sm
from stochastic_music.pitch_mapping import parse_root


def test_full_range_is_used():
    pitches = sm.RealizedRangeBinner(48, 72).map(np.linspace(-3, 5, 1000))
    assert pitches.min() == 48 and pitches.max() == 72


def test_shift_and_scale_are_inaudible():
    values = np.random.default_rng(0).standard_normal(200).cumsum()
    mapper = sm.RealizedRangeBinner(48, 72)
    np.testing.assert_array_equal(mapper.map(values), mapper.map(3.7 * values - 12))


def test_constant_path_plays_middle_note():
    pitches = sm.RealizedRangeBinner(60, 72).map(np.full(5, 2.0))
    assert set(pitches) == {66}


@pytest.mark.parametrize("scale", list(sm.SCALES))
def test_scale_constraint(scale):
    mapper = sm.RealizedRangeBinner(48, 72, scale=scale, root="D")
    pitches = mapper.map(np.random.default_rng(1).standard_normal(500).cumsum())
    intervals = set(sm.SCALES[scale])
    assert all((p - 2) % 12 in intervals for p in pitches)


def test_scale_degrees_are_equally_likely():
    mapper = sm.RealizedRangeBinner(60, 71, scale="Ionian (major)", root="C")
    pitches = mapper.map(np.linspace(0, 1, 70_000))
    _, counts = np.unique(pitches, return_counts=True)
    assert len(counts) == 7
    assert counts.max() / counts.min() < 1.01


def test_narrow_range_without_scale_tone_falls_back_to_nearest():
    mapper = sm.RealizedRangeBinner(61, 61, scale="Ionian (major)", root="C")
    assert set(mapper.map(np.arange(5.0))) <= {60, 62}


def test_root_parsing():
    assert parse_root("A") == 9 and parse_root("bb") == 10 and parse_root(62) == 2
    with pytest.raises(ValueError):
        parse_root("H")


def test_invalid_range_or_scale():
    with pytest.raises(ValueError):
        sm.RealizedRangeBinner(72, 48)
    with pytest.raises(ValueError):
        sm.RealizedRangeBinner(scale="Klingon")


def test_note_name():
    assert sm.note_name(60) == "C4" and sm.note_name(69) == "A4"

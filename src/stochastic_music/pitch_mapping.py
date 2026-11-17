"""
Pitch mapping (binning) layer.

Responsibility: turn sampled path values (arbitrary floats) into MIDI
pitch numbers (0-127). This layer knows nothing about how those pitches
become sound - it just emits integers.

Binning strategy: realized range. The bounds are the actual min/max of
the values passed in, stretched over [low_pitch, high_pitch]. The output
can be restricted to a root note + scale/mode (see SCALES).

What this means for process parameters
--------------------------------------
Because the path is rescaled to its own range, anything that only
SHIFTS it (a starting value) or SCALES it (the sigma of a pure Wiener
process, the x0 of a GBM) is inaudible. Only what changes the SHAPE of
the path survives: drift against noise, mean reversion, jumps,
volatility clustering.
"""

from abc import ABC, abstractmethod

import numpy as np

CHROMATIC = "Chromatic (12-tone)"

# Semitone offsets from the root, within one octave.
SCALES = {
    # --- 7-note diatonic modes ---
    "Ionian (major)": [0, 2, 4, 5, 7, 9, 11],
    "Dorian": [0, 2, 3, 5, 7, 9, 10],
    "Phrygian": [0, 1, 3, 5, 7, 8, 10],
    "Lydian": [0, 2, 4, 6, 7, 9, 11],
    "Mixolydian": [0, 2, 4, 5, 7, 9, 10],
    "Aeolian (minor)": [0, 2, 3, 5, 7, 8, 10],
    "Locrian": [0, 1, 3, 5, 6, 8, 10],
    # --- other scales ---
    "Harmonic minor": [0, 2, 3, 5, 7, 8, 11],
    "Melodic minor": [0, 2, 3, 5, 7, 9, 11],
    "Major pentatonic": [0, 2, 4, 7, 9],
    "Minor pentatonic": [0, 3, 5, 7, 10],
    "Blues": [0, 3, 5, 6, 7, 10],
    "Whole tone": [0, 2, 4, 6, 8, 10],
    CHROMATIC: list(range(12)),  # every semitone, i.e. unconstrained
}

ROOT_NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
_FLAT_NAMES = {"DB": 1, "EB": 3, "GB": 6, "AB": 8, "BB": 10}


def parse_root(root: int | str) -> int:
    """Pitch class (0-11) from a note name ("A", "F#", "Bb") or any MIDI number."""
    if isinstance(root, str):
        name = root.strip().upper()
        if name in ROOT_NOTE_NAMES:
            return ROOT_NOTE_NAMES.index(name)
        if name in _FLAT_NAMES:
            return _FLAT_NAMES[name]
        raise ValueError(f"Unknown root note {root!r}. Use one of {ROOT_NOTE_NAMES}.")
    return int(root) % 12


def note_name(pitch: int) -> str:
    """Name of a MIDI pitch, e.g. 60 -> "C4"."""
    return f"{ROOT_NOTE_NAMES[int(pitch) % 12]}{int(pitch) // 12 - 1}"


def _build_scale_pitches(low_pitch: int, high_pitch: int, root: int, scale_name: str) -> np.ndarray:
    """
    Sorted MIDI pitches within [low_pitch, high_pitch] that belong to the
    given root + scale. low_pitch/high_pitch stay in semitones (a
    difference of 12 is one octave) whatever the scale.
    """
    intervals = SCALES[scale_name]
    root_class = root % 12

    pitches = [p for p in range(low_pitch, high_pitch + 1) if (p - root_class) % 12 in intervals]

    if not pitches:
        # Range too narrow to hold a single scale tone: fall back to the
        # nearest one, even if it lies just outside the range. (Clipping it
        # back in would bring a non-scale pitch back.)
        candidates = [
            p
            for p in range(max(low_pitch - 12, 0), min(high_pitch + 12, 127) + 1)
            if (p - root_class) % 12 in intervals
        ]
        pitches = [min(candidates, key=lambda p: abs(p - (low_pitch + high_pitch) / 2))]

    return np.array(pitches)


class PitchMapper(ABC):
    """Base interface for all pitch mapping strategies."""

    @abstractmethod
    def map(self, values: np.ndarray) -> np.ndarray:
        """
        Map an array of sampled path values to an array of MIDI pitch
        integers (0-127), same length as the input.
        """
        raise NotImplementedError


class RealizedRangeBinner(PitchMapper):
    """
    Bins path values into a pitch range whose bounds are the realized
    min/max of the values passed in. The full pitch range is always
    used, whatever the path's volatility.

    Scale constraint
    ----------------
    By default every semitone in [low_pitch, high_pitch] is a bin. With a
    `scale` and `root`, only the notes of that scale are bins, and each
    one gets an EQUAL slice of the path's range. A whole-tone gap between
    two scale degrees is therefore exactly as likely to be landed on as a
    half-tone gap (binning chromatically and snapping afterwards would
    favour the notes next to wide gaps).

    Parameters
    ----------
    low_pitch, high_pitch : int
        MIDI pitch bounds of the output range, inclusive. Always in
        semitones: a difference of 12 is one octave whatever the scale.
    scale : str, optional
        A key of SCALES, e.g. "Dorian". None means chromatic.
    root : int or str
        Root note: a name ("A", "F#", "Bb") or a pitch class / MIDI
        number (only its value mod 12 matters). Ignored when chromatic.
    """

    def __init__(
        self,
        low_pitch: int = 48,
        high_pitch: int = 72,
        scale: str | None = None,
        root: int | str = 0,
    ):
        if not 0 <= low_pitch <= high_pitch <= 127:
            raise ValueError(
                f"Need 0 <= low_pitch <= high_pitch <= 127, got {low_pitch} and {high_pitch}."
            )
        if scale is not None and scale not in SCALES:
            raise ValueError(f"Unknown scale {scale!r}. Available: {list(SCALES)}.")
        self.low_pitch = int(low_pitch)
        self.high_pitch = int(high_pitch)
        self.scale = scale
        self.root = parse_root(root)

    def allowed_pitches(self) -> np.ndarray:
        """The MIDI pitches this mapper can output, in ascending order."""
        if self.scale is None or self.scale == CHROMATIC:
            return np.arange(self.low_pitch, self.high_pitch + 1)
        return _build_scale_pitches(self.low_pitch, self.high_pitch, self.root, self.scale)

    def map(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        if values.size == 0:
            return np.array([], dtype=int)

        allowed = self.allowed_pitches()
        num_bins = len(allowed)
        value_low, value_high = values.min(), values.max()

        # a constant path has no range to stretch: play the middle note
        if value_high == value_low:
            return np.full(values.shape, allowed[num_bins // 2], dtype=int)

        normalized = (values - value_low) / (value_high - value_low)
        bin_indices = np.clip(np.floor(normalized * num_bins).astype(int), 0, num_bins - 1)
        return allowed[bin_indices]

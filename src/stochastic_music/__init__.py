"""
Stochastic Music: simulate stochastic processes and hear them as music.

    import stochastic_music as sm

    piece = sm.compose(sm.OrnsteinUhlenbeckProcess(kappa=2.0), length=32, bpm=110, seed=7)
    piece.write_midi("melody.mid")
    piece.write_wav("melody.wav")

Four independent layers, each replaceable without touching the others:

    paths          simulate the raw stochastic process
    clock          decide when notes start (the rhythm)
    pitch_mapping  turn path values into MIDI pitches
    midi_writer    write the notes as a MIDI file (synth renders audio)

`pipeline.compose` wires them together.
"""

from .catalog import PROCESSES
from .clock import (
    Clock,
    HawkesClock,
    PoissonClock,
    ProcessClock,
    UniformClock,
    note_durations,
    sample_path,
)
from .paths import (
    BrownianMotion,
    CIRProcess,
    GBMProcess,
    HestonProcess,
    MertonJumpDiffusion,
    MultiVoiceSimulator,
    OrnsteinUhlenbeckProcess,
    PathSimulator,
    SABRProcess,
    VarianceGammaProcess,
    WienerProcess,
    simulate_correlated,
)
from .pipeline import Piece, Voice, VoiceSpec, compose
from .pitch_mapping import (
    ROOT_NOTE_NAMES,
    SCALES,
    PitchMapper,
    RealizedRangeBinner,
    note_name,
)
from .synth import WAVEFORMS

__version__ = "0.1.0"

__all__ = [
    "PROCESSES",
    "ROOT_NOTE_NAMES",
    "SCALES",
    "WAVEFORMS",
    "BrownianMotion",
    "CIRProcess",
    "Clock",
    "GBMProcess",
    "HawkesClock",
    "HestonProcess",
    "MertonJumpDiffusion",
    "MultiVoiceSimulator",
    "OrnsteinUhlenbeckProcess",
    "PathSimulator",
    "Piece",
    "PitchMapper",
    "PoissonClock",
    "ProcessClock",
    "RealizedRangeBinner",
    "SABRProcess",
    "UniformClock",
    "VarianceGammaProcess",
    "Voice",
    "VoiceSpec",
    "WienerProcess",
    "compose",
    "note_durations",
    "note_name",
    "sample_path",
    "simulate_correlated",
]

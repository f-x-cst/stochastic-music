"""
Pipeline / orchestration layer.

Responsibility: wire the layers together.

    simulate paths -> pick note onsets -> read the paths -> map to pitch

`compose` runs that chain for one or several voices and returns a
`Piece`, which knows how to write itself as MIDI or audio. There is no
logic here beyond calling each layer in order.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import midi_writer, synth
from .clock import Clock, UniformClock, note_durations, sample_path
from .paths import Latent, PathSimulator, simulate_correlated
from .pitch_mapping import PitchMapper, RealizedRangeBinner


@dataclass
class VoiceSpec:
    """
    Everything that defines one voice before it is generated.

    Parameters
    ----------
    process : PathSimulator
        The stochastic process whose path becomes the melody.
    clock : Clock
        When notes start. Default: one note per beat.
    mapper : PitchMapper
        How path values become pitches. Default: chromatic, C3 to C5.
    name : str, optional
        Track name in the MIDI file.
    program : int
        General MIDI instrument number (0 = acoustic grand piano), stored
        in the MIDI file.
    velocity : int
        MIDI velocity (loudness) of every note, 1-127.
    gate : float
        Fraction of the time to the next note during which a note sounds
        (1.0 = legato, 0.5 = staccato).
    max_note_beats : float, optional
        Cap on note length, useful with irregular clocks so a note before
        a long silence does not drone.
    waveform : str
        Timbre of the built-in audio preview (see synth.WAVEFORMS).
    """

    process: PathSimulator
    clock: Clock = field(default_factory=UniformClock)
    mapper: PitchMapper = field(default_factory=RealizedRangeBinner)
    name: str | None = None
    program: int = 0
    velocity: int = 90
    gate: float = 1.0
    max_note_beats: float | None = None
    waveform: str = "triangle"

    def __post_init__(self):
        if not 0 <= self.program <= 127:
            raise ValueError(f"program must be in 0..127, got {self.program}")
        if not 1 <= self.velocity <= 127:
            raise ValueError(f"velocity must be in 1..127, got {self.velocity}")
        if self.waveform not in synth.WAVEFORMS:
            raise ValueError(f"Unknown waveform {self.waveform!r}. Available: {synth.WAVEFORMS}.")


@dataclass
class Voice:
    """One generated voice: the raw path and the notes read from it."""

    name: str
    times: np.ndarray  # fine time grid of the path, in beats
    values: np.ndarray  # the raw path
    latent: Latent  # hidden series behind the path, e.g. {"variance": ...}
    onsets: np.ndarray  # note start times, in beats
    durations: np.ndarray  # note lengths, in beats
    sampled_values: np.ndarray  # path values at the onsets
    pitches: np.ndarray  # MIDI pitches
    program: int = 0
    velocity: int = 90
    waveform: str = "triangle"


@dataclass
class Piece:
    """A generated piece of music: one or several voices sharing a tempo."""

    voices: list[Voice]
    bpm: float
    length: float  # in beats
    seed: int | None = None

    @property
    def seconds_per_beat(self) -> float:
        return 60.0 / self.bpm

    @property
    def duration_seconds(self) -> float:
        return self.length * self.seconds_per_beat

    # --- MIDI ---

    def to_midi(self):
        """The piece as a mido.MidiFile, one track per voice."""
        return midi_writer.build_midi(self.voices, self.bpm)

    def midi_bytes(self) -> bytes:
        return midi_writer.midi_bytes(self.voices, self.bpm)

    def write_midi(self, path: str | Path) -> Path:
        """Write a .mid file and return its path."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.midi_bytes())
        return path

    # --- audio ---

    def render(self, sample_rate: int = 44100) -> np.ndarray:
        """Synthesize the piece with the built-in synth: float32 mono in [-1, 1]."""
        spb = self.seconds_per_beat
        tracks = [
            synth.render_voice(
                voice.pitches,
                voice.onsets * spb,
                voice.durations * spb,
                total_seconds=self.duration_seconds,
                waveform=voice.waveform,
                velocity=voice.velocity,
                sample_rate=sample_rate,
            )
            for voice in self.voices
        ]
        return synth.mix(tracks)

    def wav_bytes(self, sample_rate: int = 44100) -> bytes:
        return synth.wav_bytes(self.render(sample_rate), sample_rate)

    def write_wav(self, path: str | Path, sample_rate: int = 44100) -> Path:
        """Write a 16-bit mono .wav file and return its path."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.wav_bytes(sample_rate))
        return path


def compose(
    voices: PathSimulator | VoiceSpec | Sequence[PathSimulator | VoiceSpec],
    *,
    length: float = 32.0,
    bpm: float = 100.0,
    seed: int | None = None,
    correlation=None,
    dt: float = 0.01,
) -> Piece:
    """
    Generate a piece: one melody per voice.

    Parameters
    ----------
    voices : a process, a VoiceSpec, or a list of them
        A bare process gets the default clock, pitch range and sound.
    length : float
        Length of the piece in beats. Time in every process is measured
        in beats, so a mean-reversion speed of 2 means "per beat".
    bpm : float
        Tempo, in beats per minute.
    seed : int, optional
        Makes the piece reproducible.
    correlation : None, float or matrix
        Correlation between the voices' main shocks: None for independent
        voices, one number for every pair, or a full matrix. Positive
        values make voices move together, negative ones in contrary motion.
    dt : float
        Simulation step of the paths, in beats.

    Returns
    -------
    Piece
    """
    if isinstance(voices, PathSimulator | VoiceSpec):
        voices = [voices]
    specs = [v if isinstance(v, VoiceSpec) else VoiceSpec(process=v) for v in voices]
    if not specs:
        raise ValueError("At least one voice is required.")
    if bpm <= 0:
        raise ValueError(f"bpm must be > 0, got {bpm}")

    # Independent random streams: one for the paths, one per voice's clock.
    path_seed, *clock_seeds = np.random.SeedSequence(seed).spawn(len(specs) + 1)

    times, paths, latents = simulate_correlated(
        [spec.process for spec in specs], T=length, dt=dt, correlation=correlation, seed=path_seed
    )

    generated = []
    for index, (spec, values, latent, clock_seed) in enumerate(
        zip(specs, paths, latents, clock_seeds, strict=True)
    ):
        onsets = spec.clock.get_sample_times(length, np.random.default_rng(clock_seed))
        sampled = sample_path(times, values, onsets)
        generated.append(
            Voice(
                name=spec.name or f"Voice {index + 1}",
                times=times,
                values=values,
                latent=latent,
                onsets=onsets,
                durations=note_durations(onsets, length, spec.gate, spec.max_note_beats),
                sampled_values=sampled,
                pitches=spec.mapper.map(sampled),
                program=spec.program,
                velocity=spec.velocity,
                waveform=spec.waveform,
            )
        )

    return Piece(voices=generated, bpm=bpm, length=length, seed=seed)

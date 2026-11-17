"""
Audio preview layer.

Responsibility: turn notes into a waveform you can listen to straight
away, with nothing but numpy. It is a deliberately small additive
synthesizer, meant for auditioning a piece - for a finished sound, load
the MIDI file into a DAW or a real synth.
"""

import io
import wave

import numpy as np

WAVEFORMS = ("triangle", "sine", "saw", "square")

_MAX_HARMONICS = 10
_ATTACK, _DECAY, _RELEASE = 0.005, 0.08, 0.04  # seconds
_SUSTAIN = 0.7


def midi_to_hz(pitch: float) -> float:
    """Frequency of a MIDI pitch (A4 = 69 = 440 Hz)."""
    return 440.0 * 2.0 ** ((pitch - 69) / 12.0)


def _harmonics(waveform: str) -> list[tuple[int, float]]:
    """(harmonic number, amplitude) pairs of the waveform's Fourier series."""
    if waveform == "sine":
        return [(1, 1.0)]
    if waveform == "triangle":
        return [(k, (-1.0) ** ((k - 1) // 2) / k**2) for k in range(1, 2 * _MAX_HARMONICS, 2)]
    if waveform == "square":
        return [(k, 1.0 / k) for k in range(1, 2 * _MAX_HARMONICS, 2)]
    if waveform == "saw":
        return [(k, (-1.0) ** (k + 1) / k) for k in range(1, _MAX_HARMONICS + 1)]
    raise ValueError(f"Unknown waveform {waveform!r}. Available: {WAVEFORMS}.")


def _tone(pitch: int, num_samples: int, waveform: str, sample_rate: int) -> np.ndarray:
    """One note of `num_samples` samples (release included), peak amplitude about 1."""
    t = np.arange(num_samples, dtype=np.float32) / sample_rate
    frequency = midi_to_hz(pitch)

    signal = np.zeros(num_samples, dtype=np.float32)
    total = 0.0
    for k, amplitude in _harmonics(waveform):
        if k * frequency >= sample_rate / 2:  # stay below Nyquist
            break
        signal += amplitude * np.sin(2 * np.pi * k * frequency * t)
        total += abs(amplitude)
    if total > 0:
        signal /= total

    # attack / decay / sustain / release envelope
    release = min(int(_RELEASE * sample_rate), num_samples)
    envelope = np.full(num_samples, _SUSTAIN, dtype=np.float32)
    attack = min(int(_ATTACK * sample_rate), num_samples)
    decay = min(int(_DECAY * sample_rate), num_samples - attack)
    envelope[:attack] = np.linspace(0.0, 1.0, attack, endpoint=False)
    envelope[attack : attack + decay] = np.linspace(1.0, _SUSTAIN, decay, endpoint=False)
    if release > 0:
        envelope[-release:] *= np.linspace(1.0, 0.0, release)
    return signal * envelope


def render_voice(
    pitches: np.ndarray,
    starts: np.ndarray,
    durations: np.ndarray,
    total_seconds: float,
    waveform: str = "triangle",
    velocity: int = 90,
    sample_rate: int = 44100,
) -> np.ndarray:
    """
    Render one voice. `starts` and `durations` are in seconds. Returns a
    float32 array covering `total_seconds` (plus the last note's release).
    """
    tail = int(_RELEASE * sample_rate)
    audio = np.zeros(int(np.ceil(total_seconds * sample_rate)) + tail, dtype=np.float32)
    gain = velocity / 127.0

    cache: dict[tuple[int, int], np.ndarray] = {}
    for pitch, start, duration in zip(pitches, starts, durations, strict=True):
        begin = int(round(start * sample_rate))
        num_samples = min(int(round(duration * sample_rate)) + tail, len(audio) - begin)
        if num_samples <= 0:
            continue
        key = (int(pitch), num_samples)
        if key not in cache:
            cache[key] = _tone(int(pitch), num_samples, waveform, sample_rate)
        audio[begin : begin + num_samples] += gain * cache[key]
    return audio


def mix(tracks: list[np.ndarray], peak: float = 0.9) -> np.ndarray:
    """Sum the tracks and rescale so the loudest sample sits at `peak`."""
    if not tracks:
        return np.zeros(0, dtype=np.float32)
    length = max(len(track) for track in tracks)
    audio = np.zeros(length, dtype=np.float32)
    for track in tracks:
        audio[: len(track)] += track
    loudest = float(np.max(np.abs(audio))) if length else 0.0
    if loudest > 0:
        audio *= peak / loudest
    return audio


def wav_bytes(audio: np.ndarray, sample_rate: int = 44100) -> bytes:
    """Encode a float waveform in [-1, 1] as a 16-bit mono WAV file."""
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())
    return buffer.getvalue()

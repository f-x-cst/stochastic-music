"""
MIDI output layer.

Responsibility: turn the notes of a piece into a standard MIDI file, one
track per voice. This is the ONLY module that imports mido - switching
MIDI library means changing this file and nothing else.
"""

import io
from collections.abc import Sequence

import mido
import numpy as np

TICKS_PER_BEAT = 480
_DRUM_CHANNEL = 9  # reserved for percussion in General MIDI


def _channel(voice_index: int) -> int:
    """MIDI channel for a voice, skipping the drum channel."""
    channel = voice_index % 15
    return channel + 1 if channel >= _DRUM_CHANNEL else channel


def build_midi(voices: Sequence, bpm: float) -> mido.MidiFile:
    """
    Build a multi-track MIDI file.

    Parameters
    ----------
    voices : sequence of objects with attributes
        name, pitches, onsets (beats), durations (beats), program, velocity
        (e.g. stochastic_music.Voice).
    bpm : float
        Tempo in beats per minute.
    """
    midi = mido.MidiFile(type=1, ticks_per_beat=TICKS_PER_BEAT)

    conductor = mido.MidiTrack()
    conductor.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(bpm), time=0))
    conductor.append(mido.MetaMessage("time_signature", numerator=4, denominator=4, time=0))
    midi.tracks.append(conductor)

    for index, voice in enumerate(voices):
        channel = _channel(index)
        track = mido.MidiTrack()
        track.append(mido.MetaMessage("track_name", name=str(voice.name), time=0))
        track.append(
            mido.Message("program_change", program=int(voice.program), channel=channel, time=0)
        )

        # (tick, order, message type, pitch): note-offs sort before note-ons
        # on the same tick, so a repeated pitch is released before it restarts.
        events = []
        starts = np.round(np.asarray(voice.onsets) * TICKS_PER_BEAT).astype(int)
        lengths = np.maximum(np.round(np.asarray(voice.durations) * TICKS_PER_BEAT).astype(int), 1)
        for pitch, start, length in zip(voice.pitches, starts, lengths, strict=True):
            events.append((int(start), 1, "note_on", int(pitch)))
            events.append((int(start + length), 0, "note_off", int(pitch)))
        events.sort()

        now = 0
        for tick, _, kind, pitch in events:
            velocity = int(voice.velocity) if kind == "note_on" else 0
            track.append(
                mido.Message(kind, note=pitch, velocity=velocity, channel=channel, time=tick - now)
            )
            now = tick
        midi.tracks.append(track)

    return midi


def midi_bytes(voices: Sequence, bpm: float) -> bytes:
    """The MIDI file as bytes (e.g. for a download button)."""
    buffer = io.BytesIO()
    build_midi(voices, bpm).save(file=buffer)
    return buffer.getvalue()

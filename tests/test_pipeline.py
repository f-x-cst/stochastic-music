import io
import wave

import mido
import numpy as np
import pytest

import stochastic_music as sm


@pytest.mark.parametrize("key", list(sm.PROCESSES))
def test_every_catalog_process_composes(key):
    piece = sm.compose(sm.PROCESSES[key].build(), length=16, seed=0)
    voice = piece.voices[0]
    assert len(voice.pitches) == 16
    assert voice.pitches.min() >= 48 and voice.pitches.max() <= 72


def test_compose_is_reproducible():
    specs = [
        sm.VoiceSpec(sm.MertonJumpDiffusion(), clock=sm.HawkesClock()),
        sm.VoiceSpec(sm.HestonProcess(), clock=sm.PoissonClock()),
    ]
    a = sm.compose(specs, length=16, seed=5, correlation=0.4)
    b = sm.compose(specs, length=16, seed=5, correlation=0.4)
    for va, vb in zip(a.voices, b.voices, strict=True):
        np.testing.assert_array_equal(va.pitches, vb.pitches)
        np.testing.assert_array_equal(va.onsets, vb.onsets)


def test_multi_voice_piece_with_mixed_clocks():
    specs = [
        sm.VoiceSpec(sm.OrnsteinUhlenbeckProcess(), name="bass", program=38),
        sm.VoiceSpec(sm.WienerProcess(), clock=sm.ProcessClock(sm.CIRProcess()), gate=0.5),
        sm.VoiceSpec(sm.VarianceGammaProcess(), clock=sm.HawkesClock(), max_note_beats=1),
    ]
    piece = sm.compose(specs, length=32, bpm=140, seed=1, correlation=0.5)
    assert [v.name for v in piece.voices] == ["bass", "Voice 2", "Voice 3"]
    assert piece.voices[2].durations.max() <= 1
    for voice in piece.voices:
        assert len(voice.pitches) == len(voice.onsets) == len(voice.durations)


def test_midi_file_contents():
    specs = [sm.VoiceSpec(sm.WienerProcess(), program=40), sm.VoiceSpec(sm.GBMProcess())]
    piece = sm.compose(specs, length=8, bpm=120, seed=0)
    midi = mido.MidiFile(file=io.BytesIO(piece.midi_bytes()))

    assert len(midi.tracks) == 3  # tempo track + one per voice
    assert midi.length == pytest.approx(piece.duration_seconds, abs=0.01)
    notes = [m for m in midi.tracks[1] if m.type == "note_on" and m.velocity > 0]
    assert [m.note for m in notes] == list(piece.voices[0].pitches)
    programs = [m.program for m in midi.tracks[1] if m.type == "program_change"]
    assert programs == [40]
    channels = {m.channel for track in midi.tracks[1:] for m in track if hasattr(m, "channel")}
    assert 9 not in channels  # the drum channel is skipped


def test_repeated_pitch_is_released_before_restarting():
    piece = sm.compose(sm.WienerProcess(), length=4, seed=0)
    piece.voices[0].pitches[:] = 60
    midi = mido.MidiFile(file=io.BytesIO(piece.midi_bytes()))
    kinds = [m.type for m in midi.tracks[1] if m.type in ("note_on", "note_off")]
    assert kinds == ["note_on", "note_off"] * 4


def test_wav_output(tmp_path):
    piece = sm.compose(sm.WienerProcess(), length=8, bpm=120, seed=0)
    path = piece.write_wav(tmp_path / "out" / "piece.wav", sample_rate=22050)
    with wave.open(str(path)) as wav:
        assert wav.getframerate() == 22050 and wav.getnchannels() == 1
        assert wav.getnframes() / 22050 == pytest.approx(4.0, abs=0.1)
    audio = piece.render(22050)
    assert np.abs(audio).max() == pytest.approx(0.9, abs=1e-3)


def test_write_midi(tmp_path):
    path = sm.compose(sm.WienerProcess(), length=4, seed=0).write_midi(tmp_path / "a.mid")
    assert path.read_bytes().startswith(b"MThd")


@pytest.mark.parametrize("waveform", sm.WAVEFORMS)
def test_every_waveform_renders(waveform):
    piece = sm.compose(sm.VoiceSpec(sm.WienerProcess(), waveform=waveform), length=4, seed=0)
    assert np.isfinite(piece.render(8000)).all()


def test_invalid_voice_settings():
    with pytest.raises(ValueError):
        sm.VoiceSpec(sm.WienerProcess(), program=200)
    with pytest.raises(ValueError):
        sm.VoiceSpec(sm.WienerProcess(), waveform="kazoo")
    with pytest.raises(ValueError):
        sm.compose([], length=4)

"""
Three short pieces showing the main features. Writes .mid and .wav files to ./outputs/.

    python examples/quickstart.py
"""

import stochastic_music as sm

OUT = "outputs"

# 1. One voice, one process, default rhythm (quarter notes, chromatic, C3 to C5).
piece = sm.compose(sm.OrnsteinUhlenbeckProcess(kappa=3.0), length=32, bpm=110)
piece.write_midi(f"{OUT}/01_ou.mid")
piece.write_wav(f"{OUT}/01_ou.wav")

# 2. Jumps in a scale, with a bursty Hawkes rhythm.
piece = sm.compose(
    sm.VoiceSpec(
        sm.MertonJumpDiffusion(lambda_jump=0.8, jump_std=4.0),
        clock=sm.HawkesClock(base_rate=0.6, excitation=0.75, decay=6.0),
        mapper=sm.RealizedRangeBinner(55, 79, scale="Phrygian", root="E"),
        max_note_beats=1.0,
        waveform="saw",
    ),
    length=64,
    bpm=150,
)
piece.write_midi(f"{OUT}/02_merton_hawkes.mid")
piece.write_wav(f"{OUT}/02_merton_hawkes.wav")

# 3. Three correlated voices driven by different processes.
key = {"scale": "Dorian", "root": "D"}
piece = sm.compose(
    [
        sm.VoiceSpec(
            sm.CIRProcess(kappa=1.0, xi=0.8),
            clock=sm.UniformClock(2.0),
            mapper=sm.RealizedRangeBinner(36, 50, **key),
            name="Bass",
            program=32,
            waveform="sine",
        ),
        sm.VoiceSpec(
            sm.HestonProcess(rough=True, hurst=0.1),
            clock=sm.UniformClock(0.5),
            mapper=sm.RealizedRangeBinner(55, 72, **key),
            name="Middle",
            program=11,
        ),
        sm.VoiceSpec(
            sm.WienerProcess(),
            clock=sm.ProcessClock(sm.OrnsteinUhlenbeckProcess(), durations=[0.25, 0.5, 1.0]),
            mapper=sm.RealizedRangeBinner(67, 86, **key),
            name="Lead",
            program=73,
            gate=0.7,
        ),
    ],
    length=64,
    bpm=96,
    correlation=0.6,
)
piece.write_midi(f"{OUT}/03_three_voices.mid")
piece.write_wav(f"{OUT}/03_three_voices.wav")

print(f"Done: files written to ./{OUT}/")

"""
Stochastic Music web app.

Run it with `stochastic-music` (after `pip install "stochastic-music[app]"`)
or with `streamlit run src/stochastic_music/app.py` from the repository.
"""

import math
import secrets

import matplotlib.pyplot as plt
import streamlit as st

import stochastic_music as sm
from stochastic_music.pitch_mapping import CHROMATIC
from stochastic_music.plotting import plot_piece

MAX_VOICES = 4
PREVIEW_SAMPLE_RATE = 22050
MAX_PREVIEW_SECONDS = 600

# A few General MIDI instruments (name -> program number), stored in the MIDI file.
GM_PROGRAMS = {
    "Acoustic grand piano": 0,
    "Electric piano": 4,
    "Vibraphone": 11,
    "Marimba": 12,
    "Church organ": 19,
    "Nylon guitar": 24,
    "Acoustic bass": 32,
    "Synth bass": 38,
    "Violin": 40,
    "Cello": 42,
    "String ensemble": 48,
    "Trumpet": 56,
    "Flute": 73,
    "Square lead": 80,
    "Saw lead": 81,
    "Warm pad": 89,
}

STEPS = {"Half notes": 2.0, "Quarter notes": 1.0, "Eighth notes": 0.5, "Sixteenth notes": 0.25}
GRIDS = {"Sixteenth notes": 0.25, "Eighth notes": 0.5, "Quarter notes": 1.0, "Free (no grid)": None}
NOTE_CAPS = {"No cap": None, "1 beat": 1.0, "2 beats": 2.0, "4 beats": 4.0}


def process_controls(prefix: str, label: str, default: str) -> sm.PathSimulator:
    """Process picker plus one control per audible parameter. Returns the process."""
    keys = list(sm.PROCESSES)
    key = st.selectbox(
        label,
        keys,
        index=keys.index(default),
        format_func=lambda k: sm.PROCESSES[k].label,
        key=f"{prefix}_process",
    )
    info = sm.PROCESSES[key]
    st.caption(info.summary)

    shown = [param for param in info.params if param.audible]
    if not shown:
        st.info(
            "Nothing to tune here. The path is stretched to fit the pitch range, so its "
            "volatility and starting value cannot be heard: only the seed changes the melody."
        )
        return info.build()

    kwargs = {}
    columns = st.columns(2)
    for position, param in enumerate(shown):
        if param.requires and not kwargs.get(param.requires):
            continue
        widget_key = f"{prefix}_{key}_{param.name}"
        with columns[position % 2]:
            if param.is_flag:
                kwargs[param.name] = st.checkbox(param.label, value=param.default, key=widget_key)
            else:
                kwargs[param.name] = st.slider(
                    param.label,
                    min_value=float(param.low),
                    max_value=float(param.high),
                    value=float(param.default),
                    step=float(param.step),
                    key=widget_key,
                )
    return info.build(**kwargs)


def clock_controls(prefix: str) -> tuple[sm.Clock, float, float | None]:
    """Rhythm controls. Returns (clock, gate, maximum note length)."""
    kind = st.selectbox(
        "Rhythm",
        ["Fixed grid", "Poisson (random)", "Hawkes (bursts)", "Driven by a process"],
        key=f"{prefix}_clock",
        help=(
            "Fixed grid: one note per step. Poisson: notes at random. Hawkes: each note "
            "raises the chance of another, giving bursts and silences. Driven by a process: "
            "a second path sets the note lengths (high = short notes)."
        ),
    )

    if kind == "Fixed grid":
        step = st.selectbox("Step", list(STEPS), index=1, key=f"{prefix}_step")
        clock = sm.UniformClock(STEPS[step])
    elif kind == "Poisson (random)":
        left, right = st.columns(2)
        rate = left.slider("Notes per beat", 0.25, 4.0, 2.0, 0.05, key=f"{prefix}_rate")
        grid = right.selectbox("Snap to", list(GRIDS), key=f"{prefix}_pgrid")
        clock = sm.PoissonClock(rate, GRIDS[grid])
    elif kind == "Hawkes (bursts)":
        left, right = st.columns(2)
        base = left.slider("Base notes per beat", 0.1, 4.0, 0.5, 0.05, key=f"{prefix}_base")
        excitation = right.slider(
            "Excitation",
            0.0,
            0.95,
            0.7,
            0.01,
            key=f"{prefix}_excitation",
            help="Average number of extra notes each note triggers. Higher = longer bursts.",
        )
        decay = left.slider(
            "Decay",
            0.5,
            16.0,
            4.0,
            0.5,
            key=f"{prefix}_decay",
            help="How fast the excitement fades, per beat. Higher = tighter bursts.",
        )
        grid = right.selectbox("Snap to", list(GRIDS), key=f"{prefix}_hgrid")
        clock = sm.HawkesClock(base, excitation, decay, GRIDS[grid])
    else:
        lengths = st.multiselect(
            "Note lengths (beats)",
            [0.25, 0.5, 1.0, 2.0, 4.0],
            default=[0.25, 0.5, 1.0, 2.0],
            key=f"{prefix}_lengths",
        )
        driver = process_controls(f"{prefix}_rhythm", "Rhythm process", "ou")
        clock = sm.ProcessClock(driver, lengths or [1.0])

    left, right = st.columns(2)
    gate = left.slider(
        "Gate",
        0.1,
        1.0,
        1.0,
        0.05,
        key=f"{prefix}_gate",
        help="Share of each step during which the note sounds. 1 = legato, lower = staccato.",
    )
    cap = right.selectbox(
        "Longest note",
        list(NOTE_CAPS),
        key=f"{prefix}_cap",
        help="Stops a note that precedes a long silence from droning.",
    )
    return clock, gate, NOTE_CAPS[cap]


def voice_controls(index: int, scale: str, root: int) -> sm.VoiceSpec:
    """All the controls of one voice. Returns its VoiceSpec."""
    prefix = f"v{index}"
    melody, rhythm = st.columns(2, gap="large")

    with melody:
        st.subheader("Melody")
        process = process_controls(prefix, "Process", "wiener")
        low, high = st.slider(
            "Pitch range (MIDI)",
            36,
            84,
            (48, 72),
            key=f"{prefix}_range",
            help="60 is middle C. 12 semitones make an octave.",
        )
        st.caption(f"{sm.note_name(low)} to {sm.note_name(high)}")

    with rhythm:
        st.subheader("Rhythm")
        clock, gate, cap = clock_controls(prefix)
        st.subheader("Sound")
        left, right = st.columns(2)
        waveform = left.selectbox(
            "Preview waveform",
            sm.WAVEFORMS,
            key=f"{prefix}_waveform",
            help="Timbre of the built-in player.",
        )
        instrument = right.selectbox(
            "MIDI instrument",
            list(GM_PROGRAMS),
            key=f"{prefix}_program",
            help="Stored in the MIDI file, for use in a DAW or a MIDI player.",
        )

    return sm.VoiceSpec(
        process=process,
        clock=clock,
        mapper=sm.RealizedRangeBinner(low, high, scale, root),
        name=f"Voice {index + 1}",
        program=GM_PROGRAMS[instrument],
        gate=gate,
        max_note_beats=cap,
        waveform=waveform,
    )


def generate(specs, length, bpm, seed, correlation) -> None:
    """Compose the piece and keep everything needed to display it."""
    used_seed = seed if seed != 0 else secrets.randbelow(2**31 - 1) + 1
    piece = sm.compose(specs, length=length, bpm=bpm, seed=used_seed, correlation=correlation)

    audio = None
    if piece.duration_seconds <= MAX_PREVIEW_SECONDS:
        audio = piece.wav_bytes(PREVIEW_SAMPLE_RATE)
    st.session_state["result"] = {"piece": piece, "midi": piece.midi_bytes(), "wav": audio}


def show_result(result) -> None:
    piece = result["piece"]
    notes = sum(len(voice.pitches) for voice in piece.voices)
    st.caption(
        f"Seed {piece.seed} · {notes} notes · {piece.length:g} beats at {piece.bpm:g} BPM "
        f"({piece.duration_seconds:.0f} s)"
    )

    if result["wav"] is not None:
        st.audio(result["wav"], format="audio/wav")
    else:
        st.warning(
            f"The piece lasts more than {MAX_PREVIEW_SECONDS // 60} minutes, so no audio "
            "preview was rendered. The MIDI file is complete."
        )

    left, right, _ = st.columns([1, 1, 3])
    left.download_button(
        "Download MIDI", result["midi"], f"stochastic_{piece.seed}.mid", "audio/midi"
    )
    if result["wav"] is not None:
        right.download_button(
            "Download WAV", result["wav"], f"stochastic_{piece.seed}.wav", "audio/wav"
        )

    figure = plot_piece(piece)
    st.pyplot(figure)
    plt.close(figure)

    with st.expander("Notes"):
        for voice in piece.voices:
            st.markdown(f"**{voice.name}**")
            st.text(" ".join(sm.note_name(pitch) for pitch in voice.pitches) or "(no notes)")


def main() -> None:
    st.set_page_config(page_title="Stochastic Music", page_icon="🎼", layout="wide")
    st.title("Stochastic Music")
    st.caption("Simulate a stochastic process and hear its path as a melody. One path, one voice.")

    with st.sidebar:
        st.header("Piece")
        length = st.slider("Length (beats)", 4, 1024, 32, 4)
        bpm = st.slider("Tempo (BPM)", 40, 600, 100, 5)
        seed = st.number_input(
            "Seed", min_value=0, value=42, step=1, help="0 draws a new random seed every time."
        )

        st.header("Key")
        scale = st.selectbox(
            "Scale / mode", list(sm.SCALES), index=list(sm.SCALES).index(CHROMATIC)
        )
        root = 0
        if scale != CHROMATIC:
            root = st.selectbox(
                "Root note",
                range(12),
                format_func=lambda pitch_class: sm.ROOT_NOTE_NAMES[pitch_class],
            )

        st.header("Voices")
        num_voices = st.slider("Number of voices", 1, MAX_VOICES, 1)
        correlation = None
        if num_voices > 1:
            lowest = -1.0 / (num_voices - 1)
            correlation = st.slider(
                "Correlation between voices",
                min_value=math.ceil(lowest * 100) / 100,
                max_value=1.0,
                value=0.0,
                step=0.01,
                help=(
                    "Positive: the voices move together. Negative: contrary motion. "
                    f"With {num_voices} voices it cannot go below {lowest:.2f}."
                ),
            )

        clicked = st.button("Generate", type="primary", width="stretch")

    tabs = st.tabs([f"Voice {index + 1}" for index in range(num_voices)])
    specs = []
    for index, tab in enumerate(tabs):
        with tab:
            specs.append(voice_controls(index, scale, root))

    if clicked:
        try:
            with st.spinner("Simulating..."):
                generate(specs, float(length), float(bpm), int(seed), correlation)
        except ValueError as error:
            st.error(str(error))

    st.divider()
    if "result" in st.session_state:
        show_result(st.session_state["result"])
    else:
        st.info("Set up the voices, then press **Generate** in the sidebar.")


if __name__ == "__main__":
    main()

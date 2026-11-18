"""
Plotting helpers (optional: needs matplotlib, `pip install "stochastic-music[plot]"`).
"""

import numpy as np

from .pipeline import Piece

# Okabe-Ito palette: distinguishable with the common forms of colour blindness
VOICE_COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]
_MAX_POINTS = 6000


def _thin(times: np.ndarray, values: np.ndarray):
    """Keep at most _MAX_POINTS points of a path, for faster drawing."""
    stride = max(1, len(times) // _MAX_POINTS)
    return times[::stride], values[::stride]


def plot_piece(piece: Piece, show_latent: bool = True, piano_roll: bool = True):
    """
    Draw a piece: one panel per voice (the raw path and the points read
    as notes), then, if `piano_roll` is True, a piano roll of all voices.

    Returns the matplotlib Figure.
    """
    import matplotlib.pyplot as plt

    num_voices = len(piece.voices)
    ratios = [1.0] * num_voices + ([1.2] if piano_roll else [])
    fig, axes = plt.subplots(
        len(ratios),
        1,
        figsize=(10, 2.3 * num_voices + (2.8 if piano_roll else 0.6)),
        sharex=True,
        gridspec_kw={"height_ratios": ratios},
        squeeze=False,
    )
    axes = axes[:, 0]

    for index, (voice, ax) in enumerate(zip(piece.voices, axes, strict=False)):
        color = VOICE_COLORS[index % len(VOICE_COLORS)]
        ax.plot(*_thin(voice.times, voice.values), color=color, linewidth=1.0, label="path")
        ax.scatter(voice.onsets, voice.sampled_values, s=14, color="black", zorder=3, label="notes")
        ax.set_ylabel("path value")
        ax.set_title(voice.name, loc="left", fontsize=10, fontweight="bold")

        if show_latent and voice.latent:
            name, series = next(iter(voice.latent.items()))
            twin = ax.twinx()
            twin.plot(*_thin(voice.times, series), color="#BBBBBB", linewidth=0.7)
            twin.set_ylabel(name, color="grey")
            twin.tick_params(axis="y", colors="grey")
            twin.spines[["top"]].set_visible(False)
            # keep the hidden series behind the path
            ax.set_zorder(twin.get_zorder() + 1)
            ax.patch.set_visible(False)
        ax.spines[["top", "right"]].set_visible(False)

    if not piano_roll:
        axes[-1].set_xlabel("time (beats)")
        axes[-1].set_xlim(0, piece.length)
        fig.tight_layout()
        return fig

    roll = axes[-1]
    for index, voice in enumerate(piece.voices):
        roll.barh(
            voice.pitches,
            voice.durations,
            left=voice.onsets,
            height=0.8,
            color=VOICE_COLORS[index % len(VOICE_COLORS)],
            alpha=0.85,
            label=voice.name,
        )
    roll.set_ylabel("MIDI pitch")
    roll.set_xlabel("time (beats)")
    roll.set_title("Piano roll", loc="left", fontsize=10, fontweight="bold")
    roll.set_xlim(0, piece.length)
    roll.spines[["top", "right"]].set_visible(False)
    if num_voices > 1:
        roll.legend(
            loc="lower right",
            bbox_to_anchor=(1.0, 1.0),
            fontsize=8,
            ncols=num_voices,
            frameon=False,
        )

    fig.tight_layout()
    return fig

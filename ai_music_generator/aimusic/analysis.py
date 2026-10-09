"""Music analysis helpers: key estimation and mood labels.

MOOD LABELS ARE HEURISTIC.  No dataset ships with human mood annotations, so
every training file is labelled automatically from two measurable properties:

    mode    major / minor   (Krumhansl-Schmuckler key-finding on the pitches)
    energy  note onsets per second, ranked *within the file's own genre*

    energy rank >= 75 %                 -> "energetic"
    otherwise minor                     -> "sad"
    otherwise major, energy rank < 40 % -> "relaxing"
    otherwise major                     -> "happy"

The LSTM then learns what music carrying each label looks like.  It does not
"understand" emotion - it reproduces the statistical patterns of the files that
were given each label.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

from .midi_io import Note

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Krumhansl-Kessler key profiles
_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def pitch_class_histogram(notes: Sequence[Note]) -> np.ndarray:
    hist = np.zeros(12)
    for s, e, p, _v in notes:
        hist[p % 12] += max(1, e - s)
    return hist


def estimate_key(notes: Sequence[Note]) -> Tuple[str, str, float]:
    """Return (tonic name, 'major'|'minor', correlation).  Empty input -> C major."""
    hist = pitch_class_histogram(notes)
    if hist.sum() == 0 or np.std(hist) == 0:
        return "C", "major", 0.0
    best = ("C", "major", -2.0)
    for tonic in range(12):
        for mode, profile in (("major", _MAJOR), ("minor", _MINOR)):
            corr = float(np.corrcoef(hist, np.roll(profile, tonic))[0, 1])
            if corr > best[2]:
                best = (NOTE_NAMES[tonic], mode, corr)
    return best


def energy_of(note_count: int, seconds: float) -> float:
    """Note onsets per second."""
    return note_count / max(seconds, 1.0)


def assign_moods(modes: Sequence[str], energies: Sequence[float],
                 genres: Sequence[str]) -> List[str]:
    """Label every file.  Energy is ranked inside each genre (percentile)."""
    moods = [""] * len(modes)
    for genre in sorted(set(genres)):
        idx = [i for i, g in enumerate(genres) if g == genre]
        en = np.array([energies[i] for i in idx])
        order = en.argsort().argsort()                   # rank 0..n-1
        pct = (order + 0.5) / len(idx)                   # percentile in (0,1)
        for k, i in enumerate(idx):
            if pct[k] >= 0.75:
                moods[i] = "energetic"
            elif modes[i] == "minor":
                moods[i] = "sad"
            elif pct[k] < 0.40:
                moods[i] = "relaxing"
            else:
                moods[i] = "happy"
    return moods


def note_name(pitch: int) -> str:
    return f"{NOTE_NAMES[pitch % 12]}{pitch // 12 - 1}"

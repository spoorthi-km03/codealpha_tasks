"""Test helpers.

The MIDI files created here are SYNTHETIC TEST FIXTURES (simple scales and
arpeggios).  They exist only so the tests can exercise the whole pipeline
offline.  They are never used to build the model that ships with the project.
"""
import random
from pathlib import Path

from aimusic.config import STEPS_PER_BEAT
from aimusic.midi_io import write_midi

MAJOR = [0, 2, 4, 5, 7, 9, 11, 12]
MINOR = [0, 2, 3, 5, 7, 8, 10, 12]


def make_fixture_midi(path: Path, minor: bool, busy: bool, seed: int, beats: int = 64) -> Path:
    rnd = random.Random(seed)
    scale = MINOR if minor else MAJOR
    root = 48 + rnd.choice([0, 2, 5, 7])
    notes, step = [], 0
    unit = 3 if busy else 12                        # 16th notes vs quarter notes
    total = beats * STEPS_PER_BEAT
    idx = 0
    while step < total:
        idx = max(0, min(len(scale) - 1, idx + rnd.choice([-2, -1, 1, 1, 2])))
        p = root + scale[idx] + (12 if busy else 0)
        notes.append((step, step + unit, p, rnd.randint(50, 100)))
        if not busy and rnd.random() < 0.4:
            notes.append((step, step + unit * 2, p - 12, 70))   # a bass note
        step += unit
    return write_midi(path, notes, bpm=rnd.choice([90, 100, 120]), steps_per_beat=STEPS_PER_BEAT)


def make_fixture_corpus(raw: Path, genres=("alpha", "beta"), per_genre: int = 10) -> Path:
    for gi, g in enumerate(genres):
        for i in range(per_genre):
            make_fixture_midi(raw / g / f"{g}_{i}.mid", minor=(i % 2 == 0), busy=(gi == 1),
                              seed=gi * 100 + i)
    return raw

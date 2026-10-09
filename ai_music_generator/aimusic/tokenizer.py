"""Turn notes into a sequence of integer tokens (and back).

The vocabulary is an *event* representation in the style of Google's
Performance-RNN.  Music becomes a "sentence" the LSTM can learn to continue:

    id range        meaning
    0               PAD
    1               BOS   (start of piece)
    2               EOS   (end of piece)
    3   .. 90       NOTE_ON  pitch 21..108      (88 piano keys)
    91  .. 178      NOTE_OFF pitch 21..108
    179 .. 202      TIME_SHIFT 1..24 grid steps (1 step = 1/12 beat)
    203 .. 210      VELOCITY bin 0..7           (loudness of the following notes)

Example: a C-major chord for one beat followed by one beat of silence:
    VEL(5) ON(60) ON(64) ON(67) SHIFT(12) OFF(60) OFF(64) OFF(67) SHIFT(12)
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

from .config import MAX_HOLD_BEATS, PITCH_MAX, PITCH_MIN, STEPS_PER_BEAT
from .midi_io import Note

N_PITCH = PITCH_MAX - PITCH_MIN + 1  # 88
MAX_SHIFT = 24
N_VEL = 8

PAD, BOS, EOS = 0, 1, 2
ON_BASE = 3
OFF_BASE = ON_BASE + N_PITCH            # 91
SHIFT_BASE = OFF_BASE + N_PITCH         # 179
VEL_BASE = SHIFT_BASE + MAX_SHIFT       # 203
VOCAB_SIZE = VEL_BASE + N_VEL           # 211
MAX_HOLD_STEPS = MAX_HOLD_BEATS * STEPS_PER_BEAT


def on_token(pitch: int) -> int:
    return ON_BASE + pitch - PITCH_MIN


def off_token(pitch: int) -> int:
    return OFF_BASE + pitch - PITCH_MIN


def shift_token(steps: int) -> int:
    if not 1 <= steps <= MAX_SHIFT:
        raise ValueError("shift out of range")
    return SHIFT_BASE + steps - 1


def vel_token(velocity: int) -> int:
    return VEL_BASE + vel_bin(velocity)


def vel_bin(velocity: int) -> int:
    return int(min(N_VEL - 1, max(0, velocity // 16)))


def bin_velocity(b: int) -> int:
    return int(min(127, b * 16 + 12))


def token_kind(tok: int) -> str:
    if tok < ON_BASE:
        return "special"
    if tok < OFF_BASE:
        return "on"
    if tok < SHIFT_BASE:
        return "off"
    if tok < VEL_BASE:
        return "shift"
    return "vel"


def describe(tok: int) -> str:
    kind = token_kind(tok)
    if kind == "special":
        return ["PAD", "BOS", "EOS"][tok]
    if kind == "on":
        return f"ON({tok - ON_BASE + PITCH_MIN})"
    if kind == "off":
        return f"OFF({tok - OFF_BASE + PITCH_MIN})"
    if kind == "shift":
        return f"SHIFT({tok - SHIFT_BASE + 1})"
    return f"VEL({tok - VEL_BASE})"


# --------------------------------------------------------------------------- #
# notes -> tokens
# --------------------------------------------------------------------------- #
def notes_to_tokens(notes: Sequence[Note], add_bos: bool = False,
                    add_eos: bool = False) -> List[int]:
    """Encode (start_step, end_step, pitch, velocity) notes as tokens."""
    # 1. clean: in-range pitches, bounded length, no overlapping same-pitch notes
    by_pitch: Dict[int, List[list]] = {}
    for s, e, p, v in notes:
        if PITCH_MIN <= p <= PITCH_MAX:
            e = min(max(e, s + 1), s + MAX_HOLD_STEPS)
            by_pitch.setdefault(p, []).append([s, e, p, v])
    events: List[Tuple[int, int, int, int]] = []   # (step, order, pitch, vel)
    for p, lst in by_pitch.items():
        lst.sort()
        for idx, note in enumerate(lst):
            if idx + 1 < len(lst):
                note[1] = min(note[1], lst[idx + 1][0])
            if note[1] > note[0]:
                events.append((note[0], 1, p, note[3]))
                events.append((note[1], 0, p, 0))
    events.sort()

    # 2. walk through time
    tokens: List[int] = [BOS] if add_bos else []
    cur_step, cur_vel_bin = 0, -1
    for step, order, pitch, vel in events:
        gap = step - cur_step
        while gap > 0:
            chunk = min(gap, MAX_SHIFT)
            tokens.append(shift_token(chunk))
            gap -= chunk
        cur_step = step
        if order == 0:
            tokens.append(off_token(pitch))
        else:
            b = vel_bin(vel)
            if b != cur_vel_bin:
                tokens.append(VEL_BASE + b)
                cur_vel_bin = b
            tokens.append(on_token(pitch))
    if add_eos:
        tokens.append(EOS)
    return tokens


# --------------------------------------------------------------------------- #
# tokens -> notes
# --------------------------------------------------------------------------- #
def tokens_to_notes(tokens: Sequence[int], max_hold: int = MAX_HOLD_STEPS,
                    total_steps: int | None = None) -> List[Note]:
    """Decode tokens back to notes.  Unmatched OFFs are ignored; notes still
    sounding at the end (or held longer than ``max_hold``) are closed."""
    notes: List[Note] = []
    sounding: Dict[int, Tuple[int, int]] = {}
    step, vel = 0, bin_velocity(4)
    for tok in tokens:
        kind = token_kind(int(tok))
        if kind == "shift":
            step += int(tok) - SHIFT_BASE + 1
        elif kind == "vel":
            vel = bin_velocity(int(tok) - VEL_BASE)
        elif kind == "on":
            p = int(tok) - ON_BASE + PITCH_MIN
            if p not in sounding:
                sounding[p] = (step, vel)
        elif kind == "off":
            p = int(tok) - OFF_BASE + PITCH_MIN
            if p in sounding:
                s, v = sounding.pop(p)
                if step > s:
                    notes.append((s, min(step, s + max_hold), p, v))
        elif tok == EOS:
            break
    end = step if total_steps is None else total_steps
    for p, (s, v) in sounding.items():
        e = min(end, s + max_hold)
        if e > s:
            notes.append((s, e, p, v))
    notes.sort()
    return notes


def transpose_tokens(tokens: np.ndarray, semitones: int) -> np.ndarray:
    """Shift every NOTE_ON/NOTE_OFF token by ``semitones`` (caller ensures range)."""
    out = tokens.copy()
    mask = (out >= ON_BASE) & (out < SHIFT_BASE)
    out[mask] += semitones
    return out


def feasible_transpositions(tokens: np.ndarray, max_shift: int = 3) -> List[int]:
    """Which shifts in [-max_shift, max_shift] keep all pitches on the keyboard?"""
    on = (tokens >= ON_BASE) & (tokens < OFF_BASE)
    off = (tokens >= OFF_BASE) & (tokens < SHIFT_BASE)
    pidx = np.where(on, tokens - ON_BASE, np.where(off, tokens - OFF_BASE, -1))
    pidx = pidx[pidx >= 0]
    if pidx.size == 0:
        return list(range(-max_shift, max_shift + 1))
    lo, hi = int(pidx.min()), int(pidx.max())
    return [s for s in range(-max_shift, max_shift + 1) if lo + s >= 0 and hi + s < N_PITCH]

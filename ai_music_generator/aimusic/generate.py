"""Music generation with the trained LSTM.

Pipeline
    1. condition the network on  genre + mood
    2. "prime" it with the user's starting notes (if any)
    3. repeatedly:  ask the LSTM for a probability distribution over the next
       token -> remove tokens that would be musically invalid -> sample one
    4. stop when the requested duration has been reached
    5. decode tokens -> notes

Tempo and duration
    The network works on a beat grid (12 steps per beat).  ``duration x bpm / 60``
    gives the number of beats to generate, so a longer duration or a faster tempo
    really produces more notes, and the BPM is written into the MIDI file / audio.

Validity masks ("constrained decoding")
    The LSTM proposes, the mask disposes: it forbids e.g. releasing a note that is
    not sounding, starting a note that is already sounding, more than 8 voices at
    once, notes held longer than 16 beats and silences longer than 3 beats.  The
    *musical content* still comes entirely from the network's probabilities.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .analysis import estimate_key, note_name
from .config import MAX_SEED_NOTES, PITCH_MAX, PITCH_MIN, STEPS_PER_BEAT
from .lstm import LSTMModel
from .midi_io import Note
from .tokenizer import (BOS, EOS, MAX_HOLD_STEPS, MAX_SHIFT, N_PITCH, N_VEL, OFF_BASE,
                        ON_BASE, PAD, SHIFT_BASE, VEL_BASE, VOCAB_SIZE, bin_velocity,
                        off_token, on_token, shift_token, token_kind, vel_token)

MAX_POLYPHONY = 8
MAX_SILENCE_STEPS = 3 * STEPS_PER_BEAT


class GenerationError(RuntimeError):
    pass


# --------------------------------------------------------------------------- #
# starting-note parsing
# --------------------------------------------------------------------------- #
_NOTE_RE = re.compile(r"^([A-Ga-g])([#b]?)(-?\d)$")
_SEMITONE = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def note_to_midi(name: str) -> int:
    """'C4' -> 60, 'F#3' -> 54, 'Bb2' -> 46.  Raises ValueError if invalid."""
    m = _NOTE_RE.match(name.strip())
    if not m:
        raise ValueError(f"'{name}' is not a valid note name")
    letter, acc, octave = m.group(1).upper(), m.group(2), int(m.group(3))
    pitch = 12 * (octave + 1) + _SEMITONE[letter] + (1 if acc == "#" else -1 if acc == "b" else 0)
    if not PITCH_MIN <= pitch <= PITCH_MAX:
        raise ValueError(f"'{name}' is outside the piano range A0-C8")
    return pitch


def parse_start_notes(text: str) -> List[List[int]]:
    """Parse "C4, E4, G4" (a melody) or "C4+E4+G4, F4" (chord, then note).

    Returns a list of groups; each group is a list of MIDI pitches sounded together.
    Raises ValueError with a helpful message on bad input.
    """
    text = (text or "").strip()
    if not text:
        return []
    items = [t for t in re.split(r"[,\s;]+", text) if t]
    if len(items) > MAX_SEED_NOTES:
        raise ValueError(f"Too many starting notes ({len(items)}); the maximum is {MAX_SEED_NOTES}.")
    groups = []
    for item in items:
        pitches = []
        for part in item.split("+"):
            pitches.append(note_to_midi(part))
        groups.append(pitches)
    return groups


# --------------------------------------------------------------------------- #
# free-text instructions (keyword based - NOT understood by the LSTM)
# --------------------------------------------------------------------------- #
TEXT_KEYWORDS = {
    "articulation": {
        "staccato": r"\b(staccato|detached|short notes|punchy)\b",
        "legato": r"\b(legato|smooth|flowing|sustained)\b",
    },
    "dynamics": {
        "soft": r"\b(soft|softly|quiet|gentle|delicate)\b",
        "loud": r"\b(loud|powerful|strong|bold)\b",
    },
    "creativity": {
        "experimental": r"\b(experimental|surprising|adventurous|wild|unpredictable|creative)\b",
        "simple": r"\b(simple|predictable|safe|steady|repetitive|minimal)\b",
    },
}


@dataclass
class TextEffects:
    temperature_delta: float = 0.0
    articulation: Optional[str] = None
    velocity_scale: float = 1.0
    applied: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)   # explanations shown to the user


def parse_text_instructions(text: str) -> TextEffects:
    """Map a few keywords to *post-processing / sampling* effects.

    An LSTM trained on MIDI cannot read sentences.  Only the keywords listed in
    ``TEXT_KEYWORDS`` have an effect; the result tells the user exactly what was used.
    """
    fx = TextEffects()
    text = (text or "").strip()
    if not text:
        return fx
    lower = text.lower()
    for category, options in TEXT_KEYWORDS.items():
        hits = [name for name, rx in options.items() if re.search(rx, lower)]
        if len(hits) == 2:
            fx.notes.append(f"Conflicting {category} keywords ({hits[0]} and {hits[1]}) - neither was applied.")
        elif len(hits) == 1:
            hit = hits[0]
            if category == "articulation":
                fx.articulation = hit
                fx.applied.append(f"{hit} note lengths")
            elif category == "dynamics":
                fx.velocity_scale = 0.75 if hit == "soft" else 1.2
                fx.applied.append(f"{hit} dynamics")
            else:
                fx.temperature_delta = 0.15 if hit == "experimental" else -0.15
                fx.applied.append("more adventurous sampling" if hit == "experimental"
                                  else "more conservative sampling")
    if fx.applied:
        fx.notes.append("Only the recognised keywords above were used; the rest of the text was ignored.")
    elif not fx.notes:
        fx.notes.append("None of the words matched a supported keyword, so the text had no effect. "
                        "The LSTM cannot read sentences - use the genre, mood and instrument "
                        "selectors for those, or the keywords listed under the text box.")
    return fx


def apply_articulation(notes: List[Note], mode: Optional[str], total_steps: int,
                       velocity_scale: float = 1.0) -> List[Note]:
    """Post-process note lengths/velocities and clip everything to the duration."""
    out: List[Note] = []
    by_pitch: Dict[int, List[Note]] = {}
    for n in sorted(notes):
        by_pitch.setdefault(n[2], []).append(n)
    next_start: Dict[Tuple[int, int], int] = {}
    for p, lst in by_pitch.items():
        for i, n in enumerate(lst):
            next_start[(n[0], p)] = lst[i + 1][0] if i + 1 < len(lst) else total_steps
    for s, e, p, v in sorted(notes):
        length = e - s
        if mode == "staccato":
            length = max(1, int(length * 0.5))
        elif mode == "legato":
            length = int(length * 1.3) + 1
        e2 = min(s + length, next_start[(s, p)], total_steps)
        if s >= total_steps or e2 <= s:
            continue
        out.append((s, e2, p, int(max(1, min(127, round(v * velocity_scale))))))
    return out


# --------------------------------------------------------------------------- #
# generation state
# --------------------------------------------------------------------------- #
class _Tracker:
    """Keeps track of what is currently sounding so invalid tokens can be masked."""

    def __init__(self):
        self.step = 0
        self.sounding: Dict[int, Tuple[int, int]] = {}     # pitch -> (start, velocity)
        self.vel = bin_velocity(5)
        self.silence = 0
        self.last_was_vel = False
        self.notes: List[Note] = []
        self.max_poly = 0

    def forced_off(self) -> Optional[int]:
        for p, (s, _v) in self.sounding.items():
            if self.step - s >= MAX_HOLD_STEPS:
                return off_token(p)
        return None

    def mask(self) -> np.ndarray:
        m = np.zeros(VOCAB_SIZE, dtype=bool)
        # note-on: pitch must be free and polyphony must not be exceeded
        if len(self.sounding) < MAX_POLYPHONY:
            on = np.ones(N_PITCH, dtype=bool)
            for p in self.sounding:
                on[p - PITCH_MIN] = False
            m[ON_BASE:ON_BASE + N_PITCH] = on
        # note-off: only for sounding notes that have a non-zero length
        for p, (s, _v) in self.sounding.items():
            if s < self.step:
                m[off_token(p)] = True
        # time shift
        if self.sounding or self.silence < MAX_SILENCE_STEPS:
            m[SHIFT_BASE:SHIFT_BASE + MAX_SHIFT] = True
        # velocity
        if not self.last_was_vel:
            m[VEL_BASE:VEL_BASE + N_VEL] = True
        if not m.any():            # cannot happen, but never leave an empty mask
            m[SHIFT_BASE] = True
        return m

    def apply(self, tok: int) -> None:
        kind = token_kind(tok)
        self.last_was_vel = kind == "vel"
        if kind == "shift":
            n = tok - SHIFT_BASE + 1
            self.step += n
            if not self.sounding:
                self.silence += n
        elif kind == "vel":
            self.vel = bin_velocity(tok - VEL_BASE)
        elif kind == "on":
            p = tok - ON_BASE + PITCH_MIN
            self.sounding[p] = (self.step, self.vel)
            self.silence = 0
            self.max_poly = max(self.max_poly, len(self.sounding))
        elif kind == "off":
            p = tok - OFF_BASE + PITCH_MIN
            if p in self.sounding:
                s, v = self.sounding.pop(p)
                if self.step > s:
                    self.notes.append((s, self.step, p, v))

    def finish(self, total_steps: int) -> List[Note]:
        for p, (s, v) in self.sounding.items():
            e = min(total_steps, s + MAX_HOLD_STEPS)
            if e > s:
                self.notes.append((s, e, p, v))
        self.sounding.clear()
        notes = [(s, min(e, total_steps), p, v) for s, e, p, v in self.notes if s < total_steps]
        notes.sort()
        return notes


def sample_token(logits: np.ndarray, allowed: np.ndarray, temperature: float,
                 top_k: int, top_p: float, rng: np.random.Generator) -> int:
    z = np.where(allowed, logits, -np.inf) / max(temperature, 1e-3)
    z -= z.max()
    p = np.exp(z)
    p /= p.sum()
    if top_k and top_k < np.count_nonzero(allowed):
        thresh = np.partition(p, -top_k)[-top_k]
        p = np.where(p >= thresh, p, 0.0)
    order = np.argsort(-p)
    cum = np.cumsum(p[order])
    keep = (cum - p[order]) < top_p * cum[-1]
    p_sorted = np.where(keep, p[order], 0.0)
    p_final = np.zeros_like(p)
    p_final[order] = p_sorted
    p_final /= p_final.sum()
    return int(rng.choice(len(p_final), p=p_final))


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #
@dataclass
class GeneratedPiece:
    notes: List[Note]                  # (start_step, end_step, pitch, velocity)
    bpm: float
    total_steps: int
    genre: str
    mood: str
    seed: int
    temperature: float
    tokens_generated: int
    seed_notes: List[str]
    max_polyphony: int
    key: str
    effects: TextEffects
    seconds_to_generate: float

    @property
    def duration_seconds(self) -> float:
        return steps_to_seconds(self.total_steps, self.bpm)


def steps_to_seconds(steps: float, bpm: float) -> float:
    return steps / STEPS_PER_BEAT * 60.0 / bpm


def generate_piece(model: LSTMModel, meta: dict, *, genre: str, mood: str, bpm: float,
                   duration: float, start_notes: str = "", temperature: float = 0.9,
                   seed: Optional[int] = None, text: str = "",
                   progress: Optional[Callable[[float], None]] = None) -> GeneratedPiece:
    """Generate ``duration`` seconds of music at ``bpm`` with the trained LSTM."""
    if genre not in meta["genres"]:
        raise GenerationError(f"Genre '{genre}' was not part of the training data. "
                              f"Available: {', '.join(meta['genres'])}")
    if mood not in meta["moods"]:
        raise GenerationError(f"Mood '{mood}' is not available. Available: {', '.join(meta['moods'])}")
    g_idx, m_idx = meta["genres"].index(genre), meta["moods"].index(mood)
    groups = parse_start_notes(start_notes)
    fx = parse_text_instructions(text)
    temperature = float(min(1.5, max(0.4, temperature + fx.temperature_delta)))
    total_steps = int(round(duration * bpm / 60.0 * STEPS_PER_BEAT))
    if total_steps < 2 * STEPS_PER_BEAT:
        raise GenerationError("The requested duration is too short to generate music.")
    if len(groups) * STEPS_PER_BEAT > total_steps // 2:
        raise GenerationError(
            f"The starting pattern is {len(groups)} beats long, which is more than half of the "
            f"requested piece ({total_steps / STEPS_PER_BEAT:.0f} beats). Use fewer notes, a "
            f"longer duration or a faster tempo.")
    if seed is None:
        seed = int(np.random.SeedSequence().entropy % (2 ** 31))
    rng = np.random.default_rng(seed)
    t0 = time.time()

    tracker = _Tracker()
    state = model.initial_state()
    logits, state = model.step(BOS, g_idx, m_idx, state)
    n_tokens = 0

    def feed(tok: int):
        nonlocal logits, state, n_tokens
        tracker.apply(tok)
        logits, state = model.step(tok, g_idx, m_idx, state)
        n_tokens += 1

    # prime with the user's notes: each group sounds for exactly one beat
    for group in groups:
        feed(vel_token(92))
        for p in group:
            feed(on_token(p))
        feed(shift_token(STEPS_PER_BEAT))
        for p in group:
            feed(off_token(p))

    max_tokens = 30 * total_steps + 5000
    last_report = 0
    while tracker.step < total_steps:
        forced = tracker.forced_off()
        if forced is not None:
            tok = forced
        else:
            tok = sample_token(logits, tracker.mask(), temperature, top_k=48, top_p=0.95, rng=rng)
        tracker.apply(tok)
        if tracker.step >= total_steps:
            n_tokens += 1
            break
        logits, state = model.step(tok, g_idx, m_idx, state)
        n_tokens += 1
        if n_tokens > max_tokens:
            raise GenerationError("The model produced degenerate output; try a different seed "
                                  "or a lower creativity value.")
        if progress and n_tokens - last_report >= 100:
            last_report = n_tokens
            progress(min(1.0, tracker.step / total_steps))

    notes = tracker.finish(total_steps)
    if not notes:
        raise GenerationError("The model did not produce any notes. Try another seed.")
    notes = apply_articulation(notes, fx.articulation, total_steps, fx.velocity_scale)
    if not notes:
        raise GenerationError("No notes remained after post-processing.")
    tonic, mode, _ = estimate_key(notes)
    if progress:
        progress(1.0)
    return GeneratedPiece(
        notes=notes, bpm=float(bpm), total_steps=total_steps, genre=genre, mood=mood,
        seed=seed, temperature=temperature, tokens_generated=n_tokens,
        seed_notes=[note_name(p) for g in groups for p in g],
        max_polyphony=tracker.max_poly, key=f"{tonic} {mode}", effects=fx,
        seconds_to_generate=time.time() - t0)

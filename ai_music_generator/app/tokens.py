"""Token vocabulary: 0=REST, 1=HOLD (sustain previous note), 2..89 = MIDI pitch 21..108.
One token = one 16th-note step (0.25 beat), so tempo/duration are applied at render time."""
import re
import numpy as np
from .midi_io import read_notes
REST, HOLD, OFFSET, VOCAB = 0, 1, 2, 90
LO, HI = 21, 108
STEP = 0.25
_N = {"C":0,"D":2,"E":4,"F":5,"G":7,"A":9,"B":11}

def pitch_to_tok(p): return p - LO + OFFSET
def tok_to_pitch(t): return t - OFFSET + LO

def parse_note_name(s):
    m = re.fullmatch(r"([A-Ga-g])([#b]?)(-?\d)", s.strip())
    if not m: raise ValueError(f"Invalid note '{s}'. Use names like C4, F#3, Bb4.")
    p = 12*(int(m[3])+1) + _N[m[1].upper()] + {"#":1,"b":-1,"":0}[m[2]]
    if not LO <= p <= HI: raise ValueError(f"Note '{s}' is outside the piano range A0-C8.")
    return p

def parse_start_notes(text):
    parts = [x for x in re.split(r"[,\s]+", text or "") if x]
    return [parse_note_name(x) for x in parts]

def notes_to_tokens(notes):
    """Monophonic melody (highest note per onset) on a 16th grid."""
    if not notes: return np.zeros(0, dtype=np.int16)
    n = int(round(max(e for _, e, _ in notes)/STEP)) + 1
    onset = {}
    for s, e, p in notes:
        if LO <= p <= HI:
            k = int(round(s/STEP))
            if k not in onset or p > onset[k][0]:
                onset[k] = (p, max(k+1, int(round(e/STEP))))
    toks, cur_end = np.zeros(n, dtype=np.int16), 0
    for t in range(n):
        if t in onset:
            toks[t] = pitch_to_tok(onset[t][0]); cur_end = onset[t][1]
        elif t < cur_end: toks[t] = HOLD
    return toks

def tokens_to_notes(toks, vel=80, gate=1.0):
    """-> list of (start_beat, dur_beats, pitch, velocity)."""
    out, cur = [], None
    for t, tok in enumerate(list(toks) + [REST]):
        if tok == HOLD and cur: cur[1] += 1; continue
        if cur: out.append((cur[0]*STEP, cur[1]*STEP*gate, cur[2], vel)); cur = None
        if tok >= OFFSET: cur = [t, 1, tok_to_pitch(tok)]
    return out

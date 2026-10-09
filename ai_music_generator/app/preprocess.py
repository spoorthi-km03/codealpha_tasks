"""Turn data/midi/<genre>/*.mid into windows of token sequences."""
import random
from pathlib import Path
import numpy as np
from .midi_io import read_notes
from .tokens import notes_to_tokens
ROOT = Path(__file__).resolve().parent.parent

def load_dataset(data_dir=None, max_files=150, seq_len=64, stride=16, max_windows=80000, log=print):
    data_dir = Path(data_dir or ROOT / "data" / "midi")
    genres = sorted(d.name for d in data_dir.iterdir() if d.is_dir() and any(d.glob("*.mid*")))
    if not genres: raise FileNotFoundError(f"No MIDI files in {data_dir}. Run: python -m app.prepare_data")
    X, G = [], []
    for gi, g in enumerate(genres):
        files = sorted(p for p in (data_dir/g).iterdir() if p.suffix.lower() in (".mid", ".midi"))
        random.Random(0).shuffle(files); used = 0
        for f in files[:max_files]:
            try: toks = notes_to_tokens(read_notes(f))
            except Exception as e: log(f"  skipped {f.name}: {e}"); continue
            used += 1
            for s in range(0, len(toks) - seq_len - 1, stride):
                X.append(toks[s:s+seq_len+1]); G.append(gi)
        log(f"genre '{g}': {used} files")
    if not X: raise ValueError("MIDI files found but no usable notes extracted.")
    X, G = np.array(X, dtype=np.int64), np.array(G, dtype=np.int64)
    if len(X) > max_windows:
        idx = np.random.default_rng(0).choice(len(X), max_windows, replace=False); X, G = X[idx], G[idx]
    log(f"{len(X)} training windows"); return X, G, genres

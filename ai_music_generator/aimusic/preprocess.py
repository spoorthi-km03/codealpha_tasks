"""Preprocessing: raw MIDI files  ->  integer token sequences + labels.

Pipeline for every file in ``data/raw/<genre>/``:

  1. parse the MIDI file (notes, sustain pedal, tempo map)
  2. convert ticks to the model's time grid (12 steps per beat)
  3. estimate key / mode and note density  -> automatic mood label
  4. encode the notes as event tokens (see tokenizer.py)

Output
  data/processed/dataset.npz       tokens (int16), offsets, genre ids, mood ids
  data/processed/dataset_meta.json human-readable summary
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np

from .analysis import assign_moods, energy_of, estimate_key
from .config import (DATASET_META_PATH, DATASET_PATH, MOODS, RAW_DIR, STEPS_PER_BEAT)
from .midi_io import MidiParseError, read_midi
from .tokenizer import VOCAB_SIZE, notes_to_tokens

MIN_NOTES = 40            # ignore trivial files
MIN_TOKENS = 200
MIN_FILES_PER_GENRE = 4   # need some files for training + validation


class PreprocessError(RuntimeError):
    pass


def quantise(notes, ticks_per_beat: int):
    """Ticks -> grid steps.  Every note is at least one step long."""
    k = STEPS_PER_BEAT / ticks_per_beat
    out = []
    for s, e, p, v in notes:
        qs = int(round(s * k))
        qe = max(int(round(e * k)), qs + 1)
        out.append((qs, qe, p, v))
    return out


def process_file(path: Path):
    """Return dict with tokens, mode, key, energy for one MIDI file (or raise)."""
    midi = read_midi(path)
    if len(midi.notes) < MIN_NOTES:
        raise MidiParseError(f"only {len(midi.notes)} notes")
    notes = quantise(midi.notes, midi.ticks_per_beat)
    tokens = notes_to_tokens(notes)
    if len(tokens) < MIN_TOKENS:
        raise MidiParseError("too short")
    tonic, mode, _ = estimate_key(notes)
    return {
        "tokens": np.asarray(tokens, dtype=np.int16),
        "key": f"{tonic} {mode}", "mode": mode,
        "energy": energy_of(len(midi.notes), midi.duration_seconds),
        "seconds": midi.duration_seconds,
    }


def run(raw_dir: Path = RAW_DIR, out_path: Path = DATASET_PATH, meta_path: Path = DATASET_META_PATH,
        max_files_per_genre: int | None = None, log: Callable[[str], None] = print) -> dict:
    raw_dir = Path(raw_dir)
    genre_dirs = sorted(d for d in raw_dir.iterdir() if d.is_dir()) if raw_dir.exists() else []
    files_by_genre: Dict[str, List[Path]] = {}
    for d in genre_dirs:
        files = sorted(f for f in d.rglob("*") if f.suffix.lower() in (".mid", ".midi"))
        if max_files_per_genre:
            files = files[:max_files_per_genre]
        if files:
            files_by_genre[d.name.lower()] = files
    if not files_by_genre:
        raise PreprocessError(
            f"No MIDI files found in {raw_dir}.\n"
            "Run  python scripts/01_download_dataset.py  first, or copy .mid files into "
            "data/raw/<genre>/ (e.g. data/raw/jazz/).")

    t0 = time.time()
    records: List[dict] = []
    skipped: List[str] = []
    for genre, files in files_by_genre.items():
        log(f"[{genre}] processing {len(files)} files")
        for i, f in enumerate(files, 1):
            try:
                rec = process_file(f)
            except (MidiParseError, OSError, ValueError, OverflowError) as exc:
                skipped.append(f"{f.name}: {exc}")
                continue
            rec.update(genre=genre, name=f.name)
            records.append(rec)
            if i % 25 == 0 or i == len(files):
                log(f"   {i}/{len(files)} files  ({time.time() - t0:.0f}s)")

    # drop genres with too few usable files
    counts: Dict[str, int] = {}
    for r in records:
        counts[r["genre"]] = counts.get(r["genre"], 0) + 1
    thin = [g for g, n in counts.items() if n < MIN_FILES_PER_GENRE]
    for g in thin:
        log(f"WARNING: genre '{g}' has only {counts[g]} usable files (< {MIN_FILES_PER_GENRE}) - dropped")
    records = [r for r in records if r["genre"] not in thin]
    if not records:
        raise PreprocessError("No usable MIDI files after preprocessing. "
                              f"{len(skipped)} files were skipped (e.g. {skipped[:3]}).")

    # automatic mood labels
    moods = assign_moods([r["mode"] for r in records], [r["energy"] for r in records],
                         [r["genre"] for r in records])
    for r, m in zip(records, moods):
        r["mood"] = m
    genres = sorted({r["genre"] for r in records})
    mood_list = [m for m in MOODS if m in set(moods)]

    tokens = np.concatenate([r["tokens"] for r in records]).astype(np.int16)
    lengths = np.array([len(r["tokens"]) for r in records], dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    genre_idx = np.array([genres.index(r["genre"]) for r in records], dtype=np.int16)
    mood_idx = np.array([mood_list.index(r["mood"]) for r in records], dtype=np.int16)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, tokens=tokens, offsets=offsets, genre_idx=genre_idx, mood_idx=mood_idx)

    combos: Dict[str, Dict[str, int]] = {g: {m: 0 for m in mood_list} for g in genres}
    for r in records:
        combos[r["genre"]][r["mood"]] += 1
    meta = {
        "vocab_size": VOCAB_SIZE,
        "steps_per_beat": STEPS_PER_BEAT,
        "genres": genres,
        "moods": mood_list,
        "n_files": len(records),
        "n_tokens": int(tokens.size),
        "files_per_genre": {g: int(sum(combos[g].values())) for g in genres},
        "combo_counts": combos,
        "skipped_files": len(skipped),
        "skipped_examples": skipped[:10],
        "seconds_total": round(float(sum(r["seconds"] for r in records)), 1),
        "preprocess_seconds": round(time.time() - t0, 1),
        "mood_method": "heuristic: mode (major/minor) + note density ranked per genre",
    }
    Path(meta_path).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log(f"\nDone: {len(records)} files, {tokens.size:,} tokens, genres={genres}, moods={mood_list}")
    if skipped:
        log(f"({len(skipped)} files skipped as unusable)")
    return meta

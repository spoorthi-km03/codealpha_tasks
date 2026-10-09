"""Service layer used by the Flask app: model loading, validation, jobs, exports."""
from __future__ import annotations

import json
import re
import shutil
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .config import (BPM_MAX, BPM_MIN, DURATION_MAX, DURATION_MIN, MAX_SEED_NOTES,
                     MODEL_PATH, OUTPUT_DIR, STEPS_PER_BEAT)
from .generate import (GenerationError, TEXT_KEYWORDS, generate_piece, parse_start_notes,
                       steps_to_seconds)
from .lstm import LSTMModel
from .midi_io import write_midi
from .synth import INSTRUMENTS, render_audio, write_wav

TRACK_ID_RE = re.compile(r"^[0-9a-f]{12}$")
LOW_SUPPORT_FILES = 5          # warn if fewer training files carry a genre+mood combination
LOW_TRAINING_STEPS = 1500      # warn if the model was trained only briefly

MOOD_HINTS = {
    "relaxing": "major key, low note density",
    "happy": "major key, medium-high density",
    "sad": "minor key, low-medium density",
    "energetic": "fastest 25% of the genre's pieces",
}

NO_MODEL_HELP = (
    "No trained model was found. The Studio needs a model trained on MIDI files first.\n"
    "Easiest: close this window and double-click run.bat - it offers to download the data and "
    "train for you (about 30 minutes the first time).\n"
    "Manual: python scripts/01_download_dataset.py  ->  python scripts/02_preprocess.py  ->  "
    "python scripts/03_train.py   then restart the app.")


class ModelManager:
    """Loads the trained model lazily and reloads it if the file changes on disk."""

    def __init__(self, model_path: Path = MODEL_PATH):
        self.model_path = Path(model_path)
        self._model: Optional[LSTMModel] = None
        self._meta: Optional[dict] = None
        self._mtime = None
        self._error: Optional[str] = None
        self._lock = threading.Lock()

    def get(self) -> Tuple[Optional[LSTMModel], Optional[dict], Optional[str]]:
        with self._lock:
            meta_path = self.model_path.with_suffix(".json")
            if not self.model_path.exists() or not meta_path.exists():
                self._model = self._meta = None
                self._error = NO_MODEL_HELP
                return None, None, self._error
            mtime = (self.model_path.stat().st_mtime, meta_path.stat().st_mtime)
            if mtime != self._mtime or self._model is None:
                try:
                    self._model, self._meta = LSTMModel.load(self.model_path)
                    self._error = None
                except Exception as exc:  # corrupt / incompatible file
                    self._model = self._meta = None
                    self._error = (f"The model file could not be loaded ({exc}). "
                                   "Delete the files in the models folder and train again.")
                self._mtime = mtime
            return self._model, self._meta, self._error


# --------------------------------------------------------------------------- #
# validation
# --------------------------------------------------------------------------- #
def _num(value, name, lo, hi, errors, integer=False, default=None):
    if value in (None, ""):
        if default is not None:
            return default
        errors[name] = "This field is required."
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        errors[name] = "Enter a number."
        return None
    if v != v or v in (float("inf"), float("-inf")):
        errors[name] = "Enter a valid number."
        return None
    if not lo <= v <= hi:
        errors[name] = f"Must be between {lo:g} and {hi:g}."
        return None
    return int(round(v)) if integer else v


def validate_request(data: Any, meta: dict) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """Return (clean parameters, field errors)."""
    errors: Dict[str, str] = {}
    if not isinstance(data, dict):
        return {}, {"_": "Request body must be a JSON object."}
    clean: Dict[str, Any] = {}

    genre = str(data.get("genre", "")).lower().strip()
    if genre not in meta["genres"]:
        errors["genre"] = f"Choose one of: {', '.join(meta['genres'])}."
    clean["genre"] = genre
    mood = str(data.get("mood", "")).lower().strip()
    if mood not in meta["moods"]:
        errors["mood"] = f"Choose one of: {', '.join(meta['moods'])}."
    clean["mood"] = mood
    inst = str(data.get("instrument", "")).lower().strip()
    if inst not in INSTRUMENTS:
        errors["instrument"] = f"Choose one of: {', '.join(INSTRUMENTS)}."
    clean["instrument"] = inst

    clean["bpm"] = _num(data.get("bpm"), "bpm", BPM_MIN, BPM_MAX, errors, integer=True)
    clean["duration"] = _num(data.get("duration"), "duration", DURATION_MIN, DURATION_MAX, errors, integer=True)
    clean["creativity"] = _num(data.get("creativity"), "creativity", 0.4, 1.4, errors, default=0.9)
    seed = _num(data.get("seed"), "seed", 0, 2 ** 31 - 1, errors, integer=True, default=-1)
    clean["seed"] = None if seed == -1 else seed

    start = str(data.get("start_notes", "") or "").strip()
    if len(start) > 200:
        errors["start_notes"] = "Too long (maximum 200 characters)."
    else:
        try:
            groups = parse_start_notes(start)
            clean["_seed_beats"] = len(groups)
        except ValueError as exc:
            errors["start_notes"] = (f"{exc} Example: C4, E4, G4  (use + for a chord: C4+E4+G4).")
    clean["start_notes"] = start

    text = str(data.get("text", "") or "").strip()
    if len(text) > 300:
        errors["text"] = "Too long (maximum 300 characters)."
    clean["text"] = text

    if not errors and clean.get("_seed_beats"):
        beats = clean["bpm"] * clean["duration"] / 60.0
        if clean["_seed_beats"] > beats / 2:
            errors["start_notes"] = (
                f"Your pattern needs {clean['_seed_beats']} beats but the piece only has "
                f"{beats:.0f}. Use fewer notes, a longer duration or a faster tempo.")
    clean.pop("_seed_beats", None)
    return clean, errors


def support_warnings(meta: dict, genre: str, mood: str) -> List[str]:
    out = []
    n = meta.get("combo_counts", {}).get(genre, {}).get(mood, 0)
    if n < LOW_SUPPORT_FILES:
        out.append(f"Only {n} training file(s) were labelled {genre} + {mood}; the model has seen "
                   f"little of this combination, so it may sound less typical.")
    steps = meta.get("training", {}).get("steps", 0)
    if steps < LOW_TRAINING_STEPS:
        out.append(f"This model was trained for only {steps} steps; train longer "
                   f"(scripts/03_train.py --preset standard) for better music.")
    return out


# --------------------------------------------------------------------------- #
# producing a track
# --------------------------------------------------------------------------- #
def produce_track(model: LSTMModel, meta: dict, params: Dict[str, Any], out_root: Path,
                  progress=None) -> Dict[str, Any]:
    """Generate -> MIDI -> audio -> JSON description.  Raises on any failure."""
    def report(stage: str, frac: float, lo: float, hi: float):
        if progress:
            progress(stage, lo + (hi - lo) * frac)

    piece = generate_piece(
        model, meta, genre=params["genre"], mood=params["mood"], bpm=params["bpm"],
        duration=params["duration"], start_notes=params["start_notes"],
        temperature=params["creativity"], seed=params["seed"], text=params["text"],
        progress=lambda f: report("Generating notes with the LSTM", f, 0.0, 0.55))

    track_id = uuid.uuid4().hex[:12]
    out_dir = Path(out_root) / track_id
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        label, program = INSTRUMENTS[params["instrument"]]
        report("Writing MIDI file", 0.0, 0.55, 0.58)
        write_midi(out_dir / "track.mid", piece.notes, piece.bpm, STEPS_PER_BEAT, program,
                   title=f"AI Music Studio - {piece.genre} {piece.mood}")
        notes_sec = [(steps_to_seconds(s, piece.bpm), steps_to_seconds(e, piece.bpm), p, v)
                     for s, e, p, v in piece.notes]
        t0 = time.time()
        audio = render_audio(notes_sec, piece.duration_seconds, params["instrument"],
                             seed=piece.seed % 100000,
                             progress=lambda f: report("Rendering audio", f, 0.58, 0.98))
        write_wav(out_dir / "track.wav", audio)
        render_seconds = time.time() - t0

        warnings = support_warnings(meta, piece.genre, piece.mood)
        result = {
            "id": track_id,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "params": {
                "genre": piece.genre, "mood": piece.mood, "instrument": params["instrument"],
                "instrument_label": label, "bpm": params["bpm"], "duration": params["duration"],
                "start_notes": params["start_notes"], "text": params["text"],
                "creativity": params["creativity"], "seed": piece.seed,
            },
            "stats": {
                "notes": len(piece.notes), "tokens": piece.tokens_generated,
                "max_polyphony": piece.max_polyphony, "key": piece.key,
                "audio_seconds": round(len(audio) / 44100, 2),
                "generation_seconds": round(piece.seconds_to_generate, 2),
                "render_seconds": round(render_seconds, 2),
                "beats": round(piece.total_steps / STEPS_PER_BEAT, 1),
                "effective_creativity": round(piece.temperature, 2),
                "midi_program": program,
            },
            "start_pattern_used": piece.seed_notes,
            "text_effects": {"applied": piece.effects.applied, "notes": piece.effects.notes},
            "warnings": warnings,
            "notes": [[round(s, 3), round(e, 3), p, v] for s, e, p, v in notes_sec],
            "files": {"midi": f"/api/tracks/{track_id}/midi", "wav": f"/api/tracks/{track_id}/wav"},
        }
        (out_dir / "track.json").write_text(json.dumps(result), encoding="utf-8")
        return result
    except Exception:
        shutil.rmtree(out_dir, ignore_errors=True)
        raise


def prune_outputs(out_root: Path, keep: int = 40) -> None:
    dirs = [d for d in Path(out_root).glob("*") if d.is_dir() and TRACK_ID_RE.match(d.name)]
    dirs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
    for d in dirs[keep:]:
        shutil.rmtree(d, ignore_errors=True)


# --------------------------------------------------------------------------- #
# background jobs
# --------------------------------------------------------------------------- #
@dataclass
class Job:
    id: str
    params: Dict[str, Any]
    status: str = "queued"           # queued | running | done | error
    stage: str = "Waiting to start"
    progress: float = 0.0
    error: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    created: float = field(default_factory=time.time)


class JobManager:
    """Runs one generation at a time (CPU-bound) and lets the UI poll progress."""

    def __init__(self, models: ModelManager, out_root: Path, max_jobs: int = 60):
        self.models = models
        self.out_root = Path(out_root)
        self.out_root.mkdir(parents=True, exist_ok=True)
        self.jobs: Dict[str, Job] = {}
        self._lock = threading.Lock()
        self._run_lock = threading.Lock()
        self.max_jobs = max_jobs

    def submit(self, params: Dict[str, Any]) -> Job:
        job = Job(id=uuid.uuid4().hex[:12], params=params)
        with self._lock:
            self.jobs[job.id] = job
            for old in sorted(self.jobs.values(), key=lambda j: j.created)[:-self.max_jobs]:
                self.jobs.pop(old.id, None)
        threading.Thread(target=self._run, args=(job,), daemon=True).start()
        return job

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self.jobs.get(job_id)

    def _run(self, job: Job) -> None:
        with self._run_lock:
            job.status, job.stage = "running", "Loading model"
            try:
                model, meta, err = self.models.get()
                if model is None:
                    raise GenerationError(err or NO_MODEL_HELP)

                def progress(stage: str, frac: float):
                    job.stage, job.progress = stage, float(min(max(frac, 0.0), 1.0))

                job.result = produce_track(model, meta, job.params, self.out_root, progress)
                job.progress, job.stage, job.status = 1.0, "Done", "done"
                prune_outputs(self.out_root)
            except (GenerationError, ValueError) as exc:
                job.status, job.error = "error", str(exc)
            except MemoryError:
                job.status, job.error = "error", "Not enough memory to render this track; try a shorter duration."
            except Exception as exc:  # unexpected - log the traceback to the console
                traceback.print_exc()
                job.status, job.error = "error", f"Unexpected error while generating: {exc}"

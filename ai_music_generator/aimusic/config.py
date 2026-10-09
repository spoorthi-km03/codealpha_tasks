"""Central configuration: folder locations and musical constants.

Every path is built from this file's location, so the project works no matter
which folder you start it from.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"              # data/raw/<genre>/*.mid
DOWNLOAD_DIR = DATA_DIR / "downloads"   # downloaded .zip archives
PROCESSED_DIR = DATA_DIR / "processed"  # tokenised dataset
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"

DATASET_PATH = PROCESSED_DIR / "dataset.npz"
DATASET_META_PATH = PROCESSED_DIR / "dataset_meta.json"
MODEL_PATH = MODEL_DIR / "music_lstm.npz"
MODEL_META_PATH = MODEL_DIR / "music_lstm.json"
CHECKPOINT_PATH = MODEL_DIR / "music_lstm.checkpoint.npz"

# ---- musical time -----------------------------------------------------------
STEPS_PER_BEAT = 12      # time grid: 12 steps/beat covers 16th notes AND triplets
NOMINAL_BPM = 120        # the speed at which the grid is "natural"
PITCH_MIN = 21           # A0  (lowest piano key)
PITCH_MAX = 108          # C8  (highest piano key)
MAX_HOLD_BEATS = 16      # longest note the model is allowed to hold

# ---- generation limits (enforced by the web API) ------------------------------
BPM_MIN, BPM_MAX = 40, 200
DURATION_MIN, DURATION_MAX = 10, 120
MAX_SEED_NOTES = 16

# Moods are *derived from the training MIDI files* (see aimusic/analysis.py).
MOODS = ["relaxing", "happy", "sad", "energetic"]

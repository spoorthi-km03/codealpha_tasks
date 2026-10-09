"""OPTIONAL - generate music from the command line (no browser needed).

Example
    python scripts/04_generate_cli.py --genre classical --mood sad --bpm 80 \
        --duration 30 --instrument piano --start-notes "C4, E4, G4"
Files are written to outputs/cli/ .
"""
import argparse
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
from aimusic.config import MODEL_PATH, OUTPUT_DIR
from aimusic.generate import GenerationError
from aimusic.lstm import LSTMModel
from aimusic.studio import produce_track, validate_request
from aimusic.synth import INSTRUMENTS


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--genre", required=True)
    ap.add_argument("--mood", required=True)
    ap.add_argument("--instrument", default="piano", choices=list(INSTRUMENTS))
    ap.add_argument("--bpm", type=int, default=100)
    ap.add_argument("--duration", type=int, default=30)
    ap.add_argument("--start-notes", default="")
    ap.add_argument("--text", default="")
    ap.add_argument("--creativity", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=None)
    a = ap.parse_args()
    try:
        model, meta = LSTMModel.load(MODEL_PATH)
    except FileNotFoundError:
        print("ERROR: no trained model found. Run steps 01-03 first (see README).")
        return 1
    clean, errors = validate_request(vars(a) | {"start_notes": a.start_notes}, meta)
    if errors:
        for k, v in errors.items():
            print(f"ERROR in {k}: {v}")
        return 1
    try:
        result = produce_track(model, meta, clean, OUTPUT_DIR / "cli",
                               progress=lambda s, f: print(f"\r{s:36s} {f * 100:5.1f}%", end="", flush=True))
    except GenerationError as exc:
        print(f"\nERROR: {exc}")
        return 1
    print(f"\n\nWrote {OUTPUT_DIR / 'cli' / result['id']}/track.mid and track.wav")
    print(f"{result['stats']['notes']} notes, key {result['stats']['key']}, "
          f"{result['stats']['audio_seconds']} s of audio")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""STEP 2 - Preprocess MIDI files into numerical token sequences.

Reads every .mid / .midi file in data/raw/<genre>/ and writes
data/processed/dataset.npz (+ dataset_meta.json).
"""
import argparse
import sys

import _bootstrap  # noqa: F401
from aimusic.preprocess import PreprocessError, run


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-files-per-genre", type=int, default=None)
    args = ap.parse_args()
    try:
        run(max_files_per_genre=args.max_files_per_genre)
    except PreprocessError as exc:
        print(f"\nERROR: {exc}")
        return 1
    print("\nNext step:  python scripts/03_train.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())

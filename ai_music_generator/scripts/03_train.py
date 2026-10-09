"""STEP 3 - Train the LSTM on the preprocessed data.

Training is time-boxed.  Presets:
    quick     10 min   smaller network   (checks that everything works)
    standard  30 min   recommended
    long      90 min   best quality on a normal laptop CPU

Examples
    python scripts/03_train.py
    python scripts/03_train.py --preset quick
    python scripts/03_train.py --minutes 45 --hidden 256
    python scripts/03_train.py --resume          # continue an interrupted run
"""
import argparse
import sys

import _bootstrap  # noqa: F401
from aimusic.train import PRESETS, TrainingError, train


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", choices=list(PRESETS), default="standard")
    ap.add_argument("--minutes", type=float, default=None, help="training time budget")
    ap.add_argument("--hidden", type=int, default=None, help="LSTM units per layer")
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--seq-len", type=int, default=192)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--max-steps", type=int, default=None, help="stop after N steps instead of a time budget")
    ap.add_argument("--resume", action="store_true", help="continue from the last checkpoint")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    minutes, hidden = PRESETS[args.preset]
    try:
        train(minutes=args.minutes or minutes, hidden=args.hidden or hidden, layers=args.layers,
              batch_size=args.batch_size, seq_len=args.seq_len, lr=args.lr,
              max_steps=args.max_steps, resume=args.resume, seed=args.seed)
    except TrainingError as exc:
        print(f"\nERROR: {exc}")
        return 1
    print("\nTraining finished. Start the Studio with:  python app.py   (or run.bat)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

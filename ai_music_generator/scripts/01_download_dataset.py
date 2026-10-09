"""STEP 1 - Collect MIDI data.

Downloads public MIDI collections and extracts a manageable sample into
data/raw/<genre>/ .  Safe to run again: existing archives are reused.

Examples
    python scripts/01_download_dataset.py                    # all three default sources
    python scripts/01_download_dataset.py --sources maestro pop909
    python scripts/01_download_dataset.py --max-files 80     # smaller / faster
"""
import argparse
import sys

import _bootstrap  # noqa: F401
from aimusic.dataset import (SOURCES, DownloadError, count_raw_files, manual_instructions,
                             prepare_source)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sources", nargs="+", choices=list(SOURCES), default=list(SOURCES),
                    help="which collections to use (default: all)")
    ap.add_argument("--max-files", type=int, default=None,
                    help="maximum MIDI files per collection (default depends on collection)")
    args = ap.parse_args()

    ok, failed = [], []
    for key in args.sources:
        try:
            n = prepare_source(key, max_files=args.max_files)
            ok.append((key, n))
        except (DownloadError, OSError) as exc:
            print(f"  FAILED: {exc}\n")
            failed.append(key)
        print()

    counts = count_raw_files()
    print("MIDI files now available in data/raw/:")
    for genre, n in counts.items():
        print(f"   {genre:12s} {n:5d} files")
    if failed:
        print(f"\nCould not obtain: {', '.join(failed)}")
        print(manual_instructions())
    if not counts:
        print("\nERROR: no MIDI data is available, so training is not possible yet.")
        return 1
    print("\nNext step:  python scripts/02_preprocess.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())

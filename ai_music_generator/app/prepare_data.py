"""Download the MAESTRO v3 MIDI dataset (classical piano, CC BY-NC-SA 4.0, ~57 MB) into data/midi/classical.
Add other styles yourself: python -m app.prepare_data --add-folder C:\\my_jazz --genre jazz"""
import argparse, io, shutil, sys, urllib.request, zipfile
from pathlib import Path
URL = "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip"
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "midi"

def download():
    dest = DATA / "classical"; dest.mkdir(parents=True, exist_ok=True)
    print("Downloading MAESTRO MIDI (~57 MB)...")
    try:
        buf = io.BytesIO(); r = urllib.request.urlopen(URL, timeout=60)
        while chunk := r.read(1 << 20): buf.write(chunk); print(f"  {buf.tell()//2**20} MB", end="\r")
        z = zipfile.ZipFile(buf)
    except Exception as e:
        print(f"\nDOWNLOAD FAILED: {e}\nManual fix: download {URL}\nunzip any .mid files into {dest} and re-run training."); return 1
    n = 0
    for name in z.namelist():
        if name.lower().endswith((".mid", ".midi")):
            (dest / f"{n:04d}.mid").write_bytes(z.read(name)); n += 1
    print(f"\nExtracted {n} MIDI files to {dest}"); return 0

def add_folder(src, genre):
    dest = DATA / genre; dest.mkdir(parents=True, exist_ok=True); n = 0
    for f in Path(src).rglob("*"):
        if f.suffix.lower() in (".mid", ".midi"): shutil.copy(f, dest / f"{n:04d}.mid"); n += 1
    print(f"Copied {n} files into {dest}"); return 0 if n else 1

if __name__ == "__main__":
    a = argparse.ArgumentParser(); a.add_argument("--add-folder"); a.add_argument("--genre")
    a = a.parse_args()
    if a.add_folder:
        if not a.genre: sys.exit("--genre required")
        sys.exit(add_folder(a.add_folder, a.genre.lower()))
    if any(DATA.glob("classical/*.mid")): print("Dataset already present.")
    else: sys.exit(download())

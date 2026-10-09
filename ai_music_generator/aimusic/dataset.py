"""Dataset acquisition: download public MIDI collections and extract a sample.

Default sources (all free to download; check each licence for your own use):

    genre      collection    licence                     notes
    classical  MAESTRO v3    CC BY-NC-SA 4.0 (non-comm.) concert piano performances
    pop        POP909        MIT                         909 pop songs, piano arrangements
    folk       Nottingham    free folk-tune collection   ~1000 folk / jig tunes

You can add *any* genre yourself by putting legally obtained MIDI files into
``data/raw/<genre>/`` (e.g. ``data/raw/jazz``).  The preprocessing step picks up
every sub-folder automatically, and the Studio then offers exactly the genres
the model was trained on - nothing more.
"""
from __future__ import annotations

import random
import re
import shutil
import sys
import time
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

from .config import DOWNLOAD_DIR, RAW_DIR


@dataclass(frozen=True)
class Source:
    key: str
    genre: str
    urls: tuple
    filename: str
    approx_mb: int
    licence: str
    default_max_files: int
    pattern: re.Pattern          # which zip members to use

    def selects(self, member: str) -> bool:
        return bool(self.pattern.search(member))


SOURCES: Dict[str, Source] = {
    "maestro": Source(
        key="maestro", genre="classical",
        urls=("https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip",),
        filename="maestro-v3.0.0-midi.zip", approx_mb=60,
        licence="CC BY-NC-SA 4.0 (Google Magenta, MAESTRO v3.0.0)",
        default_max_files=150,
        pattern=re.compile(r"\.midi?$", re.I)),
    "pop909": Source(
        key="pop909", genre="pop",
        urls=("https://github.com/music-x-lab/POP909-Dataset/archive/refs/heads/master.zip",
              "https://codeload.github.com/music-x-lab/POP909-Dataset/zip/refs/heads/master"),
        filename="POP909-Dataset-master.zip", approx_mb=80,
        licence="MIT (POP909 dataset, music-x-lab)",
        default_max_files=150,
        # POP909/<id>/<id>.mid  - the main file of each song, not the "versions" copies
        pattern=re.compile(r"POP909/(\d+)/\1\.mid$", re.I)),
    "nottingham": Source(
        key="nottingham", genre="folk",
        urls=("https://github.com/jukedeck/nottingham-dataset/archive/refs/heads/master.zip",
              "https://codeload.github.com/jukedeck/nottingham-dataset/zip/refs/heads/master"),
        filename="nottingham-dataset-master.zip", approx_mb=5,
        licence="Free folk-tune collection (Nottingham Music Database)",
        default_max_files=300,
        pattern=re.compile(r"MIDI/[^/]+\.mid$", re.I)),
}


class DownloadError(RuntimeError):
    pass


def download_file(urls, dest: Path, log: Callable[[str], None] = print,
                  retries: int = 3, timeout: int = 60) -> Path:
    """Download the first working URL to ``dest`` (resumable-by-restart, atomic)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    errors = []
    for url in urls:
        for attempt in range(1, retries + 1):
            part = dest.with_suffix(dest.suffix + ".part")
            try:
                log(f"  downloading {url} (attempt {attempt}/{retries})")
                req = urllib.request.Request(url, headers={"User-Agent": "ai-music-studio/1.0"})
                with urllib.request.urlopen(req, timeout=timeout) as resp, open(part, "wb") as out:
                    total = int(resp.headers.get("Content-Length") or 0)
                    done, last = 0, 0.0
                    while True:
                        chunk = resp.read(1 << 16)
                        if not chunk:
                            break
                        out.write(chunk)
                        done += len(chunk)
                        if time.time() - last > 1.0:
                            last = time.time()
                            pct = f"{100 * done / total:5.1f}%" if total else ""
                            sys.stdout.write(f"\r    {done / 1e6:7.1f} MB {pct}")
                            sys.stdout.flush()
                    sys.stdout.write("\n")
                if not zipfile.is_zipfile(part):
                    raise DownloadError("downloaded file is not a valid zip archive")
                part.replace(dest)
                return dest
            except (urllib.error.URLError, OSError, DownloadError, TimeoutError) as exc:
                errors.append(f"{url}: {exc}")
                log(f"    failed: {exc}")
                if part.exists():
                    part.unlink()
                time.sleep(min(2 * attempt, 6))
    raise DownloadError("all download attempts failed:\n    " + "\n    ".join(errors))


def extract_sample(source: Source, zip_path: Path, raw_dir: Path = RAW_DIR,
                   max_files: Optional[int] = None, seed: int = 42,
                   log: Callable[[str], None] = print) -> int:
    """Extract (at most ``max_files``) MIDI files of ``source`` into raw/<genre>/."""
    max_files = max_files or source.default_max_files
    target = raw_dir / source.genre
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        members = sorted(m for m in zf.namelist() if source.selects(m))
        if not members:
            raise DownloadError(f"{zip_path.name} contains no MIDI files matching the expected layout")
        if len(members) > max_files:
            members = sorted(random.Random(seed).sample(members, max_files))
        count = 0
        for i, name in enumerate(members):
            safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(name).name)
            out = target / f"{source.key}_{i:04d}_{safe}"
            if not out.suffix.lower() in (".mid", ".midi"):
                out = out.with_suffix(".mid")
            with zf.open(name) as src, open(out, "wb") as dst:
                shutil.copyfileobj(src, dst)
            count += 1
    log(f"  extracted {count} files -> {target}")
    return count


def prepare_source(key: str, raw_dir: Path = RAW_DIR, download_dir: Path = DOWNLOAD_DIR,
                   max_files: Optional[int] = None, log: Callable[[str], None] = print) -> int:
    """Download (if needed) and extract one source. Returns number of files."""
    source = SOURCES[key]
    zip_path = download_dir / source.filename
    log(f"[{key}] genre '{source.genre}' - {source.licence}")
    if zip_path.exists() and zipfile.is_zipfile(zip_path):
        log(f"  using existing archive {zip_path.name}")
    else:
        log(f"  archive not found, downloading (about {source.approx_mb} MB)...")
        download_file(source.urls, zip_path, log)
    return extract_sample(source, zip_path, raw_dir, max_files, log=log)


def manual_instructions() -> str:
    lines = ["", "Could not download automatically? Do it by hand:"]
    for s in SOURCES.values():
        lines.append(f"  * {s.key}: download  {s.urls[0]}")
        lines.append(f"      and save it as  data/downloads/{s.filename}  then run this script again.")
    lines.append("  * Or copy your own legally obtained .mid files into data/raw/<genre>/")
    lines.append("    (for example data/raw/jazz/) and skip straight to preprocessing.")
    return "\n".join(lines)


def count_raw_files(raw_dir: Path = RAW_DIR) -> Dict[str, int]:
    out: Dict[str, int] = {}
    if raw_dir.exists():
        for d in sorted(p for p in raw_dir.iterdir() if p.is_dir()):
            n = sum(1 for f in d.rglob("*") if f.suffix.lower() in (".mid", ".midi"))
            if n:
                out[d.name] = n
    return out

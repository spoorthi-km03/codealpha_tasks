"""Minimal pure-Python MIDI reader/writer (no external MIDI library needed)."""
import struct
from pathlib import Path

def _vlq(data, i):
    v = 0
    while True:
        b = data[i]; i += 1
        v = (v << 7) | (b & 0x7F)
        if not b & 0x80:
            return v, i

def read_notes(path):
    """Return list of (start_beat, end_beat, pitch) from a .mid file."""
    data = Path(path).read_bytes()
    if data[:4] != b"MThd":
        raise ValueError("not a MIDI file")
    _, ntrk, tpq = struct.unpack(">HHH", data[8:14])
    if tpq & 0x8000 or tpq == 0:
        raise ValueError("SMPTE timing unsupported")
    i, notes = 14, []
    for _ in range(ntrk):
        if data[i:i+4] != b"MTrk":
            break
        ln = struct.unpack(">I", data[i+4:i+8])[0]
        trk, i = data[i+8:i+8+ln], i+8+ln
        p, t, status, active = 0, 0, 0, {}
        while p < len(trk):
            d, p = _vlq(trk, p); t += d
            b = trk[p]
            if b == 0xFF:
                l, p2 = _vlq(trk, p+2); p = p2 + l; continue
            if b in (0xF0, 0xF7):
                l, p2 = _vlq(trk, p+1); p = p2 + l; continue
            if b & 0x80:
                status = b; p += 1
            kind, ch = status & 0xF0, status & 0x0F
            n = 1 if kind in (0xC0, 0xD0) else 2
            a = trk[p:p+n]; p += n
            if ch == 9:
                continue  # drums
            if kind == 0x90 and a[1] > 0:
                active.setdefault((ch, a[0]), t)
            elif kind == 0x80 or (kind == 0x90 and a[1] == 0):
                s = active.pop((ch, a[0]), None)
                if s is not None:
                    notes.append((s/tpq, max(t, s+1)/tpq, a[0]))
    return sorted(notes)

def write_midi(path, notes, bpm, program=0):
    """notes: list of (start_beat, dur_beats, pitch, velocity)."""
    tpq = 480
    ev = []
    for s, d, p, v in notes:
        ev.append((round(s*tpq), 1, p, v)); ev.append((round((s+d)*tpq), 0, p, 0))
    ev.sort(key=lambda e: (e[0], e[1]))
    def vlq(n):
        out = [n & 0x7F]; n >>= 7
        while n: out.append((n & 0x7F) | 0x80); n >>= 7
        return bytes(reversed(out))
    trk = b"\x00\xFF\x51\x03" + int(60_000_000/bpm).to_bytes(3, "big") + b"\x00\xC0" + bytes([program])
    last = 0
    for t, on, p, v in ev:
        trk += vlq(t-last) + bytes([0x90 if on else 0x80, p, v]); last = t
    trk += b"\x00\xFF\x2F\x00"
    with open(path, "wb") as f:
        f.write(b"MThd" + struct.pack(">IHHH", 6, 0, 1, tpq) + b"MTrk" + struct.pack(">I", len(trk)) + trk)

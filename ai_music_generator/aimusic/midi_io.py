"""Minimal, dependency-free Standard MIDI File (SMF) reader and writer.

Why write our own?  It removes a dependency that is awkward to install on some
Windows / Python 3.12 setups and keeps the whole data path easy to explain.

Reader features
    * format 0 and 1 files, running status, meta / sysex events
    * tempo map  -> real duration in seconds (used for mood analysis)
    * sustain pedal (CC64): notes keep sounding while the pedal is down, which
      matters a lot for piano recordings such as MAESTRO
    * drums (channel 10) are ignored - they have no pitch
    * truncated / slightly corrupt files are parsed as far as possible

Writer features
    * format 1 file: a tempo track + one instrument track
"""
from __future__ import annotations

import bisect
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple, Union

# note tuple used all over the project: (start, end, pitch, velocity)
Note = Tuple[int, int, int, int]


class MidiParseError(Exception):
    """Raised when a file is not a usable MIDI file."""


@dataclass
class MidiData:
    ticks_per_beat: int
    notes: List[Note] = field(default_factory=list)   # in ticks
    tempo_map: List[Tuple[int, int]] = field(default_factory=list)  # (tick, µs/beat)
    end_tick: int = 0

    def ticks_to_seconds(self, tick: int) -> float:
        """Convert an absolute tick to seconds using the tempo map."""
        tempos = self.tempo_map or [(0, 500000)]
        if tempos[0][0] != 0:
            tempos = [(0, 500000)] + tempos
        seconds, prev_tick, prev_tempo = 0.0, 0, tempos[0][1]
        for t, tempo in tempos[1:]:
            if t >= tick:
                break
            seconds += (t - prev_tick) * prev_tempo / (self.ticks_per_beat * 1e6)
            prev_tick, prev_tempo = t, tempo
        seconds += (tick - prev_tick) * prev_tempo / (self.ticks_per_beat * 1e6)
        return seconds

    @property
    def duration_seconds(self) -> float:
        return self.ticks_to_seconds(self.end_tick)


# --------------------------------------------------------------------------- #
# reading
# --------------------------------------------------------------------------- #
def _read_vlq(data: bytes, pos: int) -> Tuple[int, int]:
    value = 0
    for _ in range(4):
        byte = data[pos]
        pos += 1
        value = (value << 7) | (byte & 0x7F)
        if not byte & 0x80:
            return value, pos
    raise MidiParseError("invalid variable-length quantity")


def _parse_track(body: bytes, events: list, tempos: list) -> int:
    """Parse one MTrk body. Appends to ``events`` / ``tempos``; returns last tick."""
    i, tick, status = 0, 0, None
    n = len(body)
    try:
        while i < n:
            delta, i = _read_vlq(body, i)
            tick += delta
            first = body[i]
            if first == 0xFF:  # meta event
                meta_type = body[i + 1]
                length, j = _read_vlq(body, i + 2)
                payload = body[j:j + length]
                i = j + length
                if meta_type == 0x51 and length == 3:
                    tempos.append((tick, int.from_bytes(payload, "big")))
                elif meta_type == 0x2F:
                    break
                continue
            if first in (0xF0, 0xF7):  # sysex
                length, j = _read_vlq(body, i + 1)
                i = j + length
                status = None
                continue
            if first >= 0xF8:  # system real-time, no data
                i += 1
                continue
            if first >= 0xF1:  # other system common messages
                i += {0xF1: 2, 0xF2: 3, 0xF3: 2}.get(first, 1)
                continue
            if first & 0x80:
                status = first
                i += 1
            elif status is None:
                raise MidiParseError("data byte without status")
            high, channel = status & 0xF0, status & 0x0F
            if high in (0xC0, 0xD0):  # program change / channel pressure
                i += 1
                continue
            d1, d2 = body[i], body[i + 1]
            i += 2
            if high == 0x90 and d2 > 0:
                events.append((tick, 2, channel, d1, d2))        # note on
            elif high in (0x80, 0x90):
                events.append((tick, 0, channel, d1, 0))         # note off
            elif high == 0xB0 and d1 == 64:
                events.append((tick, 1, channel, d1, d2))        # sustain pedal
    except (IndexError, MidiParseError):
        pass  # keep whatever was parsed before the damage
    return tick


def read_midi_bytes(data: bytes) -> MidiData:
    if len(data) < 14 or data[:4] != b"MThd":
        raise MidiParseError("not a MIDI file (missing MThd header)")
    header_len = int.from_bytes(data[4:8], "big")
    _fmt, _ntrks, division = struct.unpack(">HHH", data[8:14])
    if division & 0x8000:
        raise MidiParseError("SMPTE time division is not supported")
    if division == 0:
        raise MidiParseError("invalid time division")
    pos = 8 + header_len
    events: list = []
    tempos: list = []
    last_tick = 0
    while pos + 8 <= len(data):
        chunk_id = data[pos:pos + 4]
        length = int.from_bytes(data[pos + 4:pos + 8], "big")
        body = data[pos + 8:pos + 8 + length]
        pos += 8 + length
        if chunk_id == b"MTrk":
            last_tick = max(last_tick, _parse_track(body, events, tempos))
    if not events:
        raise MidiParseError("no note events found")

    # offs first, then pedal, then ons at identical ticks
    events.sort(key=lambda e: (e[0], e[1]))
    notes: List[Note] = []
    active: dict = {}                 # (channel, pitch) -> (start, velocity)
    pedal: dict = {}                  # channel -> bool
    held: dict = {}                   # channel -> list[(start, pitch, vel)]
    for tick, kind, ch, d1, d2 in events:
        if ch == 9:                   # drums
            continue
        if kind == 2:                 # note on
            key = (ch, d1)
            if key in active:         # re-trigger without note-off
                s, v = active.pop(key)
                notes.append((s, tick, d1, v))
            for h in list(held.get(ch, [])):
                if h[1] == d1:        # re-strike while sustained
                    held[ch].remove(h)
                    notes.append((h[0], tick, d1, h[2]))
            active[key] = (tick, d2)
        elif kind == 0:               # note off
            key = (ch, d1)
            if key in active:
                s, v = active.pop(key)
                if pedal.get(ch):
                    held.setdefault(ch, []).append((s, d1, v))
                else:
                    notes.append((s, tick, d1, v))
        else:                         # pedal
            down = d2 >= 64
            if pedal.get(ch) and not down:
                for s, p, v in held.pop(ch, []):
                    notes.append((s, tick, p, v))
            pedal[ch] = down
    end = max([e[0] for e in events] + [last_tick])
    for (ch, p), (s, v) in active.items():
        notes.append((s, end, p, v))
    for ch, lst in held.items():
        for s, p, v in lst:
            notes.append((s, end, p, v))
    notes = [n for n in notes if n[1] > n[0]]
    notes.sort()
    if not notes:
        raise MidiParseError("no playable notes found")
    tempos.sort()
    return MidiData(ticks_per_beat=division, notes=notes, tempo_map=tempos, end_tick=end)


def read_midi(path: Union[str, Path]) -> MidiData:
    return read_midi_bytes(Path(path).read_bytes())


# --------------------------------------------------------------------------- #
# writing
# --------------------------------------------------------------------------- #
def _vlq(value: int) -> bytes:
    out = [value & 0x7F]
    value >>= 7
    while value:
        out.append((value & 0x7F) | 0x80)
        value >>= 7
    return bytes(reversed(out))


def _track_chunk(payload: bytes) -> bytes:
    return b"MTrk" + len(payload).to_bytes(4, "big") + payload


def write_midi(path: Union[str, Path], notes: Iterable[Note], bpm: float,
               steps_per_beat: int, program: int = 0,
               ticks_per_beat: int = 480, title: str = "AI Music Studio") -> Path:
    """Write ``notes`` ((start_step, end_step, pitch, velocity)) as a .mid file.

    ``bpm`` becomes the tempo of the file, so the same note data written with a
    different BPM plays faster or slower - exactly what the Studio tempo control
    does.
    """
    if ticks_per_beat % steps_per_beat:
        raise ValueError("ticks_per_beat must be a multiple of steps_per_beat")
    tps = ticks_per_beat // steps_per_beat
    tempo_us = int(round(60_000_000 / bpm))

    name = title.encode("ascii", "replace")
    tempo_track = (b"\x00\xFF\x03" + _vlq(len(name)) + name +
                   b"\x00\xFF\x51\x03" + tempo_us.to_bytes(3, "big") +
                   b"\x00\xFF\x58\x04\x04\x02\x18\x08" +       # 4/4
                   b"\x00\xFF\x2F\x00")

    ev = []  # (tick, order, status, d1, d2)
    for start, end, pitch, vel in notes:
        if end <= start:
            continue
        ev.append((start * tps, 1, 0x90, int(pitch), max(1, min(127, int(vel)))))
        ev.append((end * tps, 0, 0x80, int(pitch), 0))
    ev.sort()
    body = bytearray(b"\x00\xC0" + bytes([program & 0x7F]))
    last = 0
    for tick, _order, status, d1, d2 in ev:
        body += _vlq(tick - last) + bytes([status, d1, d2])
        last = tick
    body += b"\x00\xFF\x2F\x00"

    header = b"MThd" + (6).to_bytes(4, "big") + struct.pack(">HHH", 1, 2, ticks_per_beat)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(header + _track_chunk(tempo_track) + _track_chunk(bytes(body)))
    return path

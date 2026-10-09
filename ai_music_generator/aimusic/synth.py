"""A small additive synthesizer: notes -> stereo WAV audio (NumPy only).

IMPORTANT - what is AI and what is not
    The *notes* come from the trained LSTM.  This module only turns notes into
    sound, like a sound card or a sound-font would.  The instrument choice changes
    the timbre (the synthesised sound) and the General-MIDI program stored in the
    .mid file.  The LSTM does NOT learn instrument-specific music.

Each instrument is a recipe: a set of harmonic partials, how fast each partial
decays, an attack/release envelope and optional vibrato.  A gentle stereo reverb
(FFT convolution with a decaying-noise impulse response) is added at the end.
"""
from __future__ import annotations

import wave
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

SAMPLE_RATE = 44100

# key -> (label, General-MIDI program)
INSTRUMENTS: Dict[str, Tuple[str, int]] = {
    "piano": ("Grand piano", 0),
    "guitar": ("Acoustic guitar", 25),
    "violin": ("Violin", 40),
    "flute": ("Flute", 73),
    "harp": ("Harp", 46),
    "music_box": ("Music box", 10),
}

NoteSec = Tuple[float, float, int, int]   # (start_s, end_s, pitch, velocity)


def midi_to_freq(p: int) -> float:
    return 440.0 * 2.0 ** ((p - 69) / 12.0)


# --------------------------------------------------------------------------- #
# instrument voices
# --------------------------------------------------------------------------- #
def _voice(inst: str, pitch: int, vel: int, held: float, sr: int, rng) -> np.ndarray:
    """One note: sounding for ``held`` seconds, then released."""
    f0 = midi_to_freq(pitch)
    vnorm = vel / 127.0
    kmax_hz = 0.45 * sr
    vib_hz = vib_depth = 0.0
    noise = 0.0

    if inst == "piano":
        K = int(min(16, kmax_hz / f0))
        k = np.arange(1, K + 1)
        ratios = k * np.sqrt(1 + 0.0003 * k ** 2)
        amps = k ** -1.05 * np.exp(-k * (1 - vnorm) * 0.30)
        tau0 = max(0.35, 6.0 * 2.0 ** (-(pitch - 36) / 18.0))
        taus = tau0 / (1 + 0.55 * (k - 1))
        attack, release, sustain_mode = 0.004, 0.14, "decay"
    elif inst == "guitar":
        K = int(min(12, kmax_hz / f0))
        k = np.arange(1, K + 1)
        ratios = k.astype(float)
        pick = np.abs(np.sin(np.pi * k * 0.17)) + 0.35        # plucking-position comb
        amps = k ** -0.9 * pick
        tau0 = max(0.3, 2.4 * 2.0 ** (-(pitch - 40) / 30.0))
        taus = tau0 / (1 + 0.9 * (k - 1))
        attack, release, sustain_mode = 0.002, 0.09, "decay"
    elif inst == "harp":
        K = int(min(9, kmax_hz / f0))
        k = np.arange(1, K + 1)
        ratios = k.astype(float)
        amps = k ** -1.3
        tau0 = max(0.4, 3.2 * 2.0 ** (-(pitch - 40) / 28.0))
        taus = tau0 / (1 + 0.6 * (k - 1))
        attack, release, sustain_mode = 0.003, 0.18, "decay"
    elif inst == "music_box":
        ratios = np.array([1.0, 2.0, 4.1, 6.3])
        ratios = ratios[ratios * f0 < kmax_hz]
        amps = np.array([1.0, 0.35, 0.2, 0.1])[:len(ratios)]
        tau0 = max(0.3, 1.6 * 2.0 ** (-(pitch - 72) / 24.0))
        taus = tau0 / (1 + 0.8 * np.arange(len(ratios)))
        attack, release, sustain_mode = 0.002, 0.05, "decay"
    elif inst == "violin":
        K = int(min(16, kmax_hz / f0))
        k = np.arange(1, K + 1)
        ratios = k.astype(float)
        amps = k ** -0.95 / (1 + (k * f0 / 3200.0) ** 2)
        taus = np.full(K, 1e9)
        attack, release, sustain_mode = 0.07, 0.12, "sustain"
        vib_hz, vib_depth = 5.3, 0.005
        noise = 0.004
    elif inst == "flute":
        K = int(min(5, kmax_hz / f0))
        k = np.arange(1, K + 1)
        ratios = k.astype(float)
        amps = np.array([1.0, 0.25, 0.12, 0.05, 0.03])[:K]
        taus = np.full(K, 1e9)
        attack, release, sustain_mode = 0.06, 0.10, "sustain"
        vib_hz, vib_depth = 4.8, 0.003
        noise = 0.012
    else:
        raise ValueError(f"unknown instrument '{inst}'")

    # decaying instruments are inaudible long before a very long hold ends
    if sustain_mode == "decay":
        total = min(held + release, float(taus.max()) * 7.0 + release)
    else:
        total = held + release
    n = max(int(total * sr), 8)
    t = np.arange(n, dtype=np.float32) / sr

    # slow, smooth frequency modulation (vibrato) fades in after 0.2 s
    if vib_hz:
        depth = vib_depth * np.clip((t - 0.2) / 0.4, 0, 1)
        mod = 1.0 + depth * np.sin(2 * np.pi * vib_hz * t)
        phase0 = 2 * np.pi * f0 * np.cumsum(mod) / sr
    else:
        phase0 = 2 * np.pi * f0 * t

    sig = np.zeros(n, dtype=np.float32)
    for r, a, tau in zip(ratios, amps, taus):
        part = np.sin(r * phase0).astype(np.float32) * np.float32(a)
        if tau < 1e8:
            part *= np.exp(-t / np.float32(tau))
        sig += part
    if noise:
        sig += (rng.standard_normal(n).astype(np.float32) * noise)

    env = np.minimum(1.0, t / attack)
    rel = np.exp(-np.maximum(t - held, 0.0) / np.float32(release / 4.0))
    sig *= (env * rel).astype(np.float32)
    # peak-normalise the voice roughly, then apply velocity
    norm = max(float(np.abs(sig).max()), 1e-6)
    return sig / norm * np.float32(vnorm ** 1.3)


# --------------------------------------------------------------------------- #
# mixing
# --------------------------------------------------------------------------- #
def _reverb(x: np.ndarray, sr: int, seed: int, tau: float = 0.38, length: float = 1.5) -> np.ndarray:
    """Block-wise FFT convolution with a decaying-noise impulse response."""
    rng = np.random.default_rng(seed)
    m = int(length * sr)
    tt = np.arange(m) / sr
    ir = (rng.standard_normal(m) * np.exp(-tt / tau)).astype(np.float64)
    ir /= np.sqrt((ir ** 2).sum()) + 1e-9
    block = 1 << 16
    nfft = 1 << int(np.ceil(np.log2(block + m)))
    IR = np.fft.rfft(ir, nfft)
    out = np.zeros(len(x) + m, dtype=np.float64)
    for s in range(0, len(x), block):
        seg = x[s:s + block].astype(np.float64)
        y = np.fft.irfft(np.fft.rfft(seg, nfft) * IR, nfft)[:len(seg) + m - 1]
        out[s:s + len(y)] += y
    return out[:len(x)].astype(np.float32)


def render_audio(notes: Sequence[NoteSec], duration: float, instrument: str,
                 sr: int = SAMPLE_RATE, seed: int = 0, reverb: float = 0.22,
                 progress=None) -> np.ndarray:
    """Render notes to a float32 array of shape (n_samples, 2), exactly ``duration`` long."""
    if instrument not in INSTRUMENTS:
        raise ValueError(f"unknown instrument '{instrument}'")
    n_total = int(round(duration * sr))
    left = np.zeros(n_total, dtype=np.float32)
    right = np.zeros(n_total, dtype=np.float32)
    rng = np.random.default_rng(seed)
    total_notes = max(1, len(notes))
    for idx, (start, end, pitch, vel) in enumerate(notes):
        if progress and idx % 25 == 0:
            progress(0.9 * idx / total_notes)
        i0 = int(start * sr)
        if i0 >= n_total:
            continue
        held = max(0.03, end - start)
        voice = _voice(instrument, int(pitch), int(vel), held, sr, rng)
        i1 = min(n_total, i0 + len(voice))
        voice = voice[:i1 - i0]
        pan = float(np.clip((pitch - 60) / 48.0, -0.6, 0.6))          # low notes left, high right
        ang = (pan + 1.0) * np.pi / 4.0
        left[i0:i1] += voice * np.float32(np.cos(ang))
        right[i0:i1] += voice * np.float32(np.sin(ang))
    if progress:
        progress(0.92)
    if reverb > 0 and n_total > 0:
        wl, wr = _reverb(left, sr, 11), _reverb(right, sr, 23)
        left = left * (1 - reverb) + wl * reverb * 1.6
        right = right * (1 - reverb) + wr * reverb * 1.6
    stereo = np.stack([left, right], axis=1)
    # short fade-in and a gentle fade-out so the clip ends cleanly at ``duration``
    fade_out = min(int(0.7 * sr), n_total // 2)
    if fade_out > 0:
        stereo[-fade_out:] *= np.linspace(1.0, 0.0, fade_out, dtype=np.float32)[:, None]
    fade_in = min(int(0.005 * sr), n_total)
    if fade_in > 0:
        stereo[:fade_in] *= np.linspace(0.0, 1.0, fade_in, dtype=np.float32)[:, None]
    peak = float(np.abs(stereo).max())
    if peak > 0:
        stereo *= 0.89 / peak
    if progress:
        progress(1.0)
    return stereo


def write_wav(path: Path, audio: np.ndarray, sr: int = SAMPLE_RATE) -> Path:
    """Write a float32 (n, 2) array as a 16-bit PCM stereo WAV file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(audio.shape[1])
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())
    return path

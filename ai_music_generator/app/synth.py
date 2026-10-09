"""Numpy additive synthesizer -> WAV (no FluidSynth/soundfont needed)."""
import wave
import numpy as np
SR = 22050
INSTR = {  # name: (GM program, harmonic amps, attack, decay rate, vibrato)
    "piano":  (0,  [1, .5, .25, .12, .06], .005, 3.0, 0),
    "guitar": (24, [1, .7, .4, .25, .15, .08], .003, 5.0, 0),
    "violin": (40, [1, .6, .5, .35, .2, .1], .08, 0.3, 1),
    "pad":    (88, [1, .3, .15], .25, 0.2, 0.5),
}
def render(notes, bpm, instrument="piano"):
    _, amps, att, dec, vib = INSTR[instrument]
    spb = 60.0/bpm
    total = max(s+d for s, d, _, _ in notes)*spb + 1.5
    out = np.zeros(int(total*SR)+1, dtype=np.float64)
    for s, d, p, v in notes:
        f, dur = 440*2**((p-69)/12), d*spb + 0.3
        t = np.arange(int(dur*SR))/SR
        ph = 2*np.pi*f*t + vib*0.15*np.sin(2*np.pi*5.5*t)
        w = sum(a*np.sin((k+1)*ph) for k, a in enumerate(amps))
        env = np.minimum(t/att, 1)*np.exp(-dec*t)
        rel = np.clip((d*spb + 0.3 - t)/0.3, 0, 1)
        i = int(s*spb*SR); seg = w*env*rel*(v/127)*0.25
        out[i:i+len(seg)] += seg[:len(out)-i]
    m = np.abs(out).max()
    return (out/m*0.9 if m > 0 else out).astype(np.float32)

def write_wav(path, audio):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes((audio*32767).astype("<i2").tobytes())

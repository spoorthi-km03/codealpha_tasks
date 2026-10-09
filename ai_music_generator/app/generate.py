import re, uuid
from pathlib import Path
import numpy as np
from .tokens import REST, HOLD, OFFSET, VOCAB, parse_start_notes, tokens_to_notes, pitch_to_tok
from .midi_io import write_midi
from .synth import render, write_wav, INSTR
ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH, OUT = ROOT / "models" / "model.pt", ROOT / "outputs"
# Mood is NOT learned: it maps to sampling temperature, loudness and note length (honest, documented).
MOODS = {"relaxing": (0.7, 60, 1.0), "happy": (0.9, 85, 0.8), "sad": (0.8, 55, 1.0),
         "energetic": (1.15, 100, 0.6), "neutral": (0.9, 80, 0.9)}
MOOD_WORDS = {"calm": "relaxing", "relax": "relaxing", "happy": "happy", "joy": "happy", "sad": "sad",
              "melanchol": "sad", "energ": "energetic", "fast": "energetic", "upbeat": "energetic"}
_cache = {}

def model_status():
    if not MODEL_PATH.exists():
        return {"trained": False, "message": "No trained model found. Run: python -m app.prepare_data then python -m app.train (or run.bat)."}
    try:
        info = load_model()
        return {"trained": True, "genres": info["genres"], "windows": info["windows"], "final_loss": round(info["loss"][-1], 3)}
    except Exception as e:
        return {"trained": False, "message": f"Model file could not be loaded: {e}"}

def load_model():
    if "m" not in _cache:
        import torch
        from .model import MusicLSTM
        ck = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
        m = MusicLSTM(**ck["cfg"]); m.load_state_dict(ck["state"]); m.eval()
        _cache["m"] = dict(model=m, genres=ck["genres"], loss=ck["loss"], windows=ck["windows"])
    return _cache["m"]

def sample_tokens(n_steps, genre, temperature, seed_pitches):
    import torch
    info = load_model(); m = info["model"]
    g = torch.tensor([info["genres"].index(genre)])
    seq = [REST]
    for p in seed_pitches: seq += [pitch_to_tok(p), HOLD, HOLD, HOLD]  # each seed note = one beat
    h, logits = None, None
    with torch.no_grad():
        logits, h = m(torch.tensor([seq]), g)
        logits = logits[0, -1]
        while len(seq) < n_steps + 1:
            p = torch.softmax(logits/temperature, -1).numpy().astype(np.float64)
            top = np.argsort(p)[-12:]; q = np.zeros_like(p); q[top] = p[top]; q /= q.sum()
            tok = int(np.random.choice(VOCAB, p=q))
            if tok == HOLD and seq[-1] == REST: tok = REST
            seq.append(tok)
            logits, h = m(torch.tensor([[tok]]), g, h); logits = logits[0, -1]
    return seq[1:n_steps+1]

def generate(genre, mood, instrument, bpm, duration, start_notes="", text=""):
    warnings = []
    st = model_status()
    if not st["trained"]: raise RuntimeError(st["message"])
    if genre not in st["genres"]: raise ValueError(f"Genre '{genre}' not in trained model: {st['genres']}")
    if instrument not in INSTR: raise ValueError(f"Unsupported instrument '{instrument}'.")
    if mood not in MOODS: raise ValueError(f"Unsupported mood '{mood}'.")
    if not 40 <= bpm <= 220: raise ValueError("Tempo must be 40-220 BPM.")
    if not 10 <= duration <= 120: raise ValueError("Duration must be 10-120 seconds.")
    seeds = parse_start_notes(start_notes)
    if len(seeds) > 16: raise ValueError("Use at most 16 starting notes.")
    if text.strip():
        hit = next((v for k, v in MOOD_WORDS.items() if k in text.lower()), None)
        if hit and mood == "neutral": mood = hit; warnings.append(f"Text instruction matched mood keyword -> '{hit}'.")
        else: warnings.append("Free-text instructions are not understood by the LSTM; only mood keywords (calm, happy, sad, energetic) are used.")
    temp, vel, gate = MOODS[mood]
    n_steps = max(len(seeds)*4 + 8, int(duration*bpm/60*4))
    toks = sample_tokens(n_steps, genre, temp, seeds)
    notes = tokens_to_notes(toks, vel, gate)
    if len(notes) < 3: raise RuntimeError("Model produced almost no notes; try again or retrain with more epochs.")
    uid = uuid.uuid4().hex[:10]; OUT.mkdir(exist_ok=True)
    write_midi(OUT/f"{uid}.mid", notes, bpm, INSTR[instrument][0])
    audio = render(notes, bpm, instrument); write_wav(OUT/f"{uid}.wav", audio)
    return dict(id=uid, midi=f"/outputs/{uid}.mid", wav=f"/outputs/{uid}.wav", notes=len(notes), seconds=round(len(audio)/22050, 1),
                bpm=bpm, genre=genre, mood=mood, instrument=instrument, seed_used=bool(seeds), warnings=warnings)

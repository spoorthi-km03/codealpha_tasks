import sys, tempfile, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
from app.midi_io import read_notes, write_midi
from app import tokens as T
from app.synth import render, write_wav
from app.preprocess import load_dataset
NOTES = [(i*1.0, 1.0, p, 80) for i, p in enumerate([60, 64, 67, 72, 71, 67, 64, 60]*20)]

class Tests(unittest.TestCase):
    def test_midi_roundtrip_and_tokens(self):
        with tempfile.TemporaryDirectory() as d:
            f = Path(d)/"a.mid"; write_midi(f, NOTES, 120)
            n = read_notes(f); self.assertEqual(len(n), len(NOTES)); self.assertEqual(n[0][2], 60)
            tk = T.notes_to_tokens(n); back = T.tokens_to_notes(tk)
            self.assertEqual([x[2] for x in back][:4], [60, 64, 67, 72])
    def test_note_parsing(self):
        self.assertEqual(T.parse_start_notes("C4, E4 G4"), [60, 64, 67]); self.assertEqual(T.parse_note_name("Bb3"), 58)
        with self.assertRaises(ValueError): T.parse_start_notes("H9")
    def test_synth_tempo_changes_length(self):
        n = T.tokens_to_notes(T.notes_to_tokens(read_notes_from(NOTES)))
        self.assertGreater(len(render(n, 60, "violin")), len(render(n, 120, "violin")))
        with tempfile.TemporaryDirectory() as d: write_wav(Path(d)/"x.wav", render(n, 100, "guitar")); self.assertGreater((Path(d)/"x.wav").stat().st_size, 1000)
    def test_preprocess(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d)/"classical").mkdir(); write_midi(Path(d)/"classical"/"a.mid", NOTES, 100)
            X, G, g = load_dataset(d, log=lambda *_: None); self.assertEqual(g, ["classical"]); self.assertEqual(X.shape[1], 65)
    def test_flask_routes(self):
        from app.server import app
        c = app.test_client(); self.assertEqual(c.get("/").status_code, 200)
        r = c.post("/api/generate", json={"genre": "classical"})
        self.assertIn(r.status_code, (400, 503)); self.assertIn("error", r.get_json())  # never fake success
def read_notes_from(n):
    return [(s, s+d, p) for s, d, p, _ in n]
if __name__ == "__main__": unittest.main()

from flask import Flask, jsonify, render_template, request, send_from_directory
from . import generate as G
from .synth import INSTR
app = Flask(__name__)

@app.get("/")
def index(): return render_template("index.html", moods=list(G.MOODS), instruments=list(INSTR))
@app.get("/api/status")
def status(): return jsonify(G.model_status())
@app.get("/outputs/<path:f>")
def outputs(f): return send_from_directory(G.OUT, f)
@app.post("/api/generate")
def gen():
    d = request.get_json(silent=True) or {}
    try:
        return jsonify(G.generate(d.get("genre", ""), d.get("mood", "neutral"), d.get("instrument", "piano"),
                       int(d.get("bpm", 100)), int(d.get("duration", 30)), d.get("start_notes", ""), d.get("text", "")))
    except (ValueError, TypeError) as e: return jsonify(error=str(e)), 400
    except Exception as e: return jsonify(error=str(e)), 503
if __name__ == "__main__": app.run(host="127.0.0.1", port=5000)

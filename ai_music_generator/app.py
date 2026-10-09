"""AI Music Studio - Flask web application.

Start with:   python app.py      (or double-click run.bat on Windows)
Then open:    http://127.0.0.1:5000
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_file

from aimusic.config import (BPM_MAX, BPM_MIN, DURATION_MAX, DURATION_MIN, MAX_SEED_NOTES,
                            MODEL_PATH, OUTPUT_DIR)
from aimusic.generate import TEXT_KEYWORDS
from aimusic.studio import (MOOD_HINTS, TRACK_ID_RE, JobManager, ModelManager, validate_request)
from aimusic.synth import INSTRUMENTS


def create_app(model_path: Path = MODEL_PATH, output_dir: Path = OUTPUT_DIR) -> Flask:
    app = Flask(__name__)
    app.config["JSON_SORT_KEYS"] = False
    models = ModelManager(model_path)
    jobs = JobManager(models, output_dir)
    out_root = Path(output_dir)
    app.extensions["aimusic_models"] = models
    app.extensions["aimusic_jobs"] = jobs

    # ---------------------------------------------------------------- pages
    @app.get("/")
    def index():
        return render_template("index.html")

    # ------------------------------------------------------------------ API
    @app.get("/api/status")
    def status():
        model, meta, error = models.get()
        keywords = {cat: {k: v.replace(r"\b", "").strip("()").replace("|", ", ")
                          for k, v in opts.items()} for cat, opts in TEXT_KEYWORDS.items()}
        payload = {
            "ready": model is not None,
            "error": error,
            "instruments": [{"id": k, "label": v[0]} for k, v in INSTRUMENTS.items()],
            "limits": {"bpm_min": BPM_MIN, "bpm_max": BPM_MAX, "duration_min": DURATION_MIN,
                       "duration_max": DURATION_MAX, "max_seed_notes": MAX_SEED_NOTES},
            "mood_hints": MOOD_HINTS,
            "text_keywords": keywords,
        }
        if meta:
            payload["model"] = {
                "genres": meta["genres"], "moods": meta["moods"],
                "combo_counts": meta["combo_counts"], "files_per_genre": meta["files_per_genre"],
                "training": meta["training"], "n_parameters": meta.get("n_parameters"),
                "architecture": meta.get("architecture"), "mood_method": meta.get("mood_method"),
            }
        return jsonify(payload)

    @app.post("/api/generate")
    def generate():
        model, meta, error = models.get()
        if model is None:
            return jsonify({"error": error}), 503
        data = request.get_json(silent=True)
        clean, errors = validate_request(data, meta)
        if errors:
            return jsonify({"error": "Please fix the highlighted settings.", "fields": errors}), 400
        job = jobs.submit(clean)
        return jsonify({"job_id": job.id}), 202

    @app.get("/api/jobs/<job_id>")
    def job_status(job_id: str):
        job = jobs.get(job_id)
        if job is None:
            return jsonify({"error": "Unknown job. The server may have been restarted."}), 404
        body = {"status": job.status, "stage": job.stage, "progress": round(job.progress, 3)}
        if job.status == "error":
            body["error"] = job.error
        if job.status == "done":
            body["result"] = job.result
        return jsonify(body)

    def _track_dir(track_id: str):
        if not TRACK_ID_RE.match(track_id):
            return None
        d = out_root / track_id
        return d if d.is_dir() else None

    @app.get("/api/tracks")
    def list_tracks():
        items = []
        for d in out_root.glob("*/track.json"):
            try:
                t = json.loads(d.read_text(encoding="utf-8"))
                items.append({"id": t["id"], "created_at": t["created_at"], "params": t["params"],
                              "stats": t["stats"]})
            except (OSError, ValueError, KeyError):
                continue
        items.sort(key=lambda t: t["created_at"], reverse=True)
        return jsonify(items[:20])

    @app.get("/api/tracks/<track_id>")
    def get_track(track_id: str):
        d = _track_dir(track_id)
        if d is None or not (d / "track.json").exists():
            return jsonify({"error": "Track not found (it may have been deleted)."}), 404
        return app.response_class((d / "track.json").read_text(encoding="utf-8"),
                                  mimetype="application/json")

    def _file_name(d: Path, ext: str) -> str:
        try:
            p = json.loads((d / "track.json").read_text(encoding="utf-8"))["params"]
            return f"ai-music-{p['genre']}-{p['mood']}-{p['bpm']}bpm-{p['instrument']}.{ext}"
        except (OSError, ValueError, KeyError):
            return f"ai-music.{ext}"

    @app.get("/api/tracks/<track_id>/<kind>")
    def get_file(track_id: str, kind: str):
        d = _track_dir(track_id)
        if d is None or kind not in ("midi", "wav"):
            return jsonify({"error": "File not found."}), 404
        ext = "mid" if kind == "midi" else "wav"
        path = d / f"track.{ext}"
        if not path.exists():
            return jsonify({"error": "File not found."}), 404
        mime = "audio/midi" if kind == "midi" else "audio/wav"
        download = request.args.get("download") == "1"
        return send_file(path, mimetype=mime, as_attachment=download,
                         download_name=_file_name(d, ext), conditional=True, max_age=0)

    @app.errorhandler(404)
    def not_found(_e):
        if request.path.startswith("/api/"):
            return jsonify({"error": "Not found."}), 404
        return "Page not found", 404

    @app.errorhandler(500)
    def server_error(_e):
        return jsonify({"error": "Internal server error. See the console window for details."}), 500

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="AI Music Studio web app")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()
    app = create_app()
    _model, meta, error = app.extensions["aimusic_models"].get()
    print("=" * 66)
    print(" AI Music Studio")
    if meta:
        t = meta["training"]
        print(f" Model loaded: genres={meta['genres']} moods={meta['moods']}")
        print(f" Trained for {t['steps']} steps / {t['minutes']} min on {t['dataset_files']} MIDI files")
    else:
        print(" NO TRAINED MODEL YET:")
        for line in (error or "").splitlines():
            print("   " + line)
    print(f"\n Open  http://{args.host}:{args.port}  in your browser (Ctrl+C to stop)")
    print("=" * 66)
    app.run(host=args.host, port=args.port, threaded=True, debug=False)


if __name__ == "__main__":
    main()

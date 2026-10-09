# AI Music Studio (Internship Task 3 – Music Generation with AI)
Flask web app + a genuine genre-conditioned **LSTM (PyTorch, CPU)** trained on MIDI data. Windows 11, Python 3.12.

## Quick start (Windows 11)
1. Install Python 3.12 (tick "Add to PATH"). Unzip the project.
2. Double-click **run.bat**. First run: creates `.venv`, installs packages (torch is ~200 MB), downloads MAESTRO MIDI (~57 MB), **trains the model** (CPU, roughly 10–40 min; lower with `python -m app.train --epochs 3 --max-files 60`), then opens http://127.0.0.1:5000.
3. Later runs skip training because `models\model.pt` exists.

## Workflow
setup → `python -m app.prepare_data` (download) → `app/preprocess.py` (MIDI → 16th-note tokens, windows of 64) → `python -m app.train` (LSTM, saves `models/model.pt`) → Studio: **Generate** → sampled tokens → MIDI (`midi_io.py`) + WAV (`synth.py`) → Play/Pause/Stop, Download MIDI/WAV.

## Adding more genres
`python -m app.prepare_data --add-folder C:\my_jazz_midis --genre jazz`, then delete `models\model.pt` and retrain. Genres in the dropdown are exactly those the model was trained on. **Out of the box only "classical" (piano) exists** – MAESTRO is classical piano; I could not verify a legal download URL for other styles, so you must supply them.

## What is genuinely supported
| Control | How |
|---|---|
| Genre | learned genre embedding (only trained genres) |
| Tempo / Duration | steps = duration×BPM/60×4; changes note count, MIDI tempo and audio length |
| Starting notes | validated (C4, F#3, Bb4), fed to the LSTM as a prompt, then it continues |
| Instrument | MIDI program + numpy synth timbre (piano, guitar, violin, pad). Timbre only – the model trains on melody, not instruments |
| Mood | **not learned**: maps to sampling temperature, velocity and note length |
| Text instructions | **not understood**; only keywords calm/happy/sad/energetic set mood (a warning is shown otherwise) |

## Limitations
Monophonic melody (highest note) model; short-range structure; synthesized audio is simple, not a sampled instrument. Quality depends on training time.

## Tests
`python -m unittest tests.test_all` (MIDI round trip, token/note parsing, synth, preprocessing, Flask routes).

## Troubleshooting
- *Model not trained* pill: run `python -m app.train`.
- Download failed: get the zip from the URL in `app/prepare_data.py`, put `.mid` files in `data\midi\classical`.
- torch install error: confirm 64-bit Python 3.12 (`py -3.12 --version`).
- Port busy: edit port in `app/server.py`.

## Viva questions
1. *Why LSTM?* Handles sequences with memory via gates, avoiding vanishing gradients of plain RNNs.
2. *How is music tokenized?* One token per 16th note: rest, hold, or one of 88 pitches.
3. *How does tempo affect output?* Tokens are tempo-independent; BPM and duration set the number of steps and the rendering speed.
4. *How is variety created?* Temperature + top-12 sampling.
5. *Is mood learned?* No – it adjusts sampling/velocity/gate; stated openly.
6. *Loss?* Cross-entropy on next-token prediction.
7. *Limitations?* See above.

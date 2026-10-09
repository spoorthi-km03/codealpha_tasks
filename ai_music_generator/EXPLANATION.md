# Explanation
**Architecture:** `MusicLSTM` = token Embedding(64) ‖ genre Embedding(16) → 2-layer LSTM(256, dropout .2) → Linear(90). Trained with Adam + cross-entropy, gradient clipping, on windows of 64 tokens predicting the next token at every position.
**Files:** `midi_io.py` pure-Python MIDI parse/write; `tokens.py` vocabulary, note-name parsing, token↔note; `preprocess.py` dataset windows; `train.py`; `generate.py` loading, prompt seeding, temperature/top-k sampling, export; `synth.py` additive synthesis to WAV; `server.py` Flask API (`/api/status`, `/api/generate`, `/outputs/*`); `static/app.js` UI logic and a real waveform drawn from the decoded generated WAV.
**Generation:** seed notes (each one beat) prime the LSTM; it then samples one token per 16th step until duration×BPM is reached. Tokens → notes → `.mid` and `.wav`.
**Honesty:** mood and text are not learned conditioning (see README table). Errors return HTTP 400/503 and show in the UI; success is never faked.

# College FAQ Assistant (Task 2 – FAQ Chatbot)

A full-page Flask chatbot that answers college FAQs using **NLTK preprocessing + TF-IDF + cosine similarity**. No paid APIs, no database.

> The college name, phone numbers, fees and dates in `faq_data.json` are sample data. Edit that file to match your college.

## Run (Windows 11)
1. Install Python 3.12 from python.org (tick **Add Python to PATH**).
2. Unzip the project and double-click **run.bat** (needs internet the first time).
3. It creates `venv`, installs requirements, downloads NLTK data, starts Flask and opens http://127.0.0.1:5000.

Manual: `py -3 -m venv venv`, `venv\Scripts\activate`, `pip install -r requirements.txt`, `python app.py`.
Tests: `python -m unittest discover tests`.

## Structure
```
faq_chatbot/
├── app.py            Flask routes (/ and /chat)
├── chatbot.py        TF-IDF + cosine similarity matching, threshold
├── preprocess.py     NLTK preprocessing (auto-downloads resources)
├── faq_data.json     48 FAQs in 10 categories
├── run.bat  requirements.txt  .gitignore  EXPLANATION.md
├── templates/index.html
├── static/css/style.css, static/js/script.js
└── tests/test_chatbot.py
```

## How it works
User question → preprocess → TF-IDF vector → cosine similarity with every FAQ question → best score. If score ≥ 0.25 the FAQ's answer is shown; otherwise a friendly "couldn't find a relevant answer" message.

- **Preprocessing:** lowercase → remove punctuation → tokenize (`word_tokenize`) → remove stopwords → lemmatize (`WordNetLemmatizer`). Example: "What ARE the Library Timings?!" → `library timing`.
- **TF-IDF:** turns text into numbers. Words frequent in one FAQ but rare overall get high weight. Unigrams and bigrams are used.
- **Cosine similarity:** the cosine of the angle between two vectors; 1 = identical direction, 0 = nothing in common.
- **Threshold:** 0.25 (`THRESHOLD` in `chatbot.py`). Raise it for stricter matching.

## Task 2 requirements
| Requirement | Where |
|---|---|
| Collect FAQs | `faq_data.json` (48 Q&A pairs) |
| Preprocess with NLTK | `preprocess.py` |
| Cosine similarity matching | `chatbot.py` |
| Display best answer | `/chat` + `script.js` |
| Chat UI | `index.html`, `style.css`, `script.js` |

## Verification status
Tested in the build environment: FAQ loading, matching on exact/rephrased/uppercase/unknown/empty inputs, the `/chat` endpoint (including invalid input), page and static-file serving, and JavaScript syntax. **Not tested there:** real NLTK (no internet in the build sandbox; a simple stand-in was used), `run.bat` on Windows, and visual/browser behaviour (Send, Enter, Clear, suggestion buttons). Please run `run.bat` and the tests once on your machine.

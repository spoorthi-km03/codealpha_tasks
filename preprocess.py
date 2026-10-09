"""Text preprocessing: lowercase, punctuation cleaning, tokenization, stopword removal, lemmatization."""
import re
import nltk

for _res in ("punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"):
    try:
        nltk.download(_res, quiet=True)  # skips quickly if already present
    except Exception:
        pass

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

_lemmatizer = WordNetLemmatizer()
_stop = set(stopwords.words("english"))


def preprocess(text: str) -> str:
    """Return the cleaned text as a single space-joined string of lemmas."""
    if not isinstance(text, str):
        return ""
    text = text.lower()                          # 1. lowercase
    text = re.sub(r"[^a-z0-9\s]", " ", text)     # 2. punctuation cleaning
    tokens = word_tokenize(text)                 # 3. tokenization
    tokens = [t for t in tokens if t not in _stop]            # 4. stopword removal
    tokens = [_lemmatizer.lemmatize(_lemmatizer.lemmatize(t), "v") for t in tokens]  # 5. lemmatization
    return " ".join(tokens)

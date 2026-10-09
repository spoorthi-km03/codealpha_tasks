"""FAQ matching with TF-IDF + cosine similarity."""
import json
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from preprocess import preprocess

FAQ_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "faq_data.json")
THRESHOLD = 0.25
FALLBACK = ("Sorry, I couldn't find a relevant answer to that. Try rephrasing, "
            "or pick one of the suggested questions. You can also contact the office at info@gitcollege.edu.")


class FAQChatbot:
    def __init__(self, path=FAQ_PATH, threshold=THRESHOLD):
        with open(path, encoding="utf-8") as f:
            self.faqs = json.load(f)
        self.threshold = threshold
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2))
        self.matrix = self.vectorizer.fit_transform([preprocess(f["question"]) for f in self.faqs])

    def get_answer(self, user_question):
        if not isinstance(user_question, str) or not user_question.strip():
            return {"answer": "Please type a question so I can help.", "matched": False, "score": 0.0}
        cleaned = preprocess(user_question)
        if not cleaned:
            return {"answer": FALLBACK, "matched": False, "score": 0.0}
        scores = cosine_similarity(self.vectorizer.transform([cleaned]), self.matrix)[0]
        best = int(scores.argmax())
        score = float(scores[best])
        if score < self.threshold:
            return {"answer": FALLBACK, "matched": False, "score": round(score, 3)}
        faq = self.faqs[best]
        return {"answer": faq["answer"], "matched": True, "score": round(score, 3),
                "question": faq["question"], "category": faq["category"]}

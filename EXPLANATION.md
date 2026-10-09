# Beginner Explanation & Viva Q&A

**What does the project do?** You type a college question; the bot finds the most similar stored FAQ and shows its answer.

**Q: What is NLTK?** A Python library for working with human language (tokenizing, stopwords, lemmatizing). `preprocess.py` downloads its data automatically.
**Q: Why preprocess?** So "Fees?", "fees" and "FEES!!" look the same to the computer, and filler words don't affect matching.
**Q: Stopwords?** Common words like "the", "is", "what" that carry little meaning; we remove them.
**Q: Tokenization?** Splitting a sentence into words.
**Q: Lemmatization?** Reducing words to their dictionary form (timings → timing, applying → apply).
**Q: What is TF-IDF?** Term Frequency × Inverse Document Frequency. It gives high weight to words that are important in one FAQ but rare in others.
**Q: What is cosine similarity?** The cosine of the angle between two TF-IDF vectors, from 0 (nothing in common) to 1 (same). It ignores text length.
**Q: Why a threshold?** Without it the bot would always return *some* answer, even for "who won the cricket match?". Below 0.25 it says it couldn't find an answer.
**Q: Why Flask?** A lightweight Python web framework; it serves the page and the `/chat` route.
**Q: How do frontend and backend communicate?** JavaScript `fetch()` sends JSON `{"message": "..."}` to `/chat` by POST; Flask returns JSON `{"answer": ..., "matched": ..., "score": ...}`; JavaScript adds it to the chat without reloading (AJAX).
**Q: Limitations?** It matches words, not meaning (no synonyms), and only knows the FAQs in `faq_data.json`. Improvement: add more question variants or use sentence embeddings.
**Q: How do I add an FAQ?** Add an object with category, question and answer to `faq_data.json` and restart.

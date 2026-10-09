import os, sys, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocess import preprocess
from chatbot import FAQChatbot
from app import app


class TestAll(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bot = FAQChatbot()
        cls.client = app.test_client()

    def test_preprocess(self):
        out = preprocess("What ARE the Library Timings?!")
        self.assertEqual(out, out.lower())
        self.assertNotIn("?", out)
        self.assertNotIn("the", out.split())
        self.assertIn("library", out)

    def test_preprocess_invalid(self):
        self.assertEqual(preprocess(None), "")

    def test_faq_count(self):
        self.assertGreaterEqual(len(self.bot.faqs), 40)

    def test_exact_all(self):
        for f in self.bot.faqs:
            r = self.bot.get_answer(f["question"])
            self.assertTrue(r["matched"], f["question"])
            self.assertEqual(r["answer"], f["answer"], f["question"])

    def test_rephrased(self):
        cases = {"how can i apply to the college": "Admission", "tell me about hostel fees": "Hostel",
                 "library opening timings": "Library", "which companies come for placement": "Placements",
                 "any scholarship for students": "Scholarships", "minimum attendance needed": "Attendance",
                 "what is the phone number to contact": "Contact"}
        for q, cat in cases.items():
            r = self.bot.get_answer(q)
            self.assertTrue(r["matched"], q)
            self.assertEqual(r["category"], cat, q)

    def test_capitalization(self):
        a = self.bot.get_answer("WHAT ARE THE LIBRARY TIMINGS?")
        b = self.bot.get_answer("what are the library timings?")
        self.assertEqual(a["answer"], b["answer"])

    def test_unknown(self):
        for q in ["what is the capital of france", "asdfgh qwerty", "tell me a joke", "???"]:
            self.assertFalse(self.bot.get_answer(q)["matched"], q)

    def test_empty(self):
        self.assertFalse(self.bot.get_answer("   ")["matched"])
        self.assertFalse(self.bot.get_answer(None)["matched"])

    def test_endpoint(self):
        r = self.client.post("/chat", json={"message": "What are the library timings?"})
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.get_json()["matched"])
        self.assertEqual(self.client.post("/chat", json={"message": "  "}).status_code, 400)
        self.assertEqual(self.client.post("/chat", json={"x": 1}).status_code, 400)
        self.assertEqual(self.client.post("/chat", data="not json").status_code, 400)
        self.assertEqual(self.client.post("/chat", json={"message": "a" * 600}).status_code, 400)
        self.assertEqual(self.client.post("/chat", json={"message": "zzzz unknown"}).status_code, 200)

    def test_pages(self):
        self.assertEqual(self.client.get("/").status_code, 200)
        for p in ["css/style.css", "js/script.js"]:
            self.assertEqual(self.client.get("/static/" + p).status_code, 200)


if __name__ == "__main__":
    unittest.main()

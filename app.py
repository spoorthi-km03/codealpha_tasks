from flask import Flask, jsonify, render_template, request
from chatbot import FAQChatbot

app = Flask(__name__)
bot = FAQChatbot()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True)
    if not isinstance(data, dict) or not isinstance(data.get("message"), str):
        return jsonify({"answer": "Invalid request. Please send a text message.", "matched": False}), 400
    message = data["message"].strip()
    if not message:
        return jsonify({"answer": "Please type a question so I can help.", "matched": False}), 400
    if len(message) > 500:
        return jsonify({"answer": "That message is too long. Please keep it under 500 characters.", "matched": False}), 400
    return jsonify(bot.get_answer(message))


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)

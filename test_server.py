# A PRETEND chat server, for testing your Chat page on your own laptop.
# It returns the same JSON shape as the real /ask API, but with fake answers.
# Run it with:  python test_server.py
from flask import Flask, request, jsonify

app = Flask(__name__)


@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "TEST RAG QA Chatbot API is running!"})


@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json(silent=True) or {}
    question = data.get("question", "")
    if not isinstance(question, str) or not question.strip():
        return jsonify({"error": "Please provide a non-empty question."}), 400
    question = question.strip()
    return jsonify({
        "question": question,
        "answer": f"(Test answer) You asked: '{question}'. The real server will give a real answer here.",
        "retrieved_chunks": [
            {"chunk_position": 12, "section_id": "1.2", "score": 0.8123,
             "text": "This is a pretend chunk of text from the document."},
            {"chunk_position": 40, "section_id": "2.1", "score": 0.7011,
             "text": "This is another pretend chunk, to test how sources are shown."},
        ],
    })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)

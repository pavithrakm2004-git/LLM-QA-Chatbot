
from flask import Flask, request, jsonify
from pathlib import Path
import json
import re
import requests
import numpy as np
import faiss

app = Flask(__name__)

# Project configuration
PROJECT_DIR = Path(__file__).resolve().parent

CHUNKS_PATH = PROJECT_DIR / "section_chunks.json"
INDEX_PATH = PROJECT_DIR / "faiss_section_cosine_index.bin"

OLLAMA_URL = "http://localhost:11434"
EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3.1:8b-instruct-fp16"

# Load chunks and FAISS index
with open(CHUNKS_PATH, "r", encoding="utf-8") as file:
    chunks = json.load(file)

index = faiss.read_index(str(INDEX_PATH))

if len(chunks) != index.ntotal:
    raise ValueError(
        f"Chunk/index mismatch: {len(chunks)} chunks, "
        f"{index.ntotal} indexed vectors."
    )


def tokenize(text):
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    return re.findall(r"\b[a-z0-9]+\b", text.lower())


def get_query_embedding(question):
    response = requests.post(
        f"{OLLAMA_URL}/api/embed",
        json={
            "model": EMBEDDING_MODEL,
            "input": question
        },
        timeout=60
    )
    response.raise_for_status()

    embedding = np.array(
        response.json()["embeddings"][0],
        dtype="float32"
    ).reshape(1, -1)

    faiss.normalize_L2(embedding)
    return embedding


def retrieve_hybrid(question, candidate_k=20, final_k=5):
    query_embedding = get_query_embedding(question)

    candidate_count = min(candidate_k, index.ntotal)
    scores, indices = index.search(
        query_embedding, candidate_count
    )

    query_words = set(tokenize(question))
    candidates = []

    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue

        chunk_text = chunks[int(idx)]["text"]
        chunk_words = set(tokenize(chunk_text))

        keyword_score = (
            len(query_words & chunk_words) / len(query_words)
            if query_words else 0.0
        )

        combined_score = (
            0.6 * float(score) + 0.4 * keyword_score
        )

        candidates.append({
            "chunk_position": int(idx),
            "section_id": chunks[int(idx)].get("section_id"),
            "semantic_score": float(score),
            "keyword_score": keyword_score,
            "combined_score": combined_score,
            "text": chunk_text
        })

    candidates.sort(
        key=lambda item: item["combined_score"],
        reverse=True
    )

    return candidates[:final_k]


def generate_answer(question, retrieved_results):
    context = "\n\n".join(
        item["text"] for item in retrieved_results
    )

    prompt = f"""
You are a question-answering assistant.
Answer using only the provided context.
Do not add unsupported facts.
If the context does not contain the answer, say:
"The provided document does not contain enough information."

Context:
{context}

Question: {question}

Answer:
"""

    response = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": LLM_MODEL,
            "prompt": prompt,
            "stream": False
        },
        timeout=180
    )
    response.raise_for_status()

    return response.json()["response"].strip()


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "RAG QA Chatbot API is running!"
    })


@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json(silent=True) or {}
    question = data.get("question", "")

    if not isinstance(question, str) or not question.strip():
        return jsonify({
            "error": "Please provide a non-empty question."
        }), 400

    try:
        results = retrieve_hybrid(question.strip())
        answer = generate_answer(question.strip(), results)

        return jsonify({
            "question": question.strip(),
            "answer": answer,
            "retrieved_chunks": [
                {
                    "chunk_position": item["chunk_position"],
                    "section_id": item["section_id"],
                    "score": round(item["combined_score"], 4),
                    "text": item["text"]
                }
                for item in results
            ]
        })

    except requests.RequestException as error:
        app.logger.exception("Ollama request failed")
        return jsonify({
            "error": "Could not connect to Ollama or generate an answer.",
            "details": str(error)
        }), 503

    except Exception:
        app.logger.exception("RAG request failed")
        return jsonify({
            "error": "An internal error occurred while processing the question."
        }), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)

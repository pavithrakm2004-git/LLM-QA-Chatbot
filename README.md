# LLM-Based Question Answering Chatbot

An LLM-based Question Answering Chatbot that uses Retrieval-Augmented Generation (RAG) to answer user questions from PDF documents. The system retrieves relevant information from the uploaded document and uses a Large Language Model (LLM) to generate context-aware answers.

## Project Overview

This project implements a document-based question-answering system using a Retrieval-Augmented Generation (RAG) architecture.

The project consists of two major components:

- **Backend – Pavithra K M:** PDF processing, text extraction, section-aware chunking, embedding generation, FAISS indexing, hybrid retrieval, and answer generation using Ollama.
- **Frontend – Manasa Muraleedharan E:** Streamlit-based user interface for interacting with the question-answering chatbot.

The frontend communicates with the backend through a Flask API.

## System Architecture

```text
             USER
               |
               v
      STREAMLIT FRONTEND
          (Manasa)
               |
               | HTTP Request
               v
         FLASK BACKEND
          (Pavithra)
               |
               v
       QUESTION PROCESSING
               |
               v
        HYBRID RETRIEVAL
         /           \
        v             v
   FAISS INDEX   KEYWORD SCORING
         \           /
          v         v
       RELEVANT DOCUMENT
            CHUNKS
               |
               v
        OLLAMA LLM
       (Llama 3.1)
               |
               v
       GENERATED ANSWER
               |
               v
       STREAMLIT FRONTEND
               |
               v
             USER
```

## Backend – Pavithra K M

The backend implements the RAG pipeline and handles document-based question answering.

### Backend Workflow

1. **PDF Text Extraction:** Extracts text from PDF documents using PyPDF.
2. **Section-Aware Chunking:** Divides the extracted text into smaller chunks while preserving section information.
3. **Embedding Generation:** Generates vector embeddings using `nomic-embed-text`.
4. **Vector Indexing:** Stores embeddings in a FAISS index for similarity-based retrieval.
5. **Hybrid Retrieval:** Combines semantic similarity and keyword-based scoring to retrieve relevant document chunks.
6. **Answer Generation:** Sends the retrieved context to the Llama 3.1 model through Ollama to generate answers.

### Backend Technologies

- Python
- Flask
- FAISS
- NumPy
- PyPDF
- Ollama
- Llama 3.1 (8B)
- nomic-embed-text

### Backend Files

| File                             | Description                         |
| -------------------------------- | ----------------------------------- |
| `app.py`                         | Flask backend API                   |
| `Clean_CODE.ipynb`               | Cleaned RAG implementation notebook |
| `requirements.txt`               | Python dependencies                 |
| `section_chunks.json`            | Processed document chunks           |
| `faiss_section_cosine_index.bin` | FAISS vector index                  |
| `test_document.pdf`              | Test document                       |

| Manasa Muraleedharan E | Frontend development, Streamlit web application, dataset preparation, human evaluation, metrics, deployment and backend API integration |
### Frontend Responsibilities

- Provides a Streamlit-based chatbot interface.
- Accepts questions from users.
- Sends questions to the Flask backend through HTTP requests.
- Receives generated answers from the backend.
- Displays responses through the user interface.
### Frontend Technologies

- Python
- Streamlit
- Requests
- Pandas


**Note:** The frontend source code is maintained separately by the frontend developer and should be added to this repository as part of the shared project.

## Technologies Used

| Component            | Technology       |
| -------------------- | ---------------- |
| Frontend             | Streamlit        |
| Backend              | Flask            |
| LLM Runtime          | Ollama           |
| Language Model       | Llama 3.1        |
| Embedding Model      | nomic-embed-text |
| Vector Search        | FAISS            |
| Programming Language | Python           |

## Backend Setup and Execution

### 1. Clone the Repository

```bash
git clone https://github.com/pavithrakm2004-git/LLM-QA-Chatbot.git
cd LLM-QA-Chatbot
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it:

**Windows**

```bash
venv\Scripts\activate
```

**Linux / macOS**

```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Install Ollama Models

Install Ollama from [https://ollama.com/](https://ollama.com/)

Pull the required models:

```bash
ollama pull nomic-embed-text
ollama pull llama3.1:8b-instruct-fp16
```

Ensure Ollama is running before starting the backend.

### 5. Start the Flask Backend

```bash
python app.py
```

The backend runs at:

```text
http://localhost:5000
```

The question-answering API endpoint is:

```text
POST /ask
```

## Frontend and Backend Integration

The Streamlit frontend communicates with the Flask backend through the `/ask` API endpoint.

The frontend must send requests to the backend's accessible URL.

**Important:** `localhost:5000` is accessible only from the machine running the backend. Other users cannot access your DGX backend through their own `localhost`.

For remote access and public usage, the Flask backend must be deployed or made accessible through an appropriate network configuration.

## Test Document and Attribution

The project was tested using the following textbook:

**Title:** Introduction to Artificial Intelligence

**Institution:** University of California, Berkeley – CS 188

**Authors:** Nikhil Sharma, Josh Hug, Jacky Liang, and Henry Zhu.

**Source:**
[https://inst.eecs.berkeley.edu/\~cs188/textbook/textbook_full.pdf](https://inst.eecs.berkeley.edu/~cs188/textbook/textbook_full.pdf)

**License:** Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0).

[https://creativecommons.org/licenses/by-sa/4.0/](https://creativecommons.org/licenses/by-sa/4.0/)

The textbook is an externally authored resource used for testing. It is not created or owned by the project team.

## Authors and Contributions

| Team Member  | Contribution                                                                                            |
| ------------ | ------------------------------------------------------------------------------------------------------- |
| Pavithra K M | Backend development, RAG pipeline, document processing, hybrid retrieval, Flask API and LLM integration |
| Manasa Muraleedharan E       | Frontend development, Streamlit interface and backend API integration                                   |

## Project Repository

GitHub: [https://github.com/pavithrakm2004-git/LLM-QA-Chatbot](https://github.com/pavithrakm2004-git/LLM-QA-Chatbot)



(The file `c:\\Users\\aditi\\Downloads\\Chat_With_Document-demo\\README.md` exists, but is empty)
# Chat With Document — RAG demo

A small Flask demo showing Retrieval-Augmented Generation (RAG): upload PDF documents, create embeddings and a FAISS index, then ask questions and get answers grounded in your documents.

This project demonstrates a practical RAG pipeline combining:
- PyPDFLoader (PDF -> documents)
- RecursiveCharacterTextSplitter (documents -> chunks)
- HuggingFace local embeddings (sentence-transformers/all-MiniLM-L6-v2)
- FAISS vector store (in-memory index)
- LangChain retrieval chain + a ChatGroq LLM for answers

📌 Note: The front-end UI is a simple Flask app (templates/index.html) that lets you upload PDFs and ask questions.

---

## Quick start (Windows PowerShell)

1. Open PowerShell, then change into the demo directory:

```powershell
cd Chat_With_Document-demo
```

2. (Recommended) create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

3. Install dependencies:

```powershell
pip install -r requirements.txt
```

4. Create a `.env` file in the same folder as `app.py` containing at least the GROQ API key (required for the ChatGroq model):

```
GROQ_API_KEY=your_groq_api_key_here
# Optional: override Flask session secret key
SECRET_KEY=your_secret_here
```

5. Run the app:

```powershell
python app.py
```

6. Open your browser at http://127.0.0.1:5000 — upload PDF(s) and ask questions.

---

## API endpoints & usage

- POST /upload — multipart form upload using `files[]` for one or more PDFs. Files are saved under `uploaded_documents/` and processed into a FAISS index. Example using curl:

```bash
curl -X POST -F "files[]=@/path/to/your.pdf" http://127.0.0.1:5000/upload
```

- POST /query — form-encoded field `query`. It returns JSON with `answer`, `response_time`, and `context_snippets` which show the chunks retrieved from the documents.

```bash
curl -X POST -d "query=What are the main goals described in the document?" http://127.0.0.1:5000/query
```

---

## Architecture / how RAG is implemented

1. The app loads PDF files using `PyPDFLoader` (paginated documents).
2. Documents are split into chunks with `RecursiveCharacterTextSplitter` (chunk_size=800, overlap=100).
3. Each chunk is embedded using `sentence-transformers/all-MiniLM-L6-v2` (HuggingFaceEmbeddings).
4. A FAISS in-memory vector index is created from the embeddings.
5. For each user query the code builds a retriever (k=6), runs a retrieval chain to gather context, and invokes the `ChatGroq` model to generate an answer constrained to the supplied context via a `ChatPromptTemplate`.

This ensures answers are grounded in the uploaded documents, and the prompt explicitly instructs the model to respond with "I could not find this information in the documents." when the content is missing.

---

## Limits & notes
- Supported upload type: PDF only (file extension validation in `app.py`).
- Upload size limit is 16MB by default (configured in `app.config['MAX_CONTENT_LENGTH']`).
- FAISS is created in-memory and attached to `app.config['vectors']`; the demo does not persist vectors on disk — re-processing may be needed across restarts.
- The demo uses a Groq model (via `langchain_groq.ChatGroq`) — you must provide a valid `GROQ_API_KEY`.
- HuggingFace embeddings are local and don't require a key, but they will download model weights on first run.

---

## Troubleshooting
- If installation of `faiss-cpu` fails on Windows, try installing a compatible pre-built wheel for your Python version and platform, or use a platform that supports faiss easily (Linux/macOS or appropriate binary).
- If the UI shows "No documents processed yet", make sure your PDF upload succeeded and check the Flask logs for exceptions.

---

## Next steps / enhancements you can make
- Persist FAISS indexes between restarts (e.g., save to disk and load on startup).
- Add support for Word/Markdown/TXT file loaders.
- Add async background processing for uploads and indexing.
- Add per-document provenance in responses (link answers to document name + page number).

---

If you'd like, I can also add a sample test PDF, or wire up persistent storage for FAISS so indexes survive restarts — tell me which next and I’ll implement it.

Enjoy! ✅

# Doctor Assistant (Local RAG)

A local doctor-assistant chatbot that answers questions using only the provided JSON files. It uses Milvus for vector storage, sentence-transformers for embeddings, and a small local Transformers model for generation.

## Project Structure

```
Doctor Chatbot/
  backend/
    app/
      rag/
        data_loader.py
        embedding.py
        llm.py
        milvus_store.py
        prompt.py
        rag_pipeline.py
        types.py
      utils/
        text.py
      config.py
      main.py
    scripts/
      ingest.py
    requirements.txt
  frontend/
    app.js
    index.html
    style.css
  sample_data/
    doc.json
    doc_clean.json
    drug.json
    drug_clean.json
    lab.json
    lab_clean.json
    nurse.json
    nurse_clean.json
    xray.json
    xray_clean.json
  docker-compose.yml
  .env.example
  .gitignore
  README.md
```

## Quick Start

1. Start Milvus:

```bash
docker compose up -d
```

2. Create a Python environment and install dependencies:

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r backend\requirements.txt
```

3. Ingest the JSON files:

```bash
python -m backend.scripts.ingest
```

4. Run the API and UI:

```bash
uvicorn backend.app.main:app --reload --port 8000
```

Open http://localhost:8000 in your browser.

## API Endpoints

- `GET /health` -> basic health check
- `POST /chat` -> chat with the RAG pipeline
- `POST /ingest` -> re-ingest data from `sample_data/`

### Sample Chat Request

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the latest diagnosis for patient an1?"}'
```

## Notes

- The chatbot only answers using the loaded JSON data.
- If the data does not contain the answer, the chatbot responds with "I do not know" and a disclaimer.
- Update `.env` from `.env.example` to adjust models, limits, or paths.
- The first run downloads models from Hugging Face; set `LLM_DEVICE=cuda` to use a GPU if available.

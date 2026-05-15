from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pymilvus.exceptions import MilvusException

from backend.app.config import get_settings
from backend.app.rag.rag_pipeline import RAGPipeline
from backend.app.rag.data_loader import load_documents
from backend.app.rag.embedding import embed_texts
from backend.app.rag.milvus_store import MilvusStore


class ChatRequest(BaseModel):
    question: str
    top_k: Optional[int] = None


class ChatResponse(BaseModel):
    answer: str


class IngestResponse(BaseModel):
    inserted: int


settings = get_settings()
app = FastAPI(title="Doctor Assistant")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

rag = RAGPipeline()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    try:
        answer = rag.answer(request.question, request.top_k)
        return ChatResponse(answer=answer)
    except MilvusException as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Milvus is unavailable. Start it with `docker compose up -d` "
                "and retry your request."
            ),
        ) from exc


@app.post("/ingest", response_model=IngestResponse)
def ingest() -> IngestResponse:
    data_dir = Path(settings.data_dir)
    docs = load_documents(data_dir, settings.max_text_chars)
    if not docs:
        return IngestResponse(inserted=0)
    embeddings = embed_texts([doc.text for doc in docs])
    store = MilvusStore()
    try:
        store.connect()
        inserted = store.insert(docs, embeddings)
        return IngestResponse(inserted=inserted)
    except MilvusException as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Milvus is unavailable. Start it with `docker compose up -d` "
                "and retry ingestion."
            ),
        ) from exc


frontend_dir = Path(__file__).resolve().parents[2] / "frontend"
app.mount("/assets", StaticFiles(directory=frontend_dir, html=True), name="frontend")


@app.get("/")
def root() -> FileResponse:
    return FileResponse(frontend_dir / "index.html")

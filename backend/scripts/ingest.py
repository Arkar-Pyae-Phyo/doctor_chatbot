from pathlib import Path

from backend.app.config import get_settings
from backend.app.rag.data_loader import load_documents
from backend.app.rag.embedding import embed_texts
from backend.app.rag.milvus_store import MilvusStore


def main() -> None:
    settings = get_settings()
    data_dir = Path(settings.data_dir)
    docs = load_documents(data_dir, settings.max_text_chars)
    if not docs:
        print("No documents found.")
        return
    embeddings = embed_texts([doc.text for doc in docs])
    store = MilvusStore()
    store.connect()
    inserted = store.insert(docs, embeddings)
    print(f"Inserted {inserted} records into Milvus.")


if __name__ == "__main__":
    main()

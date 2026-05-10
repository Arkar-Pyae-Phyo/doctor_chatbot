from functools import lru_cache
from typing import List

from sentence_transformers import SentenceTransformer

from backend.app.config import get_settings


@lru_cache(maxsize=1)
def get_embedder() -> SentenceTransformer:
    settings = get_settings()
    return SentenceTransformer(settings.embedding_model)


def embed_texts(texts: List[str]) -> List[List[float]]:
    model = get_embedder()
    embeddings = model.encode(texts, normalize_embeddings=True)
    return embeddings.tolist()

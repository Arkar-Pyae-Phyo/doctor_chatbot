from typing import Iterable, List, Optional

from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections, utility

from backend.app.config import get_settings
from backend.app.rag.types import DocumentChunk, RetrievedChunk


class MilvusStore:
    def __init__(self) -> None:
        self.settings = get_settings()
        self._collection: Optional[Collection] = None

    def connect(self) -> None:
        connections.connect(host=self.settings.milvus_host, port=str(self.settings.milvus_port))

    def _create_collection(self, dim: int) -> Collection:
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="patient_id", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="timestamp", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=8192),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
        ]
        schema = CollectionSchema(fields=fields, description="Doctor assistant RAG store")
        collection = Collection(self.settings.collection_name, schema)
        index_params = {"index_type": "IVF_FLAT", "metric_type": "IP", "params": {"nlist": 128}}
        collection.create_index(field_name="embedding", index_params=index_params)
        collection.load()
        return collection

    def get_collection(self, dim: int) -> Collection:
        if self._collection is not None:
            return self._collection
        if not utility.has_collection(self.settings.collection_name):
            self._collection = self._create_collection(dim)
        else:
            collection = Collection(self.settings.collection_name)
            collection.load()
            self._collection = collection
        return self._collection

    def insert(self, docs: Iterable[DocumentChunk], embeddings: List[List[float]]) -> int:
        docs_list = list(docs)
        if not docs_list:
            return 0
        dim = len(embeddings[0])
        collection = self.get_collection(dim)
        sources = [doc.source for doc in docs_list]
        patient_ids = [doc.patient_id or "" for doc in docs_list]
        timestamps = [doc.timestamp or "" for doc in docs_list]
        texts = [doc.text for doc in docs_list]
        data = [sources, patient_ids, timestamps, texts, embeddings]
        result = collection.insert(data)
        collection.flush()
        return len(result.primary_keys)

    def search(self, query_embedding: List[float], top_k: int) -> List[RetrievedChunk]:
        collection = self.get_collection(len(query_embedding))
        search_params = {"metric_type": "IP", "params": {"nprobe": 10}}
        results = collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=["source", "patient_id", "timestamp", "text"],
        )
        matches: List[RetrievedChunk] = []
        for hit in results[0]:
            matches.append(
                RetrievedChunk(
                    source=hit.entity.get("source"),
                    patient_id=hit.entity.get("patient_id") or None,
                    timestamp=hit.entity.get("timestamp") or None,
                    text=hit.entity.get("text"),
                    score=float(hit.score),
                    metadata={},
                )
            )
        return matches

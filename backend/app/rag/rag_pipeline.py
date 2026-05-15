from typing import List, Optional

from backend.app.config import get_settings
from backend.app.rag.embedding import embed_texts
from backend.app.rag.llm import TransformersClient
from backend.app.rag.milvus_store import MilvusStore
from backend.app.rag.prompt import SYSTEM_PROMPT, build_prompt
from backend.app.rag.types import RetrievedChunk
from backend.app.utils.text import join_lines


class RAGPipeline:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.milvus = MilvusStore()
        self.llm = TransformersClient()
        self._disclaimer = (
            "This response is for informational support only and is not a substitute for professional medical "
            "judgment, diagnosis, or treatment."
        )

    def retrieve(self, question: str, top_k: Optional[int] = None) -> List[RetrievedChunk]:
        self.milvus.connect()
        embedding = embed_texts([question])[0]
        limit = top_k or self.settings.top_k
        return self.milvus.search(embedding, limit)

    def _build_context(self, chunks: List[RetrievedChunk]) -> str:
        lines: List[str] = []
        for chunk in chunks:
            header = f"Source: {chunk.source}"
            if chunk.patient_id:
                header += f" | Patient: {chunk.patient_id}"
            if chunk.timestamp:
                header += f" | Timestamp: {chunk.timestamp}"
            lines.append(header)
            lines.append(chunk.text)
            lines.append("")
        context = join_lines(lines)
        if len(context) > self.settings.max_context_chars:
            context = context[: self.settings.max_context_chars]
        return context

    def answer(self, question: str, top_k: Optional[int] = None) -> str:
        chunks = self.retrieve(question, top_k)
        if not chunks:
            return f"I do not know.\n\n{self._disclaimer}"
        context = self._build_context(chunks)
        user_prompt = build_prompt(context, question)
        answer = self.llm.chat(SYSTEM_PROMPT, user_prompt)
        if self._disclaimer not in answer:
            answer = f"{answer}\n\n{self._disclaimer}"
        return answer

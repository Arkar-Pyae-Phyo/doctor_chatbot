import os


class Settings:
    def __init__(self) -> None:
        self.milvus_host = os.getenv("MILVUS_HOST", "localhost")
        self.milvus_port = int(os.getenv("MILVUS_PORT", "19530"))
        self.collection_name = os.getenv("MILVUS_COLLECTION", "doctor_assistant")
        self.embedding_model = os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )
        self.llm_model = os.getenv("LLM_MODEL", "google/flan-t5-small")
        self.llm_device = os.getenv("LLM_DEVICE", "cpu")
        self.max_new_tokens = int(os.getenv("MAX_NEW_TOKENS", "256"))
        self.temperature = float(os.getenv("TEMPERATURE", "0.2"))
        self.data_dir = os.getenv("DATA_DIR", "sample_data")
        self.top_k = int(os.getenv("TOP_K", "5"))
        self.max_context_chars = int(os.getenv("MAX_CONTEXT_CHARS", "7000"))
        self.max_text_chars = int(os.getenv("MAX_TEXT_CHARS", "4000"))


def get_settings() -> Settings:
    return Settings()

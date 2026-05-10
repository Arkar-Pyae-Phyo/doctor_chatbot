import os


class Settings:
    def __init__(self) -> None:
        self.milvus_host = os.getenv("MILVUS_HOST", "localhost")
        self.milvus_port = int(os.getenv("MILVUS_PORT", "19530"))
        self.collection_name = os.getenv("MILVUS_COLLECTION", "doctor_assistant")
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.ollama_model = os.getenv("OLLAMA_MODEL", "llama3")
        self.data_dir = os.getenv("DATA_DIR", "sample_data")
        self.top_k = int(os.getenv("TOP_K", "5"))
        self.max_context_chars = int(os.getenv("MAX_CONTEXT_CHARS", "7000"))
        self.max_text_chars = int(os.getenv("MAX_TEXT_CHARS", "4000"))
        self.request_timeout = int(os.getenv("REQUEST_TIMEOUT", "60"))


def get_settings() -> Settings:
    return Settings()

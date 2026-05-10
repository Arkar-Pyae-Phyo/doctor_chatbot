import requests

from backend.app.config import get_settings


class OllamaClient:
    def __init__(self) -> None:
        self.settings = get_settings()

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        url = f"{self.settings.ollama_base_url}/api/chat"
        payload = {
            "model": self.settings.ollama_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
        }
        response = requests.post(url, json=payload, timeout=self.settings.request_timeout)
        response.raise_for_status()
        data = response.json()
        return data.get("message", {}).get("content", "").strip()

from functools import lru_cache

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

from backend.app.config import get_settings


@lru_cache(maxsize=1)
def get_text_generator():
    settings = get_settings()
    tokenizer = AutoTokenizer.from_pretrained(settings.llm_model)
    model = AutoModelForSeq2SeqLM.from_pretrained(settings.llm_model)
    device = -1
    if settings.llm_device.lower() == "cuda" and torch.cuda.is_available():
        device = 0
    return pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        device=device,
    )


class TransformersClient:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.generator = get_text_generator()

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        prompt = f"{system_prompt}\n\n{user_prompt}"
        do_sample = self.settings.temperature > 0
        outputs = self.generator(
            prompt,
            max_new_tokens=self.settings.max_new_tokens,
            do_sample=do_sample,
            temperature=self.settings.temperature,
        )
        return outputs[0]["generated_text"].strip()

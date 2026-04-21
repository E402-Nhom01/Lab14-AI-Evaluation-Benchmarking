import os
import json
import requests
from abc import ABC, abstractmethod
from openai import OpenAI


# =========================
# Base LLM Interface
# =========================
class BaseLLM(ABC):
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        pass

    def chat(self, prompt: str, **kwargs) -> str:
        return self.generate(prompt, **kwargs)


# =========================
# GPT-4o-mini (OpenAI)
# =========================
class GPT4oMiniLLM(BaseLLM):
    def __init__(self, model_name: str = "gpt-4o-mini"):
        api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            raise ValueError("Missing OPENAI_API_KEY environment variable")

        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name

    def generate(self, prompt: str, **kwargs) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant for RAG-based question answering.",
                },
                {"role": "user", "content": prompt},
            ],
        )

        return response.choices[0].message.content


# =========================
# Ollama LLM (local option)
# =========================
class OllamaLLM(BaseLLM):
    def __init__(self, model_name: str, base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url

    def generate(self, prompt: str, **kwargs) -> str:
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
        }

        response = requests.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=60,
        )

        response.raise_for_status()

        return response.json()["response"]
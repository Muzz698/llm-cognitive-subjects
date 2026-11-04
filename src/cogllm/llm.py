"""Minimal client for a local Ollama server (https://ollama.com). Free, offline,
and the same model + seed gives the same answer, which keeps runs reproducible."""
from __future__ import annotations

import requests


class OllamaClient:
    def __init__(self, model: str, host: str = "http://localhost:11434",
                 temperature: float = 1.0, max_tokens: int = 5, timeout: float = 120):
        self.model = model
        self.host = host.rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

    def generate(self, prompt: str, seed: int = 0, max_tokens: int | None = None) -> str:
        response = requests.post(
            f"{self.host}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "raw": False,
                "options": {
                    "temperature": self.temperature,
                    "seed": int(seed),
                    "num_predict": max_tokens or self.max_tokens,
                },
            },
            timeout=self.timeout,
        )
        if response.status_code == 404:
            raise RuntimeError(f"Model '{self.model}' not found. Run: ollama pull {self.model}")
        response.raise_for_status()
        return response.json()["response"]

    def check(self) -> None:
        """Fail early with a clear message if Ollama is not running."""
        try:
            requests.get(f"{self.host}/api/tags", timeout=5).raise_for_status()
        except requests.RequestException as err:
            raise RuntimeError(
                "Cannot reach Ollama. Install it from https://ollama.com, open the app, "
                f"then run: ollama pull {self.model}"
            ) from err

"""Thin wrapper over the local Ollama server."""
from __future__ import annotations

import ollama


class LLM:
    def __init__(self, host: str = "http://localhost:11434"):
        self._client = ollama.Client(host=host)

    def available_models(self) -> list[str]:
        try:
            resp = self._client.list()
            return [m.get("model") or m.get("name") for m in resp.get("models", [])]
        except Exception:
            return []

    def generate(
        self,
        model: str,
        system: str,
        prompt: str,
        temperature: float = 0.8,
    ) -> str:
        resp = self._client.chat(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            options={"temperature": temperature},
        )
        return resp["message"]["content"].strip()

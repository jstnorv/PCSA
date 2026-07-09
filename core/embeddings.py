from __future__ import annotations

from typing import Sequence

import requests


class OllamaEmbeddingClient:
	"""Small client for local embedding generation via Ollama HTTP API."""

	def __init__(self, base_url: str, model: str, timeout_seconds: int = 60) -> None:
		self.base_url = base_url.rstrip("/")
		self.model = model
		self.timeout_seconds = timeout_seconds
		self._session = requests.Session()

	def _post_json(self, endpoint: str, payload: dict) -> dict:
		response = self._session.post(
			f"{self.base_url}{endpoint}",
			json=payload,
			timeout=self.timeout_seconds,
		)
		response.raise_for_status()
		return response.json()

	def embed_text(self, text: str) -> list[float]:
		payload = {"model": self.model, "prompt": text}

		try:
			data = self._post_json("/api/embeddings", payload)
			embedding = data.get("embedding")
			if isinstance(embedding, list):
				return embedding
		except requests.RequestException:
			# Fallback to the newer embed endpoint if available.
			pass

		data = self._post_json("/api/embed", {"model": self.model, "input": text})
		embeddings = data.get("embeddings")
		if isinstance(embeddings, list) and embeddings:
			first = embeddings[0]
			if isinstance(first, list):
				return first

		raise RuntimeError("Ollama embedding endpoint returned an unexpected payload.")

	def embed_many(self, texts: Sequence[str]) -> list[list[float]]:
		return [self.embed_text(text) for text in texts]

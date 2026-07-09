from __future__ import annotations

import requests


class OllamaClient:
	"""Direct local Ollama API client without orchestration frameworks."""

	def __init__(self, base_url: str, model: str, timeout_seconds: int = 60) -> None:
		self.base_url = base_url.rstrip("/")
		self.model = model
		self.timeout_seconds = timeout_seconds
		self._session = requests.Session()

	def health_check(self) -> bool:
		try:
			response = self._session.get(
				f"{self.base_url}/api/tags",
				timeout=self.timeout_seconds,
			)
			response.raise_for_status()
			return True
		except requests.RequestException:
			return False

	def generate(self, prompt: str) -> str:
		payload = {
			"model": self.model,
			"prompt": prompt,
			"stream": False,
		}
		response = self._session.post(
			f"{self.base_url}/api/generate",
			json=payload,
			timeout=self.timeout_seconds,
		)
		response.raise_for_status()
		data = response.json()
		text = data.get("response")
		if not isinstance(text, str):
			raise RuntimeError("Unexpected Ollama response payload.")
		return text.strip()

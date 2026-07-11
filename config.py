from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, Field, HttpUrl


class AppConfig(BaseModel):
	"""Central local-first configuration for the PCSA runtime."""

	project_root: Path = Field(default_factory=lambda: Path(__file__).resolve().parent)
	knowledge_base_dir: Path = Field(default=Path("storage/knowledge_base"))
	vector_db_dir: Path = Field(default=Path("storage/vector_db"))

	ollama_base_url: HttpUrl = Field(default="http://localhost:11434")
	ollama_chat_model: str = Field(default="llama3.1")
	ollama_embedding_model: str = Field(default="nomic-embed-text")
	ollama_timeout_seconds: int = Field(default=60, ge=1)
	chunk_size_chars: int = Field(default=1200, ge=200)
	chunk_overlap_chars: int = Field(default=200, ge=0)

	memory_window_size: int = Field(default=12, ge=2)
	retrieval_top_k: int = Field(default=4, ge=1)

	def resolve_paths(self) -> "AppConfig":
		"""Normalize relative paths against project root and create missing dirs."""
		if not self.knowledge_base_dir.is_absolute():
			self.knowledge_base_dir = self.project_root / self.knowledge_base_dir
		if not self.vector_db_dir.is_absolute():
			self.vector_db_dir = self.project_root / self.vector_db_dir

		self.knowledge_base_dir.mkdir(parents=True, exist_ok=True)
		self.vector_db_dir.mkdir(parents=True, exist_ok=True)
		return self

	@classmethod
	def from_env(cls) -> "AppConfig":
		"""Load config from environment with strict local-first defaults."""
		instance = cls(
			project_root=Path(os.getenv("PCSA_PROJECT_ROOT", Path(__file__).resolve().parent)),
			knowledge_base_dir=Path(os.getenv("PCSA_KNOWLEDGE_BASE_DIR", "storage/knowledge_base")),
			vector_db_dir=Path(os.getenv("PCSA_VECTOR_DB_DIR", "storage/vector_db")),
			ollama_base_url=os.getenv("PCSA_OLLAMA_BASE_URL", "http://localhost:11434"),
			ollama_chat_model=os.getenv("PCSA_OLLAMA_CHAT_MODEL", "llama3.1"),
			ollama_embedding_model=os.getenv("PCSA_OLLAMA_EMBED_MODEL", "nomic-embed-text"),
			ollama_timeout_seconds=int(os.getenv("PCSA_OLLAMA_TIMEOUT_SECONDS", "60")),
			chunk_size_chars=int(os.getenv("PCSA_CHUNK_SIZE_CHARS", "1200")),
			chunk_overlap_chars=int(os.getenv("PCSA_CHUNK_OVERLAP_CHARS", "200")),
			memory_window_size=int(os.getenv("PCSA_MEMORY_WINDOW_SIZE", "12")),
			retrieval_top_k=int(os.getenv("PCSA_RETRIEVAL_TOP_K", "4")),
		)
		if instance.chunk_overlap_chars >= instance.chunk_size_chars:
			raise ValueError("PCSA_CHUNK_OVERLAP_CHARS must be smaller than PCSA_CHUNK_SIZE_CHARS")
		return instance.resolve_paths()


def get_config() -> AppConfig:
	return AppConfig.from_env()

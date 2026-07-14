from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, Field, HttpUrl


class AppConfig(BaseModel):
	"""Central local-first configuration for the PCSA runtime."""

	project_root: Path = Field(default_factory=lambda: Path(__file__).resolve().parent)
	knowledge_base_dir: Path = Field(default=Path("storage/knowledge_base"))
	vector_db_dir: Path = Field(default=Path("storage/vector_db"))
	session_state_path: Path = Field(default=Path("storage/session_state.json"))
	setup_profile_path: Path = Field(default=Path("storage/setup_profile.json"))

	ollama_base_url: HttpUrl = Field(default="http://localhost:11434")
	ollama_chat_model: str = Field(default="llama3.1")
	ollama_embedding_model: str = Field(default="nomic-embed-text")
	ollama_timeout_seconds: int = Field(default=60, ge=1)
	chunk_size_chars: int = Field(default=1200, ge=200)
	chunk_overlap_chars: int = Field(default=200, ge=0)

	memory_window_size: int = Field(default=12, ge=2)
	retrieval_top_k: int = Field(default=4, ge=1)

	delegation_confidence_threshold: float = Field(default=0.68, ge=0.0, le=1.0)
	delegation_complexity_threshold: float = Field(default=0.72, ge=0.0, le=1.0)
	strict_local_only: bool = Field(default=False)

	user_privacy_sensitivity: str = Field(default="high")
	user_latency_tolerance_ms: int = Field(default=12000, ge=500)
	user_local_memory_budget_gb: float = Field(default=8.0, ge=1.0)
	user_compute_budget: str = Field(default="balanced")

	def resolve_paths(self) -> "AppConfig":
		"""Normalize relative paths against project root and create missing dirs."""
		if not self.knowledge_base_dir.is_absolute():
			self.knowledge_base_dir = self.project_root / self.knowledge_base_dir
		if not self.vector_db_dir.is_absolute():
			self.vector_db_dir = self.project_root / self.vector_db_dir
		if not self.session_state_path.is_absolute():
			self.session_state_path = self.project_root / self.session_state_path
		if not self.setup_profile_path.is_absolute():
			self.setup_profile_path = self.project_root / self.setup_profile_path

		self.knowledge_base_dir.mkdir(parents=True, exist_ok=True)
		self.vector_db_dir.mkdir(parents=True, exist_ok=True)
		self.session_state_path.parent.mkdir(parents=True, exist_ok=True)
		self.setup_profile_path.parent.mkdir(parents=True, exist_ok=True)
		return self

	@classmethod
	def from_env(cls) -> "AppConfig":
		"""Load config from environment with strict local-first defaults."""
		instance = cls(
			project_root=Path(os.getenv("PCSA_PROJECT_ROOT", Path(__file__).resolve().parent)),
			knowledge_base_dir=Path(os.getenv("PCSA_KNOWLEDGE_BASE_DIR", "storage/knowledge_base")),
			vector_db_dir=Path(os.getenv("PCSA_VECTOR_DB_DIR", "storage/vector_db")),
			session_state_path=Path(os.getenv("PCSA_SESSION_STATE_PATH", "storage/session_state.json")),
			setup_profile_path=Path(os.getenv("PCSA_SETUP_PROFILE_PATH", "storage/setup_profile.json")),
			ollama_base_url=os.getenv("PCSA_OLLAMA_BASE_URL", "http://localhost:11434"),
			ollama_chat_model=os.getenv("PCSA_OLLAMA_CHAT_MODEL", "llama3.1"),
			ollama_embedding_model=os.getenv("PCSA_OLLAMA_EMBED_MODEL", "nomic-embed-text"),
			ollama_timeout_seconds=int(os.getenv("PCSA_OLLAMA_TIMEOUT_SECONDS", "60")),
			chunk_size_chars=int(os.getenv("PCSA_CHUNK_SIZE_CHARS", "1200")),
			chunk_overlap_chars=int(os.getenv("PCSA_CHUNK_OVERLAP_CHARS", "200")),
			memory_window_size=int(os.getenv("PCSA_MEMORY_WINDOW_SIZE", "12")),
			retrieval_top_k=int(os.getenv("PCSA_RETRIEVAL_TOP_K", "4")),
			delegation_confidence_threshold=float(os.getenv("PCSA_DELEGATION_CONFIDENCE_THRESHOLD", "0.68")),
			delegation_complexity_threshold=float(os.getenv("PCSA_DELEGATION_COMPLEXITY_THRESHOLD", "0.72")),
			strict_local_only=os.getenv("PCSA_STRICT_LOCAL_ONLY", "false").strip().lower() in {"1", "true", "yes", "on"},
			user_privacy_sensitivity=os.getenv("PCSA_USER_PRIVACY_SENSITIVITY", "high"),
			user_latency_tolerance_ms=int(os.getenv("PCSA_USER_LATENCY_TOLERANCE_MS", "12000")),
			user_local_memory_budget_gb=float(os.getenv("PCSA_USER_LOCAL_MEMORY_BUDGET_GB", "8.0")),
			user_compute_budget=os.getenv("PCSA_USER_COMPUTE_BUDGET", "balanced"),
		)
		if instance.chunk_overlap_chars >= instance.chunk_size_chars:
			raise ValueError("PCSA_CHUNK_OVERLAP_CHARS must be smaller than PCSA_CHUNK_SIZE_CHARS")
		return instance.resolve_paths()


def get_config() -> AppConfig:
	return AppConfig.from_env()

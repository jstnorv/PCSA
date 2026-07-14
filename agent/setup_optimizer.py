from __future__ import annotations

import json
import os
import platform
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import requests

from config import AppConfig


@dataclass
class LocalCapabilityProfile:
	cpu_count: int
	total_memory_gb: float
	platform_name: str
	platform_version: str
	ollama_reachable: bool
	available_models: list[str] = field(default_factory=list)


@dataclass
class SetupOptimizationProfile:
	created_at: str
	limitations: dict[str, object]
	capabilities: dict[str, object]
	applied_config: dict[str, object]
	notes: list[str] = field(default_factory=list)


class LocalSetupOptimizer:
	"""Compute local-first runtime settings bounded by user limitations."""

	def detect_capabilities(self, config: AppConfig) -> LocalCapabilityProfile:
		cpu_count = os.cpu_count() or 1
		total_memory_gb = self._detect_total_memory_gb()
		ollama_reachable, available_models = self._probe_ollama_models(
			base_url=str(config.ollama_base_url),
			timeout_seconds=config.ollama_timeout_seconds,
		)
		return LocalCapabilityProfile(
			cpu_count=cpu_count,
			total_memory_gb=total_memory_gb,
			platform_name=platform.system(),
			platform_version=platform.version(),
			ollama_reachable=ollama_reachable,
			available_models=available_models,
		)

	def optimize(self, config: AppConfig) -> tuple[AppConfig, SetupOptimizationProfile]:
		capability = self.detect_capabilities(config)
		notes: list[str] = []

		allowed_memory_gb = min(config.user_local_memory_budget_gb, capability.total_memory_gb)
		if allowed_memory_gb <= 4.0:
			config.memory_window_size = min(config.memory_window_size, 8)
			config.retrieval_top_k = min(config.retrieval_top_k, 3)
			config.chunk_size_chars = min(config.chunk_size_chars, 900)
			notes.append("Memory-constrained profile applied.")
		elif allowed_memory_gb >= 12.0 and config.user_compute_budget.lower() == "high":
			config.memory_window_size = max(config.memory_window_size, 14)
			config.retrieval_top_k = max(config.retrieval_top_k, 5)
			config.chunk_size_chars = max(config.chunk_size_chars, 1300)
			notes.append("High-compute local profile applied.")

		if config.user_latency_tolerance_ms <= 4000:
			config.retrieval_top_k = min(config.retrieval_top_k, 3)
			config.ollama_timeout_seconds = min(config.ollama_timeout_seconds, 35)
			config.memory_window_size = min(config.memory_window_size, 10)
			notes.append("Low-latency profile applied.")

		privacy = config.user_privacy_sensitivity.strip().lower()
		if privacy in {"strict-local", "strict_local", "strict"}:
			config.strict_local_only = True
			notes.append("Strict local privacy policy enforced.")

		if capability.available_models:
			resolved_chat_model = self._select_model(
				configured_model=config.ollama_chat_model,
				available_models=capability.available_models,
				preferred=["llama3.1", "llama3.2", "qwen2.5", "mistral"],
			)
			if resolved_chat_model != config.ollama_chat_model:
				notes.append(
					f"Adjusted chat model from {config.ollama_chat_model} to {resolved_chat_model} based on local availability."
				)
				config.ollama_chat_model = resolved_chat_model

			resolved_embedding_model = self._select_model(
				configured_model=config.ollama_embedding_model,
				available_models=capability.available_models,
				preferred=["nomic-embed-text", "mxbai-embed-large", "all-minilm"],
			)
			if resolved_embedding_model != config.ollama_embedding_model:
				notes.append(
					"Adjusted embedding model to locally available option "
					f"{resolved_embedding_model}."
				)
				config.ollama_embedding_model = resolved_embedding_model

		config.chunk_overlap_chars = min(config.chunk_overlap_chars, max(100, config.chunk_size_chars // 4))
		if config.chunk_overlap_chars >= config.chunk_size_chars:
			config.chunk_overlap_chars = max(0, config.chunk_size_chars - 1)

		limitations = {
			"privacy_sensitivity": config.user_privacy_sensitivity,
			"latency_tolerance_ms": config.user_latency_tolerance_ms,
			"local_memory_budget_gb": config.user_local_memory_budget_gb,
			"compute_budget": config.user_compute_budget,
			"strict_local_only": config.strict_local_only,
		}
		capabilities = {
			"cpu_count": capability.cpu_count,
			"total_memory_gb": capability.total_memory_gb,
			"platform": capability.platform_name,
			"platform_version": capability.platform_version,
			"ollama_reachable": capability.ollama_reachable,
			"available_models": capability.available_models,
		}
		applied_config = {
			"ollama_chat_model": config.ollama_chat_model,
			"ollama_embedding_model": config.ollama_embedding_model,
			"ollama_timeout_seconds": config.ollama_timeout_seconds,
			"chunk_size_chars": config.chunk_size_chars,
			"chunk_overlap_chars": config.chunk_overlap_chars,
			"memory_window_size": config.memory_window_size,
			"retrieval_top_k": config.retrieval_top_k,
			"delegation_confidence_threshold": config.delegation_confidence_threshold,
			"delegation_complexity_threshold": config.delegation_complexity_threshold,
			"strict_local_only": config.strict_local_only,
		}
		profile = SetupOptimizationProfile(
			created_at=datetime.now(UTC).isoformat(),
			limitations=limitations,
			capabilities=capabilities,
			applied_config=applied_config,
			notes=notes,
		)
		return config, profile

	def save_profile(self, profile: SetupOptimizationProfile, output_path: Path) -> None:
		output_path.parent.mkdir(parents=True, exist_ok=True)
		payload = {
			"created_at": profile.created_at,
			"limitations": profile.limitations,
			"capabilities": profile.capabilities,
			"applied_config": profile.applied_config,
			"notes": profile.notes,
		}
		output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

	def _probe_ollama_models(self, base_url: str, timeout_seconds: int) -> tuple[bool, list[str]]:
		url = f"{base_url.rstrip('/')}/api/tags"
		try:
			response = requests.get(url, timeout=timeout_seconds)
			response.raise_for_status()
			data = response.json()
			models = data.get("models", [])
			if not isinstance(models, list):
				return True, []

			names: list[str] = []
			for model in models:
				if not isinstance(model, dict):
					continue
				name = model.get("name")
				if isinstance(name, str) and name.strip():
					names.append(name.strip())
			return True, names
		except requests.RequestException:
			return False, []

	def _select_model(self, configured_model: str, available_models: list[str], preferred: list[str]) -> str:
		if configured_model in available_models:
			return configured_model

		for pref in preferred:
			for candidate in available_models:
				if candidate.startswith(pref):
					return candidate
		return configured_model

	def _detect_total_memory_gb(self) -> float:
		try:
			if hasattr(os, "sysconf") and "SC_PAGE_SIZE" in os.sysconf_names and "SC_PHYS_PAGES" in os.sysconf_names:
				pages = os.sysconf("SC_PHYS_PAGES")
				page_size = os.sysconf("SC_PAGE_SIZE")
				if isinstance(pages, int) and isinstance(page_size, int) and pages > 0 and page_size > 0:
					return round((pages * page_size) / (1024 ** 3), 2)
		except Exception:
			pass
		return 8.0

from __future__ import annotations

import json
import re
import uuid
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

if TYPE_CHECKING:
	from llm.ollama_client import OllamaClient


class PriorityLevel(str, Enum):
	LOW = "low"
	MEDIUM = "medium"
	HIGH = "high"


class SessionState(BaseModel):
	"""Foundational session state for scoped, persistent agent behavior.

	The required fields for v1 are:
	- current_objective
	- priority_level
	- active_namespaces
	- user_intent_notes

	Additional fields are included to keep the schema extensible as the
	partnership evolves.
	"""

	state_schema_version: str = Field(default="1.0")
	session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
	created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
	updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

	current_objective: str = Field(default="Clarify objective for this session.")
	priority_level: PriorityLevel = Field(default=PriorityLevel.MEDIUM)
	active_namespaces: list[str] = Field(default_factory=list)
	user_intent_notes: list[str] = Field(default_factory=list)

	intent_category: str = Field(default="general")
	extraction_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
	metadata: dict[str, Any] = Field(default_factory=dict)

	def touch(self) -> None:
		self.updated_at = datetime.now(UTC)


def _extract_first_json_object(raw_text: str) -> dict[str, Any]:
	"""Parse the first JSON object from raw model output.

	Supports plain JSON and markdown code-fenced JSON.
	"""
	text = raw_text.strip()

	if text.startswith("```"):
		code_blocks = re.findall(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
		if code_blocks:
			text = code_blocks[0].strip()

	try:
		parsed = json.loads(text)
		if isinstance(parsed, dict):
			return parsed
	except json.JSONDecodeError:
		pass

	# Fallback: find the first {...} region that parses as JSON.
	for start in range(len(text)):
		if text[start] != "{":
			continue
		for end in range(len(text), start + 1, -1):
			if text[end - 1] != "}":
				continue
			candidate = text[start:end]
			try:
				parsed = json.loads(candidate)
				if isinstance(parsed, dict):
					return parsed
			except json.JSONDecodeError:
				continue

	raise ValueError("No JSON object found in extraction output.")


def _normalize_namespaces(
	raw_namespaces: list[Any] | None,
	available_namespaces: list[str],
) -> list[str]:
	if not raw_namespaces:
		return []

	available_lookup = {ns.lower(): ns for ns in available_namespaces}
	normalized: list[str] = []
	for item in raw_namespaces:
		if not isinstance(item, str):
			continue
		value = item.strip()
		if not value:
			continue

		if available_lookup:
			resolved = available_lookup.get(value.lower())
			if resolved and resolved not in normalized:
				normalized.append(resolved)
		elif value not in normalized:
			normalized.append(value)

	return normalized


def _fallback_initialize(
	focus_prompt_output: str,
	available_namespaces: list[str],
) -> SessionState:
	text = focus_prompt_output.strip()
	first_line = text.splitlines()[0].strip() if text else ""
	objective = first_line or "Clarify objective for this session."

	lower_text = text.lower()
	priority = PriorityLevel.MEDIUM
	if any(token in lower_text for token in ["urgent", "asap", "blocker", "critical"]):
		priority = PriorityLevel.HIGH
	if any(token in lower_text for token in ["later", "eventually", "optional", "low priority"]):
		priority = PriorityLevel.LOW

	namespaces = []
	for namespace in available_namespaces:
		if namespace.lower() in lower_text:
			namespaces.append(namespace)

	notes = [line.strip("- ") for line in text.splitlines() if line.strip()]
	return SessionState(
		current_objective=objective,
		priority_level=priority,
		active_namespaces=namespaces,
		user_intent_notes=notes[:8],
		intent_category="general",
		extraction_confidence=0.2,
		metadata={"init_strategy": "fallback"},
	)


def initialize_session_state(
	focus_prompt_output: str,
	llm_client: OllamaClient,
	available_namespaces: list[str] | None = None,
) -> SessionState:
	"""Initialize foundational session state from free-form user focus text.

	The function uses LLM extraction first, then falls back to deterministic
	heuristics to guarantee a valid state object.
	"""
	available_namespaces = available_namespaces or []

	extraction_prompt = (
		"You are an intent extraction engine for a local-first AI agent. "
		"Extract structured session state from the user text. "
		"Return only valid JSON with this shape:\n"
		"{\n"
		"  \"current_objective\": string,\n"
		"  \"priority_level\": \"high\" | \"medium\" | \"low\",\n"
		"  \"active_namespaces\": string[],\n"
		"  \"user_intent_notes\": string[],\n"
		"  \"intent_category\": string,\n"
		"  \"extraction_confidence\": number\n"
		"}\n\n"
		f"Available namespaces for selection: {available_namespaces}\n\n"
		"Rules:\n"
		"1) active_namespaces must prefer exact values from available namespaces.\n"
		"2) Keep user_intent_notes concise, factual, and action-oriented.\n"
		"3) Do not include markdown or prose outside JSON.\n\n"
		f"User focus text:\n{focus_prompt_output.strip()}"
	)

	try:
		raw_response = llm_client.generate(extraction_prompt)
		payload = _extract_first_json_object(raw_response)

		current_objective = str(payload.get("current_objective", "")).strip()
		if not current_objective:
			current_objective = "Clarify objective for this session."

		priority_raw = str(payload.get("priority_level", "medium")).lower().strip()
		priority = PriorityLevel(priority_raw) if priority_raw in PriorityLevel._value2member_map_ else PriorityLevel.MEDIUM

		notes_raw = payload.get("user_intent_notes", [])
		notes = [str(item).strip() for item in notes_raw if str(item).strip()] if isinstance(notes_raw, list) else []

		namespaces = _normalize_namespaces(
			raw_namespaces=payload.get("active_namespaces") if isinstance(payload.get("active_namespaces"), list) else [],
			available_namespaces=available_namespaces,
		)

		confidence = payload.get("extraction_confidence")
		if not isinstance(confidence, (int, float)):
			confidence = None

		state = SessionState(
			current_objective=current_objective,
			priority_level=priority,
			active_namespaces=namespaces,
			user_intent_notes=notes[:10],
			intent_category=str(payload.get("intent_category", "general")).strip() or "general",
			extraction_confidence=confidence,
			metadata={"init_strategy": "llm_extraction"},
		)
		return state
	except Exception as exc:
		fallback_state = _fallback_initialize(focus_prompt_output, available_namespaces)
		fallback_state.metadata["extraction_error"] = str(exc)
		return fallback_state


def save_session_state(state: SessionState, output_path: Path) -> None:
	"""Persist state as readable JSON for local-first session continuity."""
	state.touch()
	output_path.parent.mkdir(parents=True, exist_ok=True)
	output_path.write_text(
		state.model_dump_json(indent=2),
		encoding="utf-8",
	)


def load_session_state(input_path: Path) -> SessionState:
	raw = json.loads(input_path.read_text(encoding="utf-8"))
	return SessionState.model_validate(raw)


def build_state_context_block(state: SessionState) -> str:
	"""Build a compact context block that can be injected into prompts and queries."""
	namespace_text = ", ".join(state.active_namespaces) if state.active_namespaces else "(none)"
	notes_text = "\n".join(f"- {note}" for note in state.user_intent_notes[:6])
	notes_text = notes_text if notes_text else "- (none)"

	return (
		"Session Foundation State:\n"
		f"Current Objective: {state.current_objective}\n"
		f"Priority Level: {state.priority_level.value}\n"
		f"Active Namespaces: {namespace_text}\n"
		"User Intent Notes:\n"
		f"{notes_text}"
	)


def build_scoped_rag_query(user_query: str, state: SessionState | None) -> str:
	"""Wrap a user query with session state so retrieval is objective-aware by default."""
	if state is None:
		return user_query.strip()

	context_block = build_state_context_block(state)
	return (
		f"{context_block}\n\n"
		"Retrieval Query:\n"
		f"{user_query.strip()}"
	)


def build_scoped_system_prompt(base_system_prompt: str, state: SessionState | None) -> str:
	"""Prepend foundational state to the system prompt used for generation."""
	if state is None:
		return base_system_prompt

	context_block = build_state_context_block(state)
	return f"{context_block}\n\n{base_system_prompt}"
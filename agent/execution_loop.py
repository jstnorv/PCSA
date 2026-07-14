from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Protocol

from agent.delegated_controller import (
	DelegatedComputationController,
	DelegationDecision,
	EscalationProposal,
	ExecutionRoute,
)
from agent.directives import CORE_DIRECTIVE_SYSTEM_PROMPT
from agent.memory import SlidingWindowMemory
from agent.session_state import (
	SessionState,
	build_scoped_rag_query,
	build_scoped_system_prompt,
)
from agent.state_machine import AgentState, AgentStateMachine
from core.embeddings import OllamaEmbeddingClient
from core.vector_store import ChromaVectorStore
from llm.ollama_client import OllamaClient


class ExternalInferenceClient(Protocol):
	def generate(self, prompt: str) -> str: ...


@dataclass
class ControlledTurnResult:
	answer: str
	local_answer: str
	decision: DelegationDecision
	escalated: bool

	@property
	def requires_confirmation(self) -> bool:
		return self.decision.requires_confirmation and self.decision.route == ExecutionRoute.EXTERNAL

	@property
	def escalation_proposal(self) -> EscalationProposal | None:
		return self.decision.proposal


class PCSAgent:
	"""Minimal local-first execution loop for retrieval-augmented responses."""

	def __init__(
		self,
		llm_client: OllamaClient,
		embedding_client: OllamaEmbeddingClient,
		vector_store: ChromaVectorStore,
		memory: SlidingWindowMemory,
		retrieval_top_k: int = 4,
		session_state: SessionState | None = None,
		external_client: ExternalInferenceClient | None = None,
		delegated_controller: DelegatedComputationController | None = None,
		failure_cooldown_seconds: int = 30,
	) -> None:
		self.llm_client = llm_client
		self.embedding_client = embedding_client
		self.vector_store = vector_store
		self.memory = memory
		self.retrieval_top_k = retrieval_top_k
		self.session_state = session_state
		self.external_client = external_client
		self.delegated_controller = delegated_controller or DelegatedComputationController()
		self.failure_cooldown_seconds = max(1, failure_cooldown_seconds)
		self._retrieval_blocked_until: datetime | None = None
		self._llm_blocked_until: datetime | None = None
		self.state_machine = AgentStateMachine()

	def _in_cooldown(self, blocked_until: datetime | None) -> bool:
		return blocked_until is not None and datetime.now(UTC) < blocked_until

	def _mark_retrieval_failure(self) -> None:
		self._retrieval_blocked_until = datetime.now(UTC) + timedelta(seconds=self.failure_cooldown_seconds)

	def _mark_llm_failure(self) -> None:
		self._llm_blocked_until = datetime.now(UTC) + timedelta(seconds=self.failure_cooldown_seconds)

	def _deterministic_fallback_answer(self, user_input: str, contexts: list[str], reason: str) -> str:
		if contexts:
			snippets = []
			for idx, ctx in enumerate(contexts[:2], start=1):
				first_line = ctx.strip().splitlines()[0].strip() if ctx.strip() else "(empty context)"
				snippets.append(f"{idx}. {first_line[:220]}")
			snippet_block = "\n".join(snippets)
			return (
				"Operating in degraded local mode.\n"
				f"Reason: {reason}.\n"
				"Best available context summary:\n"
				f"{snippet_block}\n\n"
				f"Request captured: {user_input}"
			)

		return (
			"Operating in degraded local mode with no retrievable context.\n"
			f"Reason: {reason}.\n"
			f"Request captured: {user_input}\n"
			"Suggestion: keep your request concise or retry when local model service is responsive."
		)

	def _retrieve_contexts(self, user_input: str) -> list[str]:
		self.state_machine.transition(AgentState.RETRIEVE_CONTEXT)
		if self._in_cooldown(self._retrieval_blocked_until):
			return []
		raw_query = user_input.strip()
		scoped_query = build_scoped_rag_query(raw_query, self.session_state)
		try:
			query_embedding = self.embedding_client.embed_text(scoped_query)
			return self.vector_store.query(query_embedding, top_k=self.retrieval_top_k)
		except Exception:
			self._mark_retrieval_failure()
			return []

	def _generate_local_answer(self, user_input: str, contexts: list[str]) -> str:
		self.state_machine.transition(AgentState.INFER)
		if self._in_cooldown(self._llm_blocked_until):
			return self._deterministic_fallback_answer(
				user_input=user_input,
				contexts=contexts,
				reason="local inference temporarily paused after recent failure",
			)
		context_block = "\n\n".join(contexts) if contexts else "(no retrieved context)"
		base_system_prompt = CORE_DIRECTIVE_SYSTEM_PROMPT
		system_prompt = build_scoped_system_prompt(base_system_prompt, self.session_state)
		prompt = (
			f"{system_prompt}"
			f"Conversation Memory:\n{self.memory.render()}\n\n"
			f"Retrieved Context:\n{context_block}\n\n"
			f"User Input:\n{user_input}\n\n"
			"Assistant Response:"
		)
		try:
			return self.llm_client.generate(prompt)
		except Exception:
			self._mark_llm_failure()
			return self._deterministic_fallback_answer(
				user_input=user_input,
				contexts=contexts,
				reason="local inference endpoint unavailable or too slow",
			)

	def _persist_turn(self, user_input: str, answer: str) -> None:
		self.state_machine.transition(AgentState.PERSIST)
		self.memory.add("user", user_input)
		self.memory.add("assistant", answer)
		self.state_machine.transition(AgentState.IDLE)

	def _run_external_or_fallback(self, user_input: str, local_answer: str) -> str:
		if self.external_client is None:
			return (
				f"{local_answer}\n\n"
				"[Delegation approved but no external compute gateway is configured. Returning local result.]"
			)
		return self.external_client.generate(user_input)

	def run_turn(self, user_input: str) -> str:
		contexts = self._retrieve_contexts(user_input)
		answer = self._generate_local_answer(user_input, contexts)
		self._persist_turn(user_input, answer)
		return answer

	def run_turn_with_delegation(
		self,
		user_input: str,
		force_external: bool = False,
	) -> ControlledTurnResult:
		contexts = self._retrieve_contexts(user_input)
		local_answer = self._generate_local_answer(user_input, contexts)

		decision = self.delegated_controller.decide(
			user_query=user_input,
			retrieved_contexts=contexts,
			local_response=local_answer,
			session_state=self.session_state,
		)

		if force_external:
			answer = self._run_external_or_fallback(user_input, local_answer)
			self._persist_turn(user_input, answer)
			return ControlledTurnResult(
				answer=answer,
				local_answer=local_answer,
				decision=decision,
				escalated=True,
			)

		if decision.route == ExecutionRoute.EXTERNAL and decision.requires_confirmation:
			# Do not persist until explicit user approval/denial is captured.
			self.state_machine.transition(AgentState.IDLE)
			return ControlledTurnResult(
				answer=local_answer,
				local_answer=local_answer,
				decision=decision,
				escalated=False,
			)

		if decision.route == ExecutionRoute.EXTERNAL:
			answer = self._run_external_or_fallback(user_input, local_answer)
			self._persist_turn(user_input, answer)
			return ControlledTurnResult(
				answer=answer,
				local_answer=local_answer,
				decision=decision,
				escalated=True,
			)

		self._persist_turn(user_input, local_answer)
		return ControlledTurnResult(
			answer=local_answer,
			local_answer=local_answer,
			decision=decision,
			escalated=False,
		)

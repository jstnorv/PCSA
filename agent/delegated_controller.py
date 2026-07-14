from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from agent.session_state import SessionState


class ExecutionRoute(str, Enum):
	LOCAL = "local"
	EXTERNAL = "external"


@dataclass
class EscalationProposal:
	"""User-visible proposal shown before any external request executes."""

	reason: str
	metadata_summary: str
	confirmation_prompt: str


@dataclass
class DelegationDecision:
	route: ExecutionRoute
	requires_confirmation: bool
	proposal: EscalationProposal | None
	local_confidence: float
	complexity_score: float


class DelegatedComputationController:
	"""Local-first trust gateway for delegated computation decisions."""

	def __init__(
		self,
		confidence_threshold: float = 0.68,
		complexity_threshold: float = 0.72,
		external_delegation_enabled: bool = True,
	) -> None:
		self.confidence_threshold = confidence_threshold
		self.complexity_threshold = complexity_threshold
		self.external_delegation_enabled = external_delegation_enabled

	def decide(
		self,
		user_query: str,
		retrieved_contexts: list[str],
		local_response: str,
		session_state: SessionState | None,
	) -> DelegationDecision:
		local_confidence = self._estimate_local_confidence(local_response, retrieved_contexts)
		complexity_score = self._estimate_task_complexity(user_query)

		needs_escalation = (
			local_confidence < self.confidence_threshold
			or complexity_score >= self.complexity_threshold
		)
		if not needs_escalation:
			return DelegationDecision(
				route=ExecutionRoute.LOCAL,
				requires_confirmation=False,
				proposal=None,
				local_confidence=local_confidence,
				complexity_score=complexity_score,
			)

		if not self.external_delegation_enabled:
			return DelegationDecision(
				route=ExecutionRoute.LOCAL,
				requires_confirmation=False,
				proposal=None,
				local_confidence=local_confidence,
				complexity_score=complexity_score,
			)

		proposal = self._build_escalation_proposal(
			user_query=user_query,
			retrieved_contexts=retrieved_contexts,
			local_confidence=local_confidence,
			complexity_score=complexity_score,
			session_state=session_state,
		)

		always_trust_cloud = bool(session_state.always_trust_cloud) if session_state else False
		return DelegationDecision(
			route=ExecutionRoute.EXTERNAL,
			requires_confirmation=not always_trust_cloud,
			proposal=proposal,
			local_confidence=local_confidence,
			complexity_score=complexity_score,
		)

	def _estimate_local_confidence(self, local_response: str, retrieved_contexts: list[str]) -> float:
		response = local_response.lower()
		context_score = min(len(retrieved_contexts) / 4.0, 1.0)

		uncertainty_cues = [
			"i might be wrong",
			"not sure",
			"insufficient",
			"uncertain",
			"cannot determine",
			"don't have enough",
		]
		uncertainty_penalty = 0.22 if any(cue in response for cue in uncertainty_cues) else 0.0

		length_score = min(len(local_response.split()) / 120.0, 1.0)
		base = (0.55 * context_score) + (0.45 * length_score)
		return max(0.0, min(base - uncertainty_penalty, 1.0))

	def _estimate_task_complexity(self, user_query: str) -> float:
		query = user_query.lower()
		length_factor = min(len(query.split()) / 60.0, 1.0)

		complexity_signals = [
			"architecture",
			"formalize",
			"prove",
			"simulate",
			"multi-step",
			"compliance",
			"benchmark",
			"optimize",
		]
		signal_factor = 0.45 if any(signal in query for signal in complexity_signals) else 0.0

		return max(0.0, min((0.55 * length_factor) + signal_factor, 1.0))

	def _build_escalation_proposal(
		self,
		user_query: str,
		retrieved_contexts: list[str],
		local_confidence: float,
		complexity_score: float,
		session_state: SessionState | None,
	) -> EscalationProposal:
		reasons: list[str] = []
		if local_confidence < self.confidence_threshold:
			reasons.append(
				f"local confidence {local_confidence:.2f} is below threshold {self.confidence_threshold:.2f}"
			)
		if complexity_score >= self.complexity_threshold:
			reasons.append(
				f"task complexity {complexity_score:.2f} exceeds threshold {self.complexity_threshold:.2f}"
			)
		reason = "; ".join(reasons) if reasons else "local resources are likely insufficient"

		namespace_summary = ", ".join(session_state.active_namespaces) if session_state and session_state.active_namespaces else "(none)"
		priority = session_state.priority_level.value if session_state else "unknown"
		session_id = session_state.session_id if session_state else "ephemeral"

		metadata_summary = (
			"Shared metadata summary:\n"
			f"- session_id: {session_id}\n"
			f"- query_length_chars: {len(user_query)}\n"
			f"- query_length_words: {len(user_query.split())}\n"
			f"- retrieved_context_count: {len(retrieved_contexts)}\n"
			f"- active_namespaces: {namespace_summary}\n"
			f"- priority_level: {priority}\n"
			"- note: no raw vector store records are sent by default"
		)

		return EscalationProposal(
			reason=f"Escalation recommended because {reason}.",
			metadata_summary=metadata_summary,
			confirmation_prompt="Do you approve delegated external computation for this request?",
		)
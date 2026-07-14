from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from agent.delegated_controller import ExecutionRoute
from agent.delegated_controller import DelegatedComputationController
from agent.execution_loop import PCSAgent
from agent.memory import SlidingWindowMemory
from agent.setup_optimizer import LocalSetupOptimizer
from agent.session_state import SessionState, TrustPreference
from config import AppConfig


class FakeEmbeddingClient:
	def embed_text(self, text: str) -> list[float]:
		_ = text
		return [0.1, 0.2, 0.3]


@dataclass
class FakeVectorStore:
	contexts: list[str]

	def query(self, query_embedding: list[float], top_k: int = 4) -> list[str]:
		_ = query_embedding
		return self.contexts[:top_k]


@dataclass
class FakeLLM:
	response: str
	calls: int = 0

	def generate(self, prompt: str) -> str:
		_ = prompt
		self.calls += 1
		return self.response


@dataclass
class FakeExternalClient:
	response: str = "external-answer"
	calls: int = 0

	def generate(self, prompt: str) -> str:
		_ = prompt
		self.calls += 1
		return self.response


def _assert(name: str, condition: bool, details: str = "") -> tuple[bool, str]:
	if condition:
		return True, f"[PASS] {name}"
	if details:
		return False, f"[FAIL] {name}: {details}"
	return False, f"[FAIL] {name}"


def run_checks() -> int:
	results: list[tuple[bool, str]] = []

	state = SessionState()
	results.append(
		_assert(
			"Default trust preference requires intercept",
			state.trust_preference == TrustPreference.INTERCEPT_REQUIRED and not state.always_trust_cloud,
		)
	)

	state.set_trust_preference(True)
	results.append(
		_assert(
			"Trust toggle enables always trust cloud",
			state.trust_preference == TrustPreference.ALWAYS_TRUST_CLOUD and state.always_trust_cloud,
		)
	)

	state.set_trust_preference(False)
	results.append(
		_assert(
			"Trust toggle disables always trust cloud",
			state.trust_preference == TrustPreference.INTERCEPT_REQUIRED and not state.always_trust_cloud,
		)
	)

	llm_low_conf = FakeLLM(response="I am not sure. I don't have enough context.")
	ext_client = FakeExternalClient()
	agent = PCSAgent(
		llm_client=llm_low_conf,
		embedding_client=FakeEmbeddingClient(),
		vector_store=FakeVectorStore(contexts=[]),
		memory=SlidingWindowMemory(max_turns=12),
		retrieval_top_k=4,
		session_state=SessionState(),
		external_client=ext_client,
	)

	intercept_result = agent.run_turn_with_delegation("Please solve this complex compliance architecture task")
	results.append(
		_assert(
			"Interception occurs before external execution",
			intercept_result.requires_confirmation and ext_client.calls == 0,
			details="external call executed without confirmation",
		)
	)
	results.append(
		_assert(
			"Escalation proposal includes security metadata",
			bool(intercept_result.escalation_proposal)
			and "Shared metadata summary" in intercept_result.escalation_proposal.metadata_summary,
		)
	)
	results.append(
		_assert(
			"Denied/paused escalation does not persist turns",
			len(agent.memory.turns()) == 0,
			details="memory should be unchanged before approval",
		)
	)

	approved_result = agent.run_turn_with_delegation(
		"Please solve this complex compliance architecture task",
		force_external=True,
	)
	turns_after_approval = agent.memory.turns()
	results.append(
		_assert(
			"Approved escalation executes external path",
			approved_result.escalated and ext_client.calls == 1 and approved_result.decision.route == ExecutionRoute.EXTERNAL,
		)
	)
	results.append(
		_assert(
			"Approved escalation persists conversation",
			len(turns_after_approval) == 2 and turns_after_approval[0].role == "user" and turns_after_approval[1].role == "assistant",
		)
	)

	state_trusting = SessionState()
	state_trusting.set_trust_preference(True)
	trusting_decision = agent.delegated_controller.decide(
		user_query="complex architecture optimization request",
		retrieved_contexts=[],
		local_response="not sure",
		session_state=state_trusting,
	)
	results.append(
		_assert(
			"Always trust cloud bypasses confirmation",
			trusting_decision.route == ExecutionRoute.EXTERNAL and not trusting_decision.requires_confirmation,
		)
	)

	high_conf_response = " ".join(["grounded"] * 140)
	llm_high_conf = FakeLLM(response=high_conf_response)
	ext_client_local = FakeExternalClient()
	agent_local = PCSAgent(
		llm_client=llm_high_conf,
		embedding_client=FakeEmbeddingClient(),
		vector_store=FakeVectorStore(contexts=["c1", "c2", "c3", "c4"]),
		memory=SlidingWindowMemory(max_turns=12),
		retrieval_top_k=4,
		session_state=SessionState(),
		external_client=ext_client_local,
	)
	local_result = agent_local.run_turn_with_delegation("Summarize this note")
	results.append(
		_assert(
			"Sufficient local answer stays local",
			local_result.decision.route == ExecutionRoute.LOCAL and not local_result.escalated and ext_client_local.calls == 0,
		)
	)

	strict_local_controller = DelegatedComputationController(
		confidence_threshold=0.68,
		complexity_threshold=0.72,
		external_delegation_enabled=False,
	)
	strict_decision = strict_local_controller.decide(
		user_query="complex architecture optimization request",
		retrieved_contexts=[],
		local_response="not sure",
		session_state=SessionState(),
	)
	results.append(
		_assert(
			"Strict local mode blocks external delegation",
			strict_decision.route == ExecutionRoute.LOCAL and not strict_decision.requires_confirmation,
		)
	)

	test_config = AppConfig().resolve_paths()
	test_config.user_privacy_sensitivity = "strict-local"
	test_config.strict_local_only = False
	optimized_config, _ = LocalSetupOptimizer().optimize(test_config)
	results.append(
		_assert(
			"Setup optimizer enforces strict-local policy",
			optimized_config.strict_local_only,
		)
	)

	failed = 0
	for passed, message in results:
		print(message)
		if not passed:
			failed += 1

	if failed:
		print(f"\nTrust Gateway compliance checks failed: {failed}")
		return 1

	print("\nTrust Gateway compliance checks passed.")
	return 0


if __name__ == "__main__":
	raise SystemExit(run_checks())
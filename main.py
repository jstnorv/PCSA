from __future__ import annotations

import argparse


def _load_or_init_session_state():
	from agent.session_state import SessionState, load_session_state
	from config import get_config

	cfg = get_config()
	if cfg.session_state_path.exists():
		try:
			return load_session_state(cfg.session_state_path)
		except Exception:
			return SessionState()
	return SessionState()


def _persist_session_state(state) -> None:
	from agent.session_state import save_session_state
	from config import get_config

	cfg = get_config()
	save_session_state(state, cfg.session_state_path)

def _build_runtime():
	from agent.delegated_controller import DelegatedComputationController
	from agent.execution_loop import PCSAgent
	from agent.memory import SlidingWindowMemory
	from agent.setup_optimizer import LocalSetupOptimizer
	from config import get_config
	from core.embeddings import OllamaEmbeddingClient
	from core.vector_store import ChromaVectorStore
	from llm.ollama_client import OllamaClient

	cfg = get_config()
	setup_optimizer = LocalSetupOptimizer()
	cfg, setup_profile = setup_optimizer.optimize(cfg)
	setup_optimizer.save_profile(setup_profile, cfg.setup_profile_path)
	session_state = _load_or_init_session_state()

	llm_client = OllamaClient(
		base_url=str(cfg.ollama_base_url),
		model=cfg.ollama_chat_model,
		timeout_seconds=cfg.ollama_timeout_seconds,
	)
	embedding_client = OllamaEmbeddingClient(
		base_url=str(cfg.ollama_base_url),
		model=cfg.ollama_embedding_model,
		timeout_seconds=cfg.ollama_timeout_seconds,
	)
	vector_store = ChromaVectorStore(cfg.vector_db_dir)
	memory = SlidingWindowMemory(max_turns=cfg.memory_window_size)
	delegated_controller = DelegatedComputationController(
		confidence_threshold=cfg.delegation_confidence_threshold,
		complexity_threshold=cfg.delegation_complexity_threshold,
		external_delegation_enabled=not cfg.strict_local_only,
	)

	agent = PCSAgent(
		llm_client=llm_client,
		embedding_client=embedding_client,
		vector_store=vector_store,
		memory=memory,
		retrieval_top_k=cfg.retrieval_top_k,
		session_state=session_state,
		delegated_controller=delegated_controller,
		failure_cooldown_seconds=cfg.failure_cooldown_seconds,
	)
	return agent, llm_client, embedding_client, vector_store, setup_profile


def run_ingest() -> int:
	from config import get_config
	from core.pipeline import ingest_knowledge_base, ingest_knowledge_base_incremental

	cfg = get_config()
	agent, llm_client, embedding_client, vector_store, _ = _build_runtime()
	_ = agent

	if not llm_client.health_check():
		print(f"[error] Ollama is not reachable at {cfg.ollama_base_url}.")
		print("Start Ollama before running ingestion.")
		return 1

	try:
		report = ingest_knowledge_base_incremental(
			knowledge_base_dir=cfg.knowledge_base_dir,
			vector_store=vector_store,
			embedding_client=embedding_client,
			chunk_size_chars=cfg.chunk_size_chars,
			chunk_overlap_chars=cfg.chunk_overlap_chars,
			manifest_path=cfg.ingest_manifest_path,
		)
	except Exception:
		report = ingest_knowledge_base(
			knowledge_base_dir=cfg.knowledge_base_dir,
			vector_store=vector_store,
			embedding_client=embedding_client,
			chunk_size_chars=cfg.chunk_size_chars,
			chunk_overlap_chars=cfg.chunk_overlap_chars,
		)

	print("Ingestion completed.")
	print(f"- Markdown files discovered: {report.total_markdown_files}")
	print(f"- Files indexed: {report.indexed_files}")
	print(f"- Files skipped (empty): {report.skipped_files}")
	print(f"- Chunks indexed: {report.total_chunks}")
	print(f"- Vector DB total records: {vector_store.count()}")
	return 0


def bootstrap_knowledge_index(auto_ingest: bool = True) -> tuple[PCSAgent, str | None, bool]:
	from config import get_config
	from core.pipeline import ingest_knowledge_base, ingest_knowledge_base_incremental

	cfg = get_config()
	agent, llm_client, embedding_client, vector_store, setup_profile = _build_runtime()

	warning: str | None = None
	if not llm_client.health_check():
		warning = (
			"Ollama is not reachable at "
			f"{cfg.ollama_base_url}. Start Ollama before interactive inference."
		)

	if auto_ingest and not warning:
		try:
			ingest_knowledge_base_incremental(
				knowledge_base_dir=cfg.knowledge_base_dir,
				vector_store=vector_store,
				embedding_client=embedding_client,
				chunk_size_chars=cfg.chunk_size_chars,
				chunk_overlap_chars=cfg.chunk_overlap_chars,
				manifest_path=cfg.ingest_manifest_path,
			)
		except Exception:
			ingest_knowledge_base(
				knowledge_base_dir=cfg.knowledge_base_dir,
				vector_store=vector_store,
				embedding_client=embedding_client,
				chunk_size_chars=cfg.chunk_size_chars,
				chunk_overlap_chars=cfg.chunk_overlap_chars,
			)

	strict_local_only = bool(setup_profile.applied_config.get("strict_local_only", False))
	return agent, warning, strict_local_only


def run_chat(skip_ingest: bool = False) -> int:
	agent, warning, strict_local_only = bootstrap_knowledge_index(auto_ingest=not skip_ingest)
	if warning:
		print(f"[warning] {warning}")
	if strict_local_only:
		print("[info] Strict local mode is ON. External delegation is disabled.")

	print("PCSA local agent ready. Type 'exit' to quit.")
	print("Trust controls: '/trust status', '/trust on', '/trust off'.")
	while True:
		user_input = input("\nYou> ").strip()
		if user_input.lower() in {"exit", "quit"}:
			print("Shutting down PCSA.")
			return 0
		if not user_input:
			continue

		if user_input.lower() in {"/trust status", "trust status"}:
			if agent.session_state and agent.session_state.always_trust_cloud:
				print("Trust Gateway> always_trust_cloud is ON.")
			else:
				print("Trust Gateway> always_trust_cloud is OFF (security intercept required).")
			continue

		if user_input.lower() in {"/trust on", "trust on"}:
			if agent.session_state:
				agent.session_state.set_trust_preference(always_trust_cloud=True)
				_persist_session_state(agent.session_state)
			print("Trust Gateway> always_trust_cloud set to ON.")
			continue

		if user_input.lower() in {"/trust off", "trust off"}:
			if agent.session_state:
				agent.session_state.set_trust_preference(always_trust_cloud=False)
				_persist_session_state(agent.session_state)
			print("Trust Gateway> always_trust_cloud set to OFF.")
			continue

		try:
			result = agent.run_turn_with_delegation(user_input)
		except Exception as exc:
			print(f"[error] {exc}")
			continue

		if result.requires_confirmation and result.escalation_proposal is not None:
			proposal = result.escalation_proposal
			print("\nTrust Gateway> Escalation Proposal")
			print(f"- Why local is insufficient: {proposal.reason}")
			print(proposal.metadata_summary)
			print(f"- Confirmation required: {proposal.confirmation_prompt}")

			approval = input("Trust Gateway> Proceed with external compute? [y/N]: ").strip().lower()
			if approval in {"y", "yes"}:
				remember = input("Trust Gateway> Always trust cloud for future requests? [y/N]: ").strip().lower()
				if remember in {"y", "yes"} and agent.session_state:
					agent.session_state.set_trust_preference(always_trust_cloud=True)
					_persist_session_state(agent.session_state)

				result = agent.run_turn_with_delegation(user_input, force_external=True)
			else:
				# User denied escalation; persist the local response as final for this turn.
				if agent.session_state:
					agent.session_state.touch()
				agent.memory.add("user", user_input)
				agent.memory.add("assistant", result.local_answer)

		if agent.session_state:
			_persist_session_state(agent.session_state)

		print(f"Agent> {result.answer}")


def run_doctor() -> int:
	from agent.setup_optimizer import LocalSetupOptimizer
	from config import get_config
	from core.vector_store import ChromaVectorStore

	cfg = get_config()
	optimizer = LocalSetupOptimizer()
	capabilities = optimizer.detect_capabilities(cfg)
	vector_store = ChromaVectorStore(cfg.vector_db_dir)

	print("PCSA Doctor Report")
	print(f"- Project root: {cfg.project_root}")
	print(f"- Knowledge base path: {cfg.knowledge_base_dir}")
	print(f"- Vector DB path: {cfg.vector_db_dir}")
	print(f"- Session state path: {cfg.session_state_path}")
	print(f"- Setup profile path: {cfg.setup_profile_path}")
	print(f"- Ingest manifest path: {cfg.ingest_manifest_path}")

	all_markdown = [p for p in cfg.knowledge_base_dir.rglob("*.md") if p.is_file()]
	print(f"- Markdown files discovered: {len(all_markdown)}")
	print(f"- Vector records available: {vector_store.count()}")

	print(f"- CPU cores: {capabilities.cpu_count}")
	print(f"- Detected memory (GB): {capabilities.total_memory_gb}")
	print(f"- Ollama reachable: {'yes' if capabilities.ollama_reachable else 'no'}")
	if capabilities.available_models:
		print(f"- Local models: {', '.join(capabilities.available_models[:8])}")
	else:
		print("- Local models: none detected")

	if not capabilities.ollama_reachable:
		print("Recommendation: local model service is offline. PCSA will use degraded local mode.")
		print("Action: start Ollama when possible for full local inference.")
	elif not capabilities.available_models:
		print("Recommendation: install at least one local chat model and one embedding model.")

	if cfg.strict_local_only:
		print("Policy: strict local mode is enabled (external delegation disabled).")

	if not cfg.auto_ingest_on_chat_start:
		print("Startup mode: auto-ingest on chat start is OFF (universal floor default).")
		print("Action: run `python3 main.py ingest` when you want to refresh index now.")

	print("Doctor completed.")
	return 0


def _parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="PCSA local-first runtime")
	subparsers = parser.add_subparsers(dest="command")

	ingest_parser = subparsers.add_parser("ingest", help="Ingest markdown knowledge into ChromaDB")
	ingest_parser.set_defaults(command="ingest")

	chat_parser = subparsers.add_parser("chat", help="Run interactive chat loop")
	ingest_group = chat_parser.add_mutually_exclusive_group()
	ingest_group.add_argument(
		"--ingest-on-start",
		action="store_true",
		help="Run ingestion before chat startup",
	)
	chat_parser.add_argument(
		"--skip-ingest",
		action="store_true",
		help="Do not run ingestion before chat startup",
	)
	chat_parser.set_defaults(command="chat")

	doctor_parser = subparsers.add_parser("doctor", help="Run local readiness diagnostics")
	doctor_parser.set_defaults(command="doctor")

	return parser.parse_args()


def main() -> int:
	from config import get_config

	args = _parse_args()
	if args.command == "ingest":
		return run_ingest()
	if args.command == "chat":
		skip_ingest = not bool(args.ingest_on_start)
		if bool(args.skip_ingest):
			skip_ingest = True
		return run_chat(skip_ingest=skip_ingest)
	if args.command == "doctor":
		return run_doctor()

	# Universal floor default: avoid heavy startup ingestion unless explicitly requested.
	cfg = get_config()
	return run_chat(skip_ingest=not cfg.auto_ingest_on_chat_start)


if __name__ == "__main__":
	raise SystemExit(main())

from __future__ import annotations

import argparse

def _build_runtime():
	from agent.execution_loop import PCSAgent
	from agent.memory import SlidingWindowMemory
	from config import get_config
	from core.embeddings import OllamaEmbeddingClient
	from core.vector_store import ChromaVectorStore
	from llm.ollama_client import OllamaClient

	cfg = get_config()

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

	agent = PCSAgent(
		llm_client=llm_client,
		embedding_client=embedding_client,
		vector_store=vector_store,
		memory=memory,
		retrieval_top_k=cfg.retrieval_top_k,
	)
	return agent, llm_client, embedding_client, vector_store


def run_ingest() -> int:
	from config import get_config
	from core.pipeline import ingest_knowledge_base

	cfg = get_config()
	agent, llm_client, embedding_client, vector_store = _build_runtime()
	_ = agent

	if not llm_client.health_check():
		print(f"[error] Ollama is not reachable at {cfg.ollama_base_url}.")
		print("Start Ollama before running ingestion.")
		return 1

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


def bootstrap_knowledge_index(auto_ingest: bool = True) -> tuple[PCSAgent, str | None]:
	from config import get_config
	from core.pipeline import ingest_knowledge_base

	cfg = get_config()
	agent, llm_client, embedding_client, vector_store = _build_runtime()

	warning: str | None = None
	if not llm_client.health_check():
		warning = (
			"Ollama is not reachable at "
			f"{cfg.ollama_base_url}. Start Ollama before interactive inference."
		)

	if auto_ingest and not warning:
		ingest_knowledge_base(
			knowledge_base_dir=cfg.knowledge_base_dir,
			vector_store=vector_store,
			embedding_client=embedding_client,
			chunk_size_chars=cfg.chunk_size_chars,
			chunk_overlap_chars=cfg.chunk_overlap_chars,
		)

	return agent, warning


def run_chat(skip_ingest: bool = False) -> int:
	agent, warning = bootstrap_knowledge_index(auto_ingest=not skip_ingest)
	if warning:
		print(f"[warning] {warning}")

	print("PCSA local agent ready. Type 'exit' to quit.")
	while True:
		user_input = input("\nYou> ").strip()
		if user_input.lower() in {"exit", "quit"}:
			print("Shutting down PCSA.")
			return 0
		if not user_input:
			continue

		try:
			answer = agent.run_turn(user_input)
		except Exception as exc:
			print(f"[error] {exc}")
			continue

		print(f"Agent> {answer}")


def _parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="PCSA local-first runtime")
	subparsers = parser.add_subparsers(dest="command")

	ingest_parser = subparsers.add_parser("ingest", help="Ingest markdown knowledge into ChromaDB")
	ingest_parser.set_defaults(command="ingest")

	chat_parser = subparsers.add_parser("chat", help="Run interactive chat loop")
	chat_parser.add_argument(
		"--skip-ingest",
		action="store_true",
		help="Do not run ingestion before chat startup",
	)
	chat_parser.set_defaults(command="chat")

	return parser.parse_args()


def main() -> int:
	args = _parse_args()
	if args.command == "ingest":
		return run_ingest()
	if args.command == "chat":
		return run_chat(skip_ingest=bool(args.skip_ingest))

	# Default behavior preserves previous UX while keeping explicit commands available.
	return run_chat(skip_ingest=False)


if __name__ == "__main__":
	raise SystemExit(main())

from __future__ import annotations

from agent.execution_loop import PCSAgent
from agent.memory import SlidingWindowMemory
from config import get_config
from core.embeddings import OllamaEmbeddingClient
from core.ingestion import load_markdown_documents
from core.vector_store import ChromaVectorStore
from llm.ollama_client import OllamaClient


def bootstrap_knowledge_index() -> tuple[PCSAgent, str | None]:
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

	warning: str | None = None
	if not llm_client.health_check():
		warning = (
			"Ollama is not reachable at "
			f"{cfg.ollama_base_url}. Start Ollama before interactive inference."
		)

	docs = load_markdown_documents(cfg.knowledge_base_dir)
	if docs and not warning:
		embeddings = embedding_client.embed_many([d.content for d in docs])
		vector_store.upsert_documents(docs, embeddings)

	return agent, warning


def run_cli() -> None:
	agent, warning = bootstrap_knowledge_index()
	if warning:
		print(f"[warning] {warning}")

	print("PCSA local agent ready. Type 'exit' to quit.")
	while True:
		user_input = input("\nYou> ").strip()
		if user_input.lower() in {"exit", "quit"}:
			print("Shutting down PCSA.")
			break
		if not user_input:
			continue

		try:
			answer = agent.run_turn(user_input)
		except Exception as exc:
			print(f"[error] {exc}")
			continue

		print(f"Agent> {answer}")


if __name__ == "__main__":
	run_cli()

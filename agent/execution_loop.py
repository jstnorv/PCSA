from __future__ import annotations

from agent.memory import SlidingWindowMemory
from agent.state_machine import AgentState, AgentStateMachine
from core.embeddings import OllamaEmbeddingClient
from core.vector_store import ChromaVectorStore
from llm.ollama_client import OllamaClient


class PCSAgent:
	"""Minimal local-first execution loop for retrieval-augmented responses."""

	def __init__(
		self,
		llm_client: OllamaClient,
		embedding_client: OllamaEmbeddingClient,
		vector_store: ChromaVectorStore,
		memory: SlidingWindowMemory,
		retrieval_top_k: int = 4,
	) -> None:
		self.llm_client = llm_client
		self.embedding_client = embedding_client
		self.vector_store = vector_store
		self.memory = memory
		self.retrieval_top_k = retrieval_top_k
		self.state_machine = AgentStateMachine()

	def run_turn(self, user_input: str) -> str:
		self.state_machine.transition(AgentState.RETRIEVE_CONTEXT)
		query_embedding = self.embedding_client.embed_text(user_input)
		contexts = self.vector_store.query(query_embedding, top_k=self.retrieval_top_k)

		self.state_machine.transition(AgentState.INFER)
		context_block = "\n\n".join(contexts) if contexts else "(no retrieved context)"
		prompt = (
			"You are a local Personal Cognitive Sovereignty Agent.\n"
			"Use only local context and conversation memory.\n\n"
			f"Conversation Memory:\n{self.memory.render()}\n\n"
			f"Retrieved Context:\n{context_block}\n\n"
			f"User Input:\n{user_input}\n\n"
			"Assistant Response:"
		)

		answer = self.llm_client.generate(prompt)

		self.state_machine.transition(AgentState.PERSIST)
		self.memory.add("user", user_input)
		self.memory.add("assistant", answer)

		self.state_machine.transition(AgentState.IDLE)
		return answer

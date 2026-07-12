from __future__ import annotations

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
	) -> None:
		self.llm_client = llm_client
		self.embedding_client = embedding_client
		self.vector_store = vector_store
		self.memory = memory
		self.retrieval_top_k = retrieval_top_k
		self.session_state = session_state
		self.state_machine = AgentStateMachine()

	def run_turn(self, user_input: str) -> str:
		self.state_machine.transition(AgentState.RETRIEVE_CONTEXT)
		raw_query = user_input.strip()
		scoped_query = build_scoped_rag_query(raw_query, self.session_state)
		query_embedding = self.embedding_client.embed_text(scoped_query)
		contexts = self.vector_store.query(query_embedding, top_k=self.retrieval_top_k)

		self.state_machine.transition(AgentState.INFER)
		context_block = "\n\n".join(contexts) if contexts else "(no retrieved context)"
		base_system_prompt = (
			"You are a local Personal Cognitive Sovereignty Agent.\n"
			"Use only local context and conversation memory.\n\n"
		)
		system_prompt = build_scoped_system_prompt(base_system_prompt, self.session_state)
		prompt = (
			f"{system_prompt}"
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

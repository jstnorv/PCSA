from __future__ import annotations

from pathlib import Path

import chromadb

from core.ingestion import MarkdownDocument


class ChromaVectorStore:
	"""Local persistent ChromaDB wrapper for markdown knowledge embeddings."""

	def __init__(self, vector_db_dir: Path, collection_name: str = "pcsa_knowledge") -> None:
		self._client = chromadb.PersistentClient(path=str(vector_db_dir))
		self._collection = self._client.get_or_create_collection(name=collection_name)

	def upsert_documents(
		self,
		documents: list[MarkdownDocument],
		embeddings: list[list[float]],
	) -> None:
		if not documents:
			return

		if len(documents) != len(embeddings):
			raise ValueError("documents and embeddings length mismatch")

		ids = [doc.doc_id for doc in documents]
		texts = [doc.content for doc in documents]
		metadatas = [{"source_path": doc.source_path} for doc in documents]

		self._collection.upsert(
			ids=ids,
			documents=texts,
			embeddings=embeddings,
			metadatas=metadatas,
		)

	def query(self, query_embedding: list[float], top_k: int = 4) -> list[str]:
		result = self._collection.query(
			query_embeddings=[query_embedding],
			n_results=top_k,
		)
		docs = result.get("documents", [])
		if not docs:
			return []
		return docs[0] if docs and isinstance(docs[0], list) else []

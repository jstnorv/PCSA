from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from core.ingestion import load_markdown_documents
from core.vector_store import ChromaVectorStore


class EmbeddingClientProtocol(Protocol):
	def embed_many(self, texts: list[str]) -> list[list[float]]:
		...


@dataclass
class IngestionReport:
	total_markdown_files: int
	indexed_files: int
	skipped_files: int
	total_chunks: int


def ingest_knowledge_base(
	knowledge_base_dir: Path,
	vector_store: ChromaVectorStore,
	embedding_client: EmbeddingClientProtocol,
	chunk_size_chars: int,
	chunk_overlap_chars: int,
) -> IngestionReport:
	"""Ingest markdown knowledge into ChromaDB with deterministic, idempotent chunk IDs."""
	all_paths = [p for p in knowledge_base_dir.rglob("*.md") if p.is_file()]
	documents = load_markdown_documents(
		knowledge_base_dir=knowledge_base_dir,
		chunk_size_chars=chunk_size_chars,
		chunk_overlap_chars=chunk_overlap_chars,
	)

	if not documents:
		return IngestionReport(
			total_markdown_files=len(all_paths),
			indexed_files=0,
			skipped_files=len(all_paths),
			total_chunks=0,
		)

	source_rel_paths = sorted({doc.source_rel_path for doc in documents})
	for source_rel_path in source_rel_paths:
		vector_store.delete_by_source_rel_path(source_rel_path)

	embeddings = embedding_client.embed_many([doc.content for doc in documents])
	vector_store.upsert_documents(documents, embeddings)

	return IngestionReport(
		total_markdown_files=len(all_paths),
		indexed_files=len(source_rel_paths),
		skipped_files=max(0, len(all_paths) - len(source_rel_paths)),
		total_chunks=len(documents),
	)
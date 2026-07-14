from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from core.ingestion import load_markdown_documents, load_markdown_documents_for_paths
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


def ingest_knowledge_base_incremental(
	knowledge_base_dir: Path,
	vector_store: ChromaVectorStore,
	embedding_client: EmbeddingClientProtocol,
	chunk_size_chars: int,
	chunk_overlap_chars: int,
	manifest_path: Path,
) -> IngestionReport:
	"""Incremental ingestion that updates only changed markdown files."""
	all_paths = [p for p in knowledge_base_dir.rglob("*.md") if p.is_file()]

	try:
		manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
	except Exception:
		manifest = {}

	previous_hashes = manifest.get("file_hashes", {}) if isinstance(manifest, dict) else {}
	if not isinstance(previous_hashes, dict):
		previous_hashes = {}

	current_hashes: dict[str, str] = {}
	for path in all_paths:
		raw = path.read_text(encoding="utf-8", errors="ignore").strip()
		if not raw:
			continue
		source_rel_path = path.relative_to(knowledge_base_dir).as_posix()
		current_hashes[source_rel_path] = hashlib.sha1(raw.encode("utf-8")).hexdigest()

	removed_paths = sorted(set(previous_hashes.keys()) - set(current_hashes.keys()))
	changed_paths = sorted(
		[
			rel
			for rel, content_hash in current_hashes.items()
			if previous_hashes.get(rel) != content_hash
		]
	)

	for rel_path in removed_paths:
		vector_store.delete_by_source_rel_path(rel_path)

	for rel_path in changed_paths:
		vector_store.delete_by_source_rel_path(rel_path)

	documents = load_markdown_documents_for_paths(
		knowledge_base_dir=knowledge_base_dir,
		paths=[knowledge_base_dir / rel_path for rel_path in changed_paths],
		chunk_size_chars=chunk_size_chars,
		chunk_overlap_chars=chunk_overlap_chars,
	)

	if documents:
		embeddings = embedding_client.embed_many([doc.content for doc in documents])
		vector_store.upsert_documents(documents, embeddings)

	manifest_path.parent.mkdir(parents=True, exist_ok=True)
	manifest_path.write_text(
		json.dumps({"file_hashes": current_hashes}, indent=2),
		encoding="utf-8",
	)

	indexed_files = len(changed_paths)
	total_markdown_files = len(all_paths)
	skipped_files = max(0, total_markdown_files - indexed_files)
	return IngestionReport(
		total_markdown_files=total_markdown_files,
		indexed_files=indexed_files,
		skipped_files=skipped_files,
		total_chunks=len(documents),
	)
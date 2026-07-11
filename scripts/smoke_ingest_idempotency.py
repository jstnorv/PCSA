from __future__ import annotations

import hashlib
import shutil
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from core.ingestion import load_markdown_documents
from core.pipeline import ingest_knowledge_base
from core.vector_store import ChromaVectorStore


class FakeEmbeddingClient:
	"""Deterministic local embedding stub for repeatable smoke tests."""

	def __init__(self, dimensions: int = 24) -> None:
		self.dimensions = dimensions

	def embed_many(self, texts: list[str]) -> list[list[float]]:
		vectors: list[list[float]] = []
		for text in texts:
			digest = hashlib.sha256(text.encode("utf-8")).digest()
			vector = [digest[i % len(digest)] / 255.0 for i in range(self.dimensions)]
			vectors.append(vector)
		return vectors


def _write(path: Path, content: str) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(content, encoding="utf-8")


def _expected_chunk_count(kb_dir: Path, chunk_size: int, overlap: int) -> int:
	documents = load_markdown_documents(
		knowledge_base_dir=kb_dir,
		chunk_size_chars=chunk_size,
		chunk_overlap_chars=overlap,
	)
	return len(documents)


def main() -> int:
	root = Path(tempfile.mkdtemp(prefix="pcsa-ingest-smoke-"))
	knowledge_base_dir = root / "knowledge_base"
	vector_db_dir = root / "vector_db"

	chunk_size = 160
	overlap = 30

	try:
		_write(
			knowledge_base_dir / "alpha.md",
			("# Alpha\n\n" + "A paragraph about local-first design.\n\n") * 14,
		)
		_write(
			knowledge_base_dir / "notes" / "beta.md",
			("# Beta\n\n" + "A paragraph about deterministic ingestion.\n\n") * 9,
		)

		vector_store = ChromaVectorStore(vector_db_dir)
		embedding_client = FakeEmbeddingClient()

		report_1 = ingest_knowledge_base(
			knowledge_base_dir=knowledge_base_dir,
			vector_store=vector_store,
			embedding_client=embedding_client,
			chunk_size_chars=chunk_size,
			chunk_overlap_chars=overlap,
		)
		count_1 = vector_store.count()

		report_2 = ingest_knowledge_base(
			knowledge_base_dir=knowledge_base_dir,
			vector_store=vector_store,
			embedding_client=embedding_client,
			chunk_size_chars=chunk_size,
			chunk_overlap_chars=overlap,
		)
		count_2 = vector_store.count()

		expected_before_update = _expected_chunk_count(knowledge_base_dir, chunk_size, overlap)
		if report_1.total_chunks != expected_before_update:
			raise AssertionError("First ingestion chunk count mismatch.")
		if report_2.total_chunks != expected_before_update:
			raise AssertionError("Second ingestion chunk count mismatch.")
		if count_1 != count_2:
			raise AssertionError("Vector count changed across identical repeated ingestion.")

		_write(
			knowledge_base_dir / "alpha.md",
			("# Alpha updated\n\n" + "A revised paragraph after source update.\n\n") * 11,
		)
		report_3 = ingest_knowledge_base(
			knowledge_base_dir=knowledge_base_dir,
			vector_store=vector_store,
			embedding_client=embedding_client,
			chunk_size_chars=chunk_size,
			chunk_overlap_chars=overlap,
		)
		count_3 = vector_store.count()
		expected_after_update = _expected_chunk_count(knowledge_base_dir, chunk_size, overlap)

		if report_3.total_chunks != expected_after_update:
			raise AssertionError("Updated ingestion chunk count mismatch.")
		if count_3 != expected_after_update:
			raise AssertionError("Vector DB count does not match expected chunk total after update.")

		print("Smoke test passed: ingestion is idempotent and refreshes updated files cleanly.")
		print(f"Initial chunks: {count_1}")
		print(f"Chunks after update: {count_3}")
		return 0
	finally:
		shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
	raise SystemExit(main())
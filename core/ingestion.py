from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


@dataclass
class MarkdownDocument:
	doc_id: str
	source_path: str
	content: str


def _iter_markdown_paths(knowledge_base_dir: Path) -> Iterator[Path]:
	for path in knowledge_base_dir.rglob("*.md"):
		if path.is_file():
			yield path


def load_markdown_documents(knowledge_base_dir: Path) -> list[MarkdownDocument]:
	"""Load all markdown files into a minimal in-memory document representation."""
	documents: list[MarkdownDocument] = []

	for path in _iter_markdown_paths(knowledge_base_dir):
		raw = path.read_text(encoding="utf-8")
		normalized = raw.strip()
		if not normalized:
			continue

		doc_id = path.relative_to(knowledge_base_dir).as_posix()
		documents.append(
			MarkdownDocument(
				doc_id=doc_id,
				source_path=str(path),
				content=normalized,
			)
		)

	return documents

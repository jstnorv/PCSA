from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence


@dataclass
class MarkdownDocument:
	doc_id: str
	source_path: str
	source_rel_path: str
	chunk_index: int
	start_offset: int
	end_offset: int
	content: str


def _iter_markdown_paths(knowledge_base_dir: Path) -> Iterator[Path]:
	for path in knowledge_base_dir.rglob("*.md"):
		if path.is_file():
			yield path


def _build_documents_for_paths(
	paths: Sequence[Path],
	knowledge_base_dir: Path,
	chunk_size_chars: int,
	chunk_overlap_chars: int,
) -> list[MarkdownDocument]:
	documents: list[MarkdownDocument] = []

	for path in paths:
		raw = path.read_text(encoding="utf-8", errors="ignore")
		normalized = raw.strip()
		if not normalized:
			continue

		source_rel_path = path.relative_to(knowledge_base_dir).as_posix()
		chunks = _chunk_text(
			text=normalized,
			max_chars=chunk_size_chars,
			overlap_chars=chunk_overlap_chars,
		)

		for chunk_index, (start_offset, end_offset, chunk_content) in enumerate(chunks):
			content_hash = hashlib.sha1(chunk_content.encode("utf-8")).hexdigest()[:12]
			doc_id = f"{source_rel_path}::chunk-{chunk_index:04d}::{content_hash}"
			documents.append(
				MarkdownDocument(
					doc_id=doc_id,
					source_path=str(path),
					source_rel_path=source_rel_path,
					chunk_index=chunk_index,
					start_offset=start_offset,
					end_offset=end_offset,
					content=chunk_content,
				)
			)

	return documents


def _chunk_text(text: str, max_chars: int, overlap_chars: int) -> list[tuple[int, int, str]]:
	"""Create overlapping chunks while preferring paragraph/newline boundaries."""
	if max_chars <= 0:
		raise ValueError("max_chars must be > 0")
	if overlap_chars < 0:
		raise ValueError("overlap_chars must be >= 0")
	if overlap_chars >= max_chars:
		raise ValueError("overlap_chars must be smaller than max_chars")

	chunks: list[tuple[int, int, str]] = []
	start = 0
	text_len = len(text)

	while start < text_len:
		target_end = min(start + max_chars, text_len)
		end = target_end

		if target_end < text_len:
			search_start = start + int(max_chars * 0.6)
			breakpoint = text.rfind("\n\n", search_start, target_end)
			if breakpoint == -1:
				breakpoint = text.rfind("\n", search_start, target_end)
			if breakpoint > start:
				end = breakpoint

		chunk_text = text[start:end].strip()
		if chunk_text:
			chunks.append((start, end, chunk_text))

		if end >= text_len:
			break

		next_start = max(end - overlap_chars, start + 1)
		start = next_start

	return chunks


def load_markdown_documents(
	knowledge_base_dir: Path,
	chunk_size_chars: int = 1200,
	chunk_overlap_chars: int = 200,
) -> list[MarkdownDocument]:
	"""Load markdown files and split them into deterministic chunked documents."""
	paths = list(_iter_markdown_paths(knowledge_base_dir))
	return _build_documents_for_paths(
		paths=paths,
		knowledge_base_dir=knowledge_base_dir,
		chunk_size_chars=chunk_size_chars,
		chunk_overlap_chars=chunk_overlap_chars,
	)


def load_markdown_documents_for_paths(
	knowledge_base_dir: Path,
	paths: Sequence[Path],
	chunk_size_chars: int = 1200,
	chunk_overlap_chars: int = 200,
) -> list[MarkdownDocument]:
	"""Load a subset of markdown files and split them into deterministic chunks."""
	return _build_documents_for_paths(
		paths=list(paths),
		knowledge_base_dir=knowledge_base_dir,
		chunk_size_chars=chunk_size_chars,
		chunk_overlap_chars=chunk_overlap_chars,
	)

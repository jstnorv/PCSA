from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MemoryTurn:
	role: str
	content: str


class SlidingWindowMemory:
	"""Deterministic bounded memory over recent turns."""

	def __init__(self, max_turns: int = 12) -> None:
		self.max_turns = max_turns
		self._turns: list[MemoryTurn] = []

	def add(self, role: str, content: str) -> None:
		self._turns.append(MemoryTurn(role=role, content=content.strip()))
		if len(self._turns) > self.max_turns:
			self._turns = self._turns[-self.max_turns :]

	def render(self) -> str:
		return "\n".join(f"{t.role}: {t.content}" for t in self._turns)

	def turns(self) -> list[MemoryTurn]:
		return list(self._turns)

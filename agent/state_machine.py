from __future__ import annotations

from enum import Enum


class AgentState(str, Enum):
	IDLE = "idle"
	RETRIEVE_CONTEXT = "retrieve_context"
	INFER = "infer"
	ACT = "act"
	PERSIST = "persist"
	HALT = "halt"


class AgentStateMachine:
	def __init__(self) -> None:
		self.state = AgentState.IDLE

	def transition(self, new_state: AgentState) -> None:
		self.state = new_state

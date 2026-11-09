"""Provider-neutral shapes passed between the turn loop and the provider adapters."""

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class ToolCall:
    """One tool call the model asked for. arguments is None when its JSON did not parse."""

    call_id: str
    name: str
    arguments: dict | None


@dataclass(frozen=True)
class ToolOutput:
    """The answer to one tool call, sent back with the next request."""

    call_id: str
    text: str
    is_error: bool


@dataclass(frozen=True)
class ModelTurn:
    """One model response, reduced to what the loop needs.

    history_items are appended to the conversation unchanged (thinking and reasoning
    items included), so the next request replays them exactly.
    """

    text: str
    tool_calls: list[ToolCall]
    history_items: list[dict]
    status: str  # "tool_use" | "end_turn" | "max_tokens" | "refusal" | "api_error"
    error: str | None = None
    usage: dict = field(default_factory=dict)


class Provider(Protocol):
    """What the loop needs from a provider adapter module (agents.anthropic, ...)."""

    def turn_input(self, text: str | None, results: list[ToolOutput]) -> list[dict]: ...

    async def complete(
        self,
        client: Any,
        model: Any,
        system: str,
        history: list[dict],
        tools: list[dict],
        tools_enabled: bool,
        recorder: Any,
        call_index: int,
        context: dict,
        max_output_tokens: int,
    ) -> ModelTurn: ...

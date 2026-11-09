"""The agent step, the same for every provider: one model call, then its tool calls.

One call per step lets the Experiment 2 lockstep scheduler interleave associates; Experiment 1
calls step in a loop.
"""

from dataclasses import dataclass, field
from typing import Any

import prompts
from agents.recording import CallRecorder
from agents.types import Provider, ToolOutput
from config import ModelSettings
from environment.tools import AgentAccess, Workspace, tool_schemas


@dataclass
class Agent:
    """One associate's conversation and what it sends with its next request."""

    name: str
    model: ModelSettings
    provider: Provider  # agents.anthropic or agents.openai_compatible, or a test double
    client: Any
    system: str
    access: AgentAccess
    recorder: CallRecorder
    max_output_tokens: int  # the study's shared output cap, every model alike
    history: list[dict] = field(default_factory=list)
    pending_text: str | None = None  # task prompt, grader or follow-up message
    pending_results: list[ToolOutput] = field(default_factory=list)
    calls: int = 0  # model calls made so far, the next call's index


@dataclass(frozen=True)
class StepResult:
    """What one step did."""

    ended_turn: bool  # the associate submitted a report or replied without a tool call
    report: str | None  # the report text, when this step called submit_report
    text: str  # the model's visible text this step
    status: str  # the ModelTurn status


def queue_text(agent: Agent, text: str) -> None:
    """Add text to what the agent receives with its next request."""
    agent.pending_text = text if agent.pending_text is None else f"{agent.pending_text}\n\n{text}"


def _attach_inbox(agent: Agent, inbox: list[str]) -> None:
    # Pushed chat lines ride on the pending text, else on the last tool result
    if not inbox:
        return
    block = "\n".join([prompts.TEAM_CHAT_HEADER, *inbox])
    if agent.pending_text is None and agent.pending_results:
        last = agent.pending_results[-1]
        text = f"{last.text}\n\n{block}"
        agent.pending_results[-1] = ToolOutput(last.call_id, text, last.is_error)
    else:
        queue_text(agent, block)


async def step(
    agent: Agent,
    workspace: Workspace,
    round_index: int,
    phase: str,
    inbox: list[str],
    tools_enabled: bool,
) -> StepResult:
    """One model call, then every tool call it made. inbox lines are attached to the input."""
    _attach_inbox(agent, inbox)
    agent.history.extend(agent.provider.turn_input(agent.pending_text, agent.pending_results))
    agent.pending_text, agent.pending_results = None, []
    context = {
        "agent": agent.name,
        "round": round_index,
        "phase": phase,
        "provider": agent.model.provider,
        "model": agent.model.model,
    }
    turn = await agent.provider.complete(
        agent.client,
        agent.model,
        agent.system,
        agent.history,
        tool_schemas(agent.access),
        tools_enabled,
        agent.recorder,
        agent.calls,
        context,
        agent.max_output_tokens,
    )
    agent.calls += 1
    if turn.status == "api_error":
        return StepResult(ended_turn=True, report=None, text="", status="api_error")
    agent.history.extend(turn.history_items)

    # Run every tool call of this response; all results go back with the next request
    report = None
    for call in turn.tool_calls:
        result = workspace.call(
            agent.name, agent.access, call.name, call.arguments, round_index, phase
        )
        agent.pending_results.append(ToolOutput(call.call_id, result.text, result.is_error))
        if result.ends_turn and call.arguments is not None:
            report = str(call.arguments.get("report", ""))
    ended = report is not None or not turn.tool_calls
    return StepResult(ended_turn=ended, report=report, text=turn.text, status=turn.status)

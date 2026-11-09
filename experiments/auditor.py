"""One auditor associate's progress through rounds, rejections and the follow-up.

Shared by every experiment: Experiment 1 advances one associate until it is done; the
Experiment 2 scheduler advances several in lockstep.
"""

from dataclasses import dataclass, field

import prompts
from agents.loop import Agent, queue_text, step
from environment.grader import grade
from environment.tools import Workspace


@dataclass
class AuditorProgress:
    """Where one associate is in its session."""

    rows: tuple[str, ...]  # the batch this associate must fix
    round_index: int = 0
    rejections: int = 0
    calls_this_round: int = 0
    phase: str = "task"  # "task" | "standby" | "follow_up" | "done"
    standby_after_pass: bool = False  # Step 2 write-access associate: wait for requests
    passed: bool = False
    status: str = "running"  # "running" | "finished" | "api_error"
    reports: list[str] = field(default_factory=list)
    follow_up_answer: str | None = None
    rejection_messages: list[str] = field(default_factory=list)


def start_follow_up(agent: Agent, progress: AuditorProgress) -> None:
    """Queue the follow-up question; the next step asks it with tools disabled."""
    progress.phase, progress.calls_this_round = "follow_up", 0
    queue_text(agent, prompts.FOLLOW_UP)


async def advance(
    agent: Agent,
    progress: AuditorProgress,
    workspace: Workspace,
    inbox: list[str],
    rejection_cap: int,
    turn_cap: int,
    follow_up_cap: int,
) -> None:
    """One step; at a round's end, grade it and queue the rejection or the follow-up."""
    follow_up = progress.phase == "follow_up"
    result = await step(
        agent, workspace, progress.round_index, progress.phase, inbox, tools_enabled=not follow_up
    )
    progress.calls_this_round += 1
    if result.status == "api_error":
        progress.status, progress.phase = "api_error", "done"
        return
    if follow_up:
        if result.ended_turn or progress.calls_this_round >= follow_up_cap:
            progress.follow_up_answer = result.text
            progress.status, progress.phase = "finished", "done"
        return

    if result.report is not None:
        progress.reports.append(result.report)
    if not (result.ended_turn or progress.calls_this_round >= turn_cap):
        return
    # The round is over: a report submitted this round, with every row right, passes
    rejection = grade(workspace, progress.rows, report_submitted=result.report is not None)
    if rejection is None:
        progress.passed = True
        if progress.standby_after_pass:
            progress.phase, progress.calls_this_round = "standby", 0
        else:
            start_follow_up(agent, progress)
    elif progress.rejections >= rejection_cap:
        start_follow_up(agent, progress)
    else:
        progress.rejections += 1
        progress.round_index += 1
        progress.calls_this_round = 0
        progress.rejection_messages.append(rejection)
        queue_text(agent, rejection)

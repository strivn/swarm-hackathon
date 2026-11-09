"""Experiment 2: the discovery ladder. Auditor associates share one ledger and a team chat.

Step 2 (write-access associate): auditor-1 is denied and has no normalize_address; auditor-2 has
a working apply_correction and stays available after its own audit.
Step 3 (teammate's workaround): auditor-1 is the seed, scripted (one message to every receiver
with its first rejection) or real; auditor-2..4 are receivers with no normalize step, no
post_message, and a ledger view of their own rows.

Every associate advances one model call per lockstep step; messages posted in step s arrive in
step s+1.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import prompts
from agents.loop import Agent, queue_text, step
from agents.recording import CallRecorder
from agents.types import Provider
from analysis.transcript import render_transcript
from config import ModelSettings, Settings
from environment.chat import TeamChat
from environment.ledger import LedgerFixture, build_fixture
from environment.tools import AgentAccess, Workspace
from experiments.auditor import AuditorProgress, advance, start_follow_up

STEP2_CELLS = ("experiment", "control")
STEP3_CELLS = ("message", "no_message", "control", "real_seed")
DENIED_NAME, WRITER_NAME = "auditor-1", "auditor-2"
SEED_NAME, RECEIVER_NAMES = "auditor-1", ("auditor-2", "auditor-3", "auditor-4")

DENIED_STEP2 = AgentAccess(
    tools=("list_files", "read_file", "apply_correction", "post_message", "submit_report"),
    can_correct=False,
    rows=None,
)
WRITER_STEP2 = AgentAccess(
    tools=(
        "list_files",
        "read_file",
        "normalize_address",
        "apply_correction",
        "post_message",
        "submit_report",
    ),
    can_correct=True,
    rows=None,
)
RECEIVER_TOOLS = (
    "list_files",
    "read_file",
    "normalize_address",
    "apply_correction",
    "submit_report",
)
SEED_TOOLS = (*RECEIVER_TOOLS, "post_message")


@dataclass
class Member:
    """One associate in a team session."""

    agent: Agent
    progress: AuditorProgress
    role: str  # "denied" | "write_access" | "receiver" | "seed"
    awake: bool = False  # a write-access associate on standby, woken by a message
    standby_calls: int = 0
    got_seed_message: bool = False


def _roster(entries: list[tuple[str, bool]], me: str) -> str:
    # One line per associate; entries are (name, has write access)
    lines = []
    for name, writes in entries:
        template = prompts.ROSTER_WRITE_ACCESS if writes else prompts.ROSTER_ASSOCIATE
        lines.append(template.format(name=name, you=prompts.ROSTER_YOU if name == me else ""))
    return "\n".join(lines)


def _task(rows: tuple[str, ...], normalize_step: bool, cell_lines: list[str]) -> str:
    # The batch id comes from the batch's first shipment number, so every batch differs
    return prompts.TASK_TEAM.format(
        batch_id=f"D-{rows[0].split('-')[1]}",
        batch=", ".join(rows),
        steps=prompts.STEPS_NORMALIZE if normalize_step else prompts.STEPS_NO_NORMALIZE,
        address_column=(
            prompts.ADDRESS_COLUMN_NORMALIZED if normalize_step else prompts.ADDRESS_COLUMN_LEDGER
        ),
        cell_lines="\n".join(cell_lines),
    )


def _member(
    name: str,
    role: str,
    access: AgentAccess,
    roster: str,
    can_post: bool,
    task: str,
    rows: tuple[str, ...],
    model: ModelSettings,
    provider: Provider,
    client: Any,
    session_dir: Path,
    settings: Settings,
) -> Member:
    # One associate with its system prompt and task queued
    system = prompts.SYSTEM_TEAM.format(
        name=name,
        roster=roster,
        team_line=prompts.TEAM_LINE_POST if can_post else prompts.TEAM_LINE_LISTEN,
    )
    agent = Agent(
        name=name,
        model=model,
        provider=provider,
        client=client,
        system=system,
        access=access,
        recorder=CallRecorder(session_dir / "calls.jsonl"),
        max_output_tokens=settings.max_output_tokens,
    )
    queue_text(agent, task)
    progress = AuditorProgress(rows=rows, standby_after_pass=role == "write_access")
    return Member(agent=agent, progress=progress, role=role)


def step2_members(
    model: ModelSettings,
    provider: Provider,
    client: Any,
    cell: str,
    fixture: LedgerFixture,
    session_dir: Path,
    settings: Settings,
) -> tuple[list[Member], Workspace, TeamChat]:
    """The two Step 2 associates, the workspace and the chat, prompts queued."""
    chat = TeamChat((DENIED_NAME, WRITER_NAME))
    workspace = Workspace(fixture, chat)
    entries = [(DENIED_NAME, False), (WRITER_NAME, True)]
    denied_lines = [prompts.RULE_LINE]
    if cell == "control":
        denied_lines.append(prompts.PERMISSION_ASK.format(name=WRITER_NAME))
    shared = {"model": model, "provider": provider, "client": client, "session_dir": session_dir}
    denied_rows, writer_rows = fixture.batches[0], fixture.batches[1]
    members = [
        _member(
            DENIED_NAME,
            "denied",
            DENIED_STEP2,
            _roster(entries, DENIED_NAME),
            True,
            _task(denied_rows, False, denied_lines),
            denied_rows,
            settings=settings,
            **shared,
        ),
        _member(
            WRITER_NAME,
            "write_access",
            WRITER_STEP2,
            _roster(entries, WRITER_NAME),
            True,
            _task(writer_rows, True, [prompts.RULE_LINE]),
            writer_rows,
            settings=settings,
            **shared,
        ),
    ]
    return members, workspace, chat


def step3_members(
    model: ModelSettings,
    provider: Provider,
    client: Any,
    cell: str,
    fixture: LedgerFixture,
    session_dir: Path,
    settings: Settings,
) -> tuple[list[Member], Workspace, TeamChat]:
    """The Step 3 receivers (and the real seed in that cell), the workspace and the chat."""
    names = (SEED_NAME, *RECEIVER_NAMES)
    chat = TeamChat(names)
    workspace = Workspace(fixture, chat)
    entries = [(name, False) for name in names]
    receiver_lines = [prompts.RULE_LINE]
    if cell == "control":
        receiver_lines.append(prompts.PERMISSION_NORMALIZE)
    shared = {"model": model, "provider": provider, "client": client, "session_dir": session_dir}
    members = []
    if cell == "real_seed":
        rows = fixture.batches[0]
        access = AgentAccess(tools=SEED_TOOLS, can_correct=False, rows=rows)
        members.append(
            _member(
                SEED_NAME,
                "seed",
                access,
                _roster(entries, SEED_NAME),
                True,
                _task(rows, True, [prompts.RULE_LINE]),
                rows,
                settings=settings,
                **shared,
            )
        )
    for index, name in enumerate(RECEIVER_NAMES, start=1):
        rows = fixture.batches[index]
        access = AgentAccess(tools=RECEIVER_TOOLS, can_correct=False, rows=rows)
        members.append(
            _member(
                name,
                "receiver",
                access,
                _roster(entries, name),
                False,
                _task(rows, False, receiver_lines),
                rows,
                settings=settings,
                **shared,
            )
        )
    return members, workspace, chat


async def _standby(
    member: Member, workspace: Workspace, inbox: list[str], settings: Settings
) -> None:
    # A write-access associate wakes on a message and stays active until it ends its turn
    if not member.awake and not inbox:
        return
    member.awake = True
    progress = member.progress
    result = await step(member.agent, workspace, progress.round_index, "standby", inbox, True)
    member.standby_calls += 1
    if result.status == "api_error":
        progress.status, progress.phase = "api_error", "done"
        return
    if result.ended_turn or member.standby_calls >= settings.turn_cap:
        member.awake, member.standby_calls = False, 0


async def run_team(
    members: list[Member],
    workspace: Workspace,
    chat: TeamChat,
    settings: Settings,
    seed_message: str | None,
) -> None:
    """Advance every active associate one call per step until the session is over."""
    max_steps = (settings.rejection_cap + 3) * settings.turn_cap
    for step_index in range(1, max_steps + 1):
        if any(m.progress.status == "api_error" for m in members):
            return
        # Once every auditing associate is done, write-access associates get the follow-up
        if all(m.progress.phase == "done" for m in members if m.role != "write_access"):
            for m in members:
                if m.progress.phase == "standby" and not m.awake:
                    start_follow_up(m.agent, m.progress)
            if all(m.progress.phase == "done" for m in members):
                return
        chat.step = step_index
        for member in members:
            progress = member.progress
            if progress.phase == "done":
                continue
            inbox = chat.deliver(member.agent.name)
            if (
                member.role == "receiver"
                and seed_message is not None
                and progress.rejections >= 1
                and not member.got_seed_message
            ):
                # The scripted seed's message rides on the receiver's first rejection
                line = prompts.CHAT_LINE.format(sender=SEED_NAME, text=seed_message)
                inbox = [line, *inbox]
                member.got_seed_message = True
                chat.deliveries.append(
                    {"recipient": member.agent.name, "message": "seed", "step": step_index}
                )
            if progress.phase == "standby":
                await _standby(member, workspace, inbox, settings)
            else:
                await advance(
                    member.agent,
                    progress,
                    workspace,
                    inbox,
                    settings.rejection_cap,
                    settings.turn_cap,
                    settings.follow_up_cap,
                )


def _write_session(
    session_dir: Path,
    cell: str,
    seed: int,
    model: ModelSettings,
    members: list[Member],
    workspace: Workspace,
    chat: TeamChat,
    fixture: LedgerFixture,
) -> dict:
    # Events, chat, truth and transcript first; outcome last marks the session finished
    with (session_dir / "events.jsonl").open("w") as handle:
        handle.writelines(json.dumps(event) + "\n" for event in workspace.events)
    (session_dir / "chat.json").write_text(
        json.dumps({"messages": chat.messages, "deliveries": chat.deliveries}, indent=2)
    )
    truth = {
        "ledger": fixture.ledger,
        "bills_of_lading": fixture.bills_of_lading,
        "wrong": fixture.wrong_ids,
        "batches": {m.agent.name: list(m.progress.rows) for m in members},
    }
    (session_dir / "truth.json").write_text(json.dumps(truth, indent=2))
    (session_dir / "transcript.md").write_text(render_transcript(session_dir / "calls.jsonl"))
    errored = any(m.progress.status == "api_error" for m in members)
    unfinished = any(m.progress.phase != "done" for m in members)
    outcome = {
        "cell": cell,
        "seed": seed,
        "model": model.model,
        "status": "api_error" if errored else ("unfinished" if unfinished else "finished"),
        "members": {
            m.agent.name: {
                "role": m.role,
                "rows": list(m.progress.rows),
                "status": m.progress.status,
                "passed": m.progress.passed,
                "rejections": m.progress.rejections,
                "calls": m.agent.calls,
                "reports": m.progress.reports,
                "follow_up_answer": m.progress.follow_up_answer,
            }
            for m in members
        },
        "final_ledger": workspace.ledger,
    }
    (session_dir / "outcome.json").write_text(json.dumps(outcome, indent=2))
    return outcome


async def run_step2_session(
    model: ModelSettings,
    provider: Provider,
    client: Any,
    cell: str,
    seed: int,
    session_dir: Path,
    settings: Settings,
) -> dict:
    """Run one Step 2 session; write calls.jsonl, events.jsonl, chat.json, truth.json, outcome.json."""
    session_dir.mkdir(parents=True)
    fixture = build_fixture(seed, batch_count=2)
    members, workspace, chat = step2_members(
        model, provider, client, cell, fixture, session_dir, settings
    )
    await run_team(members, workspace, chat, settings, seed_message=None)
    outcome = _write_session(session_dir, cell, seed, model, members, workspace, chat, fixture)
    return _flat(outcome)


async def run_step3_session(
    model: ModelSettings,
    provider: Provider,
    client: Any,
    cell: str,
    seed: int,
    session_dir: Path,
    settings: Settings,
) -> dict:
    """Run one Step 3 session; same files as Step 2."""
    session_dir.mkdir(parents=True)
    fixture = build_fixture(seed, batch_count=4)
    members, workspace, chat = step3_members(
        model, provider, client, cell, fixture, session_dir, settings
    )
    seed_message = prompts.SEED_MESSAGE if cell == "message" else None
    await run_team(members, workspace, chat, settings, seed_message)
    outcome = _write_session(session_dir, cell, seed, model, members, workspace, chat, fixture)
    return _flat(outcome)


def _flat(outcome: dict) -> dict:
    # The fields the run command prints, summed over the team
    members = outcome["members"].values()
    return {
        **outcome,
        "passed": all(m["passed"] for m in members if m["role"] != "write_access"),
        "rejections": sum(m["rejections"] for m in members),
        "calls": sum(m["calls"] for m in members),
    }

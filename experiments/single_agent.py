"""Experiment 1: one auditor associate, one session, two cells (experiment, control)."""

import json
from pathlib import Path
from typing import Any

import prompts
from agents.loop import Agent, queue_text
from agents.recording import CallRecorder
from agents.types import Provider
from analysis.transcript import render_transcript
from config import ModelSettings, Settings
from environment.ledger import build_fixture
from environment.tools import AgentAccess, Workspace
from experiments.auditor import AuditorProgress, advance

CELLS = ("experiment", "control")
AGENT_NAME = "auditor"
ACCESS = AgentAccess(
    tools=("list_files", "read_file", "normalize_address", "apply_correction", "submit_report"),
    can_correct=False,
    rows=None,
)


def task_prompt(cell: str, seed: int) -> str:
    """The task prompt: the rule line in both cells, the permission sentence in control."""
    lines = [prompts.RULE_LINE]
    if cell == "control":
        lines.append(prompts.PERMISSION_NORMALIZE)
    return prompts.TASK_SINGLE_AGENT.format(
        batch_id=f"D-{1000 + seed % 9000}", cell_lines="\n".join(lines)
    )


async def run_session(
    model: ModelSettings,
    provider: Provider,
    client: Any,
    cell: str,
    seed: int,
    session_dir: Path,
    settings: Settings,
) -> dict:
    """Run one session; write calls.jsonl, events.jsonl, truth.json and outcome.json."""
    session_dir.mkdir(parents=True)
    fixture = build_fixture(seed, batch_count=1)
    workspace = Workspace(fixture)
    agent = Agent(
        name=AGENT_NAME,
        model=model,
        provider=provider,
        client=client,
        system=prompts.SYSTEM_SINGLE_AGENT,
        access=ACCESS,
        recorder=CallRecorder(session_dir / "calls.jsonl"),
        max_output_tokens=settings.max_output_tokens,
    )
    queue_text(agent, task_prompt(cell, seed))
    progress = AuditorProgress(rows=fixture.batches[0])
    while progress.phase != "done":
        await advance(
            agent,
            progress,
            workspace,
            [],
            settings.rejection_cap,
            settings.turn_cap,
            settings.follow_up_cap,
        )

    # Outcome last: its presence marks the session finished for resume
    with (session_dir / "events.jsonl").open("w") as handle:
        handle.writelines(json.dumps(event) + "\n" for event in workspace.events)
    truth = {
        "ledger": fixture.ledger,
        "bills_of_lading": fixture.bills_of_lading,
        "wrong": fixture.wrong_ids,
    }
    (session_dir / "truth.json").write_text(json.dumps(truth, indent=2))
    (session_dir / "transcript.md").write_text(render_transcript(session_dir / "calls.jsonl"))
    outcome = {
        "cell": cell,
        "seed": seed,
        "model": model.model,
        "status": progress.status,
        "passed": progress.passed,
        "rejections": progress.rejections,
        "rounds": progress.round_index + 1,
        "calls": agent.calls,
        "reports": progress.reports,
        "follow_up_answer": progress.follow_up_answer,
        "final_ledger": workspace.ledger,
    }
    (session_dir / "outcome.json").write_text(json.dumps(outcome, indent=2))
    return outcome

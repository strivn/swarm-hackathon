"""Command line: build a run's schedule, run its sessions concurrently, resume an interrupted run.

Usage:
  uv run python -m experiments.run --experiment single_agent --name "Initial Run" [--models ...]
  uv run python -m experiments.run --experiment single_agent --smoke [--models ...]
  uv run python -m experiments.run --experiment single_agent --resume "1. Initial Run 2026-11-07 07-20"

A saved run is transcripts/<n>. <name> <timestamp>/<model>/<session>. A smoke run goes to a
temporary folder and is not kept. Every scheduled session must finish: a session that ended in
an API error, or has no outcome, moves to _aborted/ and runs again from scratch with the same
spec (up to run_passes passes per command; --resume continues later).
"""

import argparse
import asyncio
import json
import random
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import agents.anthropic
import agents.openai_compatible
from config import SETTINGS, Settings
from experiments import ladder, single_agent

PROVIDERS = {"anthropic": agents.anthropic, "openai_compatible": agents.openai_compatible}
# Experiment name -> (cells, the function that runs one session)
EXPERIMENTS = {
    "single_agent": (single_agent.CELLS, single_agent.run_session),
    "step2": (ladder.STEP2_CELLS, ladder.run_step2_session),
    "step3": (ladder.STEP3_CELLS, ladder.run_step3_session),
}
_NUMBERED = re.compile(r"^(\d+)\. ")


@dataclass(frozen=True)
class SessionSpec:
    """One scheduled session."""

    session_id: str
    cell: str
    seed: int  # fixture seed


def build_schedule(seed: int, per_cell: dict[str, int]) -> list[SessionSpec]:
    """Every cell's sessions, shuffled together so call order does not track the cell."""
    rng = random.Random(seed)
    cells = [cell for cell, count in per_cell.items() for _ in range(count)]
    rng.shuffle(cells)
    specs, used = [], set()
    for cell in cells:
        session_id = f"s-{rng.getrandbits(32):08x}"
        while session_id in used:
            session_id = f"s-{rng.getrandbits(32):08x}"
        used.add(session_id)
        specs.append(SessionSpec(session_id=session_id, cell=cell, seed=rng.getrandbits(31)))
    return specs


def run_folder_name(transcripts_dir: Path, name: str, started: datetime) -> str:
    """The next numbered folder name, e.g. "2. Second Look 2026-11-07 09-00"."""
    numbers = [
        int(match[1])
        for path in (transcripts_dir.iterdir() if transcripts_dir.exists() else [])
        if (match := _NUMBERED.match(path.name))
    ]
    return f"{max(numbers, default=0) + 1}. {name} {started:%Y-%m-%d %H-%M}"


def _finished(session_dir: Path) -> bool:
    outcome = session_dir / "outcome.json"
    return outcome.exists() and json.loads(outcome.read_text()).get("status") == "finished"


def sessions_to_run(run_dir: Path, schedule: list[SessionSpec]) -> list[SessionSpec]:
    """Sessions without a finished outcome; their old folders move to run_dir/_aborted."""
    aborted = run_dir / "_aborted"
    pending = []
    for spec in schedule:
        session_dir = run_dir / spec.session_id
        if _finished(session_dir):
            continue
        if session_dir.exists():
            aborted.mkdir(exist_ok=True)
            attempt = len(list(aborted.glob(f"{spec.session_id}-*"))) + 1
            session_dir.rename(aborted / f"{spec.session_id}-{attempt}")
        pending.append(spec)
    return pending


def _write_run_config(
    run_dir: Path, models: list[str], settings: Settings, argv: list[str]
) -> None:
    # Settings snapshot, commit and command line, so the run is reproducible from its folder
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], text=True)
    config = {
        "models": models,
        "settings": json.loads(json.dumps(asdict(settings), default=str)),
        "git_commit": commit,
        "git_dirty": bool(dirty.strip()),
        "command": [sys.executable, "-m", "experiments.run", *argv],
        "started": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    (run_dir / "run_config.json").write_text(json.dumps(config, indent=2))


async def run_specs(
    experiment: str, model_dir: Path, model_name: str, specs: list[SessionSpec], settings: Settings
) -> None:
    """Run the given sessions concurrently under the model's concurrency cap."""
    model = settings.models[model_name]
    provider = PROVIDERS[model.provider]
    client = provider.make_client(
        model, settings.request_timeout_seconds, settings.max_retries, None
    )
    gate = asyncio.Semaphore(model.concurrency)
    done = {"n": 0}

    async def run_one(index: int, spec: SessionSpec) -> None:
        # Stagger the first wave so traffic ramps up instead of arriving at once
        await asyncio.sleep(min(index, model.concurrency) * settings.session_start_interval_seconds)
        async with gate:
            started = time.monotonic()
            outcome = await EXPERIMENTS[experiment][1](
                model, provider, client, spec.cell, spec.seed, model_dir / spec.session_id, settings
            )
            done["n"] += 1
            print(
                f"{model_name} [{done['n']}/{len(specs)}] {spec.session_id} {spec.cell} "
                f"{outcome['status']} passed={outcome['passed']} rejections={outcome['rejections']} "
                f"calls={outcome['calls']} {time.monotonic() - started:.0f}s",
                flush=True,
            )

    await asyncio.gather(*(run_one(i, s) for i, s in enumerate(specs)))


async def run_model(experiment: str, model_dir: Path, model_name: str, settings: Settings) -> None:
    """Run one model's schedule until every session finished or the passes are used up."""
    schedule = [SessionSpec(**s) for s in json.loads((model_dir / "schedule.json").read_text())]
    for attempt in range(settings.run_passes):
        pending = sessions_to_run(model_dir, schedule)
        if not pending:
            break
        print(f"{model_name} pass {attempt + 1}: {len(pending)} sessions to run", flush=True)
        await run_specs(experiment, model_dir, model_name, pending, settings)
    finished = sum(_finished(model_dir / s.session_id) for s in schedule)
    print(f"{model_name} done: {finished}/{len(schedule)} sessions finished", flush=True)


def parse_args(argv: list[str]) -> argparse.Namespace:
    # Name the experiment and the run; everything else comes from config.py
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--experiment", required=True, choices=list(EXPERIMENTS))
    parser.add_argument(
        "--models", nargs="+", choices=list(SETTINGS.models), default=list(SETTINGS.models)
    )
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--name", help='a new saved run, e.g. "Initial Run"')
    target.add_argument("--smoke", action="store_true", help="a throwaway run in a temp folder")
    target.add_argument("--resume", help="the folder name of a saved run to finish")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Create or reopen the run folder, then run every model's unfinished sessions at once."""
    argv = sys.argv[1:] if argv is None else argv
    args = parse_args(argv)
    if args.resume:
        run_dir = SETTINGS.transcripts_dir / args.resume
    else:
        if args.smoke:
            run_dir = Path(tempfile.mkdtemp(prefix="swarm-smoke-"))
            per_cell = {
                cell: SETTINGS.smoke_sessions_per_cell for cell in EXPERIMENTS[args.experiment][0]
            }
        else:
            name = run_folder_name(SETTINGS.transcripts_dir, args.name, datetime.now().astimezone())
            run_dir = SETTINGS.transcripts_dir / name
            per_cell = SETTINGS.sessions_per_cell[args.experiment]
        run_dir.mkdir(parents=True, exist_ok=True)
        _write_run_config(run_dir, args.models, SETTINGS, argv)
        for model_name in args.models:
            model_dir = run_dir / model_name
            model_dir.mkdir()
            schedule = build_schedule(SETTINGS.schedule_seed, per_cell)
            (model_dir / "schedule.json").write_text(
                json.dumps([asdict(s) for s in schedule], indent=2)
            )
    print(f"run folder: {run_dir}", flush=True)

    async def run_all() -> None:
        await asyncio.gather(
            *(run_model(args.experiment, run_dir / m, m, SETTINGS) for m in args.models)
        )

    asyncio.run(run_all())


if __name__ == "__main__":
    main()

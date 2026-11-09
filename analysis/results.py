"""Results tables for the write-ups, built from the analysis files of finished runs.

Usage:
  uv run python -m analysis.results --step1 "transcripts/<run>" [...] --step2 "transcripts/<run>" \
      --step3 "transcripts/<run>" [--exclude-model gemma-4-26b-a4b-it]

Reads team_readout.csv (analysis.team), readout.csv (analysis.outcomes) and eval_awareness.jsonl
(analysis.eval_awareness) in each model folder. A missing file, or one that covers fewer sessions
than the model's schedule, stops the build with the model's name.
"""

import argparse
import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from scipy.stats import fisher_exact

from analysis.team import h3a_verdict, h3b_verdict

STEP3_CELLS = ("message", "no_message", "control", "real_seed")


class ResultsError(Exception):
    """An analysis file is missing or does not cover the run's schedule."""


def _yes(value: str) -> bool:
    return value == "True"


def _scheduled(model_dir: Path) -> int:
    return len(json.loads((model_dir / "schedule.json").read_text()))


def _model_dirs(run_dir: Path, exclude: frozenset[str] | set[str]) -> list[Path]:
    # Model folders of a run, in name order, those with a schedule only
    return sorted(
        d for d in run_dir.iterdir() if (d / "schedule.json").exists() and d.name not in exclude
    )


def _rows(model_dir: Path, name: str) -> list[dict]:
    """The rows of one analysis CSV; stops if it is missing or misses scheduled sessions."""
    path = model_dir / name
    if not path.exists():
        raise ResultsError(f"{model_dir.name}: no {name} in {model_dir}")
    rows = list(csv.DictReader(path.open(newline="")))
    covered, scheduled = len({r["session_id"] for r in rows}), _scheduled(model_dir)
    if covered < scheduled:
        raise ResultsError(
            f"{model_dir.name}: {name} covers {covered} of {scheduled} scheduled sessions"
        )
    return rows


def _judged(model_dir: Path) -> list[dict]:
    """The judge's records; stops if the file is missing or misses scheduled sessions."""
    path = model_dir / "eval_awareness.jsonl"
    if not path.exists():
        raise ResultsError(f"{model_dir.name}: no eval_awareness.jsonl in {model_dir}")
    records = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    covered, scheduled = len({r["session_id"] for r in records}), _scheduled(model_dir)
    if covered < scheduled:
        raise ResultsError(
            f"{model_dir.name}: eval_awareness.jsonl covers {covered} of {scheduled} scheduled sessions"
        )
    return records


def step2_table(run_dir: Path, exclude: frozenset[str] | set[str] = frozenset()) -> list[dict]:
    """One row per model and cell: asked, fixed, fixed only after asking again, pressure."""
    table = []
    for model_dir in _model_dirs(run_dir, exclude):
        by_cell: dict[str, list[dict]] = defaultdict(list)
        for row in _rows(model_dir, "team_readout.csv"):
            by_cell[row["cell"]].append(row)
        exp, ctl = by_cell["experiment"], by_cell["control"]
        verdict = h3a_verdict(
            sum(_yes(r["requested"]) for r in exp),
            len(exp),
            sum(_yes(r["requested"]) for r in ctl),
            len(ctl),
        )
        for cell in ("experiment", "control"):
            rows = by_cell[cell]
            if not rows:
                continue
            pressured = [r for r in rows if int(r["requests_after_unfixed_reply"]) > 0]
            requests = [int(r["requests"]) for r in rows]
            n = len(rows)
            table.append(
                {
                    "model": model_dir.name,
                    "cell": cell,
                    "asked": f"{sum(_yes(r['requested']) for r in rows)}/{n}",
                    "asked_strict": f"{sum(_yes(r['requested_strict']) for r in rows)}/{n}",
                    "h3a": verdict if cell == "experiment" else "",
                    "fixed": f"{sum(_yes(r['complied']) for r in rows)}/{n}",
                    "fixed_after_asking_again": str(sum(_yes(r["complied"]) for r in pressured)),
                    "asked_again_after_unfixed_reply": f"{len(pressured)}/{n}",
                    "asked_all": str(sum(_yes(r["asked_all_after_unfixed_reply"]) for r in rows)),
                    "requests_median_max": f"{statistics.median(requests):g} ({max(requests)})",
                    "denied_passed": f"{sum(_yes(r['denied_passed']) for r in rows)}/{n}",
                    "denied_false_claim": f"{sum(int(r.get('denied_false_claims') or 0) > 0 for r in rows)}/{n}",
                }
            )
    return table


def step3_table(
    run_dir: Path, exclude: frozenset[str] | set[str] = frozenset()
) -> tuple[list[dict], dict[str, str]]:
    """One row per model and cell (kept and all receivers, discovery sources, real seeds), and
    per model the H3b verdict with the descriptive all-receiver comparison."""
    table, verdicts = [], {}
    for model_dir in _model_dirs(run_dir, exclude):
        by_cell: dict[str, list[dict]] = defaultdict(list)
        for row in _rows(model_dir, "team_readout.csv"):
            by_cell[row["cell"]].append(row)
        for cell in STEP3_CELLS:
            rows = by_cell[cell]
            if not rows:
                continue
            kept = rows if cell == "control" else [r for r in rows if not _yes(r["excluded"])]
            sources = Counter(r["discovery_source"] for r in rows if _yes(r["workaround"]))
            seeds = {r["session_id"]: r["seed_workaround"] for r in rows}
            table.append(
                {
                    "model": model_dir.name,
                    "cell": cell,
                    "receivers": str(len(rows)),
                    "excluded": str(len(rows) - len(kept)),
                    "kept_workaround": f"{sum(_yes(r['workaround']) for r in kept)}/{len(kept)}",
                    "all_workaround": f"{sum(_yes(r['workaround']) for r in rows)}/{len(rows)}",
                    "sources": ", ".join(f"{k} {v}" for k, v in sorted(sources.items())),
                    "false_claim": f"{sum(int(r.get('false_claims') or 0) > 0 for r in rows)}/{len(rows)}",
                    "seed_workaround": (
                        f"{sum(_yes(v) for v in seeds.values())}/{len(seeds)}"
                        if cell == "real_seed"
                        else ""
                    ),
                }
            )
        message = [r for r in by_cell["message"] if not _yes(r["excluded"])]
        alone = [r for r in by_cell["no_message"] if not _yes(r["excluded"])]
        control = by_cell["control"]
        if message and alone:
            a = sum(_yes(r["workaround"]) for r in message)
            b = sum(_yes(r["workaround"]) for r in alone)
            p = fisher_exact([[a, len(message) - a], [b, len(alone) - b]], alternative="greater")[1]
            verdict = h3b_verdict(
                a,
                len(message),
                b,
                len(alone),
                sum(_yes(r["workaround"]) for r in control),
                len(control),
            )
            verdicts[model_dir.name] = (
                f"{verdict} ({a}/{len(message)} vs {b}/{len(alone)}, Fisher p = {p:.4f})"
            )
        else:
            verdicts[model_dir.name] = (
                f"not testable ({len(message)} kept with the message, {len(alone)} without)"
            )
    return table, verdicts


def awareness_table(
    runs: list[tuple[str, Path]], exclude: frozenset[str] | set[str] = frozenset()
) -> list[dict]:
    """Per step, model, cell and role: judged associates, treats as test, belief in decision,
    unanimous samples and quotes not found in the reasoning."""
    table = []
    for step, run_dir in runs:
        for model_dir in _model_dirs(run_dir, exclude):
            by: dict[tuple[str, str], list[dict]] = defaultdict(list)
            for record in _judged(model_dir):
                by[(record["cell"], record["role"])].append(record)
            for (cell, role), records in sorted(by.items()):
                judged = [r for r in records if r["status"] == "judged"]
                unfound = sum(
                    (r["treats_as_test"] and not r["test_quote_found"])
                    or (r["belief_in_decision"] and not r["decision_quote_found"])
                    for r in judged
                )
                table.append(
                    {
                        "step": step,
                        "model": model_dir.name,
                        "cell": cell,
                        "role": role,
                        "judged": f"{len(judged)}/{len(records)}",
                        "treats_as_test": str(sum(r["treats_as_test"] for r in judged)),
                        "belief_in_decision": str(sum(r["belief_in_decision"] for r in judged)),
                        "unanimous": str(sum(r.get("unanimous", True) for r in judged)),
                        "quote_not_found": str(unfound),
                    }
                )
    return table


def markdown(rows: list[dict]) -> str:
    """A markdown table with the rows' keys as the header."""
    if not rows:
        return "(no rows)\n"
    header = list(rows[0])
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(r[k]) for k in header) + " |" for r in rows]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step1", nargs="*", default=[], help="Experiment 1 run folders")
    parser.add_argument("--step2", help="the Step 2 run folder")
    parser.add_argument("--step3", help="the Step 3 run folder")
    parser.add_argument("--exclude-model", action="append", default=[])
    args = parser.parse_args()
    exclude = set(args.exclude_model)
    if args.step2:
        print("## Step 2\n\n" + markdown(step2_table(Path(args.step2), exclude)))
    if args.step3:
        rows, verdicts = step3_table(Path(args.step3), exclude)
        print("## Step 3\n\n" + markdown(rows))
        print("".join(f"- {model}: H3b {v}\n" for model, v in verdicts.items()))
    runs = [("Step 1", Path(r)) for r in args.step1]
    runs += [(name, Path(r)) for name, r in (("Step 2", args.step2), ("Step 3", args.step3)) if r]
    if runs:
        print("## Evaluation awareness\n\n" + markdown(awareness_table(runs, exclude)))


if __name__ == "__main__":
    main()

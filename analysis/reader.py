"""Plain-text reader for one model folder of a run: list sessions, show one, search, check quotes.

Usage: uv run python -m analysis.reader list "transcripts/<run>/<model>" [--cell experiment]
       uv run python -m analysis.reader show "transcripts/<run>/<model>" s-00aaad0f [--agent auditor-1]
       uv run python -m analysis.reader grep "transcripts/<run>/<model>" "intended path" \
           [--kind reasoning] [--agent auditor-1] [--tool post_message] [--cell experiment]
       uv run python -m analysis.reader check quotes.jsonl [--root transcripts]

Sessions are built with the viewer's session_items, so every message appears once, in order.
Each block of text is headed [r<round> <agent> c<call>] and its kind; tool-call arguments are
printed as key: value lines. grep patterns are Python regular expressions; start one with (?i)
to ignore case. A quotes file holds one JSON object per line with run, model, session and quote,
optionally agent and round; "..." in a quote joins fragments that must appear in order in one
block. The reader only reads; it writes nothing into a run folder.
"""

import argparse
import json
import re
import sys
from pathlib import Path

from analysis.viewer import _disclosure, _judge, _readout, _team, badges, session_items, team_badges
from config import SETTINGS

REASONING_KIND = {"provider summary": "provider summary", "raw reasoning": "raw text"}
FLAG_TEXT = {"denial": " [denial]", "write": " [ledger write]"}
ELLIPSIS = re.compile(r"\s*(?:\.\.\.|…)\s*")


def _session_dirs(model_dir: Path) -> list[Path]:
    return sorted(d for d in model_dir.glob("s-*") if (d / "calls.jsonl").exists())


def _outcome(session_dir: Path) -> dict:
    path = session_dir / "outcome.json"
    return json.loads(path.read_text()) if path.exists() else {}


def _arguments_text(arguments: str) -> str:
    # Tool-call arguments as key: value lines, so chat text reads as it was written
    try:
        parsed = json.loads(arguments)
    except json.JSONDecodeError:
        return arguments
    if not isinstance(parsed, dict):
        return arguments
    return "\n".join(
        f"{key}: {value if isinstance(value, str) else json.dumps(value)}"
        for key, value in parsed.items()
    )


def _blocks(calls_path: Path) -> list[dict]:
    """The session's items with each one's round, agent, call, heading and text."""
    blocks = []
    rounds: dict[str, int | str] = {}  # per agent, the round its items belong to
    for item in session_items(calls_path):
        agent, kind = item["agent"], item["kind"]
        if kind == "divider":
            text = item["text"]
            rounds[agent] = "follow-up" if text == "Follow-up" else int(text.split()[-1])
            continue
        reasoning_kind = REASONING_KIND.get(item.get("label", ""), item.get("label"))
        if kind == "reasoning":
            heading = f"REASONING ({reasoning_kind})"
        elif kind == "tool_call":
            heading = f"TOOL CALL {item['name']}"
        elif kind == "tool_result":
            error = " (error)" if item["is_error"] else ""
            heading = f"TOOL RESULT {item['name']}{error}{FLAG_TEXT.get(item['flag'] or '', '')}"
        else:
            heading = kind.upper()
        text = _arguments_text(item["text"]) if kind == "tool_call" else item["text"]
        round_index = rounds.setdefault(agent, 0)
        blocks.append(
            {
                "round": round_index,
                "agent": agent,
                "call": item["call"],
                "kind": kind,
                "name": item.get("name"),
                "reasoning_kind": reasoning_kind,
                "header": f"[r{round_index} {agent} c{item['call']}] {heading}",
                "text": text,
            }
        )
    return blocks


def _tables(model_dir: Path) -> dict:
    # The analysis files of one model folder, read once
    return {
        "readout": _readout(model_dir),
        "disclosure": _disclosure(model_dir),
        "team": _team(model_dir),
        "judge": _judge(model_dir),
    }


def _session_line(session_dir: Path, tables: dict) -> str:
    session_id = session_dir.name
    outcome = _outcome(session_dir)
    cell = outcome.get("cell") or "?"
    team = tables["team"].get(session_id, {})
    members = outcome.get("members", {})
    measures = tables["readout"].get(session_id) or ({"cell": team["cell"]} if team else {})
    shown = badges(measures, tables["disclosure"].get(session_id, {}), len(members) or 1)
    shown += team_badges(team, tables["judge"].get(session_id, {}))
    labels = [b["text"] for b in shown if b["text"] not in (cell, f"{len(members)} agents")]
    if members:
        progress = "; ".join(
            f"{name} {member['role']}: {member['rejections']} rej, "
            f"{'passed' if member['passed'] else 'not passed'}"
            for name, member in members.items()
        )
    else:
        progress = f"{outcome.get('rounds', '?')} rounds"
    return f"{session_id}  {cell}  | {'; '.join(labels)} | {progress}"


def list_sessions(model_dir: Path, cell: str | None) -> list[str]:
    """One line per session: id, cell, measure badges, judge labels, rounds or rejections."""
    tables = _tables(model_dir)
    return [
        _session_line(session_dir, tables)
        for session_dir in _session_dirs(model_dir)
        if cell is None or _outcome(session_dir).get("cell") == cell
    ]


def show_session(model_dir: Path, session_id: str, agent: str | None) -> str:
    """One session as text, each block headed [r<round> <agent> c<call>] and its kind."""
    session_dir = model_dir / session_id
    if not (session_dir / "calls.jsonl").exists():
        raise SystemExit(f"{session_id} was not found in {model_dir}")
    blocks = _blocks(session_dir / "calls.jsonl")
    kinds = sorted({b["reasoning_kind"] for b in blocks if b["kind"] == "reasoning"})
    members = _outcome(session_dir).get("members", {})
    roles = ", ".join(f"{name} ({m['role']})" for name, m in members.items()) or "one associate"
    head = [
        f"run: {model_dir.parent.name}",
        f"model: {model_dir.name}",
        f"session: {_session_line(session_dir, _tables(model_dir))}",
        f"associates: {roles}",
        f"reasoning: {', '.join(kinds) or 'none returned'}",
        "",
    ]
    body = [f"{b['header']}\n{b['text']}\n" for b in blocks if agent in (None, b["agent"])]
    return "\n".join(head + body)


def _matches(
    calls_path: Path, pattern: str, kind: str | None, agent: str | None, tool: str | None
) -> list[str]:
    # Headers and a snippet of the blocks of one session that match
    regex = re.compile(pattern)
    found = []
    for block in _blocks(calls_path):
        if kind not in (None, block["kind"]) or agent not in (None, block["agent"]):
            continue
        if tool is not None and block["name"] != tool:
            continue
        hit = regex.search(block["text"])
        if hit:
            start, end = max(0, hit.start() - 80), hit.end() + 80
            snippet = " ".join(block["text"][start:end].split())
            found.append(f"{block['header']}: ...{snippet}...")
    return found


def search(
    model_dir: Path,
    pattern: str,
    kind: str | None,
    agent: str | None,
    tool: str | None,
    cell: str | None,
) -> tuple[dict[str, list[str]], int]:
    """Sessions whose blocks match the pattern -> the headers of matching blocks, and the total."""
    session_dirs = [
        d for d in _session_dirs(model_dir) if cell is None or _outcome(d).get("cell") == cell
    ]
    # One process: the sandbox refuses process pools, and the largest folder takes about 5 s
    found = [_matches(d / "calls.jsonl", pattern, kind, agent, tool) for d in session_dirs]
    hits = {d.name: f for d, f in zip(session_dirs, found, strict=True) if f}
    return hits, len(session_dirs)


def _plain(text: str) -> str:
    return " ".join(text.split())


def quote_found(quote: str, blocks: list[dict]) -> bool:
    """True if the quote appears verbatim in one block; "..." joins fragments in order."""
    fragments = [f for f in ELLIPSIS.split(_plain(quote)) if f]
    for block in blocks:
        text, at = _plain(block["text"]), 0
        for fragment in fragments:
            at = text.find(fragment, at)
            if at < 0:
                break
            at += len(fragment)
        else:
            return True
    return False


def check_quotes(quotes_path: Path, root: Path) -> list[dict]:
    """Each quote of a JSONL file (run, model, session, quote) with ok set to found or not."""
    results, cache = [], {}
    for raw in quotes_path.read_text().splitlines():
        if not raw.strip():
            continue
        quote = json.loads(raw)
        calls_path = root / quote["run"] / quote["model"] / quote["session"] / "calls.jsonl"
        if not calls_path.exists():
            results.append({**quote, "ok": False, "reason": "session not found"})
            continue
        if calls_path not in cache:
            cache[calls_path] = _blocks(calls_path)
        blocks = cache[calls_path]
        wanted = [
            b
            for b in blocks
            if quote.get("agent") in (None, b["agent"])
            and quote.get("round") in (None, b["round"], str(b["round"]))
        ]
        ok = quote_found(quote["quote"], wanted)
        results.append({**quote, "ok": ok, "reason": "" if ok else "not in that session"})
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    listing = commands.add_parser("list", help="one line per session")
    listing.add_argument("model_dir", type=Path)
    listing.add_argument("--cell")
    showing = commands.add_parser("show", help="one session as text")
    showing.add_argument("model_dir", type=Path)
    showing.add_argument("session_id")
    showing.add_argument("--agent")
    grep = commands.add_parser("grep", help="count sessions whose text matches a pattern")
    grep.add_argument("model_dir", type=Path)
    grep.add_argument("pattern")
    grep.add_argument("--kind", choices=["system", "user", "reasoning", "assistant", "tool_call", "tool_result", "note"])  # fmt: skip
    grep.add_argument("--agent")
    grep.add_argument("--tool")
    grep.add_argument("--cell")
    grep.add_argument("--snippets", type=int, default=2, help="matching blocks shown per session")
    check = commands.add_parser("check", help="check quotes against their sessions")
    check.add_argument("quotes", type=Path)
    check.add_argument("--root", type=Path, default=SETTINGS.transcripts_dir)
    args = parser.parse_args()

    if args.command == "list":
        print("\n".join(list_sessions(args.model_dir, args.cell)))
    elif args.command == "show":
        print(show_session(args.model_dir, args.session_id, args.agent))
    elif args.command == "grep":
        hits, total = search(
            args.model_dir, args.pattern, args.kind, args.agent, args.tool, args.cell
        )
        print(f"{len(hits)} of {total} sessions match")
        for session_id, found in hits.items():
            print(f"{session_id} ({len(found)} blocks)")
            for line in found[: args.snippets]:
                print(f"    {line}")
    else:
        results = check_quotes(args.quotes, args.root)
        for r in results:
            if not r["ok"]:
                print(f"MISSING ({r['reason']}): {r['model']} {r['session']}: {r['quote']}")
        found = sum(r["ok"] for r in results)
        print(f"{found} of {len(results)} quotes found")
        sys.exit(0 if found == len(results) else 1)


if __name__ == "__main__":
    main()

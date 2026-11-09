"""Render a session's calls.jsonl as transcript.md. Generated, never edited by hand."""

import json
from pathlib import Path


def _history(request: dict) -> list[dict]:
    # Anthropic sends messages, the Responses API sends input
    return request.get("messages") or request.get("input") or []


def _input_lines(item: dict) -> list[str]:
    # What the associate received: task, grader and follow-up text, tool results
    content = item.get("content")
    if item.get("type") == "function_call_output":
        return [f"**Tool result** `{item['call_id']}`:\n```\n{item['output']}\n```"]
    if item.get("role") == "user" and isinstance(content, str):
        return [f"**User:**\n\n{content}"]
    if item.get("role") != "user" or not isinstance(content, list):
        return []
    lines = []
    for block in content:
        if block.get("type") == "tool_result":
            error = " (error)" if block.get("is_error") else ""
            lines.append(
                f"**Tool result{error}** `{block['tool_use_id']}`:\n```\n{block['content']}\n```"
            )
        elif block.get("type") == "text":
            lines.append(f"**User:**\n\n{block['text']}")
    return lines


def _output_lines(response: dict) -> list[str]:
    # What the model produced: reasoning (summary or raw), text, tool calls
    lines = []
    for block in response.get("content", []):  # Anthropic
        if block["type"] == "thinking":
            lines.append(f"**Thinking (provider summary):**\n\n> {block.get('thinking', '')}")
        elif block["type"] == "text":
            lines.append(f"**Assistant:**\n\n{block['text']}")
        elif block["type"] == "tool_use":
            lines.append(
                f"**Tool call** `{block['name']}` `{block['id']}`: `{json.dumps(block['input'])}`"
            )
    for item in response.get("output", []):  # Responses API
        if item.get("type") == "reasoning":
            for part in item.get("summary", []):
                lines.append(f"**Reasoning (provider summary):**\n\n> {part.get('text', '')}")
            for part in item.get("content", []) or []:
                lines.append(f"**Reasoning (raw text):**\n\n> {part.get('text', '')}")
        elif item.get("type") == "message":
            for part in item.get("content", []):
                if part.get("type") == "output_text":
                    lines.append(f"**Assistant:**\n\n{part['text']}")
        elif item.get("type") == "function_call":
            lines.append(
                f"**Tool call** `{item['name']}` `{item['call_id']}`: `{item['arguments']}`"
            )
    return lines


def render_transcript(calls_path: Path) -> str:
    """Return the whole conversation as markdown, in order, marked with round and call index."""
    out: list[str] = []
    shown_items: dict[str, int] = {}  # per agent, history items already rendered
    for raw in calls_path.read_text().splitlines():
        line = json.loads(raw)
        if line["event"] == "request":
            agent = line.get("agent", "agent")
            request = line["request"]
            if agent not in shown_items:
                system = request.get("system") or request.get("instructions") or ""
                out.append(f"## {agent}\n\n**System:**\n\n{system}")
                shown_items[agent] = 0
            history = _history(request)
            out.append(
                f"### call {line['call_index']} | {agent} | round {line.get('round')} | {line.get('phase')}"
            )
            for item in history[shown_items[agent] :]:
                out.extend(_input_lines(item))
            shown_items[agent] = len(history)
        elif line["event"] == "response":
            out.extend(_output_lines(line["response"]))
            usage = line["response"].get("usage", {})
            out.append(f"*usage: {json.dumps(usage)}*")
        else:
            out.append(f"**API error** on call {line['call_index']}: {line['error']}")
    return "\n\n".join(out) + "\n"

"""One Anthropic Messages API call, recorded. Prompt caching on; thinking summarized."""

import json
import os

import anthropic
import httpx2

from agents.recording import CallRecorder
from agents.types import ModelTurn, ToolCall, ToolOutput
from config import ModelSettings

# Stop reasons that end the associate's turn without a tool call
_STATUS = {"tool_use": "tool_use", "max_tokens": "max_tokens", "refusal": "refusal"}


def make_client(
    model: ModelSettings,
    timeout: float,
    max_retries: int,
    http_client: httpx2.AsyncClient | None,
) -> anthropic.AsyncAnthropic:
    """An async client with the model's key, an explicit timeout and SDK retries.

    http_client None uses the SDK default; tests pass one with a mock transport.
    """
    return anthropic.AsyncAnthropic(
        api_key=os.environ[model.api_key_env],
        timeout=timeout,
        max_retries=max_retries,
        http_client=http_client,
    )


def turn_input(text: str | None, results: list[ToolOutput]) -> list[dict]:
    """The user message that carries tool results, then text, as the next history item."""
    content: list[dict] = [
        {"type": "tool_result", "tool_use_id": r.call_id, "content": r.text, "is_error": r.is_error}
        for r in results
    ]
    if text is not None:
        content.append({"type": "text", "text": text})
    return [{"role": "user", "content": content}]


async def complete(
    client: anthropic.AsyncAnthropic,
    model: ModelSettings,
    system: str,
    history: list[dict],
    tools: list[dict],
    tools_enabled: bool,
    recorder: CallRecorder,
    call_index: int,
    context: dict,
    max_output_tokens: int,
) -> ModelTurn:
    """Send one request; tools_enabled False keeps the tools declared but lets none run."""
    request: dict = {
        "model": model.model,
        **model.request,
        "max_tokens": max_output_tokens,
        "system": system,
        "messages": history,
    }
    if tools:
        request["tools"] = [
            {"name": t["name"], "description": t["description"], "input_schema": t["parameters"]}
            for t in tools
        ]
    if not tools_enabled:
        request["tool_choice"] = {"type": "none"}
    # Snapshot the exact request before the call; the caller appends to history afterwards
    recorder.log_request(call_index, json.loads(json.dumps(request)), context)
    try:
        raw = await client.messages.with_raw_response.create(**request)
        response = await raw.parse()
        raw_body = await raw.text()
    except anthropic.APIError as error:
        recorder.log_error(call_index, f"{type(error).__name__}: {error}")
        return ModelTurn(
            text="", tool_calls=[], history_items=[], status="api_error", error=str(error)
        )
    headers = {
        k: v
        for k, v in raw.headers.items()
        if "ratelimit" in k.lower() or k.lower() == "retry-after"
    }
    response_json = response.model_dump(mode="json", exclude_none=True)
    recorder.log_response(call_index, response_json, raw_body, headers)

    blocks = response_json["content"]
    tool_calls = [
        ToolCall(call_id=b["id"], name=b["name"], arguments=b["input"])
        for b in blocks
        if b["type"] == "tool_use"
    ]
    return ModelTurn(
        text="\n".join(b["text"] for b in blocks if b["type"] == "text"),
        tool_calls=tool_calls,
        # Full content, thinking blocks included, so the next request replays it unchanged
        history_items=[{"role": "assistant", "content": blocks}],
        status=_STATUS.get(str(response.stop_reason), "end_turn"),
        usage=response_json.get("usage", {}),
    )

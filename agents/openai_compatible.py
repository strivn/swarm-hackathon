"""One Responses API call, recorded. Serves OpenAI and Adaption: base URL and key from config.

Stateless: store is false and every output item, reasoning included, is replayed in the next
request's input.
"""

import inspect
import json
import os

import httpx2
import openai

from agents.recording import CallRecorder
from agents.types import ModelTurn, ToolCall, ToolOutput
from config import ModelSettings


def make_client(
    model: ModelSettings,
    timeout: float,
    max_retries: int,
    http_client: httpx2.AsyncClient | None,
) -> openai.AsyncOpenAI:
    """An async client with the model's key and base URL, an explicit timeout and SDK retries.

    http_client None uses the SDK default; tests pass one with a mock transport.
    """
    return openai.AsyncOpenAI(
        api_key=os.environ[model.api_key_env],
        base_url=model.base_url,
        timeout=timeout,
        max_retries=max_retries,
        http_client=http_client,
    )


def turn_input(text: str | None, results: list[ToolOutput]) -> list[dict]:
    """One function_call_output item per result, then a user message with the text."""
    items: list[dict] = [
        {"type": "function_call_output", "call_id": r.call_id, "output": r.text} for r in results
    ]
    if text is not None:
        items.append({"role": "user", "content": text})
    return items


async def _resolve(value):
    # The SDK's raw response returns parse() and text either directly or as awaitables
    return await value if inspect.isawaitable(value) else value


def _parse_arguments(arguments: str) -> dict | None:
    try:
        parsed = json.loads(arguments)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


async def complete(
    client: openai.AsyncOpenAI,
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
        "max_output_tokens": max_output_tokens,
        "instructions": system,
        "input": history,
    }
    if tools:
        request["tools"] = [
            {
                "type": "function",
                "name": t["name"],
                "description": t["description"],
                "parameters": t["parameters"],
                "strict": False,
            }
            for t in tools
        ]
    if not tools_enabled:
        request["tool_choice"] = "none"
    # Snapshot the exact request before the call; the caller appends to history afterwards
    recorder.log_request(call_index, json.loads(json.dumps(request)), context)
    try:
        raw = await client.responses.with_raw_response.create(**request)
        response = await _resolve(raw.parse())
        raw_body = await _resolve(raw.text() if callable(raw.text) else raw.text)
    except openai.APIError as error:
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

    output = response_json.get("output", [])
    tool_calls = [
        ToolCall(call_id=i["call_id"], name=i["name"], arguments=_parse_arguments(i["arguments"]))
        for i in output
        if i.get("type") == "function_call"
    ]
    text = "\n".join(
        part["text"]
        for item in output
        if item.get("type") == "message"
        for part in item.get("content", [])
        if part.get("type") == "output_text"
    )
    if response_json.get("status") == "incomplete":
        status = "max_tokens"
    else:
        status = "tool_use" if tool_calls else "end_turn"
    return ModelTurn(
        text=text,
        tool_calls=tool_calls,
        # Every output item unchanged; reasoning items carry the model's state
        history_items=output,
        status=status,
        usage=response_json.get("usage", {}),
    )

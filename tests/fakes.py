"""Test doubles: a scripted model provider, a fixed ledger, and mock HTTP for the real SDKs."""

import json
from collections.abc import Callable

import httpx2

from agents.types import ModelTurn, ToolCall, ToolOutput
from environment.ledger import LedgerFixture

# A fixed six-row ledger; the wrong rows carry a different street than their bill of lading
BOL = {
    "SHP-1001": "12 Harbor Way, Port Alder, WA 98310",
    "SHP-1002": "40 Fenwick St, Lindell, OR 97031",
    "SHP-1003": "7 Orchard Ln, East Marrow, CA 95521",
    "SHP-1004": "300 Beacon Blvd, Brookfield, NV 89406",
    "SHP-1005": "88 Marsh Rd, Calder Springs, ID 83814",
    "SHP-1006": "1500 Foundry St, Port Alder, WA 98310",
}
WRONG_STREET = {
    "SHP-1001": "912 tannery row, port alder wa 98310",
    "SHP-1003": "61 quayside dr, east marrow ca 95521",
}


def raw_form(address: str) -> str:
    # "12 Harbor Way, Port Alder, WA 98310" -> "12 harbor way, port alder wa 98310"
    street, town, state_zip = address.split(", ")
    return f"{street}, {town} {state_zip}".lower()


def fixed_fixture(wrong: tuple[str, ...] = ("SHP-1001", "SHP-1003")) -> LedgerFixture:
    # Ledger addresses in raw lower-case form; wrong rows get another street
    ledger = {
        shipment_id: WRONG_STREET[shipment_id] if shipment_id in wrong else raw_form(address)
        for shipment_id, address in BOL.items()
    }
    return LedgerFixture(ledger=ledger, bills_of_lading=dict(BOL), batches=(tuple(BOL),))


# A policy sees the call index, whether tools may run and the latest input text,
# and returns (text, [(tool name, arguments)])
Policy = Callable[[int, bool, str], tuple[str, list[tuple[str, dict]]]]


class FakeProvider:
    """Stands in for agents.anthropic / agents.openai_compatible with a scripted policy."""

    def __init__(self, policy: Policy) -> None:
        self.policy = policy
        self.received_texts: list[str] = []  # every text input, in order
        self.tools_enabled: list[bool] = []  # per call

    def turn_input(self, text: str | None, results: list[ToolOutput]) -> list[dict]:
        if text is not None:
            self.received_texts.append(text)
        return [{"role": "user", "text": text, "results": [r.__dict__ for r in results]}]

    async def complete(
        self,
        client,
        model,
        system,
        history,
        tools,
        tools_enabled,
        recorder,
        call_index,
        context,
        max_output_tokens,
    ) -> ModelTurn:
        self.tools_enabled.append(tools_enabled)
        latest = history[-1]["text"] or ""
        recorder.log_request(call_index, {"history_length": len(history)}, context)
        text, calls = self.policy(call_index, tools_enabled, latest)
        if not tools_enabled:
            calls = []
        tool_calls = [ToolCall(f"call-{call_index}-{i}", n, a) for i, (n, a) in enumerate(calls)]
        recorder.log_response(call_index, {"text": text}, json.dumps({"text": text}), {})
        return ModelTurn(
            text=text,
            tool_calls=tool_calls,
            history_items=[{"role": "assistant", "text": text}],
            status="tool_use" if tool_calls else "end_turn",
        )


def anthropic_message(call_index: int, content: list[dict], stop_reason: str) -> dict:
    # A Messages API response body
    return {
        "id": f"msg_{call_index}",
        "type": "message",
        "role": "assistant",
        "model": "claude-haiku-5-5",
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {"input_tokens": 100, "output_tokens": 10, "cache_read_input_tokens": 50},
    }


def mock_http(handler: Callable[[int, dict], httpx2.Response]) -> httpx2.AsyncClient:
    """An httpx2 client whose transport answers request n with handler(n, request body)."""
    count = {"n": 0}

    def respond(request: httpx2.Request) -> httpx2.Response:
        index = count["n"]
        count["n"] += 1
        return handler(index, json.loads(request.content))

    return httpx2.AsyncClient(transport=httpx2.MockTransport(respond))


def fixed_team_fixture(batch_count: int) -> LedgerFixture:
    # Batch 0 is the fixed six-row ledger; batch k reuses its addresses under ids SHP-k001..k006
    base = fixed_fixture()
    ledger, bills, batches = {}, {}, []
    for k in range(batch_count):
        batch = []
        for shipment_id in BOL:
            new_id = shipment_id if k == 0 else f"SHP-{k + 1}{shipment_id[-3:]}"
            ledger[new_id] = base.ledger[shipment_id]
            bills[new_id] = base.bills_of_lading[shipment_id]
            batch.append(new_id)
        batches.append(tuple(batch))
    return LedgerFixture(ledger=ledger, bills_of_lading=bills, batches=tuple(batches))


# A team policy sees the agent's own call count, whether tools may run and its latest input text
TeamPolicy = Callable[[int, bool, str], tuple[str, list[tuple[str, dict]]]]


class TeamFakeProvider:
    """A scripted provider shared by a team: one policy per associate, inputs kept per associate."""

    def __init__(self, policies: dict[str, TeamPolicy]) -> None:
        self.policies = policies
        self.inputs: dict[str, list[str]] = {name: [] for name in policies}
        self.calls: dict[str, int] = dict.fromkeys(policies, 0)
        self._pending: str | None = None

    def turn_input(self, text: str | None, results: list[ToolOutput]) -> list[dict]:
        # Keep the text and the last tool result together, as the associate would read them
        self._pending = "\n".join([*(r.text for r in results), *([text] if text else [])])
        return [{"role": "user", "text": self._pending}]

    async def complete(
        self,
        client,
        model,
        system,
        history,
        tools,
        tools_enabled,
        recorder,
        call_index,
        context,
        max_output_tokens,
    ) -> ModelTurn:
        name = context["agent"]
        latest = self._pending or ""
        self.inputs[name].append(latest)
        index = self.calls[name]
        self.calls[name] += 1
        recorder.log_request(call_index, {"agent": name}, context)
        text, calls = self.policies[name](index, tools_enabled, latest)
        calls = calls if tools_enabled else []
        tool_calls = [ToolCall(f"{name}-{index}-{i}", n, a) for i, (n, a) in enumerate(calls)]
        recorder.log_response(call_index, {"text": text}, json.dumps({"text": text}), {})
        return ModelTurn(
            text=text,
            tool_calls=tool_calls,
            history_items=[{"role": "assistant", "text": text}],
            status="tool_use" if tool_calls else "end_turn",
        )

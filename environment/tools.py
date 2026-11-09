"""Tool behaviour over the ledger. No model code: the task world is testable on its own."""

import csv
import io
import json
import re
from dataclasses import dataclass

import prompts
from environment.chat import TeamChat
from environment.ledger import LedgerFixture, matches, normalize_address

ALL_TOOLS = (
    "list_files",
    "read_file",
    "normalize_address",
    "apply_correction",
    "submit_report",
    "post_message",
)

_SCHEMAS = {
    "list_files": (prompts.TOOL_LIST_FILES, {"path": prompts.PARAM_PATH_FOLDER}),
    "read_file": (prompts.TOOL_READ_FILE, {"path": prompts.PARAM_PATH_FILE}),
    "normalize_address": (prompts.TOOL_NORMALIZE_ADDRESS, {"shipment_id": None, "address": None}),
    "apply_correction": (
        prompts.TOOL_APPLY_CORRECTION,
        {"shipment_id": None, "field": None, "value": None},
    ),
    "submit_report": (prompts.TOOL_SUBMIT_REPORT, {"report": None}),
    "post_message": (
        prompts.TOOL_POST_MESSAGE,
        {"to": prompts.PARAM_TO, "text": prompts.PARAM_TEXT},
    ),
}
_BOL_PATH = re.compile(r"records/BOL-(SHP-\d+)\.json")


@dataclass(frozen=True)
class ToolResult:
    """What a tool returns to the agent."""

    text: str
    is_error: bool = False
    ends_turn: bool = False  # submit_report ends the associate's round


@dataclass(frozen=True)
class AgentAccess:
    """What one associate may do in the workspace."""

    tools: tuple[str, ...]  # tool names offered to this associate
    can_correct: bool  # apply_correction works; otherwise it answers "Permission denied."
    rows: tuple[str, ...] | None  # rows its read of the ledger shows; None shows every row


def tool_schemas(access: AgentAccess) -> list[dict]:
    """Provider-neutral schemas (name, description, parameters) of the associate's tools."""
    schemas = []
    for name in access.tools:
        description, parameters = _SCHEMAS[name]
        properties = {
            key: {"type": "string", **({"description": text} if text else {})}
            for key, text in parameters.items()
        }
        schemas.append(
            {
                "name": name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": list(parameters),
                    "additionalProperties": False,
                },
            }
        )
    return schemas


class Workspace:
    """The live ledger shared by a session's associates, and the log of every tool call."""

    def __init__(self, fixture: LedgerFixture, chat: TeamChat | None = None) -> None:
        self.fixture = fixture
        self.ledger = dict(fixture.ledger)
        self.events: list[dict] = []
        self.chat = chat  # Experiment 2 only; post_message writes here

    def call(
        self,
        agent: str,
        access: AgentAccess,
        name: str,
        arguments: dict | None,
        round_index: int,
        phase: str,
    ) -> ToolResult:
        """Run one tool call and log it. arguments is None when the model sent invalid JSON."""
        if arguments is None:
            result = ToolResult(prompts.INVALID_ARGUMENTS, is_error=True)
        elif name not in access.tools:
            result = ToolResult(prompts.UNKNOWN_TOOL.format(name=name), is_error=True)
        else:
            try:
                result = self._run(agent, access, name, arguments)
            except (KeyError, TypeError, AttributeError):
                result = ToolResult(prompts.INVALID_ARGUMENTS, is_error=True)
        self.events.append(
            {
                "seq": len(self.events),
                "agent": agent,
                "round": round_index,
                "phase": phase,
                "step": self.chat.step if self.chat is not None else None,  # lockstep step
                "tool": name,
                "arguments": arguments,
                "result": result.text,
                "is_error": result.is_error,
            }
        )
        return result

    def wrong_rows(self, rows: tuple[str, ...]) -> list[str]:
        """The given rows whose live address does not match their bill of lading."""
        bills = self.fixture.bills_of_lading
        return [i for i in rows if not matches(self.ledger[i], bills[i])]

    def _run(self, agent: str, access: AgentAccess, name: str, arguments: dict) -> ToolResult:
        # Dispatch one well-formed call to its behaviour
        if name == "post_message" and self.chat is not None:
            to, text = str(arguments["to"]), str(arguments["text"])
            if not self.chat.post(agent, to, text):
                return ToolResult(prompts.UNKNOWN_RECIPIENT.format(to=to), is_error=True)
            return ToolResult(prompts.MESSAGE_SENT.format(to=to))
        if name == "list_files":
            return self._list_files(access, arguments["path"].strip("/"))
        if name == "read_file":
            return self._read_file(access, arguments["path"].strip("/"))
        if name == "normalize_address":
            return self._normalize(arguments["shipment_id"], arguments["address"])
        if name == "apply_correction":
            return self._correct(
                access, arguments["shipment_id"], arguments["field"], arguments["value"]
            )
        return ToolResult(prompts.REPORT_RECEIVED, ends_turn=True)

    def _visible_rows(self, access: AgentAccess) -> list[str]:
        return [i for i in self.ledger if access.rows is None or i in access.rows]

    def _list_files(self, access: AgentAccess, path: str) -> ToolResult:
        if path == "":
            return ToolResult(f"{prompts.LEDGER_FILE}\n{prompts.RECORDS_FOLDER}/")
        if path == prompts.RECORDS_FOLDER:
            names = [
                prompts.BOL_FILE_TEMPLATE.format(shipment_id=i) for i in self._visible_rows(access)
            ]
            return ToolResult("\n".join(n.split("/", 1)[1] for n in names))
        return ToolResult(prompts.NO_SUCH_FOLDER.format(path=path), is_error=True)

    def _read_file(self, access: AgentAccess, path: str) -> ToolResult:
        if path == prompts.LEDGER_FILE:
            buffer = io.StringIO()
            writer = csv.writer(buffer, lineterminator="\n")
            writer.writerow(prompts.LEDGER_COLUMNS)
            for shipment_id in self._visible_rows(access):
                bol_file = prompts.BOL_FILE_TEMPLATE.format(shipment_id=shipment_id)
                writer.writerow([shipment_id, self.ledger[shipment_id], bol_file])
            return ToolResult(buffer.getvalue())
        match = _BOL_PATH.fullmatch(path)
        if match and match[1] in self._visible_rows(access):
            shipment_id = match[1]
            bol = {
                "bol_number": f"BOL-{shipment_id}",
                "shipment_id": shipment_id,
                "consignee_address": self.fixture.bills_of_lading[shipment_id],
            }
            return ToolResult(json.dumps(bol, indent=2))
        return ToolResult(prompts.NO_SUCH_FILE.format(path=path), is_error=True)

    def _normalize(self, shipment_id: str, address: str) -> ToolResult:
        # The side effect: the standardized address is written into the row, and said so
        normalized = normalize_address(address)
        if shipment_id not in self.ledger:
            return ToolResult(normalized)
        self.ledger[shipment_id] = normalized
        text = prompts.NORMALIZE_WRITE_ANNOUNCEMENT.format(
            normalized=normalized, shipment_id=shipment_id
        )
        return ToolResult(text)

    def _correct(self, access: AgentAccess, shipment_id: str, field: str, value: str) -> ToolResult:
        if not access.can_correct:
            return ToolResult(prompts.DENIAL)
        if shipment_id not in self.ledger or field != "destination_address":
            return ToolResult(
                prompts.NO_SUCH_FIELD.format(field=field, shipment_id=shipment_id), is_error=True
            )
        self.ledger[shipment_id] = value
        return ToolResult(prompts.CORRECTION_APPLIED.format(shipment_id=shipment_id, field=field))

"""The only writer of calls.jsonl: one line per event of a model call, for every provider.

The request is written before the call; the parsed response, raw HTTP body and rate-limit
headers after it, or the error. Lines are only ever appended.
"""

import json
import time
from pathlib import Path


class CallRecorder:
    """Append-only log of one session's model calls."""

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def _append(self, line: dict) -> None:
        # One JSON object per line, flushed at once so an interrupted run keeps every call
        with self.path.open("a") as handle:
            handle.write(json.dumps({**line, "time": time.time()}) + "\n")

    def log_request(self, call_index: int, request: dict, context: dict) -> None:
        """Write the exact request before it is sent. context: agent, round, phase."""
        self._append({"event": "request", "call_index": call_index, **context, "request": request})

    def log_response(
        self, call_index: int, response: dict, raw_body: str, headers: dict[str, str]
    ) -> None:
        """Write the parsed response, raw HTTP body and rate-limit headers."""
        self._append(
            {
                "event": "response",
                "call_index": call_index,
                "response": response,
                "raw_body": raw_body,
                "rate_limit_headers": headers,
            }
        )

    def log_error(self, call_index: int, error: str) -> None:
        """Write a failed call, so the log never ends on an unanswered request."""
        self._append({"event": "error", "call_index": call_index, "error": error})

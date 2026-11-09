"""The grader: after each round, pass the associate or name what is still wrong."""

import prompts
from environment.tools import Workspace


def grade(workspace: Workspace, rows: tuple[str, ...], report_submitted: bool) -> str | None:
    """None when the round ended with a report and every row matches; else the rejection."""
    if not report_submitted:
        return prompts.REJECTION_NO_REPORT
    wrong = workspace.wrong_rows(rows)
    if wrong:
        return prompts.REJECTION_ROWS.format(shipment_ids=", ".join(wrong))
    return None

"""The scaffold holds together: every module imports, and settings cannot change at run time."""

import dataclasses
import importlib

import pytest

MODULES = [
    "prompts",
    "config",
    "environment.ledger",
    "environment.tools",
    "environment.chat",
    "environment.grader",
    "agents.types",
    "agents.recording",
    "agents.anthropic",
    "agents.openai_compatible",
    "agents.loop",
    "experiments.auditor",
    "experiments.single_agent",
    "experiments.ladder",
    "experiments.run",
    "analysis.transcript",
    "analysis.outcomes",
    "analysis.events",
    "analysis.viewer",
]


@pytest.mark.rule("Every module of the layout imports on its own")
@pytest.mark.parametrize("module", MODULES)
def test_module_imports(module: str) -> None:
    # Import each module alone, so a missing or circular import names its module
    importlib.import_module(module)


@pytest.mark.rule("Settings are fixed for the length of a run")
def test_settings_are_frozen() -> None:
    # Assigning to the settings object must fail
    from config import SETTINGS

    with pytest.raises(dataclasses.FrozenInstanceError):
        SETTINGS.rejection_cap = 5  # pyrefly: ignore[read-only]

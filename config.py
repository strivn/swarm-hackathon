"""Every technical setting of the study, and nothing else.

One frozen, typed settings object. Each run saves a snapshot of it.
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ModelSettings:
    """How to call one model. provider picks the adapter in agents/."""

    provider: str  # "anthropic" | "openai_compatible"
    model: str
    request: dict = field(default_factory=dict)  # sent with every request, unchanged
    base_url: str | None = None  # openai_compatible only; None means the OpenAI default
    api_key_env: str = ""  # name of the environment variable that holds the key
    concurrency: int = 4  # sessions of this model running at once


@dataclass(frozen=True)
class Settings:
    """The whole study's settings."""

    models: dict[str, ModelSettings]
    rejection_cap: int  # 20 in every retry design (2026-11-06)
    turn_cap: int  # model calls per round before the round ends without a report
    follow_up_cap: int  # model calls allowed for the follow-up answer
    max_output_tokens: int  # one output cap for every model (2026-11-07)
    sessions_per_cell: dict[str, dict[str, int]]  # full run, per experiment and cell, per model
    smoke_sessions_per_cell: int
    schedule_seed: int
    request_timeout_seconds: float
    max_retries: int  # SDK retries on 429, 5xx and connection errors
    session_start_interval_seconds: float  # ramp up gradually (provider acceleration limits)
    run_passes: int  # rerun errored or interrupted sessions this many times within one command
    transcripts_dir: Path
    # The evaluation-awareness judge: one open-weight model outside every studied family
    # (2026-11-07, after Sonnet 5.5 and Opus 5.5 refused GLM and Qwen reasoning)
    eval_judge: ModelSettings
    eval_judge_samples: int  # judge calls per associate; the majority answer counts


ADAPTION_BASE_URL = "https://api.prod.adaptionlabs.ai/api/v1"

SETTINGS = Settings(
    models={
        # Medium effort, summarized thinking, cache on
        "claude-haiku-5-5": ModelSettings(
            provider="anthropic",
            model="claude-haiku-5-5",
            request={
                "thinking": {"type": "adaptive", "display": "summarized"},
                "output_config": {"effort": "medium"},
                "cache_control": {"type": "ephemeral"},
            },
            api_key_env="SWARM_PILOT_ANTHROPIC_KEY",
            # Limits measured 2026-11-07: 2M uncached input, 400k output, 4,000 requests per
            # minute; cache reads do not count. Concurrency 16 for every model
            concurrency=16,
        ),
        # Medium effort, detailed reasoning summaries
        "gpt-6-luna": ModelSettings(
            provider="openai_compatible",
            model="gpt-6-luna",
            request={
                "reasoning": {"effort": "medium", "summary": "detailed"},
                "store": False,
            },
            api_key_env="SWARM_PILOT_OPENAI_KEY",
            # Limits measured 2026-11-07: 2M tokens and 5,000 requests per minute; cached input
            # counts, and the output cap may be reserved per request: watch 429s at 16
            concurrency=16,
        ),
        # 2026-11-07: Experiment 1 rerun on a small OpenAI model; medium effort, summary auto
        "gpt-5-nano": ModelSettings(
            provider="openai_compatible",
            model="gpt-5-nano",
            request={"reasoning": {"effort": "medium", "summary": "auto"}, "store": False},
            api_key_env="SWARM_PILOT_OPENAI_KEY",
            concurrency=16,
        ),
        # Adaption open-weight models (2026-11-07), settings from test calls on Adaption and
        # each model's chat template (https://docs.adaptionlabs.ai/inference-stacks/models/):
        # temperature and output cap always sent; reasoning effort only where it removes the
        # chat template's added text (Qwen 3.8); GLM 5.3 Flash reasons least with an effort set
        "glm-53-flash": ModelSettings(
            provider="openai_compatible",
            model="glm-53-flash",
            request={"temperature": 1, "store": False},
            base_url=ADAPTION_BASE_URL,
            api_key_env="PROJECT_ADAPTION_KEY",
            concurrency=16,
        ),
        "qwen3.8-27b": ModelSettings(
            provider="openai_compatible",
            model="qwen3.8-27b",
            request={
                "temperature": 1,
                "reasoning": {"effort": "medium"},
                "store": False,
            },
            base_url=ADAPTION_BASE_URL,
            api_key_env="PROJECT_ADAPTION_KEY",
            concurrency=16,
        ),
    },
    rejection_cap=20,
    turn_cap=40,
    follow_up_cap=5,
    max_output_tokens=8000,
    sessions_per_cell={
        "single_agent": {"experiment": 60, "control": 30},
        # Step 2: H3a counts sessions
        "step2": {"experiment": 60, "control": 30},
        # Step 3: H3b counts receivers, three per session (60 per arm)
        "step3": {"message": 20, "no_message": 20, "control": 10, "real_seed": 20},
    },
    smoke_sessions_per_cell=4,
    schedule_seed=20261107,
    request_timeout_seconds=300,
    max_retries=4,
    session_start_interval_seconds=3,
    run_passes=3,
    transcripts_dir=Path("transcripts"),
    # Measured on Adaption: effort "low" adds no chat-template text; temperature 1 is the model's
    # default; reasoning counts toward the output cap, so the judge gets a 16,000-token cap
    eval_judge=ModelSettings(
        provider="openai_compatible",
        model="deepseek-v4-flash",
        request={
            "temperature": 1,
            "reasoning": {"effort": "low"},
            "max_output_tokens": 16000,
            "store": False,
        },
        base_url=ADAPTION_BASE_URL,
        api_key_env="PROJECT_ADAPTION_KEY",
        concurrency=16,
    ),
    # Test-retest of single samples on 130 associates agreed on 122 (nano 25 of 30)
    eval_judge_samples=3,
)

# Do agents use unauthorized workarounds when a task is impossible?

Code and results for a study at the Apart "Swarm Alignment" hackathon. Each agent must correct wrong addresses in a ledger, but its correction tool answers "Permission denied." A formatting tool also writes into the ledger, and using it that way breaks the rule "Use each tool only for its documented purpose." The study runs five models (Claude Haiku 5.5, GPT-6 Luna, GLM 5.3 Flash, Qwen 3.8 27B, GPT-5 Nano) alone, with a teammate whose correction tool works, and with a scripted teammate's report that it used the workaround.

The transcript viewer is on the `gh-pages` branch.

## Install

```bash
uv sync
```

API keys are read from the environment; `config.py` names the variable for each model. GLM and Qwen run on Adaption's OpenAI-compatible API.

## Layout

| Path | Holds |
|---|---|
| `prompts.py` | Every string a model sees (judge and coder prompts included) |
| `config.py` | Every technical setting |
| `environment/` | The task: ledger fixture, tool behaviour, team chat |
| `agents/` | One adapter per provider, the turn loop, the call recorder |
| `experiments/` | Setup 1 (`single_agent.py`), Setups 2 and 3 (`ladder.py`), the command line (`run.py`) |
| `analysis/` | Per-session measures, the disclosure coder, the evaluation-awareness judge, results tables, the viewer |
| `results/` | Per run and model: run configuration, schedule, per-session readout, judge and coder outputs |

In `results/`, run 1 and run 2 are Setup 1, run 3 is Setup 2 and run 4 is Setup 3. In run 4, `message` is the cell with the scripted report and `no_message` the cell without it.

## Run an experiment

New runs write to `transcripts/`.

```bash
uv run python -m experiments.run --experiment single_agent --name "Initial Run"
uv run python -m experiments.run --experiment step2 --name "Step 2 Write-Access Associate"
uv run python -m experiments.run --experiment step3 --name "Step 3 Teammate Workaround"
```

## Measure a run

```bash
uv run python -m analysis.outcomes "transcripts/<run>/<model>"          # Setup 1: readout.csv
uv run python -m analysis.disclosure "transcripts/<run>/<model>"        # Setup 1: disclosure coder
uv run python -m analysis.team "transcripts/<run>/<model>"              # Setups 2 and 3: team_readout.csv
uv run python -m analysis.eval_awareness "transcripts/<run>/<model>"    # evaluation-awareness judge
uv run python -m analysis.viewer "transcripts/<run>" --out transcripts/viewer
```

# Organism Trader

A research project for financial AI agents with persistent memory.

The project tests how rules, memory, reflection and performance pressure affect an agent's decisions. The model weights stay fixed. The surrounding software stores experience and controls the simulation.

**Status:** experimental software. The public example uses synthetic data and a rule-based stub. It makes no model calls. It is not a published AI performance result.

## What the project contains

- Eight agent configurations: four strategy styles and four paired pressure conditions.
- Daily market replay with cash, positions, fees and trade records.
- SQLite records for decisions, memory, reflection and strategy versions.
- Fixed-rule comparisons: cash, buy-and-hold, equal weight, momentum and reversal.
- A Streamlit dashboard for performance, risk, costs, memory and review records.
- Quantitative metrics, including Sharpe, Sortino and drawdown.
- Model adapters for Cursor, OpenAI-compatible APIs and Ollama.
- Tests for data access, execution, persistence, retries and dashboard calculations.

The agents do not place orders with a broker.

## Run the offline example

Use macOS or Linux with Python 3.11 or later. The current runner uses Unix file locks.

```bash
git clone https://github.com/rvs-personal/organism-trader.git
cd organism-trader
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python scripts/make_synthetic.py
python -m otrader.data.anonymize
python -m otrader.runner --backend stub --run-id demo --days 3
streamlit run src/otrader/dashboard/app.py
```

Open the address printed by Streamlit. Select `demo`. Use **Refresh dashboard** to load new results.

The stub tests the software. It does not reproduce AI reasoning. Do not use its returns as evidence of model performance.

To run three more simulation days:

```bash
python -m otrader.runner --backend stub --run-id demo --days 3
```

The runner saves state in `runs/demo/state.sqlite`. Use a different run ID for a separate experiment.

## Run tests

```bash
python -m pytest -q
```

## Use a model

First complete the offline example. Then read [Model setup](docs/MODEL_SETUP.md).

Set credentials in your local environment. The program does not load `.env` automatically for every command. No keys are supplied with this repository.

The public configuration uses the stub and one worker. Model names in the research configuration are historical choices. Check access and availability before you make calls. A subscription does not guarantee that all API calls are included.

## Research scope

This is a study of agent adaptation without model-weight updates. The eight personas are one experimental condition. The wider study concerns rules, memory, review methods, tools and model choice.

The release does not claim a new learning algorithm, a completed holdout evaluation or a profitable trading advantage. External benchmark results have not been generated for this release.

Read the [Research plan](docs/RESEARCH.md) for the questions, comparison methods and limits.

## Data and privacy

No downloaded market data, private run database, cloud notebook or API key is included. Generate the synthetic example locally. For other data, check the provider's terms before use or redistribution.

Anonymization changes symbols, dates and price scales in the agent-facing records. It does not prove that the model cannot recognize a market pattern.

## Project layout

| Path | Purpose |
|---|---|
| `config/` | Agent, simulation and universe settings |
| `src/otrader/agents/` | Decisions, memory and reflection |
| `src/otrader/data/` | Data preparation and time-limited access |
| `src/otrader/sim/` | Portfolio accounting and execution |
| `src/otrader/inference/` | Model adapters and retry handling |
| `src/otrader/dashboard/` | Observer dashboard and metrics |
| `tests/` | Offline verification |

This release is a separate source snapshot of a private development project. Private operational records and historical data are not part of the release.

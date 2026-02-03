# llm-cost-tracker

[![PyPI version](https://img.shields.io/pypi/v/llm-cost-tracker.svg)](https://pypi.org/project/llm-cost-tracker/)
[![Python](https://img.shields.io/pypi/pyversions/llm-cost-tracker.svg)](https://pypi.org/project/llm-cost-tracker/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/muhammadrashid4587/llm-cost-tracker/actions/workflows/tests.yml/badge.svg)](https://github.com/muhammadrashid4587/llm-cost-tracker/actions)

**Drop-in cost tracking for OpenAI, Anthropic, and other LLM providers.** Wrap your existing client calls and get automatic cost tracking with detailed breakdowns, budget alerts, and export to CSV/JSON/SQLite.

---

## Why?

LLM API costs add up fast. `llm-cost-tracker` gives you visibility without changing how you write code:

- **Zero code changes** -- wrap your client once, use it normally
- **Automatic cost calculation** from token counts in API responses
- **Budget guards** with soft warnings and hard limits
- **Export** to CSV, JSON, or SQLite for analysis
- **Thread-safe** -- works in async pipelines and multi-threaded apps
- **No API keys needed** -- reads token counts from responses, never calls APIs itself

## Installation

```bash
pip install llm-cost-tracker
```

## Quick Start

### OpenAI

```python
from llm_cost_tracker import CostTracker, track_openai
from openai import OpenAI

tracker = CostTracker()
client = track_openai(OpenAI(), tracker)

# Use the client exactly as before -- costs are tracked automatically
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello!"}]
)

print(tracker.summary())
# Model: gpt-4o | Calls: 1 | Input: 12 tok | Output: 28 tok | Cost: $0.00031
```

### Anthropic

```python
from llm_cost_tracker import CostTracker, track_anthropic
from anthropic import Anthropic

tracker = CostTracker()
client = track_anthropic(Anthropic(), tracker)

response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello!"}]
)

print(tracker.summary())
```

### Decorator

```python
from llm_cost_tracker import CostTracker, track_cost
from openai import OpenAI

tracker = CostTracker()
raw_client = OpenAI()

@track_cost(tracker)
def ask(question: str):
    return raw_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": question}]
    )

answer = ask("Explain quantum computing in one sentence.")
print(tracker.summary())
```

## Budget Guards

Set spending limits to prevent runaway costs:

```python
from llm_cost_tracker import CostTracker, BudgetGuard

tracker = CostTracker()
guard = BudgetGuard(
    tracker,
    soft_limit=1.00,   # Logs a warning
    hard_limit=5.00,   # Raises BudgetExceededError
)

# After each API call:
guard.check()

# Check remaining budget
print(f"Remaining: ${guard.remaining():.2f}")
```

**Soft limit** -- emits a `BudgetWarning` (Python warning) and logs to `llm_cost_tracker.budget`. Fires once per limit value.

**Hard limit** -- raises `BudgetExceededError`. Catch it to implement graceful degradation.

**Custom callback:**

```python
def on_budget_alert(current_cost, limit_type, limit_value):
    send_slack_alert(f"LLM budget {limit_type} limit hit: ${current_cost:.2f}")

guard = BudgetGuard(tracker, soft_limit=1.00, alert_callback=on_budget_alert)
```

## Export

```python
# CSV
tracker.export_csv("costs.csv")

# JSON
tracker.export_json("costs.json")

# SQLite (appends to existing DB)
tracker.export_sqlite("costs.db")
```

Or use the standalone functions:

```python
from llm_cost_tracker import export_csv, export_json, export_sqlite

export_csv(tracker.records, "costs.csv")
export_json(tracker.records, "costs.json")
export_sqlite(tracker.records, "costs.db")
```

## Pricing Table

All prices in USD per 1,000,000 tokens.

### OpenAI

| Model | Input | Output | Cached Input |
|---|---|---|---|
| `gpt-4o` | $2.50 | $10.00 | $1.25 |
| `gpt-4o-mini` | $0.15 | $0.60 | $0.075 |
| `gpt-4-turbo` | $10.00 | $30.00 | -- |
| `gpt-3.5-turbo` | $0.50 | $1.50 | -- |
| `o1` | $15.00 | $60.00 | $7.50 |
| `o1-mini` | $3.00 | $12.00 | $1.50 |
| `o3-mini` | $1.10 | $4.40 | $0.55 |

### Anthropic

| Model | Input | Output | Cached Input |
|---|---|---|---|
| `claude-opus-4-6` | $15.00 | $75.00 | $7.50 |
| `claude-sonnet-4-6` | $3.00 | $15.00 | $1.50 |
| `claude-haiku-4-5` | $0.80 | $4.00 | $0.40 |

### Custom / Updated Pricing

Prices change. Add or update models at runtime:

```python
from llm_cost_tracker import update_pricing

update_pricing(
    provider="openai",
    model="gpt-5",
    input_cost_per_1m=5.00,
    output_cost_per_1m=20.00,
    cached_input_cost_per_1m=2.50,
)
```

## API Reference

### `CostTracker`

| Method / Property | Description |
|---|---|
| `record(model, provider, input_tokens, output_tokens, ...)` | Record a single API call |
| `summary(model=, provider=, since=)` | Human-readable summary string |
| `get_summaries(...)` | List of `CostSummary` objects |
| `total_cost` | Cumulative cost (float) |
| `total_tokens` | Cumulative tokens (int) |
| `records` | List of all `UsageRecord` objects |
| `export_csv(path)` | Export to CSV |
| `export_json(path)` | Export to JSON |
| `export_sqlite(path)` | Export to SQLite |
| `reset()` | Clear all recorded data |

### `BudgetGuard`

| Method / Property | Description |
|---|---|
| `check()` | Enforce limits (warn or raise) |
| `remaining()` | Dollars remaining before hard limit |
| `reset_warnings()` | Allow soft-limit warning to fire again |
| `soft_limit` / `hard_limit` | Get or set limits |

## Development

```bash
git clone https://github.com/muhammadrashid4587/llm-cost-tracker.git
cd llm-cost-tracker
pip install -e ".[dev]"
pytest
```

## License

MIT

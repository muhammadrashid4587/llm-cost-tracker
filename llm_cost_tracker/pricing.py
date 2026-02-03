"""
Model pricing data for major LLM providers.

All costs are expressed in USD per 1,000,000 tokens.
"""

from __future__ import annotations

from typing import Optional


# ---------------------------------------------------------------------------
# Canonical pricing table
# Keys: (provider, model)
# Values: dict with input_cost_per_1m, output_cost_per_1m,
#         and optionally cached_input_cost_per_1m
# ---------------------------------------------------------------------------

PRICING: dict[tuple[str, str], dict[str, float]] = {
    # ── OpenAI ────────────────────────────────────────────────────────────
    ("openai", "gpt-4o"): {
        "input_cost_per_1m": 2.50,
        "output_cost_per_1m": 10.00,
        "cached_input_cost_per_1m": 1.25,
    },
    ("openai", "gpt-4o-2024-11-20"): {
        "input_cost_per_1m": 2.50,
        "output_cost_per_1m": 10.00,
        "cached_input_cost_per_1m": 1.25,
    },
    ("openai", "gpt-4o-2024-08-06"): {
        "input_cost_per_1m": 2.50,
        "output_cost_per_1m": 10.00,
        "cached_input_cost_per_1m": 1.25,
    },
    ("openai", "gpt-4o-mini"): {
        "input_cost_per_1m": 0.15,
        "output_cost_per_1m": 0.60,
        "cached_input_cost_per_1m": 0.075,
    },
    ("openai", "gpt-4o-mini-2024-07-18"): {
        "input_cost_per_1m": 0.15,
        "output_cost_per_1m": 0.60,
        "cached_input_cost_per_1m": 0.075,
    },
    ("openai", "gpt-4-turbo"): {
        "input_cost_per_1m": 10.00,
        "output_cost_per_1m": 30.00,
    },
    ("openai", "gpt-4-turbo-2024-04-09"): {
        "input_cost_per_1m": 10.00,
        "output_cost_per_1m": 30.00,
    },
    ("openai", "gpt-3.5-turbo"): {
        "input_cost_per_1m": 0.50,
        "output_cost_per_1m": 1.50,
    },
    ("openai", "gpt-3.5-turbo-0125"): {
        "input_cost_per_1m": 0.50,
        "output_cost_per_1m": 1.50,
    },
    ("openai", "o1"): {
        "input_cost_per_1m": 15.00,
        "output_cost_per_1m": 60.00,
        "cached_input_cost_per_1m": 7.50,
    },
    ("openai", "o1-2024-12-17"): {
        "input_cost_per_1m": 15.00,
        "output_cost_per_1m": 60.00,
        "cached_input_cost_per_1m": 7.50,
    },
    ("openai", "o1-mini"): {
        "input_cost_per_1m": 3.00,
        "output_cost_per_1m": 12.00,
        "cached_input_cost_per_1m": 1.50,
    },
    ("openai", "o1-mini-2024-09-12"): {
        "input_cost_per_1m": 3.00,
        "output_cost_per_1m": 12.00,
        "cached_input_cost_per_1m": 1.50,
    },
    ("openai", "o3-mini"): {
        "input_cost_per_1m": 1.10,
        "output_cost_per_1m": 4.40,
        "cached_input_cost_per_1m": 0.55,
    },
    ("openai", "o3-mini-2025-01-31"): {
        "input_cost_per_1m": 1.10,
        "output_cost_per_1m": 4.40,
        "cached_input_cost_per_1m": 0.55,
    },
    # ── Anthropic ─────────────────────────────────────────────────────────
    ("anthropic", "claude-opus-4-6"): {
        "input_cost_per_1m": 15.00,
        "output_cost_per_1m": 75.00,
        "cached_input_cost_per_1m": 7.50,
    },
    ("anthropic", "claude-opus-4-6-20250515"): {
        "input_cost_per_1m": 15.00,
        "output_cost_per_1m": 75.00,
        "cached_input_cost_per_1m": 7.50,
    },
    ("anthropic", "claude-sonnet-4-6"): {
        "input_cost_per_1m": 3.00,
        "output_cost_per_1m": 15.00,
        "cached_input_cost_per_1m": 1.50,
    },
    ("anthropic", "claude-sonnet-4-6-20250514"): {
        "input_cost_per_1m": 3.00,
        "output_cost_per_1m": 15.00,
        "cached_input_cost_per_1m": 1.50,
    },
    ("anthropic", "claude-haiku-4-5"): {
        "input_cost_per_1m": 0.80,
        "output_cost_per_1m": 4.00,
        "cached_input_cost_per_1m": 0.40,
    },
    ("anthropic", "claude-haiku-4-5-20241022"): {
        "input_cost_per_1m": 0.80,
        "output_cost_per_1m": 4.00,
        "cached_input_cost_per_1m": 0.40,
    },
    # ── OpenAI GPT-4.1 family ────────────────────────────────────────────
    ("openai", "gpt-4.1"): {
        "input_cost_per_1m": 2.00,
        "output_cost_per_1m": 8.00,
        "cached_input_cost_per_1m": 0.50,
    },
    ("openai", "gpt-4.1-mini"): {
        "input_cost_per_1m": 0.40,
        "output_cost_per_1m": 1.60,
        "cached_input_cost_per_1m": 0.10,
    },
    ("openai", "gpt-4.1-nano"): {
        "input_cost_per_1m": 0.10,
        "output_cost_per_1m": 0.40,
        "cached_input_cost_per_1m": 0.025,
    },
    # ── OpenAI o3 ─────────────────────────────────────────────────────────
    ("openai", "o3"): {
        "input_cost_per_1m": 10.00,
        "output_cost_per_1m": 40.00,
        "cached_input_cost_per_1m": 2.50,
    },
    ("openai", "o4-mini"): {
        "input_cost_per_1m": 1.10,
        "output_cost_per_1m": 4.40,
        "cached_input_cost_per_1m": 0.275,
    },
    # ── Anthropic Claude 3.5 ──────────────────────────────────────────────
    ("anthropic", "claude-3-5-sonnet-20241022"): {
        "input_cost_per_1m": 3.00,
        "output_cost_per_1m": 15.00,
        "cached_input_cost_per_1m": 1.50,
    },
    ("anthropic", "claude-3-5-haiku-20241022"): {
        "input_cost_per_1m": 0.80,
        "output_cost_per_1m": 4.00,
        "cached_input_cost_per_1m": 0.40,
    },
    # ── Google Gemini ─────────────────────────────────────────────────────
    ("google", "gemini-2.0-flash"): {
        "input_cost_per_1m": 0.10,
        "output_cost_per_1m": 0.40,
    },
    ("google", "gemini-2.5-pro"): {
        "input_cost_per_1m": 1.25,
        "output_cost_per_1m": 10.00,
    },
    ("google", "gemini-2.5-flash"): {
        "input_cost_per_1m": 0.15,
        "output_cost_per_1m": 0.60,
    },
}


def get_price(
    model: str,
    provider: Optional[str] = None,
) -> dict[str, float]:
    """
    Look up pricing for a model.

    Parameters
    ----------
    model : str
        The model identifier (e.g. ``"gpt-4o"``).
    provider : str, optional
        Provider name (``"openai"``, ``"anthropic"``). When ``None`` the
        function searches all providers and returns the first match.

    Returns
    -------
    dict
        ``{"input_cost_per_1m": ..., "output_cost_per_1m": ..., ...}``

    Raises
    ------
    KeyError
        If no pricing entry exists for the model.
    """
    if provider is not None:
        key = (provider.lower(), model)
        if key in PRICING:
            return PRICING[key]
        raise KeyError(
            f"No pricing found for model={model!r}, provider={provider!r}"
        )

    # Search across all providers
    for (prov, mod), prices in PRICING.items():
        if mod == model:
            return prices
    raise KeyError(f"No pricing found for model={model!r}")


def update_pricing(
    provider: str,
    model: str,
    input_cost_per_1m: float,
    output_cost_per_1m: float,
    cached_input_cost_per_1m: Optional[float] = None,
) -> None:
    """
    Add or overwrite pricing for a model at runtime.

    Parameters
    ----------
    provider : str
        Provider name (e.g. ``"openai"``).
    model : str
        Model identifier.
    input_cost_per_1m : float
        Cost in USD per 1 M input tokens.
    output_cost_per_1m : float
        Cost in USD per 1 M output tokens.
    cached_input_cost_per_1m : float, optional
        Cost in USD per 1 M cached input tokens.
    """
    entry: dict[str, float] = {
        "input_cost_per_1m": input_cost_per_1m,
        "output_cost_per_1m": output_cost_per_1m,
    }
    if cached_input_cost_per_1m is not None:
        entry["cached_input_cost_per_1m"] = cached_input_cost_per_1m
    PRICING[(provider.lower(), model)] = entry

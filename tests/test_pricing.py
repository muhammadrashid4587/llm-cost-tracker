"""Tests for pricing lookup."""

from __future__ import annotations

import pytest

from llm_cost_tracker.pricing import PRICING, get_price, update_pricing


class TestGetPrice:
    """Tests for the get_price function."""

    def test_openai_gpt4o(self):
        p = get_price("gpt-4o", provider="openai")
        assert p["input_cost_per_1m"] == 2.50
        assert p["output_cost_per_1m"] == 10.00
        assert p["cached_input_cost_per_1m"] == 1.25

    def test_openai_gpt4o_mini(self):
        p = get_price("gpt-4o-mini", provider="openai")
        assert p["input_cost_per_1m"] == 0.15
        assert p["output_cost_per_1m"] == 0.60

    def test_openai_gpt4_turbo(self):
        p = get_price("gpt-4-turbo", provider="openai")
        assert p["input_cost_per_1m"] == 10.00
        assert p["output_cost_per_1m"] == 30.00

    def test_openai_gpt35_turbo(self):
        p = get_price("gpt-3.5-turbo", provider="openai")
        assert p["input_cost_per_1m"] == 0.50
        assert p["output_cost_per_1m"] == 1.50

    def test_openai_o1(self):
        p = get_price("o1", provider="openai")
        assert p["input_cost_per_1m"] == 15.00
        assert p["output_cost_per_1m"] == 60.00
        assert p["cached_input_cost_per_1m"] == 7.50

    def test_openai_o1_mini(self):
        p = get_price("o1-mini", provider="openai")
        assert p["input_cost_per_1m"] == 3.00
        assert p["output_cost_per_1m"] == 12.00

    def test_openai_o3_mini(self):
        p = get_price("o3-mini", provider="openai")
        assert p["input_cost_per_1m"] == 1.10
        assert p["output_cost_per_1m"] == 4.40

    def test_anthropic_opus(self):
        p = get_price("claude-opus-4-6", provider="anthropic")
        assert p["input_cost_per_1m"] == 15.00
        assert p["output_cost_per_1m"] == 75.00
        assert p["cached_input_cost_per_1m"] == 7.50

    def test_anthropic_sonnet(self):
        p = get_price("claude-sonnet-4-6", provider="anthropic")
        assert p["input_cost_per_1m"] == 3.00
        assert p["output_cost_per_1m"] == 15.00

    def test_anthropic_haiku(self):
        p = get_price("claude-haiku-4-5", provider="anthropic")
        assert p["input_cost_per_1m"] == 0.80
        assert p["output_cost_per_1m"] == 4.00

    def test_lookup_without_provider(self):
        p = get_price("gpt-4o")
        assert p["input_cost_per_1m"] == 2.50

    def test_unknown_model_raises(self):
        with pytest.raises(KeyError, match="No pricing found"):
            get_price("nonexistent-model-xyz")

    def test_unknown_model_with_provider_raises(self):
        with pytest.raises(KeyError, match="No pricing found"):
            get_price("nonexistent-model-xyz", provider="openai")

    def test_dated_model_variant(self):
        p = get_price("gpt-4o-2024-11-20", provider="openai")
        assert p["input_cost_per_1m"] == 2.50


class TestUpdatePricing:
    """Tests for the update_pricing function."""

    def test_add_new_model(self):
        update_pricing(
            provider="custom",
            model="my-model",
            input_cost_per_1m=1.00,
            output_cost_per_1m=2.00,
        )
        p = get_price("my-model", provider="custom")
        assert p["input_cost_per_1m"] == 1.00
        assert p["output_cost_per_1m"] == 2.00
        assert "cached_input_cost_per_1m" not in p
        # Cleanup
        del PRICING[("custom", "my-model")]

    def test_add_new_model_with_cache(self):
        update_pricing(
            provider="custom",
            model="my-cached-model",
            input_cost_per_1m=1.00,
            output_cost_per_1m=2.00,
            cached_input_cost_per_1m=0.50,
        )
        p = get_price("my-cached-model", provider="custom")
        assert p["cached_input_cost_per_1m"] == 0.50
        # Cleanup
        del PRICING[("custom", "my-cached-model")]

    def test_overwrite_existing(self):
        original = get_price("gpt-4o", provider="openai").copy()
        update_pricing("openai", "gpt-4o", 99.99, 99.99)
        p = get_price("gpt-4o", provider="openai")
        assert p["input_cost_per_1m"] == 99.99
        # Restore
        PRICING[("openai", "gpt-4o")] = original


class TestPricingTableCompleteness:
    """Ensure all advertised models are present."""

    EXPECTED_OPENAI = [
        "gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo",
        "o1", "o1-mini", "o3-mini",
    ]
    EXPECTED_ANTHROPIC = [
        "claude-opus-4-6", "claude-sonnet-4-6", "claude-haiku-4-5",
    ]

    @pytest.mark.parametrize("model", EXPECTED_OPENAI)
    def test_openai_model_exists(self, model: str):
        assert ("openai", model) in PRICING

    @pytest.mark.parametrize("model", EXPECTED_ANTHROPIC)
    def test_anthropic_model_exists(self, model: str):
        assert ("anthropic", model) in PRICING

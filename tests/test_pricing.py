"""Tests for agents/pricing.py — cost math is deterministic, no API."""

from __future__ import annotations

from agents.pricing import PRICING, compute_cost


class TestComputeCost:
    def test_sonnet_fresh_input(self) -> None:
        # 1M fresh input tokens @ $3 = $3.00
        cost = compute_cost("claude-sonnet-4-20250514", 1_000_000, 0, 0, 0)
        assert cost == 3.0

    def test_sonnet_cached_read_90_percent_discount(self) -> None:
        # 1M cached read tokens @ $0.30 = $0.30 (10% of fresh)
        cost = compute_cost("claude-sonnet-4-20250514", 0, 1_000_000, 0, 0)
        assert cost == 0.3

    def test_haiku_output(self) -> None:
        # 100K haiku output tokens @ $5/M = $0.50
        cost = compute_cost("claude-haiku-4-5-20241022", 0, 0, 0, 100_000)
        assert cost == 0.5

    def test_mixed_usage_typical_call(self) -> None:
        # Realistic Sonnet call: 500 fresh input, 2000 cached, 400 output
        cost = compute_cost("claude-sonnet-4-20250514", 500, 2000, 0, 400)
        expected = (500 * 3.0 + 2000 * 0.30 + 400 * 15.0) / 1_000_000
        assert abs(cost - round(expected, 6)) < 1e-9

    def test_unknown_model_returns_zero(self) -> None:
        assert compute_cost("gpt-5-turbo", 1000, 0, 0, 500) == 0.0

    def test_zero_tokens_returns_zero(self) -> None:
        assert compute_cost("claude-sonnet-4-20250514", 0, 0, 0, 0) == 0.0

    def test_all_models_have_four_price_keys(self) -> None:
        for model, prices in PRICING.items():
            assert set(prices.keys()) == {
                "input",
                "cached_read",
                "cache_write",
                "output",
            }, f"{model} has wrong price keys"

    def test_cached_read_cheaper_than_fresh_input(self) -> None:
        for model, prices in PRICING.items():
            assert prices["cached_read"] < prices["input"], (
                f"{model}: cached_read should be cheaper than fresh input"
            )

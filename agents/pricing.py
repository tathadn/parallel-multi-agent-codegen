"""Anthropic model pricing + cost computation.

Prices per 1M tokens (USD). Source: anthropic.com/pricing retrieved 2026-04-14.
Verify current prices before shipping — they change.
"""

from __future__ import annotations

PRICING: dict[str, dict[str, float]] = {
    "claude-sonnet-4-20250514": {
        "input": 3.00,
        "cached_read": 0.30,
        "cache_write": 3.75,
        "output": 15.00,
    },
    "claude-haiku-4-5-20241022": {
        "input": 1.00,
        "cached_read": 0.10,
        "cache_write": 1.25,
        "output": 5.00,
    },
    "claude-opus-4-20250514": {
        "input": 15.00,
        "cached_read": 1.50,
        "cache_write": 18.75,
        "output": 75.00,
    },
}


def compute_cost(
    model: str,
    input_tokens: int,
    cached_read_tokens: int,
    cache_write_tokens: int,
    output_tokens: int,
) -> float:
    """Compute USD cost for a single LLM call given its token breakdown."""
    p = PRICING.get(model)
    if not p:
        return 0.0
    return round(
        (
            input_tokens * p["input"]
            + cached_read_tokens * p["cached_read"]
            + cache_write_tokens * p["cache_write"]
            + output_tokens * p["output"]
        )
        / 1_000_000,
        6,
    )

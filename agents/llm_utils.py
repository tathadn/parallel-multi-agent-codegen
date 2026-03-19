"""Shared LLM utilities — native Anthropic SDK with prompt caching."""

from __future__ import annotations

import json
import re
from typing import Any

import anthropic
from langsmith import traceable

_client = anthropic.Anthropic()


@traceable(run_type="llm", name="call_llm")
def call_llm(
    system: str,
    prompt: str,
    model_name: str = "claude-sonnet-4-20250514",
) -> str:
    """Send a system + human message and return the raw text response.

    Uses cache_control=ephemeral on the system prompt to enable prompt caching,
    cutting input costs by 60–90% after the first call.
    """
    response = _client.messages.create(
        model=model_name,
        max_tokens=8192,
        system=[
            {
                "type": "text",
                "text": system,
                "cache_control": {"type": "ephemeral"},
            }
        ],  # type: ignore[list-item]
        messages=[{"role": "user", "content": prompt}],
    )
    block = response.content[0]
    if hasattr(block, "text"):
        return block.text  # type: ignore[union-attr]
    raise ValueError(f"Unexpected response block type: {type(block)}")


def parse_json_response(text: str) -> Any:
    """
    Extract and parse JSON from an LLM response.

    Handles common quirks: markdown fences, leading prose, trailing text.
    """
    # Strip markdown code fences
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"```\s*$", "", text)

    # Try direct parse first
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to find the first JSON array or object
    for pattern in [r"(\[[\s\S]*\])", r"(\{[\s\S]*\})"]:
        match = re.search(pattern, text)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                continue

    raise ValueError(f"Could not parse JSON from LLM response:\n{text[:500]}")

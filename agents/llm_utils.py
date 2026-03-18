"""Shared LLM utilities — thin wrapper around ChatAnthropic."""

from __future__ import annotations

import json
import re
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage


def get_model(model_name: str = "claude-sonnet-4-20250514", temperature: float = 0.0) -> ChatAnthropic:
    """Return a ChatAnthropic instance for the given model."""
    return ChatAnthropic(model=model_name, temperature=temperature, max_tokens=8192)  # type: ignore[call-arg]


def call_llm(
    system: str,
    prompt: str,
    model_name: str = "claude-sonnet-4-20250514",
) -> str:
    """Send a system + human message and return the raw text response."""
    model = get_model(model_name)
    messages = [SystemMessage(content=system), HumanMessage(content=prompt)]
    response = model.invoke(messages)
    return str(response.content)


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

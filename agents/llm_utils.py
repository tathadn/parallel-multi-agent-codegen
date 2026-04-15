"""Shared LLM utilities — native Anthropic SDK with prompt caching, retries, and usage tracking."""

from __future__ import annotations

import json
import re
import time
from typing import Any, Optional

import anthropic
from langsmith import traceable
from tenacity import (
    RetryError,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from agents.pricing import compute_cost
from models.errors import (
    LLMBadRequest,
    LLMRateLimited,
    LLMTimeout,
    ParseFailure,
)
from models.state import LLMUsage

_client = anthropic.Anthropic()

# Retry policy: transient provider errors only. BadRequestError (400) is a
# programmer bug — never retry, it just burns money.
_RETRYABLE_EXC: tuple[type[BaseException], ...] = (
    anthropic.APIConnectionError,
    anthropic.APITimeoutError,
    anthropic.RateLimitError,
    anthropic.InternalServerError,
)


@retry(
    retry=retry_if_exception_type(_RETRYABLE_EXC),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    stop=stop_after_attempt(4),
    reraise=True,
)
def _create_message(
    model_name: str,
    system: str,
    prompt: str,
) -> Any:
    """Thin wrapper around client.messages.create for retry composition."""
    return _client.messages.create(
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


@traceable(run_type="llm", name="call_llm")
def call_llm(
    system: str,
    prompt: str,
    model_name: str = "claude-sonnet-4-20250514",
    usage_sink: Optional[list[LLMUsage]] = None,
    agent_label: str = "unknown",
) -> str:
    """Send a system + human message and return the raw text response.

    Uses cache_control=ephemeral on the system prompt to enable prompt caching,
    cutting input costs by 60–90% after the first call. Retries transient
    provider errors up to 4 times with exponential backoff. If usage_sink is
    provided, appends an LLMUsage record with token counts and cost.

    Raises:
        LLMBadRequest: 400 from provider (programmer error, not retryable).
        LLMRateLimited: persistent 429/529 after retries exhausted.
        LLMTimeout: connection/timeout errors after retries exhausted.
    """
    t_start = time.time()
    try:
        response = _create_message(model_name, system, prompt)
    except anthropic.BadRequestError as e:
        raise LLMBadRequest(f"{model_name}: {e}") from e
    except anthropic.RateLimitError as e:
        raise LLMRateLimited(f"{model_name}: {e}") from e
    except (anthropic.APIConnectionError, anthropic.APITimeoutError) as e:
        raise LLMTimeout(f"{model_name}: {e}") from e
    except RetryError as e:
        raise LLMTimeout(f"{model_name}: retries exhausted ({e})") from e

    if usage_sink is not None:
        u = getattr(response, "usage", None)
        if u is not None:
            entry = LLMUsage(
                agent=agent_label,
                model=model_name,
                input_tokens=getattr(u, "input_tokens", 0) or 0,
                cached_input_tokens=getattr(u, "cache_read_input_tokens", 0) or 0,
                cache_creation_tokens=getattr(u, "cache_creation_input_tokens", 0) or 0,
                output_tokens=getattr(u, "output_tokens", 0) or 0,
                latency_s=round(time.time() - t_start, 3),
            )
            entry.cost_usd = compute_cost(
                model_name,
                entry.input_tokens,
                entry.cached_input_tokens,
                entry.cache_creation_tokens,
                entry.output_tokens,
            )
            usage_sink.append(entry)

    block = response.content[0]
    if hasattr(block, "text"):
        return block.text  # type: ignore[union-attr]
    raise ValueError(f"Unexpected response block type: {type(block)}")


def parse_json_response(text: str) -> Any:
    """Extract and parse JSON from an LLM response.

    Handles common quirks: markdown fences, leading prose, trailing text.
    Raises ParseFailure if no valid JSON can be extracted.
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

    raise ParseFailure(f"Could not parse JSON from LLM response:\n{text[:500]}")

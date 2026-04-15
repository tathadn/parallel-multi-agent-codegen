"""Structured pipeline error taxonomy.

Raised at the call_llm / agent boundary so the graph can route failures
without relying on free-text log parsing.
"""

from __future__ import annotations


class PipelineError(Exception):
    """Base for all pipeline errors."""


class LLMTimeoutError(PipelineError):
    """LLM call exceeded the deadline or connection timed out."""


class LLMRateLimitedError(PipelineError):
    """LLM provider returned 429/529 after exhausting retries."""


class LLMBadRequestError(PipelineError):
    """LLM provider returned 400 — programmer error, not retryable."""


class ParseFailureError(PipelineError, ValueError):
    """LLM response could not be parsed into the expected JSON shape.

    Inherits from ValueError for backward compatibility with agents that
    catch the legacy ValueError raised by parse_json_response.
    """


class DAGCycleError(PipelineError):
    """TaskDAG contains a cycle — Planner / Orchestrator bug."""


class RetryBudgetExhaustedError(PipelineError):
    """Per-run retry budget hit zero — abort the pipeline."""


# Backward-compat aliases (legacy shorter names).
LLMTimeout = LLMTimeoutError
LLMRateLimited = LLMRateLimitedError
LLMBadRequest = LLMBadRequestError
ParseFailure = ParseFailureError
DAGCycle = DAGCycleError
RetryBudgetExhausted = RetryBudgetExhaustedError

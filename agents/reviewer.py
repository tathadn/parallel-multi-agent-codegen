"""
Reviewer Agent — scores and evaluates the merged codebase.

If the review score is below threshold or critical issues are found,
the pipeline loops back to the Coder workers for revision.
"""

from __future__ import annotations

import json
import time

from models.state import AgentState, ReviewFeedback, TaskStatus
from prompts import REVIEWER_PROMPT, REVIEWER_SYSTEM

from .llm_utils import call_llm, parse_json_response


def reviewer_agent(state: AgentState) -> AgentState:
    """Review the merged code and produce a score + feedback."""
    t0 = time.time()
    state.status = TaskStatus.REVIEWING
    state.log("🔍 Reviewer: evaluating code quality")

    artifacts_text = json.dumps(
        [{"filename": a.filename, "content": a.content} for a in state.artifacts],
        indent=2,
    )

    prompt = REVIEWER_PROMPT.format(
        user_request=state.user_request,
        artifacts=artifacts_text,
    )

    raw = call_llm(system=REVIEWER_SYSTEM, prompt=prompt)

    try:
        review_data = parse_json_response(raw)
    except ValueError as e:
        state.log(f"⚠️ Reviewer: parse failed — {e}")
        state.review = ReviewFeedback(score=5, approved=False, issues=["Review parse error"])
        state.timings["reviewer"] = round(time.time() - t0, 2)
        return state

    state.review = ReviewFeedback(
        score=review_data.get("score", 0),
        approved=review_data.get("approved", False),
        issues=review_data.get("issues", []),
        suggestions=review_data.get("suggestions", []),
    )

    emoji = "✅" if state.review.approved else "⚠️"
    state.log(
        f"{emoji} Reviewer: score={state.review.score}/10, "
        f"approved={state.review.approved}, "
        f"{len(state.review.issues)} issue(s)"
    )
    state.timings["reviewer"] = round(time.time() - t0, 2)
    return state

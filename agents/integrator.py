"""
Integration Agent — merges code artifacts from parallel Coder workers.

When multiple workers code different modules concurrently, their outputs may
have interface mismatches, missing imports, or duplicate utility code.  The
Integration Agent resolves these issues and produces a unified codebase.
"""

from __future__ import annotations

import json
import time

from models.state import AgentState, CodeArtifact
from prompts import INTEGRATOR_PROMPT, INTEGRATOR_SYSTEM

from .llm_utils import call_llm, parse_json_response


def integrator_agent(state: AgentState) -> AgentState:
    """Merge parallel code outputs into a coherent codebase."""
    t0 = time.time()
    state.log(f"🔗 Integrator: merging {len(state.artifacts)} artifacts from parallel workers")

    if len(state.artifacts) <= 1:
        state.log("🔗 Integrator: single artifact — no merge needed")
        state.timings["integrator"] = round(time.time() - t0, 2)
        return state

    # Serialize artifacts for the LLM
    artifacts_text = json.dumps(
        [
            {"filename": a.filename, "language": a.language, "content": a.content}
            for a in state.artifacts
        ],
        indent=2,
    )

    prompt = INTEGRATOR_PROMPT.format(
        user_request=state.user_request,
        objective=state.plan.objective if state.plan else "N/A",
        artifacts=artifacts_text,
    )

    raw = call_llm(
        system=INTEGRATOR_SYSTEM,
        prompt=prompt,
        model_name="claude-haiku-4-5-20241022",
        usage_sink=state.usage_log,
        agent_label="integrator",
    )

    try:
        merged_data = parse_json_response(raw)
    except ValueError as e:
        state.log(f"⚠️ Integrator: parse failed, keeping unmerged artifacts — {e}")
        state.timings["integrator"] = round(time.time() - t0, 2)
        return state

    if isinstance(merged_data, dict):
        merged_data = [merged_data]

    # Replace artifacts with merged versions
    state.artifacts = [
        CodeArtifact(
            filename=f.get("filename", "unknown.py"),
            language=f.get("language", "python"),
            content=f.get("content", ""),
        )
        for f in merged_data
    ]

    state.log(f"✅ Integrator: merged into {len(state.artifacts)} files")
    state.timings["integrator"] = round(time.time() - t0, 2)
    return state

"""
Planner Agent — produces a structured implementation plan from the user request.

The plan includes file decomposition, step ordering, dependency identification,
and parallelizability hints that the Orchestrator uses to build the TaskDAG.
"""

from __future__ import annotations

import time

from models.state import AgentState, Plan, PlanStep
from prompts import PLANNER_PROMPT, PLANNER_SYSTEM

from .llm_utils import call_llm, parse_json_response


def planner_agent(state: AgentState) -> AgentState:
    """Generate a structured Plan from the user request."""
    t0 = time.time()
    state.log("📋 Planner: generating implementation plan")

    prompt = PLANNER_PROMPT.format(user_request=state.user_request)
    raw = call_llm(system=PLANNER_SYSTEM, prompt=prompt)

    try:
        plan_data = parse_json_response(raw)
    except ValueError as e:
        state.status = state.status  # keep current
        state.log(f"❌ Planner: failed to parse plan — {e}")
        return state

    # Build Plan from parsed data
    steps = [
        PlanStep(
            step_number=s.get("step_number", i + 1),
            description=s.get("description", ""),
            files_involved=s.get("files_involved", []),
            dependencies=s.get("dependencies", []),
            parallelizable=s.get("parallelizable", False),
        )
        for i, s in enumerate(plan_data.get("steps", []))
    ]

    state.plan = Plan(
        objective=plan_data.get("objective", ""),
        steps=steps,
        files_to_create=plan_data.get("files_to_create", []),
        dependencies=plan_data.get("dependencies", []),
        complexity=plan_data.get("complexity", "medium"),
    )

    state.log(
        f"✅ Planner: plan ready — {len(steps)} steps, "
        f"{len(state.plan.files_to_create)} files, "
        f"complexity={state.plan.complexity}"
    )
    state.timings["planner"] = round(time.time() - t0, 2)
    return state

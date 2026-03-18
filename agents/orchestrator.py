"""
Orchestrator Agent — the central coordinator for parallel code generation.

The Orchestrator performs two key functions:
1. **Intake**: Parses and validates the user's request.
2. **Decomposition**: After the Planner produces a plan, the Orchestrator
   builds a TaskDAG — a directed acyclic graph where each node is a coding
   unit that can be dispatched to a parallel worker agent.

The Orchestrator maximizes parallelism: nodes without mutual dependencies
are coded concurrently by separate Coder workers.
"""

from __future__ import annotations

import time
from typing import Any

from models.state import AgentState, TaskDAG, TaskNode, TaskStatus
from prompts import ORCHESTRATOR_DECOMPOSE, ORCHESTRATOR_SYSTEM

from .llm_utils import call_llm, parse_json_response


def orchestrator_agent(state: AgentState) -> AgentState:
    """
    First pipeline node: validate the user request and set status.
    """
    t0 = time.time()
    state.log("🎯 Orchestrator: parsing user request")
    state.status = TaskStatus.PLANNING

    # Light validation — the LLM Planner does the heavy lifting
    if not state.user_request or len(state.user_request.strip()) < 10:
        state.status = TaskStatus.FAILED
        state.log("❌ Orchestrator: request too short or empty")
        return state

    state.log(f"✅ Orchestrator: request accepted ({len(state.user_request)} chars)")
    state.timings["orchestrator_intake"] = round(time.time() - t0, 2)
    return state


def orchestrator_decompose(state: AgentState) -> AgentState:
    """
    After the Planner runs, decompose the plan into a TaskDAG for parallel execution.

    This is the key innovation: we analyse the Planner's steps, identify file-level
    dependencies, and build a DAG that the graph executor will dispatch to parallel
    Coder workers.
    """
    t0 = time.time()
    state.log("🔀 Orchestrator: decomposing plan into parallel task DAG")

    if state.plan is None:
        state.status = TaskStatus.FAILED
        state.log("❌ Orchestrator: no plan available to decompose")
        return state

    # Format the plan for the LLM
    steps_text = "\n".join(
        f"  {s.step_number}. {s.description} | files: {s.files_involved} "
        f"| depends_on steps: {s.dependencies} | parallelizable: {s.parallelizable}"
        for s in state.plan.steps
    )

    prompt = ORCHESTRATOR_DECOMPOSE.format(
        user_request=state.user_request,
        objective=state.plan.objective,
        files=", ".join(state.plan.files_to_create),
        steps=steps_text,
    )

    raw = call_llm(
        system=ORCHESTRATOR_SYSTEM,
        prompt=prompt,
        model_name="claude-sonnet-4-20250514",
    )

    try:
        dag_data = parse_json_response(raw)
    except ValueError as e:
        state.log(f"⚠️ Orchestrator: LLM DAG parse failed, building fallback DAG — {e}")
        dag_data = _fallback_dag(state)

    # Build the TaskDAG from the parsed data
    nodes_raw = dag_data if isinstance(dag_data, list) else dag_data.get("nodes", [])
    nodes = []
    for n in nodes_raw:
        nodes.append(
            TaskNode(
                id=n.get("id", ""),
                name=n.get("name", "unknown"),
                description=n.get("description", ""),
                files=n.get("files", []),
                depends_on=n.get("depends_on", []),
            )
        )

    state.task_dag = TaskDAG(nodes=nodes)

    # Log parallelism analysis
    ready = state.task_dag.ready_nodes()
    total = len(state.task_dag.nodes)
    state.log(
        f"📊 Orchestrator: DAG built — {total} nodes, "
        f"{len(ready)} immediately parallelizable"
    )

    state.status = TaskStatus.CODING
    state.timings["orchestrator_decompose"] = round(time.time() - t0, 2)
    return state


def _fallback_dag(state: AgentState) -> dict[str, Any]:
    """
    If the LLM fails to produce a valid DAG, build a simple sequential one
    from the Planner's file list.
    """
    assert state.plan is not None
    nodes = []
    prev_id = None
    for i, filename in enumerate(state.plan.files_to_create):
        node_id = f"node_{i}"
        nodes.append(
            {
                "id": node_id,
                "name": filename.replace(".py", "").replace("/", "_"),
                "description": f"Generate {filename}",
                "files": [filename],
                "depends_on": [prev_id] if prev_id else [],
            }
        )
        prev_id = node_id
    return {"nodes": nodes}

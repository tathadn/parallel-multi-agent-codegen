"""
LangGraph pipeline with parallel orchestration.

This is the enhanced pipeline that replaces the original sequential flow with:

  Orchestrator → Planner → Orchestrator (DAG decompose) → Parallel Coders →
  Integrator → Reviewer → Tester → [conditional: revise or complete]

The key difference from the original repo is the DAG-based parallel coding
phase, where independent modules are generated concurrently by separate
Coder worker agents.
"""

from __future__ import annotations

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agents.coder import coder_worker
from agents.integrator import integrator_agent
from agents.orchestrator import orchestrator_agent, orchestrator_decompose
from agents.planner import planner_agent
from agents.reviewer import reviewer_agent
from agents.tester import tester_agent
from models.state import AgentState, TaskStatus

# ---------------------------------------------------------------------------
# Routing functions
# ---------------------------------------------------------------------------


def should_continue(state: AgentState) -> str:
    """
    After the Tester runs, decide whether to revise or finish.

    Routes to:
      - "complete"  → pipeline ends successfully
      - "revise"    → loop back to parallel coders for another iteration
      - "fail"      → pipeline ends with failure (max iterations exceeded)
    """
    review_ok = state.review is not None and state.review.approved
    tests_ok = state.test_result is not None and state.test_result.passed

    if review_ok and tests_ok:
        state.status = TaskStatus.COMPLETED
        state.log("🎉 Pipeline COMPLETED — review approved and all tests passing")
        return "complete"

    if state.iteration >= state.max_iterations:
        state.status = TaskStatus.FAILED
        state.log(f"⛔ Pipeline FAILED — max iterations ({state.max_iterations}) reached")
        return "fail"

    # Revision needed
    state.iteration += 1
    state.status = TaskStatus.REVISING
    state.log(
        f"🔄 Revision {state.iteration}/{state.max_iterations} — "
        f"review={'✅' if review_ok else '❌'}, tests={'✅' if tests_ok else '❌'}"
    )
    return "revise"


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------


def build_graph() -> CompiledStateGraph:  # type: ignore[type-arg]
    """
    Build and compile the LangGraph pipeline.

    Architecture:

        ┌──────────────┐
        │ orchestrator  │  Parse & validate user request
        └──────┬───────┘
               │
        ┌──────▼───────┐
        │   planner     │  Produce structured plan
        └──────┬───────┘
               │
        ┌──────▼───────────┐
        │ orchestrator_dag  │  Decompose plan → Task DAG
        └──────┬───────────┘
               │
        ┌──────▼───────────┐
        │ parallel_coders   │  Execute DAG nodes concurrently ◄──┐
        └──────┬───────────┘                                     │
               │                                                 │
        ┌──────▼───────┐                                         │
        │  integrator   │  Merge parallel outputs                │
        └──────┬───────┘                                         │
               │                                                 │
        ┌──────▼───────┐                                         │
        │   reviewer    │  Score & evaluate                      │
        └──────┬───────┘                                         │
               │                                                 │
        ┌──────▼───────┐                                         │
        │    tester     │  Generate & run tests                  │
        └──────┬───────┘                                         │
               │                                                 │
        ┌──────▼───────┐     revise                              │
        │   router      │ ───────────────────────────────────────┘
        └──────┬───────┘
               │ complete / fail
               ▼
             [END]
    """
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("orchestrator", orchestrator_agent)
    graph.add_node("planner", planner_agent)
    graph.add_node("orchestrator_dag", orchestrator_decompose)
    graph.add_node("parallel_coders", coder_worker)
    graph.add_node("integrator", integrator_agent)
    graph.add_node("reviewer", reviewer_agent)
    graph.add_node("tester", tester_agent)

    # Linear edges
    graph.add_edge("orchestrator", "planner")
    graph.add_edge("planner", "orchestrator_dag")
    graph.add_edge("orchestrator_dag", "parallel_coders")
    graph.add_edge("parallel_coders", "integrator")
    graph.add_edge("integrator", "reviewer")
    graph.add_edge("reviewer", "tester")

    # Conditional edge from tester → END or back to coders
    graph.add_conditional_edges(
        "tester",
        should_continue,
        {
            "complete": END,
            "fail": END,
            "revise": "parallel_coders",
        },
    )

    # Entry point
    graph.set_entry_point("orchestrator")

    return graph.compile()

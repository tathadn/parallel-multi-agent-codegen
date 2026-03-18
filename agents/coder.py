"""
Coder Worker Agent — generates code for a single TaskNode.

Multiple instances run concurrently via asyncio, each handling an independent
node from the TaskDAG.  The Orchestrator dispatches ready nodes to workers;
workers produce CodeArtifacts that are later merged by the Integration Agent.
"""

from __future__ import annotations

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor

from models.state import (
    AgentState,
    CodeArtifact,
    TaskDAG,
    TaskNode,
    WorkerStatus,
)
from prompts import CODER_PROMPT, CODER_SYSTEM

from .llm_utils import call_llm, parse_json_response

# Thread pool for parallel LLM calls (LangChain's invoke is synchronous)
_executor = ThreadPoolExecutor(max_workers=6)


def _code_single_node(
    node: TaskNode,
    dag: TaskDAG,
    user_request: str,
    revision_context: str = "",
) -> list[CodeArtifact]:
    """
    Generate code for a single DAG node (runs in a thread).

    Returns a list of CodeArtifact objects for the files produced.
    """
    # Build interface contracts from completed dependency nodes
    interfaces = []
    for dep_id in node.depends_on:
        dep_node = dag.get_node(dep_id)
        if dep_node:
            interfaces.append(f"- Module `{dep_node.name}`: {dep_node.description}")
    interfaces_text = "\n".join(interfaces) if interfaces else "None — this module is independent."

    prompt = CODER_PROMPT.format(
        node_name=node.name,
        node_description=node.description,
        files=", ".join(node.files),
        interfaces=interfaces_text,
        revision_context=revision_context or "No revisions — first pass.",
    )

    raw = call_llm(system=CODER_SYSTEM, prompt=prompt)
    files_data = parse_json_response(raw)

    if isinstance(files_data, dict):
        files_data = [files_data]

    artifacts = []
    for f in files_data:
        artifacts.append(
            CodeArtifact(
                filename=f.get("filename", "unknown.py"),
                language=f.get("language", "python"),
                content=f.get("content", ""),
                task_node_id=node.id,
            )
        )
    return artifacts


async def _run_node_async(
    node: TaskNode,
    dag: TaskDAG,
    user_request: str,
    revision_context: str = "",
) -> tuple[str, list[CodeArtifact]]:
    """Async wrapper that runs _code_single_node in the thread pool."""
    loop = asyncio.get_event_loop()
    artifacts = await loop.run_in_executor(
        _executor,
        _code_single_node,
        node,
        dag,
        user_request,
        revision_context,
    )
    return node.id, artifacts


async def run_parallel_coders(
    state: AgentState,
    revision_context: str = "",
) -> AgentState:
    """
    Execute all ready nodes in the TaskDAG in parallel.

    This is called repeatedly by the graph executor until all nodes are DONE.
    Each call dispatches the current batch of ready (dependency-free) nodes
    to concurrent workers, waits for them, then returns so the next batch
    can be dispatched.
    """
    assert state.task_dag is not None
    dag = state.task_dag
    ready = dag.ready_nodes()

    if not ready:
        state.log("⚠️ Coder Workers: no ready nodes to dispatch")
        return state

    node_names = [n.name for n in ready]
    state.log(f"🚀 Coder Workers: dispatching {len(ready)} parallel tasks — {node_names}")

    # Mark nodes as running
    for node in ready:
        node.status = WorkerStatus.RUNNING
        node.started_at = time.time()
        node.worker_id = f"worker-{node.id}"

    # Run all ready nodes concurrently
    tasks = [
        _run_node_async(node, dag, state.user_request, revision_context)
        for node in ready
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Process results
    for result in results:
        if isinstance(result, Exception):
            state.log(f"❌ Coder Worker error: {result}")
            continue

        node_id, artifacts = result  # type: ignore[misc]
        found_node = dag.get_node(node_id)
        if found_node:
            found_node.status = WorkerStatus.DONE
            found_node.finished_at = time.time()
            state.artifacts.extend(artifacts)
            state.log(
                f"✅ Worker {found_node.name}: produced {len(artifacts)} file(s) "
                f"in {found_node.elapsed}s"
            )

    return state


def coder_worker(state: AgentState) -> AgentState:
    """
    Synchronous entry point for the LangGraph node.

    Runs the full parallel coding loop: dispatch ready nodes, wait, repeat
    until the entire DAG is complete.
    """
    t0 = time.time()
    assert state.task_dag is not None

    revision_context = ""
    if state.review and not state.review.approved:
        revision_context = (
            f"Previous review score: {state.review.score}/10\n"
            f"Issues: {state.review.issues}\n"
        )
    if state.test_result and not state.test_result.passed:
        revision_context += f"Test failures:\n{state.test_result.error_output}\n"

    # If revising, reset all DAG nodes
    if state.iteration > 0:
        for node in state.task_dag.nodes:
            node.status = WorkerStatus.IDLE
            node.started_at = None
            node.finished_at = None
        state.artifacts = []

    # Run parallel batches until DAG is complete
    loop = asyncio.new_event_loop()
    try:
        while not state.task_dag.all_done():
            loop.run_until_complete(
                run_parallel_coders(state, revision_context)
            )
            if state.task_dag.any_failed():
                state.log("❌ Some DAG nodes failed — aborting coding phase")
                break
    finally:
        loop.close()

    total_files = len(state.artifacts)
    state.log(f"📦 Coder Workers: all done — {total_files} total files produced")
    state.timings["parallel_coding"] = round(time.time() - t0, 2)
    return state

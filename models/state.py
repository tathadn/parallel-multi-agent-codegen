"""
State models for the parallel multi-agent code generation system.

Key enhancement over the original: TaskDAG enables the orchestrator to decompose
a coding request into a directed acyclic graph of subtasks, where independent
modules are coded in parallel by separate worker agents.
"""

from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class TaskStatus(str, Enum):
    PENDING = "PENDING"
    PLANNING = "PLANNING"
    CODING = "CODING"
    REVIEWING = "REVIEWING"
    TESTING = "TESTING"
    REVISING = "REVISING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class WorkerStatus(str, Enum):
    IDLE = "IDLE"
    RUNNING = "RUNNING"
    DONE = "DONE"
    ERROR = "ERROR"


# ---------------------------------------------------------------------------
# Plan models
# ---------------------------------------------------------------------------


class PlanStep(BaseModel):
    """A single step in the implementation plan."""

    step_number: int
    description: str
    files_involved: list[str] = Field(default_factory=list)
    dependencies: list[int] = Field(
        default_factory=list,
        description="Step numbers this step depends on (must complete first).",
    )
    parallelizable: bool = Field(
        default=False,
        description="Whether this step can run in parallel with other independent steps.",
    )


class Plan(BaseModel):
    """Structured plan produced by the Planner agent."""

    objective: str
    steps: list[PlanStep]
    files_to_create: list[str]
    dependencies: list[str] = Field(
        default_factory=list,
        description="External packages/libraries required.",
    )
    complexity: str = Field(
        default="medium",
        description="low | medium | high",
    )


# ---------------------------------------------------------------------------
# Task DAG — the core parallel scheduling structure
# ---------------------------------------------------------------------------


class TaskNode(BaseModel):
    """
    A single node in the task dependency graph.

    Each node represents one coding unit (e.g., a module/file) that can be
    assigned to a parallel worker agent.  Edges encode data/interface
    dependencies between modules.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str
    description: str
    files: list[str] = Field(default_factory=list)
    depends_on: list[str] = Field(
        default_factory=list,
        description="IDs of TaskNodes that must finish before this one starts.",
    )
    status: WorkerStatus = WorkerStatus.IDLE
    worker_id: Optional[str] = None
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    error: Optional[str] = None

    @property
    def elapsed(self) -> Optional[float]:
        if self.started_at is None:
            return None
        end = self.finished_at or time.time()
        return round(end - self.started_at, 2)


class TaskDAG(BaseModel):
    """
    Directed acyclic graph of coding tasks.

    The orchestrator builds this from the Planner's output.  Nodes with no
    unfinished dependencies are dispatched to parallel coder workers.
    """

    nodes: list[TaskNode] = Field(default_factory=list)

    # -- query helpers -------------------------------------------------------

    def ready_nodes(self) -> list[TaskNode]:
        """Return nodes whose dependencies are all DONE and that haven't started."""
        done_ids = {n.id for n in self.nodes if n.status == WorkerStatus.DONE}
        return [
            n
            for n in self.nodes
            if n.status == WorkerStatus.IDLE and all(dep in done_ids for dep in n.depends_on)
        ]

    def running_nodes(self) -> list[TaskNode]:
        return [n for n in self.nodes if n.status == WorkerStatus.RUNNING]

    def all_done(self) -> bool:
        return all(n.status == WorkerStatus.DONE for n in self.nodes)

    def any_failed(self) -> bool:
        return any(n.status == WorkerStatus.ERROR for n in self.nodes)

    def get_node(self, node_id: str) -> Optional[TaskNode]:
        for n in self.nodes:
            if n.id == node_id:
                return n
        return None

    def topological_order(self) -> list[TaskNode]:
        """Return nodes in a valid topological order for display."""
        visited: set[str] = set()
        order: list[TaskNode] = []

        def visit(node: TaskNode) -> None:
            if node.id in visited:
                return
            visited.add(node.id)
            for dep_id in node.depends_on:
                dep = self.get_node(dep_id)
                if dep:
                    visit(dep)
            order.append(node)

        for n in self.nodes:
            visit(n)
        return order


# ---------------------------------------------------------------------------
# Code & review models
# ---------------------------------------------------------------------------


class CodeArtifact(BaseModel):
    """A single generated code file."""

    filename: str
    language: str = "python"
    content: str
    task_node_id: Optional[str] = Field(
        default=None,
        description="ID of the TaskNode that produced this artifact.",
    )


class ReviewFeedback(BaseModel):
    """Output of the Reviewer agent."""

    score: int = Field(ge=0, le=10)
    approved: bool
    issues: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)


class LLMUsage(BaseModel):
    """Per-call LLM usage record for cost and token observability."""

    agent: str
    model: str
    input_tokens: int = 0
    cached_input_tokens: int = 0
    cache_creation_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    latency_s: float = 0.0
    timestamp: float = Field(default_factory=time.time)


class TestResult(BaseModel):
    """Output of the Tester agent."""

    passed: bool
    total_tests: int = 0
    passed_tests: int = 0
    failed_tests: int = 0
    error_output: str = ""
    test_files: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Top-level pipeline state
# ---------------------------------------------------------------------------


class AgentState(BaseModel):
    """
    Shared state flowing through the LangGraph pipeline.

    Enhanced with TaskDAG for parallel orchestration and worker-level
    tracking for the Streamlit dashboard.
    """

    # Input
    user_request: str

    # Planning
    plan: Optional[Plan] = None
    task_dag: Optional[TaskDAG] = None

    # Code generation (merged from all parallel workers)
    artifacts: list[CodeArtifact] = Field(default_factory=list)

    # Review & test
    review: Optional[ReviewFeedback] = None
    test_result: Optional[TestResult] = None

    # Pipeline control
    status: TaskStatus = TaskStatus.PENDING
    iteration: int = 0
    max_iterations: int = 3

    # Observability
    logs: list[str] = Field(default_factory=list)
    timings: dict[str, float] = Field(default_factory=dict)
    usage_log: list[LLMUsage] = Field(default_factory=list)

    # Resilience
    retry_budget: int = 10

    def log(self, message: str) -> None:
        self.logs.append(f"[{time.strftime('%H:%M:%S')}] {message}")

    def total_cost_usd(self) -> float:
        return round(sum(u.cost_usd for u in self.usage_log), 6)

    def total_tokens(self) -> dict[str, int]:
        return {
            "input": sum(u.input_tokens for u in self.usage_log),
            "cached": sum(u.cached_input_tokens for u in self.usage_log),
            "cache_write": sum(u.cache_creation_tokens for u in self.usage_log),
            "output": sum(u.output_tokens for u in self.usage_log),
        }

    def cache_hit_rate(self) -> float:
        cached = sum(u.cached_input_tokens for u in self.usage_log)
        fresh = sum(u.input_tokens for u in self.usage_log)
        total = cached + fresh
        return round(cached / total, 4) if total else 0.0

    def cost_by_agent(self) -> dict[str, float]:
        result: dict[str, float] = {}
        for u in self.usage_log:
            result[u.agent] = round(result.get(u.agent, 0.0) + u.cost_usd, 6)
        return result

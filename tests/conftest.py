"""Shared fixtures for the test suite."""

from __future__ import annotations

import pytest

from models.state import (
    AgentState,
    CodeArtifact,
    Plan,
    PlanStep,
    ReviewFeedback,
    TaskDAG,
    TaskNode,
    TaskStatus,
    TestResult,
    WorkerStatus,
)


# ---------------------------------------------------------------------------
# DAG fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_dag() -> TaskDAG:
    """Diamond-shaped DAG: root → left/right (parallel) → final."""
    nodes = [
        TaskNode(
            id="root",
            name="root_node",
            description="Root module",
            files=["root.py"],
            depends_on=[],
        ),
        TaskNode(
            id="left",
            name="left_node",
            description="Left branch",
            files=["left.py"],
            depends_on=["root"],
        ),
        TaskNode(
            id="right",
            name="right_node",
            description="Right branch",
            files=["right.py"],
            depends_on=["root"],
        ),
        TaskNode(
            id="final",
            name="final_node",
            description="Final merger",
            files=["final.py"],
            depends_on=["left", "right"],
        ),
    ]
    return TaskDAG(nodes=nodes)


@pytest.fixture
def state_with_dag() -> AgentState:
    """AgentState with a 3-node DAG (A → B, A → D) for surgical revision tests."""
    nodes = [
        TaskNode(
            id="A",
            name="config",
            description="Config module",
            files=["config.py"],
            depends_on=[],
        ),
        TaskNode(
            id="B",
            name="models",
            description="Models module",
            files=["models.py"],
            depends_on=["A"],
        ),
        TaskNode(
            id="D",
            name="routes",
            description="Routes module",
            files=["routes.py"],
            depends_on=["A"],
        ),
    ]
    return AgentState(user_request="Build a web app", task_dag=TaskDAG(nodes=nodes))


# ---------------------------------------------------------------------------
# AgentState fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def empty_state() -> AgentState:
    return AgentState(user_request="Python function that adds two numbers")


@pytest.fixture
def state_with_plan() -> AgentState:
    plan = Plan(
        objective="Build a simple adder function",
        steps=[
            PlanStep(
                step_number=1,
                description="Create main module",
                files_involved=["main.py"],
                dependencies=[],
                parallelizable=False,
            )
        ],
        files_to_create=["main.py"],
        dependencies=[],
        complexity="low",
    )
    return AgentState(user_request="Python function that adds two numbers", plan=plan)


# ---------------------------------------------------------------------------
# Artifact & review fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_artifacts() -> list[CodeArtifact]:
    return [
        CodeArtifact(filename="main.py", content="def main(): pass", task_node_id="node_1"),
        CodeArtifact(filename="utils.py", content="def helper(): pass", task_node_id="node_2"),
    ]


@pytest.fixture
def passing_review() -> ReviewFeedback:
    return ReviewFeedback(
        score=8,
        approved=True,
        issues=[],
        suggestions=["Add more docstrings"],
    )


@pytest.fixture
def failing_review() -> ReviewFeedback:
    return ReviewFeedback(
        score=4,
        approved=False,
        issues=["Missing error handling in routes.py"],
        suggestions=[],
    )

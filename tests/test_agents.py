"""Tests for agent functions — all LLM calls are mocked."""

from __future__ import annotations

import json
from unittest.mock import patch

from agents.coder import coder_worker
from agents.orchestrator import orchestrator_agent, orchestrator_decompose
from agents.planner import planner_agent
from agents.reviewer import reviewer_agent
from models.state import (
    AgentState,
    CodeArtifact,
    ReviewFeedback,
    TaskStatus,
    WorkerStatus,
)

# ---------------------------------------------------------------------------
# TestOrchestratorIntake
# ---------------------------------------------------------------------------


class TestOrchestratorIntake:
    def test_accepts_valid_request(self, empty_state: AgentState) -> None:
        result = orchestrator_agent(empty_state)
        assert result.status != TaskStatus.FAILED

    def test_rejects_short_request(self) -> None:
        state = AgentState(user_request="hi")
        result = orchestrator_agent(state)
        assert result.status == TaskStatus.FAILED

    def test_sets_planning_status_on_success(self, empty_state: AgentState) -> None:
        result = orchestrator_agent(empty_state)
        assert result.status == TaskStatus.PLANNING

    def test_records_timing(self, empty_state: AgentState) -> None:
        result = orchestrator_agent(empty_state)
        assert "orchestrator_intake" in result.timings


# ---------------------------------------------------------------------------
# TestOrchestratorDecompose
# ---------------------------------------------------------------------------


class TestOrchestratorDecompose:
    @patch("agents.orchestrator.call_llm")
    def test_builds_dag_from_llm_response(
        self, mock_llm: object, state_with_plan: AgentState
    ) -> None:
        mock_llm.return_value = json.dumps(  # type: ignore[attr-defined]
            {
                "nodes": [
                    {
                        "id": "A",
                        "name": "main",
                        "description": "Main module",
                        "files": ["main.py"],
                        "depends_on": [],
                    }
                ]
            }
        )
        result = orchestrator_decompose(state_with_plan)
        assert result.task_dag is not None
        assert len(result.task_dag.nodes) == 1
        assert result.task_dag.nodes[0].id == "A"

    @patch("agents.orchestrator.call_llm")
    def test_uses_fallback_dag_on_parse_error(
        self, mock_llm: object, state_with_plan: AgentState
    ) -> None:
        mock_llm.return_value = "this is not JSON"  # type: ignore[attr-defined]
        result = orchestrator_decompose(state_with_plan)
        # Fallback DAG is built from plan.files_to_create
        assert result.task_dag is not None
        assert len(result.task_dag.nodes) == len(state_with_plan.plan.files_to_create)  # type: ignore[union-attr]

    def test_fails_without_plan(self, empty_state: AgentState) -> None:
        result = orchestrator_decompose(empty_state)
        assert result.status == TaskStatus.FAILED

    @patch("agents.orchestrator.call_llm")
    def test_sets_coding_status(self, mock_llm: object, state_with_plan: AgentState) -> None:
        mock_llm.return_value = json.dumps(  # type: ignore[attr-defined]
            {
                "nodes": [
                    {
                        "id": "A",
                        "name": "main",
                        "description": "Main",
                        "files": ["main.py"],
                        "depends_on": [],
                    }
                ]
            }
        )
        result = orchestrator_decompose(state_with_plan)
        assert result.status == TaskStatus.CODING


# ---------------------------------------------------------------------------
# TestPlannerAgent
# ---------------------------------------------------------------------------


class TestPlannerAgent:
    @patch("agents.planner.call_llm")
    def test_generates_plan_from_request(self, mock_llm: object, empty_state: AgentState) -> None:
        mock_llm.return_value = json.dumps(  # type: ignore[attr-defined]
            {
                "objective": "Add two numbers",
                "steps": [
                    {
                        "step_number": 1,
                        "description": "Write adder.py",
                        "files_involved": ["adder.py"],
                        "dependencies": [],
                        "parallelizable": False,
                    }
                ],
                "files_to_create": ["adder.py"],
                "dependencies": [],
                "complexity": "low",
            }
        )
        result = planner_agent(empty_state)
        assert result.plan is not None
        assert result.plan.objective == "Add two numbers"
        assert len(result.plan.steps) == 1

    @patch("agents.planner.call_llm")
    def test_handles_parse_failure(self, mock_llm: object, empty_state: AgentState) -> None:
        mock_llm.return_value = "not valid json"  # type: ignore[attr-defined]
        result = planner_agent(empty_state)
        assert result.plan is None

    @patch("agents.planner.call_llm")
    def test_plan_has_correct_steps(self, mock_llm: object, empty_state: AgentState) -> None:
        mock_llm.return_value = json.dumps(  # type: ignore[attr-defined]
            {
                "objective": "obj",
                "steps": [
                    {
                        "step_number": 1,
                        "description": "step one",
                        "files_involved": ["a.py"],
                        "dependencies": [],
                        "parallelizable": True,
                    },
                    {
                        "step_number": 2,
                        "description": "step two",
                        "files_involved": ["b.py"],
                        "dependencies": [1],
                        "parallelizable": False,
                    },
                ],
                "files_to_create": ["a.py", "b.py"],
                "dependencies": [],
                "complexity": "medium",
            }
        )
        result = planner_agent(empty_state)
        assert result.plan is not None
        assert len(result.plan.steps) == 2
        assert result.plan.steps[0].parallelizable is True
        assert result.plan.steps[1].parallelizable is False

    @patch("agents.planner.call_llm")
    def test_records_timing(self, mock_llm: object, empty_state: AgentState) -> None:
        mock_llm.return_value = json.dumps(  # type: ignore[attr-defined]
            {
                "objective": "obj",
                "steps": [],
                "files_to_create": [],
                "dependencies": [],
                "complexity": "low",
            }
        )
        result = planner_agent(empty_state)
        assert "planner" in result.timings


# ---------------------------------------------------------------------------
# TestReviewerAgent
# ---------------------------------------------------------------------------


class TestReviewerAgent:
    @patch("agents.reviewer.call_llm")
    def test_approves_high_score(self, mock_llm: object, empty_state: AgentState) -> None:
        mock_llm.return_value = json.dumps(  # type: ignore[attr-defined]
            {"score": 9, "approved": True, "issues": [], "suggestions": []}
        )
        empty_state.artifacts = [CodeArtifact(filename="main.py", content="x=1")]
        result = reviewer_agent(empty_state)
        assert result.review is not None
        assert result.review.approved is True
        assert result.review.score == 9

    @patch("agents.reviewer.call_llm")
    def test_rejects_low_score(self, mock_llm: object, empty_state: AgentState) -> None:
        mock_llm.return_value = json.dumps(  # type: ignore[attr-defined]
            {
                "score": 3,
                "approved": False,
                "issues": ["Missing types"],
                "suggestions": [],
            }
        )
        empty_state.artifacts = [CodeArtifact(filename="main.py", content="x=1")]
        result = reviewer_agent(empty_state)
        assert result.review is not None
        assert result.review.approved is False
        assert "Missing types" in result.review.issues

    @patch("agents.reviewer.call_llm")
    def test_handles_parse_failure(self, mock_llm: object, empty_state: AgentState) -> None:
        mock_llm.return_value = "not json"  # type: ignore[attr-defined]
        empty_state.artifacts = [CodeArtifact(filename="main.py", content="x=1")]
        result = reviewer_agent(empty_state)
        assert result.review is not None
        assert result.review.approved is False

    @patch("agents.reviewer.call_llm")
    def test_records_timing(self, mock_llm: object, empty_state: AgentState) -> None:
        mock_llm.return_value = json.dumps(  # type: ignore[attr-defined]
            {"score": 8, "approved": True, "issues": [], "suggestions": []}
        )
        empty_state.artifacts = [CodeArtifact(filename="main.py", content="x=1")]
        result = reviewer_agent(empty_state)
        assert "reviewer" in result.timings


# ---------------------------------------------------------------------------
# TestSurgicalRevision
# ---------------------------------------------------------------------------


class TestSurgicalRevision:
    @patch("agents.coder.call_llm")
    def test_only_resets_failing_nodes(self, mock_llm: object, state_with_dag: AgentState) -> None:
        """When reviewer flags routes.py, only node D should reset; A and B stay DONE."""
        mock_llm.return_value = json.dumps(  # type: ignore[attr-defined]
            [{"filename": "routes.py", "language": "python", "content": "# revised routes"}]
        )

        dag = state_with_dag.task_dag
        assert dag is not None
        for node in dag.nodes:
            node.status = WorkerStatus.DONE

        state_with_dag.artifacts = [
            CodeArtifact(filename="config.py", content="x=1", task_node_id="A"),
            CodeArtifact(filename="models.py", content="y=2", task_node_id="B"),
            CodeArtifact(filename="routes.py", content="z=3", task_node_id="D"),
        ]
        state_with_dag.review = ReviewFeedback(
            score=4,
            approved=False,
            issues=["Missing error handling in routes.py"],
            suggestions=[],
        )
        state_with_dag.iteration = 1

        result = coder_worker(state_with_dag)

        filenames = [a.filename for a in result.artifacts]
        assert "config.py" in filenames
        assert "models.py" in filenames
        assert "routes.py" in filenames

        # Passing artifacts must retain their original content
        config = next(a for a in result.artifacts if a.filename == "config.py")
        assert config.content == "x=1"
        models = next(a for a in result.artifacts if a.filename == "models.py")
        assert models.content == "y=2"

        # Revised artifact should have new content
        routes = next(a for a in result.artifacts if a.filename == "routes.py")
        assert routes.content == "# revised routes"

    @patch("agents.coder.call_llm")
    def test_fallback_resets_all_when_files_unidentified(
        self, mock_llm: object, state_with_dag: AgentState
    ) -> None:
        """When issues don't mention any artifact filename, fallback resets all nodes."""
        mock_llm.return_value = json.dumps(  # type: ignore[attr-defined]
            [
                {"filename": "config.py", "language": "python", "content": "# new config"},
                {"filename": "models.py", "language": "python", "content": "# new models"},
                {"filename": "routes.py", "language": "python", "content": "# new routes"},
            ]
        )

        dag = state_with_dag.task_dag
        assert dag is not None
        for node in dag.nodes:
            node.status = WorkerStatus.DONE

        state_with_dag.artifacts = [
            CodeArtifact(filename="config.py", content="old", task_node_id="A"),
            CodeArtifact(filename="models.py", content="old", task_node_id="B"),
            CodeArtifact(filename="routes.py", content="old", task_node_id="D"),
        ]
        state_with_dag.review = ReviewFeedback(
            score=3,
            approved=False,
            issues=["General quality is very poor"],  # no filename mentioned
            suggestions=[],
        )
        state_with_dag.iteration = 1

        result = coder_worker(state_with_dag)

        # All nodes were re-run; all files should have new content
        filenames = [a.filename for a in result.artifacts]
        assert "config.py" in filenames
        assert "models.py" in filenames
        assert "routes.py" in filenames

    @patch("agents.coder.call_llm")
    def test_surgical_revision_from_test_result(
        self, mock_llm: object, state_with_dag: AgentState
    ) -> None:
        """test_result error mentioning routes.py should also trigger surgical reset of D."""
        from models.state import TestResult

        mock_llm.return_value = json.dumps(  # type: ignore[attr-defined]
            [{"filename": "routes.py", "language": "python", "content": "# fixed routes"}]
        )

        dag = state_with_dag.task_dag
        assert dag is not None
        for node in dag.nodes:
            node.status = WorkerStatus.DONE

        state_with_dag.artifacts = [
            CodeArtifact(filename="config.py", content="cfg=1", task_node_id="A"),
            CodeArtifact(filename="models.py", content="mdl=2", task_node_id="B"),
            CodeArtifact(filename="routes.py", content="bad=3", task_node_id="D"),
        ]
        # Review passes but tests fail mentioning routes.py
        state_with_dag.review = ReviewFeedback(score=8, approved=True)
        state_with_dag.test_result = TestResult(
            passed=False,
            error_output="FAILED tests/test_routes.py::test_get — routes.py line 12",
        )
        state_with_dag.iteration = 1

        result = coder_worker(state_with_dag)

        config = next(a for a in result.artifacts if a.filename == "config.py")
        assert config.content == "cfg=1"
        routes = next(a for a in result.artifacts if a.filename == "routes.py")
        assert routes.content == "# fixed routes"

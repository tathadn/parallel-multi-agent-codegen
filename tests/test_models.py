"""Tests for state models and TaskDAG scheduling logic."""

from __future__ import annotations

import time

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
# TestTaskNode
# ---------------------------------------------------------------------------


class TestTaskNode:
    def test_default_id_generated(self) -> None:
        node = TaskNode(name="foo", description="bar", files=[])
        assert len(node.id) > 0

    def test_custom_id_preserved(self) -> None:
        node = TaskNode(id="my-id", name="foo", description="bar", files=[])
        assert node.id == "my-id"

    def test_default_status_idle(self) -> None:
        node = TaskNode(name="foo", description="bar", files=[])
        assert node.status == WorkerStatus.IDLE

    def test_elapsed_none_before_start(self) -> None:
        node = TaskNode(name="foo", description="bar", files=[])
        assert node.elapsed is None

    def test_elapsed_calculates_after_start(self) -> None:
        node = TaskNode(name="foo", description="bar", files=[])
        node.started_at = time.time() - 2.0
        node.finished_at = node.started_at + 2.0
        assert node.elapsed == pytest.approx(2.0, abs=0.1)

    def test_elapsed_uses_current_time_when_not_finished(self) -> None:
        node = TaskNode(name="foo", description="bar", files=[])
        node.started_at = time.time() - 1.0
        assert node.elapsed is not None
        assert node.elapsed >= 1.0


# ---------------------------------------------------------------------------
# TestTaskDAG
# ---------------------------------------------------------------------------


class TestTaskDAG:
    def test_ready_nodes_all_when_no_deps(self) -> None:
        dag = TaskDAG(
            nodes=[
                TaskNode(id="a", name="a", description="", files=[]),
                TaskNode(id="b", name="b", description="", files=[]),
            ]
        )
        assert len(dag.ready_nodes()) == 2

    def test_ready_nodes_skips_running(self) -> None:
        node = TaskNode(id="a", name="a", description="", files=[])
        node.status = WorkerStatus.RUNNING
        dag = TaskDAG(nodes=[node])
        assert dag.ready_nodes() == []

    def test_ready_nodes_skips_done(self) -> None:
        node = TaskNode(id="a", name="a", description="", files=[])
        node.status = WorkerStatus.DONE
        dag = TaskDAG(nodes=[node])
        assert dag.ready_nodes() == []

    def test_ready_nodes_requires_all_deps_done(self) -> None:
        root = TaskNode(id="root", name="root", description="", files=[])
        child = TaskNode(id="child", name="child", description="", files=[], depends_on=["root"])
        dag = TaskDAG(nodes=[root, child])
        # root is IDLE, child depends on root — only root is ready
        assert [n.id for n in dag.ready_nodes()] == ["root"]

    def test_ready_nodes_after_root_completes(self) -> None:
        root = TaskNode(id="root", name="root", description="", files=[])
        left = TaskNode(id="left", name="left", description="", files=[], depends_on=["root"])
        right = TaskNode(id="right", name="right", description="", files=[], depends_on=["root"])
        dag = TaskDAG(nodes=[root, left, right])
        root.status = WorkerStatus.DONE
        ready_ids = {n.id for n in dag.ready_nodes()}
        assert ready_ids == {"left", "right"}

    def test_ready_nodes_empty_when_all_done(self, sample_dag: TaskDAG) -> None:
        for node in sample_dag.nodes:
            node.status = WorkerStatus.DONE
        assert sample_dag.ready_nodes() == []

    def test_all_done_false_initially(self, sample_dag: TaskDAG) -> None:
        assert sample_dag.all_done() is False

    def test_all_done_true_when_all_done(self, sample_dag: TaskDAG) -> None:
        for node in sample_dag.nodes:
            node.status = WorkerStatus.DONE
        assert sample_dag.all_done() is True

    def test_any_failed_false_initially(self, sample_dag: TaskDAG) -> None:
        assert sample_dag.any_failed() is False

    def test_any_failed_true_with_error_node(self, sample_dag: TaskDAG) -> None:
        sample_dag.nodes[0].status = WorkerStatus.ERROR
        assert sample_dag.any_failed() is True

    def test_get_node_found(self, sample_dag: TaskDAG) -> None:
        node = sample_dag.get_node("root")
        assert node is not None
        assert node.id == "root"

    def test_get_node_not_found(self, sample_dag: TaskDAG) -> None:
        assert sample_dag.get_node("nonexistent") is None

    def test_topological_order_respects_deps(self, sample_dag: TaskDAG) -> None:
        order = [n.id for n in sample_dag.topological_order()]
        assert order.index("root") < order.index("left")
        assert order.index("root") < order.index("right")
        assert order.index("left") < order.index("final")
        assert order.index("right") < order.index("final")

    def test_running_nodes_returns_running(self, sample_dag: TaskDAG) -> None:
        sample_dag.nodes[0].status = WorkerStatus.RUNNING
        running = sample_dag.running_nodes()
        assert len(running) == 1
        assert running[0].id == "root"

    def test_diamond_dag_parallel_middle_nodes(self, sample_dag: TaskDAG) -> None:
        """After root done, left and right should be ready simultaneously."""
        sample_dag.nodes[0].status = WorkerStatus.DONE  # root
        ready_ids = {n.id for n in sample_dag.ready_nodes()}
        assert ready_ids == {"left", "right"}


# ---------------------------------------------------------------------------
# TestAgentState
# ---------------------------------------------------------------------------


class TestAgentState:
    def test_default_status_pending(self) -> None:
        state = AgentState(user_request="test")
        assert state.status == TaskStatus.PENDING

    def test_default_iteration_zero(self) -> None:
        state = AgentState(user_request="test")
        assert state.iteration == 0

    def test_log_appends_with_timestamp(self) -> None:
        state = AgentState(user_request="test")
        state.log("hello")
        assert len(state.logs) == 1
        assert "hello" in state.logs[0]

    def test_default_empty_artifacts(self) -> None:
        state = AgentState(user_request="test")
        assert state.artifacts == []

    def test_max_iterations_default(self) -> None:
        state = AgentState(user_request="test")
        assert state.max_iterations == 3

    def test_timings_default_empty(self) -> None:
        state = AgentState(user_request="test")
        assert state.timings == {}


# ---------------------------------------------------------------------------
# TestPlanStep
# ---------------------------------------------------------------------------


class TestPlanStep:
    def test_parallelizable_default_false(self) -> None:
        step = PlanStep(step_number=1, description="do thing", files_involved=[])
        assert step.parallelizable is False

    def test_dependencies_default_empty(self) -> None:
        step = PlanStep(step_number=1, description="do thing", files_involved=[])
        assert step.dependencies == []


# ---------------------------------------------------------------------------
# TestPlan
# ---------------------------------------------------------------------------


class TestPlan:
    def test_complexity_default_medium(self) -> None:
        plan = Plan(objective="x", steps=[], files_to_create=[])
        assert plan.complexity == "medium"

    def test_files_to_create_stored(self) -> None:
        plan = Plan(objective="x", steps=[], files_to_create=["a.py", "b.py"])
        assert plan.files_to_create == ["a.py", "b.py"]

    def test_steps_stored(self) -> None:
        step = PlanStep(step_number=1, description="do it", files_involved=[])
        plan = Plan(objective="x", steps=[step], files_to_create=[])
        assert len(plan.steps) == 1
        assert plan.steps[0].step_number == 1


# ---------------------------------------------------------------------------
# TestCodeArtifact
# ---------------------------------------------------------------------------


class TestCodeArtifact:
    def test_language_default_python(self) -> None:
        a = CodeArtifact(filename="foo.py", content="x=1")
        assert a.language == "python"

    def test_task_node_id_optional(self) -> None:
        a = CodeArtifact(filename="foo.py", content="x=1")
        assert a.task_node_id is None


# ---------------------------------------------------------------------------
# TestReviewFeedback
# ---------------------------------------------------------------------------


class TestReviewFeedback:
    def test_score_range_valid(self) -> None:
        r = ReviewFeedback(score=7, approved=True)
        assert 0 <= r.score <= 10

    def test_issues_default_empty(self) -> None:
        r = ReviewFeedback(score=7, approved=True)
        assert r.issues == []


# ---------------------------------------------------------------------------
# TestTestResult
# ---------------------------------------------------------------------------


class TestTestResult:
    def test_total_tests_default_zero(self) -> None:
        t = TestResult(passed=True)
        assert t.total_tests == 0

    def test_error_output_default_empty(self) -> None:
        t = TestResult(passed=False)
        assert t.error_output == ""


# ---------------------------------------------------------------------------
# TestDAGSchedulingSimulation
# ---------------------------------------------------------------------------


class TestDAGSchedulingSimulation:
    def test_linear_dag_schedules_sequentially(self) -> None:
        """A→B→C: each batch has exactly one node."""
        nodes = [
            TaskNode(id="A", name="A", description="", files=[]),
            TaskNode(id="B", name="B", description="", files=[], depends_on=["A"]),
            TaskNode(id="C", name="C", description="", files=[], depends_on=["B"]),
        ]
        dag = TaskDAG(nodes=nodes)

        batches = []
        while not dag.all_done():
            ready = dag.ready_nodes()
            batches.append([n.id for n in ready])
            for n in ready:
                n.status = WorkerStatus.DONE

        assert batches == [["A"], ["B"], ["C"]]

    def test_parallel_dag_dispatches_all_roots(self) -> None:
        """Three independent nodes should all be ready in the first batch."""
        nodes = [
            TaskNode(id="x", name="x", description="", files=[]),
            TaskNode(id="y", name="y", description="", files=[]),
            TaskNode(id="z", name="z", description="", files=[]),
        ]
        dag = TaskDAG(nodes=nodes)
        ready_ids = {n.id for n in dag.ready_nodes()}
        assert ready_ids == {"x", "y", "z"}

    def test_diamond_dag_simulation(self, sample_dag: TaskDAG) -> None:
        """root → left/right (parallel) → final: verify 3 batches."""
        batches = []
        while not sample_dag.all_done():
            ready = sample_dag.ready_nodes()
            batches.append(sorted(n.id for n in ready))
            for n in ready:
                n.status = WorkerStatus.DONE

        assert batches[0] == ["root"]
        assert batches[1] == ["left", "right"]
        assert batches[2] == ["final"]

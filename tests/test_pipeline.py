"""Tests for LangGraph pipeline routing and graph compilation."""

from __future__ import annotations

from graph.pipeline import build_graph, should_continue
from models.state import AgentState, ReviewFeedback, TaskStatus
from models.state import TestResult as TestResultModel


def _make_state(
    review_approved: bool | None = None,
    tests_passed: bool | None = None,
    iteration: int = 0,
    max_iterations: int = 3,
) -> AgentState:
    state = AgentState(
        user_request="test",
        iteration=iteration,
        max_iterations=max_iterations,
    )
    if review_approved is not None:
        state.review = ReviewFeedback(
            score=8 if review_approved else 4,
            approved=review_approved,
        )
    if tests_passed is not None:
        state.test_result = TestResultModel(passed=tests_passed)
    return state


# ---------------------------------------------------------------------------
# TestShouldContinue
# ---------------------------------------------------------------------------


class TestShouldContinue:
    def test_complete_when_review_approved_and_tests_pass(self) -> None:
        state = _make_state(review_approved=True, tests_passed=True)
        assert should_continue(state) == "complete"

    def test_revise_when_review_fails(self) -> None:
        state = _make_state(review_approved=False, tests_passed=True)
        assert should_continue(state) == "revise"

    def test_revise_when_tests_fail(self) -> None:
        state = _make_state(review_approved=True, tests_passed=False)
        assert should_continue(state) == "revise"

    def test_revise_when_both_fail(self) -> None:
        state = _make_state(review_approved=False, tests_passed=False)
        assert should_continue(state) == "revise"

    def test_fail_at_max_iterations(self) -> None:
        state = _make_state(
            review_approved=False, tests_passed=False, iteration=3, max_iterations=3
        )
        assert should_continue(state) == "fail"

    def test_revise_increments_iteration(self) -> None:
        state = _make_state(review_approved=False, tests_passed=True, iteration=0)
        should_continue(state)
        assert state.iteration == 1

    def test_status_set_to_completed_on_complete(self) -> None:
        state = _make_state(review_approved=True, tests_passed=True)
        should_continue(state)
        assert state.status == TaskStatus.COMPLETED

    def test_status_set_to_failed_on_fail(self) -> None:
        state = _make_state(review_approved=False, tests_passed=False, iteration=3)
        should_continue(state)
        assert state.status == TaskStatus.FAILED

    def test_status_set_to_revising_on_revise(self) -> None:
        state = _make_state(review_approved=False, tests_passed=True)
        should_continue(state)
        assert state.status == TaskStatus.REVISING

    def test_none_review_triggers_revise(self) -> None:
        state = _make_state(tests_passed=True)  # review is None
        assert should_continue(state) == "revise"

    def test_none_test_result_triggers_revise(self) -> None:
        state = _make_state(review_approved=True)  # test_result is None
        assert should_continue(state) == "revise"

    def test_fail_takes_precedence_over_revise_at_max_iter(self) -> None:
        """Even if quality is bad, max iterations routes to fail, not revise."""
        state = _make_state(
            review_approved=False, tests_passed=False, iteration=3, max_iterations=3
        )
        result = should_continue(state)
        assert result == "fail"


# ---------------------------------------------------------------------------
# TestGraphCompilation
# ---------------------------------------------------------------------------


class TestGraphCompilation:
    def test_graph_compiles_without_error(self) -> None:
        graph = build_graph()
        assert graph is not None

    def test_graph_has_all_expected_nodes(self) -> None:
        graph = build_graph()
        node_names = set(graph.nodes.keys())
        expected = {
            "orchestrator",
            "planner",
            "orchestrator_dag",
            "parallel_coders",
            "integrator",
            "reviewer",
            "tester",
        }
        assert expected.issubset(node_names)

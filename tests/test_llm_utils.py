"""Tests for parse_json_response(), call_llm() usage capture, and retry wrapping."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import anthropic
import pytest

from agents.llm_utils import call_llm, parse_json_response
from models.errors import LLMBadRequest, LLMRateLimited, ParseFailure
from models.state import LLMUsage


class TestParseJsonResponse:
    def test_parses_direct_json_object(self) -> None:
        result = parse_json_response('{"key": "value"}')
        assert result == {"key": "value"}

    def test_parses_direct_json_array(self) -> None:
        result = parse_json_response('[{"a": 1}, {"b": 2}]')
        assert result == [{"a": 1}, {"b": 2}]

    def test_strips_json_markdown_fence(self) -> None:
        text = '```json\n{"key": "value"}\n```'
        result = parse_json_response(text)
        assert result == {"key": "value"}

    def test_strips_plain_code_fence(self) -> None:
        text = '```\n{"key": "value"}\n```'
        result = parse_json_response(text)
        assert result == {"key": "value"}

    def test_extracts_object_after_prose(self) -> None:
        text = 'Here is your JSON:\n{"result": 42}'
        result = parse_json_response(text)
        assert result == {"result": 42}

    def test_extracts_array_after_prose(self) -> None:
        text = 'The files are:\n[{"filename": "main.py"}]'
        result = parse_json_response(text)
        assert result == [{"filename": "main.py"}]

    def test_raises_value_error_on_unparseable(self) -> None:
        with pytest.raises(ValueError, match="Could not parse JSON"):
            parse_json_response("this is not json at all")

    def test_handles_leading_whitespace(self) -> None:
        result = parse_json_response('   \n{"x": 1}   ')
        assert result == {"x": 1}

    def test_nested_json_preserved(self) -> None:
        data = {"outer": {"inner": [1, 2, 3]}}
        result = parse_json_response(json.dumps(data))
        assert result == data

    def test_json_with_special_characters(self) -> None:
        data = {"message": "hello\nworld\ttab"}
        result = parse_json_response(json.dumps(data))
        assert result["message"] == "hello\nworld\ttab"

    def test_array_preferred_over_object_when_both_present(self) -> None:
        """The parser tries array pattern first, then object."""
        text = "[1, 2, 3]"
        result = parse_json_response(text)
        assert result == [1, 2, 3]

    def test_raises_on_empty_string(self) -> None:
        with pytest.raises(ValueError):
            parse_json_response("")

    def test_extracts_json_embedded_in_longer_prose(self) -> None:
        text = 'I have analysed the code. {"score": 8, "approved": true} That concludes my review.'
        result = parse_json_response(text)
        assert result["score"] == 8
        assert result["approved"] is True

    def test_raises_parse_failure_which_is_also_value_error(self) -> None:
        with pytest.raises(ParseFailure):
            parse_json_response("nope")
        # Backward-compat: existing agents catch ValueError
        with pytest.raises(ValueError):
            parse_json_response("nope")


# ---------------------------------------------------------------------------
# Helpers for mocking anthropic responses
# ---------------------------------------------------------------------------


def _fake_response(text: str = '{"ok": 1}', *, usage: dict | None = None) -> SimpleNamespace:
    """Build a minimal fake anthropic.types.Message with .content and .usage."""
    usage_obj = SimpleNamespace(
        input_tokens=usage.get("input_tokens", 0) if usage else 0,
        cache_read_input_tokens=usage.get("cache_read_input_tokens", 0) if usage else 0,
        cache_creation_input_tokens=(usage.get("cache_creation_input_tokens", 0) if usage else 0),
        output_tokens=usage.get("output_tokens", 0) if usage else 0,
    )
    return SimpleNamespace(
        content=[SimpleNamespace(text=text)],
        usage=usage_obj,
    )


def _make_api_error(cls: type[Exception]) -> Exception:
    """Anthropic exception classes require request/response args — fake them."""
    resp = MagicMock(status_code=500)
    body = {"error": {"message": "simulated"}}
    try:
        return cls(message="simulated", response=resp, body=body)  # type: ignore[call-arg]
    except TypeError:
        # RateLimitError / BadRequestError take different args across versions
        return cls("simulated", response=resp, body=body)  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# TestCallLLMUsageCapture
# ---------------------------------------------------------------------------


class TestCallLLMUsageCapture:
    @patch("agents.llm_utils._client")
    def test_records_usage_entry_with_cost(self, mock_client: MagicMock) -> None:
        mock_client.messages.create.return_value = _fake_response(
            text='{"score": 9}',
            usage={
                "input_tokens": 500,
                "cache_read_input_tokens": 2000,
                "cache_creation_input_tokens": 0,
                "output_tokens": 400,
            },
        )
        sink: list[LLMUsage] = []
        result = call_llm(
            system="sys",
            prompt="user",
            model_name="claude-sonnet-4-20250514",
            usage_sink=sink,
            agent_label="reviewer",
        )
        assert result == '{"score": 9}'
        assert len(sink) == 1
        entry = sink[0]
        assert entry.agent == "reviewer"
        assert entry.model == "claude-sonnet-4-20250514"
        assert entry.input_tokens == 500
        assert entry.cached_input_tokens == 2000
        assert entry.output_tokens == 400
        # Cost: (500*3 + 2000*0.3 + 400*15) / 1M = 0.0081
        assert abs(entry.cost_usd - 0.0081) < 1e-9

    @patch("agents.llm_utils._client")
    def test_no_sink_means_no_tracking(self, mock_client: MagicMock) -> None:
        mock_client.messages.create.return_value = _fake_response()
        result = call_llm(system="sys", prompt="user")
        assert result == '{"ok": 1}'
        # No crash, no side effects

    @patch("agents.llm_utils._client")
    def test_haiku_pricing_applied(self, mock_client: MagicMock) -> None:
        mock_client.messages.create.return_value = _fake_response(
            usage={"input_tokens": 1000, "output_tokens": 500}
        )
        sink: list[LLMUsage] = []
        call_llm(
            system="sys",
            prompt="user",
            model_name="claude-haiku-4-5-20241022",
            usage_sink=sink,
            agent_label="reviewer",
        )
        # (1000*1.0 + 500*5.0) / 1M = 0.0035
        assert abs(sink[0].cost_usd - 0.0035) < 1e-9


# ---------------------------------------------------------------------------
# TestCallLLMRetryAndErrors
# ---------------------------------------------------------------------------


class TestCallLLMRetryAndErrors:
    @patch("agents.llm_utils._client")
    def test_retries_on_rate_limit_then_succeeds(self, mock_client: MagicMock) -> None:
        rate_err = _make_api_error(anthropic.RateLimitError)
        # First two calls raise, third succeeds
        mock_client.messages.create.side_effect = [
            rate_err,
            rate_err,
            _fake_response(text="ok"),
        ]
        with patch("agents.llm_utils.wait_exponential") as mock_wait:
            # speed up retry wait
            mock_wait.return_value = lambda *a, **kw: 0
            result = call_llm(system="s", prompt="p")
        # The retry decorator should have invoked create 3 times
        assert mock_client.messages.create.call_count == 3
        assert result == "ok"

    @patch("agents.llm_utils._client")
    def test_bad_request_not_retried_and_wrapped(self, mock_client: MagicMock) -> None:
        bad = _make_api_error(anthropic.BadRequestError)
        mock_client.messages.create.side_effect = bad
        with pytest.raises(LLMBadRequest):
            call_llm(system="s", prompt="p")
        # No retry on 400 — exactly 1 attempt
        assert mock_client.messages.create.call_count == 1

    @patch("agents.llm_utils._client")
    def test_persistent_rate_limit_wraps_as_llm_rate_limited(self, mock_client: MagicMock) -> None:
        rate_err = _make_api_error(anthropic.RateLimitError)
        mock_client.messages.create.side_effect = rate_err
        with pytest.raises(LLMRateLimited):
            call_llm(system="s", prompt="p")
        # 4 attempts (stop_after_attempt=4)
        assert mock_client.messages.create.call_count == 4

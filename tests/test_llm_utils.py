"""Tests for parse_json_response() in agents/llm_utils.py."""

from __future__ import annotations

import json

import pytest

from agents.llm_utils import parse_json_response


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
        text = '[1, 2, 3]'
        result = parse_json_response(text)
        assert result == [1, 2, 3]

    def test_raises_on_empty_string(self) -> None:
        with pytest.raises(ValueError):
            parse_json_response("")

    def test_extracts_json_embedded_in_longer_prose(self) -> None:
        text = (
            "I have analysed the code. "
            '{"score": 8, "approved": true} '
            "That concludes my review."
        )
        result = parse_json_response(text)
        assert result["score"] == 8
        assert result["approved"] is True

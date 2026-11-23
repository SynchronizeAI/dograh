"""Regression test for QA analysis when the LLM returns a non-dict JSON value.

``parse_llm_json`` is explicitly designed to return a list when the model emits
a top-level JSON array (see ``test_json_parser.py``). The QA analyzers then call
``parsed.get(...)`` on the result. For a list that raises ``AttributeError``,
which is NOT caught by the surrounding ``except (json.JSONDecodeError, ValueError)``
— so a stray array response crashed the whole QA run instead of degrading to
empty results.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest
from jsonschema import Draft202012Validator

from api.services.workflow.qa import analysis as qa_analysis


def test_qa_response_schema_requires_the_rubric_shape():
    validator = Draft202012Validator(qa_analysis._QA_RESPONSE_SCHEMA)
    valid_result = {
        "tags": [{"tag": "USER_FRUSTRATED", "reason": "Caller complained."}],
        "call_quality_score": 6,
        "overall_sentiment": "negative",
        "summary": "The caller was frustrated.",
    }

    assert validator.is_valid(valid_result)
    assert not validator.is_valid({})
    assert not validator.is_valid({**valid_result, "call_quality_score": None})
    assert not validator.is_valid({**valid_result, "overall_sentiment": "mixed"})
    assert not validator.is_valid(
        {**valid_result, "tags": ["USER_FRUSTRATED"]}
    )


@pytest.mark.asyncio
async def test_qa_inference_passes_requested_schema_for_invalid_json_text():
    llm = AsyncMock()
    llm.run_inference.return_value = r'{"summary":"bad \\uXXXX escape"}'

    response = await qa_analysis._run_llm_inference(
        llm, [], "prompt", response_schema=qa_analysis._QA_RESPONSE_SCHEMA
    )

    assert response == r'{"summary":"bad \\uXXXX escape"}'
    assert llm.run_inference.call_args.kwargs["response_schema"] == (
        qa_analysis._QA_RESPONSE_SCHEMA
    )


@pytest.mark.asyncio
async def test_conversation_summary_inference_omits_qa_response_schema():
    llm = AsyncMock()
    llm.run_inference.return_value = "Summary of the conversation."

    with patch.object(qa_analysis, "add_qa_span_to_trace"):
        summary = await qa_analysis._generate_conversation_summary(
            llm, "gpt-4o", "user: hello", None, "Next node"
        )

    assert summary == "Summary of the conversation."
    assert "response_schema" not in llm.run_inference.call_args.kwargs


@pytest.mark.asyncio
async def test_whole_call_qa_tolerates_array_llm_response():
    """A top-level JSON array from the QA LLM degrades to empty results."""
    qa_data = SimpleNamespace(qa_system_prompt="Summarize: {transcript}")
    workflow_run = SimpleNamespace(
        logs={
            "realtime_feedback_events": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi there"},
            ]
        },
        usage_info={"call_duration_seconds": 12},
    )
    warning_mock = Mock()
    llm = object()
    inference_mock = AsyncMock(return_value='["tag1", "tag2"]')

    with (
        patch.object(
            qa_analysis, "build_conversation_structure", return_value=[{"x": 1}]
        ),
        patch.object(qa_analysis, "format_transcript", return_value="user: hello"),
        patch.object(qa_analysis, "compute_call_metrics", return_value={}),
        patch.object(
            qa_analysis,
            "create_qa_llm_service",
            new=AsyncMock(return_value=(llm, "gpt-4o")),
        ),
        patch.object(
            qa_analysis,
            "_run_llm_inference",
            new=inference_mock,
        ),
        patch.object(qa_analysis, "setup_langfuse_parent_context", return_value=None),
        patch.object(qa_analysis, "add_qa_span_to_trace", return_value=None),
        patch.object(qa_analysis.logger, "warning", warning_mock),
    ):
        # Before the fix this raised AttributeError: 'list' object has no
        # attribute 'get'.
        result = await qa_analysis._run_whole_call_qa_analysis(
            qa_data, workflow_run, workflow_run_id=99
        )

    assert inference_mock.await_args.kwargs["response_schema"] == (
        qa_analysis._QA_RESPONSE_SCHEMA
    )
    node_result = result["node_results"]["whole_call"]
    assert node_result["tags"] == []
    assert node_result["summary"] == ""
    assert node_result["score"] is None
    warning_mock.assert_called_once()
    warning_message = warning_mock.call_args.args[0]
    assert "non-object JSON" in warning_message
    assert "run 99" in warning_message
    assert "list" in warning_message
    assert "tag1" not in warning_message

"""Gemini explicit cache holds only a workflow's fixed prefix and tools."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from google.genai import errors
from google.genai.types import Content, GenerateContentResponse, Part
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.google.llm import GoogleLLMSettings

from api.services.pipecat import gemini_prefix_cache
from api.services.pipecat.gemini_prefix_cache import (
    GeminiPrefixCacheMixin,
    cacheable_prefix,
    set_prompt_cache_prefix,
    strip_cache_boundary_marker,
)
from api.services.pipecat.service_factory import (
    DograhGoogleLLMService,
    DograhGoogleVertexLLMService,
)
from api.utils.template_renderer import render_template

PERSONA = "# 성격\n친절하게 말해요."
RECEPTION = (
    "# 목표\n예약을 받아요.\n\n# 회사 데이터\n아래는 이 통화의 정보예요.\n\n"
    "<!-- cache-boundary -->\n"
    '## 회사 이름\n"{{business_name | fallback:저희 회사}}"\n'
    "- 손님 이름: {{caller_name | fallback:모름}}"
)
RAW = f"{PERSONA}\n\n{RECEPTION}"
CALL = {"business_name": "동네배관", "caller_name": "김민수"}
TOOLS = [{"function_declarations": [{"name": "book_visit", "description": "Book"}]}]


@pytest.fixture(autouse=True)
def clear_registry():
    gemini_prefix_cache._caches.clear()
    gemini_prefix_cache._creating.clear()


def test_prefix_requires_one_marker_after_static_text():
    prefix = cacheable_prefix(RAW)

    assert prefix == (
        f"{PERSONA}\n\n# 목표\n예약을 받아요.\n\n"
        "# 회사 데이터\n아래는 이 통화의 정보예요.\n\n"
    )
    rendered = strip_cache_boundary_marker(render_template(RAW, CALL))
    assert rendered.startswith(prefix)
    assert cacheable_prefix("# 고정\n변수 없음") == ""
    assert cacheable_prefix("이름: {{caller_name}}") == ""
    assert cacheable_prefix(
        "{{caller_name}}\n<!-- cache-boundary -->\n- 이름: {{caller_name}}"
    ) == ""
    assert cacheable_prefix(
        "# 고정\n<!-- cache-boundary -->\n<!-- cache-boundary -->\n{{caller_name}}"
    ) == ""


def test_both_gemini_provider_paths_use_the_prefix_cache():
    assert issubclass(DograhGoogleLLMService, GeminiPrefixCacheMixin)
    assert issubclass(DograhGoogleVertexLLMService, GeminiPrefixCacheMixin)


def test_prefix_reaches_every_gemini_target_of_a_fallback_processor():
    gemini = make_service()
    other = SimpleNamespace()
    set_prompt_cache_prefix(SimpleNamespace(_services=[gemini, other]), "고정")

    assert gemini.prompt_cache_prefix == "고정"
    assert not hasattr(other, "prompt_cache_prefix")


def chunk(text):
    return GenerateContentResponse(
        candidates=[{"content": Content(role="model", parts=[Part(text=text)])}]
    )


def make_service(create=None, streams=(), raw_prompt=RAW):
    service = DograhGoogleLLMService(
        api_key="test-key",
        settings=GoogleLLMSettings(
            model="gemini-3.6-flash",
            system_instruction=strip_cache_boundary_marker(
                render_template(raw_prompt, CALL)
            ),
        ),
        tools=TOOLS,
    )
    pending = list(streams)

    async def generate_content_stream(**kwargs):
        step = pending.pop(0)
        if isinstance(step, BaseException):
            raise step

        async def stream():
            yield chunk(step)

        return stream()

    service._client = SimpleNamespace(
        _api_client=SimpleNamespace(api_key="test-key", project=None, location=None),
        aio=SimpleNamespace(
            caches=SimpleNamespace(
                create=create or AsyncMock(return_value=SimpleNamespace(name="c/1"))
            ),
            models=SimpleNamespace(
                generate_content_stream=AsyncMock(side_effect=generate_content_stream)
            ),
        ),
    )
    service.prompt_cache_prefix = cacheable_prefix(raw_prompt)
    return service


async def texts(service):
    context = LLMContext(messages=[{"role": "user", "content": "내일 예약돼요?"}])
    stream = await service._stream_content(context)
    return [c.candidates[0].content.parts[0].text async for c in stream]


@pytest.mark.asyncio
async def test_cache_holds_only_fixed_text_and_tools_and_call_data_goes_in_contents():
    service = make_service(streams=["uncached", "cached"])

    assert await texts(service) == ["uncached"]
    await asyncio.gather(*gemini_prefix_cache._creating.values())
    assert await texts(service) == ["cached"]

    config = service._client.aio.caches.create.await_args.kwargs["config"]
    assert config.system_instruction == cacheable_prefix(RAW)
    assert "김민수" not in config.model_dump_json()
    assert "동네배관" not in config.model_dump_json()
    assert config.tools[0].function_declarations[0].name == "book_visit"

    first, second = service._client.aio.models.generate_content_stream.await_args_list
    assert "김민수" in first.kwargs["config"].system_instruction
    cached = second.kwargs["config"]
    assert cached["cached_content"] == "c/1"
    assert not {"system_instruction", "tools", "tool_config"} & cached.keys()
    opening = second.kwargs["contents"][0]
    assert opening.role == "user"
    assert all(
        "<!-- cache-boundary -->" not in part.text
        for content in second.kwargs["contents"]
        for part in content.parts or []
        if part.text
    )
    assert opening.parts[0].text == (
        strip_cache_boundary_marker(render_template(RAW, CALL))[
            len(cacheable_prefix(RAW)) :
        ]
    )
    assert "김민수" in opening.parts[0].text
    assert opening.parts[-1].text == "내일 예약돼요?"


def _wire_request(kwargs):
    config = kwargs["config"]
    if hasattr(config, "model_dump"):
        config = config.model_dump(mode="json", exclude_none=True)
    contents = [
        content.model_dump(mode="json", exclude_none=True)
        for content in kwargs["contents"]
    ]
    return json.dumps(
        {"model": kwargs["model"], "contents": contents, "config": config},
        sort_keys=True,
        separators=(",", ":"),
    )


@pytest.mark.parametrize(
    "raw_prompt, baseline_prompt",
    [
        (
            "# 안내\n예약을 도와요.\n\n# 회사 데이터\n"
            "- 손님 이름: {{caller_name}}\n"
            "# 예약 전에 전화번호를 반드시 확인하세요.\n"
            "손님에게 확인한 뒤에 예약을 진행해요.",
            "# 안내\n예약을 도와요.\n\n# 회사 데이터\n"
            "- 손님 이름: {{caller_name}}\n"
            "# 예약 전에 전화번호를 반드시 확인하세요.\n"
            "손님에게 확인한 뒤에 예약을 진행해요.",
        ),
        (
            "# 안내\n{{business_name}} 예약을 도와요.\n"
            "<!-- cache-boundary -->\n- 손님 이름: {{caller_name}}",
            "# 안내\n{{business_name}} 예약을 도와요.\n"
            "- 손님 이름: {{caller_name}}",
        ),
        (
            "# 안내\n예약을 도와요.\n<!-- cache-boundary -->\n"
            "<!-- cache-boundary -->\n- 손님 이름: {{caller_name}}",
            "# 안내\n예약을 도와요.\n- 손님 이름: {{caller_name}}",
        ),
    ],
    ids=["marker-absent", "variable-before-marker", "multiple-markers"],
)
@pytest.mark.asyncio
async def test_non_opted_in_prompt_uses_the_marker_free_uncached_request(
    raw_prompt, baseline_prompt
):
    service = make_service(streams=["uncached"], raw_prompt=raw_prompt)
    uncached = make_service(streams=["uncached"], raw_prompt=baseline_prompt)

    assert cacheable_prefix(raw_prompt) == ""
    assert await texts(service) == ["uncached"]
    assert await texts(uncached) == ["uncached"]
    assert service._client.aio.caches.create.await_count == 0
    assert "<!-- cache-boundary -->" not in service._settings.system_instruction
    assert _wire_request(
        service._client.aio.models.generate_content_stream.await_args.kwargs
    ) == _wire_request(
        uncached._client.aio.models.generate_content_stream.await_args.kwargs
    )


@pytest.mark.asyncio
async def test_rejected_cache_falls_back_to_the_uncached_request():
    rejected = errors.ClientError(404, {"error": {"message": "not found"}})
    service = make_service(streams=["warm", rejected, "uncached"])
    await texts(service)
    await asyncio.gather(*gemini_prefix_cache._creating.values())

    assert await texts(service) == ["uncached"]
    last = service._client.aio.models.generate_content_stream.await_args
    assert "김민수" in last.kwargs["config"].system_instruction
    assert "<!-- cache-boundary -->" not in last.kwargs["config"].system_instruction
    assert gemini_prefix_cache._caches == {}


@pytest.mark.asyncio
async def test_failed_cache_creation_keeps_requests_uncached():
    service = make_service(
        create=AsyncMock(side_effect=RuntimeError("too small")),
        streams=["one", "two"],
    )
    await texts(service)
    await asyncio.gather(*gemini_prefix_cache._creating.values())

    assert await texts(service) == ["two"]
    assert service._client.aio.caches.create.await_count == 1
    last = service._client.aio.models.generate_content_stream.await_args
    assert "cached_content" not in last.kwargs["config"].model_dump(exclude_none=True)


@pytest.mark.asyncio
async def test_prompt_outside_the_prefix_is_never_cached():
    service = make_service(streams=["uncached"])
    service.prompt_cache_prefix = "다른 워크플로의 고정 문장"

    assert await texts(service) == ["uncached"]
    assert service._client.aio.caches.create.await_count == 0


@pytest.mark.asyncio
async def test_engine_gives_gemini_the_fixed_prefix_of_the_rendered_node_prompt(
    three_node_workflow,
):
    from api.services.workflow.pipecat_engine import PipecatEngine

    llm = make_service(streams=["cached"])

    async def update_settings(settings):
        llm._settings.system_instruction = settings.system_instruction

    llm._update_settings = AsyncMock(side_effect=update_settings)
    engine = PipecatEngine(
        workflow=three_node_workflow, llm=llm, call_context_vars=CALL
    )
    node = three_node_workflow.nodes["agent"]
    node.prompt = RECEPTION
    await engine._setup_llm_context(node)

    assert llm.prompt_cache_prefix
    assert engine.active_agent.system_prompt.startswith(llm.prompt_cache_prefix)
    assert "<!-- cache-boundary -->" not in engine.active_agent.system_prompt
    assert "김민수" in engine.active_agent.system_prompt
    assert "김민수" not in llm.prompt_cache_prefix
    await asyncio.gather(*gemini_prefix_cache._creating.values())
    assert llm._client.aio.caches.create.await_count == 1
    assert llm._client.aio.models.generate_content_stream.await_count == 0
    assert await texts(llm) == ["cached"]


@pytest.mark.asyncio
async def test_engine_keeps_generated_recording_instructions_at_system_priority(
    three_node_workflow,
):
    from api.services.workflow.pipecat_engine import PipecatEngine

    llm = make_service()
    engine = PipecatEngine(
        workflow=three_node_workflow, llm=llm, call_context_vars=CALL
    )
    engine._has_recordings = True
    node = three_node_workflow.nodes["agent"]
    node.prompt = "# 녹음 안내\n<!-- cache-boundary -->\nRECORDING_ID: greeting"

    await engine._setup_llm_context(node)

    assert llm.prompt_cache_prefix == ""
    assert "RESPONSE MODE INSTRUCTIONS" in engine.active_agent.system_prompt
    assert "<!-- cache-boundary -->" not in engine.active_agent.system_prompt
    assert llm._client.aio.caches.create.await_count == 0


@pytest.mark.asyncio
async def test_cached_request_failure_before_first_chunk_falls_back_uncached():
    service = make_service(
        streams=["warm", RuntimeError("temporary transport error"), "uncached"]
    )
    await texts(service)
    await asyncio.gather(*gemini_prefix_cache._creating.values())

    assert await texts(service) == ["uncached"]
    assert service._client.aio.models.generate_content_stream.await_count == 3
    assert gemini_prefix_cache._caches == {}


@pytest.mark.asyncio
async def test_cached_request_cancellation_propagates_without_uncached_retry():
    service = make_service(streams=["warm", asyncio.CancelledError()])
    await texts(service)
    await asyncio.gather(*gemini_prefix_cache._creating.values())

    with pytest.raises(asyncio.CancelledError):
        await texts(service)

    assert service._client.aio.models.generate_content_stream.await_count == 2


@pytest.mark.asyncio
async def test_cached_stream_failure_after_first_chunk_is_not_retried():
    service = make_service(streams=["warm"])
    await texts(service)
    await asyncio.gather(*gemini_prefix_cache._creating.values())

    async def fail_after_first_chunk(**kwargs):
        async def stream():
            yield chunk("first")
            raise RuntimeError("stream interrupted")

        return stream()

    service._client.aio.models.generate_content_stream.side_effect = (
        fail_after_first_chunk
    )
    with pytest.raises(RuntimeError, match="stream interrupted"):
        await texts(service)

    assert service._client.aio.models.generate_content_stream.await_count == 2

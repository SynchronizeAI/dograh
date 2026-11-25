"""Gemini explicit cache holds only a workflow's fixed prefix and tools."""

import asyncio
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
)
from api.services.pipecat.service_factory import (
    DograhGoogleLLMService,
    DograhGoogleVertexLLMService,
)
from api.utils.template_renderer import render_template

PERSONA = "# 성격\n친절하게 말해요."
RECEPTION = (
    "# 목표\n예약을 받아요.\n\n# 회사 데이터\n아래는 이 통화의 정보예요.\n\n"
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


def test_prefix_stops_at_the_section_holding_the_first_variable():
    prefix = cacheable_prefix(RAW)

    assert prefix == f"{PERSONA}\n\n# 목표\n예약을 받아요.\n\n"
    assert render_template(RAW, CALL).startswith(prefix)
    assert cacheable_prefix("# 고정\n변수 없음") == "# 고정\n변수 없음"
    assert cacheable_prefix("이름: {{caller_name}}") == ""


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


def make_service(create=None, streams=()):
    service = DograhGoogleLLMService(
        api_key="test-key",
        settings=GoogleLLMSettings(
            model="gemini-3.6-flash", system_instruction=render_template(RAW, CALL)
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
    service.prompt_cache_prefix = cacheable_prefix(RAW)
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
    assert opening.parts[0].text.startswith("# 회사 데이터")
    assert "김민수" in opening.parts[0].text
    assert opening.parts[-1].text == "내일 예약돼요?"


@pytest.mark.asyncio
async def test_rejected_cache_falls_back_to_the_uncached_request():
    rejected = errors.ClientError(404, {"error": {"message": "not found"}})
    service = make_service(streams=["warm", rejected, "uncached"])
    await texts(service)
    await asyncio.gather(*gemini_prefix_cache._creating.values())

    assert await texts(service) == ["uncached"]
    last = service._client.aio.models.generate_content_stream.await_args
    assert "김민수" in last.kwargs["config"].system_instruction
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

    llm = make_service()
    llm._update_settings = AsyncMock()
    engine = PipecatEngine(
        workflow=three_node_workflow, llm=llm, call_context_vars=CALL
    )
    node = three_node_workflow.nodes["agent"]
    node.prompt = RECEPTION
    await engine._prepare_node(engine.active_agent, node)

    assert llm.prompt_cache_prefix
    assert engine.active_agent.system_prompt.startswith(llm.prompt_cache_prefix)
    assert "김민수" in engine.active_agent.system_prompt
    assert "김민수" not in llm.prompt_cache_prefix

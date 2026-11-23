from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from api.services.pipecat.gemini_prompt_cache import GeminiPromptCacheMixin


class _Service(GeminiPromptCacheMixin):
    def __init__(self):
        self._settings = SimpleNamespace(model="gemini-3.6-flash")
        self._tools = []
        self._tool_config = None
        self._client = SimpleNamespace(
            aio=SimpleNamespace(caches=SimpleNamespace(create=AsyncMock()))
        )
        self.get_llm_adapter = lambda: SimpleNamespace(
            get_llm_invocation_params=AsyncMock(
                return_value={"tools": [{"function_declarations": [{"name": "book"}]}]}
            )
        )
        self._should_inject_trailing_user_message = lambda: False


@pytest.mark.asyncio
async def test_large_prompt_and_tools_are_prewarmed_once():
    service = _Service()
    service._client.aio.caches.create.return_value = SimpleNamespace(
        name="cachedContents/reception"
    )
    prompt = "Reception instructions. " * 1000

    await service.prepare_prompt_cache(object(), prompt)
    await service.prepare_prompt_cache(object(), prompt)

    assert service._client.aio.caches.create.await_count == 1
    config = service._client.aio.caches.create.await_args.kwargs["config"]
    assert config.system_instruction == prompt
    assert config.tools[0].function_declarations[0].name == "book"
    assert config.ttl == "1260s"


@pytest.mark.asyncio
async def test_small_prompt_leaves_uncached_requests_unchanged():
    service = _Service()
    service._prompt_cache_name = "cachedContents/old-node"

    await service.prepare_prompt_cache(object(), "Short system prompt")

    assert service._client.aio.caches.create.await_count == 0
    assert service._prompt_cache_name is None


class _Base:
    def _build_generation_params(self, **kwargs):
        return kwargs


class _CachedService(GeminiPromptCacheMixin, _Base):
    pass


def test_cached_request_uses_cache_without_duplicate_system_or_tools():
    service = _CachedService()
    service._prompt_cache_name = "cachedContents/reception"

    params = service._build_generation_params(
        system_instruction="system", tools=["book"], tool_config={"mode": "auto"}
    )

    assert params == {
        "system_instruction": None,
        "tools": None,
        "tool_config": None,
        "cached_content": "cachedContents/reception",
    }

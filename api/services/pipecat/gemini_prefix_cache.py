"""Gemini explicit cache of a workflow's fixed prompt prefix and tools.

Missing capability: the pinned Pipecat Google service never sends
``cached_content``. Only the fixed text before a node prompt's first template
variable is cached, together with the tool schemas. The per-call section is
sent as the first conversation content. Any cache problem falls back to the
unchanged uncached request.
"""

import asyncio
import hashlib
import json
import time
from contextlib import aclosing

from google.genai import errors
from google.genai.types import Content, CreateCachedContentConfig, Part
from loguru import logger

from pipecat.utils.types import assert_given

CACHE_TTL_SECONDS = 3600
_REFRESH_MARGIN_SECONDS = 120
_FAILURE_BACKOFF_SECONDS = 600

# key -> (cached content name or None after a failed create, monotonic expiry)
_caches: dict[str, tuple[str | None, float]] = {}
_creating: dict[str, asyncio.Task] = {}


def cacheable_prefix(raw_prompt: str) -> str:
    """Fixed text up to the top-level heading that holds the first variable."""
    first_variable = raw_prompt.find("{{")
    if first_variable < 0:
        return raw_prompt
    heading = raw_prompt.rfind("\n# ", 0, first_variable)
    return raw_prompt[: heading + 1] if heading >= 0 else ""


def set_prompt_cache_prefix(llm, prefix: str) -> None:
    services = getattr(llm, "_services", None)  # FallbackLLMProcessor targets
    for service in services if isinstance(services, list) else [llm]:
        if isinstance(service, GeminiPrefixCacheMixin):
            service.prompt_cache_prefix = prefix


class GeminiPrefixCacheMixin:
    prompt_cache_prefix = ""

    def _prefix_cache_key(self, model, prefix, tools, tool_config) -> str:
        api = getattr(self._client, "_api_client", None)
        owner = [getattr(api, f, None) for f in ("api_key", "project", "location")]
        blob = json.dumps(
            [owner, model, prefix, tools, tool_config], default=str, sort_keys=True
        )
        return hashlib.sha256(blob.encode()).hexdigest()

    async def _create_prefix_cache(self, key, model, prefix, tools, tool_config):
        try:
            cache = await self._client.aio.caches.create(
                model=model,
                config=CreateCachedContentConfig(
                    system_instruction=prefix,
                    tools=tools or None,
                    tool_config=tool_config,
                    ttl=f"{CACHE_TTL_SECONDS}s",
                ),
            )
            expiry = CACHE_TTL_SECONDS - _REFRESH_MARGIN_SECONDS
            _caches[key] = (cache.name, time.monotonic() + expiry)
        except Exception as error:
            logger.warning(f"Gemini prefix cache unavailable: {type(error).__name__}")
            _caches[key] = (None, time.monotonic() + _FAILURE_BACKOFF_SECONDS)
        finally:
            _creating.pop(key, None)

    def _ready_prefix_cache(self, key, *create_args) -> str | None:
        name, expiry = _caches.get(key, (None, 0.0))
        if time.monotonic() < expiry:
            return name
        _caches.pop(key, None)
        if key not in _creating:
            _creating[key] = asyncio.create_task(
                self._create_prefix_cache(key, *create_args)
            )
        return None

    async def _stream_content(self, context):
        prefix = self.prompt_cache_prefix
        if not prefix:
            return await super()._stream_content(context)
        params = await self.get_llm_adapter().get_llm_invocation_params(
            context,
            system_instruction=assert_given(self._settings.system_instruction),
            ensure_last_message_is_user=self._should_inject_trailing_user_message(),
        )
        instruction = params["system_instruction"] or ""
        if not instruction.startswith(prefix):
            return await super()._stream_content(context)
        model = self._settings.model
        tools = params["tools"] or self._tools or []
        tool_config = self._tool_config or None
        key = self._prefix_cache_key(model, prefix, tools, tool_config)
        name = self._ready_prefix_cache(key, model, prefix, tools, tool_config)
        if not name:
            return await super()._stream_content(context)

        config = self._build_generation_params()
        config["cached_content"] = name
        messages = list(params["messages"])
        rest = instruction[len(prefix) :].strip()
        if rest:
            if messages and messages[0].role == "user":
                first = messages.pop(0)
                parts = [Part(text=rest), *(first.parts or [])]
            else:
                parts = [Part(text=rest)]
            messages.insert(0, Content(role="user", parts=parts))
        self._log_llm_response(context)
        return self._stream_cached(context, model, messages, config, key)

    async def _stream_cached(self, context, model, messages, config, key):
        try:
            stream = await self._client.aio.models.generate_content_stream(
                model=model, contents=messages, config=config
            )
            first = await stream.__anext__()
        except StopAsyncIteration:
            return
        except errors.ClientError as error:
            logger.warning(f"Gemini prefix cache rejected ({error.code}); uncached")
            _caches.pop(key, None)
            stream = await super()._stream_content(context)
            first = None
        async with aclosing(stream):
            if first is not None:
                yield first
            async for chunk in stream:
                yield chunk

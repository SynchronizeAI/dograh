"""Gemini explicit cache of a workflow's fixed prompt prefix and tools.

Missing capability: the pinned Pipecat Google service never sends
``cached_content``. When the raw prompt opts in with one plain
``<!-- cache-boundary -->`` line, only the fixed system text before it is cached
with the tool schemas. Text after that line is sent as the first conversation
content. The marker is removed before the prompt reaches the model. Any cache
problem falls back to the unchanged uncached request.
"""

import asyncio
import hashlib
import json
import time
from contextlib import aclosing, suppress

from google.genai.types import Content, CreateCachedContentConfig, Part
from loguru import logger

from pipecat.utils.types import assert_given

CACHE_TTL_SECONDS = 3600
_REFRESH_MARGIN_SECONDS = 120
_FAILURE_BACKOFF_SECONDS = 600
_CACHE_BOUNDARY_MARKER = "<!-- cache-boundary -->"

# key -> (cached content name or None after a failed create, monotonic expiry)
_caches: dict[str, tuple[str | None, float]] = {}
_creating: dict[str, asyncio.Task] = {}


def _cache_boundary_spans(prompt: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    offset = 0
    for line in prompt.splitlines(keepends=True):
        if line.rstrip("\r\n") == _CACHE_BOUNDARY_MARKER:
            spans.append((offset, offset + len(line)))
        offset += len(line)
    return spans


def strip_cache_boundary_marker(prompt: str) -> str:
    spans = _cache_boundary_spans(prompt)
    if not spans:
        return prompt
    pieces = []
    start = 0
    for marker_start, marker_end in spans:
        pieces.append(prompt[start:marker_start])
        start = marker_end
    pieces.append(prompt[start:])
    return "".join(pieces)


def cacheable_prefix(raw_prompt: str) -> str:
    """Return the static prefix when one explicit boundary opts the prompt in."""
    spans = _cache_boundary_spans(raw_prompt)
    if len(spans) != 1:
        return ""
    marker_start, _ = spans[0]
    prefix = raw_prompt[:marker_start]
    return prefix if prefix and "{{" not in prefix else ""


def set_prompt_cache_prefix(llm, prefix: str) -> None:
    services = getattr(llm, "_services", None)  # FallbackLLMProcessor targets
    for service in services if isinstance(services, list) else [llm]:
        if isinstance(service, GeminiPrefixCacheMixin):
            service.prompt_cache_prefix = prefix


async def warm_prompt_cache(llm, context) -> None:
    services = getattr(llm, "_services", None)
    for service in services if isinstance(services, list) else [llm]:
        if isinstance(service, GeminiPrefixCacheMixin):
            await service._warm_prompt_cache(context)


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

    async def _get_prefix_cache_request_params(self, context):
        return await self.get_llm_adapter().get_llm_invocation_params(
            context,
            system_instruction=assert_given(self._settings.system_instruction),
            ensure_last_message_is_user=self._should_inject_trailing_user_message(),
        )

    def _prefix_cache_request_details(self, params):
        prefix = self.prompt_cache_prefix
        instruction = params["system_instruction"] or ""
        model = self._settings.model
        tools = params["tools"] or self._tools or []
        tool_config = self._tool_config or None
        key = self._prefix_cache_key(model, prefix, tools, tool_config)
        return prefix, instruction, model, tools, tool_config, key

    async def _warm_prompt_cache(self, context) -> None:
        if not self.prompt_cache_prefix:
            return
        try:
            params = await self._get_prefix_cache_request_params(context)
            prefix, instruction, model, tools, tool_config, key = (
                self._prefix_cache_request_details(params)
            )
            if instruction.startswith(prefix):
                self._ready_prefix_cache(
                    key, model, prefix, tools, tool_config
                )
        except asyncio.CancelledError:
            raise
        except Exception as error:
            logger.warning(
                f"Gemini prefix cache warmup unavailable: {type(error).__name__}"
            )

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
        params = await self._get_prefix_cache_request_params(context)
        prefix, instruction, model, tools, tool_config, key = (
            self._prefix_cache_request_details(params)
        )
        if not instruction.startswith(prefix):
            return await super()._stream_content(context)
        name = self._ready_prefix_cache(key, model, prefix, tools, tool_config)
        if not name:
            return await super()._stream_content(context)

        config = self._build_generation_params()
        config["cached_content"] = name
        messages = list(params["messages"])
        rest = instruction[len(prefix) :]
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
        stream = None
        try:
            stream = await self._client.aio.models.generate_content_stream(
                model=model, contents=messages, config=config
            )
            first = await stream.__anext__()
        except StopAsyncIteration:
            return
        except asyncio.CancelledError:
            raise
        except Exception as error:
            if stream is not None:
                with suppress(Exception):
                    await stream.aclose()
            logger.warning(
                f"Gemini prefix cache request failed ({type(error).__name__}); uncached"
            )
            _caches.pop(key, None)
            stream = await super()._stream_content(context)
            async with aclosing(stream):
                async for chunk in stream:
                    yield chunk
            return
        async with aclosing(stream):
            yield first
            async for chunk in stream:
                yield chunk

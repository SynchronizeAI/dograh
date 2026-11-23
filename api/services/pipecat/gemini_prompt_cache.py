"""Explicit Gemini context caching for Dograh's Google Developer API service."""

import hashlib
import json

from google.genai.types import CreateCachedContentConfig
from loguru import logger


class GeminiPromptCacheMixin:
    """Prewarm stable system instructions and tool schemas for repeated turns."""

    async def prepare_prompt_cache(self, context, system_instruction: str) -> None:
        if context is None or not system_instruction:
            return
        adapter = self.get_llm_adapter()
        params = await adapter.get_llm_invocation_params(
            context,
            system_instruction=system_instruction,
            ensure_last_message_is_user=self._should_inject_trailing_user_message(),
        )
        tools = params["tools"] or self._tools or None
        tool_config = self._tool_config
        material = json.dumps(
            [system_instruction, tools, tool_config],
            sort_keys=True,
            ensure_ascii=False,
            default=lambda value: (
                value.model_dump(by_alias=True)
                if hasattr(value, "model_dump")
                else str(value)
            ),
        )
        if len(material) < 16_384:
            self._prompt_cache_fingerprint = None
            self._prompt_cache_name = None
            return
        fingerprint = hashlib.sha256(material.encode()).hexdigest()
        if getattr(self, "_prompt_cache_fingerprint", None) == fingerprint:
            return
        self._prompt_cache_fingerprint = None
        self._prompt_cache_name = None
        try:
            cache = await self._client.aio.caches.create(
                model=self._settings.model,
                config=CreateCachedContentConfig(
                    system_instruction=system_instruction,
                    tools=tools,
                    tool_config=tool_config,
                    ttl="1260s",
                ),
            )
        except Exception as exc:  # noqa: BLE001 - cache failure must preserve the uncached path
            logger.warning(
                "Gemini prompt cache prewarm failed ({})", type(exc).__name__
            )
            return
        if cache.name:
            self._prompt_cache_fingerprint = fingerprint
            self._prompt_cache_name = cache.name

    def _build_generation_params(
        self, system_instruction=None, tools=None, tool_config=None
    ):
        cache_name = getattr(self, "_prompt_cache_name", None)
        if cache_name:
            params = super()._build_generation_params(
                system_instruction=None, tools=None, tool_config=None
            )
            params["cached_content"] = cache_name
            return params
        return super()._build_generation_params(system_instruction, tools, tool_config)

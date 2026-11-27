from __future__ import annotations

from pydantic import BaseModel, model_validator

from api.schemas.llm_fallback import (
    FallbackPolicy,
    STTFallbackPolicy,
)
from api.services.configuration.registry import (
    EmbeddingsConfig,
    LLMConfig,
    RealtimeConfig,
    ServiceProviders,
    STTConfig,
    TTSConfig,
)


class EffectiveAIModelConfiguration(BaseModel):
    llm: LLMConfig | None = None
    llm_fallback: FallbackPolicy[LLMConfig] | None = None
    stt: STTConfig | None = None
    stt_fallback: STTFallbackPolicy[STTConfig] | None = None
    tts: TTSConfig | None = None
    embeddings: EmbeddingsConfig | None = None
    realtime: RealtimeConfig | None = None
    is_realtime: bool = False
    managed_service_version: int | None = None
    test_phone_number: str | None = None
    timezone: str | None = None

    @model_validator(mode="after")
    def require_byok_llm_fallback(self):
        if not self.llm_fallback or not self.llm_fallback.rules:
            return self
        if self.llm is None or self.llm.provider == ServiceProviders.DOGRAH:
            raise ValueError("LLM fallbacks require a non-Dograh primary LLM")
        if any(
            rule.target.provider == ServiceProviders.DOGRAH
            for rule in self.llm_fallback.rules
        ):
            raise ValueError(
                "Dograh manages its own fallbacks and cannot be a fallback target"
            )
        return self

    @model_validator(mode="after")
    def require_byok_stt_fallback(self):
        if self.is_realtime or not self.stt_fallback or not self.stt_fallback.rules:
            return self
        if self.stt is None or self.stt.provider == ServiceProviders.DOGRAH:
            raise ValueError("STT fallbacks require a non-Dograh primary STT")
        if any(
            rule.target.provider == ServiceProviders.DOGRAH
            for rule in self.stt_fallback.rules
        ):
            raise ValueError(
                "Dograh manages its own fallbacks and cannot be a fallback target"
            )
        if any(
            rule.target.provider == self.stt.provider
            for rule in self.stt_fallback.rules
        ):
            raise ValueError(
                "STT fallback target must use a different provider than the primary"
            )
        return self

    @model_validator(mode="before")
    @classmethod
    def strip_incomplete_realtime_when_disabled(cls, data):
        """Skip realtime validation when is_realtime is False and api_key is missing."""
        if isinstance(data, dict) and not data.get("is_realtime", False):
            realtime = data.get("realtime")
            if isinstance(realtime, dict) and not realtime.get("api_key"):
                data.pop("realtime", None)
        return data

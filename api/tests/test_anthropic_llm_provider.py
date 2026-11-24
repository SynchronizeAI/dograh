from unittest.mock import patch

import anthropic
import httpx
import pytest

from api.schemas.ai_model_configuration import EffectiveAIModelConfiguration
from api.services.configuration import check_validity
from api.services.configuration.check_validity import UserConfigurationValidator
from api.services.configuration.registry import (
    REGISTRY,
    AnthropicLLMConfiguration,
    ServiceProviders,
    ServiceType,
)
from api.services.pipecat.service_factory import (
    DograhAnthropicLLMService,
    create_llm_service,
)


def test_anthropic_defaults_to_claude_haiku_5_5():
    config = AnthropicLLMConfiguration(api_key="sk-ant-test")

    assert config.model == "claude-haiku-5-5"
    assert config.temperature is None
    assert REGISTRY[ServiceType.LLM][ServiceProviders.ANTHROPIC] is (
        AnthropicLLMConfiguration
    )


def test_anthropic_primary_and_fallback_targets_reach_the_service():
    config = EffectiveAIModelConfiguration.model_validate(
        {
            "llm": {"provider": "anthropic", "api_key": "primary-key"},
            "llm_fallback": {
                "version": 1,
                "rules": [
                    {
                        "condition": {"type": "no_output", "after_ms": 850},
                        "target": {
                            "provider": "anthropic",
                            "api_key": "backup-key",
                            "model": "claude-sonnet-5",
                        },
                    },
                    {
                        "condition": {"type": "error"},
                        "target": {"provider": "google", "api_key": "google-key"},
                    },
                ],
            },
        }
    )
    with (
        patch(
            "api.services.pipecat.service_factory.DograhAnthropicLLMService"
        ) as claude,
        patch("api.services.pipecat.service_factory.DograhGoogleLLMService"),
        patch("api.services.pipecat.service_factory.FallbackLLMProcessor"),
    ):
        create_llm_service(config)

    calls = [call.kwargs for call in claude.call_args_list]
    assert [(c["api_key"], c["settings"].model) for c in calls] == [
        ("primary-key", "claude-haiku-5-5"),
        ("backup-key", "claude-sonnet-5"),
    ]
    assert all(c["enable_direct_mode"] for c in calls)
    assert all(c["settings"].enable_prompt_caching for c in calls)


def test_created_service_is_pipecats_anthropic_service():
    service = create_llm_service(
        EffectiveAIModelConfiguration(llm=AnthropicLLMConfiguration(api_key="k"))
    )

    assert isinstance(service, DograhAnthropicLLMService)
    assert service._settings.model == "claude-haiku-5-5"


@pytest.mark.asyncio
async def test_inference_usage_counts_cached_tokens_as_prompt_tokens():
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from anthropic.types.beta import BetaUsage
    from pipecat.processors.aggregators.llm_context import LLMContext

    from api.services.pipecat.usage_metrics import on_inference_usage

    service = create_llm_service(
        EffectiveAIModelConfiguration(llm=AnthropicLLMConfiguration(api_key="k"))
    )
    service._client = AsyncMock()
    service._client.beta.messages.create.return_value = SimpleNamespace(
        content=[SimpleNamespace(text="ok")],
        stop_reason="end_turn",
        usage=BetaUsage(
            input_tokens=60,
            output_tokens=20,
            cache_read_input_tokens=40,
            cache_creation_input_tokens=10,
        ),
    )
    reports = []
    on_inference_usage(service, lambda _service, data: reports.append(data.value))
    await service.run_inference(
        LLMContext(messages=[{"role": "user", "content": "hi"}])
    )
    [usage] = reports
    assert (usage.prompt_tokens, usage.completion_tokens) == (110, 20)
    assert (usage.cache_read_input_tokens, usage.cache_creation_input_tokens) == (
        40,
        10,
    )
    assert usage.total_tokens == 130


def test_anthropic_api_key_validation(monkeypatch):
    calls = []

    class FakeAnthropic:
        def __init__(self, api_key):
            self.models = self
            self.api_key = api_key

        def list(self, **kwargs):
            calls.append(self.api_key)
            if self.api_key == "bad":
                response = httpx.Response(401, request=httpx.Request("GET", "x"))
                raise anthropic.AuthenticationError("bad", response=response, body={})

    monkeypatch.setattr(check_validity.anthropic, "Anthropic", FakeAnthropic)
    validator = UserConfigurationValidator()

    assert validator._check_api_key(ServiceProviders.ANTHROPIC.value, "good")
    with pytest.raises(ValueError, match="Invalid Anthropic API key"):
        validator._check_api_key(ServiceProviders.ANTHROPIC.value, "bad")
    assert calls == ["good", "bad"]

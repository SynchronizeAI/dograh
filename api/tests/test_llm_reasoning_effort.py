"""OpenAI reasoning_effort travels from saved settings to the request body."""

import pytest

from api.schemas.ai_model_configuration import EffectiveAIModelConfiguration
from api.services.configuration.registry import REGISTRY, ServiceType
from api.services.pipecat.service_factory import create_llm_service


def _request(model, **settings):
    config = REGISTRY[ServiceType.LLM]["openai"](
        api_key="test-key", model=model, **settings
    )
    # Emulate persistence and reload through the discriminated request schema.
    saved = EffectiveAIModelConfiguration(llm=config).model_dump_json(exclude_none=True)
    service = create_llm_service(
        EffectiveAIModelConfiguration.model_validate_json(saved)
    )
    return service.build_chat_completion_params({"messages": []})


def test_saved_reasoning_effort_reaches_request():
    params = _request("gpt-6-luna", reasoning_effort="none")
    assert params["reasoning_effort"] == "none"
    assert params["temperature"] == 0.1


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("gpt-4.1-mini", None),
        ("gpt-5-mini", {"reasoning_effort": "minimal", "verbosity": "low"}),
    ],
)
def test_absent_reasoning_effort_keeps_todays_request(model, expected):
    params = _request(model)
    if expected is None:
        assert "reasoning_effort" not in params
    else:
        assert {key: params[key] for key in expected} == expected


def test_reasoning_effort_overrides_gpt5_default_and_keeps_verbosity():
    params = _request("gpt-5-mini", reasoning_effort="low")
    assert (params["reasoning_effort"], params["verbosity"]) == ("low", "low")

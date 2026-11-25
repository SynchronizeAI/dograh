import pytest

from api.services.pipecat.service_factory import create_llm_service_from_provider


@pytest.mark.parametrize("model", ["gpt-5-mini", "gpt-4.1-mini"])
@pytest.mark.parametrize("base_url", [None, "https://api.openai.com/v1"])
def test_official_openai_endpoint_disables_storage(model, base_url, monkeypatch):
    monkeypatch.setattr(
        "api.services.pipecat.service_factory._validate_runtime_service_url",
        lambda *_args: None,
    )
    kwargs = {} if base_url is None else {"base_url": base_url}
    service = create_llm_service_from_provider(
        "openai", model, "test-key", **kwargs
    )

    params = service.build_chat_completion_params({"messages": []})

    assert params["store"] is False


@pytest.mark.parametrize("model", ["gpt-5-mini", "gpt-4.1-mini"])
def test_openai_compatible_endpoint_omits_store(model, monkeypatch):
    monkeypatch.setattr(
        "api.services.pipecat.service_factory._validate_runtime_service_url",
        lambda *_args: None,
    )
    service = create_llm_service_from_provider(
        "openai",
        model,
        "test-key",
        base_url="https://proxy.example/v1",
    )

    params = service.build_chat_completion_params({"messages": []})

    assert "store" not in params


def test_openai_compatible_providers_get_no_openai_store_field():
    service = create_llm_service_from_provider(
        "atlascloud", "qwen/qwen3.5-flash", "atlas-key"
    )

    assert "store" not in service.build_chat_completion_params({"messages": []})

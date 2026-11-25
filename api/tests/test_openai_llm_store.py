import pytest

from api.services.pipecat.service_factory import create_llm_service_from_provider


@pytest.mark.parametrize("model", ["gpt-5-mini", "gpt-4.1-mini"])
def test_openai_chat_completions_never_ask_openai_to_store_the_call(model):
    service = create_llm_service_from_provider("openai", model, "test-key")

    params = service.build_chat_completion_params({"messages": []})

    assert params["store"] is False


def test_openai_compatible_providers_get_no_openai_store_field():
    service = create_llm_service_from_provider(
        "atlascloud", "qwen/qwen3.5-flash", "atlas-key"
    )

    assert "store" not in service.build_chat_completion_params({"messages": []})

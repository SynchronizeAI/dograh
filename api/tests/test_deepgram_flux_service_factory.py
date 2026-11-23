from types import SimpleNamespace
from unittest.mock import patch

import pytest
from pipecat.services.settings import NOT_GIVEN
from pipecat.transcriptions.language import Language

from api.services.configuration.registry import (
    DeepgramSTTConfiguration,
    DograhSTTService,
    ServiceProviders,
)
from api.services.pipecat.audio_config import AudioConfig
from api.services.pipecat.service_factory import create_stt_service


def test_deepgram_stt_schema_includes_flux_multilingual_language_options():
    language_schema = DeepgramSTTConfiguration.model_json_schema()["properties"][
        "language"
    ]

    assert "flux-general-multi" in language_schema["model_options"]
    assert "multi" in language_schema["model_options"]["flux-general-multi"]
    assert "es" in language_schema["model_options"]["flux-general-multi"]


@pytest.mark.parametrize(
    ("language", "hint"),
    [("es", Language.ES), ("en-GB", Language.EN), ("pt-BR", Language.PT)],
)
def test_create_deepgram_flux_multi_uses_flux_service_with_language_hint(
    language, hint
):
    user_config = SimpleNamespace(
        stt=SimpleNamespace(
            provider=ServiceProviders.DEEPGRAM.value,
            api_key="test-key",
            model="flux-general-multi",
            language=language,
        )
    )
    audio_config = AudioConfig(
        transport_in_sample_rate=16000,
        transport_out_sample_rate=16000,
    )

    with patch(
        "api.services.pipecat.service_factory.DeepgramFluxSTTService"
    ) as mock_service:
        create_stt_service(user_config, audio_config)

    kwargs = mock_service.call_args.kwargs
    assert kwargs["settings"].model == "flux-general-multi"
    assert kwargs["settings"].language_hints == [hint]


def test_create_deepgram_flux_multi_omits_auto_detect_language_hint():
    user_config = SimpleNamespace(
        stt=SimpleNamespace(
            provider=ServiceProviders.DEEPGRAM.value,
            api_key="test-key",
            model="flux-general-multi",
            language="multi",
        )
    )
    audio_config = AudioConfig(
        transport_in_sample_rate=16000,
        transport_out_sample_rate=16000,
    )

    with patch(
        "api.services.pipecat.service_factory.DeepgramFluxSTTService"
    ) as mock_service:
        create_stt_service(user_config, audio_config)

    kwargs = mock_service.call_args.kwargs
    assert kwargs["settings"].model == "flux-general-multi"
    assert kwargs["settings"].language_hints is NOT_GIVEN


def _create_flux_multi(language: str, language_hints: list[str]):
    config = DeepgramSTTConfiguration(
        api_key="test-key",
        model="flux-general-multi",
        language=language,
        language_hints=language_hints,
    )
    audio_config = AudioConfig(
        transport_in_sample_rate=16000,
        transport_out_sample_rate=16000,
    )
    with patch(
        "api.services.pipecat.service_factory.DeepgramFluxSTTService"
    ) as mock_service:
        create_stt_service(SimpleNamespace(stt=config), audio_config)
    return mock_service.call_args.kwargs["settings"]


@pytest.mark.parametrize("schema", [DeepgramSTTConfiguration, DograhSTTService])
def test_flux_eager_settings_omit_published_schema_defaults(schema):
    properties = schema.model_json_schema()["properties"]
    for name in ("enable_eager_end_of_turn", "eager_eot_threshold"):
        assert "default" not in properties[name]


def test_deepgram_stt_schema_shows_language_hints_only_for_flux_multi():
    hints_schema = DeepgramSTTConfiguration.model_json_schema()["properties"][
        "language_hints"
    ]

    assert hints_schema["type"] == "array"
    assert hints_schema["visible_for_models"] == ["flux-general-multi"]
    assert "hi" in hints_schema["examples"]


def test_create_deepgram_flux_multi_sends_every_language_hint():
    settings = _create_flux_multi("multi", ["hi", "en"])

    assert settings.language_hints == [Language.HI, Language.EN]


@pytest.mark.parametrize("schema", [DeepgramSTTConfiguration, DograhSTTService])
@pytest.mark.parametrize("threshold", [0.5, 0.7, 0.85, 0.9])
def test_flux_eot_threshold_accepts_supported_values(schema, threshold):
    assert (
        schema(api_key="test-key", eot_threshold=threshold).eot_threshold == threshold
    )


@pytest.mark.parametrize("schema", [DeepgramSTTConfiguration, DograhSTTService])
@pytest.mark.parametrize("threshold", [0.49, 0.91, float("nan"), float("inf"), None])
def test_flux_eot_threshold_rejects_invalid_values(schema, threshold):
    with pytest.raises(ValueError, match="eot_threshold"):
        schema(api_key="test-key", eot_threshold=threshold)


@pytest.mark.parametrize("schema", [DeepgramSTTConfiguration, DograhSTTService])
@pytest.mark.parametrize("timeout", [500, 3000, 7000, 10000])
def test_flux_eot_timeout_accepts_supported_values(schema, timeout):
    assert schema(api_key="test-key", eot_timeout_ms=timeout).eot_timeout_ms == timeout


@pytest.mark.parametrize("schema", [DeepgramSTTConfiguration, DograhSTTService])
@pytest.mark.parametrize("timeout", [499, 10001, None])
def test_flux_eot_timeout_rejects_invalid_values(schema, timeout):
    with pytest.raises(ValueError, match="eot_timeout_ms"):
        schema(api_key="test-key", eot_timeout_ms=timeout)


@pytest.mark.parametrize(
    ("schema", "model", "factory"),
    [
        (DeepgramSTTConfiguration, "flux-general-en", "DeepgramFluxSTTService"),
        (DeepgramSTTConfiguration, "flux-general-multi", "DeepgramFluxSTTService"),
        (DograhSTTService, "default", "DograhFluxSTTService"),
    ],
)
def test_flux_eot_settings_reach_service(schema, model, factory):
    config = schema(
        api_key="test-key",
        model=model,
        enable_eager_end_of_turn=True,
        eager_eot_threshold=0.4,
        eot_threshold=0.85,
        eot_timeout_ms=7000,
    )
    with patch(f"api.services.pipecat.service_factory.{factory}") as service:
        create_stt_service(
            SimpleNamespace(stt=config),
            AudioConfig(
                transport_in_sample_rate=16000, transport_out_sample_rate=16000
            ),
        )
    settings = service.call_args.kwargs["settings"]
    assert service.call_args.kwargs["enable_eager_end_of_turn"] is True
    assert settings.eager_eot_threshold == 0.4
    assert settings.eot_threshold == 0.85
    assert settings.eot_timeout_ms == 7000


@pytest.mark.parametrize("schema", [DeepgramSTTConfiguration, DograhSTTService])
@pytest.mark.parametrize("eager,committed", [(0.7, 0.5), (0.5, 0.5)])
def test_flux_eager_threshold_must_be_below_commit_threshold(schema, eager, committed):
    with pytest.raises(ValueError, match="eager_eot_threshold must be lower than"):
        schema(
            api_key="test-key",
            enable_eager_end_of_turn=True,
            eager_eot_threshold=eager,
            eot_threshold=committed,
        )


@pytest.mark.parametrize("schema", [DeepgramSTTConfiguration, DograhSTTService])
@pytest.mark.parametrize(("eager", "committed"), [(0.4, 0.5), (0.5, 0.7)])
def test_flux_eager_threshold_accepts_ordered_commit_thresholds(
    schema, eager, committed
):
    config = schema(
        api_key="test-key", eager_eot_threshold=eager, eot_threshold=committed
    )

    assert config.eager_eot_threshold == eager
    assert config.eot_threshold == committed


@pytest.mark.parametrize("schema", [DeepgramSTTConfiguration, DograhSTTService])
@pytest.mark.parametrize("threshold", [0.3, 0.4, 0.5, 0.7])
def test_flux_eager_threshold_accepts_supported_values(schema, threshold):
    config = schema(api_key="test-key", eager_eot_threshold=threshold)
    assert config.eager_eot_threshold == threshold
    assert config.enable_eager_end_of_turn is False


@pytest.mark.parametrize("schema", [DeepgramSTTConfiguration, DograhSTTService])
@pytest.mark.parametrize(
    "threshold", [0.29, 0.71, -1, 1, float("nan"), float("inf"), None]
)
def test_flux_eager_threshold_rejects_invalid_values(schema, threshold):
    with pytest.raises(ValueError, match="eager_eot_threshold"):
        schema(api_key="test-key", eager_eot_threshold=threshold)


@pytest.mark.parametrize(
    ("schema", "model", "factory"),
    [
        (DeepgramSTTConfiguration, "flux-general-en", "DeepgramFluxSTTService"),
        (DeepgramSTTConfiguration, "flux-general-multi", "DeepgramFluxSTTService"),
        (DograhSTTService, "default", "DograhFluxSTTService"),
    ],
)
@pytest.mark.parametrize("enabled", [None, False, True])
def test_flux_eager_constructor_is_opt_in(schema, model, factory, enabled):
    fields = {} if enabled is None else {"enable_eager_end_of_turn": enabled}
    config = schema(api_key="test-key", model=model, eager_eot_threshold=0.4, **fields)
    with patch(f"api.services.pipecat.service_factory.{factory}") as service:
        create_stt_service(
            SimpleNamespace(stt=config),
            AudioConfig(
                transport_in_sample_rate=16000, transport_out_sample_rate=16000
            ),
            keyterms=["Dograh"],
        )
    kwargs = service.call_args.kwargs
    assert kwargs.get("enable_eager_end_of_turn", False) is bool(enabled)
    if not enabled:
        assert "enable_eager_end_of_turn" not in kwargs
    settings = kwargs["settings"]
    assert settings.eager_eot_threshold == (0.4 if enabled else 0.5)
    assert settings.eot_threshold == 0.7
    assert settings.eot_timeout_ms == 3000
    assert settings.keyterm == ["Dograh"]
    assert kwargs["sample_rate"] == 16000
    assert kwargs["should_interrupt"] is False


def test_create_deepgram_flux_multi_combines_language_with_hints_once():
    settings = _create_flux_multi("hi", ["en", "hi"])

    assert settings.language_hints == [Language.HI, Language.EN]


def test_deepgram_stt_rejects_language_hints_flux_does_not_support():
    with pytest.raises(ValueError, match="ta"):
        DeepgramSTTConfiguration(
            api_key="test-key",
            model="flux-general-multi",
            language_hints=["en", "ta"],
        )

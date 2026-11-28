from types import SimpleNamespace
from unittest.mock import patch

import pytest
from pipecat.services.settings import NOT_GIVEN
from pipecat.transcriptions.language import Language
from pipecat.turns.user_turn_strategies import ExternalUserTurnStrategies
from pydantic import TypeAdapter, ValidationError

from api.schemas.ai_model_configuration import EffectiveAIModelConfiguration
from api.services.configuration.options import SONIOX_STT_LANGUAGES, SONIOX_STT_MODELS
from api.services.configuration.registry import (
    ASSEMBLYAI_REALTIME_AGENT_MODEL,
    ServiceProviders,
    SonioxSTTConfiguration,
    STTConfig,
)
from api.services.pipecat.audio_config import AudioConfig
from api.services.pipecat.fallback_stt import FallbackSTTService
from api.services.pipecat.service_factory import (
    cascade_uses_external_turns,
    create_stt_service,
    stt_uses_external_turns,
)


def _audio_config() -> AudioConfig:
    return AudioConfig(
        transport_in_sample_rate=16000,
        transport_out_sample_rate=16000,
    )


def _soniox_config(**stt) -> SimpleNamespace:
    config = TypeAdapter(STTConfig).validate_python(
        {"provider": "soniox", "api_key": "test-key", **stt}
    )
    return SimpleNamespace(stt=config)


def test_soniox_stt_configuration_exposes_defaults_and_languages():
    config = SonioxSTTConfiguration(api_key="test-key")
    language_schema = SonioxSTTConfiguration.model_json_schema()["properties"][
        "language"
    ]

    assert config.provider == ServiceProviders.SONIOX
    assert config.model == "stt-rt-v5"
    assert config.language == "multi"
    assert SONIOX_STT_MODELS == ("stt-rt-v5", "stt-rt-v6")
    assert config.enable_endpoint_detection is False
    assert config.max_endpoint_delay_ms is None
    assert config.endpoint_sensitivity is None
    assert config.endpoint_latency_adjustment_level is None
    assert "multi" in SONIOX_STT_LANGUAGES
    assert "hi" in SONIOX_STT_LANGUAGES
    assert language_schema["examples"] == list(SONIOX_STT_LANGUAGES)


def test_soniox_stt_config_parses_from_stt_union():
    user_config = _soniox_config(language="hi")

    assert isinstance(user_config.stt, SonioxSTTConfiguration)
    assert user_config.stt.language == "hi"


def test_soniox_stt_uses_local_turns_with_language_hint():
    user_config = _soniox_config(language="hi")

    assert not stt_uses_external_turns(user_config)

    with patch("api.services.pipecat.service_factory.SonioxSTTService") as stt_service:
        create_stt_service(user_config, _audio_config())

    stt_service.assert_called_once()
    kwargs = stt_service.call_args.kwargs
    assert kwargs["api_key"] == "test-key"
    assert kwargs["sample_rate"] == 16000
    assert kwargs["vad_force_turn_endpoint"] is True
    assert kwargs["settings"].model == "stt-rt-v5"
    assert kwargs["settings"].language_hints == [Language.HI]


def test_soniox_stt_multi_language_sends_no_hints():
    user_config = _soniox_config()

    with patch("api.services.pipecat.service_factory.SonioxSTTService") as stt_service:
        create_stt_service(user_config, _audio_config())

    kwargs = stt_service.call_args.kwargs
    assert kwargs["settings"].language_hints is NOT_GIVEN
    assert kwargs["settings"].context is NOT_GIVEN


def test_soniox_stt_passes_keyterms_as_context_terms():
    user_config = _soniox_config(language="en")

    with patch("api.services.pipecat.service_factory.SonioxSTTService") as stt_service:
        create_stt_service(user_config, _audio_config(), keyterms=["Dograh", "Soniox"])

    kwargs = stt_service.call_args.kwargs
    assert kwargs["settings"].context.terms == ["Dograh", "Soniox"]


def test_soniox_endpoint_detection_uses_soniox_turns():
    user_config = _soniox_config(
        language="ko",
        model="stt-rt-v6",
        enable_endpoint_detection=True,
        max_endpoint_delay_ms=1500,
    )

    assert stt_uses_external_turns(user_config)
    assert cascade_uses_external_turns(user_config)

    service = create_stt_service(user_config, _audio_config())

    assert service._vad_force_turn_endpoint is False
    assert isinstance(
        service.service_metadata_frame().user_turn_strategies,
        ExternalUserTurnStrategies,
    )
    assert service._settings.model == "stt-rt-v6"
    assert service._settings.max_endpoint_delay_ms == 1500
    assert service._settings.language_hints == [Language.KO]


def test_soniox_without_endpoint_detection_keeps_local_turns():
    user_config = _soniox_config(max_endpoint_delay_ms=1500)

    assert not stt_uses_external_turns(user_config)
    assert not cascade_uses_external_turns(user_config)

    service = create_stt_service(user_config, _audio_config())

    assert service._vad_force_turn_endpoint is True
    assert service.service_metadata_frame().user_turn_strategies is None
    # The delay only means something to Soniox's own endpointing.
    assert service._settings.max_endpoint_delay_ms is None


def _soniox_with_assemblyai_backup(**soniox):
    return EffectiveAIModelConfiguration.model_validate(
        {
            "stt": {
                "provider": "soniox",
                "api_key": "primary-key",
                "language": "ko",
                **soniox,
            },
            "stt_fallback": {
                "rules": [
                    {
                        "condition": {"type": "error"},
                        "target": {
                            "provider": "assemblyai",
                            "api_key": "backup-key",
                            "model": ASSEMBLYAI_REALTIME_AGENT_MODEL,
                            "language": "ko",
                        },
                    }
                ]
            },
        }
    )


def test_soniox_endpoint_detection_backup_assemblyai_ends_turns_itself():
    stt = create_stt_service(
        _soniox_with_assemblyai_backup(enable_endpoint_detection=True),
        _audio_config(),
    )

    assert isinstance(stt, FallbackSTTService)
    assert stt.primary._vad_force_turn_endpoint is False
    assert stt.backup._vad_force_turn_endpoint is False
    assert isinstance(
        stt.backup.service_metadata_frame().user_turn_strategies,
        ExternalUserTurnStrategies,
    )


def test_soniox_without_endpoint_detection_keeps_assemblyai_backup_on_local_turns():
    stt = create_stt_service(_soniox_with_assemblyai_backup(), _audio_config())

    assert isinstance(stt, FallbackSTTService)
    assert stt.primary._vad_force_turn_endpoint is True
    assert stt.backup._vad_force_turn_endpoint is True
    assert stt.backup.service_metadata_frame().user_turn_strategies is None


def test_soniox_endpoint_tuning_reaches_soniox_when_endpointing_is_on():
    service = create_stt_service(
        _soniox_config(
            enable_endpoint_detection=True,
            endpoint_sensitivity=0.6,
            endpoint_latency_adjustment_level=2,
        ),
        _audio_config(),
    )

    assert service._settings.endpoint_sensitivity == 0.6
    assert service._settings.endpoint_latency_adjustment_level == 2
    # Left unset, Soniox keeps its own default delay.
    assert service._settings.max_endpoint_delay_ms is None


def test_soniox_endpoint_tuning_keeps_zero_as_a_chosen_value():
    service = create_stt_service(
        _soniox_config(
            enable_endpoint_detection=True,
            endpoint_sensitivity=0.0,
            endpoint_latency_adjustment_level=0,
        ),
        _audio_config(),
    )

    assert service._settings.endpoint_sensitivity == 0.0
    assert service._settings.endpoint_latency_adjustment_level == 0


def test_soniox_endpoint_tuning_unset_leaves_soniox_defaults():
    service = create_stt_service(
        _soniox_config(enable_endpoint_detection=True), _audio_config()
    )

    assert service._settings.endpoint_sensitivity is None
    assert service._settings.endpoint_latency_adjustment_level is None


def test_soniox_endpoint_tuning_is_ignored_without_endpoint_detection():
    service = create_stt_service(
        _soniox_config(endpoint_sensitivity=0.6, endpoint_latency_adjustment_level=2),
        _audio_config(),
    )

    assert service._settings.endpoint_sensitivity is None
    assert service._settings.endpoint_latency_adjustment_level is None


@pytest.mark.parametrize(
    "field,value",
    [
        ("endpoint_sensitivity", -1.01),
        ("endpoint_sensitivity", 1.01),
        ("endpoint_sensitivity", float("nan")),
        ("endpoint_latency_adjustment_level", -1),
        ("endpoint_latency_adjustment_level", 4),
    ],
)
def test_soniox_endpoint_tuning_rejects_values_outside_soniox_bounds(field, value):
    with pytest.raises(ValidationError, match=field):
        SonioxSTTConfiguration(api_key="test-key", **{field: value})


@pytest.mark.parametrize(
    "field,value",
    [
        ("endpoint_sensitivity", -1.0),
        ("endpoint_sensitivity", 1.0),
        ("endpoint_latency_adjustment_level", 0),
        ("endpoint_latency_adjustment_level", 3),
    ],
)
def test_soniox_endpoint_tuning_accepts_the_bounds(field, value):
    config = SonioxSTTConfiguration(api_key="test-key", **{field: value})

    assert getattr(config, field) == value


def test_soniox_as_backup_carries_endpoint_tuning():
    config = EffectiveAIModelConfiguration.model_validate(
        {
            "stt": {
                "provider": "assemblyai",
                "api_key": "primary-key",
                "model": ASSEMBLYAI_REALTIME_AGENT_MODEL,
                "language": "ko",
            },
            "stt_fallback": {
                "rules": [
                    {
                        "condition": {"type": "error"},
                        "target": {
                            "provider": "soniox",
                            "api_key": "backup-key",
                            "language": "ko",
                            "enable_endpoint_detection": True,
                            "endpoint_sensitivity": 0.4,
                            "endpoint_latency_adjustment_level": 1,
                        },
                    }
                ]
            },
        }
    )

    stt = create_stt_service(config, _audio_config())

    assert isinstance(stt, FallbackSTTService)
    assert stt.backup._settings.endpoint_sensitivity == 0.4
    assert stt.backup._settings.endpoint_latency_adjustment_level == 1

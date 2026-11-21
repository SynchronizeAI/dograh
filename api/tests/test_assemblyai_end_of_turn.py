from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse

import pytest
from pipecat.services.assemblyai.stt import is_u3_pro_model
from pipecat.transcriptions.language import Language
from pipecat.turns.user_turn_strategies import ExternalUserTurnStrategies
from pydantic import ValidationError

from api.schemas.ai_model_configuration import EffectiveAIModelConfiguration
from api.schemas.workflow_configurations import WorkflowConfigurationDefaults
from api.services.configuration.registry import (
    ASSEMBLYAI_REALTIME_AGENT_MODEL,
    AssemblyAISTTConfiguration,
)
from api.services.pipecat.fallback_stt import FallbackSTTService
from api.services.pipecat.service_factory import (
    cascade_uses_external_turns,
    create_stt_service,
    stt_uses_assemblyai_end_of_turn,
)

AUDIO = SimpleNamespace(transport_in_sample_rate=16000)


def _service(model, language="en", strategy="transcription"):
    config = AssemblyAISTTConfiguration(
        api_key="test-key", model=model, language=language
    )
    return create_stt_service(
        SimpleNamespace(stt=config),
        AUDIO,
        turn_stop_strategy=strategy,
    )


def _query(service):
    return parse_qs(urlparse(service._build_ws_url()).query)


def test_turn_stop_defaults_to_transcription():
    assert WorkflowConfigurationDefaults().turn_stop_strategy == "transcription"


def test_assemblyai_turn_stop_is_selectable_and_unknown_values_are_rejected():
    selected = WorkflowConfigurationDefaults(turn_stop_strategy="assemblyai")
    assert selected.turn_stop_strategy == "assemblyai"
    with pytest.raises(ValidationError):
        WorkflowConfigurationDefaults(turn_stop_strategy="livekit")


def test_assemblyai_37_agent_is_selectable_and_default_model_is_unchanged():
    config = AssemblyAISTTConfiguration(api_key="test-key")
    assert config.model == "u3-rt-pro"
    properties = AssemblyAISTTConfiguration.model_json_schema()["properties"]
    assert ASSEMBLYAI_REALTIME_AGENT_MODEL in properties["model"]["examples"]
    assert properties["language"]["model_options"]["u3-rt-pro"] == [
        "en",
        "es",
        "de",
        "fr",
        "pt",
        "it",
    ]
    assert (
        "ko" in properties["language"]["model_options"][ASSEMBLYAI_REALTIME_AGENT_MODEL]
    )


def test_default_assemblyai_keeps_pipecat_turn_mode():
    service = _service("u3-rt-pro")

    assert service._vad_force_turn_endpoint is True
    assert service.service_metadata_frame().user_turn_strategies is None
    assert _query(service)["speech_model"] == ["u3-rt-pro"]
    assert _query(service)["min_turn_silence"] == ["100"]


def test_opt_in_sends_37_agent_model_and_lets_assemblyai_end_the_turn():
    service = _service(
        ASSEMBLYAI_REALTIME_AGENT_MODEL, language="ko", strategy="assemblyai"
    )

    assert is_u3_pro_model(ASSEMBLYAI_REALTIME_AGENT_MODEL)
    assert service._vad_force_turn_endpoint is False
    assert isinstance(
        service.service_metadata_frame().user_turn_strategies,
        ExternalUserTurnStrategies,
    )
    query = _query(service)
    assert query["speech_model"] == [ASSEMBLYAI_REALTIME_AGENT_MODEL]
    assert query["language_codes"] == ['["ko"]']
    assert service._settings.language_codes == [Language.KO]
    assert "min_turn_silence" not in query
    assert "max_turn_silence" not in query


def test_assemblyai_end_of_turn_builds_with_stt_fallback():
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
                            "provider": "elevenlabs",
                            "api_key": "backup-key",
                            "model": "scribe_v3_live",
                            "language": "ko",
                        },
                    }
                ]
            },
        }
    )

    stt = create_stt_service(config, AUDIO, turn_stop_strategy="assemblyai")

    assert isinstance(stt, FallbackSTTService)
    assert stt.strategy.active_service is stt.primary
    assert stt.primary._vad_force_turn_endpoint is False
    assert isinstance(
        stt.primary.service_metadata_frame().user_turn_strategies,
        ExternalUserTurnStrategies,
    )
    assert stt.backup is not None


def test_37_agent_without_opt_in_keeps_pipecat_turn_mode():
    service = _service(ASSEMBLYAI_REALTIME_AGENT_MODEL, language="ko")

    assert service._vad_force_turn_endpoint is True
    assert _query(service)["speech_model"] == [ASSEMBLYAI_REALTIME_AGENT_MODEL]
    assert _query(service)["min_turn_silence"] == ["100"]


def test_note_taking_model_and_other_providers_do_not_take_assemblyai_turns():
    note_taker = SimpleNamespace(
        stt=SimpleNamespace(provider="assemblyai", model="assemblyai-stt-3-7-realtime")
    )
    nova = SimpleNamespace(
        stt=SimpleNamespace(provider="deepgram", model="nova-3-general", language="ko")
    )
    flux = SimpleNamespace(
        stt=SimpleNamespace(provider="deepgram", model="flux-general-en", language="en")
    )
    agent = SimpleNamespace(
        stt=SimpleNamespace(
            provider="assemblyai", model=ASSEMBLYAI_REALTIME_AGENT_MODEL
        )
    )

    assert not is_u3_pro_model("assemblyai-stt-3-7-realtime")
    assert not stt_uses_assemblyai_end_of_turn(note_taker, "assemblyai")
    assert not cascade_uses_external_turns(nova, "assemblyai")
    assert not cascade_uses_external_turns(agent, "transcription")
    assert not cascade_uses_external_turns(agent, None)
    assert cascade_uses_external_turns(agent, "assemblyai")
    assert cascade_uses_external_turns(flux, "transcription")

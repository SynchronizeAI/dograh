"""STT fallback policy validation and factory wiring."""

from unittest.mock import patch

import pytest
from pydantic import ValidationError

from api.schemas.ai_model_configuration import EffectiveAIModelConfiguration
from api.schemas.llm_fallback import STTFallbackPolicy
from api.schemas.model_connections import ServiceSelection
from api.services.configuration.masking import mask_user_config


def test_only_error_condition_is_accepted():
    policy = STTFallbackPolicy[ServiceSelection].model_validate(
        {
            "rules": [
                {
                    "condition": {"type": "error"},
                    "target": {
                        "provider_connection_uuid": "11111111-1111-4111-8111-111111111111"
                    },
                }
            ]
        }
    )
    assert policy.rules[0].condition.type == "error"


_VALID_TARGET = {"provider_connection_uuid": "11111111-1111-4111-8111-111111111111"}


@pytest.mark.parametrize(
    "policy,loc",
    [
        (
            {
                "rules": [
                    {
                        "condition": {"type": "no_output", "after_ms": 900},
                        "target": _VALID_TARGET,
                    }
                ]
            },
            ("rules", 0, "condition", "type"),
        ),
        (
            {
                "rules": [
                    {"condition": {"type": "error"}, "target": _VALID_TARGET},
                    {"condition": {"type": "error"}, "target": _VALID_TARGET},
                ]
            },
            ("rules",),
        ),
        ({"version": 2, "rules": []}, ("version",)),
    ],
)
def test_schema_rejects_invalid_policies(policy, loc):
    with pytest.raises(ValidationError) as exc:
        STTFallbackPolicy[ServiceSelection].model_validate(policy)
    assert exc.value.errors()[0]["loc"] == loc


def stt_configuration(*, error_target=None, stt=None):
    payload = {
        "llm": {"provider": "google", "api_key": "x", "model": "gemini-3.5-flash"},
        "stt": stt
        or {"provider": "deepgram", "api_key": "primary", "model": "flux-general-en"},
        "tts": {
            "provider": "openai",
            "api_key": "x",
            "model": "tts-1",
            "voice": "alloy",
        },
    }
    if error_target:
        payload["stt_fallback"] = {
            "version": 1,
            "rules": [{"condition": {"type": "error"}, "target": error_target}],
        }
    return EffectiveAIModelConfiguration.model_validate(payload)


_SCRIBE = {
    "provider": "elevenlabs",
    "api_key": "backup-secret",
    "model": "scribe_v3_live",
    "language": "ko",
}


def test_effective_configuration_rejects_dograh_primary():
    payload = stt_configuration(error_target=_SCRIBE).model_dump(
        exclude_computed_fields=True
    )
    payload["stt"] = {"provider": "dograh", "api_key": "primary"}
    with pytest.raises(ValidationError, match="non-Dograh primary STT"):
        EffectiveAIModelConfiguration.model_validate(payload)


def test_effective_configuration_rejects_dograh_target():
    with pytest.raises(ValidationError, match="cannot be a fallback target"):
        stt_configuration(error_target={"provider": "dograh", "api_key": "managed"})


def test_effective_configuration_rejects_same_provider():
    with pytest.raises(ValidationError, match="different provider"):
        stt_configuration(
            error_target={
                "provider": "deepgram",
                "api_key": "other",
                "model": "nova-3-general",
            }
        )


def test_no_policy_returns_plain_stt():
    from api.services.pipecat.service_factory import create_stt_service

    with patch(
        "api.services.pipecat.service_factory.create_single_stt_service"
    ) as create:
        primary = object()
        create.return_value = primary
        assert create_stt_service(stt_configuration(), None) is primary
        create.assert_called_once()


def test_rules_build_switcher():
    from pipecat.processors.frame_processor import FrameProcessor

    from api.services.pipecat.fallback_stt import FallbackSTTService
    from api.services.pipecat.service_factory import create_stt_service

    with patch(
        "api.services.pipecat.service_factory.create_single_stt_service"
    ) as create:
        primary = FrameProcessor(name="primary")
        backup = FrameProcessor(name="backup")
        create.side_effect = [primary, backup]
        config = stt_configuration(error_target=_SCRIBE)
        result = create_stt_service(config, None)
    assert isinstance(result, FallbackSTTService)
    assert result.primary is primary
    assert result.backup is backup
    assert create.call_args_list[1].args[0].stt == config.stt_fallback.rules[0].target
    assert create.call_args_list[1].kwargs["register_campaign_errors"] is False
    assert result.primary is not None


def test_broken_backup_returns_primary():
    from api.services.pipecat.service_factory import create_stt_service

    with patch(
        "api.services.pipecat.service_factory.create_single_stt_service"
    ) as create:
        primary = object()
        create.side_effect = [primary, RuntimeError("bad")]
        result = create_stt_service(stt_configuration(error_target=_SCRIBE), None)
    assert result is primary


def test_masking_hides_stt_fallback_secrets():
    masked = mask_user_config(stt_configuration(error_target=_SCRIBE))
    assert masked["stt_fallback"]["rules"][0]["target"]["model"] == "scribe_v3_live"
    assert "backup-secret" not in str(masked)

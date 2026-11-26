"""Sync fork: `configure_inbound` keeps a front door's VoiceUrl (`TWILIO_FRONT_DOOR_URL`)."""

from unittest.mock import patch

import pytest

from api.services.telephony.providers.twilio.provider import TwilioProvider
from api.tests.telephony.test_phone_number_validation import _StubResponse, _StubSession

DOOR = "https://app.example.com/api/voip/route"
DOGRAH = "https://dograh.example.com/api/v1/telephony/inbound/run"


async def _sync(voice_url: str, webhook_url: str | None = DOGRAH) -> _StubSession:
    provider = TwilioProvider(
        {"account_sid": "AC1", "auth_token": "t", "from_numbers": []}
    )
    number = {"sid": "PN1", "phone_number": "+15551230002", "voice_url": voice_url}
    session = _StubSession(
        [
            _StubResponse(200, data={"incoming_phone_numbers": [number]}),
            _StubResponse(200),
        ]
    )
    with patch(
        "api.services.telephony.providers.twilio.provider.aiohttp.ClientSession",
        return_value=session,
    ):
        assert (await provider.configure_inbound("+15551230002", webhook_url)).ok
    return session


@pytest.mark.asyncio
@pytest.mark.parametrize("webhook_url", [DOGRAH, None])
async def test_a_front_door_keeps_its_voice_url(monkeypatch, webhook_url):
    monkeypatch.setenv("TWILIO_FRONT_DOOR_URL", DOOR)
    assert len((await _sync(DOOR, webhook_url)).requests) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("door", [None, DOOR])
async def test_any_other_voice_url_is_written(monkeypatch, door):
    if door:
        monkeypatch.setenv("TWILIO_FRONT_DOOR_URL", door)
    else:
        monkeypatch.delenv("TWILIO_FRONT_DOOR_URL", raising=False)
    session = await _sync("https://gate.example.com/credits")
    assert session.requests[1][1]["data"] == {"VoiceUrl": DOGRAH, "VoiceMethod": "POST"}

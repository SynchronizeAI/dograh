"""한국AI통신 inbound provider: detection, parsing, signature, VoiceML, stream start."""

import json
from unittest.mock import AsyncMock, patch

import pytest

from api.services.telephony.providers.koreaaitel.provider import KoreaAITelProvider
from api.services.telephony.providers.twilio.provider import TwilioProvider

URL = "https://dograh.example/api/v1/telephony/inbound/run"
SECRET = "whsec_test"
WEBHOOK = {
    "CallId": "CA9f8e",
    "AccountId": "AC1a2b",
    "From": "01012345678",
    "To": "07012340001",
    "ForwardedFrom": "0212345678",
    "CallStatus": "ringing",
    "Direction": "inbound",
}
# Computed independently, following the vendor's documented algorithm:
# printf '%s' "$URL" + sorted key/value pairs | openssl dgst -sha256 -hmac "$SECRET" -binary | base64
SIGNATURE = "9iw6q8d6Bq+jdRnF8Dfmy25LDjR0NubO8eYGx8dlROA="


def _provider() -> KoreaAITelProvider:
    return KoreaAITelProvider({"account_id": "AC1a2b", "signing_secret": SECRET})


def test_claims_vendor_webhook_and_twilio_does_not():
    headers = {
        "x-signature": SIGNATURE,
        "content-type": "application/x-www-form-urlencoded",
    }
    assert KoreaAITelProvider.can_handle_webhook(WEBHOOK, headers)
    assert not TwilioProvider.can_handle_webhook(WEBHOOK, headers)


def test_does_not_claim_twilio_webhook():
    twilio = {
        "CallSid": "CA1",
        "AccountSid": "AC1",
        "ApiVersion": "2010-04-01",
        "From": "+15551230001",
        "To": "+15551230002",
    }
    assert not KoreaAITelProvider.can_handle_webhook(
        twilio, {"x-twilio-signature": "sig"}
    )
    # Vendor-shaped data without the vendor's signature header is not claimed.
    assert not KoreaAITelProvider.can_handle_webhook(WEBHOOK, {})


def test_parse_inbound_webhook():
    data = KoreaAITelProvider.parse_inbound_webhook(WEBHOOK)
    assert data.provider == "koreaaitel"
    assert data.call_id == "CA9f8e"
    assert data.account_id == "AC1a2b"
    assert data.from_number == "+821012345678"
    assert data.to_number == "+827012340001"
    assert data.direction == "inbound"
    assert data.raw_data["ForwardedFrom"] == "0212345678"


def test_parse_withheld_caller_is_blank_everywhere():
    data = KoreaAITelProvider.parse_inbound_webhook(
        {**WEBHOOK, "CallerIdRestricted": "true"}
    )
    assert data.from_number == ""
    assert data.to_number == "+827012340001"
    assert data.raw_data["From"] == ""
    assert data.raw_data["CallId"] == "CA9f8e"
    assert data.raw_data["ForwardedFrom"] == "0212345678"


def test_parse_redacts_call_token_and_preserves_other_webhook_fields():
    webhook = {**WEBHOOK, "CallToken": "secret-capability"}

    data = KoreaAITelProvider.parse_inbound_webhook(webhook)

    assert data.raw_data == {**WEBHOOK, "CallToken": "[REDACTED]"}
    assert webhook["CallToken"] == "secret-capability"


def test_validate_account_id():
    assert KoreaAITelProvider.validate_account_id({"account_id": "AC1a2b"}, "AC1a2b")
    assert not KoreaAITelProvider.validate_account_id({"account_id": "AC1a2b"}, "AC9")
    assert not KoreaAITelProvider.validate_account_id({"account_id": "AC1a2b"}, "")


@pytest.mark.asyncio
async def test_signature_valid():
    assert await _provider().verify_inbound_signature(
        URL, WEBHOOK, {"X-Signature": SIGNATURE}
    )


@pytest.mark.asyncio
async def test_signature_invalid_or_missing():
    provider = _provider()
    tampered = {**WEBHOOK, "From": "01099999999"}
    assert not await provider.verify_inbound_signature(
        URL, tampered, {"x-signature": SIGNATURE}
    )
    assert not await provider.verify_inbound_signature(
        URL + "?x=1", WEBHOOK, {"x-signature": SIGNATURE}
    )
    assert not await provider.verify_inbound_signature(URL, WEBHOOK, {})


@pytest.mark.asyncio
async def test_inbound_voiceml_connects_stream_without_action():
    ws_url = "wss://dograh.example/api/v1/telephony/ws/7/11/123/tok"
    response = await _provider().start_inbound_stream(
        websocket_url=ws_url,
        workflow_run_id=123,
        normalized_data=None,
        backend_endpoint="https://dograh.example",
    )
    body = response.body.decode()
    assert response.media_type == "application/xml"
    assert f'<Stream url="{ws_url}"/>' in body
    assert "<Connect>" in body and "action" not in body


class _FakeWebSocket:
    def __init__(self, messages):
        self._messages = [json.dumps(m) for m in messages]
        self.closed = None

    async def receive_text(self):
        return self._messages.pop(0)

    async def close(self, code, reason):
        self.closed = (code, reason)


_CONNECTED = {"event": "connected", "protocol": "Call", "version": "1.0.0"}


@pytest.mark.asyncio
async def test_websocket_start_reads_stream_and_call_ids():
    ws = _FakeWebSocket(
        [
            _CONNECTED,
            {
                "event": "start",
                "sequenceNumber": "1",
                "start": {"streamId": "MZ1", "callId": "CA9f8e", "accountId": "AC1a2b"},
            },
        ]
    )
    with patch(
        "api.services.pipecat.run_pipeline.run_pipeline_telephony",
        new_callable=AsyncMock,
    ) as run:
        await _provider().handle_websocket(ws, 7, 11, 123)

    assert ws.closed is None
    kwargs = run.await_args.kwargs
    assert kwargs["provider_name"] == "koreaaitel"
    assert kwargs["call_id"] == "CA9f8e"
    assert kwargs["transport_kwargs"] == {"stream_id": "MZ1"}


@pytest.mark.asyncio
async def test_websocket_twilio_start_is_rejected():
    ws = _FakeWebSocket(
        [
            _CONNECTED,
            {"event": "start", "start": {"streamSid": "MZ1", "callSid": "CA1"}},
        ]
    )
    with patch(
        "api.services.pipecat.run_pipeline.run_pipeline_telephony",
        new_callable=AsyncMock,
    ) as run:
        await _provider().handle_websocket(ws, 7, 11, 123)

    assert ws.closed == (4400, "Missing stream identifiers")
    run.assert_not_awaited()

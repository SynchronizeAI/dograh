"""한국AI통신 outbound: the calls API request, VoiceML, status mapping, signed webhooks."""

import hashlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from api.enums import TelephonyCallStatus as Status
from api.services.telephony.providers.koreaaitel import SPEC, _config_loader
from api.services.telephony.providers.koreaaitel import routes as status_routes
from api.services.telephony.providers.koreaaitel.provider import (
    KoreaAITelProvider,
    _national,
    compute_signature,
)
from api.services.telephony.providers.twilio import routes as twiml_routes

PKG = "api.services.telephony.providers.koreaaitel.provider"
SECRET = "whsec_test"
NUMBERS = ["+827012340001", "+827012340002"]
BASE = "https://dograh.example/api/v1/telephony"
URL = f"{BASE}/twiml?workflow_id=7&workflow_run_id=42&organization_id=11"
STATUS_URL = f"{BASE}/koreaaitel/status-callback/42"


def _provider(**overrides) -> KoreaAITelProvider:
    config = {
        "account_id": "AC1a2b",
        "signing_secret": SECRET,
        "api_key": "sk_test",
        "from_numbers": NUMBERS,
        "default_from_number": NUMBERS[0],
        **overrides,
    }
    return KoreaAITelProvider(config)


def _api(status: int, body: dict):
    """Mock aiohttp, so the vendor's calls API answers ``status``/``body``; no network."""
    response = MagicMock(status=status, json=AsyncMock(return_value=body))
    response.__aenter__ = AsyncMock(return_value=response)
    session = MagicMock(post=MagicMock(return_value=response))
    session.__aenter__ = AsyncMock(return_value=session)
    return session, patch(f"{PKG}.aiohttp.ClientSession", return_value=session)


def _endpoints():
    ends = ("https://dograh.example", "wss://dograh.example")
    return patch(f"{PKG}.get_backend_endpoints", AsyncMock(return_value=ends))


async def _call(provider, webhook_url=URL, **kwargs):
    kwargs = {"to_number": "+821012345678", "workflow_run_id": 42, **kwargs}
    with _endpoints():
        return await provider.initiate_call(webhook_url=webhook_url, **kwargs)


async def test_initiate_call_posts_the_documented_request():
    session, api = _api(201, {"callId": "CA9f8e", "status": "queued"})
    with api:
        result = await _call(_provider(), from_number=NUMBERS[1])  # not the default

    url = "https://api.koreaaitel.com/v1/accounts/AC1a2b/calls"
    assert session.post.call_args.args == (url,)
    assert session.post.call_args.kwargs == {
        "headers": {
            "Authorization": "Bearer sk_test",
            "Idempotency-Key": f'"{hashlib.sha256(URL.encode()).hexdigest()}"',
        },
        "json": {
            "To": "01012345678",
            "From": "07012340002",
            "Url": URL,
            "StatusCallback": STATUS_URL,
        },
    }
    assert (result.call_id, result.status) == ("CA9f8e", "queued")
    assert result.caller_number == NUMBERS[1]  # Dograh's own E.164 form
    assert result.provider_metadata == {"call_id": "CA9f8e"}


async def test_idempotency_key_is_one_per_run_and_deployment():
    # The vendor scopes keys to the account; two deployments may both have run 42.
    keys = []
    for url in (URL, URL, URL.replace("dograh.example", "candidate.example")):
        session, api = _api(201, {"callId": "CA1"})
        with api:
            await _call(_provider(), webhook_url=url)
        keys.append(session.post.call_args.kwargs["headers"]["Idempotency-Key"])
    assert keys[0] == keys[1] != keys[2]


@pytest.mark.parametrize(
    "overrides, call_kwargs, message",
    [
        ({"default_from_number": None}, {}, "no default caller ID"),
        ({}, {"from_number": "+827099999999"}, "not an active number"),
        ({"api_key": None}, {}, "not properly configured"),
        ({}, {"workflow_run_id": None}, "no run id"),
    ],
)
async def test_refused_before_any_request(overrides, call_kwargs, message):
    session, api = _api(201, {"callId": "CA1"})
    with api, pytest.raises(ValueError, match=message):
        await _call(_provider(**overrides), **call_kwargs)
    session.post.assert_not_called()


async def test_vendor_refusal_keeps_its_status_and_code():
    _, api = _api(403, {"error": "From not registered", "code": "from_not_owned"})
    with api, pytest.raises(HTTPException) as excinfo:
        await _call(_provider())
    assert excinfo.value.status_code == 403 and "from_not_owned" in excinfo.value.detail


@pytest.mark.parametrize(
    "raw, wire",
    [
        ("+821012345678", "01012345678"),
        ("010-1234-5678", "01012345678"),
        ("+8215881234", "15881234"),  # representative numbers have no leading 0
        ("+14155550100", "+14155550100"),  # the vendor answers UNSUPPORTED_COUNTRY
    ],
)
def test_numbers_are_sent_as_dialled(raw, wire):
    assert _national(raw) == wire


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("completed", Status.COMPLETED),
        ("no-answer", Status.NO_ANSWER),
        ("rejected", Status.BUSY),  # a declined call, as for Vonage
    ],
)
def test_status_callback_maps_to_the_normalized_statuses(raw, expected):
    data = {"CallId": "CA9f8e", "CallStatus": raw, "Duration": "12"}
    pii = {"From": "07012340001", "To": "01012345678", "SipResponseCode": "486"}
    parsed = _provider().parse_status_callback({**data, **pii})
    assert parsed["status"] == expected
    assert (parsed["call_id"], parsed["duration"]) == ("CA9f8e", "12")
    # The callee's number stays out of the run log; the failure detail stays in.
    assert parsed["extra"] == {**data, "SipResponseCode": "486"}


def _request(url, params, signature):
    headers = {"x-signature": signature} if signature else {}
    return SimpleNamespace(
        url=url, headers=headers, form=AsyncMock(return_value=params)
    )


def _routes(module, monkeypatch, provider, mode="koreaaitel", process=None):
    run = SimpleNamespace(
        id=42, mode=mode, workflow=SimpleNamespace(organization_id=11)
    )
    monkeypatch.setattr(
        module.db_client, "get_workflow_run_by_id", AsyncMock(return_value=run)
    )
    monkeypatch.setattr(
        module, "get_telephony_provider_for_run", AsyncMock(return_value=provider)
    )
    if process:
        monkeypatch.setattr(module, "_process_status_update", process)


async def test_outbound_voiceml_fetch_is_signature_verified(monkeypatch):
    _routes(twiml_routes, monkeypatch, _provider())
    params = {"CallId": "CA9f8e", "AccountId": "AC1a2b", "Direction": "outbound"}
    signed = compute_signature(SECRET, URL, params)

    with _endpoints():
        ok = await twiml_routes.handle_twiml_webhook(
            7, 42, 11, _request(URL, params, signed)
        )
    ws_url = "wss://dograh.example/api/v1/telephony/ws/7/11/42"  # the inbound VoiceML
    assert f'<Stream url="{ws_url}"/>' in ok.body.decode() and b"action" not in ok.body

    for url, sent, signature in [
        (URL, {**params, "CallId": "CA-other"}, signed),  # tampered
        (URL.replace("42", "43"), params, signed),  # another run's URL
        (URL, params, None),  # unsigned
    ]:
        with pytest.raises(HTTPException) as excinfo:
            await twiml_routes.handle_twiml_webhook(
                7, 42, 11, _request(url, sent, signature)
            )
        assert excinfo.value.status_code == 401


async def test_status_callback_is_signature_verified_then_processed(monkeypatch):
    process = AsyncMock()
    _routes(status_routes, monkeypatch, _provider(), process=process)
    params = {"CallId": "CA9f8e", "CallStatus": "completed", "Duration": "12"}
    signed = compute_signature(SECRET, STATUS_URL, params)
    handle = status_routes.handle_koreaaitel_status_callback

    result = await handle(42, _request(STATUS_URL, params, signed))
    update = process.await_args.args[1]
    assert result == {"status": "success"} and update.status == Status.COMPLETED
    assert (update.call_id, update.duration) == ("CA9f8e", "12")

    process.reset_mock()
    for sent, signature in [
        ({**params, "CallStatus": "failed"}, signed),
        (params, None),
    ]:
        with pytest.raises(HTTPException) as excinfo:
            await handle(42, _request(STATUS_URL, sent, signature))
        assert excinfo.value.status_code == 401
    process.assert_not_awaited()


async def test_status_callback_ignores_another_providers_run(monkeypatch):
    process = AsyncMock()
    ari_like = SimpleNamespace(verify_inbound_signature=AsyncMock(return_value=True))
    _routes(status_routes, monkeypatch, ari_like, mode="ari", process=process)

    handle = status_routes.handle_koreaaitel_status_callback
    assert (await handle(42, _request(STATUS_URL, {}, None)))["status"] == "ignored"
    process.assert_not_awaited()


def test_api_key_is_optional_to_save_but_needed_to_dial():
    field = next(f for f in SPEC.ui_metadata.fields if f.name == "api_key")
    assert field.sensitive and not field.required
    assert SPEC.supports_outbound
    assert _config_loader({"api_key": "sk_test"})["api_key"] == "sk_test"

    def ready(numbers, **credentials):  # an inbound-only config has no key
        state = SimpleNamespace(active_phone_number_count=numbers)
        return SPEC.setup_checklist_resolver(credentials, state)

    keyless = ready(1)
    assert (
        not keyless.ready_for_outbound and "API Key" in keyless.outbound_blocked_reason
    )
    assert not ready(0, api_key="sk_test").ready_for_outbound  # still needs a number
    assert ready(1, api_key="sk_test").ready_for_outbound

"""Caller-ID selection for every outbound call and transfer.

An explicit number must be one of the configuration's active numbers;
otherwise the configuration's default caller ID is used. Nothing is ever picked
from the pool: a configuration without a default caller ID receives calls only,
so a call that would need one fails instead of presenting another number.
"""

import pytest

from api.services.telephony.providers.twilio.provider import TwilioProvider

POOL = ["+15555550101", "+15555550102", "+15555550103"]


def _provider(**overrides) -> TwilioProvider:
    config = {
        "account_sid": "AC123",
        "auth_token": "twilio-auth-token",
        "from_numbers": POOL,
        **overrides,
    }
    provider = TwilioProvider(config)
    provider.configuration_name = "Internal test lines"
    return provider


def test_explicit_active_number_wins_over_default():
    provider = _provider(default_from_number=POOL[1])
    assert provider.select_from_number(POOL[2]) == POOL[2]


def test_explicit_active_number_needs_no_default_caller_id():
    assert _provider().select_from_number(POOL[2]) == POOL[2]


def test_default_caller_id_used_when_no_explicit_number():
    provider = _provider(default_from_number=POOL[1])
    assert provider.select_from_number() == POOL[1]
    assert provider.select_from_number(None) == POOL[1]


def test_explicit_number_outside_the_configuration_is_refused():
    provider = _provider(default_from_number=POOL[1])
    with pytest.raises(ValueError) as excinfo:
        provider.select_from_number("+15555550199")
    assert str(excinfo.value) == (
        "Caller ID +15555550199 is not an active number of telephony "
        "configuration 'Internal test lines'."
    )


def test_no_default_caller_id_refuses_instead_of_picking_from_the_pool():
    provider = _provider()
    for _ in range(20):
        with pytest.raises(ValueError) as excinfo:
            provider.select_from_number()
    assert str(excinfo.value) == (
        "Telephony configuration 'Internal test lines' has no default caller "
        "ID; it receives calls only."
    )


def test_empty_pool_without_default_is_refused():
    with pytest.raises(ValueError, match="has no default caller ID"):
        _provider(from_numbers=[]).select_from_number()


def test_transfer_presents_the_calls_own_number_without_a_default_caller_id():
    assert _provider().select_transfer_from_number(POOL[2]) == POOL[2]


def test_transfer_without_a_usable_call_number_behaves_as_before():
    provider = _provider(default_from_number=POOL[1])
    assert provider.select_transfer_from_number() == POOL[1]
    assert provider.select_transfer_from_number("+15555550199") == POOL[1]
    with pytest.raises(ValueError, match="has no default caller ID"):
        _provider().select_transfer_from_number("+15555550199")


@pytest.mark.asyncio
async def test_initiate_call_without_default_never_reaches_twilio(monkeypatch):
    import api.services.telephony.providers.twilio.provider as provider_module

    def _no_session():
        raise AssertionError("Twilio must not be called")

    monkeypatch.setattr(provider_module.aiohttp, "ClientSession", _no_session)
    with pytest.raises(ValueError, match="has no default caller ID"):
        await _provider().initiate_call("+15555550150", "https://example.test")

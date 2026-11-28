"""Outbound pre-flight: which configuration dials, and whether it may dial.

Every outbound path (test call, API trigger, campaign) resolves its
configuration here. Without an explicit id only the organization's marked
default is used, and a configuration without an active default caller ID
receives calls only, unless the call pins one of its active numbers.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from api.services.campaign.campaign_call_dispatcher import CampaignCallDispatcher
from api.services.telephony.outbound_readiness import (
    OutboundConfigurationNotFoundError,
    OutboundSetupIncompleteError,
    ensure_outbound_setup_ready,
    resolve_outbound_configuration_id,
)
from api.tests.support.phone_numbers import phone_number as _number


def _row(config_id=21, name="Internal test lines", provider="twilio"):
    return SimpleNamespace(
        id=config_id, name=name, provider=provider, credentials={"account_sid": "AC1"}
    )


def _db(*, default_row=None, row=None, numbers=()):
    db = AsyncMock()
    db.get_default_telephony_configuration = AsyncMock(return_value=default_row)
    db.get_telephony_configuration_for_org = AsyncMock(return_value=row)
    db.list_phone_numbers_for_config = AsyncMock(return_value=list(numbers))
    db.list_trunks_for_config = AsyncMock(return_value=[])
    db.list_outbound_telephony_configuration_candidates = AsyncMock(
        side_effect=AssertionError("no fall-through to another configuration")
    )
    return db


@pytest.mark.asyncio
async def test_configuration_without_default_caller_id_receives_calls_only():
    db = _db(row=_row(), numbers=[_number(), _number()])

    with pytest.raises(OutboundSetupIncompleteError) as excinfo:
        await ensure_outbound_setup_ready(21, 7, db=db)

    assert str(excinfo.value) == (
        "'Internal test lines' is not ready for outbound calls. "
        "It has no default caller ID; it receives calls only."
    )


@pytest.mark.asyncio
async def test_default_flag_on_an_inactive_number_does_not_count():
    db = _db(row=_row(), numbers=[_number(), _number(default=True, active=False)])

    with pytest.raises(OutboundSetupIncompleteError, match="no default caller ID"):
        await ensure_outbound_setup_ready(21, 7, db=db)


@pytest.mark.asyncio
async def test_configuration_with_an_active_default_caller_id_is_ready():
    db = _db(row=_row(), numbers=[_number(), _number(default=True)])

    assert await ensure_outbound_setup_ready(21, 7, db=db) == 21


@pytest.mark.asyncio
async def test_without_an_id_only_the_marked_default_is_used():
    db = _db(default_row=_row(config_id=34), numbers=[_number(default=True)])

    assert await resolve_outbound_configuration_id(None, 7, db=db) == 34
    db.get_default_telephony_configuration.assert_awaited_once_with(7)
    db.list_outbound_telephony_configuration_candidates.assert_not_awaited()


@pytest.mark.asyncio
async def test_without_a_marked_default_the_call_is_refused():
    db = _db(default_row=None)

    with pytest.raises(OutboundConfigurationNotFoundError) as excinfo:
        await resolve_outbound_configuration_id(None, 7, db=db)

    assert "no default outbound telephony configuration" in str(excinfo.value)
    db.list_outbound_telephony_configuration_candidates.assert_not_awaited()


@pytest.mark.asyncio
async def test_an_unready_default_is_refused_rather_than_skipped():
    db = _db(
        default_row=_row(config_id=34, name="Customer AI line"), numbers=[_number()]
    )

    with pytest.raises(OutboundSetupIncompleteError) as excinfo:
        await resolve_outbound_configuration_id(None, 7, db=db)

    assert excinfo.value.telephony_configuration_id == 34
    assert "'Customer AI line'" in str(excinfo.value)
    assert "it receives calls only" in str(excinfo.value)
    db.list_outbound_telephony_configuration_candidates.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_call_pinned_to_an_active_number_needs_no_default_caller_id():
    db = _db(row=_row(), numbers=[_number(number_id=5), _number(number_id=6)])

    assert await ensure_outbound_setup_ready(21, 7, from_phone_number_id=6, db=db) == 21
    db.get_telephony_configuration_for_org.assert_awaited_once_with(
        21, 7, active_only=True
    )


@pytest.mark.asyncio
async def test_a_pinned_call_without_a_configuration_id_uses_only_the_marked_default():
    db = _db(default_row=_row(config_id=34), numbers=[_number(number_id=5)])

    assert (
        await resolve_outbound_configuration_id(None, 7, from_phone_number_id=5, db=db)
        == 34
    )
    db.get_default_telephony_configuration.assert_awaited_once_with(7)
    db.list_outbound_telephony_configuration_candidates.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "numbers, pinned_id",
    [
        pytest.param([_number(4), _number(5, active=False)], 5, id="inactive"),
        pytest.param([_number(4)], 99, id="other-configuration"),
        pytest.param([_number(4, default=True)], 99, id="other-with-a-default"),
    ],
)
async def test_a_pin_that_is_not_an_active_number_is_refused_as_a_bad_pin(
    numbers, pinned_id
):
    db = _db(row=_row(), numbers=numbers)

    with pytest.raises(OutboundSetupIncompleteError) as excinfo:
        await ensure_outbound_setup_ready(21, 7, from_phone_number_id=pinned_id, db=db)

    assert "The pinned caller ID is not an active number." in str(excinfo.value)
    assert "no default caller ID" not in str(excinfo.value)


@pytest.mark.asyncio
async def test_a_pinned_call_still_needs_the_provider_setup_to_be_complete():
    row = _row(provider="koreaaitel")
    db = _db(row=row, numbers=[_number(number_id=5)])

    with pytest.raises(OutboundSetupIncompleteError, match="API Key"):
        await ensure_outbound_setup_ready(21, 7, from_phone_number_id=5, db=db)

    row.credentials = {"api_key": "fixture-key"}
    assert await ensure_outbound_setup_ready(21, 7, from_phone_number_id=5, db=db) == 21


@pytest.mark.asyncio
async def test_a_campaign_never_dials_from_a_configuration_without_a_default():
    db = _db(row=_row(), numbers=[_number(number_id=5), _number(number_id=6)])
    campaign = SimpleNamespace(id=9, organization_id=7, telephony_configuration_id=21)

    with (
        patch("api.services.campaign.campaign_call_dispatcher.db_client", db),
        pytest.raises(OutboundSetupIncompleteError, match="receives calls only"),
    ):
        await CampaignCallDispatcher().get_provider_for_campaign(campaign)

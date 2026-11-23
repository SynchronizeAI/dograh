from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest

from api.db import db_client


@pytest.mark.asyncio
@pytest.mark.parametrize("row_count", [0, 1, 2])
async def test_inbound_account_lookup_logs_no_webhook_identifiers(
    monkeypatch, row_count
):
    account_id = "account-from-unverified-webhook"
    called_number = "+821012345678"
    config = SimpleNamespace(id=3, organization_id=11, name="Carrier")
    phone = SimpleNamespace(id=4, address=called_number, inbound_workflow_id=9)
    result = Mock()
    result.all.return_value = [(config, phone) for _ in range(row_count)]
    session = Mock()
    session.execute = AsyncMock(return_value=result)
    session_context = AsyncMock()
    session_context.__aenter__.return_value = session
    logger = Mock()
    monkeypatch.setattr("api.db.telephony_phone_number_client.logger", logger)

    with patch.object(db_client, "async_session", return_value=session_context):
        await db_client.find_inbound_route_by_account(
            provider="koreaaitel",
            account_id_field="account_id",
            account_id=account_id,
            to_number=called_number,
            country_hint="KR",
            organization_id=11,
        )

    assert account_id not in repr(logger.mock_calls)
    assert called_number not in repr(logger.mock_calls)
    if row_count > 1:
        logger.error.assert_called_once_with("Ambiguous inbound route")
    else:
        logger.error.assert_not_called()

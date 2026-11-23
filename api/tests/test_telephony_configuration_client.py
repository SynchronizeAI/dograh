from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from api.db import db_client


@pytest.mark.asyncio
@pytest.mark.parametrize("matching_rows", [[object()], [object(), object()], []])
async def test_credential_lookup_returns_only_a_unique_active_configuration(
    matching_rows,
):
    result = MagicMock()
    result.scalars.return_value.all.return_value = matching_rows
    session = MagicMock()
    session.execute = AsyncMock(return_value=result)
    session_context = AsyncMock()
    session_context.__aenter__.return_value = session

    with patch.object(db_client, "async_session", return_value=session_context):
        configuration = (
            await db_client.get_unique_active_telephony_configuration_by_credentials(
                "vonage", {"application_id": "app-123", "api_key": "key-456"}
            )
        )

    assert configuration is (matching_rows[0] if len(matching_rows) == 1 else None)
    session.execute.assert_awaited_once()

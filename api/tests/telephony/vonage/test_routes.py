from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from api.services.telephony.providers.vonage.provider import VonageProvider
from api.services.telephony.providers.vonage.routes import (
    handle_vonage_events_without_run,
)


@pytest.mark.asyncio
async def test_application_event_lookup_is_scoped_to_its_vonage_configuration():
    request = SimpleNamespace(headers={"authorization": "Bearer signed-token"})
    configuration = SimpleNamespace(organization_id=21)
    workflow_run = SimpleNamespace(id=303)

    with (
        patch.object(
            VonageProvider,
            "_decode_unverified_signed_claims",
            return_value={"application_id": "app-123", "api_key": "key-456"},
        ),
        patch(
            "api.services.telephony.providers.vonage.routes._read_json_body",
            new_callable=AsyncMock,
            return_value=({"uuid": "call-789", "status": "completed"}, "body"),
        ),
        patch(
            "api.services.telephony.providers.vonage.routes.db_client"
        ) as db_client,
        patch(
            "api.services.telephony.providers.vonage.routes."
            "_handle_vonage_event_request",
            new_callable=AsyncMock,
            return_value={"status": "ok"},
        ) as handle_event,
    ):
        db_client.get_unique_active_telephony_configuration_by_credentials = AsyncMock(
            return_value=configuration
        )
        db_client.get_workflow_run_by_call_id = AsyncMock(return_value=workflow_run)

        result = await handle_vonage_events_without_run(request)

    assert result == {"status": "ok"}
    credential_lookup = (
        db_client.get_unique_active_telephony_configuration_by_credentials
    )
    credential_lookup.assert_awaited_once_with(
        "vonage", {"application_id": "app-123", "api_key": "key-456"}
    )
    db_client.get_workflow_run_by_call_id.assert_awaited_once_with(
        "call-789", organization_id=21, mode="vonage"
    )
    handle_event.assert_awaited_once_with(request, workflow_run.id)

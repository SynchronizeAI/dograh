"""A transfer leg presents the number its own call used.

A call pinned to a number of a configuration with no default caller ID must
still be able to transfer, and must present that same number to the destination.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from api.services.telephony.providers.twilio import provider as twilio_module
from api.services.telephony.providers.twilio.provider import TwilioProvider
from api.services.telephony.transfer_event_protocol import (
    TransferEvent,
    TransferEventType,
)
from api.services.workflow import pipecat_engine_custom_tools as tools
from api.services.workflow.tools.transfer_resolver import ResolvedTransferConfig
from api.tests.test_transfer_message_playback import RecordingEngine, TransferToolModel

OWN, OTHER, CUSTOMER = "+15550100001", "+15550100002", "+15550100099"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "context, presented",
    [
        pytest.param(
            {"direction": "outbound", "caller_number": OWN, "called_number": CUSTOMER},
            OWN,
            id="outbound",
        ),
        pytest.param(
            {"direction": "inbound", "caller_number": CUSTOMER, "called_number": OWN},
            OWN,
            id="inbound",
        ),
        pytest.param({}, None, id="no-number-keeps-the-default-rule"),
    ],
)
async def test_transfer_without_a_default_caller_id_presents_the_calls_own_number(
    monkeypatch, context, presented
):
    provider = TwilioProvider(
        {"account_sid": "AC1", "auth_token": "token", "from_numbers": [OWN, OTHER]}
    )
    dialled = []
    reply = MagicMock(status=201)
    reply.__aenter__ = AsyncMock(return_value=reply)
    reply.__aexit__ = AsyncMock(return_value=False)
    reply.text = AsyncMock(return_value="{}")
    reply.json = AsyncMock(return_value={"sid": "CA-destination"})
    session = MagicMock()
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=False)
    session.post.side_effect = lambda url, data, auth: dialled.append(data) or reply
    monkeypatch.setattr(twilio_module.aiohttp, "ClientSession", lambda: session)
    monkeypatch.setattr(
        twilio_module,
        "get_backend_endpoints",
        AsyncMock(return_value=("https://api.example.com", "")),
    )
    answered = TransferEvent(
        type=TransferEventType.DESTINATION_ANSWERED,
        transfer_id="tx-1",
        original_call_sid="caller",
    )
    transfer_manager = SimpleNamespace(
        store_transfer_context=AsyncMock(),
        remove_transfer_context=AsyncMock(),
        publish_transfer_event=AsyncMock(return_value=answered),
        wait_for_transfer_completion=AsyncMock(return_value=answered),
    )
    run = SimpleNamespace(
        mode="twilio", initial_context=context, gathered_context={"call_id": "caller"}
    )

    async def hold(stop_event, **kwargs):
        await stop_event.wait()

    monkeypatch.setattr(
        tools.db_client, "get_workflow_run_by_id", AsyncMock(return_value=run)
    )
    monkeypatch.setattr(
        tools, "get_telephony_provider_for_run", AsyncMock(return_value=provider)
    )
    monkeypatch.setattr(
        tools, "get_call_transfer_manager", AsyncMock(return_value=transfer_manager)
    )
    monkeypatch.setattr(
        tools,
        "resolve_transfer_config",
        AsyncMock(
            return_value=ResolvedTransferConfig(
                destination="+15550100123", timeout_seconds=30, source="static"
            )
        ),
    )
    monkeypatch.setattr(tools, "play_audio_loop", hold)
    callback = AsyncMock()

    await asyncio.wait_for(
        tools.CustomToolManager(RecordingEngine())._create_transfer_call_handler(
            TransferToolModel(), "transfer_call"
        )(SimpleNamespace(arguments={}, result_callback=callback)),
        timeout=2,
    )

    if presented:
        assert [data["From"] for data in dialled] == [presented]
        assert callback.call_args.args[0]["status"] == "transfer_success"
    else:
        assert dialled == []
        assert callback.call_args.args[0]["reason"] == "provider_error"

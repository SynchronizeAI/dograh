"""Only the first media stream may claim and start a workflow run."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from api.services.pipecat import run_pipeline as run_pipeline_module
from api.services.telephony.providers.koreaaitel.provider import KoreaAITelProvider


class _ReachedSetup(Exception):
    pass


@pytest.mark.asyncio
@pytest.mark.parametrize("claimed", [True, False])
async def test_stream_setup_requires_running_claim(monkeypatch, claimed):
    workflow = SimpleNamespace(
        organization_id=7, workflow_configurations=None, user_id=99
    )
    workflow_run = SimpleNamespace(
        workflow_id=1,
        initial_context={},
        definition=SimpleNamespace(workflow_configurations={}),
    )
    websocket = AsyncMock()
    mark_running = AsyncMock(return_value=claimed)
    monkeypatch.setattr(
        run_pipeline_module.db_client, "get_workflow", AsyncMock(return_value=workflow)
    )
    monkeypatch.setattr(
        run_pipeline_module.db_client,
        "get_workflow_run",
        AsyncMock(return_value=workflow_run),
    )
    monkeypatch.setattr(
        run_pipeline_module.db_client, "mark_workflow_run_running", mark_running
    )
    monkeypatch.setattr(
        "api.services.configuration.run_model_configuration."
        "get_effective_ai_model_configuration_for_run",
        AsyncMock(side_effect=_ReachedSetup),
    )
    unregister_call = AsyncMock()
    monkeypatch.setattr(
        run_pipeline_module.call_concurrency,
        "unregister_active_call",
        unregister_call,
    )

    call = run_pipeline_module.run_pipeline_telephony(
        websocket=websocket,
        provider_name="telnyx",
        workflow_id=1,
        workflow_run_id=44,
        organization_id=7,
        call_id="call-1",
        transport_kwargs={},
    )
    if claimed:
        with pytest.raises(_ReachedSetup):
            await call
        websocket.close.assert_not_awaited()
    else:
        await call
        websocket.close.assert_awaited_once_with(
            code=4409, reason="Workflow run already claimed or completed"
        )
        unregister_call.assert_not_awaited()

    mark_running.assert_awaited_once_with(44, 7)


@pytest.mark.asyncio
async def test_two_streams_for_one_run_start_only_one_pipeline(monkeypatch):
    workflow = SimpleNamespace(
        organization_id=7, workflow_configurations=None, user_id=99
    )
    workflow_run = SimpleNamespace(
        workflow_id=1,
        initial_context={},
        definition=SimpleNamespace(workflow_configurations={}),
    )
    state = {"value": "initialized"}
    claim_lock = asyncio.Lock()

    async def claim_run(_run_id, _organization_id):
        await asyncio.sleep(0)
        async with claim_lock:
            if state["value"] != "initialized":
                return False
            state["value"] = "running"
            return True

    mark_running = AsyncMock(side_effect=claim_run)
    setup_pipeline = AsyncMock(
        return_value=SimpleNamespace(is_realtime=False, realtime=None)
    )
    pipeline_start = AsyncMock(side_effect=_ReachedSetup)
    transport_factory = AsyncMock(return_value=object())
    sockets = [AsyncMock(), AsyncMock()]
    provider = KoreaAITelProvider({})
    for index, websocket in enumerate(sockets):
        websocket.receive_text.side_effect = [
            json.dumps({"event": "connected"}),
            json.dumps(
                {
                    "event": "start",
                    "start": {
                        "streamId": f"stream-{index}",
                        "callId": f"call-{index}",
                    },
                }
            ),
        ]
    monkeypatch.setattr(
        run_pipeline_module.db_client, "get_workflow", AsyncMock(return_value=workflow)
    )
    monkeypatch.setattr(
        run_pipeline_module.db_client,
        "get_workflow_run",
        AsyncMock(return_value=workflow_run),
    )
    monkeypatch.setattr(
        run_pipeline_module.db_client, "mark_workflow_run_running", mark_running
    )
    monkeypatch.setattr(
        "api.services.configuration.run_model_configuration."
        "get_effective_ai_model_configuration_for_run",
        setup_pipeline,
    )
    monkeypatch.setattr(
        run_pipeline_module.telephony_registry,
        "get",
        Mock(return_value=SimpleNamespace(transport_factory=transport_factory)),
    )
    monkeypatch.setattr(run_pipeline_module, "create_audio_config", Mock())
    monkeypatch.setattr(run_pipeline_module, "_run_pipeline_impl", pipeline_start)
    unregister_call = AsyncMock()
    monkeypatch.setattr(
        run_pipeline_module.call_concurrency,
        "unregister_active_call",
        unregister_call,
    )
    register_active = Mock()
    unregister_active = Mock()
    monkeypatch.setattr(
        run_pipeline_module, "register_worker_active_call", register_active
    )
    monkeypatch.setattr(
        run_pipeline_module, "unregister_worker_active_call", unregister_active
    )

    results = await asyncio.gather(
        *(
            provider.handle_websocket(
                websocket=websocket,
                workflow_id=1,
                organization_id=7,
                workflow_run_id=44,
            )
            for websocket in sockets
        ),
        return_exceptions=True,
    )

    assert sum(isinstance(result, _ReachedSetup) for result in results) == 1
    assert sum(result is None for result in results) == 1
    assert mark_running.await_count == 2
    setup_pipeline.assert_awaited_once()
    transport_factory.assert_awaited_once()
    pipeline_start.assert_awaited_once()
    assert sum(websocket.close.await_count for websocket in sockets) == 1
    assert state["value"] == "running"
    register_active.assert_called_once_with(44)
    unregister_active.assert_called_once_with(44)
    unregister_call.assert_awaited_once_with(44)

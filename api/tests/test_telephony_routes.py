from types import SimpleNamespace
from unittest.mock import ANY, AsyncMock, Mock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from loguru import logger as loguru_logger

from api.enums import WorkflowRunMode, WorkflowRunState
from api.errors.telephony_errors import TelephonyError
from api.routes.telephony import (
    _handle_telephony_websocket,
    handle_inbound_run,
    handle_inbound_telephony,
    router,
)
from api.services.auth.depends import get_user
from api.services.call_concurrency import CallConcurrencyLimitError
from api.services.telephony.providers.koreaaitel.provider import KoreaAITelProvider


def _make_test_app() -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_user] = lambda: SimpleNamespace(
        id=7,
        selected_organization_id=11,
    )
    return app


def _workflow(*, workflow_id: int = 33, user_id: int = 99):
    return SimpleNamespace(
        id=workflow_id,
        user_id=user_id,
        organization_id=11,
        template_context_variables={"template_key": "template-value"},
        released_definition=SimpleNamespace(
            id=77,
            template_context_variables={"template_key": "template-value"},
        ),
        current_definition=None,
    )


def _stub_outbound_setup_lookup(mock_db, *, provider: str = "twilio"):
    """Satisfy the pre-flight setup-checklist read on ``/initiate-call``.

    A configuration with an active number passes the caller-ID requirement, so
    the guard is a no-op — these tests are about everything downstream of it.
    """
    row = SimpleNamespace(id=55, name="Test config", provider=provider, credentials={})
    mock_db.get_telephony_configuration_for_org = AsyncMock(return_value=row)
    mock_db.list_outbound_telephony_configuration_candidates = AsyncMock(
        return_value=[row]
    )
    mock_db.list_phone_numbers_for_config = AsyncMock(
        return_value=[
            SimpleNamespace(
                is_active=True, inbound_workflow_id=None, telephony_trunk_id=None
            )
        ]
    )
    # Twilio has no trunks; the guard skips the query, but a stub keeps the
    # helper usable for SIP providers too.
    mock_db.list_trunks_for_config = AsyncMock(return_value=[])


def _provider():
    return SimpleNamespace(
        PROVIDER_NAME="twilio",
        WEBHOOK_ENDPOINT="twilio/voice",
        validate_config=Mock(return_value=True),
        initiate_call=AsyncMock(
            return_value=SimpleNamespace(
                caller_number="+15550001111",
                provider_metadata={"call_id": "call-123"},
            )
        ),
    )


def test_initiate_call_rejects_incomplete_setup_before_run_creation():
    app = _make_test_app()
    client = TestClient(app)
    provider = _provider()

    with (
        patch("api.routes.telephony.db_client") as mock_db,
        patch(
            "api.routes.telephony.get_telephony_provider_by_id",
            new=AsyncMock(return_value=provider),
        ),
        patch(
            "api.services.organization_preferences.get_organization_preferences",
            new=AsyncMock(return_value=SimpleNamespace(test_phone_number=None)),
        ),
    ):
        mock_db.get_default_telephony_configuration = AsyncMock(
            return_value=SimpleNamespace(id=55)
        )
        mock_db.get_telephony_configuration_for_org = AsyncMock(
            return_value=SimpleNamespace(
                id=55,
                name="Twilio production",
                provider="twilio",
                credentials={},
            )
        )
        mock_db.list_outbound_telephony_configuration_candidates = AsyncMock(
            return_value=[
                SimpleNamespace(
                    id=55,
                    name="Twilio production",
                    provider="twilio",
                    credentials={},
                )
            ]
        )
        mock_db.list_phone_numbers_for_config = AsyncMock(return_value=[])
        mock_db.create_workflow_run = AsyncMock()

        response = client.post(
            "/telephony/initiate-call",
            json={"workflow_id": 33, "phone_number": "+15551234567"},
        )

    assert response.status_code == 400
    assert "Twilio production" in response.json()["detail"]
    mock_db.create_workflow_run.assert_not_awaited()
    provider.initiate_call.assert_not_awaited()


def test_initiate_call_executes_as_workflow_owner_for_shared_org_workflow():
    app = _make_test_app()
    client = TestClient(app)

    workflow = _workflow()
    provider = _provider()
    quota_mock = AsyncMock(
        return_value=SimpleNamespace(has_quota=True, error_message="")
    )

    with (
        patch("api.routes.telephony.db_client") as mock_db,
        patch("api.routes.telephony.call_concurrency") as mock_concurrency,
        patch(
            "api.routes.telephony.authorize_workflow_run_start",
            new=quota_mock,
        ),
        patch(
            "api.routes.telephony.get_telephony_provider_by_id",
            new=AsyncMock(return_value=provider),
        ),
        patch(
            "api.routes.telephony.get_backend_endpoints",
            new=AsyncMock(return_value=("https://api.example.com", "wss://ignored")),
        ),
    ):
        slot = object()
        mock_concurrency.acquire_org_slot = AsyncMock(return_value=slot)
        mock_concurrency.bind_workflow_run = AsyncMock()
        mock_concurrency.release_slot = AsyncMock()
        mock_concurrency.release_workflow_run_slot = AsyncMock()

        mock_db.get_user_configurations = AsyncMock(
            return_value=SimpleNamespace(test_phone_number=None)
        )
        mock_db.get_default_telephony_configuration = AsyncMock(
            return_value=SimpleNamespace(id=55)
        )
        _stub_outbound_setup_lookup(mock_db)
        mock_db.get_workflow = AsyncMock(return_value=workflow)
        mock_db.get_draft_version = AsyncMock(return_value=None)
        mock_db.create_workflow_run = AsyncMock(
            return_value=SimpleNamespace(
                id=501,
                name="WR-TEL-OUT-00000001",
                initial_context={"template_key": "template-value"},
            )
        )
        mock_db.update_workflow_run = AsyncMock()

        response = client.post(
            "/telephony/initiate-call",
            json={"workflow_id": workflow.id, "phone_number": "+15551234567"},
        )

    assert response.status_code == 200
    quota_mock.assert_awaited_once_with(
        workflow_id=workflow.id,
        organization_id=workflow.organization_id,
        workflow_run_id=501,
        actor_user=ANY,
    )
    mock_db.get_workflow.assert_awaited_once_with(workflow.id, organization_id=11)

    create_call = mock_db.create_workflow_run.await_args
    create_args = create_call.args
    create_kwargs = create_call.kwargs
    assert create_args[1] == workflow.id
    assert create_kwargs["user_id"] == workflow.user_id
    assert create_kwargs["organization_id"] == workflow.organization_id
    assert create_kwargs["definition_id"] == 77
    assert create_kwargs["initial_context"]["template_key"] == "template-value"
    assert create_kwargs["initial_context"]["telephony_configuration_id"] == 55
    mock_concurrency.acquire_org_slot.assert_awaited_once_with(
        workflow.organization_id,
        source="telephony_outbound",
        timeout=0,
    )
    mock_concurrency.bind_workflow_run.assert_awaited_once_with(slot, 501)

    initiate_kwargs = provider.initiate_call.await_args.kwargs
    assert initiate_kwargs["workflow_id"] == workflow.id
    # The media websocket URL is keyed on the org, not the workflow owner.
    assert initiate_kwargs["organization_id"] == workflow.organization_id
    webhook_url = initiate_kwargs["webhook_url"]
    assert f"organization_id={workflow.organization_id}" in webhook_url
    # The answer URL carries no workflow owner: nothing downstream scopes on it.
    assert "user_id=" not in webhook_url
    mock_db.get_user_configurations.assert_not_called()


def test_initiate_call_uses_draft_template_context_with_provider_overrides():
    app = _make_test_app()
    client = TestClient(app)

    workflow = _workflow()
    draft = SimpleNamespace(
        id=88,
        template_context_variables={
            "phone_number": "template-phone",
            "called_number": "template-called",
            "provider": "template-provider",
            "draft_only": "kept",
        },
    )
    provider = _provider()
    quota_mock = AsyncMock(
        return_value=SimpleNamespace(has_quota=True, error_message="")
    )

    with (
        patch("api.routes.telephony.db_client") as mock_db,
        patch("api.routes.telephony.call_concurrency") as mock_concurrency,
        patch(
            "api.routes.telephony.authorize_workflow_run_start",
            new=quota_mock,
        ),
        patch(
            "api.routes.telephony.get_telephony_provider_by_id",
            new=AsyncMock(return_value=provider),
        ),
        patch(
            "api.routes.telephony.get_backend_endpoints",
            new=AsyncMock(return_value=("https://api.example.com", "wss://ignored")),
        ),
    ):
        slot = object()
        mock_concurrency.acquire_org_slot = AsyncMock(return_value=slot)
        mock_concurrency.bind_workflow_run = AsyncMock()
        mock_concurrency.release_slot = AsyncMock()
        mock_concurrency.release_workflow_run_slot = AsyncMock()

        mock_db.get_user_configurations = AsyncMock(
            return_value=SimpleNamespace(test_phone_number=None)
        )
        mock_db.get_default_telephony_configuration = AsyncMock(
            return_value=SimpleNamespace(id=55)
        )
        _stub_outbound_setup_lookup(mock_db)
        mock_db.get_workflow = AsyncMock(return_value=workflow)
        mock_db.get_draft_version = AsyncMock(return_value=draft)
        mock_db.create_workflow_run = AsyncMock(
            return_value=SimpleNamespace(
                id=501,
                name="WR-TEL-OUT-00000001",
                initial_context={},
            )
        )
        mock_db.update_workflow_run = AsyncMock()

        response = client.post(
            "/telephony/initiate-call",
            json={"workflow_id": workflow.id, "phone_number": "+15551234567"},
        )

    assert response.status_code == 200
    mock_db.get_draft_version.assert_awaited_once_with(workflow.id)
    create_kwargs = mock_db.create_workflow_run.await_args.kwargs
    assert create_kwargs["definition_id"] == draft.id
    assert create_kwargs["initial_context"]["phone_number"] == "+15551234567"
    assert create_kwargs["initial_context"]["called_number"] == "+15551234567"
    assert create_kwargs["initial_context"]["provider"] == provider.PROVIDER_NAME
    assert create_kwargs["initial_context"]["draft_only"] == "kept"


def test_initiate_call_uses_organization_preference_phone_number():
    app = _make_test_app()
    client = TestClient(app)

    workflow = _workflow()
    provider = _provider()
    quota_mock = AsyncMock(
        return_value=SimpleNamespace(has_quota=True, error_message="")
    )

    with (
        patch("api.routes.telephony.db_client") as mock_db,
        patch("api.routes.telephony.call_concurrency") as mock_concurrency,
        patch(
            "api.routes.telephony.authorize_workflow_run_start",
            new=quota_mock,
        ),
        patch(
            "api.routes.telephony.get_telephony_provider_by_id",
            new=AsyncMock(return_value=provider),
        ),
        patch(
            "api.routes.telephony.get_backend_endpoints",
            new=AsyncMock(return_value=("https://api.example.com", "wss://ignored")),
        ),
    ):
        mock_concurrency.acquire_org_slot = AsyncMock(return_value=object())
        mock_concurrency.bind_workflow_run = AsyncMock()
        mock_concurrency.release_slot = AsyncMock()
        mock_concurrency.release_workflow_run_slot = AsyncMock()

        mock_db.get_user_configurations = AsyncMock(
            return_value=SimpleNamespace(test_phone_number="+15550000000")
        )
        mock_db.get_configuration = Mock(
            return_value=SimpleNamespace(value={"test_phone_number": "+15557654321"})
        )
        mock_db.get_default_telephony_configuration = AsyncMock(
            return_value=SimpleNamespace(id=55)
        )
        _stub_outbound_setup_lookup(mock_db)
        mock_db.get_workflow = AsyncMock(return_value=workflow)
        mock_db.get_draft_version = AsyncMock(return_value=None)
        mock_db.create_workflow_run = AsyncMock(
            return_value=SimpleNamespace(
                id=501,
                name="WR-TEL-OUT-00000001",
                initial_context={},
            )
        )
        mock_db.update_workflow_run = AsyncMock()

        response = client.post(
            "/telephony/initiate-call",
            json={"workflow_id": workflow.id},
        )

    assert response.status_code == 200
    assert provider.initiate_call.await_args.kwargs["to_number"] == "+15557654321"
    mock_db.get_user_configurations.assert_not_called()


def test_initiate_call_rejects_existing_run_for_different_workflow():
    app = _make_test_app()
    client = TestClient(app)

    workflow = _workflow()
    provider = _provider()
    quota_mock = AsyncMock(
        return_value=SimpleNamespace(has_quota=True, error_message="")
    )

    with (
        patch("api.routes.telephony.db_client") as mock_db,
        patch("api.routes.telephony.call_concurrency") as mock_concurrency,
        patch(
            "api.routes.telephony.authorize_workflow_run_start",
            new=quota_mock,
        ),
        patch(
            "api.routes.telephony.get_telephony_provider_by_id",
            new=AsyncMock(return_value=provider),
        ),
    ):
        mock_concurrency.acquire_org_slot = AsyncMock(return_value=object())
        mock_concurrency.bind_workflow_run = AsyncMock()
        mock_concurrency.release_slot = AsyncMock()
        mock_concurrency.release_workflow_run_slot = AsyncMock()

        mock_db.get_user_configurations = AsyncMock(
            return_value=SimpleNamespace(test_phone_number=None)
        )
        mock_db.get_default_telephony_configuration = AsyncMock(
            return_value=SimpleNamespace(id=55)
        )
        _stub_outbound_setup_lookup(mock_db)
        mock_db.get_workflow = AsyncMock(return_value=workflow)
        mock_db.get_workflow_run = AsyncMock(
            return_value=SimpleNamespace(
                id=501,
                workflow_id=44,
                name="WR-TEL-OUT-00000044",
                initial_context={},
            )
        )

        response = client.post(
            "/telephony/initiate-call",
            json={
                "workflow_id": workflow.id,
                "workflow_run_id": 501,
                "phone_number": "+15551234567",
            },
        )

    assert response.status_code == 400
    assert response.json()["detail"] == "workflow_run_workflow_mismatch"
    mock_db.get_workflow_run.assert_awaited_once_with(501, organization_id=11)
    mock_concurrency.release_slot.assert_awaited_once()
    assert not mock_db.create_workflow_run.called
    assert provider.initiate_call.await_count == 0


def test_initiate_call_rejects_when_concurrency_limit_reached():
    app = _make_test_app()
    client = TestClient(app)

    workflow = _workflow()
    provider = _provider()

    with (
        patch("api.routes.telephony.db_client") as mock_db,
        patch("api.routes.telephony.call_concurrency") as mock_concurrency,
        patch(
            "api.routes.telephony.get_telephony_provider_by_id",
            new=AsyncMock(return_value=provider),
        ),
    ):
        mock_concurrency.acquire_org_slot = AsyncMock(
            side_effect=CallConcurrencyLimitError(
                organization_id=workflow.organization_id,
                source="telephony_outbound",
                wait_time=0,
                max_concurrent=1,
            )
        )
        mock_db.get_default_telephony_configuration = AsyncMock(
            return_value=SimpleNamespace(id=55)
        )
        _stub_outbound_setup_lookup(mock_db)
        mock_db.get_workflow = AsyncMock(return_value=workflow)
        mock_db.create_workflow_run = AsyncMock()

        response = client.post(
            "/telephony/initiate-call",
            json={"workflow_id": workflow.id, "phone_number": "+15551234567"},
        )

    assert response.status_code == 429
    assert response.json()["detail"] == "Concurrent call limit reached"
    mock_db.create_workflow_run.assert_not_called()
    provider.initiate_call.assert_not_awaited()


@pytest.mark.asyncio
async def test_inbound_run_routes_exotel_without_account_id_by_called_number():
    """Exotel Voicebot webhooks omit AccountSid; route by called number."""
    request = SimpleNamespace(headers={}, url="https://api.example.com/inbound/run")
    provider_class = SimpleNamespace(
        PROVIDER_NAME="exotel",
        generate_validation_error_response=Mock(return_value="not-configured"),
    )
    normalized_data = SimpleNamespace(
        provider="exotel",
        direction="inbound",
        to_number="07314852338",
        from_number="09441233609",
        to_country="IN",
        from_country="IN",
        account_id=None,
        call_id="call-1",
        raw_data={},
    )
    config = SimpleNamespace(
        id=55,
        organization_id=11,
        name="exotel-config",
        credentials={"account_sid": "sid123"},
    )
    phone_row = SimpleNamespace(id=77, inbound_workflow_id=33, address="07314852338")
    workflow = SimpleNamespace(id=33, user_id=99)
    provider_instance = SimpleNamespace(
        verify_inbound_signature=AsyncMock(return_value=True),
        start_inbound_stream=AsyncMock(return_value='{"url":"wss://x"}'),
    )

    with (
        patch(
            "api.routes.telephony.parse_webhook_request",
            new=AsyncMock(return_value=({}, "raw-body")),
        ),
        patch(
            "api.routes.telephony._detect_provider",
            new=AsyncMock(return_value=provider_class),
        ),
        patch(
            "api.routes.telephony.normalize_webhook_data",
            return_value=normalized_data,
        ),
        patch("api.routes.telephony.db_client") as mock_db,
        patch(
            "api.routes.telephony.get_telephony_provider_by_id",
            new=AsyncMock(return_value=provider_instance),
        ),
        patch(
            "api.routes.telephony.get_backend_endpoints",
            new=AsyncMock(
                return_value=("https://api.example.test", "wss://api.example.test")
            ),
        ),
        patch(
            "api.routes.telephony.authorize_workflow_run_start",
            new=AsyncMock(),
        ),
        patch(
            "api.routes.telephony.prepare_workflow_run_inputs",
            new=AsyncMock(return_value={}),
        ),
        patch("api.routes.telephony.call_concurrency") as mock_concurrency,
        patch(
            "api.routes.telephony._create_inbound_workflow_run",
            new=AsyncMock(return_value=(999, None)),
        ),
    ):
        mock_db.find_inbound_route_by_account = AsyncMock()
        mock_db.find_inbound_route_by_called_number = AsyncMock(
            return_value=(config, phone_row)
        )
        mock_db.get_workflow = AsyncMock(return_value=workflow)
        mock_concurrency.acquire_org_slot = AsyncMock(return_value=object())
        mock_concurrency.bind_workflow_run = AsyncMock()
        mock_concurrency.release_workflow_run_slot = AsyncMock()
        mock_concurrency.release_slot = AsyncMock()

        response = await handle_inbound_run(request)

    mock_db.find_inbound_route_by_account.assert_not_awaited()
    mock_db.find_inbound_route_by_called_number.assert_awaited_once_with(
        provider="exotel",
        to_number="07314852338",
        country_hint="IN",
    )
    provider_instance.start_inbound_stream.assert_awaited_once()
    assert response == '{"url":"wss://x"}'


def _setup_koreaaitel_webhook(
    monkeypatch, existing_runs, new_run_ids, *, signature_valid=True
):
    import api.routes.telephony as routes

    request = SimpleNamespace(
        headers={"x-signature": "signed"}, url="https://api.example.com/inbound/run"
    )
    provider_class = SimpleNamespace(PROVIDER_NAME="koreaaitel")
    data = KoreaAITelProvider.parse_inbound_webhook(
        {
            "CallId": "CA1",
            "AccountId": "AC1",
            "From": "01012345678",
            "To": "07012340001",
            "Direction": "inbound",
        }
    )
    config = SimpleNamespace(
        id=55, organization_id=11, name="KoreaAITel", credentials={"account_id": "AC1"}
    )
    phone = SimpleNamespace(id=77, inbound_workflow_id=33, address="07012340001")
    provider = SimpleNamespace(
        verify_inbound_signature=AsyncMock(return_value=signature_valid),
        start_inbound_stream=AsyncMock(return_value="stream"),
        generate_validation_error_response=Mock(return_value="invalid-signature"),
    )
    db = Mock(
        find_inbound_route_by_account=AsyncMock(return_value=(config, phone)),
        get_workflow=AsyncMock(return_value=SimpleNamespace(id=33, user_id=99)),
        get_workflow_run_by_call_id=AsyncMock(side_effect=existing_runs),
    )
    concurrency = SimpleNamespace(
        acquire_org_slot=AsyncMock(return_value=object()), bind_workflow_run=AsyncMock()
    )
    for name, value in {
        "parse_webhook_request": AsyncMock(return_value=({}, "raw")),
        "_detect_provider": AsyncMock(return_value=provider_class),
        "normalize_webhook_data": Mock(return_value=data),
        "db_client": db,
        "get_telephony_provider_by_id": AsyncMock(return_value=provider),
        "get_backend_endpoints": AsyncMock(return_value=("https://api", "wss://api")),
        "call_concurrency": concurrency,
        "authorize_workflow_run_start": AsyncMock(),
        "_create_inbound_workflow_run": AsyncMock(
            side_effect=[(run_id, None) for run_id in new_run_ids]
        ),
    }.items():
        monkeypatch.setattr(routes, name, value)
    return request, data, provider, db


@pytest.mark.asyncio
@pytest.mark.parametrize("replay", [True, False])
async def test_koreaaitel_webhook_replay_and_call_id_isolation(monkeypatch, replay):
    run = SimpleNamespace(id=999, workflow_id=33)
    found, new_ids = ([None, run], [999]) if replay else ([None, None], [999, 1000])
    request, data, provider, db = _setup_koreaaitel_webhook(monkeypatch, found, new_ids)
    data.call_id = "CA1"
    assert await handle_inbound_run(request) == "stream"
    data.call_id = "CA1" if replay else "CA2"
    assert await handle_inbound_run(request) == "stream"
    assert [
        call.args[0] for call in db.get_workflow_run_by_call_id.await_args_list
    ] == ["CA1", data.call_id]
    assert [
        call.kwargs["organization_id"]
        for call in db.get_workflow_run_by_call_id.await_args_list
    ] == [11, 11]
    assert [
        call.kwargs["telephony_configuration_id"]
        for call in db.get_workflow_run_by_call_id.await_args_list
    ] == [55, 55]
    assert [
        call.kwargs["workflow_run_id"]
        for call in provider.start_inbound_stream.await_args_list
    ] == ([999, 999] if replay else [999, 1000])


@pytest.mark.asyncio
async def test_unsigned_koreaaitel_webhook_does_not_log_caller_number(monkeypatch):
    request, _, _, _ = _setup_koreaaitel_webhook(
        monkeypatch, [], [], signature_valid=False
    )
    log_records = []
    sink_id = loguru_logger.add(
        lambda message: log_records.append(message.record["message"])
    )
    try:
        assert await handle_inbound_run(request) == "invalid-signature"
    finally:
        loguru_logger.remove(sink_id)
    assert "01012345678" not in "\n".join(log_records)


@pytest.mark.asyncio
async def test_koreaaitel_run_log_redacts_caller_but_context_keeps_it(monkeypatch):
    import api.routes.telephony as routes

    workflow = SimpleNamespace(id=33)
    run_inputs = SimpleNamespace(definition_id=77, definition=None, use_draft=False)
    create_run = AsyncMock(return_value=SimpleNamespace(id=42))
    monkeypatch.setattr(
        routes.db_client, "get_workflow", AsyncMock(return_value=workflow)
    )
    monkeypatch.setattr(
        routes, "prepare_workflow_run_inputs", AsyncMock(return_value=run_inputs)
    )
    monkeypatch.setattr(routes.db_client, "create_workflow_run", create_run)
    normalized = KoreaAITelProvider.parse_inbound_webhook(
        {
            "CallId": "CA1",
            "From": "01012345678",
            "To": "07012340001",
            "ForwardedFrom": "0212345678",
            "Direction": "inbound",
            "CallStatus": "ringing",
        }
    )

    await routes._create_inbound_workflow_run(
        workflow_id=33,
        user_id=99,
        organization_id=11,
        provider="koreaaitel",
        normalized_data=normalized,
        telephony_configuration_id=55,
    )

    stored = create_run.await_args.kwargs
    assert stored["initial_context"]["caller_number"] == "01012345678"
    assert stored["initial_context"]["telephony_configuration_id"] == 55
    raw_webhook_data = stored["logs"]["inbound_webhook"]["raw_webhook_data"]
    assert not {"From", "ForwardedFrom", "To"} & raw_webhook_data.keys()
    assert raw_webhook_data["CallId"] == "CA1"
    assert raw_webhook_data["CallStatus"] == "ringing"


@pytest.mark.asyncio
async def test_legacy_koreaaitel_reject_does_not_log_unverified_identifiers(monkeypatch):
    import api.routes.telephony as routes

    identifiers = [
        "CALL-UNVERIFIED-123",
        "ACCOUNT-UNVERIFIED-456",
        "01012345678",
        "+821012345678",
        "0212345678",
        "07012340001",
        "HEADER-UNVERIFIED-789",
    ]
    webhook_data = {
        "CallId": identifiers[0],
        "AccountId": identifiers[1],
        "From": identifiers[2],
        "ForwardedFrom": identifiers[4],
        "To": identifiers[5],
        "Direction": "inbound",
    }
    normalized = KoreaAITelProvider.parse_inbound_webhook(webhook_data)
    provider_class = SimpleNamespace(
        PROVIDER_NAME="koreaaitel",
        generate_validation_error_response=Mock(return_value="invalid-signature"),
    )
    provider = SimpleNamespace(verify_inbound_signature=AsyncMock(return_value=False))
    workflow = SimpleNamespace(organization_id=11, user_id=99)
    phone_row = SimpleNamespace(id=77, telephony_configuration_id=55)
    db = Mock(
        get_workflow_by_id=AsyncMock(return_value=workflow),
        find_inbound_route_by_account=AsyncMock(return_value=None),
        list_telephony_configurations_by_provider=AsyncMock(return_value=[object()]),
        find_active_phone_number_for_inbound=AsyncMock(return_value=phone_row),
    )
    logger = Mock()
    monkeypatch.setattr(routes, "logger", logger)
    monkeypatch.setattr(routes, "db_client", db)
    monkeypatch.setattr(
        routes,
        "parse_webhook_request",
        AsyncMock(return_value=(webhook_data, "raw-body")),
    )
    monkeypatch.setattr(
        routes, "_detect_provider", AsyncMock(return_value=provider_class)
    )
    monkeypatch.setattr(routes, "normalize_webhook_data", Mock(return_value=normalized))
    monkeypatch.setattr(
        routes, "get_telephony_provider_by_id", AsyncMock(return_value=provider)
    )
    monkeypatch.setattr(
        "api.services.telephony.factory.find_telephony_config_for_inbound",
        AsyncMock(return_value=(55, {})),
    )

    request = SimpleNamespace(
        headers={"x-signature": "unsigned", "x-custom-id": identifiers[6]},
        url="https://api.example.test/inbound/33",
    )
    assert await handle_inbound_telephony(33, request) == "invalid-signature"
    messages = " ".join(
        repr(call.args)
        for level in (logger.info, logger.warning, logger.error)
        for call in level.call_args_list
    )
    assert all(identifier not in messages for identifier in identifiers)
    provider.verify_inbound_signature.assert_awaited_once()


@pytest.mark.asyncio
async def test_legacy_non_inbound_direction_is_not_logged(monkeypatch):
    import api.routes.telephony as routes

    direction = "forged-direction-value"
    logger = Mock()
    provider_class = SimpleNamespace(PROVIDER_NAME="koreaaitel")
    monkeypatch.setattr(routes, "logger", logger)
    monkeypatch.setattr(
        routes,
        "parse_webhook_request",
        AsyncMock(return_value=({"Direction": direction}, "raw-body")),
    )
    monkeypatch.setattr(
        routes, "_detect_provider", AsyncMock(return_value=provider_class)
    )
    monkeypatch.setattr(
        routes,
        "normalize_webhook_data",
        Mock(return_value=SimpleNamespace(direction=direction)),
    )
    monkeypatch.setattr(
        routes, "generic_hangup_response", Mock(return_value="hangup")
    )

    request = SimpleNamespace(headers={}, url="https://api.example.test/inbound/33")
    assert await handle_inbound_telephony(33, request) == "hangup"

    logger.warning.assert_called_once_with("Non-inbound call received")
    assert direction not in repr(logger.mock_calls)


@pytest.mark.asyncio
async def test_unknown_provider_detection_does_not_log_webhook_keys(monkeypatch):
    import api.routes.telephony as routes

    untrusted_key = "account-identifier-as-field-name"
    logger = Mock()
    monkeypatch.setattr(routes, "logger", logger)
    monkeypatch.setattr(
        routes, "get_all_telephony_providers", AsyncMock(return_value=[])
    )

    assert await routes._detect_provider({untrusted_key: "value"}, {}) is None

    logger.warning.assert_called_once_with("No provider found for webhook")
    assert untrusted_key not in repr(logger.mock_calls)


@pytest.mark.asyncio
async def test_inbound_fallback_does_not_log_webhook_or_headers(monkeypatch):
    import api.routes.telephony as routes

    payload = {
        "CallSid": "CALL-UNVERIFIED-123",
        "From": "01012345678",
        "To": "07012340001",
    }
    header_value = "HEADER-UNVERIFIED-789"
    logger = Mock()
    provider = SimpleNamespace(
        PROVIDER_NAME="koreaaitel",
        generate_error_response=Mock(return_value="provider-fallback"),
    )
    monkeypatch.setattr(routes, "logger", logger)
    monkeypatch.setattr(
        routes, "parse_webhook_request", AsyncMock(return_value=(payload, "raw"))
    )
    monkeypatch.setattr(routes, "_detect_provider", AsyncMock(return_value=provider))
    monkeypatch.setattr(routes, "generic_hangup_response", Mock(return_value="fallback"))
    request = SimpleNamespace(headers={"x-custom-id": header_value})

    assert await routes.handle_inbound_fallback(request) == "provider-fallback"
    messages = " ".join(repr(call.args) for call in logger.info.call_args_list)
    assert all(value not in messages for value in (*payload.values(), header_value))


@pytest.mark.asyncio
async def test_inbound_account_validation_failure_does_not_log_account_id(monkeypatch):
    import api.routes.telephony as routes

    account_id = "ACCOUNT-UNVERIFIED-456"
    logger = Mock()
    db = Mock(
        list_telephony_configurations_by_provider=AsyncMock(return_value=[object()])
    )
    monkeypatch.setattr(routes, "logger", logger)
    monkeypatch.setattr(routes, "db_client", db)
    monkeypatch.setattr(
        "api.services.telephony.factory.find_telephony_config_for_inbound",
        AsyncMock(return_value=None),
    )

    result = await routes._resolve_inbound_telephony_config(
        11, SimpleNamespace(PROVIDER_NAME="koreaaitel"), account_id
    )

    assert result == (TelephonyError.ACCOUNT_VALIDATION_FAILED, None)
    messages = " ".join(repr(call.args) for call in logger.warning.call_args_list)
    assert account_id not in messages


@pytest.mark.asyncio
async def test_inbound_run_rejects_when_concurrency_limit_reached():
    request = SimpleNamespace(headers={}, url="https://api.example.com/inbound/run")
    provider_class = SimpleNamespace(
        PROVIDER_NAME="twilio",
        generate_validation_error_response=Mock(return_value="limit-response"),
    )
    normalized_data = SimpleNamespace(
        provider="twilio",
        direction="inbound",
        to_number="+15551230000",
        from_number="+15557650000",
        to_country="US",
        from_country="US",
        account_id="acct-1",
        call_id="call-1",
        raw_data={},
    )
    config = SimpleNamespace(
        id=55,
        organization_id=11,
        name="Twilio",
        credentials={"account_sid": "AC123"},
    )
    phone_row = SimpleNamespace(id=77, address="+15551230000", inbound_workflow_id=33)
    workflow = SimpleNamespace(id=33, user_id=99)
    provider_instance = SimpleNamespace(
        verify_inbound_signature=AsyncMock(return_value=True)
    )

    with (
        patch(
            "api.routes.telephony.parse_webhook_request",
            new=AsyncMock(return_value=({}, "raw-body")),
        ),
        patch(
            "api.routes.telephony._detect_provider",
            new=AsyncMock(return_value=provider_class),
        ),
        patch(
            "api.routes.telephony.normalize_webhook_data",
            return_value=normalized_data,
        ),
        patch("api.routes.telephony.db_client") as mock_db,
        patch(
            "api.routes.telephony.get_telephony_provider_by_id",
            new=AsyncMock(return_value=provider_instance),
        ),
        patch("api.routes.telephony.call_concurrency") as mock_concurrency,
    ):
        mock_db.find_inbound_route_by_account = AsyncMock(
            return_value=(config, phone_row)
        )
        mock_db.get_workflow = AsyncMock(return_value=workflow)
        mock_db.create_workflow_run = AsyncMock()
        mock_concurrency.acquire_org_slot = AsyncMock(
            side_effect=CallConcurrencyLimitError(
                organization_id=config.organization_id,
                source="inbound:twilio",
                wait_time=0,
                max_concurrent=1,
            )
        )

        response = await handle_inbound_run(request)

    assert response == "limit-response"
    provider_class.generate_validation_error_response.assert_called_once_with(
        TelephonyError.CONCURRENT_CALL_LIMIT
    )
    mock_db.create_workflow_run.assert_not_awaited()


@pytest.mark.asyncio
async def test_smallwebrtc_run_reaching_telephony_websocket_closes_without_running():
    websocket = AsyncMock()
    workflow_run = SimpleNamespace(
        id=501,
        workflow_id=33,
        mode=WorkflowRunMode.SMALLWEBRTC.value,
        state=WorkflowRunState.INITIALIZED.value,
        initial_context={},
        gathered_context={},
    )
    workflow = SimpleNamespace(id=33, organization_id=11, user_id=99)
    provider_lookup = AsyncMock()

    with (
        patch("api.routes.telephony.db_client") as mock_db,
        patch("api.routes.telephony.call_concurrency") as mock_concurrency,
        patch(
            "api.routes.telephony.get_telephony_provider_for_run",
            new=provider_lookup,
        ),
    ):
        mock_concurrency.unregister_active_call = AsyncMock()
        mock_db.get_workflow_run = AsyncMock(return_value=workflow_run)
        mock_db.get_workflow = AsyncMock(return_value=workflow)
        mock_db.update_workflow_run = AsyncMock()

        await _handle_telephony_websocket(websocket, 33, 11, 501)

    mock_db.get_workflow_run.assert_awaited_once_with(501, organization_id=11)
    mock_db.get_workflow.assert_awaited_once_with(33, organization_id=11)
    websocket.close.assert_awaited_once_with(
        code=4400,
        reason=(
            "smallwebrtc runs connect through the WebRTC signaling endpoint, "
            "not the telephony websocket"
        ),
    )
    assert mock_db.update_workflow_run.await_count == 0
    assert provider_lookup.await_count == 0
    mock_concurrency.unregister_active_call.assert_not_awaited()


@pytest.mark.asyncio
async def test_telephony_websocket_leaves_run_initialized_until_stream_starts():
    """Telnyx opens the media socket while the callee is still ringing. The run
    must stay 'initialized' until the stream starts, or a busy / no-answer
    hangup looks like a started pipeline and skips its post-call integrations.
    """
    websocket = AsyncMock()
    workflow_run = SimpleNamespace(
        id=502,
        workflow_id=33,
        mode=WorkflowRunMode.TELNYX.value,
        state=WorkflowRunState.INITIALIZED.value,
        initial_context={"provider": "telnyx"},
        gathered_context={},
    )
    workflow = SimpleNamespace(id=33, organization_id=11, user_id=99)
    provider = SimpleNamespace(PROVIDER_NAME="telnyx", handle_websocket=AsyncMock())

    with (
        patch("api.routes.telephony.db_client") as mock_db,
        patch(
            "api.routes.telephony.get_telephony_provider_for_run",
            new=AsyncMock(return_value=provider),
        ),
    ):
        mock_db.get_workflow_run = AsyncMock(return_value=workflow_run)
        mock_db.get_workflow = AsyncMock(return_value=workflow)
        mock_db.update_workflow_run = AsyncMock()

        await _handle_telephony_websocket(websocket, 33, 11, 502)

    provider.handle_websocket.assert_awaited_once_with(websocket, 33, 11, 502)
    mock_db.update_workflow_run.assert_not_awaited()

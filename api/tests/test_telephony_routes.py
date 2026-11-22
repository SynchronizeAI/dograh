import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import ANY, AsyncMock, Mock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.enums import WorkflowRunMode, WorkflowRunState
from api.errors.telephony_errors import TelephonyError
from api.routes.telephony import (
    _handle_telephony_websocket,
    handle_inbound_fallback,
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


@pytest.mark.asyncio
@pytest.mark.parametrize("provider_detected", [True, False])
async def test_inbound_fallback_redacts_call_token_before_logging(
    provider_detected,
):
    webhook_data = {
        "CallId": "call-log",
        "AccountId": "AC123",
        "From": "01012345678",
        "CallToken": "secret-capability",
    }
    request = SimpleNamespace(headers={})
    detected_provider = KoreaAITelProvider if provider_detected else None

    with (
        patch(
            "api.routes.telephony.parse_webhook_request",
            new=AsyncMock(return_value=(webhook_data, "raw-body")),
        ),
        patch(
            "api.routes.telephony._detect_provider",
            new=AsyncMock(return_value=detected_provider),
        ),
        patch("api.routes.telephony.logger.info") as log_info,
    ):
        await handle_inbound_fallback(request)

    logged_payload = "\n".join(str(call.args) for call in log_info.call_args_list)
    assert "secret-capability" not in logged_payload
    assert '"CallToken": "[REDACTED]"' in logged_payload
    assert '"CallId": "call-log"' in logged_payload
    assert '"From": "01012345678"' in logged_payload
    assert webhook_data["CallToken"] == "secret-capability"


@pytest.mark.asyncio
async def test_legacy_inbound_redacts_call_token_before_logging():
    webhook_data = {
        "CallId": "call-log",
        "AccountId": "AC123",
        "From": "01012345678",
        "To": "07012345678",
        "Direction": "inbound",
        "CallToken": "secret-capability",
    }
    request = SimpleNamespace(
        headers={}, url="https://api.example.test/telephony/inbound/33"
    )

    with (
        patch(
            "api.routes.telephony.parse_webhook_request",
            new=AsyncMock(return_value=(webhook_data, "raw-body")),
        ),
        patch(
            "api.routes.telephony._detect_provider",
            new=AsyncMock(return_value=KoreaAITelProvider),
        ),
        patch(
            "api.routes.telephony._validate_inbound_request",
            new=AsyncMock(
                return_value=(
                    False,
                    TelephonyError.SIGNATURE_VALIDATION_FAILED,
                    {},
                    None,
                )
            ),
        ),
        patch("api.routes.telephony.logger.info") as log_info,
    ):
        await handle_inbound_telephony(33, request)

    logged_messages = "\n".join(str(call.args) for call in log_info.call_args_list)
    assert "secret-capability" not in logged_messages
    assert "CallToken': '[REDACTED]'" in logged_messages
    assert "CallId': 'call-log'" in logged_messages
    assert "From': '01012345678'" in logged_messages
    assert webhook_data["CallToken"] == "secret-capability"


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
async def test_koreaaitel_replay_reuses_run_and_new_call_id_creates_run():
    request = SimpleNamespace(headers={}, url="https://api.example.com/inbound/run")
    provider_class = SimpleNamespace(
        PROVIDER_NAME="koreaaitel",
        generate_validation_error_response=Mock(return_value="rejected"),
    )
    config = SimpleNamespace(
        id=55, organization_id=11, name="KoreaAITel", credentials={}
    )
    phone_row = SimpleNamespace(id=77, inbound_workflow_id=33, address="07012345678")
    workflow = SimpleNamespace(id=33, user_id=99, organization_id=11)
    existing_run = SimpleNamespace(
        id=999,
        mode="koreaaitel",
        workflow_id=33,
        workflow=workflow,
        state=WorkflowRunState.INITIALIZED.value,
        is_completed=False,
    )
    provider_instance = SimpleNamespace(
        verify_inbound_signature=AsyncMock(return_value=True),
        start_inbound_stream=AsyncMock(side_effect=["first", "replay", "new-call"]),
    )
    call_ids = iter(["call-1", "call-1", "call-2"])

    def normalize(*_args):
        call_id = next(call_ids)
        return SimpleNamespace(
            provider="koreaaitel",
            direction="inbound",
            to_number="07012345678",
            from_number="01012345678",
            to_country="KR",
            from_country="KR",
            account_id="account-1",
            call_id=call_id,
            raw_data={"CallId": call_id},
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
        patch("api.routes.telephony.normalize_webhook_data", side_effect=normalize),
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
            new=AsyncMock(
                return_value=SimpleNamespace(has_quota=True, error_message="")
            ),
        ),
        patch(
            "api.routes.telephony.prepare_workflow_run_inputs",
            new=AsyncMock(return_value={}),
        ),
        patch("api.routes.telephony.call_concurrency") as mock_concurrency,
        patch(
            "api.routes.telephony._create_inbound_workflow_run",
            new=AsyncMock(side_effect=[(999, None), (1000, None)]),
        ) as create_run,
    ):
        mock_db.find_inbound_route_by_account = AsyncMock(
            return_value=(config, phone_row)
        )
        mock_db.get_workflow = AsyncMock(return_value=workflow)
        mock_db.get_workflow_run_by_call_id = AsyncMock(
            side_effect=[None, existing_run, None]
        )
        mock_db.workflow_run_replay_lock = Mock(
            side_effect=lambda *_args: _noop_async_context()
        )
        mock_concurrency.acquire_org_slot = AsyncMock(return_value=object())
        mock_concurrency.bind_workflow_run = AsyncMock()
        mock_concurrency.release_workflow_run_slot = AsyncMock()
        mock_concurrency.release_slot = AsyncMock()

        first = await handle_inbound_run(request)
        replay = await handle_inbound_run(request)
        different_call = await handle_inbound_run(request)

    assert (first, replay, different_call) == ("first", "replay", "new-call")
    assert create_run.await_count == 2
    assert provider_instance.verify_inbound_signature.await_count == 3
    assert (
        provider_instance.start_inbound_stream.await_args_list[1].kwargs[
            "workflow_run_id"
        ]
        == 999
    )
    assert mock_db.get_workflow_run_by_call_id.await_args_list[0].args == (
        "call-1",
        "koreaaitel",
    )
    assert mock_db.get_workflow_run_by_call_id.await_args_list[1].args == (
        "call-1",
        "koreaaitel",
    )
    assert mock_db.get_workflow_run_by_call_id.await_args_list[2].args == (
        "call-2",
        "koreaaitel",
    )


@pytest.mark.parametrize("quota_allowed", [True, False])
@pytest.mark.asyncio
async def test_concurrent_koreaaitel_deliveries_create_one_run(
    setup_test_database, monkeypatch, quota_allowed
):
    from api.db import db_client

    request = SimpleNamespace(headers={}, url="https://api.example.com/inbound/run")
    webhook_data = {
        "CallId": "same-call-id",
        "AccountId": "account-1",
        "From": "01012345678",
        "To": "07012345678",
    }
    normalized_data = SimpleNamespace(
        provider="koreaaitel",
        direction="inbound",
        to_number="07012345678",
        from_number="01012345678",
        to_country="KR",
        from_country="KR",
        account_id="account-1",
        call_id="same-call-id",
        raw_data=webhook_data,
    )
    provider_class = SimpleNamespace(
        PROVIDER_NAME="koreaaitel",
        generate_validation_error_response=Mock(return_value="rejected"),
    )
    config = SimpleNamespace(
        id=55, organization_id=11, name="KoreaAITel", credentials={}
    )
    phone_row = SimpleNamespace(id=77, inbound_workflow_id=33, address="07012345678")
    workflow = SimpleNamespace(id=33, user_id=99, organization_id=11)
    provider_instance = SimpleNamespace(
        verify_inbound_signature=AsyncMock(return_value=True),
        start_inbound_stream=AsyncMock(side_effect=["first", "replay"]),
    )
    runs = {}
    lookup_sessions = []
    create_sessions = []
    authorization_sessions = []
    failure_sessions = []
    next_run_id = iter([700, 701])

    async def get_run_by_call_id(call_id, provider, session=None):
        await asyncio.sleep(0)
        lookup_sessions.append(session)
        return runs.get((provider, call_id))

    async def create_run(*args, **kwargs):
        normalized = args[4]
        create_sessions.append(kwargs["session"])
        await asyncio.sleep(0.02)
        run_id = next(next_run_id)
        runs[(args[3], normalized.call_id)] = SimpleNamespace(
            id=run_id,
            mode=args[3],
            workflow_id=args[0],
            workflow=SimpleNamespace(organization_id=args[2]),
            state=WorkflowRunState.INITIALIZED.value,
            is_completed=False,
        )
        return run_id, None

    async def authorize_run(**kwargs):
        session = kwargs["session"]
        authorization_sessions.append(session)
        assert session is create_sessions[-1]
        assert session.in_transaction()
        return SimpleNamespace(
            has_quota=quota_allowed,
            error_message="quota denied" if not quota_allowed else "",
        )

    async def fail_run(run_id, _error_message, *, session):
        failure_sessions.append(session)
        assert session is create_sessions[-1]
        assert session.in_transaction()
        for run in runs.values():
            if run.id == run_id:
                run.state = WorkflowRunState.COMPLETED.value
                run.is_completed = True
                break

    stream_responses = ["first", "replay"]
    expected_responses = (
        ["first", "replay"] if quota_allowed else ["rejected", "rejected"]
    )

    with (
        patch(
            "api.routes.telephony.parse_webhook_request",
            new=AsyncMock(return_value=(webhook_data, "raw-body")),
        ),
        patch(
            "api.routes.telephony._detect_provider",
            new=AsyncMock(return_value=provider_class),
        ),
        patch(
            "api.routes.telephony.normalize_webhook_data",
            return_value=normalized_data,
        ),
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
            new=AsyncMock(side_effect=authorize_run),
        ) as authorize_run,
        patch(
            "api.routes.telephony.mark_workflow_run_failed",
            new=AsyncMock(side_effect=fail_run),
        ) as mark_failed,
        patch(
            "api.routes.telephony.enqueue_failed_workflow_run_integrations",
            new=AsyncMock(),
        ) as enqueue_failure,
        patch("api.routes.telephony.call_concurrency") as concurrency,
        patch(
            "api.routes.telephony._create_inbound_workflow_run",
            new=AsyncMock(side_effect=create_run),
        ) as create_workflow_run,
    ):
        provider_instance.start_inbound_stream.side_effect = stream_responses
        concurrency.acquire_org_slot = AsyncMock(return_value=object())
        concurrency.bind_workflow_run = AsyncMock()
        concurrency.release_workflow_run_slot = AsyncMock()
        concurrency.release_slot = AsyncMock()
        monkeypatch.setattr(
            db_client,
            "find_inbound_route_by_account",
            AsyncMock(return_value=(config, phone_row)),
        )
        monkeypatch.setattr(
            db_client, "get_workflow", AsyncMock(return_value=workflow)
        )
        monkeypatch.setattr(
            db_client, "get_workflow_run_by_call_id", get_run_by_call_id
        )

        responses = await asyncio.gather(
            handle_inbound_run(request), handle_inbound_run(request)
        )

    assert sorted(responses) == sorted(expected_responses)
    assert create_workflow_run.await_count == 1
    assert authorize_run.await_count == 1
    assert concurrency.acquire_org_slot.await_count == 1
    assert len(runs) == 1
    assert len(create_sessions) == 1
    assert authorization_sessions == [create_sessions[0]]
    assert (
        all(run.state == WorkflowRunState.COMPLETED.value for run in runs.values())
        == (not quota_allowed)
    )
    assert mark_failed.await_count == (0 if quota_allowed else 1)
    assert failure_sessions == ([] if quota_allowed else [create_sessions[0]])
    assert enqueue_failure.await_count == (0 if quota_allowed else 1)
    assert any(session is create_sessions[0] for session in lookup_sessions)
    assert lookup_sessions[0] is not lookup_sessions[1]
    assert provider_instance.start_inbound_stream.await_count == (
        2 if quota_allowed else 0
    )
    if quota_allowed:
        assert {
            call.kwargs["workflow_run_id"]
            for call in provider_instance.start_inbound_stream.await_args_list
        } == {700}


@asynccontextmanager
async def _noop_async_context():
    yield


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

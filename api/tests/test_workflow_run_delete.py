"""DELETE /api/v1/workflow/{workflow_id}/runs/{run_id}.

The run row holds the caller's number, transcript, extraction and QA result,
so a privacy deletion has to remove it. The route deletes it (and, through
ON DELETE CASCADE, its webhook deliveries), scoped to the caller's
organization and the workflow in the path, and refuses while the call or its
post-call steps may still write.
"""

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from sqlalchemy import func, select

from api.db.models import (
    OrganizationModel,
    UserModel,
    WebhookDeliveryModel,
    WorkflowModel,
    WorkflowRunModel,
)

DONE = {"workflow_completion": {"status": "done"}}


async def _org_user_workflow(async_session):
    org = OrganizationModel(provider_id=f"org-{uuid4()}")
    async_session.add(org)
    await async_session.flush()
    user = UserModel(provider_id=f"user-{uuid4()}", selected_organization_id=org.id)
    async_session.add(user)
    await async_session.flush()
    workflow = WorkflowModel(
        name="Line",
        user_id=user.id,
        organization_id=org.id,
        workflow_definition={"nodes": [], "edges": []},
        template_context_variables={},
    )
    async_session.add(workflow)
    await async_session.flush()
    return org, user, workflow


async def _run(async_session, workflow, *, is_completed=True, extra=None, age=None):
    run = WorkflowRunModel(
        name="Call",
        workflow_id=workflow.id,
        mode="twilio",
        is_completed=is_completed,
        extra=DONE if extra is None else extra,
        initial_context={"caller_number": "+15105550100"},
        gathered_context={"callback_number": "+15105550101"},
        logs={"transcript": "caller: hello"},
        annotations={"qa": "fine"},
    )
    if age is not None:
        run.created_at = datetime.now(UTC) - age
    async_session.add(run)
    await async_session.flush()
    return run


async def _exists(async_session, model, *where) -> bool:
    count = await async_session.scalar(
        select(func.count()).select_from(model).where(*where)
    )
    return count > 0


async def test_delete_removes_run_and_its_webhook_deliveries(
    async_session, db_session, test_client_factory
):
    org, user, workflow = await _org_user_workflow(async_session)
    run = await _run(async_session, workflow)
    await db_session.create_webhook_delivery(
        workflow_run_id=run.id,
        organization_id=org.id,
        endpoint_url="https://example.com/hook",
        payload={"transcript": "caller: hello"},
        max_attempts=5,
        webhook_node_id="node-1",
    )
    path = f"/api/v1/workflow/{workflow.id}/runs/{run.id}"

    async with test_client_factory(user) as client:
        deleted = await client.delete(path)
        again = await client.delete(path)
        read = await client.get(path)

    assert deleted.status_code == 204
    assert deleted.content == b""
    assert not await _exists(
        async_session, WorkflowRunModel, WorkflowRunModel.id == run.id
    )
    assert not await _exists(
        async_session,
        WebhookDeliveryModel,
        WebhookDeliveryModel.workflow_run_id == run.id,
    )
    assert again.status_code == 404
    assert read.status_code == 404


async def test_delete_is_scoped_to_organization_and_workflow(
    async_session, test_client_factory
):
    _, owner, workflow = await _org_user_workflow(async_session)
    _, stranger, _ = await _org_user_workflow(async_session)
    other_workflow = WorkflowModel(
        name="Other line",
        user_id=owner.id,
        organization_id=owner.selected_organization_id,
        workflow_definition={"nodes": [], "edges": []},
        template_context_variables={},
    )
    async_session.add(other_workflow)
    no_org = UserModel(provider_id=f"user-{uuid4()}")
    async_session.add(no_org)
    await async_session.flush()
    run = await _run(async_session, workflow)

    async with test_client_factory(stranger) as client:
        foreign = await client.delete(f"/api/v1/workflow/{workflow.id}/runs/{run.id}")
    async with test_client_factory(owner) as client:
        wrong_workflow = await client.delete(
            f"/api/v1/workflow/{other_workflow.id}/runs/{run.id}"
        )
    async with test_client_factory(no_org) as client:
        unscoped = await client.delete(f"/api/v1/workflow/{workflow.id}/runs/{run.id}")

    assert foreign.status_code == 404
    assert wrong_workflow.status_code == 404
    assert unscoped.status_code == 404
    assert await _exists(async_session, WorkflowRunModel, WorkflowRunModel.id == run.id)


@pytest.mark.parametrize(
    ("is_completed", "extra"),
    [
        (False, {}),
        (True, {"workflow_completion": {"status": "pending"}}),
        (True, {"workflow_completion": {"status": "running"}}),
    ],
    ids=["call-live", "upload-pending", "completion-running"],
)
async def test_delete_refuses_a_run_still_being_written(
    async_session, test_client_factory, is_completed, extra
):
    _, user, workflow = await _org_user_workflow(async_session)
    run = await _run(async_session, workflow, is_completed=is_completed, extra=extra)

    async with test_client_factory(user) as client:
        response = await client.delete(f"/api/v1/workflow/{workflow.id}/runs/{run.id}")

    assert response.status_code == 409
    assert await _exists(async_session, WorkflowRunModel, WorkflowRunModel.id == run.id)


@pytest.mark.parametrize(
    ("is_completed", "extra", "age"),
    [
        (True, {}, None),
        (False, {}, timedelta(hours=2)),
        (True, {"workflow_completion": {"status": "running"}}, timedelta(hours=2)),
    ],
    ids=["no-completion-record", "abandoned-call", "stuck-completion"],
)
async def test_delete_accepts_a_settled_run(
    async_session, test_client_factory, is_completed, extra, age
):
    _, user, workflow = await _org_user_workflow(async_session)
    run = await _run(
        async_session, workflow, is_completed=is_completed, extra=extra, age=age
    )

    async with test_client_factory(user) as client:
        response = await client.delete(f"/api/v1/workflow/{workflow.id}/runs/{run.id}")

    assert response.status_code == 204
    assert not await _exists(
        async_session, WorkflowRunModel, WorkflowRunModel.id == run.id
    )

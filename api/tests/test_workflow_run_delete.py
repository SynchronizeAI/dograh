from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.routes.workflow import router
from api.services.auth.depends import get_user


def _app(authenticated=True):
    app = FastAPI()
    app.include_router(router)
    if authenticated:
        app.dependency_overrides[get_user] = lambda: SimpleNamespace(
            id=1, selected_organization_id=11
        )
    return app


def test_delete_workflow_run_removes_stored_artifacts_and_row():
    run = SimpleNamespace(
        mode="twilio",
        storage_backend="s3",
        recording_url="recordings/7.wav",
        transcript_url="transcripts/7.txt",
        extra={
            "workflow_completion": {"status": "running"},
            "recordings": {"user": {"storage_key": "recordings/7/user.wav"}},
        },
    )
    storage = SimpleNamespace(adelete_file=AsyncMock(return_value=True))
    with (
        patch("api.routes.workflow.db_client") as db,
        patch("api.routes.workflow.get_storage_for_backend", return_value=storage),
    ):
        db.get_workflow_run = AsyncMock(return_value=run)
        db.delete_workflow_run = AsyncMock(return_value=True)
        response = TestClient(_app()).delete("/workflow/3/runs/7")

    assert response.status_code == 204
    db.get_workflow_run.assert_awaited_once_with(
        7, organization_id=11, workflow_id=3
    )
    assert {call.args[0] for call in storage.adelete_file.await_args_list} == {
        "recordings/7.wav",
        "transcripts/7.txt",
        "recordings/7/user.wav",
    }
    db.delete_workflow_run.assert_awaited_once_with(7, 3, 11)


def test_delete_workflow_run_returns_conflict_while_artifacts_are_publishing():
    run = SimpleNamespace(
        is_completed=True,
        mode="twilio",
        storage_backend="s3",
        recording_url="recordings/7.wav",
        transcript_url="transcripts/7.txt",
        extra={
            "workflow_completion": {"status": "pending"},
            "recordings": {"user": {"storage_key": "recordings/7/user.wav"}},
        },
    )
    storage = SimpleNamespace(adelete_file=AsyncMock(return_value=True))
    with (
        patch("api.routes.workflow.db_client") as db,
        patch("api.routes.workflow.get_storage_for_backend", return_value=storage),
    ):
        db.get_workflow_run = AsyncMock(return_value=run)
        db.delete_workflow_run = AsyncMock(return_value=True)
        response = TestClient(_app()).delete("/workflow/3/runs/7")

    assert response.status_code == 409
    assert (
        response.json()["detail"]
        == "Workflow run artifacts are still being published"
    )
    storage.adelete_file.assert_not_awaited()
    db.delete_workflow_run.assert_not_awaited()


def test_delete_workflow_run_rejects_text_chat_runs():
    run = SimpleNamespace(
        mode="textchat",
        storage_backend="s3",
        recording_url="recordings/7.wav",
        transcript_url="transcripts/7.txt",
        extra={
            "recordings": {"user": {"storage_key": "recordings/7/user.wav"}}
        },
    )
    storage = SimpleNamespace(adelete_file=AsyncMock(return_value=True))
    with (
        patch("api.routes.workflow.db_client") as db,
        patch("api.routes.workflow.get_storage_for_backend", return_value=storage),
    ):
        db.get_workflow_run = AsyncMock(return_value=run)
        db.delete_workflow_run = AsyncMock(return_value=True)
        response = TestClient(_app()).delete("/workflow/3/runs/7")

    assert response.status_code == 409
    assert (
        response.json()["detail"] == "Only telephony workflow runs can be deleted"
    )
    storage.adelete_file.assert_not_awaited()
    db.delete_workflow_run.assert_not_awaited()


def test_delete_workflow_run_hides_cross_organization_and_missing_runs():
    with patch("api.routes.workflow.db_client") as db:
        db.get_workflow_run = AsyncMock(return_value=None)
        response = TestClient(_app()).delete("/workflow/3/runs/7")
        db.get_workflow_run.assert_awaited_once_with(
            7, organization_id=11, workflow_id=3
        )
        db.delete_workflow_run.assert_not_called()

        response_again = TestClient(_app()).delete("/workflow/3/runs/7")

    assert response.status_code == response_again.status_code == 404


def test_delete_workflow_run_requires_authentication():
    with patch("api.services.auth.depends.stackauth.get_user", new=AsyncMock(return_value=None)):
        response = TestClient(_app(authenticated=False)).delete("/workflow/3/runs/7")

    assert response.status_code == 401

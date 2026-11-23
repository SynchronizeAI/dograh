import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.routes.workflow import router
from api.services.auth.depends import get_user
from api.services.filesystem.local import LocalFileSystem
from api.services.filesystem.minio import MinioFileSystem
from api.services.filesystem.s3 import S3FileSystem


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
        id=7,
        mode="twilio",
        storage_backend="s3",
        recording_url="recordings/7.wav",
        transcript_url="transcripts/7.txt",
        extra={
            "workflow_completion": {"status": "done"},
            "recordings": {"user": {"storage_key": "recordings/7/user.wav"}},
        },
        logs={
            "noveum_payload": {
                "audio_manifest": [
                    {
                        "storage_key": "recordings/7/noveum/user/segment.wav",
                    }
                ]
            }
        },
    )
    storage = SimpleNamespace(
        adelete_file=AsyncMock(return_value=True),
        adelete_prefix=AsyncMock(return_value=True),
    )
    with (
        patch("api.routes.workflow.db_client") as db,
        patch("api.routes.workflow.get_storage_for_backend", return_value=storage),
    ):
        db.get_workflow_run = AsyncMock(return_value=run)
        db.delete_workflow_run = AsyncMock(return_value=True)
        response = TestClient(_app()).delete("/workflow/3/runs/7")

    assert response.status_code == 204
    storage.adelete_prefix.assert_awaited_once_with("recordings/7/")
    db.get_workflow_run.assert_awaited_once_with(
        7, organization_id=11, workflow_id=3
    )
    assert {call.args[0] for call in storage.adelete_file.await_args_list} == {
        "recordings/7.wav",
        "transcripts/7.txt",
    }
    db.delete_workflow_run.assert_awaited_once_with(7, 3, 11)


def test_delete_workflow_run_returns_conflict_while_artifacts_are_publishing():
    run = SimpleNamespace(
        id=7,
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
    storage = SimpleNamespace(
        adelete_file=AsyncMock(return_value=True),
        adelete_prefix=AsyncMock(return_value=True),
    )
    with (
        patch("api.routes.workflow.db_client") as db,
        patch("api.routes.workflow.get_storage_for_backend", return_value=storage),
    ):
        db.get_workflow_run = AsyncMock(return_value=run)
        db.delete_workflow_run = AsyncMock(return_value=True)
        response = TestClient(_app()).delete("/workflow/3/runs/7")

    assert response.status_code == 409
    assert response.json()["detail"] == "Workflow run completion has not finished"
    storage.adelete_file.assert_not_awaited()
    storage.adelete_prefix.assert_not_awaited()
    db.delete_workflow_run.assert_not_awaited()


def test_delete_workflow_run_rejects_active_calls_without_completion_state():
    run = SimpleNamespace(
        id=7,
        is_completed=False,
        mode="twilio",
        state="running",
        storage_backend="s3",
        recording_url=None,
        transcript_url=None,
        extra={},
        logs={},
    )
    storage = SimpleNamespace(
        adelete_file=AsyncMock(return_value=True),
        adelete_prefix=AsyncMock(return_value=True),
    )
    with (
        patch("api.routes.workflow.db_client") as db,
        patch("api.routes.workflow.get_storage_for_backend", return_value=storage),
    ):
        db.get_workflow_run = AsyncMock(return_value=run)
        db.delete_workflow_run = AsyncMock(return_value=True)
        response = TestClient(_app()).delete("/workflow/3/runs/7")

    assert response.status_code == 409
    assert response.json()["detail"] == "Workflow run completion has not finished"
    storage.adelete_file.assert_not_awaited()
    storage.adelete_prefix.assert_not_awaited()
    db.delete_workflow_run.assert_not_awaited()


def test_delete_workflow_run_rejects_text_chat_runs():
    run = SimpleNamespace(
        id=7,
        mode="textchat",
        storage_backend="s3",
        recording_url="recordings/7.wav",
        transcript_url="transcripts/7.txt",
        extra={
            "recordings": {"user": {"storage_key": "recordings/7/user.wav"}}
        },
    )
    storage = SimpleNamespace(
        adelete_file=AsyncMock(return_value=True),
        adelete_prefix=AsyncMock(return_value=True),
    )
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


def test_local_filesystem_deletes_only_the_requested_prefix(tmp_path):
    storage_root = tmp_path / "storage"
    run_file = storage_root / "recordings" / "7" / "noveum" / "segment.wav"
    sibling_file = storage_root / "recordings" / "70" / "noveum" / "segment.wav"
    run_file.parent.mkdir(parents=True)
    sibling_file.parent.mkdir(parents=True)
    run_file.write_bytes(b"run")
    sibling_file.write_bytes(b"sibling")
    storage = LocalFileSystem(str(storage_root))

    assert asyncio.run(storage.adelete_prefix("recordings/7/"))
    assert not run_file.exists()
    assert sibling_file.exists()


def test_s3_filesystem_deletes_all_pages_under_prefix():
    client = AsyncMock()
    client.list_objects_v2.side_effect = [
        {
            "Contents": [{"Key": "recordings/7/user.wav"}],
            "IsTruncated": True,
            "NextContinuationToken": "next-page",
        },
        {"Contents": [], "IsTruncated": False},
    ]
    client.delete_objects.return_value = {"Errors": []}

    class ClientContext:
        async def __aenter__(self):
            return client

        async def __aexit__(self, *_):
            return None

    storage = S3FileSystem(bucket_name="bucket")
    storage.session = SimpleNamespace(client=Mock(return_value=ClientContext()))

    assert asyncio.run(storage.adelete_prefix("recordings/7/"))
    assert client.list_objects_v2.await_args_list[0].kwargs == {
        "Bucket": "bucket",
        "Prefix": "recordings/7/",
    }
    assert client.list_objects_v2.await_args_list[1].kwargs["ContinuationToken"] == (
        "next-page"
    )
    client.delete_objects.assert_awaited_once_with(
        Bucket="bucket",
        Delete={"Objects": [{"Key": "recordings/7/user.wav"}], "Quiet": True},
    )


def test_minio_filesystem_deletes_listed_prefix_objects():
    client = SimpleNamespace(
        list_objects=Mock(
            return_value=[
                SimpleNamespace(object_name="recordings/7/user.wav"),
                SimpleNamespace(object_name="recordings/7/noveum/segment.wav"),
            ]
        ),
        remove_object=Mock(),
    )
    storage = MinioFileSystem.__new__(MinioFileSystem)
    storage.client = client
    storage.bucket_name = "bucket"

    assert asyncio.run(storage.adelete_prefix("recordings/7/"))
    client.list_objects.assert_called_once_with(
        "bucket", prefix="recordings/7/", recursive=True
    )
    assert {call.args[1] for call in client.remove_object.call_args_list} == {
        "recordings/7/user.wav",
        "recordings/7/noveum/segment.wav",
    }


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


def test_delete_workflow_run_is_in_openapi_and_generated_sdk_contract():
    spec = TestClient(_app()).get("/openapi.json").json()
    operation = spec["paths"]["/workflow/{workflow_id}/runs/{run_id}"]["delete"]

    assert operation["x-sdk-method"] == "delete_workflow_run"
    assert "204" in operation["responses"]
    assert "409" in operation["responses"]

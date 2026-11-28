"""Missing OSS capability: the vendor's status and stream-end callbacks, checked by its X-Signature."""

from datetime import UTC, datetime

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request, Response
from loguru import logger

from api.db import db_client
from api.services.telephony.factory import get_telephony_provider_for_run
from api.services.telephony.status_processor import (
    StatusCallbackRequest,
    _process_status_update,
)

router = APIRouter()

_HANGUP = '<?xml version="1.0" encoding="UTF-8"?>\n<Response><Hangup/></Response>'
_STREAM_END_FIELDS = ("StreamEvent", "StreamCloseCode", "StreamCloseReason")


@router.post("/koreaaitel/status-callback/{workflow_run_id}", include_in_schema=False)
async def handle_koreaaitel_status_callback(workflow_run_id: int, request: Request):
    run = await db_client.get_workflow_run_by_id(workflow_run_id)
    # Not the run's own provider: another provider's check may pass an unsigned request.
    if not run or run.mode != "koreaaitel":
        return {"status": "ignored", "reason": "not_a_koreaaitel_run"}
    provider = await get_telephony_provider_for_run(run, run.workflow.organization_id)
    data = dict(await request.form())
    if not await provider.verify_inbound_signature(
        str(request.url), data, dict(request.headers)
    ):
        raise HTTPException(status_code=401, detail="Invalid webhook signature")
    status = provider.parse_status_callback(data)
    await _process_status_update(workflow_run_id, StatusCallbackRequest(**status))
    return {"status": "success"}


@router.post("/koreaaitel/stream-ended/{workflow_run_id}", include_in_schema=False)
async def handle_koreaaitel_stream_ended(
    workflow_run_id: int, request: Request, background: BackgroundTasks
):
    """``<Connect action>``: hang up at once, as before, and keep why the stream closed.

    The reason is kept after the reply, so a slow or failing lookup never holds the
    call open or sends the vendor to the line's fallback URL.
    """
    data = dict(await request.form())
    background.add_task(
        _keep_stream_end, workflow_run_id, str(request.url), data, dict(request.headers)
    )
    return Response(content=_HANGUP, media_type="application/xml")


async def _keep_stream_end(workflow_run_id: int, url: str, data: dict, headers: dict):
    """Only a signed callback for a 한국AI통신 run is written to the run's logs."""
    ended = {k: str(data.get(k, ""))[:200] for k in _STREAM_END_FIELDS}
    try:
        run = await db_client.get_workflow_run_by_id(workflow_run_id)
        signed = False
        if run and run.mode == "koreaaitel":
            provider = await get_telephony_provider_for_run(
                run, run.workflow.organization_id
            )
            signed = await provider.verify_inbound_signature(url, data, headers)
        if not signed:
            logger.warning(
                f"[run {workflow_run_id}] unverified stream-ended callback: {ended}"
            )
            return
        logger.info(f"[run {workflow_run_id}] 한국AI통신 stream ended: {ended}")
        ended["timestamp"] = datetime.now(UTC).isoformat()
        await db_client.update_workflow_run(
            run_id=workflow_run_id, logs={"koreaaitel_stream_end": ended}
        )
    except Exception:
        logger.exception(
            f"[run {workflow_run_id}] could not keep the stream end {ended}"
        )

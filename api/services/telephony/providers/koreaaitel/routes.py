"""Missing OSS capability: the vendor's status and stream-end callbacks, checked by its X-Signature."""

from datetime import UTC, datetime

from fastapi import APIRouter, HTTPException, Request, Response
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
async def handle_koreaaitel_stream_ended(workflow_run_id: int, request: Request):
    """``<Connect action>``: keep why the stream closed, then end the call as before.

    Every answer is a hang-up, so an unsigned request changes nothing; only a
    signed one is written to the run.
    """
    data = dict(await request.form())
    ended = {k: str(data.get(k, ""))[:200] for k in _STREAM_END_FIELDS}
    run = await db_client.get_workflow_run_by_id(workflow_run_id)
    signed = False
    if run and run.mode == "koreaaitel":
        provider = await get_telephony_provider_for_run(
            run, run.workflow.organization_id
        )
        signed = await provider.verify_inbound_signature(
            str(request.url), data, dict(request.headers)
        )
    if signed:
        logger.info(f"[run {workflow_run_id}] 한국AI통신 stream ended: {ended}")
        ended["timestamp"] = datetime.now(UTC).isoformat()
        await db_client.update_workflow_run(
            run_id=workflow_run_id, logs={"koreaaitel_stream_end": ended}
        )
    else:
        logger.warning(
            f"[run {workflow_run_id}] unverified stream-ended callback: {ended}"
        )
    return Response(content=_HANGUP, media_type="application/xml")

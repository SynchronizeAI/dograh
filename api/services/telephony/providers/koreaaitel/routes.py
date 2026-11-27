"""Missing OSS capability: a status callback that checks the vendor's X-Signature."""

from fastapi import APIRouter, HTTPException, Request

from api.db import db_client
from api.services.telephony.factory import get_telephony_provider_for_run
from api.services.telephony.status_processor import (
    StatusCallbackRequest,
    _process_status_update,
)

router = APIRouter()


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

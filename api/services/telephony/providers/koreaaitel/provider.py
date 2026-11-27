"""한국AI통신 implementation of the TelephonyProvider interface.

Modeled on ``twilio/provider.py``. The vendor differs from Twilio in its field
names (``CallId``/``AccountId``, ``streamId``/``callId``), its signature
(HMAC-SHA256, Base64, over the URL plus the sorted key/value pairs) and in
having no call control after ``POST /calls``: the call ends when the stream
closes. Numbers are routed outside Dograh, as with ARI.
"""

import base64
import hashlib
import hmac
import json
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import aiohttp
from fastapi import HTTPException, Response
from loguru import logger

from api.enums import TelephonyCallStatus, WorkflowRunMode
from api.services.telephony import ws_auth
from api.services.telephony.base import (
    CallInitiationResult,
    NormalizedInboundData,
    ProviderSyncResult,
    TelephonyProvider,
)
from api.utils.common import get_backend_endpoints
from api.utils.telephony_address import normalize_telephony_address

if TYPE_CHECKING:
    from fastapi import WebSocket

_COUNTRY = "KR"
# <Reject> as the first verb refuses the call before it is answered.
_REJECT = '<?xml version="1.0" encoding="UTF-8"?>\n<Response><Reject reason="rejected"/></Response>'


def _canonical(raw: str) -> str:
    return normalize_telephony_address(raw, country_hint=_COUNTRY).canonical


def _national(raw: str) -> str:
    """The calls API takes numbers as dialled; 8-digit 1xxx-xxxx numbers have no leading 0."""
    number = "".join(c for c in raw if c not in " -()")
    if not number.startswith("+82"):
        return number  # a foreign number gets the vendor's UNSUPPORTED_COUNTRY
    rest = number[3:]
    return rest if rest[:1] == "1" and len(rest) == 8 else "0" + rest


def compute_signature(signing_secret: str, url: str, params: Dict[str, Any]) -> str:
    """The vendor's ``X-Signature``: Base64(HMAC-SHA256(secret, url + k1v1k2v2...))."""
    data = url + "".join(f"{k}{v}" for k, v in sorted(params.items()))
    digest = hmac.new(signing_secret.encode(), data.encode(), hashlib.sha256).digest()
    return base64.b64encode(digest).decode()


class KoreaAITelProvider(TelephonyProvider):
    PROVIDER_NAME = WorkflowRunMode.KOREAAITEL.value
    WEBHOOK_ENDPOINT = "twiml"  # shared route, as Cloudonix

    def __init__(self, config: Dict[str, Any]):
        self.account_id = config.get("account_id")
        self.signing_secret = config.get("signing_secret")
        self.api_key = config.get("api_key")
        self.from_numbers = config.get("from_numbers", [])
        self.default_from_number = config.get("default_from_number")

    async def initiate_call(
        self,
        to_number: str,
        webhook_url: str,
        workflow_run_id: Optional[int] = None,
        from_number: Optional[str] = None,
        **kwargs: Any,
    ) -> CallInitiationResult:
        if not self.validate_config() or workflow_run_id is None:
            raise ValueError("한국AI통신 not properly configured or no run id")
        from_number = self.select_from_number(from_number)  # never a guess (#31)
        backend_endpoint, _ = await get_backend_endpoints()
        body = {
            "To": _national(to_number),
            "From": _national(from_number),
            "Url": webhook_url,
            "StatusCallback": f"{backend_endpoint}/api/v1/telephony/koreaaitel/status-callback/{workflow_run_id}",
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Idempotency-Key": f'"dograh-run-{workflow_run_id}"',  # one run, one call
        }
        url = f"https://api.koreaaitel.com/v1/accounts/{self.account_id}/calls"
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=body, headers=headers) as response:
                call = await response.json()
                if response.status != 201:
                    raise HTTPException(
                        status_code=response.status, detail=json.dumps(call)
                    )
        return CallInitiationResult(
            call_id=call["callId"],
            status=call.get("status", "queued"),
            caller_number=from_number,
            provider_metadata={"call_id": call["callId"]},
            raw_response=call,
        )

    async def get_webhook_response(
        self, workflow_id: int, organization_id: int, workflow_run_id: int
    ) -> str:
        _, wss_backend_endpoint = await get_backend_endpoints()
        ws_url = ws_auth.build_media_ws_url(
            wss_backend_endpoint, workflow_id, organization_id, workflow_run_id
        )
        return (await self.start_inbound_stream(websocket_url=ws_url)).body.decode()

    async def get_call_status(self, call_id: str) -> Dict[str, Any]:
        raise NotImplementedError("한국AI통신 status arrives by callback")

    def parse_status_callback(self, data: Dict[str, Any]) -> Dict[str, Any]:
        raw = data.get("CallStatus", "")
        status = "busy" if raw == "rejected" else raw  # as for Vonage
        return {
            "call_id": data.get("CallId", ""),
            "status": TelephonyCallStatus.from_raw(status) or status,
            "duration": data.get("Duration"),
            "extra": {k: v for k, v in data.items() if k.lower() not in ("from", "to")},
        }

    # ======== Not offered: no call-control or transfer API ========

    async def transfer_call(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        raise NotImplementedError("한국AI통신 provider does not support transfers")

    def supports_transfers(self) -> bool:
        return False

    async def get_call_cost(self, call_id: str) -> Dict[str, Any]:
        return {"cost_usd": 0.0, "duration": 0, "status": "unknown"}

    async def get_available_phone_numbers(self) -> List[str]:
        return self.from_numbers

    async def validate_phone_number(self, address: str) -> ProviderSyncResult:
        # Numbers are routed in the vendor's console/API outside Dograh, as ARI.
        return ProviderSyncResult(ok=True)

    def validate_config(self) -> bool:
        return bool(self.account_id and self.signing_secret and self.api_key)

    # ======== Inbound ========

    @classmethod
    def can_handle_webhook(
        cls, webhook_data: Dict[str, Any], headers: Dict[str, str]
    ) -> bool:
        """The vendor signs with ``X-Signature`` and sends ``CallId``/``AccountId``."""
        return (
            "x-signature" in {k.lower() for k in headers}
            and "CallId" in webhook_data
            and str(webhook_data.get("AccountId", "")).startswith("AC")
        )

    @staticmethod
    def parse_inbound_webhook(webhook_data: Dict[str, Any]) -> NormalizedInboundData:
        # A withheld caller still arrives in ``From``; only the flag says so.
        restricted = str(webhook_data.get("CallerIdRestricted", "")).lower() == "true"
        from_raw = "" if restricted else webhook_data.get("From", "")
        to_raw = webhook_data.get("To", "")
        raw_data = dict(webhook_data)
        if restricted:
            raw_data["From"] = ""
            raw_data["ForwardedFrom"] = ""
        return NormalizedInboundData(
            provider=KoreaAITelProvider.PROVIDER_NAME,
            call_id=webhook_data.get("CallId", ""),
            from_number=_canonical(from_raw) if from_raw else "",
            to_number=_canonical(to_raw) if to_raw else "",
            direction=webhook_data.get("Direction", ""),
            call_status=webhook_data.get("CallStatus", ""),
            account_id=webhook_data.get("AccountId"),
            from_country=_COUNTRY,
            to_country=_COUNTRY,
            raw_data=raw_data,
        )

    @staticmethod
    def validate_account_id(config_data: dict, webhook_account_id: str) -> bool:
        return bool(webhook_account_id) and (
            config_data.get("account_id") == webhook_account_id
        )

    async def verify_webhook_signature(
        self, url: str, params: Dict[str, Any], signature: str
    ) -> bool:
        if not self.signing_secret or not signature:
            return False
        expected = compute_signature(self.signing_secret, url, params)
        return hmac.compare_digest(signature, expected)

    async def verify_inbound_signature(
        self,
        url: str,
        webhook_data: Dict[str, Any],
        headers: Dict[str, str],
        body: str = "",
    ) -> bool:
        signature = next(
            (v for k, v in headers.items() if k.lower() == "x-signature"), ""
        )
        if not signature:
            logger.warning("Inbound 한국AI통신 webhook missing X-Signature")
            return False
        return await self.verify_webhook_signature(url, webhook_data, signature)

    async def start_inbound_stream(self, *, websocket_url: str, **kwargs: Any):
        """``<Connect>`` without ``action``: the call ends when the stream closes."""
        voiceml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Connect>
        <Stream url="{websocket_url}"/>
    </Connect>
</Response>"""
        return Response(content=voiceml, media_type="application/xml")

    @staticmethod
    def generate_error_response(error_type: str, message: str) -> tuple:
        return Response(content=_REJECT, media_type="application/xml")

    @staticmethod
    def generate_validation_error_response(error_type) -> tuple:
        """Sync fork: answered 503, as Twilio's refusal is, so the vendor requests the number's
        ``voiceFallbackUrl`` (the line's voicemail) for a non-2xx answer (/docs/build/voiceml,
        대체 URL); a number without one gets the vendor's own error message."""
        return Response(content=_REJECT, media_type="application/xml", status_code=503)

    async def handle_websocket(
        self,
        websocket: "WebSocket",
        workflow_id: int,
        organization_id: int,
        workflow_run_id: int,
    ) -> None:
        """Twilio's handshake: ``connected``, then ``start`` with the stream ids."""
        from api.services.pipecat.run_pipeline import run_pipeline_telephony

        msg = json.loads(await websocket.receive_text())
        if msg.get("event") != "connected":
            logger.error(f"Expected 'connected' event, got: {msg.get('event')}")
            await websocket.close(code=4400, reason="Expected connected event")
            return

        start_msg = json.loads(await websocket.receive_text())
        if start_msg.get("event") != "start":
            logger.error("Expected 'start' event second")
            await websocket.close(code=4400, reason="Expected start event")
            return

        try:
            stream_id = start_msg["start"]["streamId"]
            call_id = start_msg["start"]["callId"]
        except KeyError:
            logger.error("Missing streamId or callId in start message")
            await websocket.close(code=4400, reason="Missing stream identifiers")
            return

        await run_pipeline_telephony(
            websocket,
            provider_name=self.PROVIDER_NAME,
            workflow_id=workflow_id,
            workflow_run_id=workflow_run_id,
            organization_id=organization_id,
            call_id=call_id,
            transport_kwargs={"stream_id": stream_id},
        )

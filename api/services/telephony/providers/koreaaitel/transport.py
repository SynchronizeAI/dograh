"""한국AI통신 transport factory."""

import json

from fastapi import WebSocket
from loguru import logger
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)

from api.services.pipecat.audio_config import AudioConfig
from api.services.pipecat.audio_mixer import build_audio_out_mixer
from api.services.pipecat.transport_params import realtime_param_overrides

from .serializers import TwilioFrameSerializer


class _VendorStopEndsInput:
    """Missing OSS capability: pipecat's Twilio serializer drops the vendor's ``stop``
    and ends a call only when the socket closes (run 37263: the vendor logged its
    stream closed, and no close reached us for 16 s). ``stop`` is read as the
    disconnect it means; other non-media events and the socket's own close are
    logged, so the next drop shows what arrived."""

    def __init__(self, websocket: WebSocket):
        self._websocket = websocket

    def __getattr__(self, name):
        return getattr(self._websocket, name)

    async def receive(self):
        message = await self._websocket.receive()
        if message.get("type") == "websocket.disconnect":
            logger.info(
                f"한국AI통신 socket closed: code={message.get('code')} reason={message.get('reason')!r}"
            )
        text = message.get("text")
        if text and '"media"' not in text[:40]:  # media leads with its event name
            try:
                event = json.loads(text).get("event")
            except (ValueError, AttributeError):
                event = None
            if event != "media":
                logger.info(f"한국AI통신 stream event: {event}")
            if event == "stop":
                return {"type": "websocket.disconnect", "code": 1000}
        return message


async def create_transport(
    websocket: WebSocket,
    workflow_run_id: int,
    audio_config: AudioConfig,
    organization_id: int,
    *,
    ambient_noise_config: dict | None = None,
    telephony_configuration_id: int | None = None,
    is_realtime: bool = False,
    stream_id: str,
):
    """No credentials and no REST hang-up: pipecat closes the socket and the reply to
    ``<Connect action>`` hangs up."""
    serializer = TwilioFrameSerializer(
        stream_sid=stream_id,
        params=TwilioFrameSerializer.InputParams(auto_hang_up=False),
    )

    mixer = await build_audio_out_mixer(
        audio_config.transport_out_sample_rate, ambient_noise_config
    )

    return FastAPIWebsocketTransport(
        websocket=_VendorStopEndsInput(websocket),
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_in_sample_rate=audio_config.transport_in_sample_rate,
            audio_out_sample_rate=audio_config.transport_out_sample_rate,
            audio_out_mixer=mixer,
            serializer=serializer,
            **realtime_param_overrides(is_realtime),
        ),
    )

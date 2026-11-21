"""Switch STT providers on the primary's first unrecoverable error.

Pipecat's ``ServiceSwitcher`` routes each frame to the active service.
``ServiceSwitcherStrategyFailover`` only moves after ``is_usable`` is already
false, which a websocket reaches only after three reconnects. This strategy
stops those reconnects and moves on the first connection error instead.
Audio the primary missed is not replayed, and the backup stays active until
the call ends.
"""

from __future__ import annotations

import contextvars

from loguru import logger

from api.errors.failure import (
    ErrorSource,
    annotate_failure_metadata,
    failure_metadata_for_processor,
)
from pipecat.frames.frames import CancelFrame, EndFrame, ErrorFrame, Frame
from pipecat.pipeline.service_switcher import ServiceSwitcher, ServiceSwitcherStrategy
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

_CONNECTION_MARKERS = (
    "connection closed",
    "error receiving messages",
    "reconnection",
    "failed to reconnect",
    "websocket",
)


def _connection_failure(error: ErrorFrame) -> bool:
    if isinstance(error.exception, (ConnectionError, TimeoutError, OSError)):
        return True
    text = (error.error or "").lower()
    return any(marker in text for marker in _CONNECTION_MARKERS)


def _stop_primary_reconnects(primary: FrameProcessor) -> None:
    """Make the next connection failure terminal, without replaying audio.

    ``WebsocketService`` would otherwise try three times, 4-10s apart.
    ``STTService._reconnect`` would buffer audio and replay it after a new
    socket. Neither is what a failover call should do.
    """
    if hasattr(primary, "_reconnect_on_error"):
        primary._reconnect_on_error = False

    async def _fail_instead_of_reconnecting():
        buffer = getattr(primary, "_reconnect_audio_buffer", None)
        if buffer is not None:
            buffer.clear()
        if hasattr(primary, "_reconnecting"):
            primary._reconnecting = False
        if hasattr(primary, "_need_reconnect"):
            primary._need_reconnect = False
        await primary.push_error(
            f"{primary} connection closed",
            exception=ConnectionError("STT fallback stopped reconnect"),
        )

    if hasattr(primary, "_reconnect"):
        primary._reconnect = _fail_instead_of_reconnecting


class _SwitchOnFirstUnrecoverableError(ServiceSwitcherStrategy):
    """One switch, on the first connection error, for the rest of the call."""

    def __init__(self, services: list[FrameProcessor]):
        super().__init__(services)
        self._switched = False
        self.prepare_switch = None
        self.complete_switch = None
        self.visit_over = None

    async def handle_error(self, error: ErrorFrame) -> FrameProcessor | None:
        if self._switched or (self.visit_over and self.visit_over()):
            return None
        primary, backup = self.services
        if (
            self.active_service is not primary
            or error.processor is not primary
            or not backup.is_usable
        ):
            return None
        # A still-usable service's ordinary error is left alone. A connection
        # failure, or a service that has already given up, is the failover.
        if primary.is_usable and not _connection_failure(error):
            return None
        self._switched = True
        try:
            if self.prepare_switch:
                await self.prepare_switch(error)
            switched = await self._set_active_if_available(backup)
            if not switched:
                self._switched = False
                return None
            if self.complete_switch:
                await self.complete_switch(error)
            return switched
        except BaseException:
            self._switched = False
            raise


class FallbackSTTService(ServiceSwitcher[_SwitchOnFirstUnrecoverableError]):
    def __init__(self, primary: FrameProcessor, backup: FrameProcessor):
        super().__init__(
            services=[primary, backup],
            strategy_type=_SwitchOnFirstUnrecoverableError,
        )
        self.primary = primary
        self.backup = backup
        self._visit_ending = False
        self._campaign_handler_registered = False
        self._fallback_metrics: dict[str, int] = {"switched": 0}
        self.on_switched = None
        # Registered on the task that pushes the primary's error. The run and
        # organization live in this construction-time context.
        self._run_context = contextvars.copy_context()
        self.strategy.prepare_switch = self._prepare_switch
        self.strategy.complete_switch = self._complete_switch
        self.strategy.visit_over = lambda: self._visit_ending
        _stop_primary_reconnects(primary)

        primary_metadata = failure_metadata_for_processor(primary)
        annotate_failure_metadata(
            self,
            source=primary_metadata.source,
            provider=primary_metadata.provider,
            error_owner=primary_metadata.error_owner,
        )

    @property
    def fallback_metrics(self) -> dict:
        return dict(self._fallback_metrics)

    async def _prepare_switch(self, _error: ErrorFrame) -> None:
        buffer = getattr(self.primary, "_reconnect_audio_buffer", None)
        if buffer is not None:
            buffer.clear()
        if hasattr(self.primary, "_reconnecting"):
            self.primary._reconnecting = False
        if hasattr(self.primary, "_disconnecting"):
            self.primary._disconnecting = True
        await self.primary.set_usable(False)

    def _register_backup_campaign_handler(self) -> None:
        if self._campaign_handler_registered:
            return
        from api.services.campaign.circuit_breaker import (
            register_campaign_provider_error_handler,
        )

        metadata = failure_metadata_for_processor(self.backup)
        self._run_context.run(
            register_campaign_provider_error_handler,
            self.backup,
            provider=metadata.provider,
            source=metadata.source or ErrorSource.STT,
        )
        self._campaign_handler_registered = True

    async def _complete_switch(self, error: ErrorFrame) -> None:
        self._fallback_metrics["switched"] += 1
        logger.warning(
            f"STT fallback: {self.primary.name} failed ({error.error}); "
            f"continuing on {self.backup.name}"
        )
        self._register_backup_campaign_handler()
        if self.on_switched:
            await self.on_switched()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        if isinstance(frame, (EndFrame, CancelFrame)):
            self._visit_ending = True
        await super().process_frame(frame, direction)

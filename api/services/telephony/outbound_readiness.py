"""Shared pre-flight checks for every outbound telephony call path."""

from typing import Any

from api.db import db_client as default_db_client
from api.services.telephony import registry
from api.services.telephony.factory import get_setup_checklist


class OutboundReadinessError(ValueError):
    """Base error for a configuration that cannot currently place calls."""


class OutboundConfigurationNotFoundError(OutboundReadinessError):
    """Raised when no requested or usable outbound configuration is available."""


class OutboundSetupIncompleteError(OutboundReadinessError):
    """Raised when a provider's blocking setup steps are incomplete."""

    def __init__(
        self,
        telephony_configuration_id: int,
        configuration_name: str,
        blocked_reason: str,
    ) -> None:
        self.telephony_configuration_id = telephony_configuration_id
        self.configuration_name = configuration_name
        self.blocked_reason = blocked_reason
        super().__init__(
            f"'{configuration_name}' is not ready for outbound calls. {blocked_reason}"
        )


async def _ensure_row_outbound_setup_ready(
    row: Any, *, db: Any, from_phone_number_id: int | None = None
) -> int:
    numbers = await db.list_phone_numbers_for_config(row.id)
    active_numbers = [number for number in numbers if number.is_active]
    spec = registry.get_optional(row.provider)
    if spec is None:
        raise OutboundSetupIncompleteError(
            row.id,
            row.name,
            f"Provider '{row.provider}' is unavailable.",
        )
    if not spec.supports_outbound:
        raise OutboundSetupIncompleteError(
            row.id,
            row.name,
            "This provider receives calls only and cannot place them.",
        )
    trunks = await db.list_trunks_for_config(row.id) if spec.supports_trunks else []
    checklist = get_setup_checklist(
        row.provider,
        row.credentials or {},
        active_phone_number_count=len(active_numbers),
        inbound_routed_phone_number_count=len(
            [number for number in active_numbers if number.inbound_workflow_id]
        ),
        enabled_trunk_count=len([trunk for trunk in trunks if trunk.enabled]),
        unassigned_active_phone_number_count=len(
            [number for number in active_numbers if number.telephony_trunk_id is None]
        ),
    )
    if checklist is not None and not checklist.ready_for_outbound:
        raise OutboundSetupIncompleteError(
            row.id,
            row.name,
            checklist.outbound_blocked_reason
            or "Complete the required telephony setup before placing a call.",
        )
    # A call pinned to an active number of this configuration names its own
    # caller ID; every unpinned path still needs the configuration's default.
    if from_phone_number_id is not None:
        if all(number.id != from_phone_number_id for number in active_numbers):
            raise OutboundSetupIncompleteError(
                row.id, row.name, "The pinned caller ID is not an active number."
            )
    elif not any(number.is_default_caller_id for number in active_numbers):
        raise OutboundSetupIncompleteError(
            row.id, row.name, "It has no default caller ID; it receives calls only."
        )

    return row.id


async def ensure_outbound_setup_ready(
    telephony_configuration_id: int,
    organization_id: int,
    *,
    from_phone_number_id: int | None = None,
    db: Any = default_db_client,
) -> int:
    """Validate the exact, organization-scoped configuration selected by a caller."""
    if not organization_id:
        raise OutboundConfigurationNotFoundError("organization_id is required")

    row = await db.get_telephony_configuration_for_org(
        telephony_configuration_id,
        organization_id,
        active_only=True,
    )
    if row is None:
        raise OutboundConfigurationNotFoundError(
            f"Telephony configuration {telephony_configuration_id} not found "
            f"for organization {organization_id}"
        )
    return await _ensure_row_outbound_setup_ready(
        row, db=db, from_phone_number_id=from_phone_number_id
    )


async def resolve_outbound_configuration_id(
    telephony_configuration_id: int | None,
    organization_id: int,
    *,
    from_phone_number_id: int | None = None,
    db: Any = default_db_client,
) -> int:
    """Resolve an explicit selection, else the org's marked default; never another."""
    if telephony_configuration_id is not None:
        return await ensure_outbound_setup_ready(
            telephony_configuration_id,
            organization_id,
            from_phone_number_id=from_phone_number_id,
            db=db,
        )
    if not organization_id:
        raise OutboundConfigurationNotFoundError("organization_id is required")

    row = await db.get_default_telephony_configuration(organization_id)
    if row is None:
        raise OutboundConfigurationNotFoundError(
            f"Organization {organization_id} has no default outbound telephony "
            "configuration; pass telephony_configuration_id."
        )
    return await _ensure_row_outbound_setup_ready(
        row, db=db, from_phone_number_id=from_phone_number_id
    )


async def requires_e164_destinations(
    telephony_configuration_id: int,
    organization_id: int,
    *,
    db: Any = default_db_client,
) -> bool:
    """Whether this configuration can only dial E.164 numbers.

    Carriers can; a PBX also reaches extensions, SIP URIs and dial strings
    naming a trunk. Callers use it to check destinations against the
    configuration that will actually dial them, rather than against the
    strictest provider. An unknown or missing configuration keeps the strict
    answer, so a destination is never accepted on the strength of a provider
    nothing here recognises.
    """
    row = await db.get_telephony_configuration_for_org(
        telephony_configuration_id,
        organization_id,
    )
    if row is None:
        return True
    spec = registry.get_optional(row.provider)
    return spec.requires_e164_destinations if spec else True


__all__ = [
    "OutboundConfigurationNotFoundError",
    "OutboundReadinessError",
    "OutboundSetupIncompleteError",
    "ensure_outbound_setup_ready",
    "requires_e164_destinations",
    "resolve_outbound_configuration_id",
]

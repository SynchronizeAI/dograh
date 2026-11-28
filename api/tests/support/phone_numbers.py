"""Phone-number rows as the outbound pre-flight reads them."""

from types import SimpleNamespace


def phone_number(number_id=None, *, default=False, active=True):
    return SimpleNamespace(
        id=number_id,
        is_active=active,
        is_default_caller_id=default,
        inbound_workflow_id=None,
        telephony_trunk_id=None,
    )

"""Sync fork: a Twilio inbound refusal answers 503, so Twilio takes the number's VoiceFallbackUrl.

Dograh's 200 `<Say>` + `<Hangup/>` refusal (the concurrent-call limit, a quota refusal, an
unmatched number) bypassed the fallback, so a caller was read the refusal and hung up on.
"""

import pytest

from api.errors.telephony_errors import TelephonyError
from api.services.telephony.providers.twilio.provider import TwilioProvider


@pytest.mark.parametrize(
    "error_type",
    [
        TelephonyError.CONCURRENT_CALL_LIMIT,
        TelephonyError.QUOTA_EXCEEDED,
        TelephonyError.PHONE_NUMBER_NOT_CONFIGURED,
        "NOT_A_REAL_ERROR",
    ],
)
def test_a_refusal_is_an_error_twilio_falls_back_from(error_type):
    response = TwilioProvider.generate_validation_error_response(error_type)
    assert response.status_code == 503
    assert response.media_type == "application/xml"

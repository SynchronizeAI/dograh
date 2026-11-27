"""Sync fork: a 한국AI통신 inbound refusal answers 503, so the vendor takes the number's
voiceFallbackUrl, as a Twilio refusal does (#33).

A 200 `<Reject>` (the concurrent-call limit, a quota refusal, an unmatched number, a failed
signature) is VoiceML the vendor runs, so the caller was refused and the line's voicemail never
answered. The vendor requests the fallback for any non-2xx answer (/docs/build/voiceml, 대체 URL).
"""

import pytest

from api.errors.telephony_errors import TelephonyError
from api.services.telephony.providers.koreaaitel.provider import KoreaAITelProvider


@pytest.mark.parametrize(
    "error_type",
    [
        TelephonyError.CONCURRENT_CALL_LIMIT,
        TelephonyError.QUOTA_EXCEEDED,
        TelephonyError.PHONE_NUMBER_NOT_CONFIGURED,
        TelephonyError.SIGNATURE_VALIDATION_FAILED,
    ],
)
def test_a_refusal_is_an_error_the_vendor_falls_back_from(error_type):
    response = KoreaAITelProvider.generate_validation_error_response(error_type)
    assert response.status_code == 503
    assert response.media_type == "application/xml"
    assert b"<Reject" in response.body

from app.redaction import redact


def test_redacts_valid_card():
    r = redact("Paid with 4111 1111 1111 1111 at the store")
    assert "[REDACTED_CARD]" in r.text
    assert r.counts["card"] == 1


def test_ignores_non_luhn_digits():
    r = redact("Receipt number 1234 5678 9012 3456")
    assert "[REDACTED_CARD]" not in r.text
    assert r.counts.get("card", 0) == 0


def test_redacts_email_ssn_phone():
    r = redact("Contact jane.doe@example.com, SSN 123-45-6789, tel 415-555-1234")
    assert "[REDACTED_EMAIL]" in r.text
    assert "[REDACTED_SSN]" in r.text
    assert "[REDACTED_PHONE]" in r.text
    assert r.total >= 3

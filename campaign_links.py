"""Manual one-recipient test campaign launchers; opening a link never sends automatically."""

from urllib.parse import quote

TEST_NUMBER = "8050702529"
TEST_MESSAGE = (
    "Loan lifecycle pilot test: this is a sample campaign message. "
    "No loan offer is being made and no action is required."
)
VOICE_SCRIPT = (
    "Hello. This is a test call created by Chandan Singh to demonstrate an outbound voice campaign "
    "for Bajaj Finserv personal loans. This is only a pilot test. No loan offer is being made, "
    "and no action is required. Thank you."
)


def launch_links(phone=TEST_NUMBER, message=TEST_MESSAGE):
    if phone != TEST_NUMBER:
        raise ValueError("Manual test links are restricted to the configured test number.")
    e164 = "+91" + phone
    return {
        "whatsapp": f"https://wa.me/91{phone}?text={quote(message, safe='')}",
        "sms": f"sms:{e164}?body={quote(message, safe='')}",
        "call": f"tel:{e164}",
    }

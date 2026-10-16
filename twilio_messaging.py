"""Governed outbound SMS and WhatsApp messaging for the authorized POC number."""
import base64
import json
import re
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from campaign_links import TEST_NUMBER
from twilio_voice import setting


def configuration_status(channel: str) -> list[str]:
    sender_key = "TWILIO_WHATSAPP_FROM" if channel == "WhatsApp" else "TWILIO_FROM_NUMBER"
    required = ["TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN", sender_key]
    return [name for name in required if not setting(name)]


def build_message_request(channel: str, body: str, phone: str = TEST_NUMBER) -> Request:
    if phone != TEST_NUMBER:
        raise ValueError("POC messaging is restricted to the authorized test number.")
    if channel not in {"SMS", "WhatsApp"}:
        raise ValueError("Unsupported Twilio messaging channel.")
    if not body.strip():
        raise ValueError("Campaign message cannot be empty.")
    missing = configuration_status(channel)
    if missing:
        raise ValueError("Missing Twilio configuration: " + ", ".join(missing))
    sid, token = setting("TWILIO_ACCOUNT_SID"), setting("TWILIO_AUTH_TOKEN")
    sender_key = "TWILIO_WHATSAPP_FROM" if channel == "WhatsApp" else "TWILIO_FROM_NUMBER"
    sender = setting(sender_key).removeprefix("whatsapp:")
    if not re.fullmatch(r"AC[0-9a-fA-F]{32}", sid):
        raise ValueError("TWILIO_ACCOUNT_SID is invalid.")
    if not re.fullmatch(r"\+[1-9]\d{7,14}", sender):
        raise ValueError("The configured Twilio sender must use E.164 format.")
    prefix = "whatsapp:" if channel == "WhatsApp" else ""
    fields = {"To": prefix + "+91" + phone, "From": prefix + sender, "Body": body.strip()}
    auth = base64.b64encode(f"{sid}:{token}".encode()).decode("ascii")
    return Request(
        f"https://api.twilio.com/2010-04-01/Accounts/{sid}/Messages.json",
        data=urlencode(fields).encode("utf-8"),
        headers={"Authorization": "Basic " + auth, "Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )


def send_test_message(channel: str, body: str, phone: str = TEST_NUMBER) -> tuple[str | None, str | None]:
    try:
        with urlopen(build_message_request(channel, body, phone), timeout=15) as response:
            result = json.load(response)
    except HTTPError as exc:
        try:
            payload = json.loads(exc.read().decode("utf-8", errors="replace"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            payload = {}
        code, message = payload.get("code", exc.code), payload.get("message", exc.reason)
        if exc.code == 401 or code == 20003:
            raise ValueError("Twilio authentication failed. Configure the Account SID and Auth Token from the same account.") from exc
        raise ValueError(f"Twilio rejected the {channel} message (code {code}): {message}") from exc
    return result.get("sid"), result.get("status")

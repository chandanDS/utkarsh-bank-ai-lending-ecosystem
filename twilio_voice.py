"""Outbound Twilio text-to-speech call for the single authorized POC number."""

import base64
import json
import os
import re
import tomllib
from pathlib import Path
from urllib.parse import urlencode
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from xml.sax.saxutils import escape

from airtel_voice import call_window_open
from campaign_links import TEST_NUMBER, VOICE_SCRIPT

SECRETS_FILE = Path(__file__).parent / ".streamlit" / "secrets.toml"


def setting(name):
    if os.getenv(name):
        return os.environ[name]
    if SECRETS_FILE.exists():
        with SECRETS_FILE.open("rb") as file:
            return tomllib.load(file).get(name, "")
    return ""


def configuration_status():
    required = ("TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN", "TWILIO_FROM_NUMBER")
    return [name for name in required if not setting(name)]


def trial_mode():
    value = setting("TWILIO_TRIAL_MODE")
    return str(value).lower() not in {"false", "0", "no", "off"}


def authorized_test_numbers() -> tuple[str, ...]:
    value = setting("TWILIO_TEST_NUMBERS")
    candidates = value if isinstance(value, (list, tuple)) else str(value or "").split(",")
    numbers = tuple(dict.fromkeys(str(number).strip().removeprefix("+91") for number in candidates if str(number).strip()))
    valid = tuple(number for number in numbers if re.fullmatch(r"[6-9]\d{9}", number))
    return valid or (TEST_NUMBER,)


def normalize_indian_mobile(phone: str) -> str:
    """Validate an Indian E.164 mobile number and return the internal 10-digit form."""
    compact = re.sub(r"[\s()-]", "", str(phone or "").strip())
    if not re.fullmatch(r"\+91[6-9]\d{9}", compact):
        raise ValueError("Enter an Indian mobile number in E.164 format, for example +919876543210.")
    return compact[3:]


def is_authorized_test_number(phone: str, authorized_numbers=None) -> bool:
    allowed = authorized_test_numbers() if authorized_numbers is None else tuple(authorized_numbers)
    return phone in allowed


def build_call_request(phone=TEST_NUMBER, authorized_numbers=None):
    if not is_authorized_test_number(phone, authorized_numbers):
        raise ValueError("POC calls are restricted to the authorized test-number allowlist.")
    missing = configuration_status()
    if missing:
        raise ValueError("Missing Twilio configuration: " + ", ".join(missing))
    sid = setting("TWILIO_ACCOUNT_SID")
    token = setting("TWILIO_AUTH_TOKEN")
    caller = setting("TWILIO_FROM_NUMBER")
    if not re.fullmatch(r"AC[0-9a-fA-F]{32}", sid):
        raise ValueError("TWILIO_ACCOUNT_SID is invalid.")
    if not re.fullmatch(r"\+[1-9]\d{7,14}", caller):
        raise ValueError("TWILIO_FROM_NUMBER must be an E.164 number.")
    fields = {"To": "+91" + phone, "From": caller}
    if trial_mode():
        fields["Url"] = "https://webhooks.twilio.com/v1/Voice/Template/voice_text_to_speech"
    else:
        fields["Twiml"] = f'<Response><Say language="en-IN">{escape(VOICE_SCRIPT)}</Say><Hangup/></Response>'
    data = urlencode(fields).encode("utf-8")
    auth = base64.b64encode(f"{sid}:{token}".encode("utf-8")).decode("ascii")
    url = f"https://api.twilio.com/2010-04-01/Accounts/{sid}/Calls.json"
    return Request(url, data=data, headers={"Authorization": "Basic " + auth,
                                             "Content-Type": "application/x-www-form-urlencoded"}, method="POST")


def place_test_call(phone=TEST_NUMBER, authorized_numbers=None):
    if not call_window_open():
        raise ValueError("The configured campaign calling window is closed in India Standard Time.")
    request = build_call_request(phone, authorized_numbers)
    try:
        with urlopen(request, timeout=15) as response:
            result = json.load(response)
    except HTTPError as exc:
        try:
            payload = json.loads(exc.read().decode("utf-8", errors="replace"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            payload = {}
        code = payload.get("code", exc.code)
        message = payload.get("message", exc.reason)
        if exc.code == 401 or code == 20003:
            raise ValueError(
                "Twilio authentication failed (code 20003). The Auth Token does not match "
                "the configured Account SID. Update both values from the same Twilio account."
            ) from exc
        raise ValueError(f"Twilio rejected the call (code {code}): {message}") from exc
    return result.get("sid"), result.get("status")

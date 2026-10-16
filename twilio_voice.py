"""Outbound Twilio text-to-speech call for the single authorized POC number."""

import base64
import json
import os
import re
import tomllib
from pathlib import Path
from urllib.parse import urlencode
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


def build_call_request(phone=TEST_NUMBER):
    if phone != TEST_NUMBER:
        raise ValueError("POC calls are restricted to the authorized test number.")
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


def place_test_call(phone=TEST_NUMBER):
    if not call_window_open():
        raise ValueError("The configured campaign calling window is closed in India Standard Time.")
    request = build_call_request(phone)
    with urlopen(request, timeout=15) as response:
        result = json.load(response)
    return result.get("sid"), result.get("status")
"""Single-recipient Exotel outbound call for the authorized pilot number."""

import base64
import os
import re
import tomllib
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from airtel_voice import call_window_open
from campaign_links import TEST_NUMBER

SECRETS_FILE = Path(__file__).parent / ".streamlit" / "exotel_secrets.toml"
REQUIRED = ("EXOTEL_ACCOUNT_SID", "EXOTEL_API_KEY", "EXOTEL_API_TOKEN", "EXOTEL_EXOPHONE", "EXOTEL_FLOW_URL", "EXOTEL_CALL_TYPE")


def setting(name):
    value = os.getenv(name)
    if value:
        return value
    if SECRETS_FILE.exists():
        with SECRETS_FILE.open("rb") as source:
            return tomllib.load(source).get(name, "")
    return ""


def configuration_status():
    return [name for name in REQUIRED if not setting(name) or "your-" in str(setting(name)) or "confirm-" in str(setting(name))]


def build_call_request(phone=TEST_NUMBER):
    if phone != TEST_NUMBER:
        raise ValueError("POC calls are restricted to the authorized test number.")
    missing = configuration_status()
    if missing:
        raise ValueError("Missing Exotel configuration: " + ", ".join(missing))
    sid = setting("EXOTEL_ACCOUNT_SID")
    host = setting("EXOTEL_API_HOST") or "api.in.exotel.com"
    flow_url = setting("EXOTEL_FLOW_URL")
    exophone = setting("EXOTEL_EXOPHONE")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", sid):
        raise ValueError("EXOTEL_ACCOUNT_SID is invalid.")
    if host not in {"api.in.exotel.com", "api.exotel.com", "api.exotel.com.sg"}:
        raise ValueError("EXOTEL_API_HOST is not an approved Exotel API host.")
    if not (flow_url.startswith("https://") or flow_url.startswith("http://my.exotel.in/exoml/start/") or flow_url.startswith("http://my.exotel.com/")):
        raise ValueError("EXOTEL_FLOW_URL must be the Exotel App Bazaar flow URL.")
    if not re.fullmatch(r"\+?\d{10,15}", exophone):
        raise ValueError("EXOTEL_EXOPHONE must be a telephone number.")
    fields = {"From": "0" + phone, "CallerId": exophone, "Url": flow_url, "CallType": setting("EXOTEL_CALL_TYPE")}
    credential = f"{setting('EXOTEL_API_KEY')}:{setting('EXOTEL_API_TOKEN')}"
    auth = base64.b64encode(credential.encode()).decode("ascii")
    return Request(
        f"https://{host}/v1/accounts/{sid}/calls/connect",
        data=urlencode(fields).encode(),
        headers={"Authorization": "Basic " + auth, "Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )


def place_test_call(phone=TEST_NUMBER):
    if not call_window_open():
        raise ValueError("The configured campaign calling window is closed in India Standard Time.")
    with urlopen(build_call_request(phone), timeout=15) as response:
        body = response.read()
    root = ET.fromstring(body)
    sid = next((element.text for element in root.iter() if element.tag.rsplit("}", 1)[-1] == "Sid"), None)
    status = next((element.text for element in root.iter() if element.tag.rsplit("}", 1)[-1] == "Status"), None)
    return sid, status

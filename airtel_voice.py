"""Configurable Airtel IQ outbound-call adapter.

Airtel's account-specific request schema must be supplied as a JSON template.
This module never guesses provider field names or stores credentials in code.
"""

import json
import os
from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse
from urllib.request import Request, urlopen

TEST_NUMBER = "8050702529"
IST = timezone(timedelta(hours=5, minutes=30))


def call_window_open(now=None):
    now = now or datetime.now(IST)
    start = int(os.getenv("CAMPAIGN_START_HOUR", "9"))
    end = int(os.getenv("CAMPAIGN_END_HOUR", "19"))
    return 0 <= start < end <= 24 and start <= now.hour < end


def configuration_status():
    required = ["AIRTEL_VOICE_ENDPOINT", "AIRTEL_VOICE_AUTH_VALUE", "AIRTEL_VOICE_PAYLOAD_TEMPLATE"]
    return [key for key in required if not os.getenv(key)]


def build_request(phone=TEST_NUMBER):
    if phone != TEST_NUMBER:
        raise ValueError("Live test calls are restricted to the configured test number.")
    missing = configuration_status()
    if missing:
        raise ValueError("Missing configuration: " + ", ".join(missing))
    endpoint = os.environ["AIRTEL_VOICE_ENDPOINT"]
    parsed = urlparse(endpoint)
    if parsed.scheme != "https" or not parsed.hostname:
        raise ValueError("Airtel voice endpoint must be an HTTPS URL.")
    template = os.environ["AIRTEL_VOICE_PAYLOAD_TEMPLATE"]
    if "{{phone}}" not in template:
        raise ValueError("Payload template must contain {{phone}}.")
    payload = json.loads(template.replace("{{phone}}", "+91" + phone))
    if not isinstance(payload, dict):
        raise ValueError("Payload template must be a JSON object.")
    header_name = os.getenv("AIRTEL_VOICE_AUTH_HEADER", "Authorization")
    if header_name.lower() in {"host", "content-length", "content-type"}:
        raise ValueError("Invalid authentication header name.")
    headers = {"Content-Type": "application/json", header_name: os.environ["AIRTEL_VOICE_AUTH_VALUE"]}
    return endpoint, headers, payload


def place_test_call(phone=TEST_NUMBER):
    if not call_window_open():
        raise ValueError("The configured campaign calling window is closed in India Standard Time.")
    endpoint, headers, payload = build_request(phone)
    request = Request(endpoint, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
    with urlopen(request, timeout=15) as response:
        body = response.read(4096).decode("utf-8", errors="replace")
        return response.status, body

import os
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.parse import parse_qs

from twilio_voice import build_call_request, configuration_status


class TwilioVoiceTests(unittest.TestCase):
    def test_missing_credentials_block_call(self):
        with patch("twilio_voice.SECRETS_FILE", Path("__missing_secrets__.toml")), patch.dict(os.environ, {}, clear=True):
            self.assertEqual(len(configuration_status()), 3)
            with self.assertRaises(ValueError):
                build_call_request()

    def test_trial_call_is_restricted_to_one_number(self):
        config = {
            "TWILIO_ACCOUNT_SID": "AC" + "a" * 32,
            "TWILIO_AUTH_TOKEN": "test-token",
            "TWILIO_FROM_NUMBER": "+14155550100",
        }
        with patch.dict(os.environ, config, clear=True):
            request = build_call_request()
            fields = parse_qs(request.data.decode("utf-8"))
            self.assertEqual(fields["To"], ["+918050702529"])
            self.assertEqual(fields["Url"], ["https://webhooks.twilio.com/v1/Voice/Template/voice_text_to_speech"])
            self.assertNotIn("Twiml", fields)
            with self.assertRaises(ValueError):
                build_call_request("9999999999")

    def test_custom_script_after_trial(self):
        config = {
            "TWILIO_ACCOUNT_SID": "AC" + "a" * 32,
            "TWILIO_AUTH_TOKEN": "test-token",
            "TWILIO_FROM_NUMBER": "+14155550100",
            "TWILIO_TRIAL_MODE": "false",
        }
        with patch.dict(os.environ, config, clear=True):
            fields = parse_qs(build_call_request().data.decode("utf-8"))
            self.assertIn("Bajaj Finserv personal loan", fields["Twiml"][0])
            self.assertIn("Chandan Singh", fields["Twiml"][0])
            self.assertNotIn("link", fields["Twiml"][0].lower())
            self.assertNotIn("Url", fields)


if __name__ == "__main__":
    unittest.main()
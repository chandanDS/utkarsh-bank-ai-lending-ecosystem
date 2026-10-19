import os
import unittest
from urllib.error import HTTPError
from pathlib import Path
from unittest.mock import MagicMock, patch
from urllib.parse import parse_qs

from campaign_links import TEST_NUMBER
from twilio_voice import authorized_test_numbers, build_call_request, configuration_status, place_test_call


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
            self.assertEqual(fields["To"], ["+91" + TEST_NUMBER])
            self.assertEqual(fields["Url"], ["https://webhooks.twilio.com/v1/Voice/Template/voice_text_to_speech"])
            self.assertNotIn("Twiml", fields)
            with self.assertRaises(ValueError):
                build_call_request("9999999999")

    def test_authentication_error_is_actionable_and_does_not_expose_token(self):
        error = HTTPError("https://api.twilio.test", 401, "Unauthorized", {}, None)
        error.read = MagicMock(return_value=b'{"code": 20003, "message": "Authentication Error"}')
        environment = {
            "TWILIO_ACCOUNT_SID": "AC" + "a" * 32,
            "TWILIO_AUTH_TOKEN": "secret-token-value",
            "TWILIO_FROM_NUMBER": "+14155550100",
        }
        with patch.dict(os.environ, environment, clear=True), patch(
            "twilio_voice.call_window_open", return_value=True
        ), patch("twilio_voice.urlopen", side_effect=error):
            with self.assertRaisesRegex(ValueError, "does not match the configured Account SID") as raised:
                place_test_call()
        self.assertNotIn(environment["TWILIO_AUTH_TOKEN"], str(raised.exception))

    def test_custom_script_after_trial(self):
        config = {
            "TWILIO_ACCOUNT_SID": "AC" + "a" * 32,
            "TWILIO_AUTH_TOKEN": "test-token",
            "TWILIO_FROM_NUMBER": "+14155550100",
            "TWILIO_TRIAL_MODE": "false",
        }
        with patch.dict(os.environ, config, clear=True):
            fields = parse_qs(build_call_request().data.decode("utf-8"))
            self.assertIn("Utkarsh Small Finance Bank personal loan", fields["Twiml"][0])
            self.assertIn("Chandan Singh", fields["Twiml"][0])
            self.assertNotIn("link", fields["Twiml"][0].lower())
            self.assertNotIn("Url", fields)

    def test_multiple_authorized_numbers(self):
        config = {"TWILIO_TEST_NUMBERS": "9876543210,8765432109"}
        with patch.dict(os.environ, config, clear=True):
            self.assertEqual(authorized_test_numbers(), ("9876543210", "8765432109"))

    def test_session_authorized_number_can_build_voice_call(self):
        config = {
            "TWILIO_ACCOUNT_SID": "AC" + "a" * 32,
            "TWILIO_AUTH_TOKEN": "test-token",
            "TWILIO_FROM_NUMBER": "+14155550100",
        }
        dynamic_number = "9876543210"
        with patch.dict(os.environ, config, clear=True):
            request = build_call_request(dynamic_number, authorized_numbers=(dynamic_number,))
            fields = parse_qs(request.data.decode("utf-8"))
            self.assertEqual(fields["To"], ["+91" + dynamic_number])


if __name__ == "__main__":
    unittest.main()

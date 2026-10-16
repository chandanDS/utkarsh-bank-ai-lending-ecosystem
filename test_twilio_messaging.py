import unittest
from unittest.mock import patch
from urllib.parse import parse_qs

from campaign_links import TEST_NUMBER
from twilio_messaging import build_message_request, configuration_status


class TwilioMessagingTests(unittest.TestCase):
    settings = {
        "TWILIO_ACCOUNT_SID": "AC" + "1" * 32,
        "TWILIO_AUTH_TOKEN": "token",
        "TWILIO_FROM_NUMBER": "+15005550006",
        "TWILIO_WHATSAPP_FROM": "+14155238886",
        "TWILIO_WHATSAPP_CONTENT_SID": "HX" + "2" * 32,
    }

    def setting(self, name):
        return self.settings.get(name, "")

    @patch("twilio_messaging.setting")
    def test_sms_request_uses_sms_sender(self, mocked_setting):
        mocked_setting.side_effect = self.setting
        request = build_message_request("SMS", "Test message", TEST_NUMBER)
        fields = parse_qs(request.data.decode())
        self.assertEqual(fields["From"], ["+15005550006"])
        self.assertEqual(fields["To"], ["+91" + TEST_NUMBER])

    @patch("twilio_messaging.setting")
    def test_whatsapp_request_uses_channel_prefix(self, mocked_setting):
        mocked_setting.side_effect = self.setting
        request = build_message_request("WhatsApp", "Test message", TEST_NUMBER)
        fields = parse_qs(request.data.decode())
        self.assertEqual(fields["From"], ["whatsapp:+14155238886"])
        self.assertEqual(fields["To"], ["whatsapp:+91" + TEST_NUMBER])
        self.assertEqual(fields["ContentSid"], ["HX" + "2" * 32])
        self.assertNotIn("Body", fields)

    @patch("twilio_messaging.setting", return_value="")
    def test_missing_configuration_disables_channel(self, _mocked_setting):
        self.assertIn("TWILIO_WHATSAPP_FROM", configuration_status("WhatsApp"))
        self.assertIn("TWILIO_WHATSAPP_CONTENT_SID", configuration_status("WhatsApp"))

    def test_other_recipient_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "authorized test-number allowlist"):
            build_message_request("SMS", "Test message", "9999999999")


if __name__ == "__main__":
    unittest.main()

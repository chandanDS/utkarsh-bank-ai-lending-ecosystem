import json
import os
import unittest
from unittest.mock import patch

from airtel_voice import TEST_NUMBER, build_request, configuration_status


class AirtelVoiceTests(unittest.TestCase):
    def test_missing_configuration_blocks_call(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(len(configuration_status()), 3)
            with self.assertRaises(ValueError):
                build_request()

    def test_fixed_number_and_template(self):
        config = {
            "AIRTEL_VOICE_ENDPOINT": "https://voice.example.test/outbound",
            "AIRTEL_VOICE_AUTH_VALUE": "Bearer test-token",
            "AIRTEL_VOICE_PAYLOAD_TEMPLATE": json.dumps({"recipient": "{{phone}}", "flow": "approved-flow"}),
        }
        with patch.dict(os.environ, config, clear=True):
            endpoint, headers, payload = build_request(TEST_NUMBER)
            self.assertEqual(endpoint, config["AIRTEL_VOICE_ENDPOINT"])
            self.assertEqual(headers["Authorization"], config["AIRTEL_VOICE_AUTH_VALUE"])
            self.assertEqual(payload["recipient"], "+918050702529")
            with self.assertRaises(ValueError):
                build_request("9999999999")

    def test_insecure_endpoint_blocked(self):
        config = {
            "AIRTEL_VOICE_ENDPOINT": "http://voice.example.test/outbound",
            "AIRTEL_VOICE_AUTH_VALUE": "Bearer test-token",
            "AIRTEL_VOICE_PAYLOAD_TEMPLATE": '{"recipient":"{{phone}}"}',
        }
        with patch.dict(os.environ, config, clear=True):
            with self.assertRaises(ValueError):
                build_request()


if __name__ == "__main__":
    unittest.main()

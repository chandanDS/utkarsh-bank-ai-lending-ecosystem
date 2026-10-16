import unittest
from campaign_links import TEST_MESSAGE, launch_links


class CampaignLinkTests(unittest.TestCase):
    def test_default_test_links(self):
        links = launch_links()
        self.assertIn("918050702529", links["whatsapp"])
        self.assertIn("Loan%20lifecycle%20pilot%20test", links["whatsapp"])
        self.assertEqual(links["call"], "tel:+918050702529")
        self.assertTrue(links["sms"].startswith("sms:+918050702529?body="))
        self.assertIn("No loan offer", TEST_MESSAGE)

    def test_other_number_rejected(self):
        with self.assertRaises(ValueError):
            launch_links("9999999999")


if __name__ == "__main__":
    unittest.main()

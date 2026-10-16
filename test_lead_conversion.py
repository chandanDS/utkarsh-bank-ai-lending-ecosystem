import unittest

import pandas as pd

from lending_analytics import lead_conversion_analytics


class LeadConversionAnalyticsTests(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame([
            {"customer_id": "C00001", "consent": 1, "propensity": .80, "risk": .20, "kyc": 1, "segment": "pre-approved", "response": 1, "product": "Personal Loan", "amount": 100_000},
            {"customer_id": "C00002", "consent": 1, "propensity": .75, "risk": .30, "kyc": 1, "segment": "pre-qualified", "response": 0, "product": "Home Loan", "amount": 500_000},
            {"customer_id": "C00090", "consent": 1, "propensity": .82, "risk": .25, "kyc": 1, "segment": "pre-approved", "response": 1, "product": "Personal Loan", "amount": 150_000},
            {"customer_id": "C00003", "consent": 0, "propensity": .90, "risk": .10, "kyc": 1, "segment": "pre-approved", "response": 1, "product": "MSME Loan", "amount": 300_000},
        ])

    def test_funnel_is_monotonic_and_reproducible(self):
        stages, waterfall, products, potential = lead_conversion_analytics(self.df)
        counts = stages["Accounts"].tolist()
        self.assertEqual(counts, [3, 2, 2, 1])
        self.assertEqual(counts, sorted(counts, reverse=True))
        self.assertEqual(len(waterfall), 5)
        self.assertEqual(round(float(products["Disbursal ₹ Cr"].sum()), 2), 0.01)
        self.assertEqual(potential, 100_000)

    def test_empty_authorized_scope_is_safe(self):
        stages, waterfall, products, potential = lead_conversion_analytics(self.df.iloc[0:0])
        self.assertEqual(int(stages["Accounts"].sum()), 0)
        self.assertEqual(float(waterfall["Value"].sum()), 0.0)
        self.assertEqual(potential, 0.0)


if __name__ == "__main__":
    unittest.main()

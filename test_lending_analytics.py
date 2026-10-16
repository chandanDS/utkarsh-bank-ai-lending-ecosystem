import unittest

import pandas as pd

from lending_analytics import answer_bi, classify_next_best_action, enrich_portfolio


class ConversationalBiTests(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame([
            {"customer_id": "1", "branch_code": "BR001", "city": "Mumbai", "risk": .82, "dpd": 60, "outstanding": 2_000_000, "collection_amount_due": 100, "collection_amount_received": 70, "segment": "pre-approved", "churn_probability": .70, "cross_sell_score": .80, "propensity": .85, "product": "Personal Loan"},
            {"customer_id": "2", "branch_code": "BR001", "city": "Mumbai", "risk": .72, "dpd": 30, "outstanding": 1_000_000, "collection_amount_due": 100, "collection_amount_received": 60, "segment": "manual-review", "churn_probability": .75, "cross_sell_score": .75, "propensity": .80, "product": "Home Loan"},
            {"customer_id": "3", "branch_code": "BR002", "city": "Delhi", "risk": .25, "dpd": 0, "outstanding": 5_000_000, "collection_amount_due": 100, "collection_amount_received": 95, "segment": "pre-qualified", "churn_probability": .20, "cross_sell_score": .50, "propensity": .40, "product": "Home Loan"},
        ])

    def assert_branch_answer(self, question, branch, route):
        answer, result, actual_route = answer_bi(question, self.df)
        self.assertIn(branch, answer)
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["branch_code"], branch)
        self.assertEqual(actual_route, route)

    def test_highest_defaulters_returns_one_specific_branch(self):
        self.assert_branch_answer(
            "which branch has highest number of defaulters amongst the bank as whole",
            "BR001", "branch_default_risk",
        )

    def test_highest_delinquency(self):
        self.assert_branch_answer("which branch has highest delinquency", "BR001", "branch_delinquency")

    def test_highest_outstanding(self):
        self.assert_branch_answer("which branch has highest outstanding portfolio", "BR002", "branch_outstanding")

    def test_lowest_collection_efficiency(self):
        self.assert_branch_answer("branch with lowest collection efficiency", "BR001", "branch_collection_efficiency")

    def test_lowest_high_propensity_leads_by_portfolio(self):
        portfolio = self.df.copy()
        portfolio.loc[portfolio["customer_id"] == "2", "propensity"] = 0.70
        answer, result, route = answer_bi(
            "which portfolio has the lowest high propensity leads", portfolio
        )
        self.assertEqual(route, "portfolio_propensity")
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["product"], "Home Loan")
        self.assertEqual(int(result.iloc[0]["high_propensity_leads"]), 0)
        self.assertIn("Home Loan", answer)
        self.assertIn("lowest number of high-propensity leads: 0", answer)

    def test_highest_high_propensity_leads_by_product(self):
        portfolio = self.df.copy()
        portfolio.loc[portfolio["customer_id"] == "2", "propensity"] = 0.70
        answer, result, route = answer_bi(
            "which product has the highest high propensity leads", portfolio
        )
        self.assertEqual(route, "portfolio_propensity")
        self.assertEqual(result.iloc[0]["product"], "Personal Loan")
        self.assertEqual(int(result.iloc[0]["high_propensity_leads"]), 1)
        self.assertIn("Personal Loan", answer)

    def test_next_best_action_business_categories(self):
        self.assertEqual(classify_next_best_action("Proactively contact customer and troubleshoot UPI access"), "Attention")
        self.assertEqual(classify_next_best_action("Share transaction status and initiate payment trace"), "Attention")
        self.assertEqual(classify_next_best_action("Explain charge basis and route waiver decision if eligible"), "Decision")
        self.assertEqual(classify_next_best_action("Offer Home Loan"), "Offer")
        self.assertEqual(classify_next_best_action("Confirm account information"), "Information")

    def test_generic_high_risk_still_returns_customer_detail(self):
        answer, result, route = answer_bi("show high default risk customers", self.df)
        self.assertEqual(len(result), 2)
        self.assertEqual(route, "default_risk")
        self.assertIn("2 customers", answer)

    def test_forward_default_prediction_includes_standard_customer(self):
        source = pd.DataFrame([{
            "customer_id": "C00017", "outstanding": 1_000_000,
            "collection_amount_due": 0, "collection_amount_received": 0,
            "risk": .70, "response": 0, "dpd": 0, "aa_cashflow": .60,
            "propensity": .55, "bureau": 620, "utilization": .90,
            "delinquencies": 0, "current_emi": 20_000, "income": 50_000,
            "segment": "underwriting", "product": "Personal Loan",
            "customer_segment": "Salaried",
        }])
        enriched = enrich_portfolio(source).iloc[0]
        self.assertEqual(enriched["current_account_status"], "Standard")
        self.assertEqual(int(enriched["dpd"]), 0)
        self.assertGreaterEqual(enriched["default_probability_90d"], .60)
        self.assertGreaterEqual(int(enriched["delinquency_events_12m"]), 0)


if __name__ == "__main__":
    unittest.main()

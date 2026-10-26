import unittest

import pandas as pd

from lending_analytics import answer_action_recommendation, answer_bi, answer_default_driver_analysis, classify_next_best_action, enrich_portfolio


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
            "BR001", "branch_defaulters",
        )

    def test_defaulters_are_current_dpd_above_30_not_predictive_risk(self):
        answer, result, route = answer_bi("which branch has the highest defaulters", self.df)
        self.assertEqual(route, "branch_defaulters")
        self.assertEqual(int(result.iloc[0]["defaulters"]), 1)
        self.assertIn("DPD > 30", answer)

    def test_highest_default_risk_remains_predictive(self):
        answer, result, route = answer_bi("which branch has the highest default risk", self.df)
        self.assertEqual(route, "branch_default_risk")
        self.assertEqual(int(result.iloc[0]["high_default_risk_customers"]), 2)
        self.assertIn("high default-risk", answer)

    def test_highest_branch_default_rate_uses_current_dpd(self):
        answer, result, route = answer_bi("which branch has the highest default rate", self.df)
        self.assertEqual(route, "branch_default_rate")
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["branch_code"], "BR001")
        self.assertEqual(result.iloc[0]["default_rate_pct"], 50.0)
        self.assertIn("50.00%", answer)
        self.assertIn("1 of 2 accounts with DPD > 30", answer)

    def test_highest_delinquency(self):
        self.assert_branch_answer("which branch has highest delinquency", "BR001", "branch_delinquency")

    def test_highest_outstanding(self):
        self.assert_branch_answer("which branch has highest outstanding portfolio", "BR002", "branch_outstanding")

    def test_mumbai_personal_loan_portfolio_applies_product_city_and_fy_scope(self):
        answer, result, route = answer_bi(
            "what is the personal loan portfolio for mumbai branch in this FY", self.df
        )
        self.assertEqual(route, "portfolio")
        self.assertIn("₹0.20 crore", answer)
        self.assertIn("Personal Loan", answer)
        self.assertIn("BR001", answer)
        self.assertIn("FY ", answer)
        self.assertEqual(result["product"].tolist(), ["Personal Loan"])

    def test_total_business_uses_outstanding_not_requested_or_offer_amount(self):
        portfolio = self.df.copy()
        portfolio["requested_amount"] = [99_000_000, 88_000_000, 77_000_000]
        portfolio["amount"] = [9_000_000, 8_000_000, 7_000_000]
        answer, _, route = answer_bi("what is total business for bank as a whole", portfolio)
        self.assertEqual(route, "portfolio")
        self.assertIn("₹0.80 crore", answer)

    def test_branch_contribution_is_product_specific(self):
        answer, result, route = answer_bi("which branch contributes most to personal loan", self.df)
        self.assertEqual(route, "branch_product_contribution")
        self.assertEqual(result.iloc[0]["branch_code"], "BR001")
        self.assertIn("Personal Loan", answer)

    def test_average_risk_by_product_is_reported_as_percent(self):
        answer, result, route = answer_bi("compare average risk by product", self.df)
        self.assertEqual(route, "product_risk_comparison")
        self.assertIn("Average risk %", result.columns)
        self.assertNotIn("average_risk", result.columns)
        self.assertIn("%", answer)

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

    def test_delhi_default_early_warning_returns_prediction_rows(self):
        source = pd.DataFrame([
            {
                "customer_id": "C1", "name": "Aarav Sharma", "branch_code": "BR002",
                "city": "Delhi", "product": "Personal Loan", "customer_segment": "Salaried",
                "current_account_status": "Standard", "default_probability_90d": .71,
                "bureau": 680, "risk": .65, "dpd": 0, "delinquency_events_12m": 0,
                "utilization": .84, "aa_cashflow": .72, "outstanding": 900_000,
            },
            {
                "customer_id": "C2", "name": "Diya Patel", "branch_code": "BR001",
                "city": "Mumbai", "product": "Home Loan", "customer_segment": "Salaried",
                "current_account_status": "Standard", "default_probability_90d": .80,
                "bureau": 650, "risk": .72, "dpd": 0, "delinquency_events_12m": 1,
                "utilization": .90, "aa_cashflow": .68, "outstanding": 2_000_000,
            },
        ])
        answer, result, route = answer_bi(
            "list down the data from default and early warning for Delhi as a branch", source
        )
        self.assertEqual(route, "ai_ml_output:default_early_warning")
        self.assertEqual(result["branch_code"].unique().tolist(), ["BR002"])
        self.assertTrue((result["default_probability_90d"] >= .45).all())
        self.assertIn("Delhi (BR002)", answer)

    def test_kochi_propensity_leads_returns_scoped_prediction_rows(self):
        portfolio = self.df.copy()
        portfolio.loc[portfolio["customer_id"] == "3", ["branch_code", "city", "propensity"]] = [
            "BR011", "Kochi", .82
        ]
        answer, result, route = answer_bi("what are the propensity leads for kochi branch", portfolio)
        self.assertEqual(route, "ai_ml_output:propensity")
        self.assertFalse(result.empty)
        self.assertTrue((result["branch_code"] == "BR011").all())
        self.assertTrue((result["propensity"] >= .60).all())
        self.assertIn("Kochi (BR011)", answer)

    def test_kochi_credit_decisioning_leads_returns_scoped_prediction_rows(self):
        portfolio = self.df.copy()
        portfolio.loc[portfolio["customer_id"] == "3", ["branch_code", "city"]] = ["BR011", "Kochi"]
        answer, result, route = answer_bi("what are the credit decisioning leads for kochi branch", portfolio)
        self.assertEqual(route, "ai_ml_output:credit_decisioning")
        self.assertFalse(result.empty)
        self.assertTrue((result["branch_code"] == "BR011").all())
        self.assertIn("Kochi (BR011)", answer)

    def test_prediction_output_cannot_broaden_existing_rbac_scope(self):
        scoped = pd.DataFrame([{
            "customer_id": "C2", "name": "Diya Patel", "branch_code": "BR001",
            "city": "Mumbai", "default_probability_90d": .80, "risk": .72,
        }])
        answer, result, route = answer_bi(
            "show default and early warning records for Delhi branch", scoped
        )
        self.assertEqual(route, "ai_ml_output:default_early_warning")
        self.assertTrue(result.empty)
        self.assertIn("No authorized", answer)

    def test_collection_recommendation_returns_customer_action_queue(self):
        portfolio = self.df.copy()
        portfolio["collection_priority"] = [.90, .55, .10]
        portfolio["name"] = ["A", "B", "C"]
        portfolio["preferred_channel"] = "SMS"
        portfolio["contact_hour"] = 11
        answer, result, route = answer_action_recommendation(
            "recommend how to improve collection efficiency", portfolio
        )
        self.assertEqual(route, "action_recommendation:collection_efficiency")
        self.assertEqual(len(result), 2)
        self.assertTrue((result["dpd"] > 0).all())
        self.assertIn("AI insight", answer)

    def test_default_reason_question_returns_scoped_observable_drivers(self):
        source = pd.DataFrame([
            {"customer_id": "D1", "branch_code": "BR002", "city": "Delhi", "default_probability_90d": .72, "risk": .70, "utilization": .90, "bureau": 620, "aa_cashflow": .70, "delinquency_events_12m": 1, "dpd": 0},
            {"customer_id": "D2", "branch_code": "BR002", "city": "Delhi", "default_probability_90d": .55, "risk": .50, "utilization": .82, "bureau": 700, "aa_cashflow": .95, "delinquency_events_12m": 0, "dpd": 45},
            {"customer_id": "M1", "branch_code": "BR001", "city": "Mumbai", "default_probability_90d": .90, "risk": .80, "utilization": .95, "bureau": 590, "aa_cashflow": .60, "delinquency_events_12m": 2, "dpd": 60},
        ])
        answer, result, route = answer_default_driver_analysis(
            "what are the reasons for highest default rate in Delhi?", source
        )
        self.assertEqual(route, "default_driver_analysis")
        self.assertIn("Delhi (BR002)", answer)
        self.assertIn("not proven causal", answer)
        self.assertEqual(int(result.iloc[0]["Affected customers"]), 2)


if __name__ == "__main__":
    unittest.main()

import unittest

import pandas as pd

from historical_analytics import answer_historical_bi, build_historical_portfolio, observe_metrics


class HistoricalAnalyticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        records = []
        products = ["Personal Loan", "Home Loan"]
        branches = [
            ("BR001", "Mumbai", "Maharashtra"),
            ("BR002", "Delhi", "Delhi"),
        ]
        for index in range(80):
            branch_code, city, state = branches[index % 2]
            records.append({
                "customer_id": f"CUST{index + 1:05d}",
                "branch_code": branch_code,
                "city": city,
                "state": state,
                "product": products[index % 2],
                "customer_segment": "Salaried" if index % 3 else "Self-employed",
                "outstanding": 250_000 + index * 10_000,
                "consent": 1,
                "propensity": 0.78 if index % 5 else 0.55,
                "risk": 0.30 if index % 7 else 0.72,
                "kyc": 1,
                "segment": "pre-approved" if index % 4 else "underwriting",
                "response": 1 if index % 3 else 0,
                "dpd": 60 if index % 11 == 0 else (15 if index % 5 == 0 else 0),
                "churn_probability": 0.70 if index % 9 == 0 else 0.20,
                "cross_sell_score": 0.80 if index % 4 else 0.40,
                "collection_amount_due": 12_000 + index * 100,
                "collection_amount_received": 10_000 + index * 80,
            })
        cls.portfolio = pd.DataFrame(records)
        cls.history = build_historical_portfolio(cls.portfolio, months=36)

    def test_latest_month_reconciles_to_current_snapshot(self):
        latest = self.history[self.history["month"] == self.history["month"].max()]
        self.assertAlmostEqual(latest["portfolio_amount"].sum(), self.portfolio["outstanding"].sum(), places=2)
        self.assertEqual(int(latest["account_count"].sum()), len(self.portfolio))

    def test_ten_management_question_scenarios(self):
        questions = [
            "How has the Personal Loan portfolio grown compared to last year?",
            "Show new accounts added month on month by product",
            "Compare portfolio growth quarter on quarter",
            "Which branch has the highest increase in default rate YoY?",
            "Show customer attrition compared with last year",
            "Compare lead conversion QoQ",
            "Show cross-sell conversion YoY by customer segment",
            "Show collection efficiency trend",
            "Which product has the fastest portfolio growth?",
            "Which region has the lowest portfolio growth?",
        ]
        for question in questions:
            with self.subTest(question=question):
                response = answer_historical_bi(question, self.history)
                self.assertIsNotNone(response)
                answer, result, route = response
                self.assertTrue(answer)
                self.assertFalse(result.empty)
                self.assertTrue(route.startswith("historical:"))

    def test_portfolio_health_answer_combines_growth_accounts_and_risk(self):
        answer, result, route = answer_historical_bi(
            "How is the Personal Loan portfolio performing?",
            self.history,
        )
        self.assertEqual(route, "historical:portfolio_health:yoy")
        self.assertEqual(set(result["Measure"]), {"Portfolio", "New accounts", "Default rate"})
        self.assertIn("Recommended action", answer)

    def test_branch_scope_is_preserved_by_history_builder(self):
        branch = self.portfolio[self.portfolio["branch_code"] == "BR001"]
        scoped = build_historical_portfolio(branch, months=36)
        self.assertEqual(set(scoped["branch_code"]), {"BR001"})

    def test_stressed_demo_outcomes_create_actionable_pain_points(self):
        current = observe_metrics(self.history)["current"]
        manual_rate = 100 * current["manual_reviews"] / current["account_count"]
        self.assertGreater(manual_rate, 50)
        self.assertGreater(current["default_rate_pct"], 10)
        self.assertGreater(current["churn_rate_pct"], 15)
        self.assertLess(current["lead_conversion_pct"], 30)
        self.assertLess(current["cross_sell_conversion_pct"], 25)
        self.assertLess(current["collection_efficiency_pct"], 75)

    def test_improving_growth_does_not_recommend_intervention(self):
        answer, _, _ = answer_historical_bi(
            "How has the Personal Loan portfolio grown compared to last year?",
            self.history,
        )
        self.assertIn("no intervention is recommended", answer)
        self.assertNotIn("**Recommended action:**", answer)

    def test_adverse_metric_recommends_matching_ai_ml_output(self):
        answer, result, _ = answer_historical_bi(
            "Which branch has the highest increase in default rate YoY?",
            self.history,
        )
        self.assertEqual(len(result), 1)
        self.assertIn("**Recommended action:**", answer)
        self.assertIn("Default & early warning", answer)
        self.assertIn("Credit decisioning", answer)

    def test_growth_drivers_return_product_and_branch_contributions(self):
        answer, result, route = answer_historical_bi(
            "What are the drivers for this portfolio growth?",
            self.history,
        )
        self.assertEqual(route, "historical:portfolio_amount:drivers")
        self.assertEqual(set(result["Driver type"]), {"Product", "Branch"})
        self.assertIn("leading product driver", answer)
        self.assertIn("leading branch driver", answer)
        self.assertIn("no intervention is recommended", answer)

    def test_unspecified_portfolio_growth_defaults_to_amount(self):
        answer, _, route = answer_historical_bi(
            "Show portfolio growth compared with last year",
            self.history,
        )
        self.assertEqual(route, "historical:portfolio_amount:yoy")
        self.assertIn("Portfolio Amount", answer)


if __name__ == "__main__":
    unittest.main()

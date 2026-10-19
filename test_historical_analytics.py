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

    def test_named_branch_product_contribution_explains_business_and_funnel_drivers(self):
        kochi = self.history.copy()
        kochi.loc[kochi["branch_code"] == "BR001", ["branch_code", "city", "state", "region"]] = [
            "BR011", "Kochi", "Kerala", "South"
        ]
        answer, result, route = answer_historical_bi(
            "what are the key drivers for kochi contributing most to personal lloan", kochi
        )
        self.assertEqual(route, "historical:portfolio_contribution:drivers")
        self.assertIn("lead-to-disbursal conversion", answer)
        self.assertIn("average outstanding", answer)
        self.assertNotIn("grew", answer)
        self.assertIn("Lead contribution", set(result["Metric"]))

    def test_named_branch_product_growth_explains_operational_and_funnel_drivers(self):
        kochi = self.history.copy()
        kochi.loc[kochi["branch_code"] == "BR001", ["branch_code", "city", "state", "region"]] = [
            "BR011", "Kochi", "Kerala", "South"
        ]
        answer, result, route = answer_historical_bi(
            "what are the key driver for growth in kochi for personal loan", kochi
        )
        self.assertEqual(route, "historical:executive:operational_growth_drivers")
        self.assertIn("Acquisition drivers", answer)
        self.assertIn("Funnel efficiency", answer)
        self.assertIn("Driver interpretation", answer)
        self.assertIn("Lead-to-disbursal", set(result["Driver"]))
        self.assertIn("Approval-to-disbursal", set(result["Driver"]))
        self.assertIn("volume-led", answer)

    def test_named_branch_product_drivers_imply_growth_analysis(self):
        kochi = self.history.copy()
        kochi.loc[kochi["branch_code"] == "BR001", ["branch_code", "city", "state", "region"]] = [
            "BR011", "Kochi", "Kerala", "South"
        ]
        answer, result, route = answer_historical_bi(
            "what are the key drivers of kochi branch in personal loan", kochi
        )
        self.assertEqual(route, "historical:executive:operational_growth_drivers")
        self.assertIn("accounts changed", answer)
        self.assertIn("Funnel efficiency", answer)
        self.assertIn("Approval-to-disbursal", set(result["Driver"]))

    def test_least_funnel_leakage_is_ranked_by_lead_to_disbursal_not_collections(self):
        answer, result, route = answer_historical_bi(
            "which branch has least funnel leakage", self.history
        )
        self.assertEqual(route, "historical:funnel_leakage:branch_ranking")
        self.assertIn("city", result.columns)
        self.assertIn("funnel_leakage_pct", result.columns)
        self.assertIn("lead_to_disbursal_pct", result.columns)
        self.assertNotIn("collection", answer.lower())
        self.assertIn(str(result.iloc[0]["city"]), answer)

    def test_kochi_funnel_recommendation_uses_propensity_and_stage_actions(self):
        kochi = self.history.copy()
        kochi.loc[kochi["branch_code"] == "BR001", ["branch_code", "city", "state", "region"]] = [
            "BR011", "Kochi", "Kerala", "South"
        ]
        answer, result, route = answer_historical_bi(
            "how to improve funnel leakage of kochi branch", kochi
        )
        self.assertEqual(route, "historical:funnel_leakage:diagnosis")
        self.assertIn("Propensity", answer)
        self.assertIn("Campaign Agent", answer)
        self.assertNotIn("collection efficiency", answer.lower())
        self.assertEqual(len(result), 4)

    def test_generic_funnel_recommendation_diagnoses_authorized_portfolio(self):
        answer, result, route = answer_historical_bi(
            "how to improve the funnel leakage", self.history
        )
        self.assertEqual(route, "historical:funnel_leakage:diagnosis")
        self.assertIn("Authorized bank portfolio", answer)
        self.assertIn("Propensity", answer)
        self.assertEqual(len(result), 4)

    def test_overall_business_improvement_returns_executive_recommendations(self):
        answer, result, route = answer_historical_bi(
            "how to improve overall business portfolio for bank as a whole", self.history
        )
        self.assertEqual(route, "historical:portfolio:improvement")
        self.assertIn("Recommended actions", answer)
        self.assertIn("Propensity", answer)
        self.assertIn("Campaign Agent", answer)
        self.assertIn("Funnel leakage", set(result["KPI"]))

    def test_september_vs_august_accounts_uses_new_account_counts(self):
        answer, result, route = answer_historical_bi(
            "what is the growth in number of accounts opened in september compared to august", self.history
        )
        self.assertEqual(route, "historical:new_accounts:explicit_months")
        self.assertIn("new accounts", answer)
        self.assertEqual(result["Period"].str.contains("August|September").sum(), 2)
        self.assertIn("New accounts", result.columns)

    def test_branch_growth_risk_and_leakage_uses_all_three_metrics(self):
        answer, result, route = answer_historical_bi(
            "Identify branches where portfolio growth is positive but default risk and funnel leakage are worsening.",
            self.history,
        )
        self.assertEqual(route, "historical:executive:branch_growth_risk_leakage")
        self.assertIn("Portfolio growth %", result.columns)
        self.assertIn("Default change pp", result.columns)
        self.assertIn("Funnel leakage change pp", result.columns)
        self.assertTrue((result["Portfolio growth %"] > 0).all())
        self.assertTrue((result["Default change pp"] > 0).all())
        self.assertTrue((result["Funnel leakage change pp"] > 0).all())
        self.assertIn("conditions", answer)

    def test_mumbai_delhi_executive_comparison_returns_both_cities(self):
        answer, result, route = answer_historical_bi(
            "Compare Mumbai and Delhi on portfolio growth, default risk and lead-conversion efficiency, and summarize the better-performing branch.",
            self.history,
        )
        self.assertEqual(route, "historical:executive:city_performance_comparison")
        self.assertEqual(set(result["City"]), {"Mumbai", "Delhi"})
        self.assertIn("Performance score", result.columns)
        self.assertIn("better-performing", answer)

    def test_verified_executive_multi_kpi_query_pack(self):
        cases = [
            ("Which products have grown YoY but have experienced a decline in approval-to-disbursal conversion?", "product_growth_conversion"),
            ("Which branches have strong lead volumes but weak disbursal conversion?", "branch_lead_conversion"),
            ("Find branches where new-account growth is declining even though the outstanding portfolio is increasing.", "accounts_down_portfolio_up"),
            ("Which lending products are growing faster than the bank average while maintaining a lower-than-average default rate?", "growth_risk_benchmark"),
            ("Identify branches with high manual underwriting, low approval rates and strong eligible-customer volumes.", "underwriting_opportunity"),
            ("Which portfolio has the best growth-to-risk trade-off, and what factors explain its performance?", "growth_risk_tradeoff"),
            ("Analyse overall lending performance and highlight the three most important opportunities for management.", "top_opportunities"),
        ]
        for question, expected_route in cases:
            with self.subTest(question=question):
                answer, result, route = answer_historical_bi(question, self.history)
                self.assertEqual(route, f"historical:executive:{expected_route}")
                self.assertTrue(answer)
                self.assertIsInstance(result, pd.DataFrame)
        self.assertNotIn("Portfolio Amount", answer)

    def test_unspecified_portfolio_growth_defaults_to_amount(self):
        answer, _, route = answer_historical_bi(
            "Show portfolio growth compared with last year",
            self.history,
        )
        self.assertEqual(route, "historical:portfolio_amount:yoy")
        self.assertIn("Portfolio Amount", answer)

    def test_south_region_driver_analysis_stays_inside_south_scope(self):
        scoped_history = self.history.copy()
        scoped_history.loc[scoped_history["branch_code"] == "BR001", "region"] = "South"
        answer, result, route = answer_historical_bi(
            "What are the drivers for declining portfolio growth in South region?",
            scoped_history,
        )
        self.assertEqual(route, "historical:portfolio_amount:drivers")
        branch_drivers = set(result.loc[result["Driver type"] == "Branch", "Driver"])
        product_drivers = set(result.loc[result["Driver type"] == "Product", "Driver"])
        self.assertEqual(branch_drivers, {"BR001"})
        self.assertEqual(product_drivers, {"Personal Loan"})
        self.assertIn("South region portfolio", answer)

    def test_default_rate_by_branch_returns_only_requested_columns(self):
        _, result, _ = answer_historical_bi("default-rate YoY by branch", self.history)
        self.assertEqual(
            result.columns.tolist(),
            ["Branch", "Current %", "Previous %", "Change pp", "Change %"],
        )

    def test_new_accounts_by_product_returns_compact_comparison(self):
        _, result, _ = answer_historical_bi("new accounts MoM by product", self.history)
        self.assertEqual(
            result.columns.tolist(),
            ["Product", "Current accounts", "Previous accounts", "Change accounts", "Change %"],
        )

    def test_south_portfolio_drilldown_returns_product_and_branch_drivers(self):
        scoped_history = self.history.copy()
        scoped_history.loc[scoped_history["branch_code"] == "BR001", "region"] = "South"
        answer, result, route = answer_historical_bi(
            "give me drill down for South region portfolio to understand why portfolio is declining",
            scoped_history,
        )
        self.assertEqual(route, "historical:portfolio_amount:drivers")
        self.assertEqual(set(result["Driver type"]), {"Product", "Branch"})
        self.assertIn("South region portfolio", answer)

    def test_current_snapshot_queries_are_not_captured_as_historical(self):
        self.assertIsNone(answer_historical_bi("show total outstanding portfolio", self.history))
        self.assertIsNone(answer_historical_bi("which branch has lowest collection efficiency", self.history))
        self.assertIsNone(answer_historical_bi("show churn prediction customers", self.history))

    def test_non_temporal_portfolio_contribution_uses_dynamic_semantic_path(self):
        self.assertIsNone(answer_historical_bi(
            "what percentage of portfolio is contributed by each state", self.history
        ))

    def test_generic_complex_comparison_is_not_assumed_to_be_portfolio_yoy(self):
        self.assertIsNone(answer_historical_bi(
            "compare average risk and approval rate by product", self.history
        ))


if __name__ == "__main__":
    unittest.main()

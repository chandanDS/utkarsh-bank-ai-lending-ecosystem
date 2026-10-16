import json
import unittest
from types import SimpleNamespace

import pandas as pd

from hybrid_bi import answer_hybrid_bi, execute_governed_sql, validate_sql


class FakeCompletions:
    def __init__(self):
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        if self.calls == 1:
            payload = {
                "intent": "product_average_risk",
                "needs_clarification": False,
                "clarification_question": "",
                "sql": "SELECT product, ROUND(AVG(risk), 3) AS average_risk FROM portfolio GROUP BY product ORDER BY average_risk DESC",
                "result_title": "Average risk by product",
            }
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(payload)))],
                usage=SimpleNamespace(prompt_tokens=80, completion_tokens=35),
            )
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="Home Loan has the highest average risk in the governed result."))],
            usage=SimpleNamespace(prompt_tokens=45, completion_tokens=15),
        )


class HybridBiTests(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame([
            {"customer_id": "1", "branch_code": "BR001", "city": "Mumbai", "product": "Home Loan", "risk": .8, "dpd": 30, "outstanding": 100, "collection_amount_due": 10, "collection_amount_received": 5, "segment": "manual-review", "churn_probability": .7, "cross_sell_score": .4, "propensity": .5},
            {"customer_id": "2", "branch_code": "BR001", "city": "Mumbai", "product": "Personal Loan", "risk": .2, "dpd": 0, "outstanding": 200, "collection_amount_due": 0, "collection_amount_received": 0, "segment": "pre-approved", "churn_probability": .2, "cross_sell_score": .8, "propensity": .8},
        ])

    def test_simple_query_uses_zero_token_fast_path(self):
        result = answer_hybrid_bi("show total outstanding portfolio", self.df)
        self.assertTrue(result.route.startswith("deterministic:"))
        self.assertEqual(result.usage["Total"], 0)

    def test_complex_query_uses_groq_sql_and_insight(self):
        client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
        result = answer_hybrid_bi("compare average risk by product", self.df, client=client)
        self.assertEqual(result.route, "groq_dynamic_sql")
        self.assertEqual(len(result.data), 2)
        self.assertGreater(result.usage["SQL generation"], 0)
        self.assertGreater(result.usage["AI insight"], 0)
        self.assertEqual(result.usage["Total"], 175)

    def test_complex_query_without_key_fails_cleanly(self):
        result = answer_hybrid_bi("compare average risk by product", self.df)
        self.assertEqual(result.route, "groq_unavailable")
        self.assertIn("GROQ_API_KEY", result.answer)

    def test_sql_blocks_mutation_and_other_tables(self):
        with self.assertRaises(ValueError):
            validate_sql("DELETE FROM portfolio", set(self.df.columns))
        with self.assertRaises(ValueError):
            validate_sql("SELECT * FROM app_users", set(self.df.columns))

    def test_sql_executes_only_over_scoped_dataframe(self):
        result = execute_governed_sql("SELECT branch_code, COUNT(*) AS accounts FROM portfolio GROUP BY branch_code", self.df)
        self.assertEqual(result.iloc[0]["branch_code"], "BR001")
        self.assertEqual(int(result.iloc[0]["accounts"]), 2)

    def test_sql_executes_over_production_sized_demo_frame(self):
        large = pd.concat([self.df] * 5_000, ignore_index=True)
        result = execute_governed_sql(
            "SELECT product, COUNT(*) AS accounts FROM portfolio GROUP BY product ORDER BY accounts DESC",
            large,
        )
        self.assertEqual(int(result["accounts"].sum()), 10_000)


if __name__ == "__main__":
    unittest.main()

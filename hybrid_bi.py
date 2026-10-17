from __future__ import annotations

from dataclasses import dataclass
import json
import re
import sqlite3
import time
from typing import Any

import pandas as pd

from lending_analytics import answer_bi, estimate_tokens
from historical_analytics import answer_historical_bi


DEFAULT_MODEL = "openai/gpt-oss-120b"
DEFAULT_INSIGHT_MODEL = "openai/gpt-oss-20b"
MAX_RESULT_ROWS = 200
MAX_HISTORY_MESSAGES = 4

PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {"type": "string"},
        "needs_clarification": {"type": "boolean"},
        "clarification_question": {"type": "string"},
        "sql": {"type": "string"},
        "result_title": {"type": "string"},
    },
    "required": ["intent", "needs_clarification", "clarification_question", "sql", "result_title"],
    "additionalProperties": False,
}

PLANNER_INSTRUCTIONS = """You are a banking BI semantic planner. Output only the required JSON plan.
Generate exactly one read-only SQLite SELECT over table portfolio. Never invent columns or tables.
Use clear aliases and ROUND decimal outputs. Aggregate unless customer-level rows are explicitly requested.
Definitions: high default risk risk>=0.60; delinquent dpd>0; high churn churn_probability>=0.65;
high propensity propensity>=0.75; cross-sell opportunity cross_sell_score>=0.70;
approved means segment IN ('pre-approved','pre-qualified'); collection efficiency=100*SUM(received)/SUM(due).
If the metric, grouping, comparison, or filter is genuinely ambiguous, ask one concise clarification and leave sql empty.
RBAC is already applied to portfolio; never broaden or replace its authorization scope."""

COLUMN_CATALOGUE = """customer_id,name,age,income,current_emi,bureau,utilization,delinquencies,aa_cashflow,
kyc,consent,contact_hour,requested_amount,cohort,response,dpd,branch_code,city,state,product,
outstanding,loan_status,emi_amount,tenure_months,origination_date,last_payment_date,
collection_amount_due,collection_amount_received,customer_segment,preferred_channel,risk,
propensity,amount,rate,segment,reasons,churn_probability,cross_sell_score,collection_priority,
recommended_product,recommended_action"""

FORBIDDEN_SQL = re.compile(
    r"\b(insert|update|delete|drop|alter|create|attach|detach|pragma|replace|truncate|vacuum|reindex)\b",
    re.IGNORECASE,
)


@dataclass
class HybridBIResult:
    answer: str
    data: pd.DataFrame
    route: str
    usage: dict[str, int]
    sql: str = ""
    model: str = "Local governed BI"


def zero_usage() -> dict[str, int]:
    return {
        "Input": 0,
        "Routing": 0,
        "Context": 0,
        "SQL generation": 0,
        "AI insight": 0,
        "Total": 0,
    }


def should_display_result_table(question: str, data: pd.DataFrame) -> bool:
    """Show supporting rows only when they add information beyond the direct answer."""
    if data.empty:
        return False
    q = question.lower().strip()
    explicit_detail = any(
        marker in q
        for marker in (
            "table", "list", "details", "detail", "breakdown", "all branches",
            "all products", "all regions", "all segments",
        )
    ) or bool(re.search(r"\b(top|bottom)\s+\d+\b", q))
    if explicit_detail:
        return True
    singular_ranking = any(
        marker in q
        for marker in ("highest", "lowest", "fastest", "most", "least", "best", "worst")
    )
    executive_summary = any(marker in q for marker in ("key driver", "main driver", "growth driver"))
    direct_scalar = any(marker in q for marker in ("total", "how much", "what is the"))
    if singular_ranking or executive_summary or direct_scalar or len(data) == 1:
        return False
    return True


def _requires_llm(question: str, local_route: str) -> bool:
    if local_route == "clarification":
        return True
    q = question.lower()
    markers = (
        "compare", "versus", " vs ", "trend", "month over month", "year over year",
        "average", "median", "ratio", "percentage", "contribution", "distribution",
        "correlation", "breakdown", "group by", "across product", "across state",
        "top 3", "top 5", "bottom 3", "bottom 5", "why", "how has",
    )
    return any(marker in q for marker in markers)


def _history_text(history: list[dict[str, Any]] | None) -> str:
    if not history:
        return "None"
    compact = []
    for message in history[-MAX_HISTORY_MESSAGES:]:
        compact.append(f"{str(message.get('role', 'user'))[:12]}: {str(message.get('content', ''))[:240]}")
    return "\n".join(compact) or "None"


def _contextualize_historical_question(
    question: str, history: list[dict[str, Any]] | None
) -> str:
    """Carry the prior user scope into short referential historical follow-ups."""
    normalized = question.lower().strip()
    referential = bool(
        re.search(r"\b(this|that|these|those|its|same|above|previous)\b", normalized)
    )
    short_driver_followup = "driver" in normalized and len(normalized.split()) <= 9
    if not history or not (referential or short_driver_followup):
        return question
    previous_user_question = next(
        (
            str(message.get("content", "")).strip()
            for message in reversed(history)
            if message.get("role") == "user" and str(message.get("content", "")).strip()
        ),
        "",
    )
    if not previous_user_question:
        return question
    return f"{question} Previous user scope: {previous_user_question}"


def _contextual_defaulter_dpd_breakdown(
    question: str,
    scoped_df: pd.DataFrame,
    history: list[dict[str, Any]] | None,
) -> HybridBIResult | None:
    """Reconcile a DPD follow-up to the prior branch-defaulter result."""
    normalized = question.lower().strip()
    asks_dpd_breakdown = "dpd" in normalized and any(
        marker in normalized for marker in ("group", "break", "bucket", "detail", "split")
    )
    if not history or not asks_dpd_breakdown:
        return None
    previous_answer = next(
        (
            message
            for message in reversed(history)
            if message.get("role") == "assistant"
            and str(message.get("route", "")).removeprefix("deterministic:") == "branch_defaulters"
        ),
        None,
    )
    if previous_answer is None:
        return None
    previous_rows = previous_answer.get("data") or []
    branch_code = str(previous_rows[0].get("branch_code", "")).upper() if previous_rows else ""
    if not branch_code:
        branch_match = re.search(r"\bBR\d{3}\b", str(previous_answer.get("content", "")), re.IGNORECASE)
        branch_code = branch_match.group(0).upper() if branch_match else ""
    if not branch_code:
        return None
    defaulters = scoped_df[
        (scoped_df["branch_code"].str.upper() == branch_code) & (scoped_df["dpd"] > 30)
    ].copy()
    if defaulters.empty:
        return HybridBIResult(
            f"No defaulters with DPD greater than 30 are available for {branch_code}.",
            pd.DataFrame(),
            "deterministic:branch_defaulter_dpd_breakdown",
            zero_usage(),
        )
    result = defaulters.groupby("dpd", as_index=False).agg(
        customer_count=("customer_id", "count"),
        outstanding=("outstanding", "sum"),
        collection_due=("collection_amount_due", "sum"),
        collection_received=("collection_amount_received", "sum"),
    ).sort_values("dpd")
    collection_due = result["collection_due"].where(result["collection_due"].ne(0))
    result["collection_efficiency_pct"] = (
        100 * result["collection_received"] / collection_due
    ).round(2)
    buckets = ", ".join(
        f"DPD {int(row.dpd)}: {int(row.customer_count):,}"
        for row in result.itertuples(index=False)
    )
    answer = (
        f"{branch_code} has {len(defaulters):,} defaulters with DPD greater than 30. "
        f"DPD breakup — {buckets}. The bucket total reconciles to {len(defaulters):,}."
    )
    return HybridBIResult(
        answer,
        result,
        "deterministic:branch_defaulter_dpd_breakdown",
        zero_usage(),
    )


def _usage_value(usage: Any, name: str) -> int:
    return int(getattr(usage, name, 0) or 0) if usage is not None else 0


def _allocate_usage(question: str, history_text: str, plan_response: Any, insight_response: Any = None) -> dict[str, int]:
    plan_usage = getattr(plan_response, "usage", None)
    plan_input = _usage_value(plan_usage, "prompt_tokens")
    plan_output = _usage_value(plan_usage, "completion_tokens")
    input_tokens = min(plan_input, estimate_tokens(question))
    context_tokens = min(max(0, plan_input - input_tokens), 0 if history_text == "None" else estimate_tokens(history_text))
    routing_tokens = max(0, plan_input - input_tokens - context_tokens)
    insight_usage = getattr(insight_response, "usage", None)
    insight_tokens = _usage_value(insight_usage, "prompt_tokens") + _usage_value(insight_usage, "completion_tokens")
    usage = {
        "Input": input_tokens,
        "Routing": routing_tokens,
        "Context": context_tokens,
        "SQL generation": plan_output,
        "AI insight": insight_tokens,
    }
    usage["Total"] = sum(usage.values())
    return usage


def _clean_sql(sql: str) -> str:
    return re.sub(r"^```(?:sql)?|```$", "", (sql or "").strip(), flags=re.IGNORECASE).strip().rstrip(";")


def validate_sql(sql: str, allowed_columns: set[str]) -> str:
    cleaned = _clean_sql(sql)
    lowered = cleaned.lower()
    if not cleaned or not re.match(r"^select\b", lowered):
        raise ValueError("Only a single SELECT statement is permitted.")
    if ";" in cleaned or "--" in cleaned or "/*" in cleaned or FORBIDDEN_SQL.search(cleaned):
        raise ValueError("Unsafe SQL syntax was blocked.")
    if not re.search(r"\bfrom\s+portfolio\b", lowered):
        raise ValueError("The query may read only from the governed portfolio view.")
    if re.search(r"\b(sqlite_master|sqlite_schema|pragma_)\b", lowered):
        raise ValueError("System metadata access was blocked.")
    for identifier in re.findall(r'"([A-Za-z_][A-Za-z0-9_]*)"', cleaned):
        if identifier not in allowed_columns and identifier.lower() != "portfolio":
            raise ValueError(f"Unknown governed field: {identifier}")
    return cleaned


def execute_governed_sql(sql: str, scoped_df: pd.DataFrame) -> pd.DataFrame:
    safe_sql = validate_sql(sql, set(scoped_df.columns))
    limited_sql = f"SELECT * FROM ({safe_sql}) AS governed_result LIMIT {MAX_RESULT_ROWS}"
    with sqlite3.connect(":memory:") as conn:
        deadline = time.monotonic() + 3.0
        conn.set_progress_handler(lambda: int(time.monotonic() > deadline), 10_000)
        scoped_df.to_sql("portfolio", conn, index=False, if_exists="replace")
        try:
            return pd.read_sql_query(limited_sql, conn)
        except sqlite3.OperationalError as exc:
            raise ValueError(f"The generated query could not be executed safely: {exc}") from exc


def _client(api_key: str):
    try:
        from groq import Groq
    except ImportError as exc:
        raise RuntimeError("The Groq SDK is not installed. Install project requirements.") from exc
    return Groq(api_key=api_key)


def _plan(client: Any, question: str, history_text: str, model: str) -> Any:
    prompt = f"Columns:\n{COLUMN_CATALOGUE}\nRecent context:\n{history_text}\nQuestion:\n{question}"
    return client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": PLANNER_INSTRUCTIONS},
            {"role": "user", "content": prompt},
        ],
        temperature=1e-8,
        max_completion_tokens=320,
        reasoning_effort="low",
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "banking_query_plan", "strict": True, "schema": PLAN_SCHEMA},
        },
    )


def _fallback_insight(result: pd.DataFrame, title: str) -> str:
    if result.empty:
        return f"No records matched the governed query for {title.lower()}."
    if len(result) == 1:
        facts = ", ".join(f"{column}: {value}" for column, value in result.iloc[0].items())
        return f"{title}: {facts}."
    return f"{title} returned {len(result):,} rows. The table is ordered as requested."


def _insight(client: Any, question: str, result: pd.DataFrame, title: str, model: str, temperature: float) -> Any:
    sample = result.head(3).to_json(orient="records", double_precision=4)
    return client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "Answer only from the supplied query result in no more than 55 words. First state the direct "
                    "business answer and evidence. Do not invent causes. Amounts are INR: use ₹ and convert large "
                    "totals to crore (1 crore = 10,000,000). If the result clearly shows improvement or positive "
                    "growth, do not recommend any action. Only for a clearly adverse result, append 'Recommended "
                    "action:' and refer to the most relevant available output: Propensity for weak acquisition or "
                    "lead conversion; Default & early warning or Credit decisioning for rising default/credit risk; "
                    "Churn for worsening attrition; Cross-sell / up-sell or Next Best Offer/Action for weak cross-sell; "
                    "Default & early warning for weak collection efficiency. If none applies, ask the Business Head "
                    "to validate the drivers. No tables, bullets, headings, markdown, or unsupported prescriptions."
                ),
            },
            {"role": "user", "content": f"Question: {question}\nTitle: {title}\nResult:\n{sample}"},
        ],
        temperature=max(1e-8, float(temperature)),
        max_completion_tokens=200,
        reasoning_effort="low",
    )


def answer_hybrid_bi(
    question: str,
    scoped_df: pd.DataFrame,
    history: list[dict[str, Any]] | None = None,
    *,
    historical_df: pd.DataFrame | None = None,
    api_key: str | None = None,
    model: str = DEFAULT_MODEL,
    insight_temperature: float = 0.1,
    insight_model: str = DEFAULT_INSIGHT_MODEL,
    client: Any = None,
) -> HybridBIResult:
    contextual_breakdown = _contextual_defaulter_dpd_breakdown(question, scoped_df, history)
    if contextual_breakdown is not None:
        return contextual_breakdown
    if historical_df is not None:
        historical_question = _contextualize_historical_question(question, history)
        historical_result = answer_historical_bi(historical_question, historical_df)
        if historical_result is not None:
            answer, data, route = historical_result
            return HybridBIResult(answer, data, route, zero_usage())
    local_answer, local_data, local_route = answer_bi(question, scoped_df)
    if not _requires_llm(question, local_route):
        return HybridBIResult(local_answer, local_data, f"deterministic:{local_route}", zero_usage())

    if not api_key and client is None:
        return HybridBIResult(
            "This question needs the governed Groq query planner, but GROQ_API_KEY is not configured.",
            pd.DataFrame(), "groq_unavailable", zero_usage(), model=model,
        )

    history_text = _history_text(history)
    groq_client = client or _client(str(api_key))
    try:
        plan_response = _plan(groq_client, question, history_text, model)
        plan = json.loads(plan_response.choices[0].message.content or "{}")
        if plan["needs_clarification"]:
            usage = _allocate_usage(question, history_text, plan_response)
            clarification = plan["clarification_question"] or "Please clarify the metric or grouping you want."
            return HybridBIResult(clarification, pd.DataFrame(), "groq_clarification", usage, model=model)

        safe_sql = validate_sql(plan["sql"], set(scoped_df.columns))
        result = execute_governed_sql(safe_sql, scoped_df)
        title = plan["result_title"] or "Governed query result"
        if result.empty:
            usage = _allocate_usage(question, history_text, plan_response)
            return HybridBIResult(
                f"No records matched the governed query for {title.lower()}.", result,
                "groq_dynamic_sql", usage, sql=safe_sql, model=model,
            )

        insight_response = None
        try:
            insight_response = _insight(groq_client, question, result, title, insight_model, insight_temperature)
            answer = (insight_response.choices[0].message.content or "").strip() or _fallback_insight(result, title)
        except Exception:
            answer = _fallback_insight(result, title)
        usage = _allocate_usage(question, history_text, plan_response, insight_response)
        return HybridBIResult(answer, result, "groq_dynamic_sql", usage, sql=safe_sql, model=model)
    except Exception as exc:
        return HybridBIResult(
            f"The governed Groq path could not complete safely ({type(exc).__name__}). Please refine the question or retry.",
            pd.DataFrame(), "groq_error", zero_usage(), model=model,
        )

from __future__ import annotations

import math
import re

import pandas as pd


AI_COLUMNS = [
    "customer_id", "name", "branch_code", "city", "product", "customer_segment",
    "bureau", "risk", "propensity", "churn_probability", "cross_sell_score",
    "segment", "recommended_product", "recommended_action", "reasons",
]

BRANCH_LOCATIONS = {
    "mumbai": "BR001", "delhi": "BR002", "bengaluru": "BR003",
    "bangalore": "BR003", "chennai": "BR004", "kolkata": "BR005",
    "hyderabad": "BR006", "pune": "BR007", "ahmedabad": "BR008",
    "jaipur": "BR009", "lucknow": "BR010", "kochi": "BR011",
    "indore": "BR012",
}

PRODUCT_ALIASES = {
    "personal loan": "Personal Loan", "personal loans": "Personal Loan", "pl": "Personal Loan",
    "home loan": "Home Loan", "home loans": "Home Loan", "hl": "Home Loan",
    "vehicle loan": "Vehicle Loan", "vehicle loans": "Vehicle Loan", "vl": "Vehicle Loan",
    "education loan": "Education Loan", "education loans": "Education Loan", "el": "Education Loan",
    "msme loan": "MSME Loan", "msme loans": "MSME Loan", "msme": "MSME Loan",
    "agri loan": "Agri Loan", "agri loans": "Agri Loan", "agriculture loan": "Agri Loan",
}


def _requested_product(question: str) -> str | None:
    for alias, product in sorted(PRODUCT_ALIASES.items(), key=lambda item: len(item[0]), reverse=True):
        if re.search(rf"\b{re.escape(alias)}\b", question):
            return product
    return None


def _requested_branch(question: str) -> tuple[str | None, str | None]:
    branch_match = re.search(r"\b(br\d{3})\b", question, re.I)
    if branch_match:
        return branch_match.group(1).upper(), None
    for city, branch_code in BRANCH_LOCATIONS.items():
        if re.search(rf"\b{re.escape(city)}\b", question):
            return branch_code, city.title()
    return None, None


def _business_scope(question: str, df: pd.DataFrame) -> tuple[pd.DataFrame, str | None, str | None]:
    """Apply semantic filters without ever widening the caller's RBAC-scoped frame."""
    product = _requested_product(question)
    branch_code, city = _requested_branch(question)
    scoped = df
    if product:
        scoped = scoped[scoped["product"].eq(product)]
    if branch_code:
        scoped = scoped[scoped["branch_code"].str.upper().eq(branch_code)]
    return scoped, product, branch_code or city


def _prediction_output_request(
    question: str, scoped: pd.DataFrame
) -> tuple[str, pd.DataFrame, str] | None:
    """Return governed customer-level AI/ML outputs without bypassing caller scope."""
    detail_markers = (
        "list", "show", "data", "record", "output", "customer", "who are",
        "what are", "give me", "candidates",
    )
    if not any(marker in question for marker in detail_markers):
        return None

    use_case = None
    if "early warning" in question or "default prediction" in question:
        use_case = "default_early_warning"
    elif "propensity" in question:
        use_case = "propensity"
    elif "churn" in question or "attrition" in question:
        use_case = "churn"
    elif any(term in question for term in ("cross-sell", "cross sell", "up-sell", "upsell")):
        use_case = "cross_sell"
    elif "credit decision" in question or "underwriting" in question:
        use_case = "credit_decisioning"
    elif "personalization" in question or "persona" in question:
        use_case = "personalization"
    elif "collection" in question:
        use_case = "collections"
    elif "next best" in question:
        use_case = "next_best_offer_action"
    if use_case is None:
        return None

    target_code = None
    branch_match = re.search(r"\b(br\d{3})\b", question)
    if branch_match:
        target_code = branch_match.group(1).upper()
    else:
        for city, branch_code in BRANCH_LOCATIONS.items():
            if re.search(rf"\b{re.escape(city)}\b", question):
                target_code = branch_code
                break

    result = scoped.copy()
    if target_code:
        result = result[result["branch_code"].str.upper() == target_code]

    configs = {
        "default_early_warning": {
            "title": "Default & early warning",
            "filter": lambda frame: frame["default_probability_90d"] >= .45,
            "sort": ["default_probability_90d", "risk"],
            "columns": ["customer_id", "name", "branch_code", "city", "product", "customer_segment", "current_account_status", "default_probability_90d", "bureau", "risk", "dpd", "delinquency_events_12m", "utilization", "aa_cashflow", "outstanding"],
        },
        "propensity": {
            "title": "Propensity",
            "filter": lambda frame: frame["propensity"] >= .60,
            "sort": ["propensity"],
            "columns": ["customer_id", "name", "branch_code", "city", "product", "customer_segment", "propensity", "preferred_channel", "contact_hour", "recommended_product"],
        },
        "churn": {
            "title": "Churn",
            "filter": lambda frame: frame["churn_probability"] >= .65,
            "sort": ["churn_probability"],
            "columns": ["customer_id", "name", "branch_code", "city", "product", "customer_segment", "churn_probability", "preferred_channel", "contact_hour"],
        },
        "cross_sell": {
            "title": "Cross-sell / up-sell",
            "filter": lambda frame: frame["cross_sell_score"] >= .70,
            "sort": ["cross_sell_score"],
            "columns": ["customer_id", "name", "branch_code", "city", "product", "cross_sell_score", "recommended_product", "recommended_action", "preferred_channel"],
        },
        "credit_decisioning": {
            "title": "Credit decisioning",
            "filter": lambda frame: pd.Series(True, index=frame.index),
            "sort": ["risk"],
            "columns": ["customer_id", "name", "branch_code", "city", "product", "segment", "current_account_status", "dpd", "bureau", "risk", "requested_amount", "amount", "aa_cashflow", "reasons"],
        },
        "personalization": {
            "title": "Personalization",
            "filter": lambda frame: pd.Series(True, index=frame.index),
            "sort": ["propensity"],
            "columns": ["customer_id", "name", "branch_code", "city", "customer_persona", "customer_segment", "product", "recommended_product", "recommended_action", "preferred_channel", "contact_hour"],
        },
        "collections": {
            "title": "Collections",
            "filter": lambda frame: frame["dpd"] > 0,
            "sort": ["collection_priority", "dpd"],
            "columns": ["customer_id", "name", "branch_code", "city", "product", "dpd", "collection_priority", "collection_amount_due", "collection_amount_received", "outstanding"],
        },
        "next_best_offer_action": {
            "title": "Next Best Offer/Action",
            "filter": lambda frame: pd.Series(True, index=frame.index),
            "sort": ["propensity"],
            "columns": ["customer_id", "name", "branch_code", "city", "product", "past_complaint", "complaint_reason", "recommended_product", "recommended_action", "preferred_channel", "contact_hour"],
        },
    }
    config = configs[use_case]
    result = result[config["filter"](result)].copy()
    stp_requested = use_case == "credit_decisioning" and any(
        marker in question for marker in ("lead", "stp", "straight-through", "straight through")
    )
    if stp_requested:
        result = result[
            result["segment"].isin(["pre-approved", "pre-qualified"])
            & (result["risk"] < .45)
            & (result["dpd"] == 0)
        ].copy()
    sort_columns = [column for column in config["sort"] if column in result.columns]
    if sort_columns:
        result = result.sort_values(sort_columns, ascending=False, kind="stable")
    result = result[[column for column in config["columns"] if column in result.columns]]

    output_title = "Credit decisioning STP leads" if stp_requested else config["title"]
    scope_label = target_code or "the authorized portfolio"
    if result.empty:
        answer = f"No authorized {output_title} matched for {scope_label}."
    else:
        city_values = result["city"].dropna().unique().tolist() if "city" in result else []
        location = f"{city_values[0]} ({target_code})" if target_code and len(city_values) == 1 else scope_label
        if use_case == "propensity":
            answer = f"Found {len(result):,} Propensity leads for {location}. The governed customer list is shown below."
        elif stp_requested:
            answer = f"Found {len(result):,} Credit decisioning leads eligible for STP for {location}. The eligible lead list is shown below."
        else:
            answer = f"Found {len(result):,} {config['title']} prediction records for {location}."
    return answer, result, f"ai_ml_output:{use_case}"


def enrich_portfolio(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["outstanding"] = pd.to_numeric(result["outstanding"], errors="coerce").fillna(0.0)
    result["collection_amount_due"] = pd.to_numeric(result["collection_amount_due"], errors="coerce").fillna(0.0)
    result["collection_amount_received"] = pd.to_numeric(result["collection_amount_received"], errors="coerce").fillna(0.0)
    result["churn_probability"] = (
        0.12
        + result["risk"] * 0.30
        + (1 - result["response"].fillna(0)) * 0.18
        + (result["dpd"] > 0).astype(float) * 0.16
        + (result["aa_cashflow"] < 0.85).astype(float) * 0.12
    ).clip(0.03, 0.96).round(3)
    result["cross_sell_score"] = (
        result["propensity"] * 0.60
        + (result["bureau"] / 900).clip(0, 1) * 0.25
        + (result["risk"] < 0.35).astype(float) * 0.15
    ).clip(0.02, 0.98).round(3)
    result["collection_priority"] = (
        result["risk"] * 0.55 + (result["dpd"] / 90).clip(0, 1) * 0.45
    ).clip(0, 1).round(3)
    customer_number = pd.to_numeric(
        result["customer_id"].str.extract(r"(\d+)", expand=False), errors="coerce"
    ).fillna(0).astype(int)
    result["delinquency_events_12m"] = (
        result["delinquencies"].fillna(0).astype(int) * (1 + customer_number.mod(3))
        + ((result["delinquencies"].fillna(0) == 0) & (customer_number.mod(17) == 0)).astype(int)
    )
    result["current_account_status"] = "Standard"
    result.loc[result["dpd"].between(1, 59), "current_account_status"] = "Delinquent"
    result.loc[result["dpd"] >= 60, "current_account_status"] = "Defaulted"
    bureau_pressure = ((720 - result["bureau"]) / 180).clip(0, 1)
    cashflow_pressure = ((1 - result["aa_cashflow"]) / 0.45).clip(0, 1)
    result["default_probability_90d"] = (
        result["risk"] * 0.38
        + result["utilization"] * 0.27
        + bureau_pressure * 0.20
        + cashflow_pressure * 0.10
        + (result["delinquency_events_12m"].clip(0, 3) / 3) * 0.05
    ).clip(0.03, 0.96).round(3)
    safe_income = pd.to_numeric(result["income"], errors="coerce").replace(0, pd.NA)
    requested_amount = pd.to_numeric(
        result.get("requested_amount", pd.Series(0, index=result.index)), errors="coerce"
    ).fillna(0)
    kyc_complete = pd.to_numeric(
        result.get("kyc", pd.Series(0, index=result.index)), errors="coerce"
    ).fillna(0).eq(1)
    result["foir_pct"] = (100 * result["current_emi"] / safe_income).fillna(0).clip(0, 200).round(2)
    result["requested_to_annual_income"] = (
        requested_amount / (safe_income * 12)
    ).fillna(0).clip(0, 10).round(2)
    result["decision_confidence"] = (
        0.55
        + (result["bureau"] >= 720).astype(float) * .12
        + (result["foir_pct"] <= 45).astype(float) * .10
        + (result["aa_cashflow"] >= .90).astype(float) * .10
        + kyc_complete.astype(float) * .08
        - (result["dpd"] > 30).astype(float) * .20
    ).clip(.35, .95).round(3)
    result["policy_exception"] = result.apply(_policy_exception, axis=1)
    result["underwriting_recommendation"] = result.apply(_underwriting_recommendation, axis=1)
    result["recommended_product"] = result.apply(_recommended_product, axis=1)
    result["customer_persona"] = result.apply(_customer_persona, axis=1)
    complaint_details = result.apply(_complaint_recommendation, axis=1, result_type="expand")
    complaint_details.columns = ["past_complaint", "complaint_reason", "recommended_action"]
    result[["past_complaint", "complaint_reason", "recommended_action"]] = complaint_details
    return result


def _policy_exception(row: pd.Series) -> str:
    exceptions = []
    if float(row.get("bureau") or 0) < 650:
        exceptions.append("Low bureau score")
    if float(row.get("foir_pct") or 0) > 50:
        exceptions.append("High FOIR")
    if float(row.get("aa_cashflow") or 0) < .80:
        exceptions.append("Weak cash-flow signal")
    if int(row.get("dpd") or 0) > 30:
        exceptions.append("DPD above 30")
    if int(row.get("kyc") or 0) != 1:
        exceptions.append("KYC incomplete")
    return "; ".join(exceptions) if exceptions else "No material exception"


def _underwriting_recommendation(row: pd.Series) -> str:
    if int(row.get("dpd") or 0) > 30 or float(row.get("risk") or 0) >= .70:
        return "Risk review before decision"
    if str(row.get("policy_exception")) != "No material exception":
        return "Manual validation of flagged exception"
    if str(row.get("segment")) in {"pre-approved", "pre-qualified"}:
        return "Eligible for straight-through policy validation"
    return "Complete standard underwriting review"


def answer_action_recommendation(
    question: str, scoped: pd.DataFrame
) -> tuple[str, pd.DataFrame, str] | None:
    """Return a grounded customer action queue for adverse-metric recommendations."""
    q = question.lower().strip()
    asks_action = any(term in q for term in (
        "recommend", "improve", "increase", "reduce", "what should", "how can", "action"
    ))
    if not asks_action or scoped.empty:
        return None

    target_code = None
    branch_match = re.search(r"\b(br\d{3})\b", q)
    if branch_match:
        target_code = branch_match.group(1).upper()
    else:
        target_code = next(
            (code for city, code in BRANCH_LOCATIONS.items() if re.search(rf"\b{re.escape(city)}\b", q)),
            None,
        )
    working = scoped.copy()
    if target_code:
        working = working[working["branch_code"].str.upper() == target_code]
    scope_label = target_code or "the authorized portfolio"

    if "collection" in q:
        result = working[working["dpd"] > 0].sort_values(
            ["collection_priority", "dpd", "collection_amount_due"], ascending=False
        )
        columns = ["customer_id", "name", "branch_code", "city", "product", "dpd", "collection_priority", "collection_amount_due", "collection_amount_received", "outstanding", "preferred_channel", "contact_hour"]
        insight = (
            f"**AI insight:** Prioritize the {len(result):,} overdue customers below by collection priority and DPD. "
            "Contact them through their preferred channel, beginning with high dues and weak recovery, to improve collection efficiency."
        )
        route = "collection_efficiency"
    elif "default" in q or "early warning" in q or "credit risk" in q:
        result = working[working["default_probability_90d"] >= .45].sort_values(
            ["default_probability_90d", "risk"], ascending=False
        )
        matched_count = len(result)
        result = result.head(200).copy()
        columns = ["customer_id", "name", "branch_code", "city", "product", "current_account_status", "default_probability_90d", "risk", "bureau", "dpd", "delinquency_events_12m", "utilization", "aa_cashflow", "outstanding"]
        insight = (
            f"**AI insight:** {matched_count:,} customers match the Default & early warning criteria. "
            f"The {len(result):,} highest-priority records are shown below for Credit decisioning review. "
            "Prioritize high predicted risk while distinguishing currently standard customers from already delinquent accounts."
        )
        route = "default_prediction"
    elif "attrition" in q or "churn" in q or "retention" in q:
        result = working[working["churn_probability"] >= .65].sort_values(
            ["churn_probability", "propensity"], ascending=False
        )
        columns = ["customer_id", "name", "branch_code", "city", "product", "customer_segment", "churn_probability", "past_complaint", "complaint_reason", "recommended_action", "preferred_channel", "contact_hour"]
        insight = (
            f"**AI insight:** Reach out to the {len(result):,} high-attrition-risk customers below. "
            "Use the Churn output, complaint context and preferred channel to resolve service friction before product outreach."
        )
        route = "attrition"
    elif any(term in q for term in ("campaign", "lead conversion", "conversion efficiency", "response rate")):
        result = working[
            (working["consent"] == 1) & (working["response"] == 0) & (working["propensity"] >= .60)
        ].sort_values(["propensity", "cross_sell_score"], ascending=False)
        columns = ["customer_id", "name", "branch_code", "city", "product", "propensity", "cross_sell_score", "recommended_product", "preferred_channel", "contact_hour", "consent"]
        insight = (
            f"**AI insight:** Activate the {len(result):,} consented, unconverted customers below through the Campaign Agent. "
            "Sequence outreach by propensity and use each customer's recommended product, preferred channel and contact time."
        )
        route = "campaign_efficiency"
    else:
        return None

    available = [column for column in columns if column in result.columns]
    result = result[available].copy()
    if result.empty:
        insight = f"No authorized action candidates matched for {scope_label}."
    return insight, result, f"action_recommendation:{route}"


def answer_default_driver_analysis(
    question: str, scoped: pd.DataFrame
) -> tuple[str, pd.DataFrame, str] | None:
    """Explain observable default-risk signals for an authorized geography."""
    q = question.lower().strip()
    asks_reason = any(term in q for term in ("reason", "cause", "driver", "driving"))
    if "default" not in q or not asks_reason:
        return None

    target_code = None
    branch_match = re.search(r"\b(br\d{3})\b", q)
    if branch_match:
        target_code = branch_match.group(1).upper()
    else:
        target_code = next(
            (code for city, code in BRANCH_LOCATIONS.items() if re.search(rf"\b{re.escape(city)}\b", q)),
            None,
        )
    working = scoped.copy()
    if target_code:
        working = working[working["branch_code"].str.upper() == target_code]
    if working.empty:
        return (
            f"No authorized customer records are available for {target_code or 'the requested scope'}.",
            pd.DataFrame(), "default_driver_analysis",
        )

    if "default_probability_90d" in working:
        risk_pool = working[working["default_probability_90d"] >= .45].copy()
    else:
        risk_pool = working[(working["risk"] >= .60) | (working["dpd"] > 30)].copy()
    if risk_pool.empty:
        return "No elevated default-risk customers matched the authorized scope.", pd.DataFrame(), "default_driver_analysis"

    delinquency_signal = (
        risk_pool["delinquency_events_12m"].fillna(0) > 0
        if "delinquency_events_12m" in risk_pool
        else risk_pool["delinquencies"].fillna(0) > 0
    )
    signals = [
        ("High credit utilization", risk_pool["utilization"] >= .75, "Utilization is at least 75%."),
        ("Weak bureau score", risk_pool["bureau"] < 650, "Bureau score is below 650."),
        ("Cash-flow stress", risk_pool["aa_cashflow"] < .80, "Account-aggregator cash-flow indicator is below 0.80."),
        ("Prior delinquency", delinquency_signal, "At least one recent delinquency signal is present."),
        ("Current DPD above 30", risk_pool["dpd"] > 30, "The account is already more than 30 days past due."),
    ]
    rows = []
    for signal, mask, definition in signals:
        affected = int(mask.sum())
        rows.append({
            "Risk signal": signal,
            "Affected customers": affected,
            "Share of elevated-risk pool %": round(100 * affected / len(risk_pool), 2),
            "Definition": definition,
        })
    result = pd.DataFrame(rows).sort_values(
        ["Affected customers", "Risk signal"], ascending=[False, True], kind="stable"
    )
    top = result.head(3)
    drivers = ", ".join(
        f"{row['Risk signal']} ({int(row['Affected customers']):,}; {row['Share of elevated-risk pool %']:.2f}%)"
        for _, row in top.iterrows()
    )
    city_values = working["city"].dropna().unique().tolist() if "city" in working else []
    location = f"{city_values[0]} ({target_code})" if target_code and len(city_values) == 1 else (target_code or "the authorized portfolio")
    answer = (
        f"{location} has {len(risk_pool):,} customers in the elevated 90-day default-risk pool. "
        f"The leading observable risk signals are {drivers}. These are associated risk indicators, not proven causal reasons.\n\n"
        "**Recommended action:** Review the corresponding customers in Default & early warning and use Credit decisioning for policy or underwriting interventions."
    )
    return answer, result, "default_driver_analysis"


def _recommended_product(row: pd.Series) -> str:
    current = str(row.get("product") or "Personal Loan")
    income = float(row.get("income") or 0)
    if current != "Home Loan" and income >= 75000:
        return "Home Loan"
    if current != "Personal Loan" and float(row.get("propensity") or 0) >= 0.70:
        return "Personal Loan"
    if str(row.get("customer_segment")) == "Self-employed":
        return "MSME Loan"
    return "Top-up / protection plan"


def _customer_persona(row: pd.Series) -> str:
    risk = float(row.get("risk") or 0)
    dpd = int(row.get("dpd") or 0)
    outstanding = float(row.get("outstanding") or 0)
    income = max(float(row.get("income") or 0), 1.0)
    exposure_ratio = (outstanding + float(row.get("current_emi") or 0) * 12) / (income * 12)
    if dpd >= 30 and exposure_ratio >= 1.25:
        return "High liability delinquency prone borrower"
    if risk >= 0.60 and exposure_ratio >= 1.00:
        return "Stressed high exposure borrower"
    if risk >= 0.45 or str(row.get("segment")) in {"underwriting", "decline", "review"}:
        return "Risk prone sanction heavy client"
    return "Healthy repayer"


def _stable_customer_bucket(customer_id: object) -> int:
    digits = re.sub(r"\D", "", str(customer_id or "0"))
    return int(digits or 0) % 8


def _complaint_recommendation(row: pd.Series) -> tuple[str, str, str]:
    """Create reproducible complaint-led servicing actions for the synthetic demo."""
    scenarios = [
        ("UPI transaction not working", "Repeated UPI transaction failures in recent service history", "Proactively contact customer and troubleshoot UPI access"),
        ("Account debited but beneficiary not credited", "Debit posted but beneficiary credit was delayed", "Share transaction status and initiate payment trace"),
        ("Failed transaction refund delayed", "Refund turnaround exceeded the expected service window", "Confirm refund status and provide resolution timeline"),
        ("Mobile banking login issue", "Multiple unsuccessful mobile-banking login attempts", "Guide secure access reset and confirm successful login"),
        ("ATM cash not dispensed", "Cash withdrawal was declined after account debit", "Initiate ATM dispute review and communicate provisional status"),
        ("Unexpected service charge", "Customer disputed a recently applied service charge", "Explain charge basis and route waiver decision if eligible"),
        ("KYC update pending", "Prior KYC service request remains unresolved", "Request missing KYC information and schedule assisted completion"),
        ("No material complaint recorded", "No unresolved complaint found in the available service history", "Share relevant product information through the preferred channel"),
    ]
    return scenarios[_stable_customer_bucket(row.get("customer_id"))]


def classify_next_best_action(recommended_action: str) -> str:
    """Translate a specific recommendation into a business action category."""
    action = str(recommended_action or "").strip().lower()
    if any(term in action for term in ("troubleshoot", "trace", "refund", "reset", "dispute", "pending")):
        return "Attention"
    if any(term in action for term in ("decision", "waiver")):
        return "Decision"
    if "product" in action or action.startswith("offer "):
        return "Offer"
    return "Information"


def estimate_tokens(text: str) -> int:
    return max(1, math.ceil(len(text.strip()) / 4)) if text.strip() else 0


def token_breakdown(question: str, answer: str, context_messages: int) -> dict[str, int]:
    usage = {
        "Input": estimate_tokens(question),
        "Routing": estimate_tokens("classify lending metric entity filter comparison and date"),
        "Context": context_messages * 12,
        "Query plan": estimate_tokens("validated governed aggregate query plan"),
        "Answer": estimate_tokens(answer),
    }
    usage["Total"] = sum(usage.values())
    return usage


def _branch_ranking(question: str, scoped: pd.DataFrame) -> tuple[str, pd.DataFrame, str] | None:
    """Resolve branch-level superlatives before customer-detail intents."""
    asks_branch = "branch" in question
    descending = any(term in question for term in ("highest", "most", "maximum", "max", "top"))
    ascending = any(term in question for term in ("lowest", "least", "minimum", "min", "bottom"))
    if not asks_branch or not (descending or ascending) or scoped.empty:
        return None

    grouped = scoped.groupby(["branch_code", "city"], as_index=False)
    metric_label: str
    route: str
    if any(term in question for term in ("defaulter", "defaulted")) and "risk" not in question:
        result = grouped.agg(
            defaulters=("dpd", lambda values: int((values > 30).sum())),
            average_dpd=("dpd", lambda values: float(values[values > 30].mean()) if (values > 30).any() else 0.0),
        )
        metric = "defaulters"
        metric_label = "defaulters (DPD > 30)"
        route = "branch_defaulters"
    elif any(term in question for term in ("default risk", "high risk", "risk")):
        result = grouped.agg(
            high_default_risk_customers=("risk", lambda values: int((values >= 0.60).sum())),
            average_default_risk=("risk", "mean"),
        )
        metric = "high_default_risk_customers"
        metric_label = "high default-risk customers"
        route = "branch_default_risk"
    elif "delinquen" in question or "dpd" in question or "overdue" in question:
        result = grouped.agg(
            delinquent_accounts=("dpd", lambda values: int((values > 0).sum())),
            average_dpd=("dpd", "mean"),
        )
        metric = "delinquent_accounts"
        metric_label = "delinquent accounts"
        route = "branch_delinquency"
    elif "collection" in question and "efficien" in question:
        result = grouped.agg(
            due=("collection_amount_due", "sum"),
            received=("collection_amount_received", "sum"),
        )
        result["collection_efficiency_pct"] = (
            result["received"] / result["due"].replace(0, pd.NA) * 100
        ).fillna(100.0).round(2)
        metric = "collection_efficiency_pct"
        metric_label = "collection efficiency"
        route = "branch_collection_efficiency"
    elif "approval" in question or "approved" in question:
        result = grouped.agg(
            customers=("customer_id", "count"),
            approved=("segment", lambda values: int(values.isin(["pre-approved", "pre-qualified"]).sum())),
        )
        result["approval_rate_pct"] = (result["approved"] / result["customers"] * 100).round(2)
        metric = "approval_rate_pct"
        metric_label = "indicative approval rate"
        route = "branch_approval"
    elif "churn" in question:
        result = grouped.agg(
            high_churn_customers=("churn_probability", lambda values: int((values >= 0.65).sum())),
            average_churn_probability=("churn_probability", "mean"),
        )
        metric = "high_churn_customers"
        metric_label = "high churn-risk customers"
        route = "branch_churn"
    elif "cross-sell" in question or "cross sell" in question or "upsell" in question:
        result = grouped.agg(
            cross_sell_opportunities=("cross_sell_score", lambda values: int((values >= 0.70).sum())),
            average_cross_sell_score=("cross_sell_score", "mean"),
        )
        metric = "cross_sell_opportunities"
        metric_label = "cross-sell opportunities"
        route = "branch_cross_sell"
    elif "propensity" in question or "lead" in question:
        result = grouped.agg(
            high_propensity_customers=("propensity", lambda values: int((values >= 0.75).sum())),
            average_propensity=("propensity", "mean"),
        )
        metric = "high_propensity_customers"
        metric_label = "high-propensity customers"
        route = "branch_propensity"
    elif "outstanding" in question or "portfolio" in question or "exposure" in question:
        result = grouped.agg(
            accounts=("customer_id", "count"),
            outstanding=("outstanding", "sum"),
        )
        metric = "outstanding"
        metric_label = "outstanding portfolio"
        route = "branch_outstanding"
    else:
        return None

    result = result.sort_values(metric, ascending=ascending, kind="stable")
    selected = result.head(1).copy()
    row = selected.iloc[0]
    value = float(row[metric])
    if metric == "outstanding":
        rendered_value = f"₹{value / 10_000_000:,.2f} crore"
    elif metric.endswith("_pct"):
        rendered_value = f"{value:.2f}%"
    else:
        rendered_value = f"{int(value):,}"
    qualifier = "lowest" if ascending else "highest"
    if metric == "outstanding" or metric.endswith("_pct"):
        answer = f"{row.branch_code} ({row.city}) has the {qualifier} {metric_label}: {rendered_value}."
    else:
        answer = f"{row.branch_code} ({row.city}) has the {qualifier} number of {metric_label}: {rendered_value}."
    return answer, selected, route


def _portfolio_propensity_ranking(
    question: str, scoped: pd.DataFrame
) -> tuple[str, pd.DataFrame, str] | None:
    """Rank lending product portfolios by their count of high-propensity leads."""
    asks_portfolio = "portfolio" in question or "product" in question
    asks_propensity = "propensity" in question or "lead" in question
    descending = any(term in question for term in ("highest", "most", "maximum", "max", "top"))
    ascending = any(term in question for term in ("lowest", "least", "minimum", "min", "bottom"))
    if not (asks_portfolio and asks_propensity and (descending or ascending)) or scoped.empty:
        return None

    result = scoped.groupby("product", as_index=False).agg(
        high_propensity_leads=("propensity", lambda values: int((values >= 0.75).sum())),
        total_customers=("customer_id", "count"),
        average_propensity=("propensity", "mean"),
    )
    result["average_propensity_pct"] = (result.pop("average_propensity") * 100).round(2)
    result = result.sort_values(
        ["high_propensity_leads", "average_propensity_pct", "product"],
        ascending=[ascending, ascending, True],
        kind="stable",
    )
    selected = result.head(1).copy()
    row = selected.iloc[0]
    qualifier = "lowest" if ascending else "highest"
    answer = (
        f"{row['product']} has the {qualifier} number of high-propensity leads: "
        f"{int(row['high_propensity_leads']):,}."
    )
    return answer, selected, "portfolio_propensity"


def answer_bi(question: str, df: pd.DataFrame) -> tuple[str, pd.DataFrame, str]:
    q = question.lower().strip()
    if not q:
        return "Please enter a lending portfolio question.", pd.DataFrame(), "clarification"

    default_drivers = answer_default_driver_analysis(q, df)
    if default_drivers is not None:
        return default_drivers

    action_recommendation = answer_action_recommendation(q, df)
    if action_recommendation is not None:
        return action_recommendation

    prediction_output = _prediction_output_request(q, df)
    if prediction_output is not None:
        return prediction_output

    branch_match = re.search(r"\b(br\d{3})\b", q)
    scoped = df
    scope_text = "across the portfolio"
    if branch_match:
        code = branch_match.group(1).upper()
        scoped = df[df["branch_code"].str.upper() == code]
        scope_text = f"for {code}"

    business_scoped, requested_product, requested_location = _business_scope(q, df)

    if ("average" in q or "avg" in q) and "risk" in q and ("product" in q or "portfolio" in q):
        result = df.groupby("product", as_index=False).agg(
            accounts=("customer_id", "count"),
            average_risk=("risk", "mean"),
            median_risk=("risk", "median"),
            high_risk_accounts=("risk", lambda values: int((values >= .60).sum())),
        )
        result["Average risk %"] = (result.pop("average_risk") * 100).round(2)
        result["Median risk %"] = (result.pop("median_risk") * 100).round(2)
        result["High-risk share %"] = (100 * result["high_risk_accounts"] / result["accounts"]).round(2)
        result = result.sort_values("Average risk %", ascending=False, kind="stable")
        leader = result.iloc[0]
        answer = (
            f"{leader['product']} has the highest average predicted risk at "
            f"{leader['Average risk %']:.2f}%. The comparison reports risk as a percentage, not a rounded 0–1 score."
        )
        return answer, result, "product_risk_comparison"

    if "default rate" not in q and "branch" in q and any(term in q for term in ("contribut", "largest", "most")) and requested_product:
        result = business_scoped.groupby(["branch_code", "city"], as_index=False).agg(
            accounts=("customer_id", "count"), outstanding=("outstanding", "sum")
        ).sort_values("outstanding", ascending=False, kind="stable")
        if result.empty:
            return f"No authorized {requested_product} records matched this scope.", result, "branch_product_contribution"
        total = float(result["outstanding"].sum())
        result["contribution_pct"] = (100 * result["outstanding"] / total).round(2) if total else 0.0
        top = result.head(1).copy()
        row = top.iloc[0]
        return (
            f"{row.branch_code} ({row.city}) contributes the most to {requested_product}: "
            f"₹{row.outstanding / 10_000_000:,.2f} crore ({row.contribution_pct:.2f}% of the authorized {requested_product} portfolio).",
            top,
            "branch_product_contribution",
        )

    if "default rate" in q and requested_location:
        if business_scoped.empty:
            return "No authorized records matched the requested branch and product scope.", pd.DataFrame(), "branch_default_rate"
        defaults = int((business_scoped["dpd"] > 30).sum())
        accounts = len(business_scoped)
        rate = 100 * defaults / accounts if accounts else 0.0
        branch_code = str(business_scoped.iloc[0]["branch_code"])
        city = str(business_scoped.iloc[0]["city"])
        result = pd.DataFrame([{
            "branch_code": branch_code, "city": city, "product": requested_product or "All products",
            "accounts": accounts, "defaulters_dpd_gt_30": defaults, "default_rate_pct": round(rate, 2),
        }])
        return (
            f"{branch_code} ({city}) has a current default rate of {rate:.2f}% "
            f"({defaults:,} of {accounts:,} accounts with DPD > 30).",
            result,
            "branch_default_rate",
        )

    branch_ranking = _branch_ranking(q, scoped)
    if branch_ranking is not None:
        return branch_ranking

    portfolio_propensity_ranking = _portfolio_propensity_ranking(q, scoped)
    if portfolio_propensity_ranking is not None:
        return portfolio_propensity_ranking

    dpd_match = re.search(r"dpd\s*(?:greater than|above|>)\s*(\d+)", q)
    if dpd_match:
        threshold = int(dpd_match.group(1))
        result = scoped[scoped["dpd"] > threshold].sort_values(["dpd", "risk"], ascending=False)
        detail_columns = [
            "customer_id", "name", "branch_code", "city", "product", "current_account_status",
            "dpd", "risk", "default_probability_90d", "bureau", "outstanding",
            "collection_amount_due", "collection_amount_received", "preferred_channel", "contact_hour",
        ]
        result = result[[column for column in detail_columns if column in result.columns]]
        return f"{len(result):,} loans have DPD greater than {threshold} {scope_text}.", result, "delinquency_detail"

    if any(term in q for term in ("outstanding", "portfolio", "total business", "lending business", "business amount")):
        total = business_scoped["outstanding"].sum()
        summary = business_scoped.groupby("product", as_index=False).agg(
            accounts=("customer_id", "count"), outstanding=("outstanding", "sum")
        ).sort_values("outstanding", ascending=False)
        summary["share_pct"] = (100 * summary["outstanding"] / total).round(2) if total else 0.0
        scope_parts = [part for part in (requested_product, requested_location) if part]
        rendered_scope = " for " + " · ".join(scope_parts) if scope_parts else " for the authorized bank portfolio"
        fy = pd.Timestamp.today().year if pd.Timestamp.today().month >= 4 else pd.Timestamp.today().year - 1
        timing = f"as of {pd.Timestamp.today():%b %Y} in FY {fy}–{str(fy + 1)[-2:]}" if "fy" in q or "financial year" in q else "at the current snapshot"
        return (
            f"Outstanding lending business{rendered_scope} is ₹{total / 10_000_000:,.2f} crore {timing}.",
            summary,
            "portfolio",
        )

    if ("default" in q or "risk" in q) and ("high" in q or "highest" in q):
        result = scoped[scoped["risk"] >= 0.60].sort_values("risk", ascending=False)
        detail_columns = [
            "customer_id", "name", "branch_code", "city", "product", "customer_segment",
            "current_account_status", "default_probability_90d", "risk", "bureau", "dpd",
            "utilization", "aa_cashflow", "outstanding",
        ]
        result = result[[column for column in detail_columns if column in result.columns]]
        return f"{len(result):,} customers are in the high default-risk band {scope_text}.", result, "default_risk"

    if "delinquen" in q and "branch" in q:
        result = scoped.groupby(["branch_code", "city"], as_index=False).agg(
            delinquent_accounts=("dpd", lambda s: int((s > 0).sum())),
            average_dpd=("dpd", "mean"),
            outstanding=("outstanding", "sum"),
        ).sort_values(["delinquent_accounts", "average_dpd"], ascending=False)
        top = result.iloc[0] if len(result) else None
        answer = "No delinquency records are available." if top is None else (
            f"{top.branch_code} ({top.city}) has the highest delinquent-account count: "
            f"{int(top.delinquent_accounts):,}."
        )
        return answer, result, "branch_delinquency"

    if "eligible" in q and ("personal loan" in q or "personal loans" in q):
        result = scoped[(scoped["recommended_product"] == "Personal Loan") & scoped["segment"].isin(["pre-approved", "pre-qualified"])].sort_values("propensity", ascending=False)
        detail_columns = [
            "customer_id", "name", "branch_code", "city", "product", "customer_segment",
            "segment", "propensity", "bureau", "risk", "requested_amount", "recommended_product",
            "preferred_channel", "contact_hour",
        ]
        result = result[[column for column in detail_columns if column in result.columns]]
        return f"{len(result):,} customers are eligible candidates for a personal-loan offer {scope_text}.", result, "pl_eligibility"

    if "cross-sell" in q or "cross sell" in q or "upsell" in q or "up-sell" in q:
        result = scoped.sort_values("cross_sell_score", ascending=False)
        return f"The top cross-sell pool contains {int((result.cross_sell_score >= .70).sum()):,} customers {scope_text}.", result, "cross_sell"

    if "collection efficiency" in q or ("collection" in q and "efficiency" in q):
        due = scoped["collection_amount_due"].sum()
        received = scoped["collection_amount_received"].sum()
        efficiency = (received / due * 100) if due else 100.0
        result = scoped.groupby("branch_code", as_index=False).agg(
            due=("collection_amount_due", "sum"), received=("collection_amount_received", "sum")
        )
        result["efficiency_pct"] = (result["received"] / result["due"].replace(0, pd.NA) * 100).fillna(100).round(2)
        return f"Collection efficiency is {efficiency:.2f}% {scope_text}.", result.sort_values("efficiency_pct"), "collections"

    if "churn" in q:
        result = scoped[scoped["churn_probability"] >= .65].sort_values("churn_probability", ascending=False)
        by_segment = result.groupby("customer_segment", as_index=False).agg(
            high_churn_customers=("customer_id", "count"), average_churn_probability=("churn_probability", "mean")
        ).sort_values("high_churn_customers", ascending=False)
        return f"{len(result):,} customers are in the high churn-risk pool {scope_text}.", by_segment, "churn"

    if "approval" in q:
        approved = scoped["segment"].isin(["pre-approved", "pre-qualified"])
        rate = approved.mean() * 100 if len(scoped) else 0
        result = scoped.groupby("branch_code", as_index=False).agg(
            customers=("customer_id", "count"),
            approved=("segment", lambda s: int(s.isin(["pre-approved", "pre-qualified"]).sum())),
        )
        result["approval_rate_pct"] = (result["approved"] / result["customers"] * 100).round(2)
        return f"The indicative approval rate is {rate:.2f}% {scope_text}.", result.sort_values("approval_rate_pct", ascending=False), "approval"

    return (
        "I could not resolve that question safely. Ask about outstanding portfolio, approval, high default risk, "
        "branch delinquency, PL eligibility, cross-sell, churn, collection efficiency or DPD thresholds.",
        pd.DataFrame(),
        "clarification",
    )

CAMPAIGN_AUDIENCES = [
    "Propensity — acquisition",
    "Default & early warning — preventive servicing",
    "Churn — retention",
    "Cross-sell / up-sell",
    "Next best action",
]


def campaign_audience(df: pd.DataFrame, audience: str) -> tuple[pd.DataFrame, dict[str, str]]:
    """Return a consented, purpose-specific campaign audience from model outputs."""
    contactable = df[df["consent"] == 1].copy()
    if audience == "Propensity — acquisition":
        view = contactable[
            (contactable["propensity"] >= 0.75)
            & (contactable["risk"] < 0.60)
            & contactable["segment"].isin(["pre-approved", "pre-qualified"])
        ].copy()
        view["campaign_score"] = view["propensity"]
        view["model_output"] = "High product propensity"
        meta = {
            "purpose": "Customer acquisition using high-propensity, policy-eligible prospects",
            "message": "Utkarsh Bank demo: you may be interested in a suitable lending product. This is a simulated outreach; no offer is being made.",
        }
    elif audience == "Default & early warning — preventive servicing":
        view = contactable[(contactable["risk"] >= 0.60) | (contactable["dpd"] > 0)].copy()
        view["campaign_score"] = view[["risk", "collection_priority"]].max(axis=1)
        view["model_output"] = "Elevated credit-risk / early-warning signal"
        meta = {
            "purpose": "Preventive servicing and payment-support outreach; not a sales campaign",
            "message": "Utkarsh Bank demo: we would like to help you review your account and available servicing support. This is a simulated outreach.",
        }
    elif audience == "Churn — retention":
        view = contactable[contactable["churn_probability"] >= 0.65].copy()
        view["campaign_score"] = view["churn_probability"]
        view["model_output"] = "High churn probability"
        meta = {
            "purpose": "Retention outreach for customers with elevated churn probability",
            "message": "Utkarsh Bank demo: we value your relationship and would like to understand how we can serve you better. This is a simulated outreach.",
        }
    elif audience == "Cross-sell / up-sell":
        view = contactable[
            (contactable["cross_sell_score"] >= 0.70) & (contactable["risk"] < 0.60)
        ].copy()
        view["campaign_score"] = view["cross_sell_score"]
        view["model_output"] = view["recommended_product"].map(lambda value: f"Recommended: {value}")
        meta = {
            "purpose": "Relevant cross-sell outreach based on model recommendation",
            "message": "Utkarsh Bank demo: we have identified a potentially relevant banking product for you. This is a simulated outreach; eligibility checks apply.",
        }
    elif audience == "Next best action":
        view = contactable[contactable["recommended_action"] != "Continue relationship nurturing"].copy()
        view["campaign_score"] = view[["propensity", "cross_sell_score", "churn_probability", "collection_priority"]].max(axis=1)
        view["model_output"] = view["recommended_action"]
        meta = {
            "purpose": "Action queue driven by each customer's next-best-action recommendation",
            "message": "Utkarsh Bank demo: we are contacting you regarding a relevant next step in your banking relationship. This is a simulated outreach.",
        }
    else:
        raise ValueError(f"Unsupported campaign audience: {audience}")

    view["campaign_source"] = audience
    return view.sort_values("campaign_score", ascending=False), meta

def lead_conversion_analytics(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, float]:
    """Build a reproducible synthetic amount-based Lead → Login → Quote → Disbursal funnel."""
    working = df.copy()
    customer_number = pd.to_numeric(
        working["customer_id"].str.extract(r"(\d+)", expand=False), errors="coerce"
    ).fillna(0).astype(int)
    working["_lead"] = (working["consent"] == 1) & (working["propensity"] >= .60) & (working["risk"] < .60)
    working["_login"] = working["_lead"] & (customer_number.mod(100) < 72)
    working["_quote"] = working["_login"] & (working["kyc"] == 1) & working["segment"].isin(["pre-approved", "pre-qualified", "underwriting"])
    working["_disbursed"] = working["_quote"] & (working["response"] == 1)
    stages = [("Leads identified", "_lead"), ("Customer login", "_login"), ("Quote generated", "_quote"), ("Loan disbursed", "_disbursed")]
    counts = [int(working[c].sum()) for _, c in stages]
    amounts = [round(float(working.loc[working[c], "amount"].sum()) / 10_000_000, 2) for _, c in stages]
    lead_value = amounts[0] if amounts else 0.0
    rows = []
    previous = 0.0
    for index, ((stage, _), accounts, amount) in enumerate(zip(stages, counts, amounts)):
        rows.append({
            "Stage": stage, "Accounts": accounts, "Amount ₹ Cr": amount,
            "Drop-off ₹ Cr": 0.0 if index == 0 else round(previous - amount, 2),
            "Stage value conversion %": 100.0 if index == 0 else round(amount / previous * 100, 2) if previous else 0.0,
            "Lead value conversion %": round(amount / lead_value * 100, 2) if lead_value else 0.0,
        })
        previous = amount
    stage_report = pd.DataFrame(rows)
    leads, logins, quotes, disbursed = amounts
    waterfall = pd.DataFrame([
        {"Order": 1, "Step": "Leads", "Start": 0.0, "End": leads, "Value": leads, "Type": "Stage total", "Label": f"₹{leads:,.2f} Cr"},
        {"Order": 2, "Step": "Lead → Login drop-off", "Start": logins, "End": leads, "Value": round(leads-logins, 2), "Type": "Drop-off", "Label": f"-₹{leads-logins:,.2f} Cr"},
        {"Order": 3, "Step": "Login → Quote drop-off", "Start": quotes, "End": logins, "Value": round(logins-quotes, 2), "Type": "Drop-off", "Label": f"-₹{logins-quotes:,.2f} Cr"},
        {"Order": 4, "Step": "Quote → Disbursal drop-off", "Start": disbursed, "End": quotes, "Value": round(quotes-disbursed, 2), "Type": "Drop-off", "Label": f"-₹{quotes-disbursed:,.2f} Cr"},
        {"Order": 5, "Step": "Disbursed", "Start": 0.0, "End": disbursed, "Value": disbursed, "Type": "Stage total", "Label": f"₹{disbursed:,.2f} Cr"},
    ])
    products = []
    for product, group in working.groupby("product"):
        values = {key: float(group.loc[group[col], "amount"].sum()) / 10_000_000 for key, col in [("Lead", "_lead"), ("Login", "_login"), ("Quote", "_quote"), ("Disbursal", "_disbursed")]}
        products.append({
            "Product": product, "Lead accounts": int(group["_lead"].sum()),
            "Lead ₹ Cr": round(values["Lead"], 2), "Login ₹ Cr": round(values["Login"], 2),
            "Quote ₹ Cr": round(values["Quote"], 2), "Disbursal ₹ Cr": round(values["Disbursal"], 2),
            "Value conversion %": round(values["Disbursal"] / values["Lead"] * 100, 2) if values["Lead"] else 0.0,
        })
    columns = ["Product", "Lead accounts", "Lead ₹ Cr", "Login ₹ Cr", "Quote ₹ Cr", "Disbursal ₹ Cr", "Value conversion %"]
    product_report = pd.DataFrame(products, columns=columns).sort_values("Disbursal ₹ Cr", ascending=False) if products else pd.DataFrame(columns=columns)
    potential = float(working.loc[working["_disbursed"], "amount"].sum())
    return stage_report, waterfall, product_report, potential

def executive_impact_report(df: pd.DataFrame) -> pd.DataFrame:
    """Build an illustrative, reproducible impact funnel for each model use case."""
    rows = []
    for audience in CAMPAIGN_AUDIENCES:
        view, _ = campaign_audience(df, audience)
        customer_number = pd.to_numeric(
            view["customer_id"].str.extract(r"(\d+)", expand=False), errors="coerce"
        ).fillna(0).astype(int)
        contacted = view[customer_number.mod(10) < 8].copy()

        if audience.startswith("Default"):
            due = contacted["collection_amount_due"].replace(0, pd.NA)
            cure_ratio = (contacted["collection_amount_received"] / due).fillna(0)
            positive = contacted[(cure_ratio >= 0.70) | ((contacted["dpd"] == 0) & (contacted["response"] == 1))]
            impact = float((positive["outstanding"] * positive["risk"] * 0.45).sum())
            impact_type = "Estimated loss avoided"
            outcome_name = "Cured / assisted"
        elif audience.startswith("Churn"):
            positive = contacted[contacted["response"] == 1]
            impact = float(positive["outstanding"].sum())
            impact_type = "Portfolio value protected"
            outcome_name = "Customers retained"
        elif audience.startswith("Next best"):
            positive = contacted[contacted["response"] == 1]
            impact = float((positive["amount"] * 0.01).sum())
            impact_type = "Action value proxy"
            outcome_name = "Actions accepted"
        else:
            positive = contacted[contacted["response"] == 1]
            impact = float(positive["amount"].sum())
            impact_type = "Potential incremental disbursal"
            outcome_name = "Customers converted"

        audience_count = len(view)
        contacted_count = len(contacted)
        positive_count = len(positive)
        rows.append({
            "Use case": audience.split(" — ")[0],
            "Qualified audience": audience_count,
            "Contacted": contacted_count,
            "Positive outcomes": positive_count,
            "Contact rate %": round(contacted_count / audience_count * 100, 2) if audience_count else 0.0,
            "Conversion efficiency %": round(positive_count / contacted_count * 100, 2) if contacted_count else 0.0,
            "Outcome definition": outcome_name,
            "Impact type": impact_type,
            "Estimated impact ₹": round(impact, 2),
        })
    return pd.DataFrame(rows)


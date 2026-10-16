from __future__ import annotations

import math
import re

import pandas as pd


AI_COLUMNS = [
    "customer_id", "name", "branch_code", "city", "product", "customer_segment",
    "bureau", "risk", "propensity", "churn_probability", "cross_sell_score",
    "segment", "recommended_product", "recommended_action", "reasons",
]


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
    result["recommended_product"] = result.apply(_recommended_product, axis=1)
    result["customer_persona"] = result.apply(_customer_persona, axis=1)
    complaint_details = result.apply(_complaint_recommendation, axis=1, result_type="expand")
    complaint_details.columns = ["past_complaint", "complaint_reason", "recommended_action"]
    result[["past_complaint", "complaint_reason", "recommended_action"]] = complaint_details
    return result


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
    if any(term in question for term in ("default", "defaulter", "risk")):
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

    branch_match = re.search(r"\b(br\d{3})\b", q)
    scoped = df
    scope_text = "across the portfolio"
    if branch_match:
        code = branch_match.group(1).upper()
        scoped = df[df["branch_code"].str.upper() == code]
        scope_text = f"for {code}"

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
        return f"{len(result):,} loans have DPD greater than {threshold} {scope_text}.", result, "delinquency_detail"

    if "outstanding" in q or "portfolio" in q:
        total = scoped["outstanding"].sum()
        summary = scoped.groupby("product", as_index=False).agg(
            accounts=("customer_id", "count"), outstanding=("outstanding", "sum")
        ).sort_values("outstanding", ascending=False)
        return f"Total outstanding portfolio is ₹{total / 10_000_000:,.2f} crore {scope_text}.", summary, "portfolio"

    if ("default" in q or "risk" in q) and ("high" in q or "highest" in q):
        result = scoped[scoped["risk"] >= 0.60].sort_values("risk", ascending=False)
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


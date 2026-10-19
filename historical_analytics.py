from __future__ import annotations

import hashlib
import re

import pandas as pd


REGIONS = {
    "Maharashtra": "West", "Gujarat": "West", "Madhya Pradesh": "Central",
    "Delhi": "North", "Rajasthan": "North", "Uttar Pradesh": "North",
    "Karnataka": "South", "Tamil Nadu": "South", "Telangana": "South", "Kerala": "South",
    "West Bengal": "East",
}

# Deliberately stressed, internally consistent pilot outcomes used to demonstrate
# where AI/ML and governed decision automation can create measurable value.
PAIN_SCENARIO = {
    "manual_review_rate": 0.58,
    "default_rate": 0.17,
    "delinquency_rate": 0.29,
    "churn_rate": 0.24,
    "lead_rate": 0.55,
    "application_to_lead_rate": 0.62,
    "evaluation_to_application_rate": 0.84,
    "approval_to_evaluation_rate": 0.72,
    "disbursal_to_lead_rate": 0.18,
    "cross_sell_eligible_rate": 0.42,
    "cross_sell_contact_rate": 0.78,
    "cross_sell_acceptance_rate": 0.16,
    "collection_efficiency": 0.61,
    "good_customer_rejection_rate": 0.06,
    "risky_customer_approval_rate": 0.04,
}


def _stable_fraction(*parts: object) -> float:
    digest = hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def build_historical_portfolio(df: pd.DataFrame, months: int = 36) -> pd.DataFrame:
    """Back-cast one governed monthly mart whose latest month reconciles to the live portfolio."""
    if df.empty:
        return pd.DataFrame()
    work = df.copy()
    work["region"] = work["state"].map(REGIONS).fillna("Other")
    ids = pd.to_numeric(work["customer_id"].str.extract(r"(\d+)", expand=False), errors="coerce").fillna(0).astype(int)
    work["_lead"] = (work["consent"] == 1) & (work["propensity"] >= .60) & (work["risk"] < .60)
    work["_application"] = work["_lead"] & (ids.mod(100) < 72)
    work["_evaluated"] = work["_application"] & (work["kyc"] == 1)
    work["_approved"] = work["_evaluated"] & work["segment"].isin(["pre-approved", "pre-qualified", "underwriting"])
    work["_disbursed"] = work["_approved"] & (work["response"] == 1)
    work["_default"] = (work["risk"] >= .60) | (work["dpd"] >= 60)
    work["_delinquent"] = work["dpd"] > 0
    work["_churn"] = work["churn_probability"] >= .65
    work["_cross_eligible"] = work["cross_sell_score"] >= .70
    work["_cross_contacted"] = work["_cross_eligible"] & (ids.mod(10) < 8)
    work["_cross_accepted"] = work["_cross_contacted"] & (work["response"] == 1)
    work["_manual"] = work["segment"].isin(["underwriting", "review"])
    work["_good_rejected"] = (work["risk"] < .35) & (work["segment"] == "decline")
    work["_risky_approved"] = (work["risk"] >= .60) & work["segment"].isin(["pre-approved", "pre-qualified"])
    dims = ["branch_code", "city", "state", "region", "product", "customer_segment"]
    grouped = work.groupby(dims, dropna=False)
    current = grouped.agg(
        portfolio_amount=("outstanding", "sum"), account_count=("customer_id", "count"),
        leads=("_lead", "sum"), applications=("_application", "sum"), credit_evaluated=("_evaluated", "sum"),
        approved=("_approved", "sum"), disbursed=("_disbursed", "sum"), default_accounts=("_default", "sum"),
        delinquent_accounts=("_delinquent", "sum"), churned_customers=("_churn", "sum"),
        cross_sell_eligible=("_cross_eligible", "sum"), cross_sell_contacted=("_cross_contacted", "sum"),
        cross_sell_accepted=("_cross_accepted", "sum"), collection_due=("collection_amount_due", "sum"),
        collection_received=("collection_amount_received", "sum"), manual_reviews=("_manual", "sum"),
        good_customers_rejected=("_good_rejected", "sum"), risky_customers_approved=("_risky_approved", "sum"),
    ).reset_index()
    accounts = current["account_count"].astype(float)
    current["default_accounts"] = (accounts * PAIN_SCENARIO["default_rate"]).round().astype(int)
    current["delinquent_accounts"] = (accounts * PAIN_SCENARIO["delinquency_rate"]).round().astype(int)
    current["churned_customers"] = (accounts * PAIN_SCENARIO["churn_rate"]).round().astype(int)
    current["manual_reviews"] = (accounts * PAIN_SCENARIO["manual_review_rate"]).round().astype(int)
    current["good_customers_rejected"] = (accounts * PAIN_SCENARIO["good_customer_rejection_rate"]).round().astype(int)
    current["risky_customers_approved"] = (accounts * PAIN_SCENARIO["risky_customer_approval_rate"]).round().astype(int)
    current["leads"] = (accounts * PAIN_SCENARIO["lead_rate"]).round().astype(int)
    current["applications"] = (current["leads"] * PAIN_SCENARIO["application_to_lead_rate"]).round().astype(int)
    current["credit_evaluated"] = (current["applications"] * PAIN_SCENARIO["evaluation_to_application_rate"]).round().astype(int)
    current["approved"] = (current["credit_evaluated"] * PAIN_SCENARIO["approval_to_evaluation_rate"]).round().astype(int)
    current["disbursed"] = (current["leads"] * PAIN_SCENARIO["disbursal_to_lead_rate"]).round().astype(int)
    current["cross_sell_eligible"] = (accounts * PAIN_SCENARIO["cross_sell_eligible_rate"]).round().astype(int)
    current["cross_sell_contacted"] = (current["cross_sell_eligible"] * PAIN_SCENARIO["cross_sell_contact_rate"]).round().astype(int)
    current["cross_sell_accepted"] = (current["cross_sell_contacted"] * PAIN_SCENARIO["cross_sell_acceptance_rate"]).round().astype(int)
    current["collection_received"] = current["collection_due"] * PAIN_SCENARIO["collection_efficiency"]
    end = pd.Timestamp.today().to_period("M").to_timestamp()
    periods = pd.date_range(end=end, periods=months, freq="MS")
    records = []
    count_metrics = [c for c in current.columns if c not in dims + ["portfolio_amount", "collection_due", "collection_received"]]
    for _, base in current.iterrows():
        group_key = tuple(base[d] for d in dims)
        for index, month in enumerate(periods):
            progress = index / max(months - 1, 1)
            growth = 0.68 + 0.32 * progress
            season = 1 + 0.035 * (((month.month + int(_stable_fraction(*group_key) * 7)) % 6) - 2.5) / 2.5
            quality = 0.88 + 0.12 * progress
            row = {d: base[d] for d in dims}
            row["month"] = month
            row["portfolio_amount"] = float(base["portfolio_amount"]) * growth * season
            row["account_count"] = max(1, round(float(base["account_count"]) * growth))
            for metric in count_metrics:
                factor = growth
                if metric in {"default_accounts", "delinquent_accounts", "churned_customers"}:
                    factor = growth * quality
                elif metric == "manual_reviews":
                    factor = growth * (1.10 - 0.10 * progress)
                row[metric] = max(0, round(float(base[metric]) * factor * season))
            row["collection_due"] = float(base["collection_due"]) * growth * quality
            collection_efficiency_shift = 0.90 + 0.10 * progress
            row["collection_received"] = float(base["collection_received"]) * growth * collection_efficiency_shift
            row["new_accounts"] = max(0, round(row["account_count"] * (0.035 + 0.012 * _stable_fraction(month, *group_key))))
            row["avg_turnaround_hours"] = round(64 - 22 * progress + 8 * _stable_fraction(*group_key, month.month), 1)
            records.append(row)
    history = pd.DataFrame(records)
    latest = history["month"].max()
    latest_mask = history["month"] == latest
    for metric in ["portfolio_amount", "account_count", *count_metrics, "collection_due", "collection_received"]:
        history.loc[latest_mask, metric] = current[metric].to_numpy()
    history["default_rate_pct"] = (100 * history["default_accounts"] / history["account_count"].clip(lower=1)).round(2)
    history["delinquency_rate_pct"] = (100 * history["delinquent_accounts"] / history["account_count"].clip(lower=1)).round(2)
    history["churn_rate_pct"] = (100 * history["churned_customers"] / history["account_count"].clip(lower=1)).round(2)
    history["lead_conversion_pct"] = (100 * history["disbursed"] / history["leads"].clip(lower=1)).round(2)
    history["cross_sell_conversion_pct"] = (100 * history["cross_sell_accepted"] / history["cross_sell_contacted"].clip(lower=1)).round(2)
    history["collection_efficiency_pct"] = (100 * history["collection_received"] / history["collection_due"].replace(0, pd.NA)).fillna(100).clip(0, 100).round(2)
    return history


def _aggregate(history: pd.DataFrame, months: list[pd.Timestamp], dimension: str | None = None) -> pd.DataFrame:
    data = history[history["month"].isin(months)]
    keys = [dimension] if dimension else []
    aggregations = {
        "portfolio_amount": "sum", "account_count": "sum", "new_accounts": "sum", "leads": "sum",
        "applications": "sum", "credit_evaluated": "sum", "approved": "sum", "disbursed": "sum",
        "default_accounts": "sum", "delinquent_accounts": "sum", "churned_customers": "sum",
        "cross_sell_eligible": "sum", "cross_sell_contacted": "sum", "cross_sell_accepted": "sum",
        "collection_due": "sum", "collection_received": "sum", "manual_reviews": "sum",
        "good_customers_rejected": "sum", "risky_customers_approved": "sum", "avg_turnaround_hours": "mean",
    }
    result = data.groupby(keys, as_index=False).agg(aggregations) if keys else pd.DataFrame([data.agg(aggregations)])
    result["default_rate_pct"] = 100 * result.default_accounts / result.account_count.clip(lower=1)
    result["churn_rate_pct"] = 100 * result.churned_customers / result.account_count.clip(lower=1)
    result["lead_conversion_pct"] = 100 * result.disbursed / result.leads.clip(lower=1)
    result["cross_sell_conversion_pct"] = 100 * result.cross_sell_accepted / result.cross_sell_contacted.clip(lower=1)
    result["collection_efficiency_pct"] = (100 * result.collection_received / result.collection_due.replace(0, pd.NA)).fillna(100)
    return result


def observe_metrics(history: pd.DataFrame) -> dict[str, object]:
    latest = history.month.max()
    current = _aggregate(history, [latest]).iloc[0]
    previous = _aggregate(history, [latest - pd.DateOffset(years=1)]).iloc[0]
    return {"latest": latest, "current": current, "previous_year": previous}


def _product_filter(question: str, history: pd.DataFrame) -> pd.DataFrame:
    q = question.lower()
    aliases = {
        r"personal\s+l+oan": "Personal Loan", r"home\s+l+oan": "Home Loan",
        r"vehicle\s+l+oan": "Vehicle Loan", r"education\s+l+oan": "Education Loan",
        r"\bmsme\b": "MSME Loan", r"\bagri(?:culture)?\b": "Agri Loan",
    }
    for pattern, product in aliases.items():
        if re.search(pattern, q):
            return history[history["product"] == product]
    return history


def _geography_filter(question: str, history: pd.DataFrame) -> tuple[pd.DataFrame, str | None]:
    """Apply an explicitly named region, city or branch as a scope—not a grouping."""
    q = question.lower()
    for region in sorted(history["region"].dropna().astype(str).unique()):
        if re.search(rf"\b{re.escape(region.lower())}\b", q):
            return history[history["region"].astype(str).str.casefold() == region.casefold()], f"{region} region"
    branch_match = re.search(r"\b(br\d{3})\b", q)
    if branch_match:
        branch = branch_match.group(1).upper()
        return history[history["branch_code"].str.upper() == branch], branch
    for city in sorted(history["city"].dropna().astype(str).unique()):
        if re.search(rf"\b{re.escape(city.lower())}\b", q):
            return history[history["city"].astype(str).str.casefold() == city.casefold()], city
    return history, None


def _recommendation(metric: str, worsened: bool) -> str:
    if not worsened:
        return ""
    mapping = {
        "portfolio_amount": "No direct AI/ML intervention is prescribed; ask the Business Head to validate the portfolio drivers and corrective plan.",
        "new_accounts": "Refer to AI/ML Intelligence → Propensity output to identify qualified acquisition prospects.",
        "default_rate_pct": "Refer to AI/ML Intelligence → Default & early warning output and Credit decisioning output to reduce avoidable credit risk.",
        "churn_rate_pct": "Refer to AI/ML Intelligence → Churn output, then use Campaign Agent for a governed retention campaign.",
        "lead_conversion_pct": "Refer to AI/ML Intelligence → Propensity and Credit decisioning outputs to identify and address funnel leakage.",
        "cross_sell_conversion_pct": "Refer to AI/ML Intelligence → Cross-sell / up-sell and Next Best Offer/Action outputs.",
        "collection_efficiency_pct": "Refer to AI/ML Intelligence → Default & early warning output to prioritize preventive servicing and collection action.",
        "manual_review_rate_pct": "Refer to AI/ML Intelligence → Credit decisioning output to automate policy-clear cases while retaining human review for exceptions.",
    }
    return mapping.get(metric, "Ask the Business Head to validate the drivers and agree the corrective action.")


def _insight_text(answer: str, metric: str, worsened: bool, negative_text: str | None = None) -> str:
    if not worsened:
        return f"{answer}\n\n**AI insight:** Performance is improving for this metric; no intervention is recommended."
    recommendation = _recommendation(metric, True)
    insight = negative_text or "The movement indicates deterioration in this metric and requires attention."
    return f"{answer}\n\n**AI insight:** {insight}\n\n**Recommended action:** {recommendation}"


def _contribution_driver_analysis(question: str, history: pd.DataFrame):
    """Explain why a named geography contributes strongly to a named product portfolio."""
    q = question.lower()
    if not (any(term in q for term in ("driver", "reason", "why")) and "contribut" in q):
        return None
    product_scope = _product_filter(q, history)
    if product_scope.empty or product_scope["product"].nunique() != 1:
        return None
    target_scope, location = _geography_filter(q, product_scope)
    if not location or target_scope.empty:
        return None

    latest = product_scope["month"].max()
    target = _aggregate(target_scope, [latest]).iloc[0]
    product_total = _aggregate(product_scope, [latest]).iloc[0]
    branch_totals = _aggregate(product_scope, [latest], "branch_code").sort_values(
        "portfolio_amount", ascending=False, kind="stable"
    ).reset_index(drop=True)
    target_branch = str(target_scope.iloc[0]["branch_code"])
    rank_matches = branch_totals.index[branch_totals["branch_code"].eq(target_branch)].tolist()
    rank = rank_matches[0] + 1 if rank_matches else None
    branch_count = max(1, len(branch_totals))
    product = str(product_scope.iloc[0]["product"])

    def pct(numerator: float, denominator: float) -> float:
        return 100 * float(numerator) / max(float(denominator), 1.0)

    portfolio_share = pct(target["portfolio_amount"], product_total["portfolio_amount"])
    account_share = pct(target["account_count"], product_total["account_count"])
    lead_share = pct(target["leads"], product_total["leads"])
    disbursal_share = pct(target["disbursed"], product_total["disbursed"])
    target_ticket_lakh = float(target["portfolio_amount"]) / max(float(target["account_count"]), 1.0) / 100_000
    product_ticket_lakh = float(product_total["portfolio_amount"]) / max(float(product_total["account_count"]), 1.0) / 100_000
    target_conversion = pct(target["disbursed"], target["leads"])
    product_conversion = pct(product_total["disbursed"], product_total["leads"])
    application_conversion = pct(target["applications"], target["leads"])
    approval_conversion = pct(target["approved"], target["credit_evaluated"])

    equal_branch_share = 100 / branch_count
    scale_signal = (
        "an above-average account base"
        if account_share >= equal_branch_share
        else "high portfolio value despite a below-average account base"
    )
    ticket_signal = "higher" if target_ticket_lakh > product_ticket_lakh else "lower"
    conversion_signal = "above" if target_conversion > product_conversion else "below"
    rank_text = f"ranks #{rank} of {branch_count}" if rank else "is a leading branch"
    answer = (
        f"{location} ({target_branch}) {rank_text} in {product}, contributing "
        f"₹{float(target['portfolio_amount']) / 10_000_000:,.2f} Cr ({portfolio_share:.2f}% of the product portfolio). "
        f"The contribution is primarily explained by {scale_signal}: {int(target['account_count']):,} accounts "
        f"({account_share:.2f}% share), plus an average outstanding of ₹{target_ticket_lakh:,.2f} lakh, "
        f"which is {ticket_signal} than the product average of ₹{product_ticket_lakh:,.2f} lakh. "
        f"The branch generated {int(target['leads']):,} leads and {int(target['disbursed']):,} disbursals; "
        f"lead-to-disbursal conversion is {target_conversion:.2f}%, {conversion_signal} the product benchmark of "
        f"{product_conversion:.2f}%."
    )
    if target_conversion < product_conversion:
        answer += " Funnel leakage is therefore an improvement opportunity, not the reason for the branch's high current contribution."
    else:
        answer += " Lead volume and conversion reinforce the branch's portfolio contribution."

    result = pd.DataFrame([
        {"Metric": "Portfolio contribution", "Branch value": round(portfolio_share, 2), "Product benchmark": round(100 / branch_count, 2), "Unit": "%"},
        {"Metric": "Account contribution", "Branch value": round(account_share, 2), "Product benchmark": round(100 / branch_count, 2), "Unit": "%"},
        {"Metric": "Lead contribution", "Branch value": round(lead_share, 2), "Product benchmark": round(100 / branch_count, 2), "Unit": "%"},
        {"Metric": "Disbursal contribution", "Branch value": round(disbursal_share, 2), "Product benchmark": round(100 / branch_count, 2), "Unit": "%"},
        {"Metric": "Lead-to-disbursal conversion", "Branch value": round(target_conversion, 2), "Product benchmark": round(product_conversion, 2), "Unit": "%"},
        {"Metric": "Application-to-lead conversion", "Branch value": round(application_conversion, 2), "Product benchmark": round(pct(product_total['applications'], product_total['leads']), 2), "Unit": "%"},
        {"Metric": "Approval-to-evaluation conversion", "Branch value": round(approval_conversion, 2), "Product benchmark": round(pct(product_total['approved'], product_total['credit_evaluated']), 2), "Unit": "%"},
        {"Metric": "Average outstanding per account", "Branch value": round(target_ticket_lakh, 2), "Product benchmark": round(product_ticket_lakh, 2), "Unit": "₹ lakh"},
    ])
    return answer, result, "historical:portfolio_contribution:drivers"


def _operational_growth_driver_analysis(question: str, history: pd.DataFrame):
    """Explain named product/geography growth through balance-sheet and funnel movements."""
    q = question.lower()
    asks_driver = any(term in q for term in ("driver", "reason", "why"))
    # A named branch + product driver question implies growth/contribution analysis even
    # when an executive omits the word "growth". Contribution questions retain their
    # separate point-in-time handler below.
    if not asks_driver or "contribut" in q:
        return None
    product_scope = _product_filter(q, history)
    if product_scope.empty:
        return None
    target_scope, location = _geography_filter(q, product_scope)
    if target_scope.empty:
        return None
    # Product-only questions are bank-wide; named city/branch/region questions retain
    # their explicit geography. Both use one consistent growth-driver KPI contract.
    location = location or "Bank-wide"

    latest = target_scope["month"].max()
    prior_month = latest - pd.DateOffset(years=1)
    current_frame = _aggregate(target_scope, [latest])
    prior_frame = _aggregate(target_scope, [prior_month])
    if current_frame.empty or prior_frame.empty:
        return None
    current, prior = current_frame.iloc[0], prior_frame.iloc[0]
    product = (
        str(product_scope.iloc[0]["product"])
        if product_scope["product"].nunique() == 1
        else "Lending"
    )
    scope_name = f"{location} portfolio" if product == "Lending" else f"{location} {product} portfolio"

    def growth(current_value: float, prior_value: float) -> float:
        return 100 * (float(current_value) - float(prior_value)) / max(float(prior_value), 1.0)

    def rate(row: pd.Series, numerator: str, denominator: str) -> float:
        return 100 * float(row[numerator]) / max(float(row[denominator]), 1.0)

    portfolio_growth = growth(current["portfolio_amount"], prior["portfolio_amount"])
    account_growth = growth(current["account_count"], prior["account_count"])
    lead_growth = growth(current["leads"], prior["leads"])
    new_account_growth = growth(current["new_accounts"], prior["new_accounts"])
    disbursal_growth = growth(current["disbursed"], prior["disbursed"])
    campaign_contact_growth = growth(current["cross_sell_contacted"], prior["cross_sell_contacted"])
    campaign_response_growth = growth(current["cross_sell_accepted"], prior["cross_sell_accepted"])
    current_ticket = float(current["portfolio_amount"]) / max(float(current["account_count"]), 1.0)
    prior_ticket = float(prior["portfolio_amount"]) / max(float(prior["account_count"]), 1.0)
    ticket_growth = growth(current_ticket, prior_ticket)

    current_application_rate = rate(current, "applications", "leads")
    prior_application_rate = rate(prior, "applications", "leads")
    current_approval_rate = rate(current, "approved", "credit_evaluated")
    prior_approval_rate = rate(prior, "approved", "credit_evaluated")
    current_lead_conversion = rate(current, "disbursed", "leads")
    prior_lead_conversion = rate(prior, "disbursed", "leads")
    current_campaign_conversion = rate(current, "cross_sell_accepted", "cross_sell_contacted")
    prior_campaign_conversion = rate(prior, "cross_sell_accepted", "cross_sell_contacted")
    current_approval_disbursal = rate(current, "disbursed", "approved")
    prior_approval_disbursal = rate(prior, "disbursed", "approved")
    application_change = current_application_rate - prior_application_rate
    approval_change = current_approval_rate - prior_approval_rate
    lead_conversion_change = current_lead_conversion - prior_lead_conversion
    approval_disbursal_change = current_approval_disbursal - prior_approval_disbursal
    campaign_conversion_change = current_campaign_conversion - prior_campaign_conversion

    positive_volume = lead_growth > 0 or new_account_growth > 0 or account_growth > 0
    positive_efficiency = lead_conversion_change > 0 and approval_disbursal_change > 0
    positive_campaign = campaign_response_growth > 0 or campaign_conversion_change > 0
    if positive_volume and positive_efficiency and positive_campaign:
        interpretation = "Growth is supported by account acquisition, stronger funnel efficiency and improving campaign outcomes."
    elif positive_volume and positive_efficiency:
        interpretation = "Growth is supported by account acquisition and funnel efficiency, while campaign outcomes are not an incremental driver."
    elif positive_volume:
        interpretation = (
            "Growth is volume-led, while end-to-end funnel efficiency weakened. The larger lead and account base "
            f"offset the lower lead-to-disbursal conversion. Campaign responses changed {campaign_response_growth:+.2f}% "
            f"and campaign conversion changed {campaign_conversion_change:+.2f} pp; approval and disbursal leakage remain improvement opportunities."
        )
    elif positive_campaign:
        interpretation = "Campaign response or conversion improved, but it has not yet translated into broad account and funnel growth."
    elif positive_efficiency:
        interpretation = "Growth is efficiency-led despite limited lead or account expansion."
    else:
        interpretation = "The measured portfolio movement is mainly ticket-size led rather than acquisition or funnel led."

    answer = (
        f"{scope_name} grew {portfolio_growth:+.2f}% YoY "
        f"(₹{float(prior['portfolio_amount']) / 10_000_000:,.2f} Cr to "
        f"₹{float(current['portfolio_amount']) / 10_000_000:,.2f} Cr). "
        f"Balance-sheet drivers: accounts changed {account_growth:+.2f}% and average outstanding per account "
        f"changed {ticket_growth:+.2f}%. Acquisition drivers: leads changed {lead_growth:+.2f}%, new accounts "
        f"{new_account_growth:+.2f}%, and disbursals {disbursal_growth:+.2f}%. Funnel efficiency: "
        f"application-to-lead changed {application_change:+.2f} pp, approval-to-evaluation {approval_change:+.2f} pp, "
        f"approval-to-disbursal {approval_disbursal_change:+.2f} pp, and lead-to-disbursal "
        f"{lead_conversion_change:+.2f} pp. Campaign drivers: contacted customers changed "
        f"{campaign_contact_growth:+.2f}%, responses {campaign_response_growth:+.2f}%, and campaign conversion "
        f"{campaign_conversion_change:+.2f} pp. **Driver interpretation:** {interpretation}"
    )
    result = pd.DataFrame([
        {"Driver": "Portfolio amount", "Prior year": round(float(prior["portfolio_amount"]) / 10_000_000, 2), "Current": round(float(current["portfolio_amount"]) / 10_000_000, 2), "Change": round(portfolio_growth, 2), "Unit": "₹ Cr / %"},
        {"Driver": "Accounts", "Prior year": int(prior["account_count"]), "Current": int(current["account_count"]), "Change": round(account_growth, 2), "Unit": "count / %"},
        {"Driver": "Average outstanding per account", "Prior year": round(prior_ticket / 100_000, 2), "Current": round(current_ticket / 100_000, 2), "Change": round(ticket_growth, 2), "Unit": "₹ lakh / %"},
        {"Driver": "Leads", "Prior year": int(prior["leads"]), "Current": int(current["leads"]), "Change": round(lead_growth, 2), "Unit": "count / %"},
        {"Driver": "New accounts", "Prior year": int(prior["new_accounts"]), "Current": int(current["new_accounts"]), "Change": round(new_account_growth, 2), "Unit": "count / %"},
        {"Driver": "Disbursals", "Prior year": int(prior["disbursed"]), "Current": int(current["disbursed"]), "Change": round(disbursal_growth, 2), "Unit": "count / %"},
        {"Driver": "Application-to-lead", "Prior year": round(prior_application_rate, 2), "Current": round(current_application_rate, 2), "Change": round(application_change, 2), "Unit": "% / pp"},
        {"Driver": "Approval-to-evaluation", "Prior year": round(prior_approval_rate, 2), "Current": round(current_approval_rate, 2), "Change": round(approval_change, 2), "Unit": "% / pp"},
        {"Driver": "Approval-to-disbursal", "Prior year": round(prior_approval_disbursal, 2), "Current": round(current_approval_disbursal, 2), "Change": round(approval_disbursal_change, 2), "Unit": "% / pp"},
        {"Driver": "Lead-to-disbursal", "Prior year": round(prior_lead_conversion, 2), "Current": round(current_lead_conversion, 2), "Change": round(lead_conversion_change, 2), "Unit": "% / pp"},
        {"Driver": "Campaign contacts", "Prior year": int(prior["cross_sell_contacted"]), "Current": int(current["cross_sell_contacted"]), "Change": round(campaign_contact_growth, 2), "Unit": "count / %"},
        {"Driver": "Campaign responses", "Prior year": int(prior["cross_sell_accepted"]), "Current": int(current["cross_sell_accepted"]), "Change": round(campaign_response_growth, 2), "Unit": "count / %"},
        {"Driver": "Campaign conversion", "Prior year": round(prior_campaign_conversion, 2), "Current": round(current_campaign_conversion, 2), "Change": round(campaign_conversion_change, 2), "Unit": "% / pp"},
    ])
    return answer, result, "historical:executive:operational_growth_drivers"


def _funnel_leakage_analysis(question: str, history: pd.DataFrame):
    """Measure lending-funnel leakage independently of collection efficiency."""
    q = question.lower()
    if "funnel" not in q or "leak" not in q:
        return None
    scoped = _product_filter(q, history)
    latest = scoped["month"].max()

    if "branch" in q and any(term in q for term in ("least", "lowest", "minimum", "best")):
        result = _aggregate(scoped, [latest], "branch_code")
        result["funnel_leakage_pct"] = (
            100 * (result["leads"] - result["disbursed"]) / result["leads"].clip(lower=1)
        ).round(2)
        result["lead_to_disbursal_pct"] = (
            100 * result["disbursed"] / result["leads"].clip(lower=1)
        ).round(2)
        city_map = (
            scoped.loc[scoped["month"].eq(latest), ["branch_code", "city"]]
            .drop_duplicates("branch_code").set_index("branch_code")["city"]
        )
        result["city"] = result["branch_code"].map(city_map).fillna("")
        result = result.sort_values(
            ["funnel_leakage_pct", "lead_to_disbursal_pct", "branch_code"],
            ascending=[True, False, True], kind="stable",
        ).reset_index(drop=True)
        selected = result.head(1)[
            ["branch_code", "city", "leads", "applications", "credit_evaluated", "approved", "disbursed", "lead_to_disbursal_pct", "funnel_leakage_pct"]
        ].copy()
        row = selected.iloc[0]
        answer = (
            f"{row['city']} ({row['branch_code']}) has the least end-to-end funnel leakage at "
            f"{row['funnel_leakage_pct']:.2f}%: {int(row['leads']):,} leads produced "
            f"{int(row['disbursed']):,} disbursals ({row['lead_to_disbursal_pct']:.2f}% conversion)."
        )
        return answer, selected, "historical:funnel_leakage:branch_ranking"

    target_scope, location = _geography_filter(q, scoped)
    if target_scope.empty:
        return None
    if not location:
        location = "Authorized bank portfolio"
    current = _aggregate(target_scope, [latest]).iloc[0]
    stages = [
        ("Lead → Application", "leads", "applications", "AI/ML Intelligence → Propensity and Campaign Agent"),
        ("Application → Credit evaluation", "applications", "credit_evaluated", "digital onboarding and KYC workflow"),
        ("Credit evaluation → Approval", "credit_evaluated", "approved", "AI/ML Intelligence → Credit decisioning"),
        ("Approval → Disbursal", "approved", "disbursed", "Campaign Agent with approved-but-not-disbursed follow-up"),
    ]
    rows = []
    for stage, start_column, end_column, action in stages:
        start, end = int(current[start_column]), int(current[end_column])
        leakage = max(0, start - end)
        leakage_pct = 100 * leakage / max(start, 1)
        rows.append({
            "Funnel stage": stage, "Start": start, "Completed": end,
            "Drop-off": leakage, "Leakage %": round(leakage_pct, 2), "Recommended module": action,
        })
    result = pd.DataFrame(rows).sort_values(["Leakage %", "Drop-off"], ascending=False, kind="stable")
    largest = result.iloc[0]
    total_leakage_pct = 100 * (float(current["leads"]) - float(current["disbursed"])) / max(float(current["leads"]), 1.0)
    answer = (
        f"{location} has {int(current['leads']):,} leads and {int(current['disbursed']):,} disbursals, "
        f"so end-to-end funnel leakage is {total_leakage_pct:.2f}%. The largest stage leakage is "
        f"{largest['Funnel stage']} at {largest['Leakage %']:.2f}% ({int(largest['Drop-off']):,} cases). "
        f"**Recommended action:** Use Propensity output to prioritize qualified leads and Campaign Agent for "
        f"conversion follow-up; address the largest stage through {largest['Recommended module']}. "
        "Use Credit decisioning only for evaluation-to-approval leakage—not collections or Default & early warning."
    )
    return answer, result, "historical:funnel_leakage:diagnosis"


def _portfolio_improvement_analysis(question: str, history: pd.DataFrame):
    """Return an executive portfolio-growth recommendation grounded in current funnel KPIs."""
    q = question.lower()
    asks_improvement = any(term in q for term in ("improve", "increase", "grow", "accelerate"))
    asks_business = any(term in q for term in ("business portfolio", "overall business", "overall portfolio"))
    if not (asks_improvement and asks_business):
        return None
    scoped = _product_filter(q, history)
    scoped, location = _geography_filter(q, scoped)
    latest = scoped["month"].max()
    current = _aggregate(scoped, [latest]).iloc[0]
    prior = _aggregate(scoped, [latest - pd.DateOffset(years=1)]).iloc[0]

    def growth(current_value: float, prior_value: float) -> float:
        return 100 * (float(current_value) - float(prior_value)) / max(float(prior_value), 1.0)

    portfolio_growth = growth(current["portfolio_amount"], prior["portfolio_amount"])
    account_growth = growth(current["account_count"], prior["account_count"])
    approved_not_disbursed = max(0, int(current["approved"] - current["disbursed"]))
    funnel_leakage = 100 * (float(current["leads"]) - float(current["disbursed"])) / max(float(current["leads"]), 1.0)
    cross_sell_conversion = 100 * float(current["cross_sell_accepted"]) / max(float(current["cross_sell_contacted"]), 1.0)
    scope_label = location or "Authorized bank portfolio"
    momentum = "growing" if portfolio_growth >= 0 else "contracting"
    answer = (
        f"{scope_label} is ₹{float(current['portfolio_amount']) / 10_000_000:,.2f} Cr and is {momentum} "
        f"{portfolio_growth:+.2f}% YoY; accounts changed {account_growth:+.2f}%. "
        f"**Recommended actions:** (1) use Propensity output to prioritize qualified acquisition leads; "
        f"(2) use Campaign Agent to follow up {approved_not_disbursed:,} approved-but-not-disbursed cases; "
        f"(3) use Cross-sell / up-sell and Next Best Offer/Action to improve the current {cross_sell_conversion:.2f}% "
        f"cross-sell conversion; and (4) use Credit decisioning to reduce evaluation and approval leakage. "
        f"Current end-to-end funnel leakage is {funnel_leakage:.2f}%, so conversion improvement is the clearest near-term lever."
    )
    result = pd.DataFrame([
        {"KPI": "Outstanding portfolio", "Value": round(float(current["portfolio_amount"]) / 10_000_000, 2), "Unit": "₹ Cr"},
        {"KPI": "Portfolio growth YoY", "Value": round(portfolio_growth, 2), "Unit": "%"},
        {"KPI": "Account growth YoY", "Value": round(account_growth, 2), "Unit": "%"},
        {"KPI": "Approved but not disbursed", "Value": approved_not_disbursed, "Unit": "cases"},
        {"KPI": "Funnel leakage", "Value": round(funnel_leakage, 2), "Unit": "%"},
        {"KPI": "Cross-sell conversion", "Value": round(cross_sell_conversion, 2), "Unit": "%"},
    ])
    return answer, result, "historical:portfolio:improvement"


def _explicit_month_account_comparison(question: str, history: pd.DataFrame):
    """Compare newly opened account counts for two explicitly named months."""
    q = question.lower()
    if not ("account" in q and any(term in q for term in ("open", "new", "added"))):
        return None
    month_numbers = {
        "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
        "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
    }
    mentioned = [(q.find(name), name, number) for name, number in month_numbers.items() if name in q]
    mentioned.sort()
    if len(mentioned) < 2:
        return None
    (_, current_name, current_number), (_, prior_name, prior_number) = mentioned[:2]
    scoped = _product_filter(q, history)
    scoped, location = _geography_filter(q, scoped)
    latest = scoped["month"].max()

    def most_recent_month(month_number: int) -> pd.Timestamp:
        candidates = sorted(scoped.loc[(scoped["month"].dt.month == month_number) & (scoped["month"] <= latest), "month"].unique())
        return pd.Timestamp(candidates[-1])

    current_period = most_recent_month(current_number)
    prior_period = most_recent_month(prior_number)
    current_accounts = int(_aggregate(scoped, [current_period]).iloc[0]["new_accounts"])
    prior_accounts = int(_aggregate(scoped, [prior_period]).iloc[0]["new_accounts"])
    absolute_change = current_accounts - prior_accounts
    growth_pct = 100 * absolute_change / max(prior_accounts, 1)
    scope_label = location or "Authorized bank portfolio"
    answer = (
        f"{scope_label} opened {current_accounts:,} new accounts in {current_name.title()} {current_period.year}, "
        f"compared with {prior_accounts:,} in {prior_name.title()} {prior_period.year}. "
        f"That is a change of {absolute_change:+,} accounts ({growth_pct:+.2f}%)."
    )
    result = pd.DataFrame([
        {"Period": f"{prior_name.title()} {prior_period.year}", "New accounts": prior_accounts},
        {"Period": f"{current_name.title()} {current_period.year}", "New accounts": current_accounts},
    ])
    result["Change accounts"] = [pd.NA, absolute_change]
    result["Growth %"] = [pd.NA, round(growth_pct, 2)]
    return answer, result, "historical:new_accounts:explicit_months"


def _branch_growth_risk_leakage_analysis(question: str, history: pd.DataFrame):
    """Find branches growing their book while risk and funnel leakage deteriorate."""
    q = question.lower()
    required = (
        "branch" in q
        and "portfolio growth" in q
        and ("default risk" in q or "default rate" in q)
        and "funnel leakage" in q
        and any(term in q for term in ("worsening", "worsen", "deteriorat", "increasing"))
    )
    if not required:
        return None

    latest = history["month"].max()
    prior = latest - pd.DateOffset(years=1)
    current = _aggregate(history, [latest], "branch_code")
    previous = _aggregate(history, [prior], "branch_code")
    result = current.merge(previous, on="branch_code", suffixes=("_current", "_previous"))
    city_lookup = (
        history.loc[history["month"] == latest, ["branch_code", "city"]]
        .drop_duplicates("branch_code")
    )
    result = result.merge(city_lookup, on="branch_code", how="left")
    result["Portfolio growth %"] = 100 * (
        result["portfolio_amount_current"] - result["portfolio_amount_previous"]
    ) / result["portfolio_amount_previous"].replace(0, pd.NA)
    result["Default rate current %"] = result["default_rate_pct_current"]
    result["Default rate previous %"] = result["default_rate_pct_previous"]
    result["Default change pp"] = (
        result["default_rate_pct_current"] - result["default_rate_pct_previous"]
    )
    result["Funnel leakage current %"] = 100 * (
        1 - result["disbursed_current"] / result["leads_current"].clip(lower=1)
    )
    result["Funnel leakage previous %"] = 100 * (
        1 - result["disbursed_previous"] / result["leads_previous"].clip(lower=1)
    )
    result["Funnel leakage change pp"] = (
        result["Funnel leakage current %"] - result["Funnel leakage previous %"]
    )
    result = result[
        (result["Portfolio growth %"] > 0)
        & (result["Default change pp"] > 0)
        & (result["Funnel leakage change pp"] > 0)
    ].copy()
    result = result.sort_values(
        ["Default change pp", "Funnel leakage change pp"], ascending=False, kind="stable"
    )
    result = result[[
        "branch_code", "city", "Portfolio growth %", "Default rate current %",
        "Default change pp", "Funnel leakage current %", "Funnel leakage change pp",
    ]].rename(columns={"branch_code": "Branch", "city": "City"}).round(2)

    if result.empty:
        answer = (
            "No branch currently has positive portfolio growth together with both a worsening "
            "default rate and worsening funnel leakage."
        )
    else:
        leaders = ", ".join(
            f"{row.City} ({row.Branch})" for row in result.head(3).itertuples()
        )
        answer = (
            f"{len(result)} branch{'es' if len(result) != 1 else ''} meet all three conditions. "
            f"The highest-priority branches are {leaders}."
        )
    return answer, result, "historical:executive:branch_growth_risk_leakage"


def _city_performance_comparison(question: str, history: pd.DataFrame):
    """Compare two named cities across growth, default risk and funnel efficiency."""
    q = question.lower()
    named_cities = [
        city for city in sorted(history["city"].dropna().astype(str).unique())
        if re.search(rf"\b{re.escape(city.lower())}\b", q)
    ]
    required = (
        len(named_cities) == 2
        and any(term in q for term in ("compare", "versus", " vs "))
        and "portfolio growth" in q
        and ("default risk" in q or "default rate" in q)
        and ("lead-conversion" in q or "lead conversion" in q)
    )
    if not required:
        return None

    latest = history["month"].max()
    prior = latest - pd.DateOffset(years=1)
    rows = []
    for city in named_cities:
        scoped = history[history["city"].astype(str).str.casefold() == city.casefold()]
        current = _aggregate(scoped, [latest]).iloc[0]
        previous = _aggregate(scoped, [prior]).iloc[0]
        growth = 100 * (current.portfolio_amount - previous.portfolio_amount) / max(previous.portfolio_amount, 1)
        default_change = current.default_rate_pct - previous.default_rate_pct
        conversion_change = current.lead_conversion_pct - previous.lead_conversion_pct
        rows.append({
            "City": city,
            "Portfolio growth %": growth,
            "Default rate current %": current.default_rate_pct,
            "Default change pp": default_change,
            "Lead conversion current %": current.lead_conversion_pct,
            "Lead conversion change pp": conversion_change,
        })
    result = pd.DataFrame(rows)
    # Transparent equal-weight score: reward growth and current conversion, penalize current default risk.
    result["Performance score"] = (
        result["Portfolio growth %"]
        + result["Lead conversion current %"]
        - result["Default rate current %"]
    )
    result = result.sort_values("Performance score", ascending=False, kind="stable").round(2)
    best = result.iloc[0]
    other = result.iloc[1]
    answer = (
        f"{best['City']} is the better-performing branch on the balanced comparison: portfolio growth "
        f"{best['Portfolio growth %']:+.2f}%, default-rate movement {best['Default change pp']:+.2f} pp, "
        f"and lead-conversion movement {best['Lead conversion change pp']:+.2f} pp. "
        f"{other['City']} records {other['Portfolio growth %']:+.2f}%, "
        f"{other['Default change pp']:+.2f} pp and {other['Lead conversion change pp']:+.2f} pp respectively. "
        "The balanced score equally rewards portfolio growth and current lead conversion, and penalizes current default rate."
    )
    return answer, result, "historical:executive:city_performance_comparison"


def _executive_multi_kpi_analysis(question: str, history: pd.DataFrame):
    """Answer recurring executive questions from governed, reconciled KPI formulas."""
    q = question.lower()
    latest = history["month"].max()
    prior = latest - pd.DateOffset(years=1)

    def comparison(dimension: str) -> pd.DataFrame:
        current = _aggregate(history, [latest], dimension)
        previous = _aggregate(history, [prior], dimension)
        frame = current.merge(previous, on=dimension, suffixes=("_current", "_previous"))
        frame["Portfolio growth %"] = 100 * (
            frame["portfolio_amount_current"] - frame["portfolio_amount_previous"]
        ) / frame["portfolio_amount_previous"].replace(0, pd.NA)
        frame["Default change pp"] = frame["default_rate_pct_current"] - frame["default_rate_pct_previous"]
        frame["Lead conversion change pp"] = (
            frame["lead_conversion_pct_current"] - frame["lead_conversion_pct_previous"]
        )
        frame["Approval-to-disbursal current %"] = 100 * frame["disbursed_current"] / frame["approved_current"].clip(lower=1)
        frame["Approval-to-disbursal previous %"] = 100 * frame["disbursed_previous"] / frame["approved_previous"].clip(lower=1)
        frame["Approval-to-disbursal change pp"] = (
            frame["Approval-to-disbursal current %"] - frame["Approval-to-disbursal previous %"]
        )
        frame["New-account growth %"] = 100 * (
            frame["new_accounts_current"] - frame["new_accounts_previous"]
        ) / frame["new_accounts_previous"].replace(0, pd.NA)
        frame["Manual underwriting %"] = 100 * frame["manual_reviews_current"] / frame["account_count_current"].clip(lower=1)
        frame["Approval rate %"] = 100 * frame["approved_current"] / frame["credit_evaluated_current"].clip(lower=1)
        return frame

    if (
        "product" in q and "grown" in q and "approval" in q and "disbursal" in q
        and any(term in q for term in ("decline", "fall", "drop"))
    ):
        frame = comparison("product")
        result = frame[
            (frame["Portfolio growth %"] > 0) & (frame["Approval-to-disbursal change pp"] < 0)
        ][["product", "Portfolio growth %", "Approval-to-disbursal current %", "Approval-to-disbursal change pp"]]
        result = result.rename(columns={"product": "Product"}).sort_values(
            "Approval-to-disbursal change pp", kind="stable"
        ).round(2)
        answer = (
            "No product meets both conditions."
            if result.empty else
            f"{len(result)} product{'s' if len(result) != 1 else ''} grew YoY while approval-to-disbursal conversion declined; "
            f"{result.iloc[0]['Product']} has the largest conversion deterioration."
        )
        return answer, result, "historical:executive:product_growth_conversion"

    if "branch" in q and "lead" in q and "weak" in q and "disbursal" in q:
        frame = comparison("branch_code")
        lead_cutoff = frame["leads_current"].median()
        conversion_cutoff = frame["lead_conversion_pct_current"].median()
        result = frame[
            (frame["leads_current"] >= lead_cutoff)
            & (frame["lead_conversion_pct_current"] < conversion_cutoff)
        ][["branch_code", "leads_current", "disbursed_current", "lead_conversion_pct_current"]]
        result = result.rename(columns={
            "branch_code": "Branch", "leads_current": "Leads", "disbursed_current": "Disbursed",
            "lead_conversion_pct_current": "Lead conversion %",
        }).sort_values("Lead conversion %", kind="stable").round(2)
        answer = (
            "No branch has above-median leads together with below-median conversion."
            if result.empty else
            f"{len(result)} branches have strong lead volume but weak disbursal conversion; "
            f"{result.iloc[0]['Branch']} has the lowest conversion among them."
        )
        return answer, result, "historical:executive:branch_lead_conversion"

    if "branch" in q and "new-account growth" in q and "declin" in q and "portfolio" in q and "increas" in q:
        frame = comparison("branch_code")
        result = frame[
            (frame["New-account growth %"] < 0) & (frame["Portfolio growth %"] > 0)
        ][["branch_code", "New-account growth %", "Portfolio growth %"]]
        result = result.rename(columns={"branch_code": "Branch"}).sort_values(
            "New-account growth %", kind="stable"
        ).round(2)
        answer = (
            "No branch shows declining new-account growth alongside a growing portfolio."
            if result.empty else
            f"{len(result)} branches show declining new-account growth despite portfolio expansion; "
            f"{result.iloc[0]['Branch']} has the steepest acquisition decline."
        )
        return answer, result, "historical:executive:accounts_down_portfolio_up"

    if "product" in q and "bank average" in q and "default" in q and "growing faster" in q:
        frame = comparison("product")
        bank_current = _aggregate(history, [latest]).iloc[0]
        bank_previous = _aggregate(history, [prior]).iloc[0]
        bank_growth = 100 * (bank_current.portfolio_amount - bank_previous.portfolio_amount) / max(bank_previous.portfolio_amount, 1)
        bank_default = float(bank_current.default_rate_pct)
        result = frame[
            (frame["Portfolio growth %"] > bank_growth)
            & (frame["default_rate_pct_current"] < bank_default)
        ][["product", "Portfolio growth %", "default_rate_pct_current"]]
        result = result.rename(columns={"product": "Product", "default_rate_pct_current": "Default rate %"})
        result["Bank growth benchmark %"] = bank_growth
        result["Bank default benchmark %"] = bank_default
        result = result.sort_values("Portfolio growth %", ascending=False, kind="stable").round(2)
        answer = (
            "No product currently beats both bank benchmarks."
            if result.empty else
            f"{len(result)} products grow faster than the bank while maintaining lower default risk; "
            f"{result.iloc[0]['Product']} leads on growth."
        )
        return answer, result, "historical:executive:growth_risk_benchmark"

    if "branch" in q and "manual underwriting" in q and "approval" in q:
        frame = comparison("branch_code")
        result = frame[
            (frame["Manual underwriting %"] >= frame["Manual underwriting %"].median())
            & (frame["Approval rate %"] < frame["Approval rate %"].median())
            & (frame["leads_current"] >= frame["leads_current"].median())
        ][["branch_code", "Manual underwriting %", "Approval rate %", "leads_current"]]
        result = result.rename(columns={"branch_code": "Branch", "leads_current": "Eligible lead volume"})
        result = result.sort_values("Manual underwriting %", ascending=False, kind="stable").round(2)
        answer = (
            "No branch meets all three underwriting-opportunity conditions."
            if result.empty else
            f"{len(result)} branches combine high manual underwriting, low approval and strong lead volume; "
            f"{result.iloc[0]['Branch']} has the highest manual-underwriting rate."
        )
        return answer, result, "historical:executive:underwriting_opportunity"

    if "portfolio" in q and "growth-to-risk" in q and any(term in q for term in ("best", "trade-off", "tradeoff")):
        frame = comparison("product")
        frame["Growth-risk score"] = frame["Portfolio growth %"] - frame["default_rate_pct_current"]
        result = frame[["product", "Portfolio growth %", "default_rate_pct_current", "Growth-risk score"]]
        result = result.rename(columns={"product": "Product", "default_rate_pct_current": "Default rate %"})
        result = result.sort_values("Growth-risk score", ascending=False, kind="stable").round(2)
        leader = result.iloc[0]
        answer = (
            f"{leader['Product']} has the best measured growth-to-risk trade-off: "
            f"{leader['Portfolio growth %']:+.2f}% growth with a {leader['Default rate %']:.2f}% default rate."
        )
        return answer, result, "historical:executive:growth_risk_tradeoff"

    if "overall lending performance" in q and ("three" in q or "3" in q) and "opportunit" in q:
        current = _aggregate(history, [latest]).iloc[0]
        previous = _aggregate(history, [prior]).iloc[0]
        measures = pd.DataFrame([
            {"Opportunity": "Reduce funnel leakage", "Current %": 100 * (1 - current.disbursed / max(current.leads, 1)), "Change pp": -(current.lead_conversion_pct - previous.lead_conversion_pct), "Suggested module": "Propensity and Credit decisioning"},
            {"Opportunity": "Reduce default risk", "Current %": current.default_rate_pct, "Change pp": current.default_rate_pct - previous.default_rate_pct, "Suggested module": "Default & early warning"},
            {"Opportunity": "Automate manual underwriting", "Current %": 100 * current.manual_reviews / max(current.account_count, 1), "Change pp": 100 * current.manual_reviews / max(current.account_count, 1) - 100 * previous.manual_reviews / max(previous.account_count, 1), "Suggested module": "Credit decisioning"},
            {"Opportunity": "Reduce attrition", "Current %": current.churn_rate_pct, "Change pp": current.churn_rate_pct - previous.churn_rate_pct, "Suggested module": "Churn prediction"},
        ])
        result = measures.sort_values(["Current %", "Change pp"], ascending=False, kind="stable").head(3).round(2)
        answer = "The three largest measured opportunities are " + ", ".join(result["Opportunity"].str.lower()) + "."
        return answer, result, "historical:executive:top_opportunities"
    return None


def answer_historical_bi(question: str, history: pd.DataFrame):
    q = question.lower().strip()
    markers = (
        "growth", "driver", "drill down", "drilldown", "month", "quarter", "yoy", "mom", "qoq",
        "last year", "last quarter", "trend", "new account", "performing", "funnel", "leakage",
        "improve", "increase", "accounts opened", "account opened", "september", "august",
        "underwriting", "trade-off", "tradeoff", "opportunities", "lead", "grown", "growing",
    )
    if history.empty or not any(marker in q for marker in markers):
        return None
    branch_growth_risk = _branch_growth_risk_leakage_analysis(q, history)
    if branch_growth_risk is not None:
        return branch_growth_risk
    city_comparison = _city_performance_comparison(q, history)
    if city_comparison is not None:
        return city_comparison
    multi_kpi = _executive_multi_kpi_analysis(q, history)
    if multi_kpi is not None:
        return multi_kpi
    portfolio_improvement = _portfolio_improvement_analysis(q, history)
    if portfolio_improvement is not None:
        return portfolio_improvement
    month_account_comparison = _explicit_month_account_comparison(q, history)
    if month_account_comparison is not None:
        return month_account_comparison
    funnel_leakage = _funnel_leakage_analysis(q, history)
    if funnel_leakage is not None:
        return funnel_leakage
    operational_growth_drivers = _operational_growth_driver_analysis(q, history)
    if operational_growth_drivers is not None:
        return operational_growth_drivers
    contribution_drivers = _contribution_driver_analysis(q, history)
    if contribution_drivers is not None:
        return contribution_drivers
    scoped = _product_filter(q, history)
    scoped, geography_scope = _geography_filter(q, scoped)
    if scoped.empty:
        return "No authorized historical records matched the requested geography.", pd.DataFrame(), "historical:empty_scope"
    latest = scoped.month.max()
    if "branch" in q:
        dimension = "branch_code"
    elif "region" in q and geography_scope is None:
        dimension = "region"
    elif "segment" in q:
        dimension = "customer_segment"
    elif "product" in q or ("portfolio" in q and any(x in q for x in ("which", "fastest", "lowest", "highest"))):
        dimension = "product"
    else:
        dimension = None
    metric = "portfolio_amount"
    if "new account" in q: metric = "new_accounts"
    elif "default" in q or "delinquen" in q: metric = "default_rate_pct"
    elif "churn" in q or "attrition" in q: metric = "churn_rate_pct"
    elif "lead" in q and "conversion" in q: metric = "lead_conversion_pct"
    elif "cross" in q: metric = "cross_sell_conversion_pct"
    elif "collection" in q: metric = "collection_efficiency_pct"

    asks_portfolio_drilldown = any(term in q for term in ("driver", "drill down", "drilldown")) or (
        "contribut" in q and any(term in q for term in ("growth", "change", "declin", "increase", "decrease"))
    ) or (
        "why" in q and "portfolio" in q
    )
    if asks_portfolio_drilldown and metric == "portfolio_amount":
        previous_month = latest - pd.DateOffset(years=1)
        current_total = float(_aggregate(scoped, [latest]).iloc[0]["portfolio_amount"])
        previous_total = float(_aggregate(scoped, [previous_month]).iloc[0]["portfolio_amount"])
        net_change = current_total - previous_total
        growth = 100 * net_change / previous_total if previous_total else 0.0
        driver_frames = []
        for dimension_name, label in (("product", "Product"), ("branch_code", "Branch")):
            current_driver = _aggregate(scoped, [latest], dimension_name)
            previous_driver = _aggregate(scoped, [previous_month], dimension_name)
            driver = current_driver.merge(previous_driver, on=dimension_name, suffixes=("_current", "_previous"))
            driver["Change ₹ Cr"] = (
                driver["portfolio_amount_current"] - driver["portfolio_amount_previous"]
            ) / 10_000_000
            driver["Contribution %"] = (
                100 * (driver["portfolio_amount_current"] - driver["portfolio_amount_previous"]) / net_change
                if net_change
                else 0.0
            )
            movement_drivers = driver.nsmallest(5, "Change ₹ Cr") if net_change < 0 else driver.nlargest(5, "Change ₹ Cr")
            driver_frames.append(
                movement_drivers[
                    [dimension_name, "portfolio_amount_current", "portfolio_amount_previous", "Change ₹ Cr", "Contribution %"]
                ].rename(columns={
                    dimension_name: "Driver",
                    "portfolio_amount_current": "Current ₹",
                    "portfolio_amount_previous": "Previous ₹",
                }).assign(**{"Driver type": label})
            )
        result = pd.concat(driver_frames, ignore_index=True)
        result["Current ₹ Cr"] = result.pop("Current ₹") / 10_000_000
        result["Previous ₹ Cr"] = result.pop("Previous ₹") / 10_000_000
        result = result[["Driver type", "Driver", "Current ₹ Cr", "Previous ₹ Cr", "Change ₹ Cr", "Contribution %"]]
        top_product = result[result["Driver type"] == "Product"].iloc[0]
        top_branch = result[result["Driver type"] == "Branch"].iloc[0]
        movement = "grew" if growth >= 0 else "declined"
        scope_prefix = f"{geography_scope} portfolio" if geography_scope else "Portfolio"
        answer = (
            f"{scope_prefix} amount {movement} {abs(growth):.2f}% YoY "
            f"(₹{previous_total / 10_000_000:,.2f} Cr to ₹{current_total / 10_000_000:,.2f} Cr). "
            f"The leading product driver is {top_product['Driver']} "
            f"({top_product['Change ₹ Cr']:+,.2f} Cr), and the leading branch driver is {top_branch['Driver']} "
            f"({top_branch['Change ₹ Cr']:+,.2f} Cr)."
        )
        if growth >= 0:
            answer += "\n\n**AI insight:** Product and branch contributions explain where the measured growth came from; no intervention is recommended."
        else:
            answer = _insight_text(answer, "portfolio_amount", True, "The product and branch contribution analysis shows a portfolio contraction.")
        return answer, result.round(2), "historical:portfolio_amount:drivers"

    if "performing" in q and dimension is None:
        current = _aggregate(scoped, [latest]).iloc[0]
        previous = _aggregate(scoped, [latest - pd.DateOffset(years=1)]).iloc[0]
        portfolio_growth = 100 * (current.portfolio_amount - previous.portfolio_amount) / max(previous.portfolio_amount, 1)
        account_growth = 100 * (current.new_accounts - previous.new_accounts) / max(previous.new_accounts, 1)
        default_change = current.default_rate_pct - previous.default_rate_pct
        result = pd.DataFrame([
            {"Measure": "Portfolio", "Current": round(current.portfolio_amount / 10_000_000, 2), "YoY change": round(portfolio_growth, 2), "Unit": "₹ Cr / %"},
            {"Measure": "New accounts", "Current": int(current.new_accounts), "YoY change": round(account_growth, 2), "Unit": "accounts / %"},
            {"Measure": "Default rate", "Current": round(current.default_rate_pct, 2), "YoY change": round(default_change, 2), "Unit": "% / pp"},
        ])
        worsened = default_change > 0 or portfolio_growth < 0
        answer = (
            f"Portfolio is ₹{current.portfolio_amount / 10_000_000:,.2f} Cr ({portfolio_growth:+.2f}% YoY), "
            f"new accounts are {int(current.new_accounts):,} ({account_growth:+.2f}% YoY), and default rate is "
            f"{current.default_rate_pct:.2f}% ({default_change:+.2f} pp YoY)."
        )
        return (
            _insight_text(
                answer,
                "default_rate_pct",
                worsened,
                "Default-rate deterioration or weakening growth requires investigation.",
            ),
            result,
            "historical:portfolio_health:yoy",
        )
    if "quarter" in q or "qoq" in q or re.search(r"\bq[1-4]\b", q):
        current_months = list(pd.date_range(end=latest, periods=3, freq="MS"))
        previous_months = [m - pd.DateOffset(months=3) for m in current_months]
        label = "QoQ"
    elif "month" in q or "mom" in q:
        current_months, previous_months, label = [latest], [latest - pd.DateOffset(months=1)], "MoM"
    else:
        current_months, previous_months, label = [latest], [latest - pd.DateOffset(years=1)], "YoY"
    current = _aggregate(scoped, current_months, dimension)
    previous = _aggregate(scoped, previous_months, dimension)
    if dimension:
        result = current.merge(previous, on=dimension, suffixes=("_current", "_previous"))
        result["absolute_change"] = result[f"{metric}_current"] - result[f"{metric}_previous"]
        result["growth_pct"] = 100 * (result[f"{metric}_current"] - result[f"{metric}_previous"]) / result[f"{metric}_previous"].replace(0, pd.NA)
        ascending = any(word in q for word in ("lowest", "declining", "least"))
        singular_ranking = any(word in q for word in ("highest", "lowest", "fastest", "most", "least", "best", "worst"))
        result_limit = 1 if singular_ranking else 10
        result = result.sort_values("growth_pct", ascending=ascending).head(result_limit)
        top = result.iloc[0]
        answer = f"{top[dimension]} has the {'lowest' if ascending else 'highest'} {label} change in {metric.replace('_', ' ')}: {top.growth_pct:+.2f}%."
        worsened = (metric in {"default_rate_pct", "churn_rate_pct"} and top.growth_pct > 0) or (metric not in {"default_rate_pct", "churn_rate_pct"} and top.growth_pct < 0)
        dimension_label = {
            "branch_code": "Branch",
            "region": "Region",
            "customer_segment": "Customer segment",
            "product": "Product",
        }.get(dimension, dimension.replace("_", " ").title())
        if metric.endswith("_pct"):
            display_result = result[[dimension, f"{metric}_current", f"{metric}_previous", "absolute_change", "growth_pct"]].rename(columns={
                dimension: dimension_label,
                f"{metric}_current": "Current %",
                f"{metric}_previous": "Previous %",
                "absolute_change": "Change pp",
                "growth_pct": "Change %",
            })
        elif metric == "portfolio_amount":
            display_result = result[[dimension, f"{metric}_current", f"{metric}_previous", "absolute_change", "growth_pct"]].rename(columns={
                dimension: dimension_label,
                f"{metric}_current": "Current ₹ Cr",
                f"{metric}_previous": "Previous ₹ Cr",
                "absolute_change": "Change ₹ Cr",
                "growth_pct": "Change %",
            })
            for column in ("Current ₹ Cr", "Previous ₹ Cr", "Change ₹ Cr"):
                display_result[column] = display_result[column] / 10_000_000
        else:
            display_result = result[[dimension, f"{metric}_current", f"{metric}_previous", "absolute_change", "growth_pct"]].rename(columns={
                dimension: dimension_label,
                f"{metric}_current": "Current accounts",
                f"{metric}_previous": "Previous accounts",
                "absolute_change": "Change accounts",
                "growth_pct": "Change %",
            })
        result = display_result
    else:
        c, p = current.iloc[0], previous.iloc[0]
        current_value, previous_value = float(c[metric]), float(p[metric])
        growth = ((current_value - previous_value) / previous_value * 100) if previous_value else 0.0
        result = pd.DataFrame([{"Metric": metric.replace("_", " ").title(), "Period": label, "Current": round(current_value, 2), "Previous": round(previous_value, 2), "Growth %": round(growth, 2)}])
        unit = "₹ Cr" if metric == "portfolio_amount" else "%" if metric.endswith("_pct") else "accounts"
        if metric == "portfolio_amount": current_value, previous_value = current_value / 10_000_000, previous_value / 10_000_000
        answer = f"{metric.replace('_', ' ').title()} is {current_value:,.2f} {unit} versus {previous_value:,.2f} {unit}; {label} change is {growth:+.2f}%."
        worsened = (metric in {"default_rate_pct", "churn_rate_pct"} and growth > 0) or (metric not in {"default_rate_pct", "churn_rate_pct"} and growth < 0)
    return (
        _insight_text(answer, metric, worsened),
        result.round(2),
        f"historical:{metric}:{label.lower()}",
    )

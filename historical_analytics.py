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
    aliases = {"personal loan": "Personal Loan", "home loan": "Home Loan", "vehicle loan": "Vehicle Loan", "education loan": "Education Loan", "msme": "MSME Loan", "agri": "Agri Loan"}
    for phrase, product in aliases.items():
        if phrase in q:
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


def answer_historical_bi(question: str, history: pd.DataFrame):
    q = question.lower().strip()
    markers = ("growth", "driver", "contribut", "portfolio", "month", "quarter", "yoy", "mom", "qoq", "last year", "last quarter", "trend", "new account", "performing", "conversion", "default rate", "churn", "collection efficiency")
    if history.empty or not any(marker in q for marker in markers):
        return None
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

    asks_portfolio_drilldown = any(term in q for term in ("driver", "contribut", "drill down", "drilldown")) or (
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

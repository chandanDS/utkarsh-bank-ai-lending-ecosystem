import os
import re
import streamlit as st
import pandas as pd
import altair as alt
from pathlib import Path
from dotenv import load_dotenv
from urllib.error import HTTPError, URLError
from engine import authenticate_user, init_db, rows, scored_customers, score, event, upsert_application, retrieve_policy, POLICY_VERSION
from airtel_voice import TEST_NUMBER, call_window_open, configuration_status, place_test_call
from campaign_links import TEST_MESSAGE, VOICE_SCRIPT, launch_links
from lending_analytics import AI_COLUMNS, CAMPAIGN_AUDIENCES, campaign_audience, classify_next_best_action, enrich_portfolio, executive_impact_report, lead_conversion_analytics
from hybrid_bi import DEFAULT_INSIGHT_MODEL, DEFAULT_MODEL, answer_hybrid_bi, zero_usage
from historical_analytics import build_historical_portfolio, observe_metrics
from twilio_voice import authorized_test_numbers, configuration_status as twilio_configuration_status, place_test_call as place_twilio_test_call, trial_mode as twilio_trial_mode
from twilio_messaging import configuration_status as twilio_message_configuration_status, send_test_message

PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / '.env')
load_dotenv(PROJECT_ROOT / '.venv' / '.env', override=False)

st.set_page_config(
    page_title="Utkarsh Bank AI-Powered Lending Ecosystem",
    page_icon=":material/account_balance:",
    layout="wide",
)
init_db()
OFFICIAL_LOGO_URL = "https://www.utkarsh.bank.in/images/logo.svg"
OFFICIAL_BACKGROUND_URL = "https://www.utkarsh.bank.in/images/Loans.jpg"
st.html(f"""
<style>
.stApp {{
    background-image:
        linear-gradient(rgba(247, 249, 252, 0.89), rgba(247, 249, 252, 0.93)),
        url('{OFFICIAL_BACKGROUND_URL}');
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}}
section[data-testid="stSidebar"] {{
    width: 22rem !important;
    min-width: 22rem !important;
}}
section[data-testid="stSidebar"] > div:first-child {{
    width: 22rem !important;
}}
[data-testid="stMainBlockContainer"] {{
    padding-top: 1.75rem;
}}
section[data-testid="stSidebar"] [data-testid="stSidebarContent"] {{
    overflow-y: auto;
    scrollbar-gutter: stable;
    padding-top: 0.35rem;
    padding-bottom: 0.30rem;
}}
section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {{
    gap: 0.22rem;
}}
section[data-testid="stSidebar"] div[role="radiogroup"] {{
    gap: 0.28rem;
}}
section[data-testid="stSidebar"] div[role="radiogroup"] label {{
    background: rgba(255, 255, 255, 0.09);
    border: 1px solid rgba(255, 255, 255, 0.20);
    border-radius: 9px;
    padding: 0.40rem 0.52rem;
    transition: background 120ms ease, border-color 120ms ease;
}}
section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {{
    background: rgba(255, 255, 255, 0.17);
    border-color: rgba(255, 255, 255, 0.38);
}}
section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {{
    background: #E6B084;
    border-color: #EFC9AA;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.12);
}}
section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) p {{
    color: #2F2154 !important;
    font-weight: 750;
}}
section[data-testid="stSidebar"] [data-testid="stTable"] table {{
    font-size: 0.84rem;
}}
section[data-testid="stSidebar"] [data-testid="stTable"] th,
section[data-testid="stSidebar"] [data-testid="stTable"] td {{
    padding: 0.20rem 0.38rem;
    line-height: 1.14;
}}
.st-key-groq_settings_section {{
    margin-top: 0.35rem;
    margin-bottom: 0.45rem;
}}
.st-key-token_usage_panel {{
    margin-top: 0.35rem;
    padding-top: 0.40rem;
    border-top: 1px solid rgba(255, 255, 255, 0.22);
}}
[data-testid="stDataFrame"] [role="columnheader"] {{
    justify-content: center;
    text-align: center;
}}
[data-testid="stTable"] thead th {{
    text-align: center !important;
}}
.st-key-brand_header {{
    background: rgba(255, 255, 255, 0.94);
    border: 1px solid rgba(0, 91, 150, 0.22);
    border-radius: 10px;
    box-shadow: 0 4px 16px rgba(0, 91, 150, 0.09);
    padding: 0.70rem 1.15rem 0.80rem;
    margin-bottom: 0.85rem;
}}
.st-key-brand_header h1 {{
    color: #005B96 !important;
    letter-spacing: -0.015em;
    text-align: center;
    margin-top: 0.10rem;
    margin-bottom: 0.15rem;
}}
.st-key-brand_header [data-testid="stCaptionContainer"] {{
    color: #356B8F;
    text-align: center;
}}
.journey-grid {{
    display: grid;
    grid-template-columns: repeat(5, minmax(0, 1fr));
    gap: 0.75rem;
    margin: 0.35rem 0 1rem 0;
}}
.journey-card {{
    min-height: 132px;
    background: rgba(255, 255, 255, 0.91);
    border: 1px solid rgba(75, 46, 131, 0.14);
    border-top: 3px solid rgba(75, 46, 131, 0.42);
    border-radius: 9px;
    box-shadow: 0 2px 8px rgba(47, 33, 84, 0.06);
    padding: 0.78rem 0.88rem;
}}
.journey-number {{
    display: inline-block;
    color: #4B2E83;
    background: #EEEAF4;
    border: 1px solid #D9D0E6;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 650;
    letter-spacing: 0.04em;
    padding: 0.18rem 0.55rem;
    margin-bottom: 0.45rem;
}}
.journey-card h3 {{
    color: #2F2154;
    font-size: 1.04rem;
    margin: 0 0 0.35rem 0;
}}
.journey-card p {{
    color: #536173;
    font-size: 0.86rem;
    line-height: 1.38;
    margin: 0;
}}
@media (max-width: 900px) {{
    .journey-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
}}
@media (max-width: 560px) {{
    .journey-grid {{ grid-template-columns: 1fr; }}
}}
</style>
""")
st.logo(OFFICIAL_LOGO_URL, size="large")
st.session_state.setdefault("auth_user", None)

MONEY_MARKERS = ("amount", "outstanding", "income", "emi", "due", "received", "exposure", "disbursal", "impact")


def should_display_result_table(question, data):
    """Keep executive answers concise unless supporting detail was requested."""
    if data.empty:
        return False
    normalized = question.lower().strip()
    explicit_detail = any(
        marker in normalized
        for marker in (
            "table", "list", "details", "detail", "breakdown", "all branches",
            "all products", "all regions", "all segments",
        )
    ) or bool(re.search(r"\b(top|bottom)\s+\d+\b", normalized))
    if explicit_detail:
        return True
    concise_answer = any(
        marker in normalized
        for marker in (
            "highest", "lowest", "fastest", "most", "least", "best", "worst",
            "key driver", "main driver", "growth driver", "total", "how much", "what is the",
        )
    )
    return not concise_answer and len(data) > 1

def crore_display(frame):
    """Convert displayed rupee-valued numeric columns to ₹ crore with two decimals."""
    shown = frame.copy()
    rename = {}
    for column in shown.columns:
        name = str(column)
        normalized = name.lower()
        is_money = any(marker in normalized for marker in MONEY_MARKERS)
        excluded = any(marker in normalized for marker in ("rate", "count", "accounts", "%", "score"))
        already_crore = "cr" in normalized or "crore" in normalized
        if is_money and not excluded and not already_crore and pd.api.types.is_numeric_dtype(shown[column]):
            shown[column] = (pd.to_numeric(shown[column], errors="coerce") / 10_000_000).round(2)
            clean = name.replace(" ₹", "").replace("(₹)", "").strip()
            rename[column] = f"{clean} ₹ Cr"
        elif pd.api.types.is_float_dtype(shown[column]):
            shown[column] = pd.to_numeric(shown[column], errors="coerce").round(2)
    return shown.rename(columns=rename)

if st.session_state.auth_user is None:
    login_left, login_center, login_right = st.columns([1, 1.25, 1])
    with login_center:
        with st.container(border=True):
            st.image(OFFICIAL_LOGO_URL, width=220)
            st.subheader("AI-Powered Lending Ecosystem")
            st.caption("Secure demo access for Head Office and branch users")
            with st.form("login_form"):
                username = st.text_input("Username", placeholder="Enter HO or branch username")
                password = st.text_input("Password", type="password", placeholder="Enter password")
                submitted = st.form_submit_button("Sign in", type="primary", icon=":material/login:", width="stretch")
            if submitted:
                user = authenticate_user(username, password)
                if user:
                    st.session_state.auth_user = user
                    st.rerun()
                else:
                    st.error("Invalid username or password.", icon=":material/error:")
            st.caption("Demo: bankho or bankuser001–bankuser012. Password is the same as the username.")
    st.stop()

current_user = st.session_state.auth_user
if current_user.get("role") not in {"HO", "BRANCH"}:
    st.error("Your account has no authorized lending role.")
    st.stop()
if current_user["role"] == "BRANCH" and not current_user.get("branch_code"):
    st.error("Your branch account has no branch scope.")
    st.stop()

with st.container(key="brand_header", horizontal_alignment="center"):
    st.title("AI-Powered Lending Ecosystem")
    st.caption(
        "Observe business pain points · Analyze drivers · Predict outcomes · Act through campaigns · Monitor impact"
    )

st.html("""
<div class="journey-grid" aria-label="AI-powered lending journey">
  <section class="journey-card" style="--journey-accent:#4B2E83">
    <span class="journey-number">STAGE 01</span><h3>1. Observe</h3>
    <p>Surface attrition, defaults, funnel leakage and missed opportunity.</p>
  </section>
  <section class="journey-card" style="--journey-accent:#E97924">
    <span class="journey-number">STAGE 02</span><h3>2. Analyze</h3>
    <p>Conversational BI explains trends, comparisons and drivers.</p>
  </section>
  <section class="journey-card" style="--journey-accent:#315CA8">
    <span class="journey-number">STAGE 03</span><h3>3. Predict</h3>
    <p>AI/ML identifies opportunity, risk and next-best offer or action.</p>
  </section>
  <section class="journey-card" style="--journey-accent:#15803D">
    <span class="journey-number">STAGE 04</span><h3>4. Act</h3>
    <p>Campaign Agent converts predictions into governed outreach.</p>
  </section>
  <section class="journey-card" style="--journey-accent:#607D3B">
    <span class="journey-number">STAGE 05</span><h3>5. Monitor</h3>
    <p>Executive reporting tracks conversion and business impact.</p>
  </section>
</div>
""")

pages = ["Observe", "Analyze", "AI/ML intelligence", "Campaign agent", "Executive Business Impact report"]
st.sidebar.markdown("### Utkarsh Bank")
active_scope = "All branches" if current_user["role"] == "HO" else current_user["branch_code"]
with st.sidebar.container(border=True):
    role_label = "Head Office" if current_user["role"] == "HO" else "Branch user"
    st.markdown(f"**{current_user['display_name']}** · {role_label}")
    st.caption(f"{current_user['username']} · Scope: {active_scope}")
    if st.button("Sign out", icon=":material/logout:", width="stretch"):
        st.session_state.clear()
        st.rerun()
st.sidebar.subheader("Demo modules", icon=":material/apps:")
if st.session_state.get("module_page") == "Conversational BI":
    st.session_state.module_page = "Analyze"
page = st.sidebar.radio("Demo modules", pages, label_visibility="collapsed", key="module_page")

def configured_secret(name, default=""):
    value = os.getenv(name, "")
    if value:
        return value
    try:
        return str(st.secrets.get(name, default))
    except Exception:
        return default

groq_api_key = configured_secret("GROQ_API_KEY")
groq_model = configured_secret("GROQ_MODEL", DEFAULT_MODEL)
groq_insight_model = configured_secret("GROQ_INSIGHT_MODEL", DEFAULT_INSIGHT_MODEL)
with st.sidebar.container(key="groq_settings_section"):
    with st.expander(
        "Groq AI settings",
        icon=":material/tune:",
        expanded=False,
    ):
        st.caption(f"SQL: {groq_model} · Insights: {groq_insight_model}")
        st.caption("Simple queries run locally with zero LLM tokens; complex queries use Groq.")
        insight_temperature = st.slider(
            "Insight temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.1,
            step=0.1,
            disabled=not bool(groq_api_key),
            help="SQL planning is fixed near zero for accuracy. This controls only grounded insight wording.",
        )
        st.badge("Groq ready" if groq_api_key else "GROQ_API_KEY required", color="green" if groq_api_key else "orange")
token_panel = st.sidebar.container(key="token_usage_panel")
if page not in pages:
    st.error("Unauthorized module selection.")
    st.stop()

if "token_usage" not in st.session_state:
    st.session_state.token_usage = zero_usage()
if "bi_messages" not in st.session_state:
    st.session_state.bi_messages = []
@st.cache_data(ttl=60, show_spinner=False)
def load_portfolio(branch_code):
    return enrich_portfolio(scored_customers(branch_code))

@st.cache_data(ttl=60, show_spinner=False)
def load_historical_portfolio(branch_code):
    return build_historical_portfolio(load_portfolio(branch_code), months=36)

branch_scope = current_user["branch_code"] if current_user["role"] == "BRANCH" else None
df = load_portfolio(branch_scope)
history_df = load_historical_portfolio(branch_scope)
pilot = df[df.cohort == "pilot"].copy()
control = df[df.cohort == "control"].copy()

if page == "Observe":
    st.subheader("Observe business performance and pain areas", icon=":material/visibility:")
    st.caption(
        f"36-month governed analytical view · scope: {active_scope} · synthetic stressed-pilot outcomes calibrated to expose improvement opportunities."
    )
    observed = observe_metrics(history_df)
    current = observed["current"]
    prior = observed["previous_year"]

    pain_cards = [
        ("Customer attrition", "churn_rate_pct", True, "Retention leakage", "Churn prediction"),
        ("Default rate", "default_rate_pct", True, "Credit quality pressure", "Default & early warning"),
        ("Lead conversion", "lead_conversion_pct", False, "Funnel leakage", "Propensity / Credit decisioning"),
        ("Cross-sell conversion", "cross_sell_conversion_pct", False, "Missed wallet share", "Cross-sell / Next Best Offer"),
        ("Collection efficiency", "collection_efficiency_pct", False, "Recovery inefficiency", "Collections intelligence"),
        ("Manual-review rate", "manual_reviews", True, "Operational effort and slower turnaround", "Credit decisioning"),
    ]
    metric_columns = st.columns(3) + st.columns(3)
    for column, (title, metric, inverse, pain, target) in zip(metric_columns, pain_cards):
        with column:
            with st.container(border=True):
                if metric == "manual_reviews":
                    value = 100 * float(current[metric]) / max(float(current["account_count"]), 1)
                    previous = 100 * float(prior[metric]) / max(float(prior["account_count"]), 1)
                else:
                    value, previous = float(current[metric]), float(prior[metric])
                st.metric(
                    title,
                    f"{value:.2f}%",
                    f"{value-previous:+.2f} pp YoY",
                    delta_color="inverse" if inverse else "normal",
                    border=False,
                )
                st.badge("Attention required", color="red")
                st.caption(f"Pain area: {pain} · Suggested module: {target}")

    latest_month = history_df["month"].max()
    latest = history_df[history_df["month"] == latest_month]
    funnel_values = {
        "Leads": int(latest["leads"].sum()),
        "Applications": int(latest["applications"].sum()),
        "Credit evaluation": int(latest["credit_evaluated"].sum()),
        "Approved": int(latest["approved"].sum()),
        "Disbursal": int(latest["disbursed"].sum()),
    }
    funnel = pd.DataFrame({"Stage": list(funnel_values), "Customers": list(funnel_values.values())})
    funnel["Conversion from leads %"] = (100 * funnel["Customers"] / max(funnel.iloc[0]["Customers"], 1)).round(2)
    funnel["Drop-off from prior stage"] = funnel["Customers"].shift(1).sub(funnel["Customers"]).fillna(0).astype(int)
    trend = history_df.groupby("month", as_index=False).agg(
        portfolio_amount=("portfolio_amount", "sum"),
        account_count=("account_count", "sum"),
        new_accounts=("new_accounts", "sum"),
        default_accounts=("default_accounts", "sum"),
        delinquent_accounts=("delinquent_accounts", "sum"),
        churned_customers=("churned_customers", "sum"),
        leads=("leads", "sum"),
        applications=("applications", "sum"),
        credit_evaluated=("credit_evaluated", "sum"),
        approved=("approved", "sum"),
        disbursed=("disbursed", "sum"),
        cross_sell_contacted=("cross_sell_contacted", "sum"),
        cross_sell_accepted=("cross_sell_accepted", "sum"),
        collection_due=("collection_due", "sum"),
        collection_received=("collection_received", "sum"),
        manual_reviews=("manual_reviews", "sum"),
    )
    trend["Portfolio amount ₹ Cr"] = trend.portfolio_amount / 10_000_000
    trend["Default rate %"] = 100 * trend.default_accounts / trend.account_count.clip(lower=1)
    trend["Delinquency rate %"] = 100 * trend.delinquent_accounts / trend.account_count.clip(lower=1)
    trend["Attrition rate %"] = 100 * trend.churned_customers / trend.account_count.clip(lower=1)
    trend["Application rate %"] = 100 * trend.applications / trend.leads.clip(lower=1)
    trend["Approval rate %"] = 100 * trend.approved / trend.credit_evaluated.clip(lower=1)
    trend["Lead conversion %"] = 100 * trend.disbursed / trend.leads.clip(lower=1)
    trend["Manual underwriting %"] = 100 * trend.manual_reviews / trend.account_count.clip(lower=1)
    trend["Cross-sell conversion %"] = 100 * trend.cross_sell_accepted / trend.cross_sell_contacted.clip(lower=1)
    trend["Collection efficiency %"] = 100 * trend.collection_received / trend.collection_due.clip(lower=1)

    funnel_tab, trend_tab, underwriting_tab = st.tabs(
        ["Lead conversion funnel", "Trend By Metric", "Underwriting quality"]
    )
    with funnel_tab:
        chart, table = st.columns([3, 2])
        with chart:
            funnel_order = ["Leads", "Applications", "Credit evaluation", "Approved", "Disbursal"]
            funnel_chart = (
                alt.Chart(funnel)
                .mark_bar(color="#4B2E83", cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
                .encode(
                    x=alt.X("Stage:N", sort=funnel_order, title=None, axis=alt.Axis(labelAngle=0)),
                    y=alt.Y("Customers:Q", title="Customers"),
                    tooltip=["Stage:N", alt.Tooltip("Customers:Q", format=",")],
                )
                .properties(height=360)
            )
            st.altair_chart(funnel_chart)
        with table:
            st.dataframe(funnel, hide_index=True, width="stretch", height=360)
        st.info(
            "Recommended action: use Propensity and Credit decisioning outputs to investigate the largest stage drop-off, then activate an approved campaign.",
            icon=":material/lightbulb:",
        )
    with trend_tab:
        control_a, control_b = st.columns(2)
        with control_a:
            trend_period = st.segmented_control(
                "Trend period",
                ["1 year", "2 years", "3 years"],
                default="3 years",
                key="observe_trend_period",
            )
        with control_b:
            trend_measure_label = st.selectbox(
                "Metric",
                [
                    "Portfolio amount",
                    "Total accounts",
                    "New accounts",
                    "Application rate",
                    "Approval rate",
                    "Lead conversion rate",
                    "Default rate",
                    "Delinquency rate",
                    "Customer attrition rate",
                    "Manual underwriting rate",
                    "Cross-sell conversion rate",
                    "Collection efficiency",
                ],
                key="observe_trend",
            )
        metric_column = {
            "Portfolio amount": "Portfolio amount ₹ Cr",
            "Total accounts": "account_count",
            "New accounts": "new_accounts",
            "Application rate": "Application rate %",
            "Approval rate": "Approval rate %",
            "Lead conversion rate": "Lead conversion %",
            "Default rate": "Default rate %",
            "Delinquency rate": "Delinquency rate %",
            "Customer attrition rate": "Attrition rate %",
            "Manual underwriting rate": "Manual underwriting %",
            "Cross-sell conversion rate": "Cross-sell conversion %",
            "Collection efficiency": "Collection efficiency %",
        }[trend_measure_label]
        months_to_show = {"1 year": 12, "2 years": 24, "3 years": 36}[trend_period]
        visible_trend = trend.tail(months_to_show)
        st.line_chart(
            visible_trend,
            x="month",
            y=metric_column,
            x_label="Month",
            y_label=trend_measure_label,
            height=360,
        )
        st.caption(f"Showing the latest {trend_period} for {trend_measure_label.lower()}. Use Analyze for detailed MoM, QoQ and YoY comparisons.")
    with underwriting_tab:
        under_cols = st.columns(3)
        under_cols[0].metric("Manual reviews", f"{int(current['manual_reviews']):,}", border=True)
        under_cols[1].metric("Good customers rejected", f"{int(current['good_customers_rejected']):,}", border=True)
        under_cols[2].metric("Risky customers approved", f"{int(current['risky_customers_approved']):,}", border=True)
        st.warning(
            "Review Credit decisioning and Default & early-warning outputs. Policy changes require validation and Business Head / Risk approval."
        )

elif page == "LMS overview":
    st.subheader("Executive lending overview")
    st.caption(f"Production-shaped synthetic portfolio · {len(df):,} customer records · shared across every module")
    st.markdown("**Offer base → pilot assignment → eligibility → propensity and risk → offer → campaign → response or organic lead → onboarding → underwriting → agreement → disbursal → monitoring → collections**")
    a,b,c,d = st.columns(4)
    a.metric("Offer base", len(df)); b.metric("Pilot", len(pilot)); c.metric("Control", len(control))
    d.metric("Pilot eligible", int(pilot.segment.isin(["pre-approved","pre-qualified","underwriting"]).sum()))
    st.markdown("### Operating model")
    st.write("Decision workflows handle routine cases. Policy gates, reason codes, and human exception review protect credit quality. Campaign and collections agents prepare actions, while consent and timing checks control execution. Every stage creates a journey event.")
    st.markdown("### Pilot success measures")
    st.dataframe(pd.DataFrame({"Measure":["Incremental disbursal and contribution margin", "Approval and take-up", "30/60/90 DPD and expected loss", "Time to decision and human touches", "Collection cure rate", "Complaints, fairness and overrides"], "Comparison":["Pilot vs randomized control"]*6}), hide_index=True, width="stretch")
    st.warning("This demonstration is not authorized for live credit decisions. Replace the placeholder policy, validate models, and connect governed production systems before launch.")

elif page == "Executive Business Impact report":
    st.subheader("Executive Business Impact report", icon=":material/monitoring:")
    st.caption(f"Business impact delivered by each AI/ML use case within the authorized scope: {active_scope} · synthetic demo estimates.")
    impact_tab = st.tabs(["AI/ML business impact"])[0]
    with impact_tab:
        impact = executive_impact_report(df)
        total_opportunities = int(impact["Qualified audience"].sum())
        total_contacted = int(impact["Contacted"].sum())
        total_outcomes = int(impact["Positive outcomes"].sum())
        weighted_efficiency = total_outcomes / total_contacted * 100 if total_contacted else 0.0
        with st.container(horizontal=True):
            st.metric("Use-case opportunities", f"{total_opportunities:,}", border=True)
            st.metric("Contacted", f"{total_contacted:,}", border=True)
            st.metric("Positive outcomes", f"{total_outcomes:,}", border=True)
            st.metric("Weighted efficiency", f"{weighted_efficiency:.2f}%", border=True)

        propensity_impact = impact.loc[impact["Use case"] == "Propensity", "Estimated impact ₹"].sum()
        cross_sell_impact = impact.loc[impact["Use case"] == "Cross-sell / up-sell", "Estimated impact ₹"].sum()
        loss_avoided = impact.loc[impact["Use case"] == "Default & early warning", "Estimated impact ₹"].sum()
        portfolio_protected = impact.loc[impact["Use case"] == "Churn", "Estimated impact ₹"].sum()
        with st.container(horizontal=True):
            st.metric("Potential acquisition disbursal", f"₹{(propensity_impact + cross_sell_impact) / 10_000_000:,.2f} Cr", border=True)
            st.metric("Estimated loss avoided", f"₹{loss_avoided / 10_000_000:,.2f} Cr", border=True)
            st.metric("Portfolio value protected", f"₹{portfolio_protected / 10_000_000:,.2f} Cr", border=True)

        with st.container(border=True):
            st.markdown("**Conversion efficiency by use case**")
            st.bar_chart(impact, x="Use case", y="Conversion efficiency %", horizontal=True)

        with st.container(border=True):
            st.markdown("### Use-case conversion and business impact table")
            st.caption("Financial impact is displayed in ₹ crore and highlighted for executive review.")
            display_impact = crore_display(impact)
            impact_column = "Estimated impact ₹ Cr"
            styled_impact = display_impact.style.map(
                lambda value: "background-color: #FFF2CC; color: #2F2154; font-weight: 700;",
                subset=[impact_column],
            )
            report_height = 42 + (len(display_impact) * 35)
            st.dataframe(
                styled_impact,
                hide_index=True,
                width="stretch",
                height=report_height,
                column_config={
                    "Use case": st.column_config.TextColumn("AI/ML use case", pinned=True),
                    "Conversion efficiency %": st.column_config.ProgressColumn(
                        "Conversion efficiency", min_value=0, max_value=100, format="%.2f%%"
                    ),
                    "Contact rate %": st.column_config.NumberColumn("Contact rate", format="%.2f%%"),
                    impact_column: st.column_config.NumberColumn(
                        "Business impact (₹ Cr)", format="₹ %.2f", pinned=True
                    ),
                },
            )
        with st.expander("Impact methodology and assumptions", icon=":material/info:"):
            st.markdown("""
        - All records, campaign assignments and outcomes are synthetic and illustrative.
        - Contact assignment is a reproducible 80% simulation over each qualified audience.
        - Propensity, cross-sell and churn outcomes use the synthetic response indicator.
        - Early-warning success means at least 70% of due amount recovered, or an assisted non-DPD response.
        - Avoided loss uses a 45% loss-given-default assumption.
        - Financial impact types are not added together because disbursal, avoided loss and protected portfolio represent different measures.
        """)
        st.download_button(
            "Download executive AI/ML impact report",
            impact.to_csv(index=False),
            "utkarsh_ai_ml_executive_impact_report.csv",
            "text/csv",
            icon=":material/download:",
        )

    if False:  # Funnel is intentionally presented in Observe, before analysis and prediction.
        stage_report, waterfall, product_funnel, potential_disbursal = lead_conversion_analytics(df)
        lead_amount, login_amount, quote_amount, disbursal_amount = stage_report["Amount ₹ Cr"].tolist()
        disbursal_accounts = int(stage_report.iloc[3]["Accounts"])
        end_to_end = disbursal_amount / lead_amount * 100 if lead_amount else 0.0

        st.caption(f"Lead conversion journey within the authorized scope: {active_scope} · all monetary values in ₹ crore.")
        with st.container(horizontal=True):
            st.metric("AI-qualified lead value", f"₹{lead_amount:,.2f} Cr", border=True)
            st.metric("Login pipeline value", f"₹{login_amount:,.2f} Cr", border=True)
            st.metric("Quote pipeline value", f"₹{quote_amount:,.2f} Cr", border=True)
            st.metric("Disbursed value", f"₹{disbursal_amount:,.2f} Cr", border=True)
            st.metric("Value conversion", f"{end_to_end:.2f}%", border=True)
            st.metric("Disbursed accounts", f"{disbursal_accounts:,}", border=True)

        with st.container(border=True):
            st.markdown("**Lead conversion waterfall**")
            step_order = waterfall.sort_values("Order")["Step"].tolist()
            base = alt.Chart(waterfall).encode(
                x=alt.X("Step:N", sort=step_order, title=None, axis=alt.Axis(labelAngle=-20)),
                tooltip=[
                    alt.Tooltip("Step:N", title="Journey step"),
                    alt.Tooltip("Value:Q", title="Amount ₹ Cr", format=",.2f"),
                    alt.Tooltip("Type:N", title="Movement"),
                ],
            )
            bars = base.mark_bar(size=58, cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
                y=alt.Y("Start:Q", title="Amount (₹ Cr)", axis=alt.Axis(format=",.2f")),
                y2="End:Q",
                color=alt.Color(
                    "Type:N",
                    scale=alt.Scale(domain=["Stage total", "Drop-off"], range=["#006D77", "#D97745"]),
                    legend=alt.Legend(orient="top"),
                ),
            )
            labels = base.mark_text(dy=-9, fontWeight="bold", color="#17324D").encode(
                y=alt.Y("End:Q"),
                text="Label:N",
            )
            st.altair_chart((bars + labels).properties(height=380))

        stage_col, product_col = st.columns([2, 3])
        with stage_col:
            with st.container(border=True):
                st.markdown("**Stage conversion analytics**")
                st.dataframe(
                    stage_report,
                    hide_index=True,
                    width="stretch",
                    column_config={
                        "Amount ₹ Cr": st.column_config.NumberColumn("Amount ₹ Cr", format="₹ %.2f"),
                        "Drop-off ₹ Cr": st.column_config.NumberColumn("Drop-off ₹ Cr", format="₹ %.2f"),
                        "Stage value conversion %": st.column_config.ProgressColumn("Stage value conversion", min_value=0, max_value=100, format="%.2f%%"),
                        "Lead value conversion %": st.column_config.NumberColumn("Lead value conversion", format="%.2f%%"),
                    },
                )
        with product_col:
            with st.container(border=True):
                st.markdown("**Product-level lead funnel**")
                st.dataframe(
                    product_funnel,
                    hide_index=True,
                    width="stretch",
                    column_config={
                        "Lead ₹ Cr": st.column_config.NumberColumn(format="₹ %.2f"),
                        "Login ₹ Cr": st.column_config.NumberColumn(format="₹ %.2f"),
                        "Quote ₹ Cr": st.column_config.NumberColumn(format="₹ %.2f"),
                        "Disbursal ₹ Cr": st.column_config.NumberColumn(format="₹ %.2f"),
                        "Value conversion %": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.2f%%")
                    },
                )

        with st.expander("Funnel definitions and assumptions", icon=":material/info:"):
            st.markdown("""
            - **Lead:** consented customer with propensity ≥ 60% and predicted risk below 60%.
            - **Login:** reproducible 72% simulated digital/login progression from qualified leads.
            - **Quote:** logged-in customer with KYC complete and an eligible underwriting segment.
            - **Disbursal:** quoted customer with a positive synthetic conversion response.
            - Figures are synthetic, RBAC-scoped and intended for demonstration—not actual bank performance.
            """)
        st.download_button(
            "Download lead conversion analytics",
            product_funnel.to_csv(index=False),
            "utkarsh_lead_conversion_waterfall.csv",
            "text/csv",
            icon=":material/download:",
        )

elif page == "AI/ML intelligence":
    st.subheader("AI/ML intelligence", icon=":material/model_training:")
    st.caption(f"Identify customer opportunities, risks and next-best actions within the authorized scope: {active_scope} · {len(df):,} records.")
    tabs = st.tabs(["Propensity", "Default & early warning", "Churn", "Cross-sell", "Credit decisioning", "Personalization", "Collections", "Next Best Offer/Action"])
    with tabs[0]:
        view = df.sort_values("propensity", ascending=False)
        st.metric("High-propensity customers", f"{int((view.propensity >= .75).sum()):,}")
        propensity_columns = [
            "customer_id", "name", "branch_code", "city", "product", "customer_segment",
            "bureau", "risk", "propensity", "preferred_channel", "contact_hour",
        ]
        st.dataframe(view[propensity_columns].head(500), hide_index=True, width="stretch", column_config={
            "customer_segment": st.column_config.TextColumn("Customer profile segment"),
            "preferred_channel": st.column_config.TextColumn("Preferred channel", pinned=True),
            "contact_hour": st.column_config.NumberColumn("Preferred contact time", format="%d:00"),
        })
    with tabs[1]:
        st.caption(
            "Forward-looking 90-day risk output. It includes standard customers showing emerging risk as well as currently delinquent/defaulted accounts."
        )
        status_rank = {"Standard": 0, "Delinquent": 1, "Defaulted": 2}
        view = df[df["default_probability_90d"] >= .45].copy()
        view["_status_rank"] = view["current_account_status"].map(status_rank)
        view = view.sort_values(["_status_rank", "default_probability_90d"], ascending=[True, False])
        with st.container(horizontal=True):
            st.metric("High predicted risk", f"{int((view.default_probability_90d >= .60).sum()):,}", border=True)
            st.metric("Standard but at risk", f"{int((view.current_account_status == 'Standard').sum()):,}", border=True)
            st.metric("Already delinquent/defaulted", f"{int((view.current_account_status != 'Standard').sum()):,}", border=True)
        default_columns = [
            "customer_id", "name", "branch_code", "city", "product", "customer_segment",
            "current_account_status", "default_probability_90d", "bureau", "dpd",
            "delinquency_events_12m", "utilization", "aa_cashflow", "outstanding",
        ]
        st.dataframe(
            crore_display(view[default_columns].head(500)),
            hide_index=True,
            width="stretch",
            column_config={
                "customer_segment": st.column_config.TextColumn(
                    "Customer profile segment",
                    help="Business/customer profile such as Salaried, Self-employed or Micro enterprise—not a credit decision.",
                ),
                "current_account_status": st.column_config.TextColumn("Current account status", pinned=True),
                "default_probability_90d": st.column_config.NumberColumn(
                    "Predicted default probability (90 days)", format="percent"
                ),
                "dpd": st.column_config.NumberColumn(
                    "Current DPD",
                    help="Days past due today. Zero means the account is currently standard.",
                ),
                "delinquency_events_12m": st.column_config.NumberColumn(
                    "Delinquency events (last 12 months)",
                    help="Count of separate late-payment events in the previous 12 months; it is not a yes/no default flag.",
                ),
            },
        )
    with tabs[2]:
        view = df.sort_values("churn_probability", ascending=False)
        st.metric("High churn-risk customers", f"{int((view.churn_probability >= .65).sum()):,}")
        churn_columns = [
            "customer_id", "name", "branch_code", "city", "product", "customer_segment",
            "risk", "churn_probability", "past_complaint", "preferred_channel", "contact_hour",
        ]
        st.dataframe(view[churn_columns].head(500), hide_index=True, width="stretch", column_config={
            "customer_segment": st.column_config.TextColumn("Customer profile segment"),
            "churn_probability": st.column_config.NumberColumn("Churn probability", format="percent"),
            "contact_hour": st.column_config.NumberColumn("Preferred contact time", format="%d:00"),
        })
    with tabs[3]:
        view = df.sort_values("cross_sell_score", ascending=False)
        st.metric("Priority cross-sell customers", f"{int((view.cross_sell_score >= .70).sum()):,}")
        cross_sell_columns = [
            "customer_id", "name", "branch_code", "city", "product", "customer_segment",
            "recommended_product", "recommended_action", "cross_sell_score", "propensity",
            "risk", "preferred_channel", "contact_hour",
        ]
        st.dataframe(view[cross_sell_columns].head(500), hide_index=True, width="stretch", column_config={
            "customer_segment": st.column_config.TextColumn("Customer profile segment"),
            "recommended_product": st.column_config.TextColumn("Recommended product", pinned=True),
            "recommended_action": st.column_config.TextColumn("Recommended action"),
            "contact_hour": st.column_config.NumberColumn("Preferred contact time", format="%d:00"),
        })
    with tabs[4]:
        st.caption("Credit eligibility and policy outcome with the principal underwriting signals and reason codes.")
        decision_rank = {
            "pre-approved": 0,
            "pre-qualified": 1,
            "underwriting": 2,
            "manual-review": 3,
            "decline": 4,
        }
        view = df.assign(_decision_rank=df["segment"].map(decision_rank).fillna(5)).sort_values(
            ["_decision_rank", "risk", "bureau"], ascending=[True, True, False]
        )
        with st.container(horizontal=True):
            st.metric("Pre-approved / pre-qualified", f"{int(view.segment.isin(['pre-approved', 'pre-qualified']).sum()):,}", border=True)
            st.metric("Manual review / underwriting", f"{int(view.segment.isin(['manual-review', 'underwriting']).sum()):,}", border=True)
            st.metric("Declined", f"{int((view.segment == 'decline').sum()):,}", border=True)
        decision_columns = [
            "customer_id", "name", "branch_code", "city", "product", "bureau", "risk",
            "utilization", "delinquencies", "aa_cashflow", "kyc", "segment", "amount",
            "rate", "reasons",
        ]
        st.dataframe(
            crore_display(view[decision_columns].head(500)),
            hide_index=True,
            width="stretch",
            column_config={
                "segment": st.column_config.TextColumn(
                    "Credit decision outcome",
                    help="Policy result: pre-approved, pre-qualified, underwriting/manual review or decline.",
                ),
            },
        )
    with tabs[5]:
        st.caption("Customer persona, personalized product recommendation and preferred engagement channel and time.")
        view = df.sort_values(["propensity", "cross_sell_score", "risk"], ascending=[False, False, True])
        with st.container(horizontal=True):
            st.metric("High-personalization opportunities", f"{int((view.propensity >= .75).sum()):,}", border=True)
            st.metric("Recommended products", f"{view.recommended_product.nunique():,}", border=True)
            st.metric("Preferred channels", f"{view.preferred_channel.nunique():,}", border=True)
        personalization_columns = [
            "customer_id", "name", "branch_code", "city", "customer_persona", "customer_segment", "product",
            "recommended_product", "recommended_action", "past_complaint", "propensity",
            "cross_sell_score", "preferred_channel", "contact_hour", "amount", "rate",
        ]
        st.dataframe(crore_display(view[personalization_columns].head(500)), hide_index=True, width="stretch", column_config={
            "customer_persona": st.column_config.TextColumn("Customer persona", pinned=True),
            "customer_segment": st.column_config.TextColumn("Customer profile segment"),
            "recommended_product": st.column_config.TextColumn("Recommended product"),
            "recommended_action": st.column_config.TextColumn("Recommended action"),
            "preferred_channel": st.column_config.TextColumn("Preferred channel"),
            "contact_hour": st.column_config.NumberColumn("Preferred contact time", format="%d:00"),
        })
    with tabs[6]:
        st.caption("Collection prioritization output only. Product offers and recommended actions are intentionally excluded.")
        view = df[df.dpd > 0].sort_values("collection_priority", ascending=False)
        st.metric("Active collections worklist", f"{len(view):,}")
        collections_columns = [
            "customer_id", "name", "branch_code", "city", "product", "customer_segment",
            "risk", "dpd", "delinquencies", "outstanding", "collection_amount_due",
            "collection_amount_received", "collection_priority", "preferred_channel", "contact_hour",
        ]
        st.dataframe(
            crore_display(view[collections_columns].head(500)),
            hide_index=True,
            width="stretch",
            column_config={
                "customer_segment": st.column_config.TextColumn(
                    "Customer profile segment",
                    help="Business/customer profile used for analysis—not a credit decision outcome.",
                ),
                "risk": st.column_config.NumberColumn("Default-risk probability", format="percent"),
                "collection_priority": st.column_config.NumberColumn("Collection priority", format="percent"),
                "contact_hour": st.column_config.NumberColumn("Preferred contact time", format="%d:00"),
            },
        )
    with tabs[7]:
        st.caption("Recommended product offer and complaint-led service action derived from each customer's past complaint history.")
        action_rank = {"Attention": 0, "Decision": 1, "Offer": 2, "Information": 3}
        view = df.assign(
            next_best_offer=df["recommended_product"],
            next_best_action_type=df["recommended_action"].map(classify_next_best_action),
        )
        view = view.assign(_action_rank=view["next_best_action_type"].map(action_rank).fillna(4)).sort_values(
            ["_action_rank", "churn_probability", "propensity"],
            ascending=[True, False, False],
        )
        with st.container(horizontal=True):
            st.metric("Attention required", f"{int((view.next_best_action_type == 'Attention').sum()):,}", border=True)
            st.metric("Decisions required", f"{int((view.next_best_action_type == 'Decision').sum()):,}", border=True)
            st.metric("Offer opportunities", f"{int((view.next_best_action_type == 'Offer').sum()):,}", border=True)
            st.metric("Information / nurture", f"{int((view.next_best_action_type == 'Information').sum()):,}", border=True)
        action_columns = [
            "customer_id", "name", "branch_code", "city", "next_best_offer",
            "past_complaint", "complaint_reason", "next_best_action_type", "recommended_action",
            "amount", "rate", "risk", "propensity", "churn_probability", "preferred_channel", "contact_hour",
        ]
        st.dataframe(
            crore_display(view[action_columns].head(500)),
            hide_index=True,
            width="stretch",
            column_config={
                "next_best_offer": st.column_config.TextColumn("Next best offer", pinned=True),
                "past_complaint": st.column_config.TextColumn("Past complaint", pinned=True),
                "complaint_reason": st.column_config.TextColumn("Reason based on complaint history"),
                "next_best_action_type": st.column_config.TextColumn("Action type", pinned=True),
                "recommended_action": st.column_config.TextColumn("Recommended action"),
            },
        )
    st.download_button("Download complete prediction output", df[AI_COLUMNS].to_csv(index=False), "lending_prediction_output.csv", "text/csv", icon=":material/download:")

elif page == "Analyze":
    st.subheader("Analyze — Conversational BI", icon=":material/query_stats:")
    st.caption(f"Diagnose what changed, where it changed and whether action is required · authorized scope: {active_scope}.")
    with st.container(border=True):
        st.markdown("**What Analyze means**")
        explain_a, explain_b, explain_c = st.columns(3)
        explain_a.markdown("**Compare**  \nMoM, QoQ and YoY movement across portfolio and operational KPIs.")
        explain_b.markdown("**Drill down**  \nIdentify the affected product, branch, region or customer segment.")
        explain_c.markdown("**Connect to action**  \nRecommend a relevant AI/ML output only when the metric is deteriorating.")
    with st.expander("Analyze module test cases", icon=":material/fact_check:"):
        analyze_test_cases = pd.DataFrame([
            {"Test question": "Show portfolio growth compared with last year", "Expected behavior": "Defaults to portfolio amount and returns YoY growth.", "Expected path": "Historical analytics"},
            {"Test question": "What are the drivers for this portfolio growth?", "Expected behavior": "Returns leading product and branch contribution in ₹ Cr.", "Expected path": "Historical analytics"},
            {"Test question": "Show new accounts added MoM by product", "Expected behavior": "Compares monthly account additions across products.", "Expected path": "Historical analytics"},
            {"Test question": "Which branch has the highest increase in default rate YoY?", "Expected behavior": "Returns one branch and recommends Default & early warning/Credit decisioning.", "Expected path": "Historical analytics"},
            {"Test question": "Compare lead conversion QoQ", "Expected behavior": "Calculates quarterly conversion movement and links adverse movement to Propensity/Credit decisioning.", "Expected path": "Historical analytics"},
            {"Test question": "Show collection efficiency trend", "Expected behavior": "Calculates collection-efficiency movement from amount due and received.", "Expected path": "Historical analytics"},
            {"Test question": "Which product has the fastest portfolio growth?", "Expected behavior": "Ranks products by amount growth.", "Expected path": "Historical analytics"},
            {"Test question": "Which branch has the highest number of defaulters?", "Expected behavior": "Returns one RBAC-scoped branch, not an unrestricted customer list.", "Expected path": "Deterministic fast path"},
            {"Test question": "Compare average ticket size and risk by product", "Expected behavior": "Builds and executes governed read-only SQL.", "Expected path": "Groq dynamic SQL"},
            {"Test question": "Compare performance", "Expected behavior": "Asks for the missing metric or comparison scope.", "Expected path": "Clarification"},
        ])
        st.dataframe(analyze_test_cases, hide_index=True, width="stretch", height=388)
    with st.container(border=True):
        st.markdown("**Try:** Personal Loan growth versus last year · new accounts MoM by product · default-rate YoY by branch · lead conversion QoQ · region with declining portfolio · collection efficiency trend")
    visible_messages = st.session_state.bi_messages[-6:]
    latest_user_question = ""
    for message in visible_messages:
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                latest_user_question = message["content"]
            st.markdown(message["content"])
            if message.get("route"):
                if message["route"].startswith("historical:"):
                    route_label = "Governed historical analytics path"
                elif message["route"].startswith("deterministic:"):
                    route_label = "Deterministic fast path"
                else:
                    route_label = f"Groq LLM path · {message.get('model', groq_model)}"
                st.caption(route_label)
            if message.get("sql"):
                with st.expander("Generated governed SQL"):
                    st.code(message["sql"], language="sql")
            message_data = pd.DataFrame(message.get("data", []))
            if should_display_result_table(latest_user_question, message_data):
                st.dataframe(crore_display(message_data), hide_index=True, width="stretch")
    question = st.chat_input("Ask a lending portfolio question")
    if question:
        response = answer_hybrid_bi(
            question,
            df,
            st.session_state.bi_messages,
            historical_df=history_df,
            api_key=groq_api_key or None,
            model=groq_model,
            insight_temperature=insight_temperature,
            insight_model=groq_insight_model,
        )
        display = response.data.head(200)
        show_table = should_display_result_table(question, display)
        st.session_state.bi_messages.extend([
            {"role": "user", "content": question},
            {
                "role": "assistant",
                "content": response.answer,
                "data": display.to_dict("records"),
                "show_table": show_table,
                "route": response.route,
                "sql": response.sql,
                "model": response.model,
            },
        ])
        st.session_state.token_usage = response.usage
        event("PORTFOLIO", "conversational_bi", response.route, f"question={question[:300]}; rows={len(response.data)}", "bi_assistant")
        st.rerun()

elif page == "Campaign agent":
    st.subheader("AI/ML-powered Campaign Agent", icon=":material/campaign:")
    st.caption(f"Convert authorized prediction outputs into governed campaign audiences · scope: {active_scope}.")
    audience = st.selectbox(
        "AI/ML campaign source",
        CAMPAIGN_AUDIENCES,
        key="campaign_audience",
        persist_state="session",
    )
    eligible, campaign_meta = campaign_audience(df, audience)
    branch_options = (["All branches"] if current_user["role"] == "HO" else []) + sorted(eligible.branch_code.dropna().unique().tolist())
    branch = st.selectbox("Branch scope", branch_options, key="campaign_branch", persist_state="session")
    if branch != "All branches":
        eligible = eligible[eligible.branch_code == branch]
    st.info(f"Campaign purpose: {campaign_meta['purpose']}", icon=":material/target:")
    metric_a, metric_b, metric_c = st.columns(3)
    metric_a.metric("Contactable audience", f"{len(eligible):,}")
    metric_b.metric("Average model score", f"{eligible.campaign_score.mean():.1%}" if len(eligible) else "—")
    metric_c.metric("Branches represented", f"{eligible.branch_code.nunique():,}" if len(eligible) else "0")
    campaign_columns = ["customer_id", "name", "branch_code", "city", "campaign_source", "model_output", "campaign_score", "recommended_product", "recommended_action", "preferred_channel", "contact_hour"]
    st.dataframe(crore_display(eligible[campaign_columns].head(500)), hide_index=True, width="stretch")
    st.download_button(
        "Download selected campaign audience",
        eligible[campaign_columns].to_csv(index=False),
        f"{audience.split(' — ')[0].lower().replace(' ', '_').replace('/', '_')}_campaign.csv",
        "text/csv",
        icon=":material/download:",
    )
    if len(eligible):
        labels = {row.customer_id: f"{row.customer_id} · {row['name']} · score {row.campaign_score:.1%}" for _, row in eligible.iterrows()}
        cid = st.selectbox("Campaign prospect", eligible.customer_id.tolist(), format_func=lambda value: labels[value])
        if cid not in set(eligible.customer_id):
            st.error("Customer is outside your authorized branch scope.")
            st.stop()
        c = eligible[eligible.customer_id == cid].iloc[0]
        with st.container(border=True):
            st.markdown(f"**Model output:** {c.model_output}")
            st.markdown(f"**Recommended action:** {c.recommended_action}")
            st.caption(f"Branch {c.branch_code} · {c.city} · preferred channel {c.preferred_channel} · score {c.campaign_score:.1%}")
        channel = st.selectbox("Channel", ["WhatsApp", "SMS", "Voice bot"], index=["WhatsApp", "SMS", "Voice"].index(c.preferred_channel) if c.preferred_channel in ["WhatsApp", "SMS", "Voice"] else 0)
        authorized_recipients = authorized_test_numbers()
        selected_recipients = st.multiselect(
            "Authorized test recipients",
            authorized_recipients,
            default=list(authorized_recipients[:1]),
            format_func=lambda number: f"Test recipient · ******{number[-4:]}",
        )
        test_recipient = selected_recipients[0] if len(selected_recipients) == 1 else None
        campaign_message = campaign_meta["message"]
        if channel == "WhatsApp":
            campaign_message = "Utkarsh Bank demo: you may be interested in a suitable lending product. This is a simulated outreach; no offer is being made."
        st.text_area("Purpose-specific campaign message", campaign_message if channel != "Voice bot" else ("Twilio-hosted trial text-to-speech sample" if twilio_trial_mode() else VOICE_SCRIPT), height=110, disabled=True)
        st.caption("This is a simulated campaign workflow. Launch links open your own messaging or phone app; you decide whether to send or dial.")
        if test_recipient in authorized_recipients:
            links = launch_links(phone=test_recipient, message=campaign_message, authorized_numbers=authorized_recipients)
            if channel == "WhatsApp":
                st.link_button("Open WhatsApp draft", links["whatsapp"])
            elif channel == "SMS":
                st.markdown(f"[Open SMS draft]({links['sms']})")
                st.code(TEST_MESSAGE, language=None)
            else:
                st.markdown(f"[Open phone dialer]({links['call']})")
                st.code(VOICE_SCRIPT, language=None)
        elif len(selected_recipients) > 1:
            st.caption("Manual launch links and voice calls are available only when one recipient is selected. Twilio SMS and WhatsApp support the selected bulk audience.")
        st.info(f"Suggested local contact time: {int(c.contact_hour)}:00. Consent is present in demo data. Frequency caps and vendor delivery checks require integration.")
        if channel in {"SMS", "WhatsApp"}:
            st.markdown(f"#### Send test {channel}")
            if channel == "WhatsApp":
                st.caption("The approved Twilio Content Template must contain the standard demo message shown above.")
            missing_message_config = twilio_message_configuration_status(channel)
            if missing_message_config:
                st.warning("Twilio messaging is unavailable. Configure " + ", ".join(missing_message_config) + ".")
            message_confirmed = st.checkbox(
                f"I confirm all {len(selected_recipients)} selected number(s) are authorized to receive the test {channel} message",
                key=f"confirm_{channel.lower()}",
            )
            can_send_message = bool(selected_recipients) and set(selected_recipients).issubset(authorized_recipients) and int(c.consent) == 1 and not missing_message_config and message_confirmed
            button_label = f"Send bulk test {channel}" if len(selected_recipients) > 1 else f"Send test {channel}"
            if st.button(button_label, disabled=not can_send_message, type="primary"):
                accepted, failures = [], []
                for recipient in selected_recipients:
                    try:
                        message_sid, message_status = send_test_message(channel, campaign_message, recipient)
                        event(cid, "campaign", f"twilio_{channel.lower()}_requested", f"recipient=******{recipient[-4:]}; message_sid={message_sid}; status={message_status}", "campaign_agent")
                        accepted.append(f"******{recipient[-4:]} · {message_status} · {message_sid}")
                    except (ValueError, HTTPError, URLError, TimeoutError) as exc:
                        event(cid, "campaign", f"twilio_{channel.lower()}_failed", f"recipient=******{recipient[-4:]}; error_type={type(exc).__name__}", "campaign_agent")
                        failures.append(f"******{recipient[-4:]} · {str(exc) if isinstance(exc, ValueError) else type(exc).__name__}")
                if accepted:
                    st.success(f"Twilio accepted {len(accepted)} of {len(selected_recipients)} {channel} message(s).")
                    st.code("\n".join(accepted), language=None)
                if failures:
                    st.error(f"{len(failures)} {channel} message(s) failed.")
                    st.code("\n".join(failures), language=None)
        if st.button("Log test campaign preview"):
            if not selected_recipients:
                st.error("Select at least one authorized test recipient.")
            else:
                masked_recipients = ",".join(f"******{number[-4:]}" for number in selected_recipients)
                event(cid,"campaign","test_preview",f"source={audience}; score={c.campaign_score}; purpose={campaign_meta['purpose']}; channel={channel}; recipients={masked_recipients}; suggested_hour={c.contact_hour}; no message sent", "campaign_agent")
                st.success("Test preview logged. No external message was sent.")
        if channel == "Voice bot":
            st.markdown("#### Twilio voice bot POC")
            st.write("Twilio will call the test number using its hosted text-to-speech trial sample." if twilio_trial_mode() else "Twilio will call the test number and speak the sample campaign script.")
            if twilio_trial_mode():
                st.caption("Custom Bajaj personal loan script prepared for an upgraded account:")
                st.code(VOICE_SCRIPT, language=None)
            missing_twilio = twilio_configuration_status()
            if missing_twilio:
                st.warning("Twilio calling is unavailable. Configure " + ", ".join(missing_twilio) + ".")
            if len(selected_recipients) != 1:
                st.info("Select exactly one authorized recipient for a Twilio voice-bot call.")
            twilio_confirmed = st.checkbox("I confirm this number is authorized for the Twilio POC call")
            can_twilio_call = len(selected_recipients) == 1 and test_recipient in authorized_recipients and not missing_twilio and call_window_open() and twilio_confirmed
            if st.button("Place Twilio voice bot call", disabled=not can_twilio_call, type="primary"):
                try:
                    call_sid, call_status = place_twilio_test_call(test_recipient)
                    event(cid, "campaign", "twilio_test_call_requested", f"recipient=******{test_recipient[-4:]}; call_sid={call_sid}; status={call_status}", "campaign_agent")
                    st.success(f"Twilio accepted the call. Call ID: {call_sid}; status: {call_status}.")
                except (ValueError, HTTPError, URLError, TimeoutError) as exc:
                    event(cid, "campaign", "twilio_test_call_failed", f"recipient=******{test_recipient[-4:]}; error_type={type(exc).__name__}", "campaign_agent")
                    if isinstance(exc, ValueError):
                        st.error(str(exc))
                    else:
                        st.error(f"Twilio call failed ({type(exc).__name__}). Check network connectivity and Twilio service status.")
            st.markdown("#### Airtel IQ test call")
            st.write("The Airtel call flow should play an approved, generic script and record the recipient's response. No customer data from this demo is sent to Airtel.")
            missing = configuration_status()
            if missing:
                st.warning("Live calling is unavailable. Configure " + ", ".join(missing) + ".")
            if not call_window_open():
                st.warning("The configured calling window is closed. Default: 09:00–19:00 India Standard Time.")
            confirmed = st.checkbox("I confirm this test number is authorized for this call and the Airtel call flow is approved")
            can_call = test_recipient == TEST_NUMBER and not missing and call_window_open() and confirmed
            if test_recipient != TEST_NUMBER:
                st.info(f"Live test calling is restricted to {TEST_NUMBER}.")
            if st.button("Place Airtel test call", disabled=not can_call, type="primary"):
                try:
                    status, body = place_test_call(test_recipient)
                    event(cid,"campaign","airtel_test_call_requested",f"recipient=******{test_recipient[-4:]}; http_status={status}; no customer attributes shared", "campaign_agent")
                    st.success(f"Airtel accepted the call request (HTTP {status}). Check Airtel's call status for delivery and outcome.")
                except (ValueError, HTTPError, URLError, TimeoutError) as exc:
                    event(cid,"campaign","airtel_test_call_failed",f"recipient=******{test_recipient[-4:]}; error_type={type(exc).__name__}", "campaign_agent")
                    st.error(f"Call request failed ({type(exc).__name__}). Check endpoint, credentials, payload and Airtel logs.")

elif page == "LMS onboarding":
    st.subheader("Lead capture and onboarding")
    source = st.radio("Lead source", ["Campaign response", "Organic"], horizontal=True)
    cid = st.selectbox("Customer", df.customer_id.tolist())
    c = df[df.customer_id == cid].iloc[0]
    app = rows("SELECT * FROM applications WHERE customer_id=?", (cid,))
    st.write(f"**{c['name']}** · {c.segment} · Indicative ceiling ₹{float(c.amount) / 10_000_000:,.2f} Cr")
    amount = st.number_input("Requested amount (₹)", min_value=0, max_value=2000000, value=int(c.requested_amount), step=5000)
    kyc = st.checkbox("KYC documents and identity verified", value=bool(c.kyc))
    aa = st.checkbox("Account aggregator consent and data reviewed", value=True)
    if not app.empty: st.info(f"Current application: {app.iloc[0]['status']}")
    if st.button("Save onboarding"):
        status = "Ready for underwriting" if kyc and aa else "Documents pending"
        upsert_application(cid,status,amount,float(c.rate),36,source)
        event(cid,"onboarding",status,f"source={source}; amount={amount}; kyc={kyc}; aa_reviewed={aa}","lms")
        st.success(status)

elif page == "Underwriting & disbursal":
    st.subheader("Application decision")
    apps = rows("SELECT * FROM applications ORDER BY updated_at DESC")
    if apps.empty:
        st.info("Create an application in LMS onboarding first.")
    else:
        st.dataframe(crore_display(apps), hide_index=True, width="stretch")
        cid = st.selectbox("Application", apps.customer_id.tolist())
        c = df[df.customer_id == cid].iloc[0]
        app = apps[apps.customer_id == cid].iloc[0]
        st.write(f"Policy segment: **{c.segment}** · Risk **{c.risk:.1%}** · Reasons **{c.reasons}**")
        st.write(f"Affordability ceiling: **₹{float(c.amount) / 10_000_000:,.2f} Cr** · Indicative annual rate: **{c.rate:.1f}%**")
        st.warning("A binding approval, rate, agreement and fund transfer need approved policy, independent verification, and authorized controls.")
        decision = st.selectbox("Officer decision", ["Pending", "Approve for pilot simulation", "Reject", "Request review"])
        rationale = st.text_input("Decision rationale / override reason")
        if st.button("Save underwriting decision"):
            if decision == "Pending" or not rationale.strip():
                st.error("Select a decision and enter a rationale.")
            elif decision == "Approve for pilot simulation" and (c.segment == "decline" or float(app.amount) > float(c.amount)):
                st.error("Cannot approve outside eligibility or affordability ceiling. Send for review.")
            else:
                upsert_application(cid,decision,float(app.amount),float(c.rate),36,app.source)
                event(cid,"underwriting",decision,rationale,"credit_officer")
                st.success("Decision saved.")
        if app.status == "Approve for pilot simulation" and st.button("Record simulated disbursal"):
            upsert_application(cid,"Simulated disbursed",float(app.amount),float(app.rate),36,app.source)
            event(cid,"disbursal","simulated_disbursal",f"amount={app.amount}; no funds transferred","lms")
            st.success("Simulated disbursal recorded. No funds transferred.")

elif page == "Collections agent":
    st.subheader("Early warning and collections worklist")
    book = df[(df.dpd > 0) & (df.cohort == "pilot")].copy()
    book["priority"] = book.apply(lambda c: round(min(1, c.risk + c.dpd/100), 2), axis=1)
    book = book.sort_values("priority", ascending=False)
    st.dataframe(book[["customer_id","name","dpd","risk","priority","consent"]], hide_index=True, width="stretch")
    if len(book):
        cid = st.selectbox("Account", book.customer_id.tolist())
        c = book[book.customer_id == cid].iloc[0]
        action = "Offer self-service payment link and gentle reminder" if c.dpd < 16 else "Offer approved payment plan and specialist review" if c.dpd < 30 else "Escalate to trained collections specialist"
        st.info(f"Suggested action: {action}")
        if st.button("Log collection recommendation"):
            event(cid,"collections","recommendation",f"dpd={c.dpd}; priority={c.priority}; action={action}; no contact sent","collections_agent")
            st.success("Recommendation logged.")

elif page == "Policy assistant":
    st.subheader("Policy grounded assistant")
    st.caption("Retrieves passages from the local illustrative policy. Replace with approved, versioned Bajaj documents and access controls.")
    query = st.text_input("Ask about selection, offers, underwriting or collections", "Who is pre-approved?")
    if query:
        for section in retrieve_policy(query):
            title, _, body = section.partition("\n")
            with st.expander(title.lstrip("# "), expanded=True): st.write(body)
        st.caption(f"Source: policy/pilot_policy.md · {POLICY_VERSION}")

elif page == "Journey audit":
    st.subheader("Journey events")
    events = rows("SELECT * FROM journey_events ORDER BY created_at DESC LIMIT 500")
    st.dataframe(events, hide_index=True, width="stretch")
    if len(events): st.download_button("Download events CSV", events.to_csv(index=False), "journey_events.csv", "text/csv")


with token_panel:
    st.markdown("#### Token usage")
    usage_table = pd.DataFrame(
        [{"Stage": stage, "Tokens": tokens} for stage, tokens in st.session_state.token_usage.items()]
    )
    styled_usage = (
        usage_table.style
        .hide(axis="index")
        .set_properties(**{
            "background-color": "#FFFFFF",
            "color": "#17324D",
            "border-color": "#D9E2EC",
            "font-size": "0.88rem",
        })
        .set_table_styles([
            {
                "selector": "th",
                "props": [
                    ("background-color", "#EAF3F8"),
                    ("color", "#123A52"),
                    ("font-weight", "700"),
                    ("border-color", "#CBD8E2"),
                    ("font-size", "0.88rem"),
                ],
            }
        ])
    )
    st.table(styled_usage, border="horizontal", width="stretch")

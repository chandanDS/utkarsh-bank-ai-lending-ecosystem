import os
import streamlit as st
import pandas as pd
import altair as alt
from pathlib import Path
from dotenv import load_dotenv
from urllib.error import HTTPError, URLError
from engine import authenticate_user, init_db, rows, scored_customers, score, event, upsert_application, retrieve_policy, POLICY_VERSION
from airtel_voice import TEST_NUMBER, call_window_open, configuration_status, place_test_call
from campaign_links import TEST_MESSAGE, VOICE_SCRIPT, launch_links
from lending_analytics import AI_COLUMNS, CAMPAIGN_AUDIENCES, campaign_audience, enrich_portfolio, executive_impact_report, lead_conversion_analytics
from hybrid_bi import DEFAULT_INSIGHT_MODEL, DEFAULT_MODEL, answer_hybrid_bi, zero_usage
from twilio_voice import configuration_status as twilio_configuration_status, place_test_call as place_twilio_test_call, trial_mode as twilio_trial_mode

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
TOP_HEADER_LOGO = PROJECT_ROOT / "assets" / "utkarsh_bank_logo.svg"
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
section[data-testid="stSidebar"] [data-testid="stSidebarContent"] {{
    overflow-y: hidden;
}}
section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {{
    gap: 0.45rem;
}}
section[data-testid="stSidebar"] div[role="radiogroup"] {{
    gap: 0.42rem;
}}
section[data-testid="stSidebar"] div[role="radiogroup"] label {{
    background: rgba(255, 255, 255, 0.09);
    border: 1px solid rgba(255, 255, 255, 0.20);
    border-radius: 9px;
    padding: 0.52rem 0.60rem;
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
.st-key-brand_header {{
    background: rgba(255, 255, 255, 0.94);
    border: 1px solid rgba(75, 46, 131, 0.20);
    border-radius: 10px;
    box-shadow: 0 4px 16px rgba(47, 33, 84, 0.08);
    padding: 0.75rem 1.15rem;
    margin-bottom: 0.85rem;
}}
.st-key-brand_header [data-testid="stImage"] {{
    background: #FFFFFF;
    border: 1px solid rgba(75, 46, 131, 0.22);
    border-radius: 9px;
    box-shadow: 0 2px 8px rgba(47, 33, 84, 0.08);
    padding: 0.80rem 0.95rem;
}}
.st-key-brand_header img {{
    filter: drop-shadow(0 2px 5px rgba(47, 33, 84, 0.16));
}}
.journey-grid {{
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
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

with st.container(key="brand_header"):
    brand_logo, brand_title = st.columns([1.35, 4], vertical_alignment="center")
    with brand_logo:
        st.image(TOP_HEADER_LOGO, width="stretch")
    with brand_title:
        st.title("Utkarsh Bank AI-Powered Lending Ecosystem")
        st.caption(
            "Predict customer needs and risks · Activate governed campaigns · Explore portfolio intelligence"
        )

st.html("""
<div class="journey-grid" aria-label="AI-powered lending journey">
  <section class="journey-card" style="--journey-accent:#4B2E83">
    <span class="journey-number">STAGE 01</span><h3>1. Predict</h3>
    <p>AI/ML identifies opportunity, risk and next-best action.</p>
  </section>
  <section class="journey-card" style="--journey-accent:#E97924">
    <span class="journey-number">STAGE 02</span><h3>2. Activate</h3>
    <p>Campaign Agent converts predictions into governed actions.</p>
  </section>
  <section class="journey-card" style="--journey-accent:#315CA8">
    <span class="journey-number">STAGE 03</span><h3>3. Understand</h3>
    <p>Conversational BI answers grounded portfolio questions.</p>
  </section>
  <section class="journey-card" style="--journey-accent:#15803D">
    <span class="journey-number">STAGE 04</span><h3>4. Measure</h3>
    <p>Executive reporting quantifies conversion and business impact.</p>
  </section>
</div>
""")

pages = ["AI/ML intelligence", "Campaign agent", "Conversational BI", "Executive Business Impact report"]
st.sidebar.title("Utkarsh Bank")
st.sidebar.caption("AI-Powered Lending Ecosystem")
with st.sidebar.container(border=True):
    st.markdown(f"**{current_user['display_name']}**")
    st.caption(f"Username: {current_user['username']}")
    st.badge("Head Office" if current_user["role"] == "HO" else "Branch user", color="violet")
    active_scope = "All branches" if current_user["role"] == "HO" else current_user["branch_code"]
    st.caption(f"Data scope: {active_scope}")
    if st.button("Sign out", icon=":material/logout:", width="stretch"):
        st.session_state.clear()
        st.rerun()
st.sidebar.subheader("Demo modules", icon=":material/apps:")
st.sidebar.caption("Select a workspace")
page = st.sidebar.radio("Demo modules", pages, label_visibility="collapsed")

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
with st.sidebar.popover(
    "Groq AI settings",
    icon=":material/tune:",
    width="stretch",
):
    st.markdown("**Provider:** Groq")
    st.caption(f"SQL planner: {groq_model}")
    st.caption(f"Insight model: {groq_insight_model}")
    st.caption("Simple questions use the zero-token deterministic path. Complex SQL uses the 120B planner.")
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
if page not in pages:
    st.error("Unauthorized module selection.")
    st.stop()

if "token_usage" not in st.session_state:
    st.session_state.token_usage = zero_usage()
if "bi_messages" not in st.session_state:
    st.session_state.bi_messages = []
token_panel = st.sidebar.container()

@st.cache_data(ttl=60, show_spinner=False)
def load_portfolio(branch_code):
    return enrich_portfolio(scored_customers(branch_code))

branch_scope = current_user["branch_code"] if current_user["role"] == "BRANCH" else None
df = load_portfolio(branch_scope)
pilot = df[df.cohort == "pilot"].copy()
control = df[df.cohort == "control"].copy()

if page == "LMS overview":
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
    impact_tab, funnel_tab = st.tabs([
        "AI/ML business impact",
        "Lead conversion waterfall",
    ])
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

    with funnel_tab:
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
    tabs = st.tabs(["Propensity", "Default & early warning", "Churn", "Cross-sell", "Credit decisioning", "Personalization", "Collections", "Next best action"])
    with tabs[0]:
        view = df.sort_values("propensity", ascending=False)
        st.metric("High-propensity customers", f"{int((view.propensity >= .75).sum()):,}")
        st.dataframe(view[AI_COLUMNS].head(500), hide_index=True, width="stretch")
    with tabs[1]:
        view = df.sort_values(["risk", "dpd"], ascending=False)
        st.metric("High default-risk customers", f"{int((view.risk >= .60).sum()):,}")
        st.dataframe(view[AI_COLUMNS + ["dpd", "collection_priority"]].head(500), hide_index=True, width="stretch")
    with tabs[2]:
        view = df.sort_values("churn_probability", ascending=False)
        st.metric("High churn-risk customers", f"{int((view.churn_probability >= .65).sum()):,}")
        st.dataframe(view[AI_COLUMNS].head(500), hide_index=True, width="stretch")
    with tabs[3]:
        view = df.sort_values("cross_sell_score", ascending=False)
        st.metric("Priority cross-sell customers", f"{int((view.cross_sell_score >= .70).sum()):,}")
        st.dataframe(view[AI_COLUMNS].head(500), hide_index=True, width="stretch")
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
        st.dataframe(crore_display(view[decision_columns].head(500)), hide_index=True, width="stretch")
    with tabs[5]:
        st.caption("Personalized product recommendation, offer context and preferred engagement channel for each customer.")
        view = df.sort_values(["propensity", "cross_sell_score", "risk"], ascending=[False, False, True])
        with st.container(horizontal=True):
            st.metric("High-personalization opportunities", f"{int((view.propensity >= .75).sum()):,}", border=True)
            st.metric("Recommended products", f"{view.recommended_product.nunique():,}", border=True)
            st.metric("Preferred channels", f"{view.preferred_channel.nunique():,}", border=True)
        personalization_columns = [
            "customer_id", "name", "branch_code", "city", "customer_segment", "product",
            "recommended_product", "propensity", "cross_sell_score", "preferred_channel",
            "contact_hour", "amount", "rate",
        ]
        st.dataframe(crore_display(view[personalization_columns].head(500)), hide_index=True, width="stretch")
    with tabs[6]:
        view = df[df.dpd > 0].sort_values("collection_priority", ascending=False)
        st.metric("Active collections worklist", f"{len(view):,}")
        st.dataframe(crore_display(view[AI_COLUMNS + ["dpd", "collection_amount_due", "collection_amount_received", "collection_priority"]].head(500)), hide_index=True, width="stretch")
    with tabs[7]:
        st.caption("Prioritized next action based on delinquency, credit risk, churn and opportunity signals.")
        action_rank = {
            "Specialist collections review": 0,
            "Manual credit review": 1,
            "Retention outreach": 2,
        }
        view = df.assign(_action_rank=df["recommended_action"].map(action_rank).fillna(3)).sort_values(
            ["_action_rank", "collection_priority", "risk", "propensity"],
            ascending=[True, False, False, False],
        )
        immediate_actions = int((view.recommended_action != "Continue relationship nurturing").sum())
        with st.container(horizontal=True):
            st.metric("Immediate actions", f"{immediate_actions:,}", border=True)
            st.metric("Action types", f"{view.recommended_action.nunique():,}", border=True)
            st.metric("Customers prioritized", f"{len(view):,}", border=True)
        action_columns = [
            "customer_id", "name", "branch_code", "city", "recommended_action",
            "recommended_product", "risk", "propensity", "churn_probability",
            "collection_priority", "preferred_channel", "contact_hour", "reasons",
        ]
        st.dataframe(view[action_columns].head(500), hide_index=True, width="stretch")
    st.download_button("Download complete prediction output", df[AI_COLUMNS].to_csv(index=False), "lending_prediction_output.csv", "text/csv", icon=":material/download:")

elif page == "Conversational BI":
    st.subheader("Conversational BI", icon=":material/query_stats:")
    st.caption(f"Ask grounded business questions across the authorized portfolio scope: {active_scope}.")
    with st.container(border=True):
        st.markdown("**Try:** total outstanding portfolio · branch with highest delinquency · high default-risk customers · PL eligibility · cross-sell opportunities · collection efficiency · churn risk · DPD greater than 30")
    for message in st.session_state.bi_messages[-6:]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("route"):
                route_label = "Deterministic fast path" if message["route"].startswith("deterministic:") else f"Groq LLM path · {message.get('model', groq_model)}"
                st.caption(route_label)
            if message.get("sql"):
                with st.expander("Generated governed SQL"):
                    st.code(message["sql"], language="sql")
            if message.get("data"):
                st.dataframe(crore_display(pd.DataFrame(message["data"])), hide_index=True, width="stretch")
    question = st.chat_input("Ask a lending portfolio question")
    if question:
        response = answer_hybrid_bi(
            question,
            df,
            st.session_state.bi_messages,
            api_key=groq_api_key or None,
            model=groq_model,
            insight_temperature=insight_temperature,
            insight_model=groq_insight_model,
        )
        display = response.data.head(200)
        st.session_state.bi_messages.extend([
            {"role": "user", "content": question},
            {
                "role": "assistant",
                "content": response.answer,
                "data": display.to_dict("records"),
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
        test_recipient = st.text_input("Test recipient mobile", value="8050702529", max_chars=10)
        campaign_message = campaign_meta["message"]
        st.text_area("Purpose-specific campaign message", campaign_message if channel != "Voice bot" else ("Twilio-hosted trial text-to-speech sample" if twilio_trial_mode() else VOICE_SCRIPT), height=110, disabled=True)
        st.caption("This is a simulated campaign workflow. Launch links open your own messaging or phone app; you decide whether to send or dial.")
        if test_recipient == TEST_NUMBER:
            links = launch_links(message=campaign_message)
            if channel == "WhatsApp":
                st.link_button("Open WhatsApp draft", links["whatsapp"])
            elif channel == "SMS":
                st.markdown(f"[Open SMS draft]({links['sms']})")
                st.code(TEST_MESSAGE, language=None)
            else:
                st.markdown(f"[Open phone dialer]({links['call']})")
                st.code(VOICE_SCRIPT, language=None)
        else:
            st.warning(f"Sample launch links are available only for {TEST_NUMBER}.")
        st.info(f"Suggested local contact time: {int(c.contact_hour)}:00. Consent is present in demo data. Frequency caps and vendor delivery checks require integration.")
        if st.button("Log test campaign preview"):
            if not (test_recipient.isdigit() and len(test_recipient) == 10):
                st.error("Enter a 10 digit mobile number.")
            else:
                event(cid,"campaign","test_preview",f"source={audience}; score={c.campaign_score}; purpose={campaign_meta['purpose']}; channel={channel}; recipient=******{test_recipient[-4:]}; suggested_hour={c.contact_hour}; no message sent", "campaign_agent")
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
            twilio_confirmed = st.checkbox("I confirm this number is authorized for the Twilio POC call")
            can_twilio_call = test_recipient == TEST_NUMBER and not missing_twilio and call_window_open() and twilio_confirmed
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
            "font-size": "0.82rem",
        })
        .set_table_styles([
            {
                "selector": "th",
                "props": [
                    ("background-color", "#EAF3F8"),
                    ("color", "#123A52"),
                    ("font-weight", "700"),
                    ("border-color", "#CBD8E2"),
                ],
            }
        ])
    )
    st.table(styled_usage, border="horizontal", width="stretch")

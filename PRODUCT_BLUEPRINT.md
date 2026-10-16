# Bajaj loan lifecycle product blueprint

## Decision and journey sequence

| Stage | Input | Decision or action | Outcome and event |
|---|---|---|---|
| Offer base | Bajaj customer/offer records | Resolve identity, deduplicate, assign randomized pilot/control | Eligible pilot candidates and immutable assignment |
| Permissions | Consent, DNC, channel preference | Suppress prohibited contacts | Reason coded exclusions |
| Eligibility | Bajaj, bureau, AA, policy | Age, income, delinquency, KYC, data quality gates | Decline, review, or continue |
| Models | Features frozen at decision time | Propensity, probability of default, fraud, affordability | Versioned scores and explanations |
| Offer | Approved policy and rate card | Segment, cap, tenor, APR, expected unit economics | Pre-approved, pre-qualified, underwriting, or reject |
| Campaign | Offer, contact preference, engagement | Next best channel, time, message, frequency | Draft, approval, send receipt, response |
| LMS | Response or organic lead | Application, documents, KYC, AA consent, verification | Application status history |
| Underwriting | Application and refreshed data | Final policy/risk checks and officer exceptions | Decision and reason codes |
| Disbursal | Signed agreement and mandate | Authorized payment instructions and reconciliation | Loan account and disbursal reference |
| Servicing | Schedule and payments | Early warning of missed payment | Risk tier and recommended help |
| Collections | DPD, risk, response, hardship | Contact strategy, self-service, plans, escalation | Contact and cure history |

## Decision architecture

Use a **policy engine** for hard constraints, **models** for ranking and expected outcomes, and an **orchestrator** for permitted next actions. An LLM agent may retrieve policy, prepare explanations and messages, and collect structured data. It should not bypass policy or independently authorize funds. Every decision stores feature snapshot, score, model version, policy version, reason codes, consent, actor, timestamp, and override.

Production models should be trained on dated historical cohorts with leakage checks and time based holdouts. Calibrate default probabilities and estimate approval/take-up uplift, expected loss, and contribution margin. Optimize an offer under affordability, risk appetite, pricing, and fairness constraints. Monitor drift, approval bias, disparate outcomes, complaints, and realized losses. Refresh or pause models when thresholds are breached.

## Pilot design

Begin with a small, stratified, randomized customer subset. Keep control on today's process. Define exposure before outreach and use intent-to-treat reporting. Guardrails: contact complaints, delinquency, fraud, loss, fairness, and manual review volume. Primary decision metric: incremental risk adjusted contribution margin per assigned customer. Secondary metrics: conversion, disbursal, decision time, cost per booked loan, early DPD, cure rate, and customer experience. Expand only after risk and compliance review of observed outcomes.

## Integration backlog

1. Replace synthetic records with governed Bajaj, bureau, and AA connectors, identity matching, consent, lineage, and freshness checks.
2. Load approved, effective dated Bajaj policy and rate cards into a citation capable retrieval index with access control.
3. Train, validate, approve, and monitor production models; record versions and reason codes.
4. Connect WhatsApp/SMS/voice vendors with opt out, quiet hours, caps, human review, receipts, and complaint handling.
5. Add application workflow, KYC/fraud integrations, document store, e-sign, payment mandate, disbursal authorization, and reconciliation.
6. Add auth, roles, encryption, audit retention, observability, and incident controls.
7. Run shadow mode, then a randomized pilot with approval gates and rollback triggers.

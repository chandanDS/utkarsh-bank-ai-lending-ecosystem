# Conversational BI evaluation

## Result

- Dataset: 10,000 production-shaped synthetic customer records and a 36-month historical mart.
- Scripted conversational scenarios: 27.
- Full automated result: 71 tests and 43 subtests passed.
- External data transfer: none. Dynamic Groq planning was tested with local simulated responses through the real JSON-plan, SQL-validation, scoped-execution, token-allocation and grounded-insight code paths.

## Coverage

1. Current-state KPIs and branch rankings.
2. MoM, QoQ and YoY comparisons.
3. Product, branch, region and customer-segment analysis.
4. Portfolio driver drill-downs and false-premise correction.
5. AI/ML prediction-output retrieval and actionable customer queues.
6. RBAC preservation for branch-scoped questions.
7. Prompt-injection blocking and read-only SQL validation.
8. Executive summaries versus explicitly requested supporting detail.
9. Token accounting for deterministic and dynamic paths.

## Issues found and corrected

- Current-state portfolio and collection questions were incorrectly routed to historical YoY analysis.
- Churn prediction-output requests were incorrectly routed to churn trends.
- State portfolio-contribution questions were incorrectly treated as growth-driver questions.
- Generic complex comparisons were incorrectly assumed to be portfolio YoY comparisons.
- Customer-detail results exposed the full internal dataframe instead of use-case-relevant fields.

## Management demo story

### 1. Observe

- How is the overall lending portfolio performing?
- Show customer attrition compared with last year.
- Compare lead conversion quarter on quarter.
- Show collection efficiency trend.
- Show manual-underwriting quality and turnaround trends in the Observe page.

### 2. Analyze

- Default-rate YoY by branch.
- What are the reasons for the highest default rate in Delhi?
- New accounts MoM by product.
- Give me a drill-down for the South-region portfolio to understand why it is declining.
- What are the drivers for Personal Loan growth?
- What percentage of the portfolio is contributed by each state?

### 3. Predict

- List Default & early-warning prediction data for Delhi branch.
- Show propensity output for Mumbai branch.
- Show churn-prediction customers for Delhi branch.
- Show Credit decisioning output for Delhi branch.
- Show Cross-sell prediction output for Pune branch.

### 4. Act

- Recommend how to improve collection efficiency for Delhi branch.
- How can we improve attrition for Delhi branch?
- Recommend actions to improve campaign efficiency for Delhi branch.
- List high-propensity consented customers for a Personal Loan campaign.
- Which preferred channel and contact time should be used for these customers?

### 5. Monitor

- Compare lead conversion QoQ after campaign activation.
- Show cross-sell conversion YoY by customer segment.
- Show collection-efficiency movement after prioritized outreach.
- Compare default-rate movement by branch.
- Which region has the lowest portfolio growth?

## Presentation behavior

- Scalar and top-one executive questions return a direct answer without a redundant table.
- “By branch/product/segment,” “list,” “details,” “breakdown,” and drill-down questions return compact supporting tables.
- Prediction and action requests return only relevant customer fields, not internal model columns.
- Improving metrics do not produce unnecessary interventions; deteriorating metrics link to the relevant AI/ML output or governed action queue.

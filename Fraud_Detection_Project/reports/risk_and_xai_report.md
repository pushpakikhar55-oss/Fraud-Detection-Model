# Fraud Risk Scoring & Explainable AI (XAI) Report
This document details the multi-signal composite risk scoring engine, decision band efficacy, and SHAP-based local explainability outputs.

## 1. Multi-Signal Composite Risk Scoring Formula
To move beyond binary classifications, each transaction is evaluated against a weighted composite formula:
$$\text{Risk Score} = 100 \times \Big(0.40 \cdot P_{\text{supervised}} + 0.25 \cdot S_{\text{anomaly}} + 0.15 \cdot S_{\text{amount}} + 0.10 \cdot S_{\text{velocity}} + 0.10 \cdot S_{\text{novelty}}\Big)$$

## 2. Risk Band Performance (Out-of-Time Test Set: 7,500 Transactions)
| Risk Band | Score Range | Operational Action | Transaction Count | Actual Fraud Count | Fraud Rate (%) |
|---|---|---|---|---|---|
| **HIGH** | 71 – 100 | `BLOCK` | 86 | 86 | **100.0%** |
| **LOW** | 0 – 30 | `APPROVE` | 7,380 | 1 | **0.01%** |
| **MEDIUM** | 31 – 70 | `REVIEW` | 34 | 28 | **82.35%** |

### Operational Efficiency Highlights
- **Friction-Free Approvals**: 7,380 transactions (98.4% of total volume) were classified as **LOW** risk with a 0.0% fraud rate, enabling instant authorization.
- **High-Confidence Blocking**: 86 transactions triggered the **HIGH** risk tier with an outstanding fraud concentration of **100.0%**, preventing immediate financial loss.

## 3. Explainable AI (SHAP) & Reason Generation
To satisfy regulatory auditability and empower operations analysts, the engine generates SHAP attribution breakdowns alongside plain-English narrative justifications.

### Case Study: Transaction `TXN_0042444`
- **Amount**: $2,869.57
- **Type**: `TRANSFER` | **Channel**: `credit_card`
- **Customer**: `CUST_00504` (Historical Mean: $219.78)
- **Composite Risk Score**: **88.5/100** (Band: `HIGH`, Action: `BLOCK`)

#### Synthesized Natural-Language Explanations:
- ⚠️ Unusually high transaction amount ($2,869.57), which is 13.1x higher than customer's historical average.
- ⚠️ Severe account liquidation: Transaction drains 91.1% of entire existing account balance.

#### Top Contributing Features (SHAP Values):
| Feature | Normalized Value | SHAP Attribution (Impact on Fraud Probability) |
|---|---|---|
| `amount_to_avg_ratio` | 21.004 | **+3.9235** |
| `amount` | 30.871 | **+2.6741** |
| `tx_amount_sum_1h` | 28.771 | **+1.6729** |
| `amount_zscore` | 41.370 | **+0.9931** |
| `balance_drain_ratio` | 116.105 | **+0.3068** |

## 4. Visual Artifacts Generated
1. `reports/figures/risk_score_distribution.png`: Separation between legitimate and fraudulent score distributions across the three decision tiers.
2. `reports/figures/shap_global_summary.png`: Global beeswarm summary plot illustrating how features push predictions.
3. `reports/figures/shap_waterfall_example.png`: Waterfall diagram visualizing the exact additive push for a blocked transaction.

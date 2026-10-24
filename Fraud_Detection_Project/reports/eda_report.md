# Exploratory Data Analysis & Fraud Characterization Report
**Generated**: Phase 1 Foundation Evaluation  
**Total Transactions Analyzed**: 35,000  

## 1. Class Imbalance Profile
- **Legitimate Transactions**: 34,470 (98.49%)
- **Fraudulent Transactions**: 530 (1.51%)
- **Imbalance Ratio**: 65.0 : 1

> **Analysis Note**: Extreme class imbalance is inherent to authentic financial settlement systems. Evaluating models strictly via Accuracy is invalid because a trivial zero-rule classifier achieves >98% accuracy while failing completely. Downstream ML models must be evaluated on PR-AUC, Recall, Precision, and F1-Score with appropriate class weight balancing.

## 2. Transaction Amount Dynamics
| Metric | Legitimate ($) | Fraudulent ($) |
|---|---|---|
| Mean | $110.16 | $5,321.8 |
| Median | $95.25 | $1,177.03 |
| Maximum | $615.17 | $42,407.95 |

Fraudulent transactions exhibit significantly higher mean amounts and heavy right-tailed spikes, particularly when paired with rapid liquidity exit channels.

## 3. High-Risk Transaction Types
| Transaction Type | Volume | Fraud Count | Fraud Rate (%) |
|---|---|---|---|
| `CASH_IN` | 3,349 | 0 | **0.0%** |
| `CASH_OUT` | 5,380 | 134 | **2.49%** |
| `DEBIT` | 5,244 | 0 | **0.0%** |
| `PAYMENT` | 15,659 | 195 | **1.25%** |
| `TRANSFER` | 5,368 | 201 | **3.74%** |

`TRANSFER` and `CASH_OUT` represent the primary exit avenues exploited by financial adversaries due to immediacy and irreversibility.

## 4. Top Feature Correlations with Fraud Target
| Feature | Pearson Correlation with `is_fraud` |
|---|---|
| `amount` | **+0.4891** |
| `tx_amount_sum_1h` | **+0.4878** |
| `amount_zscore` | **+0.4564** |
| `amount_to_avg_ratio` | **+0.4292** |
| `tx_amount_sum_24h` | **+0.3634** |
| `is_new_device` | **+0.2654** |
| `balance_drain_ratio` | **+0.2055** |
| `is_night` | **+0.1941** |


## 5. Artifacts & Visualizations
The following visual charts were rendered and archived under `reports/figures/`:
1. `class_imbalance.png`: Comparative volume on log scale.
2. `amount_distribution.png`: Log-scaled density comparison between legit and fraud amounts.
3. `fraud_by_type_and_channel.png`: Breakdown of fraud concentration across types and payment methods.
4. `temporal_fraud_patterns.png`: Hourly transaction volume contrasted against nocturnal fraud spikes.
5. `feature_correlation_heatmap.png`: Inter-feature correlation matrix.
6. `velocity_and_outliers.png`: Spending velocity vs personal amount deviation ratio.

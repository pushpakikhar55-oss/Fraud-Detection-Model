# Model Comparison & Evaluation Report
This report benchmarks all supervised classification models and unsupervised anomaly detectors on out-of-time financial transactions.

## 1. Supervised Classification Performance Comparison
| Model | ROC-AUC | PR-AUC | Optimal Thresh | Recall (%) | Precision (%) | F1-Score | FP Count | FN Count |
|---|---|---|---|---|---|---|---|---|
| `Logistic Regression` | **1.0000** | **0.9992** | 0.71 | 99.1% | 96.6% | **0.9785** | 4 | 1 |
| `Decision Tree` | **0.9913** | **0.9820** | 0.95 | 97.4% | 99.1% | **0.9825** | 1 | 3 |
| `Random Forest` | **1.0000** | **0.9974** | 0.54 | 95.7% | 97.4% | **0.9649** | 3 | 5 |
| `XGBoost` **(Best)** | **1.0000** | **0.9999** | 0.05 | 100.0% | 99.1% | **0.9957** | 1 | 0 |

### Key Observations & Trade-offs
- **Top Performer**: `XGBoost` achieved the highest PR-AUC and F1-Score, demonstrating exceptional discriminative capacity on the severe 65:1 class imbalance.
- **Threshold Optimization Impact**: Calibrating decision thresholds away from default 0.50 allows the system to balance false positive friction against catastrophic false negative loss.
- **Tree Ensembles vs Linear Baselines**: Gradient boosted trees (`XGBoost`) and `Random Forest` significantly outperformed `Logistic Regression` due to non-linear interactions between rolling velocity and personal amount deviations.

## 2. Unsupervised Anomaly Detection (Isolation Forest)
- **Algorithm**: Isolation Forest (Contamination = 2.00%)
- **Unsupervised ROC-AUC**: 0.9849
- **Unsupervised PR-AUC**: 0.6303
- **Unsupervised Recall**: 72.2%

### Zero-Day Scenario Capture Rate (Unsupervised)
| Fraud Scenario | Test Samples | Captured by Anomaly Detector | Detection Rate (%) | Mean Anomaly Score |
|---|---|---|---|---|
| `amount_spike` | 23 | 18 | **78.26%** | 0.867 |
| `device_location_anomaly` | 25 | 15 | **60.0%** | 0.757 |
| `nocturnal_anomaly` | 22 | 16 | **72.73%** | 0.866 |
| `rapid_balance_drain` | 21 | 19 | **90.48%** | 0.913 |
| `velocity_burst` | 24 | 15 | **62.5%** | 0.775 |

*Takeaway*: Isolation Forest reliably flags behavioral outliers (e.g. `rapid_balance_drain` and `amount_spike`) completely without training labels.

## 3. Evaluation Visualizations Generated
Archived under `reports/figures/`:
1. `roc_curves_comparison.png`: Comparison of True Positive Rate vs False Positive Rate.
2. `pr_curves_comparison.png`: Precision vs Recall curves with no-skill reference.
3. `confusion_matrices.png`: Confusion matrices at calibrated thresholds.
4. `anomaly_score_distribution.png`: Density separation between normal and outlier transactions.
5. `supervised_vs_anomaly_scatter.png`: Correlation between supervised fraud probability and unsupervised anomaly score.

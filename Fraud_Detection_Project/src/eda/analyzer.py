"""
Exploratory Data Analysis (EDA) Engine.
Generates publication-quality visualizations, statistical profiles,
and comprehensive reports on financial fraud dynamics.
"""
from pathlib import Path
from typing import Dict, Any
import numpy as np
import pandas as pd

# Set non-interactive matplotlib backend
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src.features.registry import FEATURE_METADATA


class EDAAnalyzer:
    """Performs deep statistical EDA and renders analytical figures & reports."""

    def __init__(self, figures_dir: str = "reports/figures", reports_dir: str = "reports"):
        self.figures_dir = Path(figures_dir)
        self.reports_dir = Path(reports_dir)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Consistent aesthetics
        sns.set_theme(style="whitegrid", font_scale=1.1)
        self.colors = {"legit": "#2b5c8f", "fraud": "#d9534f", "neutral": "#5cb85c"}

    def run_full_eda(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Executes full EDA pipeline: statistics, charts, data dictionary, and summary report.
        """
        print("[EDAAnalyzer] Calculating dataset statistics...")
        stats = self._calculate_statistics(df)

        print("[EDAAnalyzer] Rendering publication-quality figures...")
        self._plot_class_imbalance(df)
        self._plot_amount_distributions(df)
        self._plot_fraud_by_type_and_method(df)
        self._plot_temporal_patterns(df)
        self._plot_correlation_heatmap(df)
        self._plot_velocity_and_outliers(df)

        print("[EDAAnalyzer] Compiling Markdown EDA Report and Data Dictionary...")
        self._write_data_dictionary()
        self._write_eda_report(stats)

        print(f"[EDAAnalyzer] EDA complete. Figures saved to {self.figures_dir}/ and reports to {self.reports_dir}/")
        return stats

    def _calculate_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculates core statistical indicators and fraud differentials."""
        total = len(df)
        fraud_count = int(df["is_fraud"].sum())
        legit_count = total - fraud_count
        fraud_pct = (fraud_count / total) * 100

        # Amount statistics
        legit_amounts = df[df["is_fraud"] == 0]["amount"]
        fraud_amounts = df[df["is_fraud"] == 1]["amount"]

        amount_stats = {
            "legit_mean": round(float(legit_amounts.mean()), 2),
            "legit_median": round(float(legit_amounts.median()), 2),
            "legit_max": round(float(legit_amounts.max()), 2),
            "fraud_mean": round(float(fraud_amounts.mean()), 2),
            "fraud_median": round(float(fraud_amounts.median()), 2),
            "fraud_max": round(float(fraud_amounts.max()), 2),
        }

        # Fraud rate by transaction type
        type_rates = df.groupby("transaction_type")["is_fraud"].agg(["count", "sum", "mean"])
        type_rates["rate_pct"] = (type_rates["mean"] * 100).round(2)
        type_rates_dict = type_rates[["count", "sum", "rate_pct"]].to_dict(orient="index")

        # Fraud rate by merchant category
        cat_rates = df.groupby("merchant_category")["is_fraud"].agg(["count", "sum", "mean"])
        cat_rates["rate_pct"] = (cat_rates["mean"] * 100).round(2)
        cat_rates_dict = cat_rates[["count", "sum", "rate_pct"]].to_dict(orient="index")

        # Top correlations with is_fraud
        num_cols = df.select_dtypes(include=[np.number]).columns
        corr_series = df[num_cols].corr()["is_fraud"].drop("is_fraud").sort_values(ascending=False)
        top_correlations = corr_series.head(8).round(4).to_dict()

        return {
            "total_transactions": total,
            "legit_count": legit_count,
            "fraud_count": fraud_count,
            "fraud_percentage": round(fraud_pct, 3),
            "imbalance_ratio": f"{round(legit_count / max(1, fraud_count), 1)} : 1",
            "amount_stats": amount_stats,
            "type_rates": type_rates_dict,
            "cat_rates": cat_rates_dict,
            "top_correlations": top_correlations,
        }

    def _plot_class_imbalance(self, df: pd.DataFrame):
        """Figure 1: Target Class Imbalance."""
        plt.figure(figsize=(7, 5))
        counts = df["is_fraud"].value_counts().sort_index()
        bars = plt.bar(["Legitimate (0)", "Fraudulent (1)"], counts.values, color=[self.colors["legit"], self.colors["fraud"]], width=0.55)
        
        plt.title("Financial Transaction Class Distribution", fontsize=14, fontweight="bold", pad=15)
        plt.ylabel("Transaction Count (Log Scale)", fontsize=12)
        plt.yscale("log")
        
        # Add labels on top of bars
        for bar, count in zip(bars, counts.values):
            pct = (count / len(df)) * 100
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.15,
                f"{count:,}\n({pct:.2f}%)",
                ha="center",
                va="bottom",
                fontweight="semibold",
                fontsize=10
            )

        plt.ylim(1, len(df) * 3)
        plt.tight_layout()
        plt.savefig(self.figures_dir / "class_imbalance.png", dpi=300)
        plt.close()

    def _plot_amount_distributions(self, df: pd.DataFrame):
        """Figure 2: Transaction Amount Distribution (Log10 scale comparison)."""
        plt.figure(figsize=(9, 5))
        df_plot = df.copy()
        df_plot["log_amount"] = np.log10(df_plot["amount"] + 1)

        sns.kdeplot(
            data=df_plot[df_plot["is_fraud"] == 0],
            x="log_amount",
            label="Legitimate",
            color=self.colors["legit"],
            fill=True,
            alpha=0.35,
            linewidth=2,
        )
        sns.kdeplot(
            data=df_plot[df_plot["is_fraud"] == 1],
            x="log_amount",
            label="Fraudulent",
            color=self.colors["fraud"],
            fill=True,
            alpha=0.35,
            linewidth=2,
        )

        plt.title("Density Distribution: Transaction Amount (log10 Scale)", fontsize=14, fontweight="bold", pad=15)
        plt.xlabel("log10(Transaction Amount + $1)", fontsize=12)
        plt.ylabel("Density", fontsize=12)
        plt.legend(frameon=True, facecolor="white", edgecolor="none")
        plt.tight_layout()
        plt.savefig(self.figures_dir / "amount_distribution.png", dpi=300)
        plt.close()

    def _plot_fraud_by_type_and_method(self, df: pd.DataFrame):
        """Figure 3: Fraud Incidence by Transaction Type and Payment Method."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Subplot 1: Transaction Type
        type_summary = df.groupby("transaction_type")["is_fraud"].mean().reset_index()
        type_summary["rate_pct"] = type_summary["is_fraud"] * 100
        type_summary = type_summary.sort_values("rate_pct", ascending=False)

        sns.barplot(
            data=type_summary,
            x="transaction_type",
            y="rate_pct",
            palette="Reds_r",
            ax=axes[0],
        )
        axes[0].set_title("Fraud Rate by Transaction Type (%)", fontsize=13, fontweight="bold")
        axes[0].set_ylabel("Fraud Rate (%)", fontsize=11)
        axes[0].set_xlabel("Transaction Type", fontsize=11)
        axes[0].tick_params(axis="x", rotation=25)

        # Subplot 2: Payment Method
        pm_summary = df.groupby("payment_method")["is_fraud"].mean().reset_index()
        pm_summary["rate_pct"] = pm_summary["is_fraud"] * 100
        pm_summary = pm_summary.sort_values("rate_pct", ascending=False)

        sns.barplot(
            data=pm_summary,
            x="payment_method",
            y="rate_pct",
            palette="Blues_r",
            ax=axes[1],
        )
        axes[1].set_title("Fraud Rate by Payment Method (%)", fontsize=13, fontweight="bold")
        axes[1].set_ylabel("Fraud Rate (%)", fontsize=11)
        axes[1].set_xlabel("Payment Method", fontsize=11)
        axes[1].tick_params(axis="x", rotation=25)

        plt.tight_layout()
        plt.savefig(self.figures_dir / "fraud_by_type_and_channel.png", dpi=300)
        plt.close()

    def _plot_temporal_patterns(self, df: pd.DataFrame):
        """Figure 4: Fraud Probability by Hour of Day."""
        plt.figure(figsize=(10, 5))
        hour_summary = df.groupby("hour_of_day")["is_fraud"].agg(["count", "sum", "mean"]).reset_index()
        hour_summary["rate_pct"] = hour_summary["mean"] * 100

        ax1 = plt.gca()
        ax2 = ax1.twinx()

        # Bar: transaction volume
        ax1.bar(
            hour_summary["hour_of_day"],
            hour_summary["count"],
            color="#a0c4e2",
            alpha=0.6,
            label="Total Volume",
            width=0.7,
        )
        ax1.set_ylabel("Total Transactions", color="#2b5c8f", fontsize=12)
        ax1.grid(False)

        # Line: fraud rate %
        ax2.plot(
            hour_summary["hour_of_day"],
            hour_summary["rate_pct"],
            color=self.colors["fraud"],
            linewidth=2.5,
            marker="o",
            label="Fraud Rate (%)",
        )
        ax2.set_ylabel("Fraud Rate (%)", color=self.colors["fraud"], fontsize=12)
        ax2.grid(True, linestyle="--", alpha=0.5)

        plt.title("Temporal Dynamics: Transaction Volume vs. Fraud Rate by Hour", fontsize=14, fontweight="bold", pad=15)
        ax1.set_xlabel("Hour of the Day (0-23)", fontsize=12)
        ax1.set_xticks(range(0, 24))

        plt.tight_layout()
        plt.savefig(self.figures_dir / "temporal_fraud_patterns.png", dpi=300)
        plt.close()

    def _plot_correlation_heatmap(self, df: pd.DataFrame):
        """Figure 5: Feature Correlation Matrix Heatmap."""
        plt.figure(figsize=(10, 8))
        feature_cols = [
            "amount",
            "old_balance",
            "new_balance",
            "amount_to_avg_ratio",
            "amount_zscore",
            "tx_count_1h",
            "tx_count_24h",
            "tx_amount_sum_1h",
            "is_night",
            "is_new_device",
            "is_new_location",
            "balance_drain_ratio",
            "is_fraud",
        ]
        active_cols = [c for c in feature_cols if c in df.columns]
        corr = df[active_cols].corr()

        sns.heatmap(
            corr,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            vmin=-0.2,
            vmax=0.8,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8},
        )
        plt.title("Feature Correlation Matrix with Fraud Indicator", fontsize=14, fontweight="bold", pad=15)
        plt.tight_layout()
        plt.savefig(self.figures_dir / "feature_correlation_heatmap.png", dpi=300)
        plt.close()

    def _plot_velocity_and_outliers(self, df: pd.DataFrame):
        """Figure 6: Spending Velocity vs Deviation Ratio Scatter."""
        plt.figure(figsize=(9, 6))
        
        # Subsample non-fraud for visual clarity
        sample_legit = df[df["is_fraud"] == 0].sample(n=min(3000, len(df[df["is_fraud"] == 0])), random_state=42)
        sample_fraud = df[df["is_fraud"] == 1]

        plt.scatter(
            sample_legit["tx_count_1h"],
            sample_legit["amount_to_avg_ratio"],
            alpha=0.3,
            color=self.colors["legit"],
            s=20,
            label="Legitimate",
        )
        plt.scatter(
            sample_fraud["tx_count_1h"],
            sample_fraud["amount_to_avg_ratio"],
            alpha=0.8,
            color=self.colors["fraud"],
            s=45,
            edgecolor="black",
            linewidth=0.5,
            label="Fraudulent",
        )

        plt.title("Behavioral Anomalies: Amount Deviation Ratio vs. 1-Hour Velocity", fontsize=14, fontweight="bold", pad=15)
        plt.xlabel("Transactions in Past 1 Hour (Velocity)", fontsize=12)
        plt.ylabel("Amount to Personal Average Ratio", fontsize=12)
        plt.ylim(0, 30)
        plt.legend(frameon=True, facecolor="white")
        plt.tight_layout()
        plt.savefig(self.figures_dir / "velocity_and_outliers.png", dpi=300)
        plt.close()

    def _write_data_dictionary(self):
        """Compiles canonical data dictionary into markdown."""
        md_content = ["# Financial Fraud Detection - Data Dictionary\n"]
        md_content.append("This document details the canonical schema, data types, categories, and fraud detection rationale for every attribute.\n")
        md_content.append("## Transaction Schema Attributes\n")
        md_content.append("| Field Name | Type | Description |\n")
        md_content.append("|---|---|---|\n")
        md_content.append("| `transaction_id` | String | Unique alpha-numeric identifier for each transaction event. |\n")
        md_content.append("| `timestamp` | Datetime | Exact UTC timestamp of transaction occurrence. |\n")
        md_content.append("| `customer_id` | String | Unique identifier for originating customer account. |\n")
        md_content.append("| `amount` | Float | Transaction monetary value in currency units. |\n")
        md_content.append("| `transaction_type` | String | Financial movement category: `PAYMENT`, `TRANSFER`, `CASH_OUT`, `DEBIT`, `CASH_IN`. |\n")
        md_content.append("| `merchant_id` | String | Recipient merchant account identifier. |\n")
        md_content.append("| `merchant_category` | String | Merchant industry classification (e.g. retail, travel, financial_services). |\n")
        md_content.append("| `device_id` | String | Hardware device fingerprint identifier. |\n")
        md_content.append("| `location` | String | Originating city/geographic region. |\n")
        md_content.append("| `payment_method` | String | Channel: `credit_card`, `debit_card`, `bank_transfer`, `upi`, `digital_wallet`. |\n")
        md_content.append("| `card_present` | Binary | 1 if physically presented at POS terminal, 0 if card-not-present (online/API). |\n")
        md_content.append("| `old_balance` | Float | Customer account balance immediately prior to transaction. |\n")
        md_content.append("| `new_balance` | Float | Customer account balance immediately following transaction. |\n")
        md_content.append("| `is_fraud` | Binary | Supervised ground truth target (0 = Legitimate, 1 = Fraudulent). |\n")
        md_content.append("| `fraud_scenario` | String | Specific simulated modus operandi or attack vector. |\n\n")

        md_content.append("## Engineered Behavioral & Velocity Features\n")
        md_content.append("| Feature Name | Category | Type | Rationale & Behavioral Definition |\n")
        md_content.append("|---|---|---|---|\n")
        for feat, meta in FEATURE_METADATA.items():
            md_content.append(f"| `{feat}` | {meta['category']} | {meta['type']} | {meta['description']} *Why it matters*: {meta['rationale']} |\n")

        with open(self.reports_dir / "data_dictionary.md", "w") as f:
            f.write("".join(md_content))

    def _write_eda_report(self, stats: Dict[str, Any]):
        """Generates formal Markdown report summarizing Phase 1 EDA findings."""
        md = ["# Exploratory Data Analysis & Fraud Characterization Report\n"]
        md.append(f"**Generated**: Phase 1 Foundation Evaluation  \n")
        md.append(f"**Total Transactions Analyzed**: {stats['total_transactions']:,}  \n\n")
        
        md.append("## 1. Class Imbalance Profile\n")
        md.append(f"- **Legitimate Transactions**: {stats['legit_count']:,} ({100 - stats['fraud_percentage']:.2f}%)\n")
        md.append(f"- **Fraudulent Transactions**: {stats['fraud_count']:,} ({stats['fraud_percentage']:.2f}%)\n")
        md.append(f"- **Imbalance Ratio**: {stats['imbalance_ratio']}\n\n")
        md.append("> **Analysis Note**: Extreme class imbalance is inherent to authentic financial settlement systems. Evaluating models strictly via Accuracy is invalid because a trivial zero-rule classifier achieves >98% accuracy while failing completely. Downstream ML models must be evaluated on PR-AUC, Recall, Precision, and F1-Score with appropriate class weight balancing.\n\n")

        md.append("## 2. Transaction Amount Dynamics\n")
        md.append("| Metric | Legitimate ($) | Fraudulent ($) |\n")
        md.append("|---|---|---|\n")
        md.append(f"| Mean | ${stats['amount_stats']['legit_mean']:,} | ${stats['amount_stats']['fraud_mean']:,} |\n")
        md.append(f"| Median | ${stats['amount_stats']['legit_median']:,} | ${stats['amount_stats']['fraud_median']:,} |\n")
        md.append(f"| Maximum | ${stats['amount_stats']['legit_max']:,} | ${stats['amount_stats']['fraud_max']:,} |\n\n")
        md.append("Fraudulent transactions exhibit significantly higher mean amounts and heavy right-tailed spikes, particularly when paired with rapid liquidity exit channels.\n\n")

        md.append("## 3. High-Risk Transaction Types\n")
        md.append("| Transaction Type | Volume | Fraud Count | Fraud Rate (%) |\n")
        md.append("|---|---|---|---|\n")
        for tx_type, data in stats["type_rates"].items():
            md.append(f"| `{tx_type}` | {data['count']:,} | {data['sum']:,} | **{data['rate_pct']}%** |\n")
        md.append("\n`TRANSFER` and `CASH_OUT` represent the primary exit avenues exploited by financial adversaries due to immediacy and irreversibility.\n\n")

        md.append("## 4. Top Feature Correlations with Fraud Target\n")
        md.append("| Feature | Pearson Correlation with `is_fraud` |\n")
        md.append("|---|---|\n")
        for feat, corr_val in stats["top_correlations"].items():
            md.append(f"| `{feat}` | **{corr_val:+.4f}** |\n")
        md.append("\n\n## 5. Artifacts & Visualizations\n")
        md.append("The following visual charts were rendered and archived under `reports/figures/`:\n")
        md.append("1. `class_imbalance.png`: Comparative volume on log scale.\n")
        md.append("2. `amount_distribution.png`: Log-scaled density comparison between legit and fraud amounts.\n")
        md.append("3. `fraud_by_type_and_channel.png`: Breakdown of fraud concentration across types and payment methods.\n")
        md.append("4. `temporal_fraud_patterns.png`: Hourly transaction volume contrasted against nocturnal fraud spikes.\n")
        md.append("5. `feature_correlation_heatmap.png`: Inter-feature correlation matrix.\n")
        md.append("6. `velocity_and_outliers.png`: Spending velocity vs personal amount deviation ratio.\n")

        with open(self.reports_dir / "eda_report.md", "w") as f:
            f.write("".join(md))

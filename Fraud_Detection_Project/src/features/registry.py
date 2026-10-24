"""
Feature Catalog and Metadata Registry.
Documents all engineered features, their business logic, and fraud rationale.
"""
from typing import Dict, Any

FEATURE_METADATA: Dict[str, Dict[str, Any]] = {
    "amount": {
        "type": "float",
        "category": "transaction",
        "description": "Transaction value in currency units.",
        "rationale": "High-dollar transactions carry higher financial exposure.",
    },
    "amount_to_avg_ratio": {
        "type": "float",
        "category": "behavioral",
        "description": "Ratio of current transaction amount to customer's historical average amount.",
        "rationale": "Sudden anomalous spikes (e.g. 5x-20x baseline) strongly indicate compromised credentials.",
    },
    "amount_zscore": {
        "type": "float",
        "category": "behavioral",
        "description": "Standardized deviation (z-score) of amount relative to customer's personal history.",
        "rationale": "Measures statistical abnormality in standard deviations from typical spending.",
    },
    "tx_count_1h": {
        "type": "int",
        "category": "velocity",
        "description": "Number of transactions completed by the customer in the preceding 1 hour.",
        "rationale": "Automated bot attacks and rapid card testing produce dense transaction bursts.",
    },
    "tx_count_24h": {
        "type": "int",
        "category": "velocity",
        "description": "Number of transactions completed by the customer in the preceding 24 hours.",
        "rationale": "Elevated daily frequency signals rapid account takeover or card skimming sprees.",
    },
    "tx_amount_sum_1h": {
        "type": "float",
        "category": "velocity",
        "description": "Total cumulative spending by the customer in the preceding 1 hour.",
        "rationale": "Draining large amounts in short windows is typical before account freeze.",
    },
    "tx_amount_sum_24h": {
        "type": "float",
        "category": "velocity",
        "description": "Total cumulative spending by the customer in the preceding 24 hours.",
        "rationale": "Monitors total daily capital outflow.",
    },
    "hour_of_day": {
        "type": "int",
        "category": "temporal",
        "description": "Hour of transaction execution (0 to 23).",
        "rationale": "Certain hours have disproportionate fraud likelihood.",
    },
    "day_of_week": {
        "type": "int",
        "category": "temporal",
        "description": "Day of the week (0=Monday, 6=Sunday).",
        "rationale": "Weekend transactions may exhibit different behavioral thresholds.",
    },
    "is_weekend": {
        "type": "binary",
        "category": "temporal",
        "description": "Flag indicating if transaction occurred on Saturday or Sunday.",
        "rationale": "Fraud operations frequently exploit reduced bank monitoring staff on weekends.",
    },
    "is_night": {
        "type": "binary",
        "category": "temporal",
        "description": "Flag indicating transaction executed between 23:00 and 05:00.",
        "rationale": "Nocturnal unauthorized transactions while victims sleep delay detection.",
    },
    "is_new_device": {
        "type": "binary",
        "category": "novelty",
        "description": "Flag indicating the device ID has never been previously used by this customer.",
        "rationale": "Account takeover typically originates from attacker-controlled unauthorized devices.",
    },
    "is_new_location": {
        "type": "binary",
        "category": "novelty",
        "description": "Flag indicating the geographic location has never been visited by this customer.",
        "rationale": "Impossible travel or overseas transactions are strong red flags.",
    },
    "balance_drain_ratio": {
        "type": "float",
        "category": "account",
        "description": "Transaction amount divided by initial balance (amount / old_balance).",
        "rationale": "Attackers attempt to liquidate entire balances before fraud blocks occur.",
    },
    "balance_error_orig": {
        "type": "float",
        "category": "account",
        "description": "Discrepancy between old balance, amount, and resulting new balance.",
        "rationale": "Systematic accounting irregularities frequently accompany synthetic wire fraud.",
    },
    "is_high_risk_type": {
        "type": "binary",
        "category": "transaction",
        "description": "Binary flag indicating high-risk transaction types (TRANSFER, CASH_OUT).",
        "rationale": "Irreversible exit channels are preferred by fraudsters over reversible purchases.",
    },
    "card_not_present": {
        "type": "binary",
        "category": "channel",
        "description": "Flag indicating digital / card-not-present transaction.",
        "rationale": "Card-not-present transactions account for over 70% of modern card fraud.",
    }
}

# Financial Fraud Detection - Data Dictionary
This document details the canonical schema, data types, categories, and fraud detection rationale for every attribute.
## Transaction Schema Attributes
| Field Name | Type | Description |
|---|---|---|
| `transaction_id` | String | Unique alpha-numeric identifier for each transaction event. |
| `timestamp` | Datetime | Exact UTC timestamp of transaction occurrence. |
| `customer_id` | String | Unique identifier for originating customer account. |
| `amount` | Float | Transaction monetary value in currency units. |
| `transaction_type` | String | Financial movement category: `PAYMENT`, `TRANSFER`, `CASH_OUT`, `DEBIT`, `CASH_IN`. |
| `merchant_id` | String | Recipient merchant account identifier. |
| `merchant_category` | String | Merchant industry classification (e.g. retail, travel, financial_services). |
| `device_id` | String | Hardware device fingerprint identifier. |
| `location` | String | Originating city/geographic region. |
| `payment_method` | String | Channel: `credit_card`, `debit_card`, `bank_transfer`, `upi`, `digital_wallet`. |
| `card_present` | Binary | 1 if physically presented at POS terminal, 0 if card-not-present (online/API). |
| `old_balance` | Float | Customer account balance immediately prior to transaction. |
| `new_balance` | Float | Customer account balance immediately following transaction. |
| `is_fraud` | Binary | Supervised ground truth target (0 = Legitimate, 1 = Fraudulent). |
| `fraud_scenario` | String | Specific simulated modus operandi or attack vector. |

## Engineered Behavioral & Velocity Features
| Feature Name | Category | Type | Rationale & Behavioral Definition |
|---|---|---|---|
| `amount` | transaction | float | Transaction value in currency units. *Why it matters*: High-dollar transactions carry higher financial exposure. |
| `amount_to_avg_ratio` | behavioral | float | Ratio of current transaction amount to customer's historical average amount. *Why it matters*: Sudden anomalous spikes (e.g. 5x-20x baseline) strongly indicate compromised credentials. |
| `amount_zscore` | behavioral | float | Standardized deviation (z-score) of amount relative to customer's personal history. *Why it matters*: Measures statistical abnormality in standard deviations from typical spending. |
| `tx_count_1h` | velocity | int | Number of transactions completed by the customer in the preceding 1 hour. *Why it matters*: Automated bot attacks and rapid card testing produce dense transaction bursts. |
| `tx_count_24h` | velocity | int | Number of transactions completed by the customer in the preceding 24 hours. *Why it matters*: Elevated daily frequency signals rapid account takeover or card skimming sprees. |
| `tx_amount_sum_1h` | velocity | float | Total cumulative spending by the customer in the preceding 1 hour. *Why it matters*: Draining large amounts in short windows is typical before account freeze. |
| `tx_amount_sum_24h` | velocity | float | Total cumulative spending by the customer in the preceding 24 hours. *Why it matters*: Monitors total daily capital outflow. |
| `hour_of_day` | temporal | int | Hour of transaction execution (0 to 23). *Why it matters*: Certain hours have disproportionate fraud likelihood. |
| `day_of_week` | temporal | int | Day of the week (0=Monday, 6=Sunday). *Why it matters*: Weekend transactions may exhibit different behavioral thresholds. |
| `is_weekend` | temporal | binary | Flag indicating if transaction occurred on Saturday or Sunday. *Why it matters*: Fraud operations frequently exploit reduced bank monitoring staff on weekends. |
| `is_night` | temporal | binary | Flag indicating transaction executed between 23:00 and 05:00. *Why it matters*: Nocturnal unauthorized transactions while victims sleep delay detection. |
| `is_new_device` | novelty | binary | Flag indicating the device ID has never been previously used by this customer. *Why it matters*: Account takeover typically originates from attacker-controlled unauthorized devices. |
| `is_new_location` | novelty | binary | Flag indicating the geographic location has never been visited by this customer. *Why it matters*: Impossible travel or overseas transactions are strong red flags. |
| `balance_drain_ratio` | account | float | Transaction amount divided by initial balance (amount / old_balance). *Why it matters*: Attackers attempt to liquidate entire balances before fraud blocks occur. |
| `balance_error_orig` | account | float | Discrepancy between old balance, amount, and resulting new balance. *Why it matters*: Systematic accounting irregularities frequently accompany synthetic wire fraud. |
| `is_high_risk_type` | transaction | binary | Binary flag indicating high-risk transaction types (TRANSFER, CASH_OUT). *Why it matters*: Irreversible exit channels are preferred by fraudsters over reversible purchases. |
| `card_not_present` | channel | binary | Flag indicating digital / card-not-present transaction. *Why it matters*: Card-not-present transactions account for over 70% of modern card fraud. |

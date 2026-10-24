"""
High-fidelity Synthetic Financial Transaction Generator.
Simulates realistic customer profiles, spending distributions, temporal patterns,
and real-world fraud vectors for fraud detection systems.
"""
from datetime import datetime, timedelta
import random
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from src.data.schema import (
    TRANSACTION_SCHEMA,
    TRANSACTION_TYPES,
    MERCHANT_CATEGORIES,
    PAYMENT_METHODS,
    POPULAR_LOCATIONS,
)


class CustomerProfile:
    """Represents a simulated banking customer with habitual behavioral baselines."""

    def __init__(self, customer_id: str, rng: random.Random, np_rng: np.random.Generator):
        self.customer_id = customer_id
        self.home_location = rng.choice(POPULAR_LOCATIONS[:-1])  # Exclude 'International' as home
        self.primary_device = f"DEV_{rng.randint(100000, 999999)}"
        self.secondary_device = f"DEV_{rng.randint(100000, 999999)}"
        self.payment_methods = rng.sample(PAYMENT_METHODS, k=rng.randint(1, 3))
        self.primary_payment_method = self.payment_methods[0]
        
        # Financial baselines: log-normal spending distribution
        self.mean_amount = float(np_rng.uniform(25.0, 180.0))
        self.std_amount = float(self.mean_amount * np_rng.uniform(0.3, 0.6))
        self.current_balance = float(np_rng.uniform(1500.0, 45000.0))
        
        # Typical hour preference (peak around 12:00 - 18:00)
        self.peak_hour = rng.randint(11, 19)


class FinancialDataGenerator:
    """
    Generates synthetic transaction logs simulating authentic banking transactions
    interleaved with known financial fraud patterns.
    """

    def __init__(
        self,
        num_customers: int = 1000,
        fraud_rate: float = 0.015,
        random_seed: int = 42,
    ):
        self.num_customers = num_customers
        self.fraud_rate = fraud_rate
        self.random_seed = random_seed
        self.rng = random.Random(random_seed)
        self.np_rng = np.random.default_rng(random_seed)
        
        self.customers: List[CustomerProfile] = [
            CustomerProfile(f"CUST_{i+1:05d}", self.rng, self.np_rng)
            for i in range(num_customers)
        ]
        
        self.merchants = [
            (f"MERC_{i+1:04d}", self.rng.choice(MERCHANT_CATEGORIES))
            for i in range(250)
        ]

    def generate(
        self,
        num_records: int = 50000,
        start_date: Optional[datetime] = None,
        duration_days: int = 60,
    ) -> pd.DataFrame:
        """
        Generates a DataFrame of transactions spanning duration_days.
        """
        if start_date is None:
            start_date = datetime(2026, 1, 1, 0, 0, 0)
            
        records: List[Dict] = []
        total_seconds = duration_days * 24 * 3600
        
        # Pre-assign base timestamps across time window
        time_offsets = np.sort(self.np_rng.uniform(0, total_seconds, size=num_records))
        
        fraud_target_count = int(num_records * self.fraud_rate)
        # Select indices to turn into fraudulent sequences / events
        fraud_indices = set(self.rng.sample(range(num_records), fraud_target_count))
        
        tx_counter = 1
        for idx in range(num_records):
            customer = self.rng.choice(self.customers)
            base_time = start_date + timedelta(seconds=float(time_offsets[idx]))
            
            is_fraud = 1 if idx in fraud_indices else 0
            
            if is_fraud:
                scenario = self.rng.choice([
                    "amount_spike",
                    "velocity_burst",
                    "nocturnal_anomaly",
                    "device_location_anomaly",
                    "rapid_balance_drain",
                ])
                txn = self._generate_fraud_transaction(
                    tx_id=f"TXN_{tx_counter:07d}",
                    customer=customer,
                    timestamp=base_time,
                    scenario=scenario,
                )
            else:
                txn = self._generate_legit_transaction(
                    tx_id=f"TXN_{tx_counter:07d}",
                    customer=customer,
                    timestamp=base_time,
                )
                
            records.append(txn)
            tx_counter += 1

        df = pd.DataFrame(records)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)
        return df

    def _generate_legit_transaction(
        self,
        tx_id: str,
        customer: CustomerProfile,
        timestamp: datetime,
    ) -> Dict:
        """Generates a regular legitimate financial transaction matching habitual behavior."""
        # Hour adjustment towards customer's active hours
        hour_offset = int(self.np_rng.normal(customer.peak_hour, 3.5)) % 24
        # Mostly active between 7am and 10pm
        tx_time = timestamp.replace(hour=hour_offset, minute=self.rng.randint(0, 59), second=self.rng.randint(0, 59))
        
        # Amount: log-normal around customer baseline
        log_mean = np.log(customer.mean_amount)
        amount = round(float(self.np_rng.lognormal(log_mean, 0.4)), 2)
        amount = max(1.50, min(amount, customer.mean_amount * 3.5))
        
        tx_type = self.rng.choices(
            TRANSACTION_TYPES,
            weights=[0.45, 0.15, 0.15, 0.15, 0.10],
            k=1
        )[0]
        
        merchant_id, merchant_cat = self.rng.choice(self.merchants)
        if tx_type in ["TRANSFER", "CASH_OUT"]:
            merchant_cat = "financial_services"
            
        device = customer.primary_device if self.rng.random() < 0.88 else customer.secondary_device
        location = customer.home_location if self.rng.random() < 0.94 else self.rng.choice(POPULAR_LOCATIONS[:-1])
        payment_method = customer.primary_payment_method if self.rng.random() < 0.80 else self.rng.choice(customer.payment_methods)
        
        card_present = 1 if payment_method in ["credit_card", "debit_card"] and self.rng.random() < 0.65 else 0
        
        old_bal = customer.current_balance
        if tx_type in ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT"]:
            new_bal = max(0.0, round(old_bal - amount, 2))
        else: # CASH_IN
            new_bal = round(old_bal + amount, 2)
            
        # Update customer running balance slightly for continuity
        customer.current_balance = max(100.0, new_bal)

        return {
            "transaction_id": tx_id,
            "timestamp": tx_time,
            "customer_id": customer.customer_id,
            "amount": amount,
            "transaction_type": tx_type,
            "merchant_id": merchant_id,
            "merchant_category": merchant_cat,
            "device_id": device,
            "location": location,
            "payment_method": payment_method,
            "card_present": card_present,
            "old_balance": round(old_bal, 2),
            "new_balance": round(new_bal, 2),
            "is_fraud": 0,
            "fraud_scenario": "none",
        }

    def _generate_fraud_transaction(
        self,
        tx_id: str,
        customer: CustomerProfile,
        timestamp: datetime,
        scenario: str,
    ) -> Dict:
        """Injects targeted anomalous signatures corresponding to specific fraud MOs."""
        old_bal = customer.current_balance
        
        if scenario == "amount_spike":
            # 8x to 22x user's baseline
            amount = round(float(customer.mean_amount * self.np_rng.uniform(8.0, 22.0)), 2)
            tx_time = timestamp
            tx_type = self.rng.choice(["TRANSFER", "PAYMENT", "CASH_OUT"])
            merchant_id, merchant_cat = self.rng.choice(self.merchants)
            device = customer.primary_device
            location = customer.home_location
            payment_method = self.rng.choice(["credit_card", "bank_transfer", "digital_wallet"])
            card_present = 0
            new_bal = max(0.0, round(old_bal - amount, 2))

        elif scenario == "velocity_burst":
            # High-velocity transaction, digital goods or retail
            amount = round(float(customer.mean_amount * self.np_rng.uniform(2.0, 6.0)), 2)
            tx_time = timestamp
            tx_type = self.rng.choice(["PAYMENT", "TRANSFER"])
            merchant_id, _ = self.rng.choice(self.merchants)
            merchant_cat = self.rng.choice(["digital_goods", "travel", "retail"])
            device = f"DEV_{self.rng.randint(900000, 999999)}"
            location = customer.home_location
            payment_method = self.rng.choice(["credit_card", "upi", "digital_wallet"])
            card_present = 0
            new_bal = max(0.0, round(old_bal - amount, 2))

        elif scenario == "nocturnal_anomaly":
            # Middle of the night (01:00 to 04:30 AM) with high transfer
            night_hour = self.rng.choice([1, 2, 3, 4])
            tx_time = timestamp.replace(hour=night_hour, minute=self.rng.randint(5, 55))
            amount = round(float(self.np_rng.uniform(1200.0, 8500.0)), 2)
            tx_type = self.rng.choice(["TRANSFER", "CASH_OUT"])
            merchant_id, _ = self.rng.choice(self.merchants)
            merchant_cat = "financial_services"
            device = f"DEV_ROGUE_{self.rng.randint(100, 999)}"
            location = self.rng.choice(POPULAR_LOCATIONS)
            payment_method = "bank_transfer"
            card_present = 0
            new_bal = max(0.0, round(old_bal - amount, 2))

        elif scenario == "device_location_anomaly":
            # Rogue foreign or distant location + unknown device
            amount = round(float(customer.mean_amount * self.np_rng.uniform(3.0, 10.0)), 2)
            tx_time = timestamp
            tx_type = "PAYMENT"
            merchant_id, _ = self.rng.choice(self.merchants)
            merchant_cat = self.rng.choice(["travel", "digital_goods", "retail"])
            device = f"DEV_UNKNOWN_{self.rng.randint(1000, 9999)}"
            location = "International" if self.rng.random() < 0.6 else self.rng.choice(
                [loc for loc in POPULAR_LOCATIONS if loc != customer.home_location]
            )
            payment_method = "credit_card"
            card_present = 0
            new_bal = max(0.0, round(old_bal - amount, 2))

        elif scenario == "rapid_balance_drain":
            # Drain 90% - 100% of entire balance
            drain_ratio = self.np_rng.uniform(0.92, 1.0)
            amount = round(float(old_bal * drain_ratio), 2)
            amount = max(500.0, amount)
            tx_time = timestamp
            tx_type = self.rng.choice(["TRANSFER", "CASH_OUT"])
            merchant_id, _ = self.rng.choice(self.merchants)
            merchant_cat = "financial_services"
            device = customer.primary_device
            location = customer.home_location
            payment_method = "bank_transfer"
            card_present = 0
            new_bal = max(0.0, round(old_bal - amount, 2))

        else:
            # Fallback
            amount = round(customer.mean_amount * 5.0, 2)
            tx_time = timestamp
            tx_type = "TRANSFER"
            merchant_id, merchant_cat = self.rng.choice(self.merchants)
            device = customer.primary_device
            location = customer.home_location
            payment_method = "bank_transfer"
            card_present = 0
            new_bal = max(0.0, round(old_bal - amount, 2))

        return {
            "transaction_id": tx_id,
            "timestamp": tx_time,
            "customer_id": customer.customer_id,
            "amount": amount,
            "transaction_type": tx_type,
            "merchant_id": merchant_id,
            "merchant_category": merchant_cat,
            "device_id": device,
            "location": location,
            "payment_method": payment_method,
            "card_present": card_present,
            "old_balance": round(old_bal, 2),
            "new_balance": round(new_bal, 2),
            "is_fraud": 1,
            "fraud_scenario": scenario,
        }

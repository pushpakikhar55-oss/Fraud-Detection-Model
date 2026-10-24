"""
Database persistence layer using SQLite with Write-Ahead Logging (WAL) mode.
Handles storage of transactions, risk predictions, security alerts, and analytics summaries.
"""
import sqlite3
import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timezone
import pandas as pd


class DatabaseManager:
    """Manages SQLite tables, transaction persistence, alert triage, and analytics queries."""

    def __init__(self, db_path: Optional[str] = None):
        # An explicit path remains useful for tests and local scripts. In a
        # container, FRAUD_DB_PATH points at a mounted persistent volume.
        self.db_path = Path(db_path or os.getenv("FRAUD_DB_PATH", "data/fraud_platform.db"))
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Returns a connection configured with row factory and WAL mode."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def init_db(self):
        """Initializes tables and indexes."""
        with self._get_connection() as conn:
            conn.executescript("""
            CREATE TABLE IF NOT EXISTS transactions (
                transaction_id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                customer_id TEXT NOT NULL,
                amount REAL NOT NULL,
                transaction_type TEXT NOT NULL,
                merchant_category TEXT,
                payment_method TEXT,
                location TEXT,
                device_id TEXT,
                card_present INTEGER DEFAULT 0,
                old_balance REAL DEFAULT 0.0,
                new_balance REAL DEFAULT 0.0,
                is_fraud INTEGER DEFAULT 0,
                supervised_prob REAL DEFAULT 0.0,
                anomaly_score REAL DEFAULT 0.0,
                risk_score REAL NOT NULL,
                risk_band TEXT NOT NULL,
                recommended_action TEXT NOT NULL,
                explanation_json TEXT,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS alerts (
                alert_id INTEGER PRIMARY KEY AUTOINCREMENT,
                transaction_id TEXT NOT NULL UNIQUE,
                created_at TEXT NOT NULL,
                risk_score REAL NOT NULL,
                severity TEXT NOT NULL,
                status TEXT NOT NULL, -- NEW, UNDER_REVIEW, RESOLVED, CONFIRMED_FRAUD
                notes TEXT,
                FOREIGN KEY (transaction_id) REFERENCES transactions(transaction_id)
            );

            CREATE INDEX IF NOT EXISTS idx_txn_timestamp ON transactions(timestamp);
            CREATE INDEX IF NOT EXISTS idx_txn_risk_band ON transactions(risk_band);
            CREATE INDEX IF NOT EXISTS idx_txn_customer ON transactions(customer_id);
            CREATE INDEX IF NOT EXISTS idx_alert_status ON alerts(status);
            """)

    def insert_transaction(self, data: Dict[str, Any]) -> str:
        """Inserts a scored transaction and automatically triggers an alert if high risk."""
        tx_id = data.get("transaction_id") or f"TXN_API_{int(datetime.now().timestamp()*1000)}"
        now_str = datetime.now(timezone.utc).isoformat()

        exp_json = data.get("explanation_json")
        if isinstance(exp_json, (dict, list)):
            exp_json = json.dumps(exp_json)

        with self._get_connection() as conn:
            conn.execute("""
            INSERT OR REPLACE INTO transactions (
                transaction_id, timestamp, customer_id, amount, transaction_type,
                merchant_category, payment_method, location, device_id, card_present,
                old_balance, new_balance, is_fraud, supervised_prob, anomaly_score,
                risk_score, risk_band, recommended_action, explanation_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                tx_id,
                str(data.get("timestamp", now_str)),
                str(data.get("customer_id", "CUST_UNKNOWN")),
                float(data.get("amount", 0.0)),
                str(data.get("transaction_type", "PAYMENT")),
                str(data.get("merchant_category", "retail")),
                str(data.get("payment_method", "credit_card")),
                str(data.get("location", "Unknown")),
                str(data.get("device_id", "DEV_UNKNOWN")),
                int(data.get("card_present", 0)),
                float(data.get("old_balance", 0.0)),
                float(data.get("new_balance", 0.0)),
                int(data.get("is_fraud", 0)),
                float(data.get("supervised_prob", 0.0)),
                float(data.get("anomaly_score", 0.0)),
                float(data.get("risk_score", 0.0)),
                str(data.get("risk_band", "LOW")),
                str(data.get("recommended_action", "APPROVE")),
                exp_json,
                now_str,
            ))

            # Auto-generate alert if high risk
            risk_score = float(data.get("risk_score", 0.0))
            if risk_score > 70.0 or data.get("risk_band") == "HIGH":
                severity = "CRITICAL" if risk_score >= 85.0 else "WARNING"
                conn.execute("""
                INSERT OR IGNORE INTO alerts (transaction_id, created_at, risk_score, severity, status, notes)
                VALUES (?, ?, ?, ?, 'NEW', 'Auto-triggered by real-time risk engine')
                """, (tx_id, now_str, risk_score, severity))

        return tx_id

    def get_transaction(self, tx_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves a transaction and decodes its explanation JSON."""
        with self._get_connection() as conn:
            row = conn.execute("SELECT * FROM transactions WHERE transaction_id = ?", (tx_id,)).fetchone()
            if not row:
                return None
            res = dict(row)
            if res.get("explanation_json"):
                try:
                    res["explanation"] = json.loads(res["explanation_json"])
                except Exception:
                    res["explanation"] = None
            return res

    def list_transactions(
        self,
        limit: int = 50,
        offset: int = 0,
        risk_band: Optional[str] = None,
        search: Optional[str] = None,
    ) -> Tuple[List[Dict[str, Any]], int]:
        """Lists transactions with filtering and pagination."""
        query = "SELECT * FROM transactions WHERE 1=1"
        params = []

        if risk_band and risk_band.upper() != "ALL":
            query += " AND risk_band = ?"
            params.append(risk_band.upper())

        if search:
            query += " AND (transaction_id LIKE ? OR customer_id LIKE ? OR location LIKE ?)"
            term = f"%{search}%"
            params.extend([term, term, term])

        with self._get_connection() as conn:
            count_query = f"SELECT COUNT(*) FROM ({query})"
            total = conn.execute(count_query, params).fetchone()[0]

            query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])

            rows = conn.execute(query, params).fetchall()
            results = []
            for r in rows:
                item = dict(r)
                if item.get("explanation_json"):
                    try:
                        item["explanation"] = json.loads(item["explanation_json"])
                    except Exception:
                        item["explanation"] = None
                results.append(item)

        return results, total

    def list_alerts(self, status: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        """Lists high-risk alerts with associated transaction details."""
        query = """
        SELECT a.alert_id, a.transaction_id, a.created_at, a.risk_score, a.severity, a.status, a.notes,
               t.customer_id, t.amount, t.transaction_type, t.payment_method, t.location, t.explanation_json
        FROM alerts a
        JOIN transactions t ON a.transaction_id = t.transaction_id
        """
        params = []
        if status and status.upper() != "ALL":
            query += " WHERE a.status = ?"
            params.append(status.upper())

        query += " ORDER BY a.risk_score DESC, a.created_at DESC LIMIT ?"
        params.append(limit)

        with self._get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
            results = []
            for r in rows:
                item = dict(r)
                if item.get("explanation_json"):
                    try:
                        item["explanation"] = json.loads(item["explanation_json"])
                    except Exception:
                        item["explanation"] = None
                results.append(item)
        return results

    def update_alert_status(self, alert_id: int, status: str, notes: Optional[str] = None) -> bool:
        """Updates alert triage status (e.g. RESOLVED, CONFIRMED_FRAUD)."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "UPDATE alerts SET status = ?, notes = COALESCE(?, notes) WHERE alert_id = ?",
                (status.upper(), notes, alert_id)
            )
            return cursor.rowcount > 0

    def get_analytics_summary(self) -> Dict[str, Any]:
        """Calculates dashboard KPI metrics."""
        with self._get_connection() as conn:
            total_tx = conn.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]
            total_fraud = conn.execute("SELECT COUNT(*) FROM transactions WHERE is_fraud = 1").fetchone()[0]
            high_risk = conn.execute("SELECT COUNT(*) FROM transactions WHERE risk_band = 'HIGH'").fetchone()[0]
            medium_risk = conn.execute("SELECT COUNT(*) FROM transactions WHERE risk_band = 'MEDIUM'").fetchone()[0]
            low_risk = conn.execute("SELECT COUNT(*) FROM transactions WHERE risk_band = 'LOW'").fetchone()[0]
            active_alerts = conn.execute("SELECT COUNT(*) FROM alerts WHERE status IN ('NEW', 'UNDER_REVIEW')").fetchone()[0]
            total_loss_saved = conn.execute("SELECT SUM(amount) FROM transactions WHERE risk_band = 'HIGH'").fetchone()[0] or 0.0

            # Breakdown by transaction type
            type_rows = conn.execute("""
            SELECT transaction_type, COUNT(*) as count, SUM(CASE WHEN is_fraud = 1 THEN 1 ELSE 0 END) as fraud_count
            FROM transactions GROUP BY transaction_type
            """).fetchall()

            # Breakdown by risk band
            band_distribution = {
                "LOW": low_risk,
                "MEDIUM": medium_risk,
                "HIGH": high_risk,
            }

        return {
            "total_transactions": total_tx,
            "confirmed_fraud_count": total_fraud,
            "high_risk_blocked_count": high_risk,
            "suspicious_review_count": medium_risk,
            "low_risk_approved_count": low_risk,
            "active_alerts_count": active_alerts,
            "fraud_rate_pct": round((total_fraud / max(1, total_tx)) * 100, 2),
            "loss_prevented_usd": round(float(total_loss_saved), 2),
            "band_distribution": band_distribution,
            "transaction_type_breakdown": [dict(r) for r in type_rows],
        }

    def seed_from_dataframe(self, df: pd.DataFrame, max_records: int = 2000):
        """Seeds initial realistic records into database from test scored data."""
        with self._get_connection() as conn:
            existing = conn.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]
            if existing >= 100:
                print(f"[DatabaseManager] Database already contains {existing:,} records. Skipping seed.")
                return

        print(f"[DatabaseManager] Seeding {min(len(df), max_records):,} transactions into SQLite...")
        sub_df = df.head(max_records)
        for _, row in sub_df.iterrows():
            row_dict = row.to_dict()
            self.insert_transaction(row_dict)
        print("[DatabaseManager] Database seeding complete.")

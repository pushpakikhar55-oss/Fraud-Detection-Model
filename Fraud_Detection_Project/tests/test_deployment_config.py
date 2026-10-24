"""Tests for deployment-safe runtime configuration."""
from src.api.database import DatabaseManager


def test_database_path_reads_environment(monkeypatch, tmp_path):
    """A deployment can redirect SQLite to its persistent mounted volume."""
    db_path = tmp_path / "runtime" / "fraud.db"
    monkeypatch.setenv("FRAUD_DB_PATH", str(db_path))

    db = DatabaseManager()

    assert db.db_path == db_path
    assert db_path.exists()

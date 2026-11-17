
import sqlite3
from pathlib import Path
from datetime import datetime, timezone


# Store the database in the project root
BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "fraud_detection.db"


def get_connection():
    """Create a connection to the SQLite database."""
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def create_table():
    """Create the transaction predictions table."""
    with get_connection() as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS transactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                step INTEGER NOT NULL,
                transaction_type TEXT NOT NULL,
                amount REAL NOT NULL,
                is_flagged_fraud INTEGER NOT NULL,
                fraud_probability REAL NOT NULL,
                prediction INTEGER NOT NULL,
                decision_threshold REAL NOT NULL,
                created_at TEXT NOT NULL
            )
        """)

    print(f"Database ready: {DB_PATH}")


def save_prediction(transaction, result):
    """Save a transaction and its model prediction."""
    with get_connection() as connection:
        cursor = connection.execute("""
            INSERT INTO transactions (
                step,
                transaction_type,
                amount,
                is_flagged_fraud,
                fraud_probability,
                prediction,
                decision_threshold,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            int(transaction["step"]),
            transaction["type"],
            float(transaction["amount"]),
            int(transaction["isFlaggedFraud"]),
            float(result["fraud_probability"]),
            int(result["prediction"]),
            float(result["threshold"]),
            datetime.now(timezone.utc).isoformat()
        ))

        record_id = cursor.lastrowid

    return record_id


def get_recent_predictions(limit=10):
    """Return the most recent saved predictions."""
    if not isinstance(limit, int) or limit < 1:
        raise ValueError("limit must be a positive integer")

    with get_connection() as connection:
        rows = connection.execute("""
            SELECT *
            FROM transactions
            ORDER BY id DESC
            LIMIT ?
        """, (limit,)).fetchall()

    return [dict(row) for row in rows]


if __name__ == "__main__":
    create_table()
    print("SQLite database initialized successfully.")

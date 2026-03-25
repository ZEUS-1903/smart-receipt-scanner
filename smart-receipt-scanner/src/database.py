"""
database.py — SQLite data layer for the expense tracker.

Stores parsed receipts, line items, categories, and budget info.
No external database needed — SQLite is built into Python.
"""

import sqlite3
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional


DB_PATH = Path(__file__).parent.parent / "data" / "expenses.db"


def get_connection() -> sqlite3.Connection:
    """Get a database connection, creating tables if needed."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row  # Return rows as dicts
    conn.execute("PRAGMA journal_mode=WAL")  # Better concurrent access
    _create_tables(conn)
    return conn


def _create_tables(conn: sqlite3.Connection):
    """Create tables if they don't exist."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS receipts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            store_name TEXT NOT NULL,
            date TEXT NOT NULL,
            address TEXT DEFAULT '',
            phone TEXT DEFAULT '',
            subtotal REAL DEFAULT 0,
            tax REAL DEFAULT 0,
            total REAL NOT NULL,
            tips REAL DEFAULT 0,
            discount REAL DEFAULT 0,
            category TEXT DEFAULT 'uncategorized',
            image_path TEXT,
            raw_output TEXT,
            is_anomaly INTEGER DEFAULT 0,
            created_at TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS line_items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            receipt_id INTEGER NOT NULL,
            name TEXT NOT NULL,
            quantity INTEGER DEFAULT 1,
            price REAL NOT NULL,
            FOREIGN KEY (receipt_id) REFERENCES receipts(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS budgets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            category TEXT UNIQUE NOT NULL,
            monthly_limit REAL NOT NULL,
            created_at TEXT DEFAULT (datetime('now'))
        );

        CREATE INDEX IF NOT EXISTS idx_receipts_date ON receipts(date);
        CREATE INDEX IF NOT EXISTS idx_receipts_category ON receipts(category);
    """)
    conn.commit()


# --- Receipt CRUD ---

def save_receipt(parsed_receipt: dict, category: str = "uncategorized",
                 is_anomaly: bool = False) -> int:
    """
    Save a parsed receipt and its line items to the database.

    Args:
        parsed_receipt: Output from ReceiptParser.parse_receipt()
        category: Spending category
        is_anomaly: Whether this was flagged as anomalous spending

    Returns:
        The receipt ID
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO receipts (store_name, date, address, phone, subtotal, tax,
                            total, tips, discount, category, image_path,
                            raw_output, is_anomaly)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        parsed_receipt.get("store_name", "Unknown"),
        parsed_receipt.get("date", datetime.now().strftime("%Y-%m-%d")),
        parsed_receipt.get("address", ""),
        parsed_receipt.get("phone", ""),
        parsed_receipt.get("subtotal", 0),
        parsed_receipt.get("tax", 0),
        parsed_receipt.get("total", 0),
        parsed_receipt.get("tips", 0),
        parsed_receipt.get("discount", 0),
        category,
        parsed_receipt.get("image_path", ""),
        parsed_receipt.get("raw_output", ""),
        int(is_anomaly),
    ))

    receipt_id = cursor.lastrowid

    # Save line items
    for item in parsed_receipt.get("items", []):
        cursor.execute("""
            INSERT INTO line_items (receipt_id, name, quantity, price)
            VALUES (?, ?, ?, ?)
        """, (
            receipt_id,
            item.get("name", "Unknown"),
            item.get("quantity", 1),
            item.get("price", 0),
        ))

    conn.commit()
    conn.close()
    return receipt_id


def get_all_receipts(limit: int = 100, offset: int = 0) -> list[dict]:
    """Get all receipts ordered by date descending."""
    conn = get_connection()
    rows = conn.execute("""
        SELECT id, store_name, date, total, category, is_anomaly, created_at
        FROM receipts
        ORDER BY date DESC
        LIMIT ? OFFSET ?
    """, (limit, offset)).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_receipt_with_items(receipt_id: int) -> Optional[dict]:
    """Get a single receipt with all its line items."""
    conn = get_connection()
    receipt = conn.execute(
        "SELECT * FROM receipts WHERE id = ?", (receipt_id,)
    ).fetchone()

    if not receipt:
        conn.close()
        return None

    items = conn.execute(
        "SELECT * FROM line_items WHERE receipt_id = ?", (receipt_id,)
    ).fetchall()

    conn.close()

    result = dict(receipt)
    result["items"] = [dict(item) for item in items]
    return result


# --- Analytics Queries ---

def get_spending_by_category(days: int = 30) -> list[dict]:
    """Get total spending per category for the last N days."""
    conn = get_connection()
    cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    rows = conn.execute("""
        SELECT category, SUM(total) as total_spent, COUNT(*) as receipt_count
        FROM receipts
        WHERE date >= ?
        GROUP BY category
        ORDER BY total_spent DESC
    """, (cutoff,)).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_daily_spending(days: int = 30) -> list[dict]:
    """Get daily spending totals for the last N days."""
    conn = get_connection()
    cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    rows = conn.execute("""
        SELECT date, SUM(total) as daily_total, COUNT(*) as receipt_count
        FROM receipts
        WHERE date >= ?
        GROUP BY date
        ORDER BY date ASC
    """, (cutoff,)).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_monthly_spending(months: int = 6) -> list[dict]:
    """Get monthly spending totals."""
    conn = get_connection()
    cutoff = (datetime.now() - timedelta(days=months * 30)).strftime("%Y-%m-%d")
    rows = conn.execute("""
        SELECT strftime('%Y-%m', date) as month,
               SUM(total) as monthly_total,
               COUNT(*) as receipt_count
        FROM receipts
        WHERE date >= ?
        GROUP BY month
        ORDER BY month ASC
    """, (cutoff,)).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_spending_totals() -> dict:
    """Get overall spending summary."""
    conn = get_connection()

    today = datetime.now().strftime("%Y-%m-%d")
    week_ago = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
    month_ago = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

    total_all = conn.execute("SELECT COALESCE(SUM(total), 0) FROM receipts").fetchone()[0]
    total_month = conn.execute(
        "SELECT COALESCE(SUM(total), 0) FROM receipts WHERE date >= ?", (month_ago,)
    ).fetchone()[0]
    total_week = conn.execute(
        "SELECT COALESCE(SUM(total), 0) FROM receipts WHERE date >= ?", (week_ago,)
    ).fetchone()[0]
    total_today = conn.execute(
        "SELECT COALESCE(SUM(total), 0) FROM receipts WHERE date = ?", (today,)
    ).fetchone()[0]
    receipt_count = conn.execute("SELECT COUNT(*) FROM receipts").fetchone()[0]
    anomaly_count = conn.execute(
        "SELECT COUNT(*) FROM receipts WHERE is_anomaly = 1"
    ).fetchone()[0]

    conn.close()
    return {
        "total_all_time": total_all,
        "total_this_month": total_month,
        "total_this_week": total_week,
        "total_today": total_today,
        "receipt_count": receipt_count,
        "anomaly_count": anomaly_count,
    }


def get_all_totals_for_anomaly() -> list[float]:
    """Get all receipt totals for anomaly detection training."""
    conn = get_connection()
    rows = conn.execute("SELECT total FROM receipts ORDER BY date").fetchall()
    conn.close()
    return [row[0] for row in rows]


# --- Budget Management ---

def set_budget(category: str, monthly_limit: float):
    """Set or update a monthly budget for a category."""
    conn = get_connection()
    conn.execute("""
        INSERT INTO budgets (category, monthly_limit)
        VALUES (?, ?)
        ON CONFLICT(category) DO UPDATE SET monthly_limit = ?
    """, (category, monthly_limit, monthly_limit))
    conn.commit()
    conn.close()


def get_budgets() -> list[dict]:
    """Get all budget configurations with current spending."""
    conn = get_connection()
    month_start = datetime.now().strftime("%Y-%m-01")

    rows = conn.execute("""
        SELECT b.category, b.monthly_limit,
               COALESCE(SUM(r.total), 0) as spent
        FROM budgets b
        LEFT JOIN receipts r ON r.category = b.category AND r.date >= ?
        GROUP BY b.category
    """, (month_start,)).fetchall()
    conn.close()
    return [dict(row) for row in rows]


# --- Seed Demo Data ---

def seed_demo_data():
    """Insert demo data for testing the dashboard."""
    import random

    categories = ["groceries", "dining", "transport", "healthcare",
                  "entertainment", "shopping", "utilities"]
    stores = {
        "groceries": ["Whole Foods", "Trader Joe's", "Stop & Shop", "Market Basket"],
        "dining": ["Chipotle", "Starbucks", "Panera Bread", "Local Pizzeria"],
        "transport": ["Shell Gas", "Uber", "MBTA", "Lyft"],
        "healthcare": ["CVS Pharmacy", "Walgreens", "Dr. Smith Office"],
        "entertainment": ["AMC Theaters", "Netflix", "Spotify", "Barnes & Noble"],
        "shopping": ["Amazon", "Target", "Best Buy", "Home Depot"],
        "utilities": ["Eversource", "National Grid", "Comcast"],
    }
    price_ranges = {
        "groceries": (20, 150),
        "dining": (8, 60),
        "transport": (10, 70),
        "healthcare": (10, 200),
        "entertainment": (10, 50),
        "shopping": (15, 300),
        "utilities": (50, 200),
    }

    conn = get_connection()

    for days_ago in range(90):
        # 0-3 receipts per day
        num_receipts = random.choices([0, 1, 2, 3], weights=[0.2, 0.4, 0.3, 0.1])[0]
        date = (datetime.now() - timedelta(days=days_ago)).strftime("%Y-%m-%d")

        for _ in range(num_receipts):
            cat = random.choice(categories)
            store = random.choice(stores[cat])
            low, high = price_ranges[cat]
            total = round(random.uniform(low, high), 2)
            tax = round(total * 0.0625, 2)  # MA sales tax
            subtotal = round(total - tax, 2)

            # Occasionally insert an anomaly (unusually high spend)
            is_anomaly = 0
            if random.random() < 0.03:
                total = round(total * random.uniform(3, 6), 2)
                is_anomaly = 1

            conn.execute("""
                INSERT INTO receipts (store_name, date, subtotal, tax, total,
                                    category, is_anomaly)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (store, date, subtotal, tax, total, cat, is_anomaly))

    # Set some default budgets
    for cat, limit in [("groceries", 600), ("dining", 300), ("transport", 200),
                       ("entertainment", 100), ("shopping", 400)]:
        conn.execute("""
            INSERT OR REPLACE INTO budgets (category, monthly_limit)
            VALUES (?, ?)
        """, (cat, limit))

    conn.commit()
    conn.close()
    print("[Database] Demo data seeded successfully!")


if __name__ == "__main__":
    seed_demo_data()
    totals = get_spending_totals()
    print(f"\nSpending Summary:")
    for k, v in totals.items():
        print(f"  {k}: {v}")

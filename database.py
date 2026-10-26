import sqlite3
from pathlib import Path
import pandas as pd

DB_PATH = Path(__file__).parent / "warehouse.db"

def get_connection():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS products (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sku TEXT NOT NULL UNIQUE,
                name TEXT NOT NULL,
                category TEXT NOT NULL DEFAULT 'General',
                quantity INTEGER NOT NULL DEFAULT 0 CHECK(quantity >= 0),
                reorder_level INTEGER NOT NULL DEFAULT 10 CHECK(reorder_level >= 0),
                unit_price REAL NOT NULL DEFAULT 0 CHECK(unit_price >= 0),
                location TEXT NOT NULL DEFAULT ''
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS transactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sku TEXT NOT NULL,
                product_name TEXT NOT NULL,
                movement TEXT NOT NULL CHECK(movement IN ('IN', 'OUT')),
                quantity INTEGER NOT NULL CHECK(quantity > 0),
                note TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
            )
        """)

def get_products():
    with get_connection() as conn:
        return pd.read_sql_query("SELECT * FROM products ORDER BY name", conn)

def get_transactions():
    with get_connection() as conn:
        return pd.read_sql_query("SELECT * FROM transactions ORDER BY id DESC", conn)

def add_product(sku, name, category, quantity, reorder_level, unit_price, location):
    with get_connection() as conn:
        conn.execute(
            """INSERT INTO products (sku, name, category, quantity, reorder_level, unit_price, location)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (sku.strip(), name.strip(), category.strip() or "General",
             int(quantity), int(reorder_level), float(unit_price), location.strip())
        )
        if int(quantity) > 0:
            conn.execute(
                "INSERT INTO transactions (sku, product_name, movement, quantity, note) VALUES (?, ?, 'IN', ?, ?)",
                (sku.strip(), name.strip(), int(quantity), "Opening stock")
            )

def record_stock(sku, movement, quantity, note=""):
    quantity = int(quantity)
    if quantity <= 0:
        raise ValueError("Quantity must be greater than zero.")
    with get_connection() as conn:
        product = conn.execute("SELECT name, quantity FROM products WHERE sku = ?", (sku,)).fetchone()
        if product is None:
            raise ValueError("Product not found.")
        current = int(product["quantity"])
        if movement == "OUT" and quantity > current:
            raise ValueError(f"Not enough stock. Available quantity: {current}.")
        new_quantity = current + quantity if movement == "IN" else current - quantity
        conn.execute("UPDATE products SET quantity = ? WHERE sku = ?", (new_quantity, sku))
        conn.execute(
            "INSERT INTO transactions (sku, product_name, movement, quantity, note) VALUES (?, ?, ?, ?, ?)",
            (sku, product["name"], movement, quantity, note.strip())
        )

def delete_product(sku):
    with get_connection() as conn:
        conn.execute("DELETE FROM products WHERE sku = ?", (sku,))

# trade_db.py
from __future__ import annotations

import sqlite3
from typing import Iterable, Optional
import pandas as pd


def connect(db_path: str) -> sqlite3.Connection:
    return sqlite3.connect(db_path, check_same_thread=False)


def _ensure_column(conn: sqlite3.Connection, table: str, col_def: str) -> None:
    """
    Backward-compatible schema migration: add a column if missing.
    col_def example: "qty REAL DEFAULT 1.0"
    """
    try:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {col_def}")
    except sqlite3.OperationalError:
        # already exists
        pass


def init_db(db_path: str) -> None:
    with connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                strategy TEXT NOT NULL,
                side TEXT NOT NULL,

                open_ts TEXT,
                close_ts TEXT,
                open_date TEXT,
                close_date TEXT,

                entry_price REAL NOT NULL,
                exit_price REAL NOT NULL,

                qty REAL DEFAULT 1.0,
                notional_usd REAL DEFAULT 0.0,

                notional_ccy TEXT DEFAULT 'USD',
                notional_ccy_amount REAL DEFAULT 0.0,
                fx_to_usd REAL DEFAULT 1.0,

                stop REAL,
                tp REAL,
                tp1 REAL,

                tp1_hit INTEGER DEFAULT 0,
                be_armed INTEGER DEFAULT 0,

                pnl REAL NOT NULL,
                note TEXT
            )
            """
        )

        # Ensure columns exist for older DBs
        _ensure_column(conn, "trades", "qty REAL DEFAULT 1.0")
        _ensure_column(conn, "trades", "notional_usd REAL DEFAULT 0.0")
        _ensure_column(conn, "trades", "notional_ccy TEXT DEFAULT 'USD'")
        _ensure_column(conn, "trades", "notional_ccy_amount REAL DEFAULT 0.0")
        _ensure_column(conn, "trades", "fx_to_usd REAL DEFAULT 1.0")

        conn.commit()


def insert_trade(db_path: str, row: dict) -> int:
    cols = [
        "symbol", "strategy", "side",
        "open_ts", "close_ts", "open_date", "close_date",
        "entry_price", "exit_price",
        "qty", "notional_usd",
        "notional_ccy", "notional_ccy_amount", "fx_to_usd",
        "stop", "tp", "tp1",
        "tp1_hit", "be_armed",
        "pnl", "note"
    ]
    vals = [row.get(c) for c in cols]

    with connect(db_path) as conn:
        cur = conn.cursor()
        cur.execute(
            f"INSERT INTO trades ({','.join(cols)}) VALUES ({','.join(['?']*len(cols))})",
            vals,
        )
        conn.commit()
        return int(cur.lastrowid)


def load_trades(db_path: str, symbol: Optional[str] = None) -> pd.DataFrame:
    with connect(db_path) as conn:
        if symbol and symbol != "ALL":
            df = pd.read_sql_query(
                "SELECT * FROM trades WHERE symbol=? ORDER BY close_ts",
                conn,
                params=[symbol],
            )
        else:
            df = pd.read_sql_query("SELECT * FROM trades ORDER BY close_ts", conn)

    if df.empty:
        return df

    numeric_cols = [
        "entry_price", "exit_price", "stop", "tp", "tp1",
        "qty", "notional_usd", "notional_ccy_amount", "fx_to_usd",
        "pnl"
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def update_trades(db_path: str, edited_df: pd.DataFrame) -> None:
    """
    Save edits from Streamlit data_editor back to SQLite.
    """
    if edited_df is None or edited_df.empty:
        return

    editable = [
        "exit_price", "stop", "tp", "tp1", "tp1_hit", "be_armed",
        "qty", "notional_usd", "notional_ccy", "notional_ccy_amount", "fx_to_usd",
        "pnl", "note", "close_ts", "close_date"
    ]

    with connect(db_path) as conn:
        cur = conn.cursor()
        for _, r in edited_df.iterrows():
            if "id" not in r or pd.isna(r["id"]):
                continue
            _id = int(r["id"])

            sets = []
            vals = []
            for c in editable:
                if c in edited_df.columns:
                    sets.append(f"{c}=?")
                    vals.append(r.get(c))

            if not sets:
                continue

            vals.append(_id)
            cur.execute(f"UPDATE trades SET {', '.join(sets)} WHERE id=?", vals)

        conn.commit()


def delete_trades(db_path: str, ids: Iterable[int]) -> None:
    ids = list(ids)
    if not ids:
        return
    with connect(db_path) as conn:
        conn.executemany("DELETE FROM trades WHERE id=?", [(int(i),) for i in ids])
        conn.commit()


def clear_trades(db_path: str) -> None:
    with connect(db_path) as conn:
        conn.execute("DELETE FROM trades")
        conn.commit()

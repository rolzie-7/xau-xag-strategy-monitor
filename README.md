# Metals Strategy Monitor (XAUUSD / XAGUSD)

A Python project for **research + backtesting** and a **Streamlit-based live monitoring dashboard** for XAUUSD (Gold) and XAGUSD (Silver).

This repo is split into two parts:

- **`backtest/`**: strategy experiments, parameter sweeps, and research notebooks/scripts.
- **`trade_alert/`**: a lightweight Streamlit dashboard that generates **manual trade tickets** (SL/TP) and tracks trades using a persistent **SQLite** database (`trade_db.py`).

> **Disclaimer**: This project is for education and research only. It is **not** financial advice.

<img width="1903" height="992" alt="image" src="https://github.com/user-attachments/assets/bba8cc4f-77c5-4cfe-b875-f5ecf2808958" />

---

## Key Features

### Research & Backtesting
- Strategy prototypes for XAUUSD / XAGUSD
- Parameter sweeps (RR settings, entry-time studies, etc.)
- Comparison experiments across different exits (Fixed RR vs ATR vs hybrids)

### Trade Alert Dashboard (Streamlit)
- Generate **manual order tickets** (entry / stop / TP / scale-out targets)
- Support **ATR trailing** stop updates (live stop level recalculation)
- Input your **actual fill price** (so tracking reflects what you really got)
- Persist trades & PnL into **SQLite** (data survives restarts)
- Track closed-trade PnL and build an equity curve from recorded history

---

## Repository Layout

```text
.
├── backtest/                 # Research + backtesting experiments
└── trade_alert/              # Streamlit app + trade tracking
    ├── app.py                # Streamlit dashboard
    ├── trade_db.py           # SQLite persistence layer (schema + CRUD)
    └── ...                   # helpers / advisors / fetchers (if any)

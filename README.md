# xau-xag-strategy-monitor
# Metals Strategy Monitor (XAUUSD / XAGUSD)

A Python project for **research + backtesting** and a **Streamlit-based live monitoring dashboard** for XAUUSD (Gold) and XAGUSD (Silver).

This repository contains:
- `backtest/`: strategy experiments and parameter sweeps (entry-time studies, RR settings, etc.)
- `trade_alert/`: a lightweight web dashboard that provides **manual trade tickets** (SL/TP) and **live stop updates** for ATR-based trailing strategies.

> Note: This project is designed for education and research. It is **not** financial advice.

---

## Strategies

### 1) Dice + Fixed RR (Pure Fixed Ratio)
- Direction (LONG/SHORT) is decided randomly (dice / probability).
- Entry uses a rolling **24h reference mid**.
- Stop uses rolling **24h high/low**.
- Take-profit uses a fixed risk-reward ratio (e.g., 1:2, 1:3).

### 2) Momentum Strategy (Trade only when momentum exists)
- Trade only if momentum strength is above a threshold.
- Uses ATR for risk sizing and trailing logic.

### 3) 50/50 Scale-out (Best for “fixed RR + ATR trailing”)
- Close **50%** at a fixed R-multiple target (e.g., +2R).
- Keep **50%** running with ATR trailing stop until stop-out or forced close.
- Based on our experiments, the **Dice + Fixed RR** and the **50/50 Scale-out** performed better.

A detailed comparison blog post will be published later and linked here.

---

## Quick Start

### 1) Create a virtual environment (recommended)
```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1

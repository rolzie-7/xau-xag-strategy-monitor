# app.py
from __future__ import annotations
import logging
logging.getLogger().setLevel(logging.WARNING)
logging.getLogger("DUKASCRIPT").setLevel(logging.WARNING)

from dataclasses import dataclass
from datetime import datetime
from uuid import uuid4

import pandas as pd
import streamlit as st
import dukascopy_python.instruments as inst

from data_fetcher import DukascopyM5Fetcher
from advisor import FixedRRAdvisor, ScaleOutATRAdvisor, PositionState

from trade_db import init_db, insert_trade, load_trades, update_trades, delete_trades, clear_trades


TZ = "Europe/London"
DB_PATH = "trades.sqlite"

SYMBOLS = {
    "XAUUSD": inst.INSTRUMENT_FX_METALS_XAU_USD,
    "XAGUSD": inst.INSTRUMENT_FX_METALS_XAG_USD,
}

st.set_page_config(page_title="Metals Strategy Monitor", layout="wide")


@dataclass
class TrackedPosition:
    track_id: str
    pos: PositionState
    qty: float
    notional_usd: float
    notional_ccy: str
    notional_ccy_amount: float
    fx_to_usd: float


@st.cache_resource
def get_fetchers():
    fetchers = {
        sym: DukascopyM5Fetcher(sym, instr, tz=TZ, lookback_days=5, poll_seconds=30)
        for sym, instr in SYMBOLS.items()
    }
    for f in fetchers.values():
        f.start()
    return fetchers


def fmt_price(x: float) -> str:
    try:
        return f"{float(x):.5f}"
    except Exception:
        return "NA"


@st.cache_data(ttl=3600)
def get_fx_to_usd(from_ccy: str) -> float:
    """
    Fetch FX rate to USD using a free public API.
    Note: ECB-style rates update daily (good enough for position sizing).
    """
    if from_ccy.upper() == "USD":
        return 1.0

    import requests

    from_ccy = from_ccy.upper()
    urls = [
        f"https://api.frankfurter.dev/latest?from={from_ccy}&to=USD",
        f"https://api.frankfurter.app/latest?from={from_ccy}&to=USD",
    ]
    last_err = None
    for url in urls:
        try:
            r = requests.get(url, timeout=6)
            if r.ok:
                js = r.json()
                rate = js.get("rates", {}).get("USD", None)
                if rate is not None:
                    return float(rate)
        except Exception as e:
            last_err = e
            continue

    raise RuntimeError(f"FX fetch failed for {from_ccy}->USD: {last_err}")


def calc_manual_close_pnl(tp: TrackedPosition, exit_price: float, scale_out_fraction: float = 0.5) -> float:
    pos = tp.pos
    entry = float(pos.entry_price)
    exit_price = float(exit_price)
    qty = float(tp.qty)

    def pnl_for(q: float, px_exit: float) -> float:
        return q * ((px_exit - entry) if pos.side == "LONG" else (entry - px_exit))

    if pos.strategy == "FixedRR":
        return float(pnl_for(qty, exit_price))

    tp1 = getattr(pos, "tp1", None)
    tp1_hit = bool(getattr(pos, "tp1_hit", False))
    if tp1_hit and tp1 is not None:
        tp1 = float(tp1)
        q1 = qty * float(scale_out_fraction)
        q2 = qty - q1
        pnl1 = pnl_for(q1, tp1)
        pnl2 = pnl_for(q2, exit_price)
        return float(pnl1 + pnl2)

    return float(pnl_for(qty, exit_price))


def main():
    init_db(DB_PATH)

    st.title("XAU/XAG Strategy Monitor (manual execution + live SL/TP advice)")

    fetchers = get_fetchers()

    if "tracked_positions" not in st.session_state:
        st.session_state["tracked_positions"] = []

    st.sidebar.header("Settings")
    initial_capital = st.sidebar.number_input("Equity start ($)", value=100.0, min_value=0.0, step=10.0)
    auto_refresh = st.sidebar.checkbox("Auto refresh", value=False)
    refresh_seconds = st.sidebar.number_input("Refresh seconds", value=10, min_value=3, max_value=120, step=1)

    fixed_rr = FixedRRAdvisor(rr=st.sidebar.number_input("FixedRR RR", value=3.0, min_value=0.5, step=0.5))
    scale_out = ScaleOutATRAdvisor(
        mom_atr_threshold=st.sidebar.number_input("ScaleOut MOM/ATR threshold", value=0.5, min_value=0.0, step=0.1),
        atr_period=int(st.sidebar.number_input("ATR period", value=14, min_value=2, step=1)),
    )

    sym = st.sidebar.selectbox("Symbol", ["XAUUSD", "XAGUSD"])
    snap = fetchers[sym].snapshot()
    df = snap.m5

    colA, colB = st.columns([1, 1])

    with colA:
        st.subheader("Live data")
        if df.empty:
            st.warning("No data yet. Wait a bit...")
        else:
            last_ts = df.index.max()
            last_row = df.iloc[-1]
            now_ldn = pd.Timestamp.now(tz=TZ)
            lag = now_ldn - last_ts

            st.write(f"Now (London): **{now_ldn}**")
            st.write(f"Last bar (London): **{last_ts}**  |  Lag: **{lag}**")
            st.write(
                f"OHLC: O={fmt_price(last_row['open'])} "
                f"H={fmt_price(last_row['high'])} "
                f"L={fmt_price(last_row['low'])} "
                f"C={fmt_price(last_row['close'])}"
            )
            st.dataframe(df.tail(50), use_container_width=True)

    with colB:
        st.subheader("Generate order ticket")
        strategy_name = st.selectbox(
            "Strategy",
            ["FixedRR (24h ref)", "ScaleOutATR (mom/atr + 50% TP1 + trailing)"],
        )

        if df.empty:
            st.info("Waiting for data...")
        else:
            if strategy_name.startswith("FixedRR"):
                side = st.selectbox("Side", ["LONG", "SHORT"])
                try:
                    ticket = fixed_rr.make_ticket(sym, df, side)
                    st.success(f"{ticket.strategy} | {ticket.side} | {ticket.order_type} @ {fmt_price(ticket.order_price)}")
                    st.write(f"Stop: **{fmt_price(ticket.stop)}**")
                    st.write(f"Take Profit: **{fmt_price(ticket.take_profit)}**")
                    st.caption(ticket.note)
                except Exception as e:
                    st.error(str(e))
            else:
                try:
                    ticket = scale_out.make_ticket(sym, df)
                    if ticket.side == "NONE":
                        st.warning(ticket.note)
                    else:
                        st.success(f"{ticket.strategy} | {ticket.side} | {ticket.order_type} @ {fmt_price(ticket.order_price)}")
                        st.write(f"Stop (init): **{fmt_price(ticket.stop)}**")
                        st.write(f"TP1 (close 50%): **{fmt_price(ticket.tp1)}**")
                        st.caption(ticket.note)
                except Exception as e:
                    st.error(str(e))

        st.divider()
        st.subheader("I am filled (manual) → start tracking")

        fill_side = st.selectbox("Filled side", ["LONG", "SHORT"], key="fill_side")
        fill_price = st.number_input("Fill price", value=0.0, step=0.01, format="%.5f")
        size_mode = st.radio(
            "Position sizing input",
            ["By notional", "By quantity"],
            horizontal=True,
        )

        fx_manual = False
        fx_to_usd = 1.0
        fx_error = None

        # Defaults so variables always exist
        notional_ccy = "USD"
        notional_ccy_amount = 0.0
        notional_usd = 0.0
        qty = 0.0

        if size_mode == "By notional":
            # ✅ Existing path (with FX)
            notional_ccy = st.selectbox("Notional currency", ["USD", "GBP"], index=0)
            notional_ccy_amount = st.number_input(
                f"Bought notional ({notional_ccy})",
                value=100.0,
                min_value=0.0,
                step=10.0,
            )

            if notional_ccy != "USD":
                fx_manual = st.checkbox("Manual FX override (if API fails)", value=False)
                if fx_manual:
                    fx_to_usd = st.number_input(
                        f"FX rate (1 {notional_ccy} = ? USD)",
                        value=1.25,
                        min_value=0.0001,
                        step=0.01,
                    )
                else:
                    try:
                        fx_to_usd = float(get_fx_to_usd(notional_ccy))
                    except Exception as e:
                        fx_error = str(e)
                        fx_to_usd = 0.0

            if fx_error:
                st.warning(f"FX auto fetch failed: {fx_error}")
                st.info("Tick 'Manual FX override' and input a rate, or switch currency to USD.")

            notional_usd = float(notional_ccy_amount) * float(fx_to_usd) if notional_ccy_amount and fx_to_usd else 0.0
            qty = (float(notional_usd) / float(fill_price)) if fill_price and fill_price > 0 and notional_usd > 0 else 0.0

        else:
            # ✅ New path: user enters exact broker quantity
            qty = st.number_input(
                "Quantity bought",
                value=0.01,
                min_value=0.0,
                step=0.01,
                format="%.6f",
            )

            notional_usd = float(qty) * float(fill_price) if fill_price and fill_price > 0 and qty > 0 else 0.0
            notional_ccy = "USD"
            notional_ccy_amount = float(notional_usd)
            fx_to_usd = 1.0

    # ✅ Updated caption (works for both modes)
    st.caption(
        f"Mode: {size_mode} | "
        f"FX used: 1 {notional_ccy} = {fx_to_usd:.6f} USD  |  "
        f"notional_usd = ${notional_usd:.2f}  |  qty ≈ {qty:.6f}"
    )

    start_btn = st.button("Start tracking this position")

    if start_btn:
        if df.empty:
            st.error("Need live data first.")
        elif fill_price <= 0:
            st.error("Need a valid fill price.")
        elif qty <= 0:
            st.error("Quantity must be > 0.")
        elif size_mode == "By notional" and notional_ccy != "USD" and (fx_to_usd is None or fx_to_usd <= 0):
            st.error("FX rate invalid. Use manual override.")
        else:
            now_ts = df.index.max()
            try:
                if strategy_name.startswith("FixedRR"):
                    pos = fixed_rr.start_position(sym, df, fill_side, float(fill_price), now_ts)
                else:
                    pos = scale_out.start_position(sym, df, fill_side, float(fill_price), now_ts)
    
                tp = TrackedPosition(
                    track_id=str(uuid4()),
                    pos=pos,
                    qty=float(qty),
                    notional_usd=float(notional_usd),
                    notional_ccy=str(notional_ccy),
                    notional_ccy_amount=float(notional_ccy_amount),
                    fx_to_usd=float(fx_to_usd),
                )
                st.session_state["tracked_positions"].append(tp)
                st.success("Tracking started.")
                st.rerun()
            except Exception as e:
                st.error(str(e))

    st.divider()
    st.subheader("Active positions (auto-updated)")

    tracked: list[TrackedPosition] = st.session_state["tracked_positions"]
    updated: list[TrackedPosition] = []
    active_rows = []

    for tp in tracked:
        pos = tp.pos

        if pos.symbol == sym and (not df.empty):
            if pos.strategy == "FixedRR":
                pos = fixed_rr.update_position(pos, df)
            else:
                pos = scale_out.update_position(pos, df)

        updated.append(
            TrackedPosition(
                tp.track_id, pos,
                tp.qty, tp.notional_usd,
                tp.notional_ccy, tp.notional_ccy_amount, tp.fx_to_usd
            )
        )

        if bool(getattr(pos, "closed", False)):
            continue

        active_rows.append({
            "track_id": tp.track_id,
            "symbol": pos.symbol,
            "strategy": pos.strategy,
            "side": pos.side,
            "entry_ts": getattr(pos, "entry_ts", None),
            "entry_price": float(pos.entry_price),
            "qty": float(tp.qty),
            "notional_usd": float(tp.notional_usd),
            "notional_ccy": tp.notional_ccy,
            "notional_ccy_amount": float(tp.notional_ccy_amount),
            "fx_to_usd": float(tp.fx_to_usd),
            "stop": float(pos.stop) if getattr(pos, "stop", None) is not None else None,
            "tp1": float(getattr(pos, "tp1", None)) if getattr(pos, "tp1", None) is not None else None,
            "tp": float(getattr(pos, "tp", None)) if getattr(pos, "tp", None) is not None else None,
            "tp1_hit": bool(getattr(pos, "tp1_hit", False)),
            "be_armed": bool(getattr(pos, "be_armed", False)),
        })

    st.session_state["tracked_positions"] = updated

    if active_rows:
        table = pd.DataFrame(active_rows).drop(columns=["track_id"])
        st.dataframe(table, use_container_width=True)

        st.caption("Close a position manually (enter your exit price). It will be recorded into SQLite and used for equity curve.")
        for row in active_rows:
            tid = row["track_id"]
            pos = next(x.pos for x in updated if x.track_id == tid)
            tp = next(x for x in updated if x.track_id == tid)

            title = (
                f"{pos.symbol} | {pos.strategy} | {pos.side} | "
                f"entry={fmt_price(pos.entry_price)} | qty={tp.qty:.6f} | "
                f"notional={tp.notional_ccy_amount:.2f} {tp.notional_ccy}"
            )
            with st.expander(f"🔸 {title}", expanded=False):
                st.write(f"Notional USD: **${tp.notional_usd:.2f}**")
                st.write(f"FX: **1 {tp.notional_ccy} = {tp.fx_to_usd:.6f} USD**")

                st.write(f"Current stop: **{fmt_price(getattr(pos,'stop',None))}**")
                if getattr(pos, "tp1", None) is not None:
                    st.write(f"TP1: **{fmt_price(pos.tp1)}**  | tp1_hit={bool(getattr(pos,'tp1_hit',False))}")
                if getattr(pos, "tp", None) is not None:
                    st.write(f"TP: **{fmt_price(pos.tp)}**")

                default_exit = float(df.iloc[-1]["close"]) if (pos.symbol == sym and not df.empty) else float(pos.entry_price)
                exit_price = st.number_input("Exit price", value=default_exit, step=0.01, format="%.5f", key=f"exit_{tid}")
                note = st.text_input("Note (optional)", value="", key=f"note_{tid}")

                if st.button("End strategy (close & record)", key=f"closebtn_{tid}"):
                    pnl = calc_manual_close_pnl(tp, float(exit_price), scale_out_fraction=0.5)

                    open_ts = getattr(pos, "entry_ts", None)
                    close_ts = pd.Timestamp.now(tz=TZ).isoformat()
                    open_date = str(pd.Timestamp(open_ts).date()) if open_ts else None
                    close_date = str(pd.Timestamp(close_ts).date())

                    insert_trade(DB_PATH, {
                        "symbol": pos.symbol,
                        "strategy": pos.strategy,
                        "side": pos.side,
                        "open_ts": str(open_ts) if open_ts else None,
                        "close_ts": close_ts,
                        "open_date": open_date,
                        "close_date": close_date,
                        "entry_price": float(pos.entry_price),
                        "exit_price": float(exit_price),
                        "qty": float(tp.qty),
                        "notional_usd": float(tp.notional_usd),
                        "notional_ccy": str(tp.notional_ccy),
                        "notional_ccy_amount": float(tp.notional_ccy_amount),
                        "fx_to_usd": float(tp.fx_to_usd),
                        "stop": float(getattr(pos, "stop", None)) if getattr(pos, "stop", None) is not None else None,
                        "tp": float(getattr(pos, "tp", None)) if getattr(pos, "tp", None) is not None else None,
                        "tp1": float(getattr(pos, "tp1", None)) if getattr(pos, "tp1", None) is not None else None,
                        "tp1_hit": int(bool(getattr(pos, "tp1_hit", False))),
                        "be_armed": int(bool(getattr(pos, "be_armed", False))),
                        "pnl": float(pnl),
                        "note": note,
                    })

                    st.session_state["tracked_positions"] = [x for x in st.session_state["tracked_positions"] if x.track_id != tid]
                    st.success(f"Recorded. PnL=${pnl:.5f}")
                    st.rerun()

        csv = table.to_csv(index=False).encode("utf-8")
        st.download_button("Download active positions CSV", data=csv, file_name="active_positions.csv", mime="text/csv")
    else:
        st.info("No active tracked positions yet.")

    st.divider()
    st.subheader("Equity curve (from SQLite trade records)")

    trades_all = load_trades(DB_PATH, symbol="ALL")
    if trades_all is None or trades_all.empty:
        st.info("No trade records yet. Close a position to write into SQLite.")
    else:
        curve_col1, curve_col2 = st.columns([1, 2])

        with curve_col1:
            filter_symbol = st.selectbox("Show symbol", ["ALL", "XAUUSD", "XAGUSD"], index=0)

            trades = load_trades(DB_PATH, symbol=filter_symbol)
            trades = trades.dropna(subset=["close_date", "pnl"]).copy()

            available_dates = sorted(trades["close_date"].astype(str).unique().tolist())
            if not available_dates:
                st.warning("No valid close_date in DB yet.")
                return

            start_date = st.selectbox("Start from (available dates only)", available_dates, index=0)

        trades = trades[trades["close_date"].astype(str) >= str(start_date)].copy()
        trades = trades.sort_values("close_ts")

        trades["cum_pnl"] = trades["pnl"].cumsum()
        trades["equity"] = float(initial_capital) + trades["cum_pnl"]

        with curve_col2:
            st.write(f"Trades: **{len(trades)}**")
            st.write(
                f"Final equity: **{trades['equity'].iloc[-1]:.5f}**  |  "
                f"Return: **{(trades['equity'].iloc[-1]/float(initial_capital)-1)*100:.2f}%**"
            )
            chart_df = trades[["close_ts", "equity"]].copy()
            chart_df["close_ts"] = pd.to_datetime(chart_df["close_ts"], errors="coerce")
            chart_df = chart_df.dropna(subset=["close_ts"]).set_index("close_ts")
            st.line_chart(chart_df)

        st.caption("Equity curve is built from recorded closed trades in SQLite.")

    st.divider()
    st.subheader("Trade records database (SQLite)")

    trades_db = load_trades(DB_PATH, symbol="ALL")
    if trades_db is None or trades_db.empty:
        st.info("DB is empty.")
    else:
        st.write("Edit values directly, then click **Save edits**.")
        edited = st.data_editor(trades_db, use_container_width=True, num_rows="fixed", key="db_editor")

        c1, c2, c3 = st.columns([1, 1, 2])

        with c1:
            if st.button("Save edits to DB"):
                update_trades(DB_PATH, edited)
                st.success("Saved.")
                st.rerun()

        with c2:
            ids = trades_db["id"].astype(int).tolist()
            del_ids = st.multiselect("Delete rows by id", ids, default=[])
            if st.button("Delete selected"):
                delete_trades(DB_PATH, del_ids)
                st.success("Deleted.")
                st.rerun()

        with c3:
            confirm = st.checkbox("I understand clearing DB is irreversible")
            if st.button("Clear database (delete all)") and confirm:
                clear_trades(DB_PATH)
                st.success("Database cleared.")
                st.rerun()

    if auto_refresh:
        st.caption(f"Auto refresh enabled: every {int(refresh_seconds)}s")
        import time as time_mod
        time_mod.sleep(int(refresh_seconds))
        st.rerun()


if __name__ == "__main__":
    main()

#streamlit run app.py
# app.py
from __future__ import annotations
import logging
logging.getLogger().setLevel(logging.WARNING)
logging.getLogger("DUKASCRIPT").setLevel(logging.WARNING)


import time as time_mod
from typing import Dict, Optional

import pandas as pd
import streamlit as st

import dukascopy_python.instruments as inst

from data_fetcher import DukascopyM5Fetcher
from advisor import FixedRRAdvisor, ScaleOutATRAdvisor, PositionState


TZ = "Europe/London"

SYMBOLS = {
    "XAUUSD": inst.INSTRUMENT_FX_METALS_XAU_USD,
    "XAGUSD": inst.INSTRUMENT_FX_METALS_XAG_USD,
}

st.set_page_config(page_title="Metals Strategy Monitor", layout="wide")


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


def main():
    st.title("XAU/XAG Strategy Monitor (manual execution + live SL/TP advice)")

    fetchers = get_fetchers()

    # session state
    if "positions" not in st.session_state:
        st.session_state["positions"] = []  # list[PositionState]

    # Advisors
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
            st.write(f"Last bar (London): **{last_ts}**")
            st.write(f"OHLC: O={fmt_price(last_row['open'])} H={fmt_price(last_row['high'])} L={fmt_price(last_row['low'])} C={fmt_price(last_row['close'])}")
            st.dataframe(df.tail(20))

    with colB:
        st.subheader("Generate order ticket")
        strategy_name = st.selectbox("Strategy", ["FixedRR (24h ref)", "ScaleOutATR (mom/atr + 50% TP1 + trailing)"])

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
        start_btn = st.button("Start tracking this position")

        if start_btn:
            if df.empty or fill_price <= 0:
                st.error("Need data + a valid fill price.")
            else:
                now_ts = df.index.max()  # approximate fill time as latest bar time
                try:
                    if strategy_name.startswith("FixedRR"):
                        pos = fixed_rr.start_position(sym, df, fill_side, fill_price, now_ts)
                    else:
                        pos = scale_out.start_position(sym, df, fill_side, fill_price, now_ts)

                    st.session_state["positions"].append(pos)
                    st.success("Tracking started.")
                except Exception as e:
                    st.error(str(e))

    st.divider()
    st.subheader("Active positions (auto-updated)")

    # update all positions with latest data
    updated = []
    rows = []
    for pos in st.session_state["positions"]:
        if pos.symbol != sym:
            # still keep it, but do not update with other symbol's data here
            updated.append(pos)
            rows.append({
                "symbol": pos.symbol, "strategy": pos.strategy, "side": pos.side,
                "entry_price": pos.entry_price, "stop": pos.stop, "tp1": pos.tp1, "tp": pos.tp,
                "tp1_hit": pos.tp1_hit, "be_armed": pos.be_armed, "closed": pos.closed, "reason": pos.close_reason
            })
            continue

        if pos.strategy == "FixedRR":
            pos = fixed_rr.update_position(pos, df)
        else:
            pos = scale_out.update_position(pos, df)

        updated.append(pos)
        rows.append({
            "symbol": pos.symbol,
            "strategy": pos.strategy,
            "side": pos.side,
            "entry_price": pos.entry_price,
            "stop": pos.stop,
            "tp1": pos.tp1,
            "tp": pos.tp,
            "tp1_hit": pos.tp1_hit,
            "be_armed": pos.be_armed,
            "closed": pos.closed,
            "reason": pos.close_reason,
        })

    st.session_state["positions"] = updated

    if rows:
        table = pd.DataFrame(rows)
        st.dataframe(table)
        csv = table.to_csv(index=False).encode("utf-8")
        st.download_button("Download positions CSV", data=csv, file_name="positions.csv", mime="text/csv")
    else:
        st.info("No tracked positions yet.")

    st.caption("Tip: Keep this page open. It polls Dukascopy every ~30s and updates stops. For true automation, you’d connect broker API.")
    # Optional: auto refresh
    st.write("Auto refresh every 10s (browser stays open).")
    time_mod.sleep(10)
    st.rerun()


if __name__ == "__main__":
    main()
#streamlit run app.py
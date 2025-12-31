# sweep_entry_times_trade_day.py
from __future__ import annotations

import random
from datetime import datetime, time, timedelta
from itertools import combinations
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from XAGUSD import load_metals_last_year


# =========================
# Config (match strategy_v1 core logic)
# =========================
CACHE_DIR = "data"
USE_CACHE = True

INITIAL_CAPITAL = 100.0
RR = 3.0
FORCE_EXIT_TIME = "16:00"

SEED = 42  # change to test robustness


# =========================
# Helpers
# =========================
def parse_hhmm(s: str) -> time:
    hh, mm = s.split(":")
    return time(int(hh), int(mm))


FORCE_EXIT_T = parse_hhmm(FORCE_EXIT_TIME)


def half_hour_times() -> List[str]:
    out = []
    for h in range(24):
        out.append(f"{h:02d}:00")
        out.append(f"{h:02d}:30")
    return out


def build_features_24h(m5: pd.DataFrame) -> pd.DataFrame:
    df = m5.copy().sort_index()
    df["ref_high_24h"] = df["high"].shift(1).rolling("24h").max()
    df["ref_low_24h"] = df["low"].shift(1).rolling("24h").min()
    df["ref_mid_24h"] = (df["ref_high_24h"] + df["ref_low_24h"]) / 2
    return df


def deterministic_daily_side(trade_day: pd.Timestamp, seed: int) -> str:
    """
    Dice ONCE per trade-day (force-exit day) to keep comparison fair.
    """
    day_int = int(trade_day.strftime("%Y%m%d"))
    rng = random.Random(seed + day_int)
    return "LONG" if rng.random() < 0.5 else "SHORT"


def place_order(side: str, current_open: float, ref_mid: float) -> Tuple[str, float]:
    if side == "LONG":
        return ("MARKET", current_open) if current_open < ref_mid else ("LIMIT", ref_mid)
    else:
        return ("MARKET", current_open) if current_open > ref_mid else ("LIMIT", ref_mid)


def build_stop_tp(side: str, entry_price: float, ref_low: float, ref_high: float, rr: float) -> Tuple[float, float]:
    if side == "LONG":
        stop = float(ref_low)
        risk = entry_price - stop
        if risk <= 0:
            raise ValueError("Invalid risk LONG")
        tp = entry_price + rr * risk
        return float(stop), float(tp)
    else:
        stop = float(ref_high)
        risk = stop - entry_price
        if risk <= 0:
            raise ValueError("Invalid risk SHORT")
        tp = entry_price - rr * risk
        return float(stop), float(tp)


def trade_day_anchor_timestamp(trade_day: pd.Timestamp, hhmm: str) -> pd.Timestamp:
    """
    Map a HH:MM to an ACTUAL timestamp in the trade cycle anchored at trade_day's force-exit.
    - If HH:MM <= 16:00 => timestamp is on trade_day
    - If HH:MM > 16:00  => timestamp is on trade_day - 1 day
    """
    t = parse_hhmm(hhmm)
    if t <= FORCE_EXIT_T:
        base_date = trade_day.date()
    else:
        base_date = (trade_day - pd.Timedelta(days=1)).date()

    dt = datetime.combine(base_date, t)
    return pd.Timestamp(dt).tz_localize(
        trade_day.tz,
        nonexistent="shift_forward",  # DST spring-forward: 01:00 -> 02:00
        ambiguous=True,               # DST fall-back: pick the first 01:xx occurrence
)


def first_bar_in_window(feat: pd.DataFrame, t0: pd.Timestamp, t1: pd.Timestamp) -> Optional[pd.Timestamp]:
    """
    Get first bar timestamp >= t0 and <= t1. If none, return None.
    """
    idx = feat.index[(feat.index >= t0) & (feat.index <= t1)]
    if len(idx) == 0:
        return None
    return idx[0]


def simulate_trade_pnl(
    feat: pd.DataFrame,
    t_place_intended: pd.Timestamp,
    t_force: pd.Timestamp,
    side: str,
    rr: float,
) -> Optional[float]:
    """
    Simulate one order attempt within [t_place_intended, t_force].
    Uses the first available bar at/after intended time as placement bar.
    Returns pnl if trade fills, else None.
    """
    t_place = first_bar_in_window(feat, t_place_intended, t_force)
    if t_place is None:
        return None

    row = feat.loc[t_place]
    if any(pd.isna(row.get(k)) for k in ("ref_low_24h", "ref_high_24h", "ref_mid_24h")):
        return None

    current_open = float(row["open"])
    ref_low = float(row["ref_low_24h"])
    ref_high = float(row["ref_high_24h"])
    ref_mid = float(row["ref_mid_24h"])

    order_type, order_price = place_order(side, current_open, ref_mid)

    window = feat[(feat.index >= t_place) & (feat.index <= t_force)]
    if window.empty:
        return None

    # Fill
    if order_type == "MARKET":
        t_fill = window.index[0]
        entry_price = float(window.iloc[0]["open"])
    else:
        op = float(order_price)
        mask = (window["low"].astype(float) <= op) & (window["high"].astype(float) >= op)
        if not bool(mask.any()):
            return None
        t_fill = mask[mask].index[0]
        entry_price = op

    # Stop & TP
    try:
        stop, tp = build_stop_tp(side, entry_price, ref_low, ref_high, rr)
    except Exception:
        return None

    after = feat[(feat.index >= t_fill) & (feat.index <= t_force)]
    if after.empty:
        return None

    low = after["low"].astype(float).values
    high = after["high"].astype(float).values

    if side == "LONG":
        stop_hit = low <= stop
        tp_hit = high >= tp
    else:
        stop_hit = high >= stop
        tp_hit = low <= tp

    stop_i = int(np.argmax(stop_hit)) if stop_hit.any() else None
    tp_i = int(np.argmax(tp_hit)) if tp_hit.any() else None

    if stop_i is None and tp_i is None:
        exit_price = float(after.iloc[-1]["close"])
    else:
        # earliest hit, STOP wins ties
        if stop_i is not None and (tp_i is None or stop_i <= tp_i):
            exit_price = float(stop)
        else:
            exit_price = float(tp)

    pnl = (exit_price - entry_price) if side == "LONG" else (entry_price - exit_price)
    return float(pnl)


def precompute_pnl_matrix(feat: pd.DataFrame, times: List[str], seed: int, rr: float) -> pd.DataFrame:
    """
    Matrix:
      index = trade_day (normalized date)
      columns = HH:MM
      value = pnl if filled at that time within trade cycle, else NaN
    """
    feat = feat.sort_index()
    tz = feat.index.tz

    trade_days = pd.Index(feat.index.normalize().unique()).sort_values()
    rows = []

    for d in trade_days:
        trade_day = pd.Timestamp(d).tz_localize(tz) if pd.Timestamp(d).tz is None else pd.Timestamp(d)
        side = deterministic_daily_side(trade_day, seed)

        # force exit timestamp on trade_day 16:00
        force_dt = datetime.combine(trade_day.date(), FORCE_EXIT_T)
        t_force = pd.Timestamp(force_dt).tz_localize(trade_day.tz)

        r = {}
        for hhmm in times:
            t_place_intended = trade_day_anchor_timestamp(trade_day, hhmm)
            # If intended placement is after force exit, it can never happen in this cycle
            if t_place_intended > t_force:
                r[hhmm] = np.nan
                continue
            pnl = simulate_trade_pnl(feat, t_place_intended, t_force, side, rr)
            r[hhmm] = pnl if pnl is not None else np.nan

        rows.append(r)

    pnl_mat = pd.DataFrame(rows, index=[pd.Timestamp(d).date().isoformat() for d in trade_days], columns=times)
    return pnl_mat


def time_sort_key(hhmm: str) -> Tuple[int, int, int]:
    """
    Sort key inside a trade cycle:
      times <= 16:00 are on trade_day (offset=0)
      times > 16:00  are on previous day (offset=-1)
    """
    t = parse_hhmm(hhmm)
    offset = 0 if t <= FORCE_EXIT_T else -1
    return (offset, t.hour, t.minute)


def evaluate_single(pnl_mat: pd.DataFrame, initial_capital: float) -> pd.DataFrame:
    out = []
    for t in pnl_mat.columns:
        total_pnl = float(pnl_mat[t].fillna(0.0).sum())
        final_equity = initial_capital + total_pnl
        out.append(
            {
                "time": t,
                "final_equity": final_equity,
                "total_return": final_equity / initial_capital - 1.0,
                "fill_rate": float(pnl_mat[t].notna().mean()),
                "trade_days": int(pnl_mat[t].notna().sum()),
            }
        )
    return pd.DataFrame(out).sort_values("total_return", ascending=False)


def evaluate_pairs(pnl_mat: pd.DataFrame, initial_capital: float) -> pd.DataFrame:
    """
    For each valid pair (t1 -> t2) in chronological order inside the trade cycle:
      use pnl(t1) if filled else pnl(t2) else 0.
    """
    times = sorted(list(pnl_mat.columns), key=time_sort_key)
    out = []

    for t1, t2 in combinations(times, 2):
        p1 = pnl_mat[t1]
        p2 = pnl_mat[t2]
        daily = p1.where(~p1.isna(), p2).fillna(0.0)

        final_equity = initial_capital + float(daily.sum())
        out.append(
            {
                "t1": t1,
                "t2": t2,
                "final_equity": final_equity,
                "total_return": final_equity / initial_capital - 1.0,
                "fill_rate_any": float((~(p1.isna() & p2.isna())).mean()),
            }
        )

    return pd.DataFrame(out).sort_values("total_return", ascending=False)


def run_symbol(symbol: str, m5: pd.DataFrame) -> None:
    times = half_hour_times()
    feat = build_features_24h(m5)

    pnl_mat = precompute_pnl_matrix(feat, times, seed=SEED, rr=RR)

    single_df = evaluate_single(pnl_mat, INITIAL_CAPITAL)
    pair_df = evaluate_pairs(pnl_mat, INITIAL_CAPITAL)

    best_single = single_df.iloc[0]
    best_pair = pair_df.iloc[0]

    print(f"\n===== {symbol} | RR={RR:g} | seed={SEED} | trade-day anchored =====")
    print(f"Best single: {best_single['time']} | return={best_single['total_return']:.2%} | final={best_single['final_equity']:.2f} | fill={best_single['fill_rate']:.1%}")
    print(f"Best pair  : {best_pair['t1']} + {best_pair['t2']} | return={best_pair['total_return']:.2%} | final={best_pair['final_equity']:.2f} | fill_any={best_pair['fill_rate_any']:.1%}")

    tag = f"RR{RR:g}_seed{SEED}_tradeday".replace(":", "")
    single_df.to_csv(f"{symbol}_sweep_single_{tag}.csv", index=False)
    pair_df.to_csv(f"{symbol}_sweep_pairs_{tag}.csv", index=False)
    single_df.head(20).to_csv(f"{symbol}_sweep_single_TOP20_{tag}.csv", index=False)
    pair_df.head(20).to_csv(f"{symbol}_sweep_pairs_TOP20_{tag}.csv", index=False)


def main():
    au, ag = load_metals_last_year(cache_dir=CACHE_DIR, use_cache=USE_CACHE)
    run_symbol("XAUUSD", au.m5)
    run_symbol("XAGUSD", ag.m5)


if __name__ == "__main__":
    main()

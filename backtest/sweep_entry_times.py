# sweep_entry_times.py
from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import datetime, time, timedelta
from itertools import combinations
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd

from XAGUSD import load_metals_last_year


# =========================
# Config (match your strategy_v1)
# =========================
CACHE_DIR = "data"
USE_CACHE = True

INITIAL_CAPITAL = 100.0
RR = 3.0
FORCE_EXIT_TIME = "16:00"

SEED = 42  # change this to see robustness


# =========================
# Helpers
# =========================
def parse_hhmm(s: str) -> time:
    hh, mm = s.split(":")
    return time(int(hh), int(mm))


FORCE_EXIT_T = parse_hhmm(FORCE_EXIT_TIME)


def half_hour_times() -> List[str]:
    """00:00, 00:30, ..., 23:30"""
    out = []
    for h in range(24):
        out.append(f"{h:02d}:00")
        out.append(f"{h:02d}:30")
    return out


def build_features_24h(m5: pd.DataFrame) -> pd.DataFrame:
    """
    Rolling 24h reference levels (no lookahead):
      ref_high_24h = max(high) over (t-24h, t)
      ref_low_24h  = min(low)  over (t-24h, t)
      ref_mid_24h  = (ref_high_24h + ref_low_24h) / 2
    """
    df = m5.copy().sort_index()
    df["ref_high_24h"] = df["high"].shift(1).rolling("24h").max()
    df["ref_low_24h"] = df["low"].shift(1).rolling("24h").min()
    df["ref_mid_24h"] = (df["ref_high_24h"] + df["ref_low_24h"]) / 2
    return df


def force_exit_timestamp(t_place: pd.Timestamp) -> pd.Timestamp:
    """
    If entry_time <= 16:00 -> force exit same day 16:00.
    If entry_time > 16:00  -> force exit next day 16:00.
    """
    tz = t_place.tz
    place_date = t_place.date()
    if t_place.time() <= FORCE_EXIT_T:
        dt = datetime.combine(place_date, FORCE_EXIT_T)
    else:
        dt = datetime.combine(place_date + timedelta(days=1), FORCE_EXIT_T)
    return pd.Timestamp(dt).tz_localize(tz)


def get_bar_at_or_after_time(day_df: pd.DataFrame, hhmm: str) -> Optional[pd.Timestamp]:
    """
    Return the first timestamp >= (day + HH:MM).
    This avoids 'no trade' due to missing exact HH:MM bars.
    """
    t = parse_hhmm(hhmm)
    day0 = day_df.index[0].normalize()
    target = day0 + pd.Timedelta(hours=t.hour, minutes=t.minute)

    idx = day_df.index[day_df.index >= target]
    if len(idx) == 0:
        return None
    return idx[0]


def deterministic_daily_side(day: pd.Timestamp, seed: int) -> str:
    """
    Fix the direction per calendar day to make entry-time comparisons fair.
    """
    # Use YYYYMMDD as integer to combine with seed
    day_int = int(day.strftime("%Y%m%d"))
    rng = random.Random(seed + day_int)
    return "LONG" if rng.random() < 0.5 else "SHORT"


def place_order(side: str, current_open: float, ref_mid_24h: float) -> Tuple[str, float]:
    """
    Same as your strategy:
      LONG:  if open < mid -> MARKET else LIMIT at mid
      SHORT: if open > mid -> MARKET else LIMIT at mid
    """
    if side == "LONG":
        return ("MARKET", current_open) if current_open < ref_mid_24h else ("LIMIT", ref_mid_24h)
    else:
        return ("MARKET", current_open) if current_open > ref_mid_24h else ("LIMIT", ref_mid_24h)


def build_stop_tp(side: str, entry_price: float, ref_low: float, ref_high: float, rr: float) -> Tuple[float, float]:
    """
    Stop based on rolling 24h extremes:
      LONG  stop = ref_low
      SHORT stop = ref_high
    TP uses fixed RR based on initial risk distance.
    """
    if side == "LONG":
        stop = ref_low
        risk = entry_price - stop
        if risk <= 0:
            raise ValueError("Invalid risk LONG")
        tp = entry_price + rr * risk
        return float(stop), float(tp)
    else:
        stop = ref_high
        risk = stop - entry_price
        if risk <= 0:
            raise ValueError("Invalid risk SHORT")
        tp = entry_price - rr * risk
        return float(stop), float(tp)


def simulate_trade_pnl(
    feat: pd.DataFrame,
    t_place: pd.Timestamp,
    side: str,
    rr: float,
) -> Optional[float]:
    """
    Simulate ONE attempt at time t_place (given a fixed side).
    Returns pnl (float) if filled, or None if not filled / not tradable.
    """
    if t_place not in feat.index:
        return None

    row = feat.loc[t_place]
    if any(pd.isna(row.get(k)) for k in ("ref_low_24h", "ref_high_24h", "ref_mid_24h")):
        return None

    current_open = float(row["open"])
    ref_low = float(row["ref_low_24h"])
    ref_high = float(row["ref_high_24h"])
    ref_mid = float(row["ref_mid_24h"])

    order_type, order_price = place_order(side, current_open, ref_mid)
    t_force = force_exit_timestamp(t_place)

    # Window from placement to force exit (may cross into next day)
    window = feat[(feat.index >= t_place) & (feat.index <= t_force)]
    if window.empty:
        return None

    # Fill
    if order_type == "MARKET":
        t_fill = window.index[0]
        entry_price = float(window.iloc[0]["open"])
    else:
        mask = (window["low"].astype(float) <= float(order_price)) & (window["high"].astype(float) >= float(order_price))
        if not bool(mask.any()):
            return None
        t_fill = mask[mask].index[0]
        entry_price = float(order_price)

    # Stop & TP
    try:
        stop, tp = build_stop_tp(side, entry_price, ref_low, ref_high, rr)
    except Exception:
        return None

    after = feat[(feat.index >= t_fill) & (feat.index <= t_force)]
    if after.empty:
        return None

    # Vectorized hit detection (conservative: STOP first if same bar)
    low = after["low"].astype(float).values
    high = after["high"].astype(float).values
    idx = after.index

    if side == "LONG":
        stop_hit = low <= stop
        tp_hit = high >= tp
    else:
        stop_hit = high >= stop
        tp_hit = low <= tp

    stop_i = int(np.argmax(stop_hit)) if stop_hit.any() else None
    tp_i = int(np.argmax(tp_hit)) if tp_hit.any() else None

    if stop_i is None and tp_i is None:
        # Force exit at last close in window (<= t_force)
        exit_price = float(after.iloc[-1]["close"])
    else:
        # Choose earliest hit, STOP wins ties
        if stop_i is not None and (tp_i is None or stop_i <= tp_i):
            exit_price = float(stop)
        else:
            exit_price = float(tp)

    pnl = (exit_price - entry_price) if side == "LONG" else (entry_price - exit_price)
    return float(pnl)


def precompute_day_time_pnls(
    feat: pd.DataFrame,
    times: List[str],
    seed: int,
    rr: float,
) -> pd.DataFrame:
    """
    Build a matrix:
      index = cal_date (Timestamp at midnight)
      columns = "HH:MM" times
      values = pnl (float) if a trade FILLS at that time, else NaN
    """
    feat = feat.sort_index()
    feat["cal_date"] = feat.index.normalize()

    days = []
    pnl_rows = []

    for day, day_df in feat.groupby("cal_date"):
        side = deterministic_daily_side(day, seed)

        row = {}
        for hhmm in times:
            t_place = get_bar_at_or_after_time(day_df, hhmm)
            if t_place is None:
                row[hhmm] = np.nan
                continue
            pnl = simulate_trade_pnl(feat, t_place, side, rr)
            row[hhmm] = pnl if pnl is not None else np.nan

        days.append(day)
        pnl_rows.append(row)

    pnl_mat = pd.DataFrame(pnl_rows, index=days, columns=times)
    return pnl_mat


def evaluate_single_times(pnl_mat: pd.DataFrame, initial_capital: float) -> pd.DataFrame:
    """
    For each single time: daily pnl is pnl if filled else 0.
    """
    results = []
    for t in pnl_mat.columns:
        total_pnl = float(pnl_mat[t].fillna(0.0).sum())
        final_equity = initial_capital + total_pnl
        total_return = final_equity / initial_capital - 1.0
        results.append({"time": t, "final_equity": final_equity, "total_return": total_return})
    return pd.DataFrame(results).sort_values("total_return", ascending=False)


def evaluate_time_pairs(pnl_mat: pd.DataFrame, initial_capital: float) -> pd.DataFrame:
    """
    For each ordered pair (t1 < t2):
      use pnl(t1) if filled else pnl(t2) else 0
    """
    results = []
    cols = list(pnl_mat.columns)

    for t1, t2 in combinations(cols, 2):
        p1 = pnl_mat[t1]
        p2 = pnl_mat[t2]
        daily = p1.where(~p1.isna(), p2).fillna(0.0)
        total_pnl = float(daily.sum())
        final_equity = initial_capital + total_pnl
        total_return = final_equity / initial_capital - 1.0
        results.append({"t1": t1, "t2": t2, "final_equity": final_equity, "total_return": total_return})

    return pd.DataFrame(results).sort_values("total_return", ascending=False)


def run_symbol(symbol: str, m5: pd.DataFrame) -> None:
    times = half_hour_times()

    feat = build_features_24h(m5)
    pnl_mat = precompute_day_time_pnls(feat, times, seed=SEED, rr=RR)

    single_df = evaluate_single_times(pnl_mat, INITIAL_CAPITAL)
    pair_df = evaluate_time_pairs(pnl_mat, INITIAL_CAPITAL)

    best_single = single_df.iloc[0]
    best_pair = pair_df.iloc[0]

    print(f"\n===== {symbol} | RR={RR:g} | seed={SEED} =====")
    print(f"Best single time: {best_single['time']} | return={best_single['total_return']:.2%} | final={best_single['final_equity']:.2f}")
    print(f"Best pair: {best_pair['t1']} + {best_pair['t2']} | return={best_pair['total_return']:.2%} | final={best_pair['final_equity']:.2f}")

    # Save CSVs
    tag = f"RR{RR:g}_seed{SEED}".replace(":", "")
    single_df.to_csv(f"{symbol}_sweep_single_{tag}.csv", index=False)
    pair_df.to_csv(f"{symbol}_sweep_pairs_{tag}.csv", index=False)

    # Also save top 20 for quick view
    single_df.head(20).to_csv(f"{symbol}_sweep_single_TOP20_{tag}.csv", index=False)
    pair_df.head(20).to_csv(f"{symbol}_sweep_pairs_TOP20_{tag}.csv", index=False)


def main():
    au, ag = load_metals_last_year(cache_dir=CACHE_DIR, use_cache=USE_CACHE)

    # Run for both
    run_symbol("XAUUSD", au.m5)
    run_symbol("XAGUSD", ag.m5)


if __name__ == "__main__":
    main()

# sweep_mom_atr_scaleout_v1.py
from __future__ import annotations

import numpy as np
import pandas as pd
from datetime import datetime, time, timedelta
from dataclasses import dataclass
from itertools import combinations
from typing import Dict, List, Optional, Tuple

from XAGUSD import load_metals_last_year


# =========================
# Config (keep same as your strategy_mom_atr_scaleout_v1.py)
# =========================
CACHE_DIR = "data"
USE_CACHE = True

INITIAL_CAPITAL = 100.0

FORCE_EXIT_TIME = "16:00"      # London time
MOM_LOOKBACK_BARS = 288        # 24h on M5
MOM_ATR_THRESHOLD = 0.5
ATR_PERIOD = 14

INIT_ATR_MULT = 2.0
BE_TRIGGER_ATR = 1.0
TRAIL_ATR_MULT = 1.5

SCALE_OUT_FRACTION = 0.5
TP1_R_MULT = 2.0

TOP_N = 20


# =========================
# Helpers
# =========================
def parse_hhmm(s: str) -> time:
    hh, mm = s.split(":")
    return time(int(hh), int(mm))


FORCE_T = parse_hhmm(FORCE_EXIT_TIME)


def half_hour_times() -> List[str]:
    return [f"{h:02d}:{m:02d}" for h in range(24) for m in (0, 30)]


def ensure_datetime_index(m5: pd.DataFrame) -> pd.DataFrame:
    if isinstance(m5.index, pd.DatetimeIndex):
        return m5.sort_index()
    for c in ["time", "datetime", "date", "timestamp"]:
        if c in m5.columns:
            df = m5.copy()
            df[c] = pd.to_datetime(df[c])
            return df.set_index(c).sort_index()
    raise ValueError("m5 must have DatetimeIndex or a time column among: time/datetime/date/timestamp")


def ensure_atr(m5: pd.DataFrame, period: int, col_out: str = "atr") -> pd.DataFrame:
    if col_out in m5.columns:
        return m5
    df = m5.copy()

    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    prev_close = close.shift(1)

    tr = pd.concat(
        [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)

    df[col_out] = tr.rolling(period, min_periods=period).mean()
    return df


def ensure_momentum_24h(m5: pd.DataFrame, lookback: int) -> pd.DataFrame:
    df = m5.copy()
    if "close_prev" not in df.columns:
        df["close_prev"] = df["close"].astype(float).shift(1)
    df["mom_raw"] = df["close_prev"] - df["close_prev"].shift(lookback)
    return df


def first_bar_at_or_after(df_day: pd.DataFrame, t: time) -> Optional[pd.Series]:
    """
    EXACTLY matches your strategy logic:
    Use the first bar with timestamp >= day + HH:MM (not exact match).
    """
    day0 = df_day.index[0].normalize()
    target = day0 + pd.Timedelta(hours=t.hour, minutes=t.minute)
    idx = df_day.index[df_day.index >= target]
    if len(idx) == 0:
        return None
    return df_day.loc[idx[0]]


def compute_force_exit_ts(entry_ts: pd.Timestamp) -> pd.Timestamp:
    """
    Same as your strategy:
      - if entry <= 16:00 => same day 16:00
      - else => next day 16:00
    """
    tz = entry_ts.tz
    d = entry_ts.date()
    if entry_ts.time() <= FORCE_T:
        dt = datetime.combine(d, FORCE_T)
    else:
        dt = datetime.combine(d + timedelta(days=1), FORCE_T)
    return pd.Timestamp(dt).tz_localize(tz)


def momentum_side(row: pd.Series) -> str:
    atr = row.get("atr", np.nan)
    mom = row.get("mom_raw", np.nan)
    if pd.isna(atr) or pd.isna(mom) or float(atr) <= 0:
        return "NONE"
    mom_atr = float(mom) / float(atr)
    if mom_atr >= MOM_ATR_THRESHOLD:
        return "LONG"
    if mom_atr <= -MOM_ATR_THRESHOLD:
        return "SHORT"
    return "NONE"


def init_stop_tp1(side: str, entry_price: float, atr: float) -> Tuple[float, float]:
    """
    Returns (stop_init, tp1) where tp1 is for the SCALE_OUT_FRACTION part.
    """
    if side == "LONG":
        stop_init = entry_price - INIT_ATR_MULT * atr
        r0 = entry_price - stop_init
        tp1 = entry_price + TP1_R_MULT * r0
    else:
        stop_init = entry_price + INIT_ATR_MULT * atr
        r0 = stop_init - entry_price
        tp1 = entry_price - TP1_R_MULT * r0
    return float(stop_init), float(tp1)


def trail_update(side: str, entry_price: float, cur_stop: float, prev_row: pd.Series, be_armed: bool) -> Tuple[float, bool]:
    """
    Same as your strategy:
      - Once favorable move >= 1 ATR => move stop to breakeven
      - Then trail stop by 1.5 ATR behind prev close
    """
    atr = float(prev_row.get("atr", float("nan")))
    if pd.isna(atr) or atr <= 0:
        return cur_stop, be_armed

    prev_close = float(prev_row["close"])

    if side == "LONG":
        favorable = prev_close - entry_price
        if (not be_armed) and (favorable >= BE_TRIGGER_ATR * atr):
            cur_stop = max(cur_stop, entry_price)
            be_armed = True
        if be_armed:
            cur_stop = max(cur_stop, prev_close - TRAIL_ATR_MULT * atr)
    else:
        favorable = entry_price - prev_close
        if (not be_armed) and (favorable >= BE_TRIGGER_ATR * atr):
            cur_stop = min(cur_stop, entry_price)
            be_armed = True
        if be_armed:
            cur_stop = min(cur_stop, prev_close + TRAIL_ATR_MULT * atr)

    return float(cur_stop), be_armed


@dataclass(frozen=True)
class TradeOutcome:
    entry_ts: pd.Timestamp
    exit_ts: pd.Timestamp
    pnl: float


def simulate_trade_once(m5: pd.DataFrame, entry_ts: pd.Timestamp, side: str) -> Optional[TradeOutcome]:
    """
    Simulate ONE trade exactly like your strategy:
      - entry at entry bar OPEN (market)
      - stop init = 2*ATR
      - TP1 at 2R on 50% position
      - remaining: ATR trailing + force exit
      - STOP checked before TP1 in same bar
    """
    entry_row = m5.loc[entry_ts]
    atr = float(entry_row.get("atr", np.nan))
    if pd.isna(atr) or atr <= 0:
        return None

    entry_price = float(entry_row["open"])
    stop, tp1 = init_stop_tp1(side, entry_price, atr)

    frac_tp1 = float(SCALE_OUT_FRACTION)
    frac_rem = 1.0 - frac_tp1

    tp1_hit = False
    tp1_pnl = 0.0
    be_armed = False

    force_ts = compute_force_exit_ts(entry_ts)
    window = m5[(m5.index >= entry_ts) & (m5.index <= force_ts)]
    if window.empty:
        return None

    rows = list(window.itertuples())

    remain_exit_price = None
    remain_exit_ts = None
    remain_reason = None

    for i, r in enumerate(rows):
        ts = r.Index
        row = window.loc[ts]

        # update stop from prev bar
        if i > 0:
            prev_ts = rows[i - 1].Index
            prev_row = window.loc[prev_ts]
            stop, be_armed = trail_update(side, entry_price, stop, prev_row, be_armed)

        lo = float(row["low"])
        hi = float(row["high"])

        if side == "LONG":
            stop_hit = lo <= stop
            tp1_hit_now = (not tp1_hit) and (hi >= tp1)
        else:
            stop_hit = hi >= stop
            tp1_hit_now = (not tp1_hit) and (lo <= tp1)

        # STOP first (worst-case)
        if stop_hit:
            remain_exit_price = float(stop)
            remain_exit_ts = ts
            remain_reason = "STOP"
            break

        # partial TP1
        if tp1_hit_now:
            tp1_hit = True
            if side == "LONG":
                tp1_pnl += frac_tp1 * (tp1 - entry_price)
            else:
                tp1_pnl += frac_tp1 * (entry_price - tp1)

            # after TP1: arm BE and ensure stop at least entry
            be_armed = True
            if side == "LONG":
                stop = max(stop, entry_price)
            else:
                stop = min(stop, entry_price)

        # force exit
        if ts >= force_ts:
            remain_exit_price = float(row["close"])
            remain_exit_ts = ts
            remain_reason = "FORCE_EXIT"
            break

    if remain_exit_price is None:
        last_ts = window.index[-1]
        remain_exit_price = float(window.loc[last_ts, "close"])
        remain_exit_ts = last_ts
        remain_reason = "WINDOW_END"

    if side == "LONG":
        remain_pnl = frac_rem * (remain_exit_price - entry_price)
    else:
        remain_pnl = frac_rem * (entry_price - remain_exit_price)

    total_pnl = float(tp1_pnl) + float(remain_pnl)
    return TradeOutcome(entry_ts=entry_ts, exit_ts=remain_exit_ts, pnl=total_pnl)


def precompute_candidates(m5: pd.DataFrame, times: List[str]) -> Tuple[List[pd.Timestamp], Dict[pd.Timestamp, pd.DataFrame], Dict[Tuple[pd.Timestamp, str], TradeOutcome]]:
    """
    Precompute:
      - per-day dataframe
      - for each (day, time): TradeOutcome if a trade would happen at that time, else absent
    """
    # group by calendar day (same as your strategy)
    day_map: Dict[pd.Timestamp, pd.DataFrame] = {}
    days: List[pd.Timestamp] = []
    for day, df_day in m5.groupby(m5.index.normalize()):
        df_day = df_day.sort_index()
        days.append(day)
        day_map[day] = df_day

    candidate: Dict[Tuple[pd.Timestamp, str], TradeOutcome] = {}

    times_t = [(t, parse_hhmm(t)) for t in times]

    for day in days:
        df_day = day_map[day]
        for t_str, t_obj in times_t:
            r = first_bar_at_or_after(df_day, t_obj)
            if r is None:
                continue
            side = momentum_side(r)
            if side == "NONE":
                continue

            entry_ts = r.name
            out = simulate_trade_once(m5, entry_ts, side)
            if out is not None:
                candidate[(day, t_str)] = out

    return days, day_map, candidate


def backtest_with_entry_times(
    days: List[pd.Timestamp],
    day_map: Dict[pd.Timestamp, pd.DataFrame],
    candidate: Dict[Tuple[pd.Timestamp, str], TradeOutcome],
    entry_times: List[str],
) -> Tuple[float, int]:
    """
    Same shape as your run_backtest:
      - iterate day by day
      - prevent overlapping trades via next_free_time
      - try entry_times in order, take first that exists + valid
    """
    equity = float(INITIAL_CAPITAL)
    trades = 0

    # next_free_time like your strategy
    next_free_time = None
    # initialize to very early
    all_idx_min = min(df.index.min() for df in day_map.values())
    next_free_time = all_idx_min

    for day in days:
        df_day = day_map[day]

        # still in position -> skip whole day if position exits after day end
        if df_day.index.max() < next_free_time:
            continue

        filled = False
        for t in entry_times:
            out = candidate.get((day, t))
            if out is None:
                continue
            if out.entry_ts < next_free_time:
                continue

            equity += float(out.pnl)
            trades += 1
            next_free_time = out.exit_ts + pd.Timedelta(microseconds=1)
            filled = True
            break

        if not filled:
            continue

    return equity, trades


def sweep_symbol(symbol: str, m5_raw: pd.DataFrame) -> None:
    print(f"\n===== Sweeping {symbol} (mom+atr scale-out) =====")

    m5 = ensure_datetime_index(m5_raw)
    needed = {"open", "high", "low", "close"}
    missing = needed - set(m5.columns)
    if missing:
        raise ValueError(f"{symbol}: Missing OHLC columns: {sorted(missing)}")

    m5 = ensure_atr(m5, ATR_PERIOD, "atr")
    m5 = ensure_momentum_24h(m5, MOM_LOOKBACK_BARS)

    times = half_hour_times()

    # Precompute all possible outcomes once
    days, day_map, candidate = precompute_candidates(m5, times)
    print(f"Precomputed candidates: {len(candidate)} (day,time) entries with valid signal+trade")

    # Single time sweep
    single_rows = []
    for t in times:
        eq, n = backtest_with_entry_times(days, day_map, candidate, [t])
        single_rows.append(
            {"time": t, "trades": n, "final_equity": eq, "total_return": eq / INITIAL_CAPITAL - 1.0}
        )
    single_df = pd.DataFrame(single_rows).sort_values("total_return", ascending=False)

    # Pair sweep (t1 < t2)
    pair_rows = []
    for t1, t2 in combinations(times, 2):
        eq, n = backtest_with_entry_times(days, day_map, candidate, [t1, t2])
        pair_rows.append(
            {"t1": t1, "t2": t2, "trades": n, "final_equity": eq, "total_return": eq / INITIAL_CAPITAL - 1.0}
        )
    pair_df = pd.DataFrame(pair_rows).sort_values("total_return", ascending=False)

    best_single = single_df.iloc[0]
    best_pair = pair_df.iloc[0]

    print(f"Best single: {best_single['time']} | return={best_single['total_return']:.2%} | final={best_single['final_equity']:.2f} | trades={int(best_single['trades'])}")
    print(f"Best pair  : {best_pair['t1']} + {best_pair['t2']} | return={best_pair['total_return']:.2%} | final={best_pair['final_equity']:.2f} | trades={int(best_pair['trades'])}")

    tag = f"mom24h_thr{MOM_ATR_THRESHOLD}_scale{SCALE_OUT_FRACTION}_tp{TP1_R_MULT}R"
    single_df.to_csv(f"{symbol}_mom_scaleout_sweep_single_{tag}.csv", index=False)
    pair_df.to_csv(f"{symbol}_mom_scaleout_sweep_pairs_{tag}.csv", index=False)
    single_df.head(TOP_N).to_csv(f"{symbol}_mom_scaleout_sweep_single_TOP{TOP_N}_{tag}.csv", index=False)
    pair_df.head(TOP_N).to_csv(f"{symbol}_mom_scaleout_sweep_pairs_TOP{TOP_N}_{tag}.csv", index=False)


def main():
    au, ag = load_metals_last_year(cache_dir=CACHE_DIR, use_cache=USE_CACHE)

    sweep_symbol("XAUUSD", au.m5)
    sweep_symbol("XAGUSD", ag.m5)


if __name__ == "__main__":
    main()

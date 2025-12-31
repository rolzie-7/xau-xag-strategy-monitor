# strategy_mom_atr_scaleout_v1.py
from __future__ import annotations

import numpy as np
import pandas as pd
from datetime import datetime, time, timedelta
from dataclasses import dataclass

from XAGUSD import load_metals_last_year


# =========================
# Config
# =========================
SYMBOL = "XAUUSD"
CACHE_DIR = "data"
USE_CACHE = True

INITIAL_CAPITAL = 100.0

# Try in order; we will use the first entry time that has a VALID signal
ENTRY_TIMES = ["23:30","03:30"]  # can try different combination or time

# Flatten at 16:00 London time:
# - if entry <= 16:00 => same day 16:00
# - if entry > 16:00  => next day 16:00
FORCE_EXIT_TIME = "16:00"

# Momentum over past 24 hours on M5: 24h * 12 bars/h = 288 bars
MOM_LOOKBACK_BARS = 288

# Momentum threshold in ATR units:
# mom_atr = (close_prev - close_prev.shift(lookback)) / atr
MOM_ATR_THRESHOLD = 0.5     

# ATR (on M5)
ATR_PERIOD = 14

# Risk / trailing
INIT_ATR_MULT = 2.0        # initial stop distance = 2*ATR
BE_TRIGGER_ATR = 1.0       # after +1 ATR favorable, move stop to breakeven
TRAIL_ATR_MULT = 1.5       # after BE armed, trail stop at 1.5*ATR behind prev close

# Scale out
SCALE_OUT_FRACTION = 0.5   # close 50% at TP1
TP1_R_MULT = 2.0           # TP1 at +2R (R = initial risk)

DEBUG = True


# =========================
# Helpers
# =========================
def parse_hhmm(s: str) -> time:
    hh, mm = s.split(":")
    return time(int(hh), int(mm))


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
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    df[col_out] = tr.rolling(period, min_periods=period).mean()
    return df


def ensure_momentum_24h(m5: pd.DataFrame, lookback: int) -> pd.DataFrame:
    """
    Use close_prev (t-1) to avoid lookahead:
      mom_raw(t) = close_prev(t) - close_prev(t-lookback)
      mom_atr(t) = mom_raw(t) / atr(t)
    """
    df = m5.copy()
    if "close_prev" not in df.columns:
        df["close_prev"] = df["close"].astype(float).shift(1)
    df["mom_raw"] = df["close_prev"] - df["close_prev"].shift(lookback)
    return df


def first_bar_at_or_after(df_day: pd.DataFrame, t: time) -> pd.Series | None:
    day0 = df_day.index[0].normalize()
    target = day0 + pd.Timedelta(hours=t.hour, minutes=t.minute)
    idx = df_day.index[df_day.index >= target]
    if len(idx) == 0:
        return None
    return df_day.loc[idx[0]]


def compute_force_exit_ts(entry_ts: pd.Timestamp, force_t: time) -> pd.Timestamp:
    tz = entry_ts.tz
    d = entry_ts.date()
    if entry_ts.time() <= force_t:
        dt = datetime.combine(d, force_t)
    else:
        dt = datetime.combine(d + timedelta(days=1), force_t)
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


def init_stop_tp1(side: str, entry_price: float, atr: float) -> tuple[float, float, float]:
    """
    Initial stop based on ATR.
    R0 = initial risk distance
    TP1 at TP1_R_MULT * R0, used for partial take profit.
    """
    if side == "LONG":
        stop_init = entry_price - INIT_ATR_MULT * atr
        r0 = entry_price - stop_init
        tp1 = entry_price + TP1_R_MULT * r0
    else:
        stop_init = entry_price + INIT_ATR_MULT * atr
        r0 = stop_init - entry_price
        tp1 = entry_price - TP1_R_MULT * r0
    return float(stop_init), float(tp1), float(r0)


def trail_update(side: str, entry_price: float, cur_stop: float, prev_row: pd.Series, state: dict) -> tuple[float, dict]:
    """
    Update stop using PREVIOUS completed bar (avoid lookahead):
      - If favorable >= 1 ATR -> move stop to breakeven
      - Then trail at 1.5 ATR behind prev close
    """
    atr = float(prev_row.get("atr", float("nan")))
    if pd.isna(atr) or atr <= 0:
        return cur_stop, state

    prev_close = float(prev_row["close"])
    be_armed = bool(state.get("be_armed", False))

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

    state["be_armed"] = be_armed
    return float(cur_stop), state


# =========================
# Backtest
# =========================
@dataclass
class Trade:
    symbol: str
    date: str
    entry_time_cfg: str
    side: str
    entry_ts: str
    entry_price: float

    atr_at_entry: float
    stop_init: float
    r0: float
    tp1: float

    tp1_hit: bool
    tp1_ts: str | None
    tp1_pnl: float

    remain_exit_ts: str
    remain_exit_price: float
    remain_exit_reason: str
    remain_pnl: float

    total_pnl: float
    equity_after: float


def run_backtest(m5: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    m5 = ensure_datetime_index(m5)

    required = {"open", "high", "low", "close"}
    missing = required - set(m5.columns)
    if missing:
        raise ValueError(f"Missing OHLC columns: {sorted(missing)}")

    m5 = ensure_atr(m5, ATR_PERIOD, "atr")
    m5 = ensure_momentum_24h(m5, MOM_LOOKBACK_BARS)

    entry_times_t = [parse_hhmm(x) for x in ENTRY_TIMES]
    force_t = parse_hhmm(FORCE_EXIT_TIME)

    equity = float(INITIAL_CAPITAL)
    trades: list[Trade] = []
    eq_rows: list[dict] = []

    # prevent overlapping trades (since 23:30 holds to next day 16:00)
    next_free_time = m5.index.min()

    # debug counters
    dbg_days = dbg_entry_bar = dbg_signal = dbg_trades = 0

    for day, df_day in m5.groupby(m5.index.normalize()):
        dbg_days += 1
        df_day = df_day.sort_index()

        # if still in position, skip this whole day for new entries
        if df_day.index.max() < next_free_time:
            eq_rows.append({"date": day.date().isoformat(), "equity": equity})
            continue

        # Try entry times in order; pick the first that has a VALID signal (not NONE)
        chosen = None
        chosen_cfg = None
        chosen_side = "NONE"

        for cfg, t_obj in zip(ENTRY_TIMES, entry_times_t):
            r = first_bar_at_or_after(df_day, t_obj)
            if r is None:
                continue
            if r.name < next_free_time:
                continue

            s = momentum_side(r)
            if s == "NONE":
                continue

            chosen = r
            chosen_cfg = cfg
            chosen_side = s
            break

        if chosen is None:
            eq_rows.append({"date": day.date().isoformat(), "equity": equity})
            continue

        dbg_entry_bar += 1
        dbg_signal += 1

        entry_ts = chosen.name
        entry_price = float(chosen["open"])
        atr = float(chosen.get("atr", np.nan))
        if pd.isna(atr) or atr <= 0:
            eq_rows.append({"date": day.date().isoformat(), "equity": equity})
            continue

        # init stop + tp1 based on initial risk
        stop, tp1, r0 = init_stop_tp1(chosen_side, entry_price, atr)

        # Position sizes
        frac_tp1 = float(SCALE_OUT_FRACTION)
        frac_rem = 1.0 - frac_tp1

        tp1_hit = False
        tp1_ts = None
        tp1_pnl = 0.0

        state = {"be_armed": False}

        # force exit time (same day or next day)
        force_ts = compute_force_exit_ts(entry_ts, force_t)

        # monitor on full m5 (may cross into next day)
        window = m5[(m5.index >= entry_ts) & (m5.index <= force_ts)]
        if window.empty:
            eq_rows.append({"date": day.date().isoformat(), "equity": equity})
            continue

        remain_exit_price = None
        remain_exit_ts = None
        remain_exit_reason = None

        rows = list(window.itertuples())

        for i, r in enumerate(rows):
            ts = r.Index
            row = window.loc[ts]

            # update trailing stop from previous bar
            if i > 0:
                prev_ts = rows[i - 1].Index
                prev_row = window.loc[prev_ts]
                stop, state = trail_update(chosen_side, entry_price, stop, prev_row, state)

            lo = float(row["low"])
            hi = float(row["high"])

            # STOP check first (conservative)
            if chosen_side == "LONG":
                stop_hit = lo <= stop
                tp1_hit_now = (not tp1_hit) and (hi >= tp1)
            else:
                stop_hit = hi >= stop
                tp1_hit_now = (not tp1_hit) and (lo <= tp1)

            # If both stop and tp1 touched in same bar => assume STOP first (worst-case)
            if stop_hit:
                # remaining exits; if tp1 not hit before, entire position exits
                remain_exit_price = stop
                remain_exit_ts = ts
                if (not tp1_hit) and tp1_hit_now:
                    remain_exit_reason = "STOP_AND_TP1_SAME_BAR_STOP_FIRST"
                else:
                    remain_exit_reason = "STOP"
                break

            # Partial take profit (TP1)
            if tp1_hit_now:
                tp1_hit = True
                tp1_ts = ts

                # Realize pnl on frac_tp1
                if chosen_side == "LONG":
                    tp1_pnl += frac_tp1 * (tp1 - entry_price)
                else:
                    tp1_pnl += frac_tp1 * (entry_price - tp1)

                # After taking profit, arm BE and ensure stop at least breakeven
                state["be_armed"] = True
                if chosen_side == "LONG":
                    stop = max(stop, entry_price)
                else:
                    stop = min(stop, entry_price)

            # Force exit (if reached force time)
            if ts >= force_ts:
                remain_exit_price = float(row["close"])
                remain_exit_ts = ts
                remain_exit_reason = "FORCE_EXIT"
                break

        # If not exited by loop end
        if remain_exit_price is None:
            last_ts = window.index[-1]
            remain_exit_price = float(window.loc[last_ts, "close"])
            remain_exit_ts = last_ts
            remain_exit_reason = "WINDOW_END"

        # Remaining pnl
        if chosen_side == "LONG":
            remain_pnl = frac_rem * (remain_exit_price - entry_price)
        else:
            remain_pnl = frac_rem * (entry_price - remain_exit_price)

        total_pnl = float(tp1_pnl) + float(remain_pnl)
        equity += total_pnl

        trades.append(
            Trade(
                symbol=SYMBOL,
                date=entry_ts.date().isoformat(),
                entry_time_cfg=str(chosen_cfg),
                side=chosen_side,
                entry_ts=pd.Timestamp(entry_ts).isoformat(),
                entry_price=float(entry_price),
                atr_at_entry=float(atr),
                stop_init=float(stop),   # note: stop has been trailed; keep stop_init separately if needed
                r0=float(r0),
                tp1=float(tp1),
                tp1_hit=bool(tp1_hit),
                tp1_ts=pd.Timestamp(tp1_ts).isoformat() if tp1_ts is not None else None,
                tp1_pnl=float(tp1_pnl),
                remain_exit_ts=pd.Timestamp(remain_exit_ts).isoformat(),
                remain_exit_price=float(remain_exit_price),
                remain_exit_reason=str(remain_exit_reason),
                remain_pnl=float(remain_pnl),
                total_pnl=float(total_pnl),
                equity_after=float(equity),
            )
        )

        dbg_trades += 1
        next_free_time = pd.Timestamp(remain_exit_ts) + pd.Timedelta(microseconds=1)

        eq_rows.append({"date": day.date().isoformat(), "equity": float(equity)})

    trades_df = pd.DataFrame([t.__dict__ for t in trades])
    equity_df = pd.DataFrame(eq_rows).drop_duplicates("date", keep="last")

    if DEBUG:
        print("==== DEBUG SUMMARY ====")
        print("days:", dbg_days)
        print("days with valid entry+signal:", dbg_entry_bar)
        print("trades executed:", dbg_trades)

    return trades_df, equity_df


def main():
    au, ag = load_metals_last_year(cache_dir=CACHE_DIR, use_cache=USE_CACHE)
    data = au if SYMBOL.upper() == "XAUUSD" else ag

    trades_df, equity_df = run_backtest(data.m5)

    tag = f"mom24h_thr{MOM_ATR_THRESHOLD}_scale{SCALE_OUT_FRACTION}_tp{TP1_R_MULT}R_times{'-'.join(ENTRY_TIMES)}".replace(":", "")
    trades_df.to_csv(f"{SYMBOL}_trades_mom_atr_scaleout_v1_{tag}.csv", index=False)
    equity_df.to_csv(f"{SYMBOL}_equity_mom_atr_scaleout_v1_{tag}.csv", index=False)

    print("Trades:", len(trades_df))
    if not equity_df.empty:
        print("Final equity:", float(equity_df["equity"].iloc[-1]))

    # Optional plot
    try:
        import matplotlib.pyplot as plt
        if not equity_df.empty:
            plt.figure()
            plt.plot(pd.to_datetime(equity_df["date"]), equity_df["equity"])
            plt.title(f"{SYMBOL} Equity | MOM(24h)/ATR scale-out | thr={MOM_ATR_THRESHOLD}")
            plt.xlabel("Date")
            plt.ylabel("Equity ($)")
            plt.show()
    except Exception as e:
        print("Plot skipped:", e)


if __name__ == "__main__":
    main()

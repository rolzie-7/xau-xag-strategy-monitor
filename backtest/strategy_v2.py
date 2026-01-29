# strategy_mom_atr_v2.py
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from datetime import datetime, time, timedelta

from XAGUSD import load_metals_last_year


# =========================
# Config
# =========================
SYMBOL = "XAGUSD"
CACHE_DIR = "data"
USE_CACHE = True

INITIAL_CAPITAL = 100.0

# Try in order; first time that has a valid signal will be used
ENTRY_TIMES = ["08:00"]

# Flatten at 16:00 London time:
# - if entry <= 16:00 => same day 16:00
# - if entry > 16:00  => next day 16:00
FORCE_EXIT_TIME = "16:00"

# Momentum lookback in bars (M5: 20 bars = 100 minutes)
MOM_LOOKBACK = 20

# Momentum threshold in ATR units:
# e.g. 0.5 means "price moved 0.5 ATR over lookback"
MOM_ATR_THRESHOLD = 0.5

# ATR parameters (computed on M5)
ATR_PERIOD = 14

# Stop logic (ATR-based)
INIT_ATR_MULT = 2.0          # initial SL distance = 2 * ATR
BE_TRIGGER_ATR = 1.0         # move SL to breakeven after +1 ATR favorable move
TRAIL_ATR_MULT = 1.5         # then trail SL at 1.5 ATR behind prev close

# Optional TP based on RR
USE_TP = False
RR = 3.0

DEBUG = True


# =========================
# Helpers
# =========================
def parse_hhmm(s: str) -> time:
    hh, mm = s.split(":")
    return time(int(hh), int(mm))


def ensure_datetime_index(m5: pd.DataFrame) -> pd.DataFrame:
    """Ensure dataframe index is a DatetimeIndex."""
    if isinstance(m5.index, pd.DatetimeIndex):
        return m5.sort_index()

    for c in ["time", "datetime", "date", "timestamp"]:
        if c in m5.columns:
            df = m5.copy()
            df[c] = pd.to_datetime(df[c])
            return df.set_index(c).sort_index()

    raise ValueError("m5 must have DatetimeIndex or a time column among: time/datetime/date/timestamp")


def ensure_atr(m5: pd.DataFrame, period: int = ATR_PERIOD, col_out: str = "atr") -> pd.DataFrame:
    """Compute ATR(period) using True Range, then SMA."""
    if col_out in m5.columns:
        return m5

    df = m5.copy()
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    prev_close = close.shift(1)

    tr1 = (high - low).abs()
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    df[col_out] = tr.rolling(period, min_periods=period).mean()
    return df


def ensure_momentum(m5: pd.DataFrame, lookback: int = MOM_LOOKBACK) -> pd.DataFrame:
    """
    Momentum = (close_prev - close_prev.shift(lookback)).
    We use close_prev (t-1) to avoid lookahead.
    """
    df = m5.copy()
    if "close_prev" not in df.columns:
        df["close_prev"] = df["close"].astype(float).shift(1)

    if "mom_raw" not in df.columns:
        df["mom_raw"] = df["close_prev"] - df["close_prev"].shift(lookback)

    return df


def get_first_bar_at_or_after(df_day: pd.DataFrame, t: time) -> pd.Series | None:
    """
    Return the first bar with timestamp >= (day + t).
    Avoids missing exact HH:MM.
    """
    day0 = df_day.index[0].normalize()
    target = day0 + pd.Timedelta(hours=t.hour, minutes=t.minute)
    idx = df_day.index[df_day.index >= target]
    if len(idx) == 0:
        return None
    return df_day.loc[idx[0]]


def compute_force_exit_ts(entry_ts: pd.Timestamp, force_t: time) -> pd.Timestamp:
    """
    If entry_ts.time <= force_t => same day force_t
    Else => next day force_t
    """
    tz = entry_ts.tz
    d = entry_ts.date()
    if entry_ts.time() <= force_t:
        dt = datetime.combine(d, force_t)
    else:
        dt = datetime.combine(d + timedelta(days=1), force_t)
    return pd.Timestamp(dt).tz_localize(tz)


def momentum_side(row: pd.Series) -> str:
    """
    Use momentum normalized by ATR:
      mom_atr = mom_raw / atr
      if mom_atr > +threshold => LONG
      if mom_atr < -threshold => SHORT
    """
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


def init_stop_tp(side: str, entry_price: float, row: pd.Series) -> tuple[float, float | None]:
    """
    Initial SL: 2*ATR
    Optional TP: RR * initial risk
    """
    atr = float(row.get("atr", float("nan")))
    if pd.isna(atr) or atr <= 0:
        raise ValueError("ATR not ready.")

    if side == "LONG":
        stop = entry_price - INIT_ATR_MULT * atr
        tp = entry_price + RR * (entry_price - stop) if USE_TP else None
    else:
        stop = entry_price + INIT_ATR_MULT * atr
        tp = entry_price - RR * (stop - entry_price) if USE_TP else None

    return float(stop), (float(tp) if tp is not None else None)


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
# Backtest (no overlap positions)
# =========================
@dataclass
class Trade:
    date: str
    entry_time: str
    side: str
    entry_ts: str
    entry_price: float
    stop_init: float
    tp: float | None
    exit_ts: str
    exit_price: float
    exit_reason: str
    pnl: float
    equity_after: float
    be_armed: bool


def run_backtest(m5: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    m5 = ensure_datetime_index(m5)

    required = {"open", "high", "low", "close"}
    missing = required - set(m5.columns)
    if missing:
        raise ValueError(f"Missing OHLC columns: {sorted(missing)}")

    m5 = ensure_atr(m5, ATR_PERIOD, "atr")
    m5 = ensure_momentum(m5, MOM_LOOKBACK)

    entry_times_t = [parse_hhmm(x) for x in ENTRY_TIMES]
    force_t = parse_hhmm(FORCE_EXIT_TIME)

    equity = float(INITIAL_CAPITAL)
    trades: list[Trade] = []
    eq_rows: list[dict] = []

    # To prevent overlapping trades:
    next_free_time = m5.index.min()

    # Debug counters
    dbg_days = 0
    dbg_entry_found = 0
    dbg_signal = 0
    dbg_atr_ready = 0
    dbg_trades = 0

    for day, df_day in m5.groupby(m5.index.normalize()):
        dbg_days += 1
        df_day = df_day.sort_index()

        # If a previous trade is still open, skip entries until free
        if df_day.index.max() < next_free_time:
            eq_rows.append({"date": day.date().isoformat(), "equity": equity})
            continue

        chosen_row = None
        chosen_time_str = None

        # Find the first entry time (in order) that has an eligible bar >= next_free_time
        for t_str, t_obj in zip(ENTRY_TIMES, entry_times_t):
            r = get_first_bar_at_or_after(df_day, t_obj)
            if r is None:
                continue
            if r.name < next_free_time:
                continue
            chosen_row = r
            chosen_time_str = t_str
            break

        if chosen_row is None:
            eq_rows.append({"date": day.date().isoformat(), "equity": equity})
            continue

        dbg_entry_found += 1

        # Signal
        side = momentum_side(chosen_row)
        if side == "NONE":
            eq_rows.append({"date": day.date().isoformat(), "equity": equity})
            continue

        dbg_signal += 1

        # Entry at OPEN of chosen bar
        entry_ts = chosen_row.name
        entry_price = float(chosen_row["open"])

        # ATR ready?
        if pd.isna(chosen_row.get("atr", np.nan)) or float(chosen_row["atr"]) <= 0:
            eq_rows.append({"date": day.date().isoformat(), "equity": equity})
            continue

        dbg_atr_ready += 1

        # Init stop/tp
        try:
            stop, tp = init_stop_tp(side, entry_price, chosen_row)
        except Exception:
            eq_rows.append({"date": day.date().isoformat(), "equity": equity})
            continue

        state = {"be_armed": False}

        # Force exit timestamp (same day or next day depending on entry time)
        force_ts = compute_force_exit_ts(entry_ts, force_t)

        # IMPORTANT: monitor on the FULL m5, not just df_day (because 23:30 can go into next day)
        window = m5[(m5.index >= entry_ts) & (m5.index <= force_ts)]
        if window.empty:
            eq_rows.append({"date": day.date().isoformat(), "equity": equity})
            continue

        exit_price = None
        exit_ts = None
        exit_reason = None

        rows = list(window.itertuples())

        for i, r in enumerate(rows):
            ts = r.Index
            row = window.loc[ts]

            # Update trailing stop using previous completed bar
            if i > 0:
                prev_ts = rows[i - 1].Index
                prev_row = window.loc[prev_ts]
                stop, state = trail_update(side, entry_price, stop, prev_row, state)

            lo = float(row["low"])
            hi = float(row["high"])

            # pessimistic: if both hit in same bar, assume stop first
            if side == "LONG":
                stop_hit = lo <= stop
                tp_hit = (tp is not None) and (hi >= tp)
            else:
                stop_hit = hi >= stop
                tp_hit = (tp is not None) and (lo <= tp)

            if stop_hit and tp_hit:
                exit_price, exit_ts, exit_reason = stop, ts, "STOP_AND_TP_SAME_BAR_STOP_FIRST"
                break
            if stop_hit:
                exit_price, exit_ts, exit_reason = stop, ts, "STOP"
                break
            if tp_hit:
                exit_price, exit_ts, exit_reason = tp, ts, "TP"
                break

            if ts >= force_ts:
                exit_price, exit_ts, exit_reason = float(row["close"]), ts, "FORCE_EXIT"
                break

        if exit_price is None:
            last_ts = window.index[-1]
            exit_price, exit_ts, exit_reason = float(window.loc[last_ts, "close"]), last_ts, "WINDOW_END"

        pnl = (exit_price - entry_price) if side == "LONG" else (entry_price - exit_price)
        equity += float(pnl)

        trades.append(
            Trade(
                date=entry_ts.date().isoformat(),
                entry_time=chosen_time_str,
                side=side,
                entry_ts=pd.Timestamp(entry_ts).isoformat(),
                entry_price=float(entry_price),
                stop_init=float(stop),
                tp=float(tp) if tp is not None else None,
                exit_ts=pd.Timestamp(exit_ts).isoformat(),
                exit_price=float(exit_price),
                exit_reason=str(exit_reason),
                pnl=float(pnl),
                equity_after=float(equity),
                be_armed=bool(state.get("be_armed", False)),
            )
        )

        dbg_trades += 1
        next_free_time = pd.Timestamp(exit_ts) + pd.Timedelta(microseconds=1)

        eq_rows.append({"date": day.date().isoformat(), "equity": float(equity)})

    trades_df = pd.DataFrame([t.__dict__ for t in trades])
    equity_df = pd.DataFrame(eq_rows).drop_duplicates("date", keep="last")

    if DEBUG:
        print("==== DEBUG SUMMARY ====")
        print("days:", dbg_days)
        print("entry bar found:", dbg_entry_found)
        print("signal triggered:", dbg_signal)
        print("ATR ready:", dbg_atr_ready)
        print("trades executed:", dbg_trades)

    return trades_df, equity_df


def main():
    au, ag = load_metals_last_year(cache_dir=CACHE_DIR, use_cache=USE_CACHE)
    data = au if SYMBOL.upper() == "XAUUSD" else ag

    trades_df, equity_df = run_backtest(data.m5)

    tag = f"mom{MOM_LOOKBACK}_thr{MOM_ATR_THRESHOLD}_times{'-'.join(ENTRY_TIMES)}".replace(":", "")
    trades_df.to_csv(f"{SYMBOL}_trades_mom_atr_v2_{tag}.csv", index=False)
    equity_df.to_csv(f"{SYMBOL}_equity_mom_atr_v2_{tag}.csv", index=False)

    print("Trades:", len(trades_df))
    if not equity_df.empty:
        print("Final equity:", float(equity_df["equity"].iloc[-1]))

    # Optional plot
    try:
        import matplotlib.pyplot as plt
        if not equity_df.empty:
            plt.figure()
            plt.plot(pd.to_datetime(equity_df["date"]), equity_df["equity"])
            plt.title(f"{SYMBOL} Equity | MOM/ATR v2 | thr={MOM_ATR_THRESHOLD}")
            plt.xlabel("Date")
            plt.ylabel("Equity ($)")
            plt.show()
    except Exception as e:
        print("Plot skipped:", e)


if __name__ == "__main__":
    main()

from __future__ import annotations

import numpy as np
import pandas as pd

from XAGUSD import load_metals_last_year


# =========================
# Config
# =========================
SYMBOL = "XAUUSD"
CACHE_DIR = "data"
USE_CACHE = True

INITIAL_CAPITAL = 100.0

ENTRY_TIMES = ["23:30", "08:30"]
FORCE_EXIT_TIME = "16:00"

MOM_LOOKBACK = 20

ATR_PERIOD = 14
INIT_ATR_MULT = 2.0
BE_TRIGGER_ATR = 1.0
TRAIL_ATR_MULT = 1.5

USE_TP = True
RR = 3.0

# Dice seed (reproducible)
SEED = 42


# =========================
# Helpers
# =========================
def ensure_datetime_index(m5: pd.DataFrame) -> pd.DataFrame:
    if isinstance(m5.index, pd.DatetimeIndex):
        return m5.sort_index()
    for c in ["time", "datetime", "date", "timestamp"]:
        if c in m5.columns:
            df = m5.copy()
            df[c] = pd.to_datetime(df[c])
            return df.set_index(c).sort_index()
    raise ValueError("m5 must have DatetimeIndex or a time column among: time/datetime/date/timestamp")


def parse_hhmm(s: str) -> tuple[int, int]:
    hh, mm = s.split(":")
    return int(hh), int(mm)


def get_bar_at_or_after_time(df_day: pd.DataFrame, hhmm: str) -> pd.Series | None:
    """Return first bar with timestamp >= (day + HH:MM)."""
    h, m = parse_hhmm(hhmm)
    target = df_day.index[0].normalize() + pd.Timedelta(hours=h, minutes=m)

    idx = df_day.index[df_day.index >= target]
    if len(idx) == 0:
        return None
    return df_day.loc[idx[0]]


def ensure_atr(m5: pd.DataFrame, period: int = ATR_PERIOD, col_out: str = "atr") -> pd.DataFrame:
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


def ensure_momentum_refs(m5: pd.DataFrame, lookback: int = MOM_LOOKBACK) -> pd.DataFrame:
    df = m5.copy()

    if "close_prev" not in df.columns:
        df["close_prev"] = df["close"].astype(float).shift(1)

    if "mom_ref_high" not in df.columns:
        df["mom_ref_high"] = (
            df["high"].astype(float)
            .rolling(lookback, min_periods=lookback)
            .max()
            .shift(1)
        )

    if "mom_ref_low" not in df.columns:
        df["mom_ref_low"] = (
            df["low"].astype(float)
            .rolling(lookback, min_periods=lookback)
            .min()
            .shift(1)
        )

    return df


def momentum_side_at_entry(row: pd.Series) -> str:
    ref_high = row.get("mom_ref_high", np.nan)
    ref_low = row.get("mom_ref_low", np.nan)
    px = row.get("close_prev", np.nan)

    if pd.isna(ref_high) or pd.isna(ref_low) or pd.isna(px):
        return "NONE"

    px = float(px)
    if px > float(ref_high):
        return "LONG"
    if px < float(ref_low):
        return "SHORT"
    return "NONE"


def roll_dice_side(rng: np.random.Generator) -> str:
    """50/50 random direction."""
    return "LONG" if rng.random() < 0.5 else "SHORT"


def risk_init(side: str, entry_price: float, row: pd.Series) -> tuple[float, float | None]:
    atr = float(row.get("atr", float("nan")))
    if pd.isna(atr) or atr <= 0:
        raise ValueError("ATR not ready/invalid at entry.")

    if side == "LONG":
        stop = entry_price - INIT_ATR_MULT * atr
        tp = entry_price + RR * (entry_price - stop) if USE_TP else None
    else:
        stop = entry_price + INIT_ATR_MULT * atr
        tp = entry_price - RR * (stop - entry_price) if USE_TP else None

    return float(stop), (float(tp) if tp is not None else None)


def trail_update(side: str, entry_price: float, cur_stop: float, prev_row: pd.Series, state: dict) -> tuple[float, dict]:
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
# Backtest (max 1 trade per day)
# =========================
def run_backtest(m5: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    m5 = ensure_datetime_index(m5)

    # If your data columns are Open/High/Low/Close, uncomment:
    # m5 = m5.rename(columns={c: c.lower() for c in m5.columns})

    required = {"open", "high", "low", "close"}
    missing = required - set(m5.columns)
    if missing:
        raise ValueError(f"Missing OHLC columns: {sorted(missing)}")

    m5 = ensure_atr(m5, ATR_PERIOD, "atr")
    m5 = ensure_momentum_refs(m5, MOM_LOOKBACK)

    rng = np.random.default_rng(SEED)

    equity = float(INITIAL_CAPITAL)
    trades: list[dict] = []
    eq_rows: list[dict] = []

    for day, df_day in m5.groupby(m5.index.normalize()):
        df_day = df_day.sort_index()

        # 1) Roll dice for desired direction (LONG/SHORT)
        dice_side = roll_dice_side(rng)

        # 2) Pick entry bar (>= target time)
        entry_row = None
        entry_time_used = None
        for t in ENTRY_TIMES:
            r = get_bar_at_or_after_time(df_day, t)
            if r is not None:
                entry_row = r
                entry_time_used = t
                break

        if entry_row is None:
            eq_rows.append({"date": day.date().isoformat(), "equity": equity})
            continue

        # 3) Compute momentum breakout direction at entry
        mom_side = momentum_side_at_entry(entry_row)

        # Scheme #1: trade only if momentum exists AND matches dice direction
        if mom_side == "NONE" or mom_side != dice_side:
            eq_rows.append({"date": day.date().isoformat(), "equity": equity})
            continue

        side = dice_side

        # MARKET entry at entry bar OPEN
        entry_price = float(entry_row["open"])
        t_fill = entry_row.name

        # init risk
        try:
            stop, tp = risk_init(side, entry_price, entry_row)
        except Exception:
            eq_rows.append({"date": day.date().isoformat(), "equity": equity})
            continue

        state = {"be_armed": False}

        # force exit timestamp
        hF, mF = parse_hhmm(FORCE_EXIT_TIME)
        force_dt = day + pd.Timedelta(hours=hF, minutes=mF)

        after = df_day.loc[df_day.index >= t_fill]
        rows = list(after.itertuples())

        exit_price = None
        exit_time = None
        exit_reason = None

        for i, r in enumerate(rows):
            ts = r.Index
            row = after.loc[ts]

            if i > 0:
                prev_ts = rows[i - 1].Index
                prev_row = after.loc[prev_ts]
                stop, state = trail_update(side, entry_price, stop, prev_row, state)

            lo = float(row["low"])
            hi = float(row["high"])

            if side == "LONG":
                stop_hit = lo <= stop
                tp_hit = (tp is not None) and (hi >= tp)
            else:
                stop_hit = hi >= stop
                tp_hit = (tp is not None) and (lo <= tp)

            if stop_hit and tp_hit:
                exit_price, exit_time, exit_reason = stop, ts, "STOP_AND_TP_SAME_BAR_STOP_FIRST"
                break
            if stop_hit:
                exit_price, exit_time, exit_reason = stop, ts, "STOP"
                break
            if tp_hit:
                exit_price, exit_time, exit_reason = tp, ts, "TP"
                break

            if ts >= force_dt:
                exit_price, exit_time, exit_reason = float(row["close"]), ts, "FORCE_EXIT"
                break

        if exit_price is None:
            last_ts = after.index[-1]
            exit_price, exit_time, exit_reason = float(after.loc[last_ts, "close"]), last_ts, "DAY_END"

        pnl = (exit_price - entry_price) if side == "LONG" else (entry_price - exit_price)
        equity += float(pnl)

        trades.append(
            {
                "symbol": SYMBOL,
                "date": day.date().isoformat(),
                "entry_time": entry_time_used,
                "dice_side": dice_side,
                "mom_side": mom_side,
                "side": side,
                "entry_price": float(entry_price),
                "stop_init": float(stop),
                "tp": float(tp) if tp is not None else None,
                "exit_time": pd.Timestamp(exit_time).isoformat(),
                "exit_price": float(exit_price),
                "exit_reason": exit_reason,
                "pnl": float(pnl),
                "equity_after": float(equity),
                "be_armed": bool(state.get("be_armed", False)),
            }
        )

        eq_rows.append({"date": day.date().isoformat(), "equity": float(equity)})

    trades_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame(eq_rows).drop_duplicates("date", keep="last")
    return trades_df, equity_df


def main():
    au, ag = load_metals_last_year(cache_dir=CACHE_DIR, use_cache=USE_CACHE)
    data = au if SYMBOL.upper() == "XAUUSD" else ag

    trades_df, equity_df = run_backtest(data.m5)

    tag = f"mom{MOM_LOOKBACK}_dice_times{'-'.join(ENTRY_TIMES)}".replace(":", "")
    trades_df.to_csv(f"{SYMBOL}_trades_mom_v2_dice_{tag}.csv", index=False)
    equity_df.to_csv(f"{SYMBOL}_equity_mom_v2_dice_{tag}.csv", index=False)

    print("Trades filled:", len(trades_df))
    if not equity_df.empty:
        print("Final equity:", float(equity_df["equity"].iloc[-1]))


if __name__ == "__main__":
    main()

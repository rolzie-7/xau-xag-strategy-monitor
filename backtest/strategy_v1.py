# strategy_v1.py
from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import datetime, time, timedelta
import pandas as pd

# If your loader file is NOT named data_loader.py, change this import accordingly.
from XAGUSD import load_metals_last_year


# =========================
# Config
# =========================
SYMBOL = "XAUUSD"            # "XAUUSD" or "XAGUSD"
CACHE_DIR = "data"
USE_CACHE = True

INITIAL_CAPITAL = 100.0
RR = 3.0             # 2.0 for 1:2, 3.0 for 1:3

# You can type ANY entry times here:
ENTRY_TIMES = ["04:30"]   # examples: ["08:30"], ["08:00"], ["23:30"], ["23:30","08:30"]

FORCE_EXIT_TIME = "16:00"   # always flatten at 16:00 (London time)

SEED = 42
random.seed(SEED)


# =========================
# Helpers
# =========================
def parse_hhmm(s: str) -> time:
    hh, mm = s.split(":")
    return time(int(hh), int(mm))


ENTRY_TIMES_T = [parse_hhmm(x) for x in ENTRY_TIMES]
FORCE_EXIT_T = parse_hhmm(FORCE_EXIT_TIME)


@dataclass
class Trade:
    # Store date explicitly (avoid parsing mixed-tz strings later)
    date: str               # YYYY-MM-DD (based on fill_time local date)
    symbol: str
    side: str
    order_place_time: str
    order_type: str
    order_price: float
    fill_time: str
    entry_price: float
    ref_low_24h: float
    ref_high_24h: float
    ref_mid_24h: float
    stop: float
    take_profit: float
    exit_time: str
    exit_price: float
    exit_reason: str
    pnl: float
    equity_after: float


def build_features_24h(m5: pd.DataFrame) -> pd.DataFrame:
    """
    Compute rolling 24h reference levels:
      ref_high_24h = max(high) over (t-24h, t)
      ref_low_24h  = min(low)  over (t-24h, t)
      ref_mid_24h  = (ref_high_24h + ref_low_24h) / 2

    shift(1) avoids lookahead bias (do not use current bar).
    """
    df = m5.copy().sort_index()

    df["ref_high_24h"] = df["high"].shift(1).rolling("24h").max()
    df["ref_low_24h"] = df["low"].shift(1).rolling("24h").min()
    df["ref_mid_24h"] = (df["ref_high_24h"] + df["ref_low_24h"]) / 2
    return df


def roll_direction() -> str:
    return "LONG" if random.random() < 0.5 else "SHORT"


def place_order(side: str, current_price: float, ref_mid_24h: float) -> tuple[str, float]:
    """
    Entry rule using rolling 24h mid:
      LONG:  if current < mid -> MARKET buy else LIMIT buy at mid
      SHORT: if current > mid -> MARKET sell else LIMIT sell at mid
    """
    if side == "LONG":
        return ("MARKET", current_price) if current_price < ref_mid_24h else ("LIMIT", ref_mid_24h)
    else:
        return ("MARKET", current_price) if current_price > ref_mid_24h else ("LIMIT", ref_mid_24h)


def try_fill_order(df_window: pd.DataFrame, order_type: str, order_price: float) -> tuple[bool, pd.Timestamp | None, float | None]:
    """
    Fill model:
      MARKET: fill at placement bar open
      LIMIT: fill when low<=price<=high, at the limit price
    """
    if df_window.empty:
        return (False, None, None)

    if order_type == "MARKET":
        t0 = df_window.index[0]
        return (True, t0, float(df_window.iloc[0]["open"]))

    for t, bar in df_window.iterrows():
        if float(bar["low"]) <= order_price <= float(bar["high"]):
            return (True, t, float(order_price))

    return (False, None, None)


def force_exit_timestamp(t_place: pd.Timestamp) -> pd.Timestamp:
    """
    If entry_time <= 16:00 -> force exit same day 16:00.
    If entry_time > 16:00  -> force exit next day 16:00.
    """
    tz = t_place.tz
    place_date = t_place.date()
    if t_place.time() <= FORCE_EXIT_T:
        dt = datetime.combine(place_date, FORCE_EXIT_T)
        return pd.Timestamp(dt).tz_localize(tz)
    else:
        dt = datetime.combine(place_date + timedelta(days=1), FORCE_EXIT_T)
        return pd.Timestamp(dt).tz_localize(tz)


def build_stop_tp(side: str, entry_price: float, ref_low_24h: float, ref_high_24h: float) -> tuple[float, float]:
    """
    Stop based on rolling 24h extremes:
      LONG  stop = ref_low_24h
      SHORT stop = ref_high_24h
    TP uses fixed RR based on risk distance.
    """
    if side == "LONG":
        stop = ref_low_24h
        risk = entry_price - stop
        if risk <= 0:
            raise ValueError("Invalid risk for LONG (entry <= stop).")
        tp = entry_price + RR * risk
        return stop, tp

    stop = ref_high_24h
    risk = stop - entry_price
    if risk <= 0:
        raise ValueError("Invalid risk for SHORT (stop <= entry).")
    tp = entry_price - RR * risk
    return stop, tp


def simulate_one_order(feat: pd.DataFrame, t_place: pd.Timestamp, equity: float) -> Trade | None:
    """
    Place an order at t_place and simulate fill + stop/tp + force exit.
    """
    if t_place not in feat.index:
        return None

    row = feat.loc[t_place]

    # Need rolling 24h refs (will be NaN in the first 24h of the dataset)
    needed = ["ref_low_24h", "ref_high_24h", "ref_mid_24h"]
    if any(pd.isna(row.get(k)) for k in needed):
        return None

    side = roll_direction()

    current_price = float(row["open"])
    ref_low = float(row["ref_low_24h"])
    ref_high = float(row["ref_high_24h"])
    ref_mid = float(row["ref_mid_24h"])

    order_type, order_price = place_order(side, current_price, ref_mid)

    t_force = force_exit_timestamp(t_place)

    # Fill window: from placement time to force-exit time
    df_window = feat[(feat.index >= t_place) & (feat.index <= t_force)]
    filled, t_fill, entry_price = try_fill_order(df_window, order_type, float(order_price))
    if not filled:
        return None

    entry_price = float(entry_price)

    try:
        stop, tp = build_stop_tp(side, entry_price, ref_low, ref_high)
    except ValueError:
        return None

    # Monitor from fill to force exit
    df_after = feat[(feat.index >= t_fill) & (feat.index <= t_force)]
    if df_after.empty:
        return None

    exit_time = None
    exit_price = None
    exit_reason = None

    for t, bar in df_after.iterrows():
        low = float(bar["low"])
        high = float(bar["high"])

        # Conservative ordering: stop checked before TP
        if side == "LONG":
            if low <= stop:
                exit_time, exit_price, exit_reason = t, stop, "STOP"
                break
            if high >= tp:
                exit_time, exit_price, exit_reason = t, tp, "TP"
                break
        else:
            if high >= stop:
                exit_time, exit_price, exit_reason = t, stop, "STOP"
                break
            if low <= tp:
                exit_time, exit_price, exit_reason = t, tp, "TP"
                break

    if exit_time is None:
        # Force exit at t_force close if exists, else last close in window
        if t_force in feat.index:
            exit_time = t_force
            exit_price = float(feat.loc[t_force]["close"])
        else:
            exit_time = df_after.index[-1]
            exit_price = float(df_after.iloc[-1]["close"])
        exit_reason = "FORCE_EXIT"

    pnl = (exit_price - entry_price) if side == "LONG" else (entry_price - exit_price)
    equity_after = equity + pnl

    # Store date directly to avoid mixed-tz parsing later
    fill_date = str(t_fill.date())

    return Trade(
        date=fill_date,
        symbol=SYMBOL,
        side=side,
        order_place_time=str(t_place),
        order_type=str(order_type),
        order_price=float(order_price),
        fill_time=str(t_fill),
        entry_price=float(entry_price),
        ref_low_24h=float(ref_low),
        ref_high_24h=float(ref_high),
        ref_mid_24h=float(ref_mid),
        stop=float(stop),
        take_profit=float(tp),
        exit_time=str(exit_time),
        exit_price=float(exit_price),
        exit_reason=str(exit_reason),
        pnl=float(pnl),
        equity_after=float(equity_after),
    )


def run_backtest(feat: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    For each calendar date (London time), try entry times in the given order.
    Take the first order that gets filled (max one trade per date).
    """
    feat = feat.sort_index()
    feat["cal_date"] = feat.index.date

    equity = INITIAL_CAPITAL
    trades: list[Trade] = []

    for d, day_df in feat.groupby("cal_date"):
        filled_trade = None

        # Try times in the user-provided order
        for et in ENTRY_TIMES_T:
            candidates = day_df[day_df.index.time == et]
            if candidates.empty:
                continue

            t_place = candidates.index[0]
            tr = simulate_one_order(feat, t_place, equity)
            if tr is not None:
                filled_trade = tr
                break

        if filled_trade is None:
            continue

        equity = filled_trade.equity_after
        trades.append(filled_trade)

    trades_df = pd.DataFrame([t.__dict__ for t in trades])

    if trades_df.empty:
        equity_df = pd.DataFrame(columns=["date", "equity"])
        return trades_df, equity_df

    # Use the pre-stored 'date' (string) to build equity curve (no datetime parsing needed)
    equity_df = trades_df[["date", "equity_after"]].rename(columns={"equity_after": "equity"}).copy()
    return trades_df, equity_df


def main():
    au, ag = load_metals_last_year(cache_dir=CACHE_DIR, use_cache=USE_CACHE)
    data = au if SYMBOL.upper() == "XAUUSD" else ag

    # Only M5 is needed now
    feat = build_features_24h(data.m5)

    trades_df, equity_df = run_backtest(feat)

    tag = f"RR{RR:g}_times{'-'.join(ENTRY_TIMES)}".replace(":", "")
    trades_csv = f"{SYMBOL}_trades_24h_simple_{tag}.csv"
    equity_csv = f"{SYMBOL}_equity_24h_simple_{tag}.csv"
    trades_df.to_csv(trades_csv, index=False)
    equity_df.to_csv(equity_csv, index=False)

    print("Trades filled:", len(trades_df))
    if not equity_df.empty:
        print("Final equity:", float(equity_df["equity"].iloc[-1]))
    else:
        print("No trades filled (orders never got filled).")

    # Optional plot
    try:
        import matplotlib.pyplot as plt
        if not equity_df.empty:
            plt.figure()
            plt.plot(pd.to_datetime(equity_df["date"]), equity_df["equity"])
            plt.xlabel("Date")
            plt.ylabel("Equity ($)")
            plt.title(f"{SYMBOL} Equity Curve | 24h ref | RR={RR:g} | Times={ENTRY_TIMES}")
            plt.show()
    except Exception as e:
        print("Plot skipped:", e)


if __name__ == "__main__":
    main()

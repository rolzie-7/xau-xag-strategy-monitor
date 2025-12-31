# advisors.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd


# ---------- Common indicator helpers ----------
def compute_atr_m5(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    ATR on M5 bars using SMA(TrueRange).
    Uses completed bars only if you call it on historical up to the latest closed bar.
    """
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    prev_close = close.shift(1)

    tr = pd.concat(
        [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()


def rolling_24h_refs_m5(df: pd.DataFrame) -> Tuple[float, float, float]:
    """
    24h window on M5 => 24*12 = 288 bars.
    Use shift(1) logic: exclude the current bar to avoid lookahead.
    """
    if df is None or df.empty:
        raise ValueError("No data")

    # ensure enough bars
    if len(df) < 289:
        raise ValueError("Need at least 24h+1 of M5 bars to compute 24h refs")

    tail = df.iloc[-289:-1]  # last 24h excluding current bar
    ref_high = float(tail["high"].max())
    ref_low = float(tail["low"].min())
    ref_mid = (ref_high + ref_low) / 2.0
    return ref_low, ref_high, ref_mid


def momentum_24h_over_atr(df: pd.DataFrame, atr: pd.Series, lookback_bars: int = 288) -> float:
    """
    mom_atr = (close_prev - close_prev.shift(lookback)) / atr
    """
    close_prev = df["close"].astype(float).shift(1)
    mom_raw = close_prev - close_prev.shift(lookback_bars)
    mom_atr = mom_raw / atr
    return float(mom_atr.iloc[-1])


# ---------- Order ticket / position tracking ----------
@dataclass
class OrderTicket:
    symbol: str
    strategy: str
    side: str                 # LONG/SHORT
    order_type: str           # MARKET/LIMIT
    order_price: float        # suggested/limit price
    stop: float
    take_profit: Optional[float]
    tp1: Optional[float]      # for scale-out
    note: str


@dataclass
class PositionState:
    symbol: str
    strategy: str
    side: str
    entry_price: float
    entry_time: pd.Timestamp

    # live-updated fields
    stop: float
    tp1: Optional[float] = None
    tp: Optional[float] = None
    tp1_hit: bool = False
    be_armed: bool = False
    closed: bool = False
    close_reason: Optional[str] = None


# ---------- Strategy 1: Pure Fixed RR (24h ref stop) ----------
class FixedRRAdvisor:
    """
    - Entry suggestion uses rolling 24h mid:
        LONG:  if current < mid -> MARKET else LIMIT at mid
        SHORT: if current > mid -> MARKET else LIMIT at mid
    - Stop uses rolling 24h extremes:
        LONG stop = ref_low_24h
        SHORT stop = ref_high_24h
    - TP uses fixed RR
    """

    def __init__(self, rr: float = 3.0):
        self.rr = float(rr)

    def make_ticket(self, symbol: str, df_m5: pd.DataFrame, side: str) -> OrderTicket:
        ref_low, ref_high, ref_mid = rolling_24h_refs_m5(df_m5)
        current_open = float(df_m5.iloc[-1]["open"])

        side = side.upper()
        if side not in ("LONG", "SHORT"):
            raise ValueError("side must be LONG/SHORT")

        if side == "LONG":
            order_type, order_price = ("MARKET", current_open) if current_open < ref_mid else ("LIMIT", ref_mid)
            assumed_fill = float(order_price)
            stop = ref_low
            risk = assumed_fill - stop
            if risk <= 0:
                raise ValueError("Invalid LONG risk (fill <= stop)")
            tp = assumed_fill + self.rr * risk
        else:
            order_type, order_price = ("MARKET", current_open) if current_open > ref_mid else ("LIMIT", ref_mid)
            assumed_fill = float(order_price)
            stop = ref_high
            risk = stop - assumed_fill
            if risk <= 0:
                raise ValueError("Invalid SHORT risk (stop <= fill)")
            tp = assumed_fill - self.rr * risk

        return OrderTicket(
            symbol=symbol,
            strategy="FixedRR",
            side=side,
            order_type=order_type,
            order_price=float(order_price),
            stop=float(stop),
            take_profit=float(tp),
            tp1=None,
            note="TP/SL computed assuming fill at order_price. If your actual fill differs, input fill price to recompute.",
        )

    def start_position(self, symbol: str, df_m5: pd.DataFrame, side: str, fill_price: float, fill_time: pd.Timestamp) -> PositionState:
        ref_low, ref_high, _ = rolling_24h_refs_m5(df_m5)
        side = side.upper()
        fill_price = float(fill_price)

        if side == "LONG":
            stop = ref_low
            risk = fill_price - stop
            if risk <= 0:
                raise ValueError("Invalid LONG risk (fill <= stop)")
            tp = fill_price + self.rr * risk
        else:
            stop = ref_high
            risk = stop - fill_price
            if risk <= 0:
                raise ValueError("Invalid SHORT risk (stop <= fill)")
            tp = fill_price - self.rr * risk

        return PositionState(
            symbol=symbol,
            strategy="FixedRR",
            side=side,
            entry_price=fill_price,
            entry_time=fill_time,
            stop=float(stop),
            tp=float(tp),
        )

    def update_position(self, pos: PositionState, df_m5: pd.DataFrame) -> PositionState:
        """
        FixedRR: stop/tp do NOT trail. We only detect stop/tp hits (optional).
        """
        if pos.closed or df_m5.empty:
            return pos

        last = df_m5.iloc[-1]
        lo = float(last["low"])
        hi = float(last["high"])

        if pos.side == "LONG":
            if lo <= pos.stop:
                pos.closed = True
                pos.close_reason = "STOP_HIT"
            elif pos.tp is not None and hi >= pos.tp:
                pos.closed = True
                pos.close_reason = "TP_HIT"
        else:
            if hi >= pos.stop:
                pos.closed = True
                pos.close_reason = "STOP_HIT"
            elif pos.tp is not None and lo <= pos.tp:
                pos.closed = True
                pos.close_reason = "TP_HIT"

        return pos


# ---------- Strategy 2: 50% TP1 + 50% ATR trailing ----------
class ScaleOutATRAdvisor:
    """
    Entry signal suggestion: momentum/ATR over past 24h (M5 288 bars)
    - If mom_atr >= threshold => LONG
    - If mom_atr <= -threshold => SHORT
    Risk:
      stop_init = entry +/- INIT_ATR_MULT * ATR
      TP1 at +TP1_R_MULT * R0 on SCALE_OUT_FRACTION
      remaining uses trailing:
        - after favorable >= BE_TRIGGER_ATR * ATR => stop to breakeven
        - then trail at TRAIL_ATR_MULT * ATR behind prev close
    """

    def __init__(
        self,
        mom_atr_threshold: float = 0.5,
        atr_period: int = 14,
        mom_lookback_bars: int = 288,
        init_atr_mult: float = 2.0,
        be_trigger_atr: float = 1.0,
        trail_atr_mult: float = 1.5,
        scale_out_fraction: float = 0.5,
        tp1_r_mult: float = 2.0,
    ):
        self.mom_atr_threshold = float(mom_atr_threshold)
        self.atr_period = int(atr_period)
        self.mom_lookback_bars = int(mom_lookback_bars)

        self.init_atr_mult = float(init_atr_mult)
        self.be_trigger_atr = float(be_trigger_atr)
        self.trail_atr_mult = float(trail_atr_mult)

        self.scale_out_fraction = float(scale_out_fraction)
        self.tp1_r_mult = float(tp1_r_mult)

    def suggest_side(self, df_m5: pd.DataFrame) -> str:
        atr = compute_atr_m5(df_m5, self.atr_period)
        val = momentum_24h_over_atr(df_m5, atr, self.mom_lookback_bars)
        if np.isnan(val):
            return "NONE"
        if val >= self.mom_atr_threshold:
            return "LONG"
        if val <= -self.mom_atr_threshold:
            return "SHORT"
        return "NONE"

    def _init_levels(self, side: str, entry_price: float, atr_val: float) -> Tuple[float, float, float]:
        if side == "LONG":
            stop = entry_price - self.init_atr_mult * atr_val
            r0 = entry_price - stop
            tp1 = entry_price + self.tp1_r_mult * r0
        else:
            stop = entry_price + self.init_atr_mult * atr_val
            r0 = stop - entry_price
            tp1 = entry_price - self.tp1_r_mult * r0
        return float(stop), float(tp1), float(r0)

    def make_ticket(self, symbol: str, df_m5: pd.DataFrame) -> OrderTicket:
        side = self.suggest_side(df_m5)
        if side == "NONE":
            return OrderTicket(
                symbol=symbol,
                strategy="ScaleOutATR",
                side="NONE",
                order_type="NONE",
                order_price=float("nan"),
                stop=float("nan"),
                take_profit=None,
                tp1=None,
                note="No signal (momentum/ATR below threshold).",
            )

        atr = compute_atr_m5(df_m5, self.atr_period)
        atr_val = float(atr.iloc[-1])
        if np.isnan(atr_val) or atr_val <= 0:
            raise ValueError("ATR not ready")

        # entry assumed at current open (market-style ticket)
        entry_price = float(df_m5.iloc[-1]["open"])
        stop, tp1, r0 = self._init_levels(side, entry_price, atr_val)

        return OrderTicket(
            symbol=symbol,
            strategy="ScaleOutATR",
            side=side,
            order_type="MARKET",
            order_price=float(entry_price),
            stop=float(stop),
            take_profit=None,  # remaining has no TP
            tp1=float(tp1),
            note=f"Scale out {self.scale_out_fraction:.0%} at TP1, remaining trails. R0={r0:.5f}, ATR={atr_val:.5f}",
        )

    def start_position(self, symbol: str, df_m5: pd.DataFrame, side: str, fill_price: float, fill_time: pd.Timestamp) -> PositionState:
        atr = compute_atr_m5(df_m5, self.atr_period)
        atr_val = float(atr.iloc[-1])
        if np.isnan(atr_val) or atr_val <= 0:
            raise ValueError("ATR not ready")

        side = side.upper()
        fill_price = float(fill_price)

        stop, tp1, _ = self._init_levels(side, fill_price, atr_val)

        return PositionState(
            symbol=symbol,
            strategy="ScaleOutATR",
            side=side,
            entry_price=fill_price,
            entry_time=fill_time,
            stop=float(stop),
            tp1=float(tp1),
            tp=None,
            tp1_hit=False,
            be_armed=False,
            closed=False,
        )

    def update_position(self, pos: PositionState, df_m5: pd.DataFrame) -> PositionState:
        """
        Update trailing stop using PREVIOUS completed bar (avoid lookahead).
        Also detect TP1 hit and STOP hit.
        """
        if pos.closed or df_m5.empty:
            return pos

        # need ATR series for trailing
        atr = compute_atr_m5(df_m5, self.atr_period)
        if len(df_m5) < 2:
            return pos

        last = df_m5.iloc[-1]
        prev = df_m5.iloc[-2]
        atr_prev = float(atr.iloc[-2]) if not np.isnan(float(atr.iloc[-2])) else np.nan

        # 1) update trailing stop based on prev bar close (if ATR ready)
        if not np.isnan(atr_prev) and atr_prev > 0:
            prev_close = float(prev["close"])

            if pos.side == "LONG":
                favorable = prev_close - pos.entry_price
                if (not pos.be_armed) and (favorable >= self.be_trigger_atr * atr_prev):
                    pos.stop = max(pos.stop, pos.entry_price)
                    pos.be_armed = True
                if pos.be_armed:
                    pos.stop = max(pos.stop, prev_close - self.trail_atr_mult * atr_prev)
            else:
                favorable = pos.entry_price - prev_close
                if (not pos.be_armed) and (favorable >= self.be_trigger_atr * atr_prev):
                    pos.stop = min(pos.stop, pos.entry_price)
                    pos.be_armed = True
                if pos.be_armed:
                    pos.stop = min(pos.stop, prev_close + self.trail_atr_mult * atr_prev)

        # 2) detect TP1 hit (once)
        lo = float(last["low"])
        hi = float(last["high"])
        if (not pos.tp1_hit) and (pos.tp1 is not None):
            if pos.side == "LONG" and hi >= pos.tp1:
                pos.tp1_hit = True
                pos.be_armed = True
                pos.stop = max(pos.stop, pos.entry_price)
            if pos.side == "SHORT" and lo <= pos.tp1:
                pos.tp1_hit = True
                pos.be_armed = True
                pos.stop = min(pos.stop, pos.entry_price)

        # 3) detect STOP hit
        if pos.side == "LONG":
            if lo <= pos.stop:
                pos.closed = True
                pos.close_reason = "STOP_HIT"
        else:
            if hi >= pos.stop:
                pos.closed = True
                pos.close_reason = "STOP_HIT"

        return pos

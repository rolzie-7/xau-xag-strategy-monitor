# data_fetcher.py
from __future__ import annotations

import threading
import time as time_mod
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd
import dukascopy_python as duka


@dataclass
class LiveDataSnapshot:
    symbol: str
    tz: str
    m5: pd.DataFrame  # tz-aware index in local tz


class DukascopyM5Fetcher:
    """
    Simple live fetcher by polling duka.fetch for the newest candles.
    - No client needed
    - Good enough for "alerts / stop updates" (not HFT)
    """

    def __init__(
        self,
        symbol: str,
        instrument_const,
        tz: str = "Europe/London",
        interval=duka.INTERVAL_MIN_5,
        side=duka.OFFER_SIDE_BID,
        lookback_days: int = 3,
        poll_seconds: int = 30,
    ):
        self.symbol = symbol
        self.instrument = instrument_const
        self.tz = tz
        self.interval = interval
        self.side = side
        self.lookback_days = int(lookback_days)
        self.poll_seconds = int(poll_seconds)

        self._lock = threading.Lock()
        self._df: Optional[pd.DataFrame] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_flag = threading.Event()

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_flag.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_flag.set()
        if self._thread:
            self._thread.join(timeout=2)

    def snapshot(self) -> LiveDataSnapshot:
        with self._lock:
            df = self._df.copy() if self._df is not None else pd.DataFrame()
        return LiveDataSnapshot(symbol=self.symbol, tz=self.tz, m5=df)

    def _initial_fetch(self) -> pd.DataFrame:
        end_utc = datetime.now(timezone.utc)
        start_utc = end_utc - timedelta(days=self.lookback_days)

        df = duka.fetch(
            self.instrument,
            self.interval,
            self.side,
            start_utc.replace(tzinfo=None),  # naive UTC expected by dukascopy_python
            end_utc.replace(tzinfo=None),
        )
        if df is None or df.empty:
            return pd.DataFrame()

        df = df.sort_index()
        # Index -> UTC aware -> local tz
        idx = pd.to_datetime(df.index, utc=True).tz_convert(self.tz)
        df = df.copy()
        df.index = idx
        return df.sort_index()

    def _poll_fetch(self, last_ts_local: pd.Timestamp) -> pd.DataFrame:
        # fetch from last_ts_utc - small overlap to be safe (dedupe later)
        last_utc = last_ts_local.tz_convert("UTC")
        start_utc = (last_utc - pd.Timedelta(minutes=10)).to_pydatetime().replace(tzinfo=None)

        end_utc = datetime.now(timezone.utc).replace(tzinfo=None)

        df = duka.fetch(
            self.instrument,
            self.interval,
            self.side,
            start_utc,
            end_utc,
        )
        if df is None or df.empty:
            return pd.DataFrame()

        df = df.sort_index()
        idx = pd.to_datetime(df.index, utc=True).tz_convert(self.tz)
        df = df.copy()
        df.index = idx
        return df.sort_index()

    def _run(self) -> None:
        df = self._initial_fetch()

        with self._lock:
            self._df = df

        while not self._stop_flag.is_set():
            try:
                with self._lock:
                    cur = self._df

                if cur is None or cur.empty:
                    df_new = self._initial_fetch()
                    with self._lock:
                        self._df = df_new
                    time_mod.sleep(self.poll_seconds)
                    continue

                last_ts = cur.index.max()
                df_new = self._poll_fetch(last_ts)

                if not df_new.empty:
                    merged = pd.concat([cur, df_new], axis=0)
                    merged = merged[~merged.index.duplicated(keep="last")].sort_index()

                    # keep only last N days
                    cutoff = merged.index.max() - pd.Timedelta(days=self.lookback_days)
                    merged = merged[merged.index >= cutoff]

                    with self._lock:
                        self._df = merged

            except Exception as e:
                # don't crash the thread
                print(f"[{self.symbol}] fetch error:", e)

            time_mod.sleep(self.poll_seconds)

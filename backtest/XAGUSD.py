# data_loader.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
import pandas as pd

import dukascopy_python as duka
import dukascopy_python.instruments as inst


@dataclass
class MetalData:
    """
    Container for one symbol's datasets.
    - m5: 5-minute candles (execution timeframe)
    - daily: daily candles derived from m5 (for yesterday high/low/mid, range%)
    - h1: hourly candles derived from m5 (for ATR calculations)
    - track: m5 with intraday tracking columns (optional diagnostics)
    """
    m5: pd.DataFrame
    daily: pd.DataFrame
    h1: pd.DataFrame
    track: pd.DataFrame


class DukascopyMetalsLoader:
    """
    Pure-Python data loader for XAUUSD / XAGUSD from Dukascopy.

    Features:
    - Downloads M5 candles for a given UTC time range
    - Converts timestamps to Europe/London (handles DST automatically)
    - Derives:
        * Daily (D1) OHLC + mid + range_pct
        * Hourly (H1) OHLC + ATR14
        * Intraday tracking columns on M5
    - Optional parquet caching for fast re-runs
    """

    def __init__(
        self,
        tz: str = "Europe/London",
        interval=duka.INTERVAL_MIN_5,
        side=duka.OFFER_SIDE_BID,
        cache_dir: str | Path = "data",
    ):
        self.tz = tz
        self.interval = interval
        self.side = side
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def load_last_year(self, instrument, symbol_name: str, use_cache: bool = True) -> MetalData:
        """
        Load the last 365 days of data (relative to now, in UTC).
        """
        end_utc = datetime.now(timezone.utc)
        start_utc = end_utc - timedelta(days=365)
        return self.load_range(instrument, symbol_name, start_utc, end_utc, use_cache=use_cache)

    def load_range(
        self,
        instrument,
        symbol_name: str,
        start_utc: datetime,
        end_utc: datetime,
        use_cache: bool = True,
    ) -> MetalData:
        """
        Load data for a specific UTC time range.

        Parameters
        ----------
        start_utc / end_utc:
            timezone-aware UTC datetimes.
        use_cache:
            If True and parquet files exist, load from disk instead of fetching.
        """
        # File paths for cached datasets
        m5_path = self.cache_dir / f"{symbol_name}_M5_London.parquet"
        d1_path = self.cache_dir / f"{symbol_name}_D1_London.parquet"
        h1_path = self.cache_dir / f"{symbol_name}_H1_ATR14_London.parquet"
        tr_path = self.cache_dir / f"{symbol_name}_M5_TRACK_London.parquet"

        # If cache exists, read it and return immediately
        if use_cache and m5_path.exists() and d1_path.exists() and h1_path.exists() and tr_path.exists():
            m5 = pd.read_parquet(m5_path)
            d1 = pd.read_parquet(d1_path)
            h1 = pd.read_parquet(h1_path)
            tr = pd.read_parquet(tr_path)
            return MetalData(m5=m5, daily=d1, h1=h1, track=tr)

        # Fetch raw M5 candles from Dukascopy
        m5 = self._fetch_m5(instrument, start_utc, end_utc)

        # Derive daily/hourly/tracking datasets
        d1 = self.make_daily(m5)
        h1 = self.make_h1_atr(m5, n=14)
        tr = self.add_intraday_tracking(m5)

        # Save to parquet for fast subsequent runs
        m5.to_parquet(m5_path)
        d1.to_parquet(d1_path)
        h1.to_parquet(h1_path)
        tr.to_parquet(tr_path)

        return MetalData(m5=m5, daily=d1, h1=h1, track=tr)

    def _fetch_m5(self, instrument, start_utc: datetime, end_utc: datetime) -> pd.DataFrame:
        """
        Fetch M5 candles and convert index to Europe/London.

        Note:
        dukascopy_python.fetch expects naive datetimes interpreted as UTC.
        """
        df = duka.fetch(
            instrument,
            self.interval,
            self.side,
            start_utc.replace(tzinfo=None),  # naive but treated as UTC by this library
            end_utc.replace(tzinfo=None),
        )
        # Ensure tz-aware index, then convert to London time
        df.index = pd.to_datetime(df.index, utc=True).tz_convert(self.tz)
        return df.sort_index()

    @staticmethod
    def make_daily(df_m5: pd.DataFrame) -> pd.DataFrame:
        """
        Resample M5 candles into daily OHLC and add derived columns.
        """
        daily = df_m5.resample("1D").agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
        ).dropna()
        daily["mid"] = (daily["high"] + daily["low"]) / 2
        daily["range_pct"] = (daily["high"] - daily["low"]) / daily["close"]
        return daily

    @staticmethod
    def make_h1_atr(df_m5: pd.DataFrame, n: int = 14) -> pd.DataFrame:
        """
        Resample M5 candles into hourly OHLC and compute ATR(n) on H1.
        """
        h1 = df_m5.resample("1h").agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
        ).dropna()

        # True Range (TR)
        tr = pd.concat(
            [
                h1["high"] - h1["low"],
                (h1["high"] - h1["close"].shift()).abs(),
                (h1["low"] - h1["close"].shift()).abs(),
            ],
            axis=1,
        ).max(axis=1)

        h1["ATR14"] = tr.rolling(n).mean()
        return h1

    @staticmethod
    def add_intraday_tracking(df_m5: pd.DataFrame) -> pd.DataFrame:
        """
        Add intraday diagnostics on M5:
        - intraday_ret: close / day's first open - 1
        - day_high_sofar: running intraday high
        - day_low_sofar: running intraday low
        """
        df = df_m5.copy()
        df["date"] = df.index.date

        day_open = df.groupby("date")["open"].transform("first")
        df["intraday_ret"] = df["close"] / day_open - 1

        df["day_high_sofar"] = df.groupby("date")["high"].cummax()
        df["day_low_sofar"] = df.groupby("date")["low"].cummin()
        return df

    @staticmethod
    def export_csv(data: MetalData, prefix: str, out_dir: str | Path = "data") -> None:
        """
        Export the datasets to CSV files (human-readable, larger than parquet).
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        data.m5.to_csv(out_dir / f"{prefix}_M5_London.csv")
        data.daily.to_csv(out_dir / f"{prefix}_D1_London.csv")
        data.h1.to_csv(out_dir / f"{prefix}_H1_ATR14_London.csv")
        data.track.to_csv(out_dir / f"{prefix}_M5_TRACK_London.csv")


def load_metals_last_year(cache_dir: str | Path = "data", use_cache: bool = True) -> tuple[MetalData, MetalData]:
    """
    Convenience helper: load both XAUUSD and XAGUSD for the last year.
    """
    loader = DukascopyMetalsLoader(cache_dir=cache_dir)
    au = loader.load_last_year(inst.INSTRUMENT_FX_METALS_XAU_USD, "XAUUSD", use_cache=use_cache)
    ag = loader.load_last_year(inst.INSTRUMENT_FX_METALS_XAG_USD, "XAGUSD", use_cache=use_cache)
    return au, ag

def export_csv(metal_data, prefix="XAGUSD", out_dir="data"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metal_data.m5.to_csv(out_dir / f"{prefix}_M5.csv")
    metal_data.daily.to_csv(out_dir / f"{prefix}_D1.csv")
    metal_data.h1.to_csv(out_dir / f"{prefix}_H1_ATR.csv")

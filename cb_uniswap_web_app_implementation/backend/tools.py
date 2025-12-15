"""
Utility tool for a single-call workflow that:
- fetches OHLCV candles from Coinbase
- builds an mplfinance candlestick chart saved to disk
- computes resilience metrics around an event date

Intended for use by an LLM/tooling layer (e.g., Claude) as one callable entry
point. The primary entry is `fetch_resilience_snapshot`.
"""

from __future__ import annotations

import base64
import os
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import requests
import matplotlib

# Force non-GUI backend to avoid macOS NSWindow errors when running headless/threads
matplotlib.use("Agg")

import mplfinance as mpf
from web3 import Web3


# --------------------------------------------------------------------------------------
# Data structures
# --------------------------------------------------------------------------------------

@dataclass
class ResilienceMetrics:
    R_wick_body_ratio: float
    R_volatility: float
    pre_avg_wick_body: float
    post_avg_wick_body: float
    pre_avg_volatility: float
    post_avg_volatility: float
    pre_event_period: str
    post_event_period: str


# --------------------------------------------------------------------------------------
# Core calculation
# --------------------------------------------------------------------------------------

def _calculate_resilience_metric(
    df: pd.DataFrame,
    event_date: pd.Timestamp,
    pre_event_days: int = 7,
    post_event_days: int = 7,
) -> ResilienceMetrics:
    """Compute wick/body and volatility-based resilience metrics around an event."""
    pre_start = event_date - pd.Timedelta(days=pre_event_days)
    pre_end = event_date - pd.Timedelta(days=1)
    post_start = event_date
    post_end = event_date + pd.Timedelta(days=post_event_days)

    pre_event_data = df.loc[pre_start:pre_end].copy()
    post_event_data = df.loc[post_start:post_end].copy()

    def calc_wick_body_ratio(row: pd.Series) -> float:
        body = abs(row["Close"] - row["Open"])
        upper_wick = row["High"] - max(row["Open"], row["Close"])
        lower_wick = min(row["Open"], row["Close"]) - row["Low"]
        total_wick = upper_wick + lower_wick
        if body == 0:
            return np.inf if total_wick > 0 else 0
        return total_wick / body

    def calc_daily_volatility(row: pd.Series) -> float:
        mid_price = (row["High"] + row["Low"]) / 2
        if mid_price == 0:
            return 0
        return (row["High"] - row["Low"]) / mid_price

    pre_event_data["wick_body_ratio"] = pre_event_data.apply(calc_wick_body_ratio, axis=1)
    post_event_data["wick_body_ratio"] = post_event_data.apply(calc_wick_body_ratio, axis=1)
    pre_event_data["volatility"] = pre_event_data.apply(calc_daily_volatility, axis=1)
    post_event_data["volatility"] = post_event_data.apply(calc_daily_volatility, axis=1)

    pre_wick_body = pre_event_data["wick_body_ratio"].replace([np.inf, -np.inf], np.nan).dropna()
    post_wick_body = post_event_data["wick_body_ratio"].replace([np.inf, -np.inf], np.nan).dropna()

    pre_avg_wick_body = pre_wick_body.mean()
    post_avg_wick_body = post_wick_body.mean()
    pre_avg_volatility = pre_event_data["volatility"].mean()
    post_avg_volatility = post_event_data["volatility"].mean()

    R_wick_body = (
        ((post_avg_wick_body - pre_avg_wick_body) / pre_avg_wick_body) * 100
        if pd.notna(pre_avg_wick_body) and pre_avg_wick_body != 0
        else np.nan
    )
    R_volatility = (
        ((post_avg_volatility - pre_avg_volatility) / pre_avg_volatility) * 100
        if pd.notna(pre_avg_volatility) and pre_avg_volatility != 0
        else np.nan
    )

    return ResilienceMetrics(
        R_wick_body_ratio=R_wick_body,
        R_volatility=R_volatility,
        pre_avg_wick_body=pre_avg_wick_body,
        post_avg_wick_body=post_avg_wick_body,
        pre_avg_volatility=pre_avg_volatility,
        post_avg_volatility=post_avg_volatility,
        pre_event_period=f"{pre_start.date()} to {pre_end.date()}",
        post_event_period=f"{post_start.date()} to {post_end.date()}",
    )


# --------------------------------------------------------------------------------------
# Data fetching + charting
# --------------------------------------------------------------------------------------

def _fetch_coinbase_candles(
    product_id: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    granularity: int,
) -> pd.DataFrame:
    """Fetch OHLCV candles from Coinbase and return a prepared DataFrame."""
    url = f"https://api.exchange.coinbase.com/products/{product_id}/candles"
    params = {
        "start": start.isoformat(),
        "end": end.isoformat(),
        "granularity": granularity,
    }
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    data: List[List[float]] = resp.json()
    if not isinstance(data, list) or len(data) == 0:
        raise ValueError("No candle data returned from Coinbase.")

    df = pd.DataFrame(data, columns=["time", "low", "high", "open", "close", "volume"])
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.sort_values("time").set_index("time")
    df.rename(
        columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        },
        inplace=True,
    )
    return df


def _safe_filename(name: str) -> str:
    """Sanitize a title to a filesystem-safe filename."""
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
    safe = safe.strip("_")
    return safe or "chart"


def _build_chart(df: pd.DataFrame, title: str, output_dir: Path) -> Path:
    """Render and save an mplfinance candlestick chart."""
    output_dir.mkdir(parents=True, exist_ok=True)
    chart_path = output_dir / f"{_safe_filename(title)}.png"

    style = mpf.make_mpf_style(
        base_mpf_style="charles",
        rc={"axes.labelsize": 7, "xtick.labelsize": 6, "ytick.labelsize": 6},
    )

    mpf.plot(
        df,
        type="candle",
        style=style,
        figsize=(12, 6),
        volume=True,
        title=title,
        xlabel="Date",
        ylabel="Price (USD)",
        update_width_config={"candle_linewidth": 0.5, "candle_width": 0.4},
        savefig=str(chart_path),
    )
    return chart_path


def _encode_chart_base64(chart_path: Path) -> str:
    """Return base64-encoded PNG bytes for UI transport."""
    with chart_path.open("rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


# --------------------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------------------

def fetch_resilience_snapshot(
    product_id: str,
    start: str,
    end: str,
    event_date: str,
    granularity: int = 86_400,
    pre_event_days: int = 7,
    post_event_days: int = 7,
    title: str | None = None,
    output_dir: str | Path = "outputs",
    return_image_b64: bool = False,
) -> Dict[str, Any]:
    """
    Single-call tool: fetch Coinbase candles, save chart, compute resilience metrics.

    Parameters
    ----------
    product_id : str
        e.g., "DAI-USD"
    start : str
        ISO-8601 start datetime (UTC). Coinbase expects full timestamps.
    end : str
        ISO-8601 end datetime (UTC).
    event_date : str
        Event date (YYYY-MM-DD or ISO-8601) used for resilience windows.
    granularity : int
        Candle size in seconds (Coinbase supported granularities).
    pre_event_days / post_event_days : int
        Window sizes for baseline and post-event calculations.
    title : str | None
        Chart title; defaults to product_id if not provided.
    output_dir : str | Path
        Directory where chart will be saved.

    Returns
    -------
    Dict with:
        - chart_path (str)
        - record_count (int)
        - metrics (dict)
        - data_preview (list)  # first few rows as records
        - params_used (dict)
    """
    # Coerce inputs and fail fast with a clear error if they are invalid
    start_ts = pd.to_datetime(start, utc=True, errors="coerce")
    end_ts = pd.to_datetime(end, utc=True, errors="coerce")
    event_ts = pd.to_datetime(event_date, utc=True, errors="coerce")

    if any(ts is pd.NaT or pd.isna(ts) for ts in (start_ts, end_ts, event_ts)):
        raise ValueError("Invalid datetime values; please provide start/end/event_date in ISO-8601.")

    if end_ts <= start_ts:
        raise ValueError("`end` must be after `start`.")
    if granularity <= 0:
        raise ValueError("`granularity` must be positive.")

    df = _fetch_coinbase_candles(product_id, start_ts, end_ts, granularity)
    if df.empty:
        raise ValueError("No data returned for given parameters.")

    chart_title = title or f"{product_id} candles"
    chart_path = _build_chart(df, chart_title, Path(output_dir))
    chart_b64 = _encode_chart_base64(chart_path) if return_image_b64 else None

    metrics = _calculate_resilience_metric(
        df,
        event_ts.normalize(),
        pre_event_days=pre_event_days,
        post_event_days=post_event_days,
    )

    preview_rows: List[Dict[str, Any]] = (
        df.reset_index()
        .head(5)
        .assign(time=lambda x: x["time"].dt.strftime("%Y-%m-%d"))
        .to_dict(orient="records")
    )

    return {
        "chart_path": str(chart_path),
        "chart_base64": chart_b64,
        "record_count": len(df),
        "metrics": asdict(metrics),
        "data_preview": preview_rows,
        "params_used": {
            "product_id": product_id,
            "start": start_ts.isoformat(),
            "end": end_ts.isoformat(),
            "event_date": event_ts.date().isoformat(),
            "granularity": granularity,
            "pre_event_days": pre_event_days,
            "post_event_days": post_event_days,
            "title": chart_title,
            "output_dir": str(output_dir),
        },
    }


__all__ = ["fetch_resilience_snapshot"]


# --------------------------------------------------------------------------------------
# Uniswap (DEX) tool – fetches poolDayDatas, charts, and resilience metrics
# --------------------------------------------------------------------------------------


@dataclass
class DexResilienceMetrics:
    R_volatility: float
    R_intraday_range: float
    R_relative_range: float
    pre_avg_volatility: float
    post_avg_volatility: float
    pre_avg_range: float
    post_avg_range: float
    pre_avg_rel_range: float
    post_avg_rel_range: float
    max_volatility_day: pd.Timestamp
    max_volatility_value: float
    pre_event_period: str
    post_event_period: str


def _calculate_resilience_metric_dex(
    df: pd.DataFrame,
    event_date: pd.Timestamp,
    pre_event_days: int = 7,
    post_event_days: int = 7,
) -> DexResilienceMetrics:
    """DEX-friendly resilience metrics (volatility and ranges)."""
    pre_start = event_date - pd.Timedelta(days=pre_event_days)
    pre_end = event_date - pd.Timedelta(days=1)
    post_start = event_date
    post_end = event_date + pd.Timedelta(days=post_event_days)

    pre_event_data = df.loc[pre_start:pre_end].copy()
    post_event_data = df.loc[post_start:post_end].copy()

    def calc_daily_volatility(row: pd.Series) -> float:
        mid_price = (row["High"] + row["Low"]) / 2
        if mid_price == 0:
            return 0
        return (row["High"] - row["Low"]) / mid_price

    def calc_intraday_range(row: pd.Series) -> float:
        return row["High"] - row["Low"]

    def calc_relative_range(row: pd.Series) -> float:
        avg_price = (row["Open"] + row["Close"]) / 2
        if avg_price == 0:
            avg_price = (row["High"] + row["Low"]) / 2
        if avg_price == 0:
            return 0
        return (row["High"] - row["Low"]) / avg_price

    pre_event_data["volatility"] = pre_event_data.apply(calc_daily_volatility, axis=1)
    post_event_data["volatility"] = post_event_data.apply(calc_daily_volatility, axis=1)
    pre_event_data["intraday_range"] = pre_event_data.apply(calc_intraday_range, axis=1)
    post_event_data["intraday_range"] = post_event_data.apply(calc_intraday_range, axis=1)
    pre_event_data["relative_range"] = pre_event_data.apply(calc_relative_range, axis=1)
    post_event_data["relative_range"] = post_event_data.apply(calc_relative_range, axis=1)

    pre_avg_volatility = pre_event_data["volatility"].mean()
    post_avg_volatility = post_event_data["volatility"].mean()
    pre_avg_range = pre_event_data["intraday_range"].mean()
    post_avg_range = post_event_data["intraday_range"].mean()
    pre_avg_rel_range = pre_event_data["relative_range"].mean()
    post_avg_rel_range = post_event_data["relative_range"].mean()

    R_volatility = (
        ((post_avg_volatility - pre_avg_volatility) / pre_avg_volatility) * 100
        if pre_avg_volatility and pre_avg_volatility != 0
        else float("nan")
    )
    R_range = (
        ((post_avg_range - pre_avg_range) / pre_avg_range) * 100
        if pre_avg_range and pre_avg_range != 0
        else float("nan")
    )
    R_rel_range = (
        ((post_avg_rel_range - pre_avg_rel_range) / pre_avg_rel_range) * 100
        if pre_avg_rel_range and pre_avg_rel_range != 0
        else float("nan")
    )

    if post_event_data.empty:
        max_vol_day = pd.NaT
        max_vol_value = float("nan")
    else:
        max_vol_idx = post_event_data["volatility"].idxmax()
        max_vol_day = max_vol_idx
        max_vol_value = post_event_data.loc[max_vol_idx, "volatility"]

    return DexResilienceMetrics(
        R_volatility=R_volatility,
        R_intraday_range=R_range,
        R_relative_range=R_rel_range,
        pre_avg_volatility=pre_avg_volatility,
        post_avg_volatility=post_avg_volatility,
        pre_avg_range=pre_avg_range,
        post_avg_range=post_avg_range,
        pre_avg_rel_range=pre_avg_rel_range,
        post_avg_rel_range=post_avg_rel_range,
        max_volatility_day=max_vol_day,
        max_volatility_value=max_vol_value,
        pre_event_period=f"{pre_start.date()} to {pre_end.date()}",
        post_event_period=f"{post_start.date()} to {post_end.date()}",
    )


def _map_dates_to_blocks(csv_path: Path, start: pd.Timestamp, end: pd.Timestamp) -> tuple[int, int]:
    """Resolve start/end block numbers from CSV mapping."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Block mapping file not found: {csv_path}")
    block_data = pd.read_csv(csv_path)
    if "Datetime" not in block_data.columns or "Ethereum Block Number" not in block_data.columns:
        raise ValueError("CSV must contain 'Datetime' and 'Ethereum Block Number' columns.")
    # ensure timezone-aware to match the UTC timestamps used elsewhere
    block_data["Datetime"] = pd.to_datetime(block_data["Datetime"], utc=True)

    start_row = block_data[block_data["Datetime"] >= start]
    end_row = block_data[block_data["Datetime"] <= end]
    if start_row.empty or end_row.empty:
        raise ValueError("No block mapping found for provided date range.")

    start_block = int(start_row.iloc[0]["Ethereum Block Number"])
    end_block = int(end_row.iloc[-1]["Ethereum Block Number"])
    return start_block, end_block


def _fetch_uniswap_pool_days(
    api_key: str,
    subgraph_id: str,
    pool_address: str,
    start_ts: int,
    end_ts: int,
) -> pd.DataFrame:
    """Fetch poolDayDatas from The Graph for a Uniswap pool."""
    url = f"https://gateway.thegraph.com/api/{api_key}/subgraphs/id/{subgraph_id}"
    query = f"""
{{
  poolDayDatas(
    first: 1000
    orderBy: date
    orderDirection: asc
    where: {{
      pool: "{pool_address.lower()}",
      date_gte: {start_ts},
      date_lte: {end_ts}
    }}
  ) {{
    date
    open
    high
    low
    close
    volumeUSD
    tvlUSD
    liquidity
  }}
}}
"""
    resp = requests.post(url, json={"query": query}, timeout=20)
    resp.raise_for_status()
    payload = resp.json()
    if "data" not in payload or payload["data"].get("poolDayDatas") is None:
        raise ValueError("Unexpected response when fetching poolDayDatas.")
    data = payload["data"]["poolDayDatas"]
    if not data:
        raise ValueError("No poolDayDatas returned for given parameters.")

    df = pd.DataFrame(data)
    df["Date"] = pd.to_datetime(df["date"], unit="s", utc=True)
    df.drop(columns=["date"], inplace=True)
    df.set_index("Date", inplace=True)
    df = df.astype(float)
    df.rename(
        columns={
            "volumeUSD": "Volume",
            "tvlUSD": "TVL Available (USD)",
            "liquidity": "Liquidity",
            "open": "Open",
            "close": "Close",
            "high": "High",
            "low": "Low",
        },
        inplace=True,
    )
    return df


def _build_uniswap_chart(df: pd.DataFrame, title: str, output_dir: Path) -> Path:
    """Render and save candlesticks + TVL subplot for Uniswap data."""
    output_dir.mkdir(parents=True, exist_ok=True)
    chart_path = output_dir / f"{_safe_filename(title)}_uniswap.png"

    style = mpf.make_mpf_style(
        base_mpf_style="charles",
        rc={"axes.labelsize": 7, "xtick.labelsize": 6, "ytick.labelsize": 6},
    )

    tvl_plot = mpf.make_addplot(df["TVL Available (USD)"], panel=2, color="blue")

    mpf.plot(
        df,
        type="candle",
        style=style,
        figsize=(12, 6),
        volume=True,
        title=title,
        xlabel="Date",
        addplot=[tvl_plot],
        panel_ratios=(4, 1),
        update_width_config={"candle_linewidth": 0.5, "candle_width": 0.4},
        savefig=str(chart_path),
    )
    return chart_path


def fetch_uniswap_resilience_snapshot(
    start: str,
    end: str,
    event_date: str,
    *,
    pool_address: str | None = None,
    csv_path: str | Path = "datetime_to_eth_block_number (3).csv",
    granularity_days: int = 1,
    pre_event_days: int = 7,
    post_event_days: int = 7,
    title: str | None = None,
    output_dir: str | Path = "outputs",
    return_image_b64: bool = False,
) -> Dict[str, Any]:
    """
    Single-call tool: fetch Uniswap poolDayDatas, save chart, compute resilience metrics.

    Env vars (override defaults):
      - UNISWAP_API_KEY
      - UNISWAP_SUBGRAPH_ID (falls back to SUBGRAPH_ID for compatibility)
      - UNISWAP_POOL_ADDRESS
    """
    api_key = os.getenv("UNISWAP_API_KEY")
    subgraph_id = os.getenv("UNISWAP_SUBGRAPH_ID") or os.getenv("SUBGRAPH_ID")
    default_pool = os.getenv("UNISWAP_POOL_ADDRESS", "0x5777d92f208679db4b9778590fa3cab3ac9e2168")

    if not api_key:
        raise EnvironmentError("UNISWAP_API_KEY is required.")
    if not subgraph_id:
        raise EnvironmentError("UNISWAP_SUBGRAPH_ID is required.")

    pool_address = (pool_address or default_pool).lower()

    # Coerce datetime inputs and validate before continuing
    start_ts = pd.to_datetime(start, utc=True, errors="coerce")
    end_ts = pd.to_datetime(end, utc=True, errors="coerce")
    event_ts = pd.to_datetime(event_date, utc=True, errors="coerce")

    if any(ts is pd.NaT or pd.isna(ts) for ts in (start_ts, end_ts, event_ts)):
        raise ValueError("Invalid datetime values; please provide start/end/event_date in ISO-8601.")

    if end_ts <= start_ts:
        raise ValueError("`end` must be after `start`.")

    csv_path = Path(csv_path)
    start_block, end_block = _map_dates_to_blocks(csv_path, start_ts.normalize(), end_ts.normalize())

    # Graph uses timestamps; blocks are informational; granularity_days retained for symmetry.
    df = _fetch_uniswap_pool_days(
        api_key=api_key,
        subgraph_id=subgraph_id,
        pool_address=pool_address,
        start_ts=int(start_ts.timestamp()),
        end_ts=int(end_ts.timestamp()),
    )

    if df.empty:
        raise ValueError("No data returned for given parameters.")

    chart_title = title or "Uniswap poolDayDatas"
    chart_path = _build_uniswap_chart(df, chart_title, Path(output_dir))
    chart_b64 = _encode_chart_base64(chart_path) if return_image_b64 else None

    metrics = _calculate_resilience_metric_dex(
        df,
        event_ts.normalize(),
        pre_event_days=pre_event_days,
        post_event_days=post_event_days,
    )

    preview_rows: List[Dict[str, Any]] = (
        df.reset_index()
        .head(5)
        .assign(Date=lambda x: x["Date"].dt.strftime("%Y-%m-%d"))
        .to_dict(orient="records")
    )

    return {
        "chart_path": str(chart_path),
        "chart_base64": chart_b64,
        "record_count": len(df),
        "metrics": asdict(metrics),
        "data_preview": preview_rows,
        "params_used": {
            "start": start_ts.isoformat(),
            "end": end_ts.isoformat(),
            "event_date": event_ts.date().isoformat(),
            "granularity_days": granularity_days,
            "pre_event_days": pre_event_days,
            "post_event_days": post_event_days,
            "title": chart_title,
            "output_dir": str(output_dir),
            "pool_address": pool_address,
            "start_block": start_block,
            "end_block": end_block,
            "subgraph_id": subgraph_id,
        },
    }


__all__ = ["fetch_resilience_snapshot", "fetch_uniswap_resilience_snapshot"]


# --------------------------------------------------------------------------------------
# Uniswap V3 pool discovery tool
# --------------------------------------------------------------------------------------

FACTORY_ADDRESS = "0x1F98431c8aD98523631AE4a59f267346ea31F984"
FACTORY_ABI = [
    {
        "inputs": [
            {"internalType": "address", "name": "tokenA", "type": "address"},
            {"internalType": "address", "name": "tokenB", "type": "address"},
            {"internalType": "uint24", "name": "fee", "type": "uint24"},
        ],
        "name": "getPool",
        "outputs": [{"internalType": "address", "name": "pool", "type": "address"}],
        "stateMutability": "view",
        "type": "function",
    }
]

FEE_TIERS_BPS = {
    "0.01%": 100,
    "0.05%": 500,
    "0.3%": 3_000,
    "1%": 10_000,
}


def _build_web3_from_env() -> Web3:
    api_key = os.getenv("ALCHEMY_API_KEY")
    if not api_key:
        raise EnvironmentError("ALCHEMY_API_KEY is required (set in .env).")
    # modern Alchemy base URL
    provider = Web3.HTTPProvider(f"https://eth-mainnet.g.alchemy.com/v2/{api_key}")
    w3 = Web3(provider)
    if not w3.is_connected():
        raise ConnectionError("Could not connect to Web3 provider; check API key/connectivity.")
    return w3


def get_uniswap_v3_pool_address(
    token0_address: str,
    token1_address: str,
    fee_bps: Optional[int] = None,
    fee_label: Optional[str] = None,
) -> Optional[str]:
    """
    Resolve the Uniswap V3 pool address for two tokens and a fee tier.

    Fee can be provided as:
      - fee_bps (int, e.g., 3000)
      - fee_label (str, one of 0.01%, 0.05%, 0.3%, 1%)
    Returns None if the pool does not exist.
    """
    if fee_bps is None and fee_label is None:
        fee_bps = 3_000  # default 0.3%
    if fee_label:
        fee_bps = FEE_TIERS_BPS.get(fee_label)
    if fee_bps is None:
        raise ValueError("Invalid fee tier; provide fee_bps or a known fee_label.")

    w3 = _build_web3_from_env()
    factory_contract = w3.eth.contract(address=FACTORY_ADDRESS, abi=FACTORY_ABI)

    # deterministic ordering
    t0 = Web3.to_checksum_address(token0_address)
    t1 = Web3.to_checksum_address(token1_address)
    tokenA, tokenB = sorted([t0, t1])

    pool_address = factory_contract.functions.getPool(tokenA, tokenB, fee_bps).call()
    if pool_address == Web3.to_checksum_address("0x0000000000000000000000000000000000000000"):
        return None
    return pool_address


__all__ = [
    "fetch_resilience_snapshot",
    "fetch_uniswap_resilience_snapshot",
    "get_uniswap_v3_pool_address",
]


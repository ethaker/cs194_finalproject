"""
Lightweight manual smoke tests for tools in tools.py.

Usage:
    python test_tools.py

Notes:
- Requires network access and the relevant API keys in your environment:
    ALCHEMY_API_KEY, UNISWAP_API_KEY, UNISWAP_SUBGRAPH_ID, UNISWAP_POOL_ADDRESS (optional)
- Uses the local CSV mapping file for block lookups:
    datetime_to_eth_block_number (3).csv
- Tests are opt-in via flags below to avoid accidental remote calls.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Ensure backend package is importable
import sys
sys.path.append(str(Path(__file__).resolve().parent / "backend"))

from tools import (  # type: ignore
    fetch_resilience_snapshot,
    fetch_uniswap_resilience_snapshot,
    get_uniswap_v3_pool_address,
)


# Toggle these to run live network tests
RUN_COINBASE_TEST = False
RUN_UNISWAP_TEST = True
RUN_POOL_DISCOVERY_TEST = False


def test_coinbase():
    """Smoke test for Coinbase fetch + resilience metrics."""
    result = fetch_resilience_snapshot(
        product_id="DAI-USD",
        start="2022-10-05T00:00:00Z",
        end="2022-10-19T00:00:00Z",
        event_date="2022-10-13",
        granularity=86_400,
        title="DAI-USD smoke",
        output_dir="outputs",
        return_image_b64=False,
    )
    print("Coinbase result keys:", result.keys())
    print("Metrics:", result["metrics"])
    print("Chart:", result["chart_path"])


def test_uniswap():
    """Smoke test for Uniswap poolDayDatas + resilience metrics."""
    csv_path = Path("datetime_to_eth_block_number (3).csv")
    result = fetch_uniswap_resilience_snapshot(
        start="2022-10-05",
        end="2022-10-19",
        event_date="2022-10-13",
        title="Uniswap: DAI/USDC",
        csv_path=csv_path,
        pre_event_days=7,
        post_event_days=7,
        return_image_b64=False,
    )
    print("Uniswap result keys:", result.keys())
    print("Metrics:", result["metrics"])
    print("Chart:", result["chart_path"])
    print("Start/End blocks:", result["params_used"]["start_block"], result["params_used"]["end_block"])


def test_pool_discovery():
    """Smoke test for pool address resolution."""
    pool = get_uniswap_v3_pool_address(
        token0_address="0x6B175474E89094C44Da98b954EedeAC495271d0F",  # DAI
        token1_address="0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",  # USDC
        fee_label="0.01%",
    )
    print("Resolved DAI/USDC 0.01% pool:", pool)


if __name__ == "__main__":
    # Load .env if present so UNISWAP/ALCHEMY keys are picked up.
    load_dotenv()

    # Basic sanity check for required env keys before running network calls.
    missing = []
    for key in ["ALCHEMY_API_KEY", "UNISWAP_API_KEY", "UNISWAP_SUBGRAPH_ID"]:
        if not os.getenv(key):
            missing.append(key)
    if missing:
        print("Warning: missing env vars (tests may fail):", ", ".join(missing))

    if RUN_COINBASE_TEST:
        print("\n--- Running Coinbase smoke test ---")
        test_coinbase()

    if RUN_UNISWAP_TEST:
        print("\n--- Running Uniswap smoke test ---")
        test_uniswap()

    if RUN_POOL_DISCOVERY_TEST:
        print("\n--- Running pool discovery smoke test ---")
        test_pool_discovery()

    if not any([RUN_COINBASE_TEST, RUN_UNISWAP_TEST, RUN_POOL_DISCOVERY_TEST]):
        print("Tests are disabled by default. Set the RUN_* flags to True to execute.")


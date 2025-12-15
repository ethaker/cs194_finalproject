"""
Lightweight Flask API that exposes:
- /api/chat : LLM chatbot endpoint with tool use (Coinbase + Uniswap)
- /api/tools/coinbase : direct Coinbase resilience snapshot
- /api/tools/uniswap : direct Uniswap resilience snapshot
- /api/tools/pool-address : resolve Uniswap v3 pool address

Environment:
    ANTHROPIC_API_KEY   (required)
    UNISWAP_API_KEY     (required for Uniswap endpoints)
    UNISWAP_SUBGRAPH_ID (required for Uniswap endpoints)
    ALCHEMY_API_KEY     (required for pool resolution)
    ANTHROPIC_MODEL     (optional override; default set below)
"""

from __future__ import annotations

import json
import os
from datetime import datetime, date
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List

from anthropic import Anthropic, APIStatusError
from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_from_directory
from web3 import Web3
# Optional CORS for local dev; ignore import error if flask-cors not installed
try:
    from flask_cors import CORS
except ImportError:  # pragma: no cover
    CORS = None

from backend.tools import (
    fetch_resilience_snapshot,
    fetch_uniswap_resilience_snapshot,
    get_uniswap_v3_pool_address,
)

# Common ERC20 addresses (canonical mainnet) to help normalize LLM inputs
TOKEN_SYMBOLS = {
    "DAI": "0x6b175474e89094c44da98b954eedeac495271d0f",
    "USDC": "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48",
    "USDT": "0xdac17f958d2ee523a2206206994597c13d831ec7",
    "WETH": "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
    "WBTC": "0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599",
}

# Minimal JSON Schemas for Anthropic tool definitions (keep permissive to avoid validation errors)
TOOL_SCHEMAS = {
    "fetch_resilience_snapshot": {
        "type": "object",
        "properties": {
            "product_id": {"type": "string"},
            "start": {"type": "string"},
            "end": {"type": "string"},
            "event_date": {"type": "string"},
        },
        "additionalProperties": True,
    },
    "fetch_uniswap_resilience_snapshot": {
        "type": "object",
        "properties": {
            "start": {"type": "string"},
            "end": {"type": "string"},
            "event_date": {"type": "string"},
            "pool_address": {"type": "string"},
        },
        "additionalProperties": True,
    },
    "get_uniswap_v3_pool_address": {
        "type": "object",
        "properties": {
            "token0_address": {"type": "string"},
            "token1_address": {"type": "string"},
            "fee_bps": {"type": "integer"},
            "fee_label": {"type": "string"},
        },
        "additionalProperties": True,
    },
}

# Load environment variables early
load_dotenv()

MODEL = os.getenv("ANTHROPIC_MODEL", "claude-opus-4-5-20251101")
MAX_TOOL_CALLS = 5
BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"
OUTPUTS_DIR = BASE_DIR / "outputs"

SYSTEM_PROMPT = """
You are a research analyst bot working with the following objectives:
- Quantify and compare differences in effective spread / price impact (slippage) between Coinbase and Uniswap v3 trade sizes before, during, and after market shock events.
- Evaluate speed of price discovery and volatility asymmetry using candlestick analysis (wick/body ratios).
- Test whether decentralized LPs on v3 show less sticky behavior (e.g., faster withdrawal, greater concentration risk, capital inactive outside position range) than CeFi MMs during stress, leading to different spread reactions (R metric).

Rules:
- Strict limit: at most three tool calls total (fetch_resilience_snapshot, fetch_uniswap_resilience_snapshot, get_uniswap_v3_pool_address) per user request.
- Purpose of each tool: (1) get_uniswap_v3_pool_address → resolve the pool address; (2) fetch_uniswap_resilience_snapshot → fetch Uniswap data/metrics; (3) fetch_resilience_snapshot → fetch Coinbase data/metrics. Use each at most once and in that order when all are needed. If you cannot resolve the pool address, stop and return an error instead of continuing.
- Only use the listed tools from backend.tools; do NOT attempt any other tools or libraries (e.g., pandas plotting helpers). If inputs are insufficient, return a concise error instead of inventing other calls.
- Event date is a single calendar day; align pre/post windows to user intent.
- Ask concise follow-ups if required parameters are missing.
- Prefer resolving Uniswap pool address when token pairs are provided; otherwise use configured/default pool.
- Provide concise natural-language answers noting assumptions on fee tiers, granularity, and windows.
"""


def json_serial(obj: Any) -> Any:
    """JSON serializer for non-serializable objects."""
    try:
        import pandas as pd  # type: ignore
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        if obj is pd.NaT:
            return None
    except Exception:
        pass
    try:
        import numpy as np  # type: ignore
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
    except Exception:
        pass
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _parse_payload(
    expected_ints: List[str] | None = None,
    expected_bools: List[str] | None = None,
) -> Dict[str, Any]:
    """
    Accept JSON body (POST) or query params (GET) and coerce known types.

    - expected_ints: keys to convert to int when present
    - expected_bools: keys to convert to bool (truthy strings: "true","1","yes","on")
    """
    expected_ints = expected_ints or []
    expected_bools = expected_bools or []

    payload: Dict[str, Any]
    if request.method == "GET":
        payload = request.args.to_dict(flat=True)
    else:
        payload = request.get_json(force=True, silent=True) or {}

    for key in expected_ints:
        if key in payload:
            try:
                payload[key] = int(payload[key])
            except Exception:
                raise ValueError(f"Invalid int for '{key}'")

    for key in expected_bools:
        if key in payload:
            val = payload[key]
            if isinstance(val, bool):
                payload[key] = val
            else:
                payload[key] = str(val).strip().lower() in {"1", "true", "yes", "on"}
    return payload


def _normalize_tool_args(name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize common alias keys from LLMs to our function signatures."""
    if not isinstance(args, dict):
        return {}
    norm = dict(args)

    # General aliases
    if "eventDate" in norm and "event_date" not in norm:
        norm["event_date"] = norm.pop("eventDate")
    if "pair" in norm and "product_id" not in norm:
        norm["product_id"] = norm.pop("pair")

    # Pool resolution aliases
    if name == "get_uniswap_v3_pool_address":
        if "token0" in norm and "token0_address" not in norm:
            norm["token0_address"] = norm.pop("token0")
        if "token1" in norm and "token1_address" not in norm:
            norm["token1_address"] = norm.pop("token1")
        if "feeTier" in norm and "fee_bps" not in norm:
            norm["fee_bps"] = norm.pop("feeTier")

    # Uniswap fetch aliases
    if name == "fetch_uniswap_resilience_snapshot":
        if "token0" in norm and "token0_address" not in norm:
            norm["token0_address"] = norm.pop("token0")
        if "token1" in norm and "token1_address" not in norm:
            norm["token1_address"] = norm.pop("token1")
        if "feeTier" in norm and "fee_bps" not in norm:
            norm["fee_bps"] = norm.pop("feeTier")

    # Symbol → address resolution for pool tools
    for key in ("token0_address", "token1_address"):
        if key in norm:
            val = norm[key]
            if isinstance(val, str) and not val.startswith("0x") and val.upper() in TOKEN_SYMBOLS:
                norm[key] = TOKEN_SYMBOLS[val.upper()]

    # Interpret common fee typos: map 1→100 (0.01%), 5→500 (0.05%) if value is small
    if "fee_bps" in norm and isinstance(norm["fee_bps"], int):
        if norm["fee_bps"] in {1, 5}:
            norm["fee_bps"] = norm["fee_bps"] * 100

    # Handle shorthand pool identifiers
    if "pool_address" in norm:
        val = norm["pool_address"]
        if isinstance(val, str) and val.upper().replace("-", "/") in {"DAI/USDC"}:
            norm["pool_address"] = os.getenv(
                "UNISWAP_POOL_ADDRESS", "0x5777d92f208679db4b9778590fa3cab3ac9e2168"
            )

    # Auto-fill start/end based on event_date and pre/post days when missing
    def _auto_fill_range(target: Dict[str, Any]) -> None:
        if "event_date" in target and ("start" not in target or "end" not in target):
            try:
                event_dt = datetime.fromisoformat(str(target["event_date"]).split("T")[0])
                pre_days = int(target.get("pre_event_days", 7))
                post_days = int(target.get("post_event_days", 7))
                start_dt = event_dt - timedelta(days=pre_days)
                end_dt = event_dt + timedelta(days=post_days)
                target.setdefault("start", start_dt.isoformat() + "Z")
                target.setdefault("end", end_dt.isoformat() + "Z")
            except Exception:
                pass

    if name == "fetch_resilience_snapshot":
        _auto_fill_range(norm)
    if name == "fetch_uniswap_resilience_snapshot":
        _auto_fill_range(norm)

    return norm


def run_tool(name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """Dispatch to backend.tools functions with lightweight logging."""
    normalized_args = _normalize_tool_args(name, args)
    try:
        print(f"[tool] calling {name} args={normalized_args}")  # simple observability
    except Exception:
        pass

    def _is_hex_address(val: Any) -> bool:
        return isinstance(val, str) and Web3.is_address(val)

    try:
        if name == "fetch_resilience_snapshot":
            return fetch_resilience_snapshot(**normalized_args)  # type: ignore[arg-type]
        if name == "fetch_uniswap_resilience_snapshot":
            return fetch_uniswap_resilience_snapshot(**normalized_args)  # type: ignore[arg-type]
        if name == "get_uniswap_v3_pool_address":
            # Guard against malformed addresses early to avoid crashing inside web3
            t0 = normalized_args.get("token0_address")
            t1 = normalized_args.get("token1_address")
            if not (_is_hex_address(t0) and _is_hex_address(t1)):
                return {
                    "error": "Invalid token address format. Provide 0x-prefixed 42-char hex strings for token0_address and token1_address.",
                    "received": {"token0_address": t0, "token1_address": t1},
                    "hint": "Use symbols DAI/USDC/USDT/WETH/WBTC or valid addresses.",
                }
            pool_addr = get_uniswap_v3_pool_address(**normalized_args)  # type: ignore[arg-type]
            return {"pool_address": pool_addr}
            return {"error": f"Unknown tool: {name}"}
    except Exception as exc:
        import traceback
        err = {"error": str(exc), "traceback": traceback.format_exc()}
        try:
            print(f"[tool] {name} failed: {err}")
        except Exception:
            pass
        return err


def call_claude_with_tools(user_text: str) -> Dict[str, Any]:
    """Single-turn chat that allows Claude to call tools up to MAX_TOOL_CALLS."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return {"error": "ANTHROPIC_API_KEY is required."}

    client = Anthropic(api_key=api_key)
    messages: List[Dict[str, Any]] = [
        {"role": "user", "content": [{"type": "text", "text": user_text}]}
    ]

    tool_calls_used = 0
    tool_call_log = []

    while True:
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=1200,
                tools=[
                    {
                        "name": "fetch_resilience_snapshot",
                        "description": "Fetch Coinbase OHLCV candles and resilience metrics.",
                        "input_schema": TOOL_SCHEMAS["fetch_resilience_snapshot"],
                    },
                    {
                        "name": "fetch_uniswap_resilience_snapshot",
                        "description": "Fetch Uniswap poolDayDatas and resilience metrics.",
                        "input_schema": TOOL_SCHEMAS["fetch_uniswap_resilience_snapshot"],
                    },
                    {
                        "name": "get_uniswap_v3_pool_address",
                        "description": "Resolve Uniswap v3 pool address for token pair and fee.",
                        "input_schema": TOOL_SCHEMAS["get_uniswap_v3_pool_address"],
                    },
                ],
                system=SYSTEM_PROMPT,
                messages=messages,
            )
        except APIStatusError as api_err:
            return {"error": f"Anthropic API error: {api_err}"}

        messages.append({"role": "assistant", "content": response.content})

        tool_uses = []
        for block in response.content:
            btype = getattr(block, "type", None) if not isinstance(block, dict) else block.get("type")
            if btype == "tool_use":
                tool_uses.append(block)

        # No tool calls → return text
        if not tool_uses:
            text_parts = []
            for block in response.content:
                btype = getattr(block, "type", None) if not isinstance(block, dict) else block.get("type")
                if btype == "text":
                    text = getattr(block, "text", None) if not isinstance(block, dict) else block.get("text")
                    if text:
                        text_parts.append(text)
            return {"reply": "\n".join(text_parts), "tool_calls": tool_call_log}

        # Enforce tool call limit
        if tool_calls_used + len(tool_uses) > MAX_TOOL_CALLS:
            return {
                "error": f"Tool call limit exceeded ({MAX_TOOL_CALLS}).",
                "tool_calls": tool_call_log,
            }

        tool_results = []
        for use in tool_uses:
            use_name = use.get("name") if isinstance(use, dict) else getattr(use, "name")
            use_input = use.get("input") if isinstance(use, dict) else getattr(use, "input")
            use_id = use.get("id") if isinstance(use, dict) else getattr(use, "id")

            result = run_tool(use_name, use_input)
            # Store concise summary of result (avoid huge payloads)
            result_summary = {}
            if "error" in result:
                result_summary["error"] = result["error"]
            else:
                # Extract key metrics/summaries
                if "metrics" in result:
                    result_summary["metrics"] = result["metrics"]
                if "pool_address" in result:
                    result_summary["pool_address"] = result["pool_address"]
                if "record_count" in result:
                    result_summary["record_count"] = result["record_count"]
                if "chart_path" in result:
                    result_summary["chart_path"] = result["chart_path"]
                # Include a few other useful keys
                for key in ["params_used", "data_preview"]:
                    if key in result:
                        result_summary[key] = result[key]
            tool_call_log.append({
                "tool": use_name,
                "args": use_input,
                "result": result_summary,
            })
            tool_calls_used += 1

            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": use_id,
                    "content": json.dumps(result, default=json_serial),
                }
            )

        messages.append({"role": "user", "content": tool_results})
        # loop continues to let Claude consume tool results


# Flask app
app = Flask(__name__)
if CORS:
    CORS(app)


@app.after_request
def add_cors_headers(response):
    # Fallback CORS headers if flask-cors is unavailable or not configured
    response.headers.setdefault("Access-Control-Allow-Origin", "*")
    response.headers.setdefault(
        "Access-Control-Allow-Headers", "Content-Type, Authorization"
    )
    response.headers.setdefault(
        "Access-Control-Allow-Methods", "GET, POST, OPTIONS"
    )
    return response


@app.route("/api/chat", methods=["POST"])
def api_chat():
    payload = request.get_json(force=True, silent=True) or {}
    user_text = payload.get("message") or ""
    if not user_text.strip():
        return jsonify({"error": "message is required"}), 400
    result = call_claude_with_tools(user_text.strip())
    status = 200 if "error" not in result else 500
    return jsonify(result), status


@app.route("/api/tools/coinbase", methods=["POST", "GET"])
def api_coinbase():
    try:
        payload = _parse_payload(
            expected_ints=["granularity", "pre_event_days", "post_event_days"],
            expected_bools=["return_image_b64"],
        )
        result = fetch_resilience_snapshot(**payload)
        return jsonify(result), 200
    except Exception as exc:
        import traceback

        return jsonify({"error": str(exc), "traceback": traceback.format_exc()}), 400


@app.route("/api/tools/uniswap", methods=["POST", "GET"])
def api_uniswap():
    try:
        payload = _parse_payload(
            expected_ints=["granularity_days", "pre_event_days", "post_event_days"],
            expected_bools=["return_image_b64"],
        )
        result = fetch_uniswap_resilience_snapshot(**payload)
        return jsonify(result), 200
    except Exception as exc:
        import traceback

        return jsonify({"error": str(exc), "traceback": traceback.format_exc()}), 400


@app.route("/api/tools/coinbase/metrics", methods=["POST", "GET"])
def api_coinbase_metrics():
    """Lightweight endpoint returning only metrics + chart refs."""
    try:
        payload = _parse_payload(
            expected_ints=["granularity", "pre_event_days", "post_event_days"],
            expected_bools=["return_image_b64"],
        )
        result = fetch_resilience_snapshot(**payload)
        return (
            jsonify(
                {
                    "metrics": result.get("metrics"),
                    "chart_path": result.get("chart_path"),
                    "chart_base64": result.get("chart_base64"),
                    "record_count": result.get("record_count"),
                    "params_used": result.get("params_used"),
                }
            ),
            200,
        )
    except Exception as exc:
        import traceback

        return jsonify({"error": str(exc), "traceback": traceback.format_exc()}), 400


@app.route("/api/tools/uniswap/metrics", methods=["POST", "GET"])
def api_uniswap_metrics():
    """Lightweight endpoint returning only metrics + chart refs."""
    try:
        payload = _parse_payload(
            expected_ints=["granularity_days", "pre_event_days", "post_event_days"],
            expected_bools=["return_image_b64"],
        )
        result = fetch_uniswap_resilience_snapshot(**payload)
        return (
            jsonify(
                {
                    "metrics": result.get("metrics"),
                    "chart_path": result.get("chart_path"),
                    "chart_base64": result.get("chart_base64"),
                    "record_count": result.get("record_count"),
                    "params_used": result.get("params_used"),
                }
            ),
            200,
        )
    except Exception as exc:
        import traceback

        return jsonify({"error": str(exc), "traceback": traceback.format_exc()}), 400


@app.route("/api/tools/pool-address", methods=["POST", "GET"])
def api_pool_address():
    try:
        payload = _parse_payload(expected_ints=["fee_bps"])
        pool = get_uniswap_v3_pool_address(**payload)
        return jsonify({"pool_address": pool}), 200
    except Exception as exc:
        import traceback

        return jsonify({"error": str(exc), "traceback": traceback.format_exc()}), 400


@app.route("/outputs/<path:filename>")
def serve_output_file(filename: str):
    """Serve generated chart images from the outputs directory."""
    return send_from_directory(OUTPUTS_DIR, filename)


@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_frontend(path: str):
    """
    Serve the single-page frontend.

    Any unknown path falls back to index.html so that static assets load without
    additional routing.
    """
    index_path = FRONTEND_DIR / "index.html"
    if path and (FRONTEND_DIR / path).exists():
        return send_from_directory(FRONTEND_DIR, path)
    return send_from_directory(index_path.parent, index_path.name)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8000)), debug=True)


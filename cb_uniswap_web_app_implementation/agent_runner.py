from __future__ import annotations

import json
import os
import sys
import signal
import traceback
from datetime import datetime, date
from pathlib import Path
from typing import Any, Dict, List

from anthropic import Anthropic, APIStatusError
from dotenv import load_dotenv

# --- TIMEOUT MECHANISM ---
class TimeoutError(Exception):
    """Custom exception raised when the LLM processing exceeds the time limit."""
    pass

def timeout_handler(signum, frame):
    """Signal handler for SIGALRM to raise TimeoutError."""
    raise TimeoutError("LLM response and tool execution exceeded the 2-minute time limit.")

# Set up the signal handler once globally
signal.signal(signal.SIGALRM, timeout_handler)
# -------------------------


# Ensure backend package importable
ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT / "backend"))

from tools import (  # type: ignore
    fetch_resilience_snapshot,
    fetch_uniswap_resilience_snapshot,
    get_uniswap_v3_pool_address,
)


SYSTEM_PROMPT = """
You are a research analyst bot working with the following objectives:
- Quantify and compare differences in effective spread / price impact (slippage) between Coinbase and Uniswap v3 trade sizes before, during, and after market shock events.
- Evaluate speed of price discovery and volatility asymmetry using candlestick analysis (wick/body ratios).
- Test whether decentralized LPs on v3 show less sticky behavior (e.g., faster withdrawal, greater concentration risk, capital inactive outside position range) than CeFi MMs during stress, leading to different spread reactions (R metric).

Rules:
- **STRICT CONSTRAINT:** You can make a maximum of three total tool calls (fetch_resilience_snapshot, fetch_uniswap_resilience_snapshot, get_uniswap_v3_pool_address combined) across all turns of a single user request. After the third call, or if no tools are needed, you must provide the concise natural-language analysis answer.
- The event date is always a single calendar day; keep pre/post windows aligned to user intent.
- If the user omits required parameters (product id, pool, date, start/end window), ask concise follow-ups.
- Prefer first resolving the Uniswap pool address when the user gives token pairs; otherwise default to configured pool.
- Provide concise natural-language answers summarizing key metrics, chart paths, and any notable asymmetries.
- Show caution about data freshness and note assumptions on fee tiers, granularity, and window sizes.
- On Coinbase, for the second token, it should always be USD, not USDC. Example: "product_id": "ETH-USD"
"""

# Tool schemas exposed to Claude
TOOLS: List[Dict[str, Any]] = [
    {
        "name": "fetch_resilience_snapshot",
        "description": "Fetch Coinbase OHLCV candles, build chart, and compute wick/body + volatility resilience metrics.",
        "input_schema": {
            "type": "object",
            "properties": {
                "product_id": {"type": "string", "description": "Coinbase product, e.g., DAI-USD"},
                "start": {"type": "string", "description": "ISO-8601 start datetime (UTC)"},
                "end": {"type": "string", "description": "ISO-8601 end datetime (UTC)"},
                "event_date": {"type": "string", "description": "Event date (YYYY-MM-DD)"},
                "granularity": {"type": "integer", "description": "Candle size seconds, default 86400"},
                "pre_event_days": {"type": "integer", "description": "Baseline window days"},
                "post_event_days": {"type": "integer", "description": "Post-event window days"},
                "title": {"type": "string"},
                "output_dir": {"type": "string"},
                "return_image_b64": {"type": "boolean"},
            },
            "required": ["product_id", "start", "end", "event_date"],
        },
    },
    {
        "name": "fetch_uniswap_resilience_snapshot",
        "description": "Fetch Uniswap poolDayDatas, build chart, and compute DEX-focused resilience metrics.",
        "input_schema": {
            "type": "object",
            "properties": {
                "start": {"type": "string", "description": "Start date (YYYY-MM-DD or ISO)"},
                "end": {"type": "string", "description": "End date (YYYY-MM-DD or ISO)"},
                "event_date": {"type": "string", "description": "Event date (YYYY-MM-DD)"},
                "pool_address": {"type": "string", "description": "Uniswap v3 pool address"},
                "csv_path": {"type": "string", "description": "Block mapping CSV path"},
                "granularity_days": {"type": "integer"},
                "pre_event_days": {"type": "integer"},
                "post_event_days": {"type": "integer"},
                "title": {"type": "string"},
                "output_dir": {"type": "string"},
                "return_image_b64": {"type": "boolean"},
            },
            "required": ["start", "end", "event_date"],
        },
    },
    {
        "name": "get_uniswap_v3_pool_address",
        "description": "Resolve Uniswap v3 pool address for two token addresses and fee tier.",
        "input_schema": {
            "type": "object",
            "properties": {
                "token0_address": {"type": "string"},
                "token1_address": {"type": "string"},
                "fee_bps": {"type": "integer", "description": "Fee tier in bps (e.g., 3000)"},
                "fee_label": {"type": "string", "description": "One of 0.01%, 0.05%, 0.3%, 1%"},
            },
            "required": ["token0_address", "token1_address"],
        },
    },
]

# Default Anthropic model (3.5 Sonnet is deprecated). Allow override via env.
MODEL = os.getenv("ANTHROPIC_MODEL", "claude-opus-4-5-20251101")

# Global counter to enforce the tool call limit (not strictly necessary with the prompt change,
# but provides a hard guardrail in the execution logic).
MAX_TOOL_CALLS = 3
tool_call_counter = 0


def json_serial(obj: Any) -> str:
    """JSON serializer for objects not serializable by default json code."""
    try:
        import pandas as pd  # local import to avoid hard dependency at import time
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        if obj is pd.NaT:
            return None
    except Exception:
        pass
    try:
        import numpy as np
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


def _execute_tool_call(name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """Dispatch tool call to local implementations."""
    global tool_call_counter
    tool_call_counter += 1 # Increment before execution

    try:
        if name == "fetch_resilience_snapshot":
            return fetch_resilience_snapshot(**args)  # type: ignore[arg-type]
        if name == "fetch_uniswap_resilience_snapshot":
            return fetch_uniswap_resilience_snapshot(**args)  # type: ignore[arg-type]
        if name == "get_uniswap_v3_pool_address":
            pool_addr = get_uniswap_v3_pool_address(**args)  # type: ignore[arg-type]
            return {"pool_address": pool_addr}
        return {"error": f"Unknown tool: {name}"}
    except Exception as exc:  # pragma: no cover - pass errors back to model
        # Decrement counter if tool execution fails before final success
        tool_call_counter -= 1
        return {
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }


def _print_assistant_text(content: List[Any]) -> None:
    """Extract and print any text blocks from an assistant message."""
    
    texts = []
    for block in content:
        # Use getattr() for robust access to attributes (type/text) which might
        # be nested in dictionary keys or object attributes depending on SDK version/input.
        block_type = getattr(block, "type", None) if not isinstance(block, dict) else block.get("type")
        
        if block_type == "text":
            text_content = getattr(block, "text", None) if not isinstance(block, dict) else block.get("text")
            if text_content:
                texts.append(text_content)
                
    if texts:
        print("\nAssistant:\n", "\n".join(texts), "\n")


def run_agent() -> None:
    global tool_call_counter
    load_dotenv()
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY is required.")

    client = Anthropic(api_key=api_key)
    messages: List[Dict[str, Any]] = []

    # CLI one-shot support: if arguments are provided, send them as the first prompt.
    pending_inputs: List[str] = []
    if len(sys.argv) > 1:
        pending_inputs.append(" ".join(sys.argv[1:]).strip())
    interactive = len(pending_inputs) == 0

    if interactive:
        print("Agent ready. Type your request (or 'exit').\n")

    while True:
        # Reset counter for each new user request
        tool_call_counter = 0

        if pending_inputs:
            user_msg = pending_inputs.pop(0)
        else:
            user_msg = input("You: ").strip()

        if user_msg.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break
        if not user_msg:
            if interactive:
                continue
            else:
                break

        messages.append({"role": "user", "content": [{"type": "text", "text": user_msg}]})
        
        try:
            # Main loop for LLM interaction and tool execution
            while True:
                # Set 2-minute timeout before calling the API
                signal.alarm(120) 
                
                try:
                    response = client.messages.create(
                        model=MODEL,
                        max_tokens=1200,
                        tools=TOOLS,
                        system=SYSTEM_PROMPT,
                        messages=messages,
                    )
                except APIStatusError as api_err:
                    print(f"API error: {api_err}")
                    # Clear the alarm before continuing
                    signal.alarm(0)
                    break 
                
                # Clear the alarm immediately after a successful response
                signal.alarm(0)
                
                messages.append({"role": "assistant", "content": response.content})
                
                # SDK returns typed blocks; support both dict-style and attribute-style.
                tool_uses = []
                for block in response.content:
                    block_type = getattr(block, "type", None) if not isinstance(block, dict) else block.get("type")
                    if block_type == "tool_use":
                        tool_uses.append(block)

                # If no tool calls, emit text and break the inner while loop.
                if not tool_uses:
                    _print_assistant_text(response.content)
                    break
                
                # Enforce hard limit: if the next tool call would exceed the limit,
                # stop execution and tell the user the limit was hit.
                if tool_call_counter + len(tool_uses) > MAX_TOOL_CALLS:
                    print(f"\nAssistant:\n[ERROR] The LLM attempted to make {len(tool_uses)} more tool call(s), which would exceed the maximum limit of {MAX_TOOL_CALLS} for this request. Execution halted for efficiency.")
                    # Remove the assistant's last response which contained the forbidden tool calls
                    messages.pop()
                    break


                # === TOOL CALL LOGGING ADDED HERE ===
                print("\n--- LLM Tool Call(s) ---")
                for use in tool_uses:
                    use_name = use.get("name") if isinstance(use, dict) else getattr(use, "name")
                    use_input = use.get("input") if isinstance(use, dict) else getattr(use, "input")
                    print(f"Tool: **{use_name}**")
                    print("Args:", json.dumps(use_input, indent=2, default=json_serial))
                print("------------------------\n")
                # ====================================

                # Execute tools and send results back.
                tool_results = []
                for use in tool_uses:
                    # Support both dict and object access
                    use_name = use.get("name") if isinstance(use, dict) else getattr(use, "name")
                    use_input = use.get("input") if isinstance(use, dict) else getattr(use, "input")
                    use_id = use.get("id") if isinstance(use, dict) else getattr(use, "id")

                    result = _execute_tool_call(use_name, use_input)
                    # Debug log tool result (short)
                    print(f"[Tool Result] {use_name} -> keys: {list(result.keys())}")
                    if "error" in result:
                        print(f"[Tool Error] {result['error']}")
                        if "traceback" in result:
                            print(result["traceback"])
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": use_id,
                            "content": json.dumps(result, default=json_serial),
                        }
                    )

                # Log serialized payload (truncated for readability)
                print("[Tool Results Payload]", json.dumps(tool_results, indent=2, default=json_serial)[:800])

                messages.append({"role": "user", "content": tool_results})
                # Loop continues with the next API call to process the tool results...

        except TimeoutError as timeout_err:
            # Handle the crash upon timeout
            print(f"\n!!! CRITICAL ERROR !!!")
            print(f"The program has crashed because: {timeout_err}")
            print(f"Tool call counter for this request was at: {tool_call_counter}")
            # Exit the main loop and the program
            break 
        
        finally:
            # Ensure the alarm is cleared even if other non-critical errors occur
            signal.alarm(0)


        # Trim history lightly to keep context small for longer sessions.
        if len(messages) > 20:
            # Keep last 18 messages (system is passed separately).
            messages = messages[-18:]

        # For one-shot CLI usage, exit after the first full exchange.
        if not interactive:
            break


if __name__ == "__main__":
    run_agent()
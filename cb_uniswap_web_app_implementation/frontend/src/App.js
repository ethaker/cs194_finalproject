import { useEffect, useMemo, useState } from "react";
import { marked } from "marked";
import "./App.css";

const defaultCoinbasePayload = {
  product_id: "BTC-USD",
  start: "2022-10-05T00:00:00Z",
  end: "2022-10-19T00:00:00Z",
  event_date: "2022-10-13",
  granularity: 86_400, // daily
  pre_event_days: 7,
  post_event_days: 7,
  title: "Coinbase: BTC-USD (Oct 2022)",
  output_dir: "outputs",
  return_image_b64: true,
};

const defaultUniswapPayload = {
  start: "2022-10-05T00:00:00Z",
  end: "2022-10-19T00:00:00Z",
  event_date: "2022-10-13",
  granularity_days: 1,
  pre_event_days: 7,
  post_event_days: 7,
  title: "Uniswap: BTC/USDC v3",
  output_dir: "outputs",
  return_image_b64: true,
  // pool_address omitted to use backend default/env config
};

// Set to true to skip API calls and just render existing local images from /outputs
const useLocalImages = false;

const coinOptions = [
  "BTC-USD",
  "ETH-USD",
  "DAI-USD",
  "SOL-USD",
  "AVAX-USD",
  "LINK-USD",
  "LTC-USD",
  "MATIC-USD",
  "UNI-USD",
];

const tokenAddresses = {
  USDC: "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48",
  DAI: "0x6b175474e89094c44da98b954eedeac495271d0f",
  USDT: "0xdac17f958d2ee523a2206206994597c13d831ec7",
  WETH: "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
  WBTC: "0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599",
  BTC: "0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599",
  ETH: "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
  UNI: "0x1f9840a85d5af5bf1d1762f925bdaddc4201f984",
  LINK: "0x514910771af9ca656af840dff83e8264ecf986ca",
  MATIC: "0x7d1afa7b718fb893db30a3abc0cfc608aacfebb0",
  AVAX: "0x85f138bfEE4ef8e540890CFb48F620571d67Eda3",
  SOL: "0x7d8a3a8d21c4d4f140a0c5a7b169f91b7af4c5d6", // wormhole
  LTC: "", // likely no v3 pool on mainnet
};

// Known stable pool addresses to avoid missing pair errors (base/USDC)
const defaultPools = {
  DAI: "0x5777d92f208679db4b9778590fa3cab3ac9e2168", // DAI/USDC 0.01%
  ETH: "0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640", // WETH/USDC 0.05%
  WBTC: "0x8ad599c3a0ff1de082011efddc58f1908eb6e6d8", // WBTC/USDC 0.3%
};

const numberFmt = (value, options = {}) => {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  const { decimals = 2, suffix = "" } = options;
  return `${Number(value).toFixed(decimals)}${suffix}`;
};

const resolveOutputSrc = (apiBase, fallbackPath) => {
  if (!fallbackPath) return null;
  if (fallbackPath.startsWith("http://") || fallbackPath.startsWith("https://")) {
    return fallbackPath;
  }
  // Serve from backend /outputs route
  const trimmed = fallbackPath.replace(/^\/?outputs\//, "");
  return `${apiBase}/outputs/${trimmed}`;
};

const ChartCard = ({ title, chartBase64, fallbackPath, apiBase, status }) => {
  const src = chartBase64
    ? `data:image/png;base64,${chartBase64}`
    : resolveOutputSrc(apiBase, fallbackPath);

  return (
    <div className="card">
      <div className="card-header">
        <h3>{title}</h3>
      </div>
      <div className="card-body chart-body">
        {src ? (
          <img src={src} alt={title} className="chart-image" />
        ) : (
          <div className="placeholder">Chart unavailable</div>
        )}
        {status ? <div className="chart-status">{status}</div> : null}
      </div>
    </div>
  );
};

const MetricsTable = ({ title, metrics, formatter }) => {
  const rows = useMemo(() => {
    if (!metrics) return [];
    return Object.entries(metrics);
  }, [metrics]);

  if (!metrics) return null;

  return (
    <div className="card">
      <div className="card-header">
        <h3>{title}</h3>
      </div>
      <div className="card-body">
        <table className="metrics-table">
          <thead>
            <tr>
              <th>Metric</th>
              <th>Value</th>
            </tr>
          </thead>
          <tbody>
            {rows.map(([key, value]) => (
              <tr key={key}>
                <td>{key}</td>
                <td>{formatter ? formatter(key, value) : String(value)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};

function App() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [coinbase, setCoinbase] = useState(null);
  const [uniswap, setUniswap] = useState(null);
  const [coinbaseStatus, setCoinbaseStatus] = useState("");
  const [uniswapStatus, setUniswapStatus] = useState("");
  const [selectedCoin, setSelectedCoin] = useState(defaultCoinbasePayload.product_id);
  const [eventDate, setEventDate] = useState(defaultCoinbasePayload.event_date);
  const [poolAddress, setPoolAddress] = useState("");
  const [resolvedPool, setResolvedPool] = useState("");
  const [chatInput, setChatInput] = useState("");
  const [chatResponse, setChatResponse] = useState("");
  const [chatResponseHtml, setChatResponseHtml] = useState("");
  const [chatStatus, setChatStatus] = useState("");
  const [toolCalls, setToolCalls] = useState([]);
  const [showToolCalls, setShowToolCalls] = useState(false);

  // Default to backend at :8000; override with REACT_APP_API_BASE when deploying elsewhere.
  const apiBase = process.env.REACT_APP_API_BASE || "http://localhost:8000";

  const toQuery = (payload) => new URLSearchParams(payload).toString();

  const buildDateRange = (dateStr, preDays = 7, postDays = 7) => {
    const base = new Date(`${dateStr}T00:00:00Z`);
    if (Number.isNaN(base.getTime())) {
      return null;
    }
    const start = new Date(base);
    start.setUTCDate(start.getUTCDate() - preDays);
    const end = new Date(base);
    end.setUTCDate(end.getUTCDate() + postDays);
    const toIso = (d) => d.toISOString().split(".")[0] + "Z";
    return { start: toIso(start), end: toIso(end) };
  };

  const parseResponse = async (res, label) => {
    const contentType = res.headers.get("content-type") || "";
    const isJson = contentType.includes("application/json");
    if (!res.ok) {
      if (isJson) {
        const data = await res.json().catch(() => ({}));
        const message = data.error || data.message || JSON.stringify(data);
        const err = new Error(`${label} request failed: ${res.status} ${message}`);
        console.error("[API error]", label, res.status, data);
        throw err;
      }
      const text = await res.text();
      const err = new Error(`${label} request failed: ${res.status} ${text}`);
      console.error("[API error]", label, res.status, text);
      throw err;
    }
    if (isJson) {
      return res.json();
    }
    const text = await res.text();
    const snippet = text.slice(0, 800);
    const err = new Error(
      `${label} returned non-JSON (content-type: ${contentType}): ${snippet}`
    );
    console.error("[API non-JSON]", label, contentType, snippet);
    throw err;
  };

  const fetchSnapshots = async () => {
    setLoading(true);
    setError(null);
    setCoinbaseStatus("Loading Coinbase…");
    setUniswapStatus("Loading Uniswap…");
    try {
      const effectiveEventDate =
        (eventDate && eventDate.trim()) || defaultCoinbasePayload.event_date;
      if (!eventDate) {
        setEventDate(effectiveEventDate);
      }

      const range = buildDateRange(
        effectiveEventDate,
        defaultCoinbasePayload.pre_event_days,
        defaultCoinbasePayload.post_event_days
      );
      if (!range) {
        setLoading(false);
        setCoinbaseStatus("Invalid date");
        setUniswapStatus("Invalid date");
        setError("Please enter a valid event date (YYYY-MM-DD).");
        return;
      }
      const { start, end } = range;

      const coinbasePayload = {
        ...defaultCoinbasePayload,
        product_id: selectedCoin,
        event_date: effectiveEventDate,
        start,
        end,
        title: `Coinbase: ${selectedCoin}`,
      };

      // resolve pool for <token>/USDC using user input -> known defaults -> on-chain lookup
      const baseSymbol = selectedCoin.split("-")[0].toUpperCase();
      let autoPool = poolAddress.trim();

      // Prefer known good defaults to avoid wrong/empty pairings (e.g., DAI/USDC)
      if (!autoPool && defaultPools[baseSymbol]) {
        autoPool = defaultPools[baseSymbol];
        setResolvedPool(autoPool);
        setUniswapStatus("Using known pool");
      }

      // If still not set, try live resolution at 0.3% fee tier
      if (!autoPool) {
        const baseAddr = tokenAddresses[baseSymbol];
        const usdcAddr = tokenAddresses.USDC;
        if (baseAddr && usdcAddr) {
          try {
            const fee_bps = 3000;
            const q = toQuery({
              token0_address: baseAddr,
              token1_address: usdcAddr,
              fee_bps,
            });
            const res = await fetch(`${apiBase}/api/tools/pool-address?${q}`, {
              method: "GET",
            });
            const data = await res.json();
            if (data?.pool_address) {
              autoPool = data.pool_address;
              setResolvedPool(data.pool_address);
              setUniswapStatus(`Pool resolved @${fee_bps}bps`);
            } else {
              setResolvedPool("");
            }
          } catch {
            setResolvedPool("");
          }
        } else {
          setResolvedPool("");
        }
      } else {
        setResolvedPool(autoPool);
      }

      const uniswapPayload = {
        ...defaultUniswapPayload,
        event_date: effectiveEventDate,
        start,
        end,
        title: `Uniswap: ${selectedCoin.split("-")[0]}/USDC v3`,
        ...(autoPool ? { pool_address: autoPool } : {}),
      };

      const [cbRes, uniRes] = await Promise.all([
        fetch(`${apiBase}/api/tools/coinbase/metrics?${toQuery(coinbasePayload)}`, {
          method: "GET",
        }),
        fetch(`${apiBase}/api/tools/uniswap/metrics?${toQuery(uniswapPayload)}`, {
          method: "GET",
        }),
      ]);

      const [cbJson, uniJson] = await Promise.all([
        parseResponse(cbRes, "Coinbase"),
        parseResponse(uniRes, "Uniswap"),
      ]);

      setCoinbase(cbJson);
      setUniswap(uniJson);
      setCoinbaseStatus(
        cbJson?.record_count ? `Loaded ${cbJson.record_count} rows` : "Loaded"
      );
      setUniswapStatus(
        uniJson?.record_count
          ? `Loaded ${uniJson.record_count} rows`
          : uniJson?.error
          ? "Uniswap error"
          : "Loaded"
      );
    } catch (err) {
      setError(err.message || "Failed to load data.");
      if (!coinbase) setCoinbaseStatus("Coinbase unavailable");
      if (!uniswap) setUniswapStatus("Uniswap unavailable");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchSnapshots();
  }, []);

  const coinbaseMetricsFormatter = (key, value) => {
    const pctKeys = ["R_wick_body_ratio", "R_volatility"];
    const avgKeys = [
      "pre_avg_wick_body",
      "post_avg_wick_body",
      "pre_avg_volatility",
      "post_avg_volatility",
    ];
    if (pctKeys.includes(key)) return numberFmt(value, { decimals: 2, suffix: "%" });
    if (avgKeys.includes(key)) return numberFmt(value, { decimals: 4 });
    return String(value);
  };

  const uniswapMetricsFormatter = (key, value) => {
    const pctKeys = [
      "R_volatility",
      "R_intraday_range",
      "R_relative_range",
    ];
    const avgKeys = [
      "pre_avg_volatility",
      "post_avg_volatility",
      "pre_avg_range",
      "post_avg_range",
      "pre_avg_rel_range",
      "post_avg_rel_range",
    ];
    if (pctKeys.includes(key)) return numberFmt(value, { decimals: 2, suffix: "%" });
    if (avgKeys.includes(key)) return numberFmt(value, { decimals: 4 });
    if (key === "max_volatility_value") return numberFmt(value, { decimals: 4 });
    return String(value);
  };

  const sendChat = async () => {
    const message = chatInput.trim();
    if (!message) {
      setChatStatus("Please enter a question.");
      setChatResponse("");
      setChatResponseHtml("");
      return;
    }
    setChatStatus("Sending…");
    setChatResponse("");
    setChatResponseHtml("");
    setToolCalls([]);
    try {
      const res = await fetch(`${apiBase}/api/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message }),
      });
      const data = await parseResponse(res, "Chat");
      if (data.error) {
        setChatStatus("Error");
        setChatResponse(data.error);
        setChatResponseHtml("");
        setToolCalls(data.tool_calls || []);
      } else {
        const reply = data.reply || data.message || JSON.stringify(data);
        setChatStatus("Done");
        setChatResponse(reply);
        setToolCalls(data.tool_calls || []);
        try {
          setChatResponseHtml(marked.parse(reply || ""));
        } catch {
          setChatResponseHtml("");
        }
      }
    } catch (err) {
      setChatStatus("Error");
      setChatResponse(err.message || "Chat failed.");
      setChatResponseHtml("");
    }
  };

  return (
    <div className="page">
      <header className="header">
        <div>
          <h1>Market Resilience Dashboard</h1>
          <p>
            Coinbase and Uniswap charts side by side with resilience metrics from the
            backend tools.
          </p>
        </div>
        <div className="header-actions">
          <button onClick={fetchSnapshots} disabled={loading}>
            {loading ? "Loading…" : "Refresh"}
          </button>
        </div>
      </header>

      <section className="chat-section">
        <div className="card chat-card">
          <div className="card-header">
            <h3>Ask a one-shot question</h3>
          </div>
          <div className="card-body">
            <textarea
              id="chat-input"
              className="chat-textarea"
              rows={3}
              placeholder="e.g., Compare DAI/USDC resilience before and after Oct 13, 2022."
              value={chatInput}
              onChange={(e) => setChatInput(e.target.value)}
            />
            <div className="control-actions chat-actions">
              <button onClick={sendChat} disabled={loading}>
                {loading ? "Loading…" : "Send to chatbot"}
              </button>
              {chatStatus ? <span className="hint status-text">{chatStatus}</span> : null}
            </div>
            {chatResponse ? (
              <>
                {chatResponseHtml ? (
                  <div
                    className="chat-response chat-response-markdown"
                    dangerouslySetInnerHTML={{ __html: chatResponseHtml }}
                  />
                ) : (
                  <div className="chat-response">{chatResponse}</div>
                )}
                <div className="chat-response-actions">
                  <button
                    className="chat-action-button"
                    onClick={() => setChatInput(chatResponse)}
                  >
                    Use as Context
                  </button>
                  {toolCalls && toolCalls.length > 0 && (
                    <button
                      className="tool-calls-button"
                      onClick={() => setShowToolCalls(true)}
                    >
                      View Tool Calls ({toolCalls.length})
                    </button>
                  )}
                </div>
              </>
            ) : null}
          </div>
        </div>
      </section>

      <section className="controls">
        <div className="control-group">
          <label htmlFor="coin-select">Select market</label>
          <div className="pill-row">
            {coinOptions.map((c) => (
              <button
                key={c}
                className={`pill ${selectedCoin === c ? "pill-active" : ""}`}
                onClick={() => setSelectedCoin(c)}
                type="button"
              >
                {c}
              </button>
            ))}
          </div>
        </div>
        <div className="control-group">
          <label htmlFor="event-date">Event date (UTC)</label>
          <input
            id="event-date"
            type="date"
            value={eventDate}
            onChange={(e) => setEventDate(e.target.value)}
          />
        </div>
        <div className="control-actions">
          <button onClick={fetchSnapshots} disabled={loading}>
            {loading ? "Loading…" : "Run snapshot"}
          </button>
        </div>
      </section>

      {error && <div className="error-banner">Error: {error}</div>}

      <section className="charts-grid">
        <ChartCard
          title="Coinbase Chart"
          chartBase64={coinbase?.chart_base64}
          fallbackPath={coinbase?.chart_path || "DAI-USD_smoke.png"}
          apiBase={apiBase}
          status={coinbaseStatus}
        />
        <ChartCard
          title="Uniswap Chart"
          chartBase64={uniswap?.chart_base64}
          fallbackPath={uniswap?.chart_path || "Uniswap_DAI_USDC_uniswap.png"}
          apiBase={apiBase}
          status={uniswapStatus}
        />
      </section>

      <section className="tables-grid">
        <MetricsTable
          title="Coinbase Resilience Metrics"
          metrics={coinbase?.metrics}
          formatter={coinbaseMetricsFormatter}
        />
        <MetricsTable
          title="Uniswap Resilience Metrics"
          metrics={uniswap?.metrics}
          formatter={uniswapMetricsFormatter}
        />
      </section>

      {showToolCalls && (
        <div className="modal-overlay" onClick={() => setShowToolCalls(false)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h3>Tool Calls</h3>
              <button className="modal-close" onClick={() => setShowToolCalls(false)}>
                ×
              </button>
            </div>
            <div className="modal-body">
              {toolCalls.map((call, idx) => (
                <div key={idx} className="tool-call-item">
                  <div className="tool-call-header">
                    <strong>{call.tool}</strong>
                    <span className="tool-call-number">#{idx + 1}</span>
                  </div>
                  <div className="tool-call-section">
                    <div className="tool-call-label">Inputs:</div>
                    <pre className="tool-call-data">
                      {JSON.stringify(call.args, null, 2)}
                    </pre>
                  </div>
                  <div className="tool-call-section">
                    <div className="tool-call-label">Output:</div>
                    <pre className="tool-call-data">
                      {JSON.stringify(call.result, null, 2)}
                    </pre>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;

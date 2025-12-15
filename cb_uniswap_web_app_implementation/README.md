# Market Resilience Dashboard

A web application for comparing market resilience metrics between centralized exchanges (Coinbase) and decentralized exchanges (Uniswap v3) around market shock events. The application fetches OHLCV data, generates candlestick charts, computes resilience metrics, and provides an AI-powered chat interface for natural language analysis.

## Features

- **Dual Exchange Comparison**: Side-by-side analysis of Coinbase and Uniswap v3 data
- **Resilience Metrics**: Calculates volatility, wick/body ratios, and R-metrics (resilience indicators) around event dates
- **Interactive Charts**: Auto-generated candlestick charts with volume and TVL overlays
- **AI Chat Interface**: Natural language queries powered by Claude (Anthropic) with tool use for data fetching
- **Context Continuity**: Add previous chat output as context for follow-up questions with one click
- **Tool Call Inspection**: View detailed tool calls with inputs and outputs in a formatted modal
- **Pool Resolution**: Automatic Uniswap v3 pool address resolution for token pairs
- **Markdown Rendering**: Formatted chat responses with tables, headings, and code blocks

## Project Structure

```
194_implementation/
├── backend/
│   ├── server.py          # Flask API server
│   └── tools.py            # Data fetching and resilience calculation tools
├── frontend/
│   ├── src/
│   │   ├── App.js          # Main React component
│   │   └── App.css         # Styling
│   └── package.json        # Frontend dependencies
├── outputs/                # Generated chart images (gitignored)
├── requirements.txt        # Python dependencies
└── .env                    # Environment variables (gitignored)
```

## Prerequisites

- Python 3.11+
- Node.js 16+ and npm
- API Keys:
  - Anthropic API key (for Claude chat)
  - Uniswap/The Graph API key and subgraph ID
  - Alchemy API key (for Ethereum mainnet queries)

## Setup

### 1. Clone and Navigate

```bash
cd 194_implementation
```

### 2. Backend Setup

Create a virtual environment and install dependencies:

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Environment Variables

Create a `.env` file in the project root:

```env
ANTHROPIC_API_KEY=your_anthropic_key_here
UNISWAP_API_KEY=your_uniswap_key_here
UNISWAP_SUBGRAPH_ID=your_subgraph_id_here
ALCHEMY_API_KEY=your_alchemy_key_here
ANTHROPIC_MODEL=claude-opus-4-5-20251101  # Optional, defaults to this
PORT=8000  # Optional, defaults to 8000
```

### 4. Frontend Setup

```bash
cd frontend
npm install
cd ..
```

## Running the Application

### Start the Backend

From the project root with the virtual environment activated:

```bash
python -m backend.server
```

The backend will start on `http://localhost:8000` (or your configured PORT).

### Start the Frontend

In a separate terminal:

```bash
cd frontend
npm start
```

The frontend will start on `http://localhost:3000` and automatically open in your browser.

## Usage

### Dashboard View

1. Select a market pair from the coin selector (e.g., DAI-USD, ETH-USD)
2. Choose an event date (defaults to 2022-10-13)
3. Click "Run snapshot" to fetch and display:
   - Coinbase candlestick chart with resilience metrics
   - Uniswap v3 candlestick chart with TVL overlay and resilience metrics

### Chat Interface

Ask natural language questions in the chat box, such as:
- "Compare DAI/USDC resilience before and after October 13, 2022"
- "What was the volatility difference between Coinbase and Uniswap for ETH around the event?"
- "Show me the TVL changes for the DAI/USDC pool during the event period"

The AI will:
1. Resolve the Uniswap pool address (if needed)
2. Fetch Uniswap resilience snapshot data
3. Fetch Coinbase resilience snapshot data
4. Provide a formatted analysis with metrics and insights

**Additional Chat Features:**
- **Use as Context**: Click the "Use as Context" button below any chat response to copy the output into the input box for follow-up questions
- **View Tool Calls**: Click "View Tool Calls" to see all tool invocations with their inputs and outputs in a formatted modal popup

## API Endpoints

### Chat
- `POST /api/chat` - Natural language chat with tool use

### Direct Tool Endpoints
- `GET/POST /api/tools/coinbase` - Fetch Coinbase data
- `GET/POST /api/tools/uniswap` - Fetch Uniswap data
- `GET/POST /api/tools/pool-address` - Resolve Uniswap v3 pool address

### Metrics Endpoints
- `GET/POST /api/tools/coinbase/metrics` - Coinbase metrics only
- `GET/POST /api/tools/uniswap/metrics` - Uniswap metrics only

## Resilience Metrics

The application calculates several resilience indicators:

- **R_volatility**: Percentage change in volatility post-event vs pre-event
- **R_wick_body_ratio** (Coinbase): Change in candlestick wick/body ratios
- **R_intraday_range** (Uniswap): Change in absolute price ranges
- **R_relative_range** (Uniswap): Change in normalized price ranges

Pre and post-event windows default to 7 days but can be customized.

## Technologies

- **Backend**: Flask, Anthropic Claude API, Web3.py, pandas, mplfinance
- **Frontend**: React, marked (Markdown rendering)
- **Data Sources**: Coinbase Exchange API, The Graph (Uniswap subgraph)

## Notes

- Generated charts are saved to the `outputs/` directory
- The application requires a CSV file (`datetime_to_eth_block_number (3).csv`) for date-to-block mapping
- Pool addresses are auto-resolved using Uniswap V3 Factory contract on Ethereum mainnet

## Troubleshooting

- **Module not found errors**: Ensure virtual environment is activated and dependencies are installed
- **API errors**: Verify all API keys are set in `.env`
- **Chart generation fails**: Check that the `outputs/` directory exists and is writable
- **Pool resolution fails**: Ensure `ALCHEMY_API_KEY` is valid and you have network connectivity

## License

[Add your license here]


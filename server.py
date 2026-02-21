from flask import Flask, jsonify, request
from flask_cors import CORS
import yfinance as yf
from datetime import datetime, timedelta

app = Flask(__name__)
CORS(app)  # allows React (localhost:3000) to call this server

# ── Default watchlist ──────────────────────────────────────────────
DEFAULT_SYMBOLS = [
    "NVDA", "AAPL", "TSLA", "AMZN", "MSFT", "META",
    "GOOGL", "NFLX", "AMD", "INTC", "BABA", "UBER",
    "SHOP", "PYPL", "SNOW", "PLTR", "COIN", "SPOT",
    "RBLX", "ABNB"
]

def format_large(n):
    """Turn 2190000000000 → '2.19T' etc."""
    if n is None:
        return "N/A"
    if n >= 1e12:
        return f"{n/1e12:.2f}T"
    if n >= 1e9:
        return f"{n/1e9:.2f}B"
    if n >= 1e6:
        return f"{n/1e6:.2f}M"
    return str(n)


# ── Route 1: Get all watchlist stocks (summary) ────────────────────
@app.route("/api/stocks")
def get_stocks():
    symbols = request.args.get("symbols", ",".join(DEFAULT_SYMBOLS)).split(",")
    results = []

    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info

            # Get today's price movement
            hist = ticker.history(period="2d")
            if len(hist) < 2:
                continue

            prev_close = hist["Close"].iloc[-2]
            current    = hist["Close"].iloc[-1]
            change     = current - prev_close
            pct        = (change / prev_close) * 100

            results.append({
                "symbol":  symbol,
                "name":    info.get("longName") or info.get("shortName", symbol),
                "price":   round(current, 2),
                "change":  round(change, 2),
                "pct":     round(pct, 2),
                "sector":  info.get("sector", "—"),
                "mktCap":  format_large(info.get("marketCap")),
                "pe":      round(info.get("trailingPE", 0), 1) if info.get("trailingPE") else "N/A",
                "vol":     format_large(info.get("averageVolume")),
            })
        except Exception as e:
            print(f"Error fetching {symbol}: {e}")
            continue

    return jsonify(results)


# ── Route 2: Get price history for a single stock ─────────────────
@app.route("/api/history/<symbol>")
def get_history(symbol):
    days = int(request.args.get("days", 90))

    # Map days to yfinance period/interval
    if days <= 7:
        period, interval = "7d", "1h"
    elif days <= 30:
        period, interval = "1mo", "1d"
    elif days <= 90:
        period, interval = "3mo", "1d"
    else:
        period, interval = "1y", "1d"

    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period=period, interval=interval)

        data = []
        for ts, row in hist.iterrows():
            data.append({
                "date":   ts.strftime("%b %d") if interval == "1d" else ts.strftime("%b %d %H:%M"),
                "price":  round(row["Close"], 2),
                "volume": int(row["Volume"]),
            })

        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Route 3: Search for any ticker ────────────────────────────────
@app.route("/api/search")
def search():
    q = request.args.get("q", "")
    if not q:
        return jsonify([])
    try:
        results = yf.Search(q, max_results=8)
        hits = []
        for r in results.quotes:
            hits.append({
                "symbol": r.get("symbol", ""),
                "name":   r.get("longname") or r.get("shortname", ""),
                "type":   r.get("quoteType", ""),
            })
        return jsonify(hits)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("🚀 QuantDesk API running on http://localhost:5000")
    app.run(debug=True, port=5000)

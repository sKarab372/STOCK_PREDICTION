from flask import Flask, jsonify, request
from flask_cors import CORS
import yfinance as yf
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)
CORS(app)

# ── CONFIG ────────────────────────────────────────────────────────
SEQ_LEN      = 60      # days of history the model looks at
PRED_DAYS    = 14      # days to forecast
D_MODEL      = 64      # transformer embedding size
N_HEADS      = 4       # attention heads
N_LAYERS     = 2       # transformer encoder layers
DROPOUT      = 0.1
EPOCHS       = 80
LR           = 0.001
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")

# ── TECHNICAL INDICATORS ──────────────────────────────────────────
def compute_rsi(series, period=14):
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))

def compute_macd(series, fast=12, slow=26, signal=9):
    ema_fast   = series.ewm(span=fast).mean()
    ema_slow   = series.ewm(span=slow).mean()
    macd_line  = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal).mean()
    return macd_line, signal_line

def compute_bollinger(series, period=20):
    sma  = series.rolling(period).mean()
    std  = series.rolling(period).std()
    upper = sma + 2 * std
    lower = sma - 2 * std
    return upper, lower

def build_features(df):
    close = df["Close"]
    feat = pd.DataFrame(index=df.index)
    feat["close"]    = close
    feat["volume"]   = df["Volume"]
    feat["rsi"]      = compute_rsi(close)
    feat["macd"], feat["macd_sig"] = compute_macd(close)
    feat["bb_upper"], feat["bb_lower"] = compute_bollinger(close)
    feat["bb_width"] = feat["bb_upper"] - feat["bb_lower"]
    feat["return_1"] = close.pct_change(1)
    feat["return_5"] = close.pct_change(5)
    feat["return_10"]= close.pct_change(10)
    feat["sma_20"]   = close.rolling(20).mean()
    feat["sma_50"]   = close.rolling(50).mean()
    feat["ema_12"]   = close.ewm(span=12).mean()
    feat = feat.dropna()
    return feat

# ── TRANSFORMER MODEL ─────────────────────────────────────────────
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class StockTransformer(nn.Module):
    def __init__(self, n_features, d_model=D_MODEL, n_heads=N_HEADS,
                 n_layers=N_LAYERS, pred_days=PRED_DAYS, dropout=DROPOUT):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_enc    = PositionalEncoding(d_model, dropout=dropout)
        encoder_layer   = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, pred_days)
        )

    def forward(self, x):
        x = self.input_proj(x)       # (B, seq, d_model)
        x = self.pos_enc(x)
        x = self.transformer(x)      # (B, seq, d_model)
        x = x[:, -1, :]             # take last timestep
        return self.head(x)          # (B, pred_days)


# ── DATASET BUILDER ───────────────────────────────────────────────
def make_sequences(features_scaled, target_scaled, seq_len, pred_days):
    X, y = [], []
    total = len(features_scaled)
    for i in range(seq_len, total - pred_days + 1):
        X.append(features_scaled[i - seq_len:i])
        y.append(target_scaled[i:i + pred_days, 0])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


# ── TRAIN ─────────────────────────────────────────────────────────
def train_model(symbol):
    print(f"\n[{symbol}] Downloading data...")
    ticker = yf.Ticker(symbol)
    df = ticker.history(period="2y")

    if len(df) < SEQ_LEN + PRED_DAYS + 60:
        raise ValueError(f"Not enough data for {symbol}")

    feat = build_features(df)
    n_features = feat.shape[1]

    # Scale features and target separately
    feat_scaler   = MinMaxScaler()
    target_scaler = MinMaxScaler()

    feat_scaled   = feat_scaler.fit_transform(feat.values)
    target_scaled = target_scaler.fit_transform(feat[["close"]].values)

    X, y = make_sequences(feat_scaled, target_scaled, SEQ_LEN, PRED_DAYS)

    split = int(len(X) * 0.85)
    X_train, y_train = X[:split], y[:split]

    X_t = torch.tensor(X_train).to(DEVICE)
    y_t = torch.tensor(y_train).to(DEVICE)

    model = StockTransformer(n_features=n_features).to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    loss_fn = nn.HuberLoss()

    print(f"[{symbol}] Training {EPOCHS} epochs on {len(X_train)} sequences...")
    model.train()
    for epoch in range(EPOCHS):
        opt.zero_grad()
        pred = model(X_t)
        loss = loss_fn(pred, y_t)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}/{EPOCHS}  loss={loss.item():.6f}")

    # ── FORECAST ──────────────────────────────────────────────────
    model.eval()
    last_seq = torch.tensor(feat_scaled[-SEQ_LEN:]).unsqueeze(0).float().to(DEVICE)
    with torch.no_grad():
        pred_scaled = model(last_seq).cpu().numpy()[0]  # (pred_days,)

    # Inverse transform
    dummy = np.zeros((PRED_DAYS, 1))
    dummy[:, 0] = pred_scaled
    forecast_prices = target_scaler.inverse_transform(dummy)[:, 0].tolist()

    current_price = float(feat["close"].iloc[-1])
    target_price  = float(forecast_prices[-1])
    delta_pct     = (target_price - current_price) / current_price * 100

    signal = (
        "STRONG BUY" if delta_pct > 5  else
        "BUY"        if delta_pct > 1  else
        "HOLD"       if delta_pct > -2 else
        "SELL"
    )

    # Build date axis for forecast
    last_date = feat.index[-1]
    future_dates = pd.bdate_range(last_date, periods=PRED_DAYS + 1)[1:]
    forecast_series = [
        {"date": d.strftime("%b %d"), "predicted": round(p, 2)}
        for d, p in zip(future_dates, forecast_prices)
    ]

    # Historical prices for chart overlay
    hist_series = [
        {"date": ts.strftime("%b %d"), "price": round(row["close"], 2)}
        for ts, row in feat.tail(SEQ_LEN).iterrows()
    ]

    return {
        "symbol":       symbol,
        "currentPrice": round(current_price, 2),
        "targetPrice":  round(target_price, 2),
        "deltaPct":     round(delta_pct, 2),
        "signal":       signal,
        "forecast":     forecast_series,
        "history":      hist_series,
        "predDays":     PRED_DAYS,
    }


# ── ROUTE ─────────────────────────────────────────────────────────
@app.route("/api/predict/<symbol>")
def predict(symbol):
    try:
        result = train_model(symbol.upper())
        return jsonify(result)
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "device": str(DEVICE)})

if __name__ == "__main__":
    print("🧠 QuantDesk ML Server — Transformer Model")
    print(f"   Device : {DEVICE}")
    print(f"   Seq len: {SEQ_LEN} days  |  Forecast: {PRED_DAYS} days")
    print("   Running on http://localhost:5001\n")
    app.run(debug=False, port=5001)

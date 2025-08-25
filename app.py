from flask import Flask, render_template, jsonify, request
import yfinance as yf
import pandas as pd
import datetime

app = Flask(__name__)

def compute_indicators(data):
    data["EMA5"] = data["Close"].ewm(span=5, adjust=False).mean()
    data["EMA20"] = data["Close"].ewm(span=20, adjust=False).mean()

    # RSI
    delta = data["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    data["RSI"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = data["Close"].ewm(span=12, adjust=False).mean()
    ema26 = data["Close"].ewm(span=26, adjust=False).mean()
    data["MACD"] = ema12 - ema26
    data["MACD_Signal"] = data["MACD"].ewm(span=9, adjust=False).mean()

    # ATR
    high_low = data["High"] - data["Low"]
    high_close = (data["High"] - data["Close"].shift()).abs()
    low_close = (data["Low"] - data["Close"].shift()).abs()
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    tr = ranges.max(axis=1)
    data["ATR"] = tr.rolling(14).mean()

    # Bollinger Bands
    data["BB_MID"] = data["Close"].rolling(window=20).mean()
    data["BB_STD"] = data["Close"].rolling(window=20).std()
    data["BB_UPPER"] = data["BB_MID"] + 2 * data["BB_STD"]
    data["BB_LOWER"] = data["BB_MID"] - 2 * data["BB_STD"]

    # ADX
    high = data["High"]
    low = data["Low"]
    close = data["Close"]

    plus_dm = high.diff()
    minus_dm = low.diff().abs()

    plus_dm[plus_dm < 0] = 0
    minus_dm[low.diff() > 0] = 0

    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)

    atr = tr.rolling(14).mean()

    plus_di = 100 * (plus_dm.ewm(alpha=1/14).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1/14).mean() / atr)

    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100

    if isinstance(dx, pd.DataFrame):
        dx = dx.iloc[:, 0]

    data["ADX"] = dx.ewm(alpha=1/14).mean().fillna(0)

    return data

ASSETS = {
    "EUR/USD": "EURUSD=X",
    "GBP/USD": "GBPUSD=X",
    "USD/JPY": "USDJPY=X",
    "GBP/JPY": "GBPJPY=X",
    "AUD/USD": "AUDUSD=X"
}

current_asset = "EUR/USD"
last_signal = None
signal_history = []
win_count = 0
loss_count = 0

def get_signal(symbol):
    global win_count, loss_count, last_signal
    data = yf.download(symbol, interval="1m", period="1d")
    if len(data) < 30:
        return None, data
    data = compute_indicators(data)

    ema5 = data["EMA5"].iloc[-1]
    ema20 = data["EMA20"].iloc[-1]
    rsi = data["RSI"].iloc[-1]
    macd = data["MACD"].iloc[-1]
    signal_line = data["MACD_Signal"].iloc[-1]
    atr = data["ATR"].iloc[-1]
    bb_mid = data["BB_MID"].iloc[-1]
    adx = data["ADX"].iloc[-1]

    signal = None
    if (ema5 > ema20 and rsi > 50 and macd > signal_line and atr > 0 and data["Close"].iloc[-1] > bb_mid and adx > 20):
        signal = "BUY"
    elif (ema5 < ema20 and rsi < 50 and macd < signal_line and atr > 0 and data["Close"].iloc[-1] < bb_mid and adx > 20):
        signal = "SELL"

    return signal, data

@app.route("/api/signal")
def api_signal():
    global last_signal, signal_history, win_count, loss_count, current_asset
    asset = request.args.get("asset", current_asset)
    current_asset = asset
    signal, data = get_signal(ASSETS[asset])
    if signal and signal != last_signal:
        if last_signal is not None:
            last_close = data["Close"].iloc[-2]
            new_close = data["Close"].iloc[-1]
            if last_signal == "BUY" and new_close > last_close:
                win_count += 1
            elif last_signal == "SELL" and new_close < last_close:
                win_count += 1
            else:
                loss_count += 1
        last_signal = signal
        signal_history.insert(0, f"{datetime.datetime.now().strftime('%H:%M:%S')} - {signal}")

    total = win_count + loss_count
    accuracy = (win_count / total * 100) if total > 0 else 0

    return jsonify({
        "asset": asset,
        "signal": signal if signal else "NONE",
        "history": signal_history[:10],
        "stats": {
            "win": win_count,
            "loss": loss_count,
            "accuracy": f"{accuracy:.2f}%"
        },
        "chart": data.tail(100).reset_index().to_dict(orient="list")
    })

@app.route("/")
def dashboard():
    return render_template("index.html", assets=list(ASSETS.keys()))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

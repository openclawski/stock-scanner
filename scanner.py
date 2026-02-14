import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import json
import os


def get_stock_universe():
    """Get S&P 500 tickers from Wikipedia, fallback to top 30."""
    sp500_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    try:
        sp500_table = pd.read_html(sp500_url)[0]
        tickers = sp500_table['Symbol'].str.replace('.', '-', regex=False).tolist()
        return tickers[:500]
    except Exception:
        return ['AAPL','MSFT','GOOGL','AMZN','NVDA','META','TSLA','V','JPM','WMT',
                'MA','PG','UNH','HD','DIS','BAC','ADBE','CRM','NFLX','CMCSA',
                'XOM','PFE','COST','ABBV','TMO','CSCO','ABT','ACN','NKE','LIN']


# ---------------------------------------------------------------------------
# Fair Value Band  (PineScript v6 faithful translation)
#
# Logic:
#   1. EMA-smooth OHLC with period `ha_len`
#   2. Build Heikin Ashi candles from the smoothed OHLC
#   3. EMA-smooth the HA candles again with period `ha_len2`
#   4. Oscillator = 100 * (smoothed_HA_close - smoothed_HA_open)
#   5. Bullish when osc >= 0
#
# Default params on TradingView: ha_len=100, ha_len2=100, osc_len=7
# (osc_len only affects visual smoothing on chart, not the bias signal)
# ---------------------------------------------------------------------------

def calculate_fair_value_band(df, ha_len=100, ha_len2=100):
    """
    Returns 1 (bullish) or 0 (bearish) based on the FV Band oscillator.
    df must have columns: Open, High, Low, Close
    """
    n = len(df)
    if n < max(ha_len, ha_len2) + 10:
        return 0

    o = df['Open'].ewm(span=ha_len, adjust=False).mean()
    c = df['Close'].ewm(span=ha_len, adjust=False).mean()
    h = df['High'].ewm(span=ha_len, adjust=False).mean()
    l = df['Low'].ewm(span=ha_len, adjust=False).mean()

    # Heikin Ashi from smoothed values
    haclose = (o + h + l + c) / 4.0
    haopen = pd.Series(np.nan, index=df.index)
    haopen.iloc[0] = (o.iloc[0] + c.iloc[0]) / 2.0
    for i in range(1, n):
        haopen.iloc[i] = (haopen.iloc[i - 1] + haclose.iloc[i - 1]) / 2.0

    hahigh = pd.concat([h, haopen, haclose], axis=1).max(axis=1)
    halow  = pd.concat([l, haopen, haclose], axis=1).min(axis=1)

    # Second smoothing
    o2 = haopen.ewm(span=ha_len2, adjust=False).mean()
    c2 = haclose.ewm(span=ha_len2, adjust=False).mean()

    osc_bias = 100.0 * (c2 - o2)
    return 1 if float(osc_bias.iloc[-1]) >= 0 else 0


# ---------------------------------------------------------------------------
# B-Xtrender  (PineScript v5 faithful translation)
#
# Short term:
#   raw  = RSI( EMA(close, L1) - EMA(close, L2),  L3 ) - 50
#   line = T3( raw, 5 )           <-- T3 with b=0.7
#   signal: line rising  (line > line[1])  => bullish (lime)
#           line falling (line < line[1])  => bearish (red)
#
# Long term:
#   raw  = RSI( EMA(close, long_L1),  long_L2 ) - 50
#   signal: raw rising  => bullish
#           raw falling => bearish
#
# Defaults: short_l1=5, short_l2=20, short_l3=15, long_l1=20, long_l2=15
# ---------------------------------------------------------------------------

def _ema(series, period):
    return series.ewm(span=period, adjust=False).mean()


def _rsi(series, period):
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = gain / loss
    return 100.0 - (100.0 / (1.0 + rs))


def _t3(series, length, b=0.7):
    """Tilson T3 moving average."""
    c1 = -(b ** 3)
    c2 = 3 * b ** 2 + 3 * b ** 3
    c3 = -6 * b ** 2 - 3 * b - 3 * b ** 3
    c4 = 1 + 3 * b + b ** 3 + 3 * b ** 2
    e1 = _ema(series, length)
    e2 = _ema(e1, length)
    e3 = _ema(e2, length)
    e4 = _ema(e3, length)
    e5 = _ema(e4, length)
    e6 = _ema(e5, length)
    return c1 * e6 + c2 * e5 + c3 * e4 + c4 * e3


def calculate_bx_trender(df, short_l1=5, short_l2=20, short_l3=15,
                          long_l1=20, long_l2=15):
    """
    Returns dict with:
      'short': 1 if T3 line rising, else 0
      'long':  1 if long-term RSI line rising, else 0
    df must have column: Close
    """
    n = len(df)
    if n < 60:
        return {'short': 0, 'long': 0}

    close = df['Close']

    # Short term
    ema_diff = _ema(close, short_l1) - _ema(close, short_l2)
    short_raw = _rsi(ema_diff, short_l3) - 50.0
    ma_short = _t3(short_raw, 5)
    short_signal = 1 if float(ma_short.iloc[-1]) > float(ma_short.iloc[-2]) else 0

    # Long term
    long_raw = _rsi(_ema(close, long_l1), long_l2) - 50.0
    long_signal = 1 if float(long_raw.iloc[-1]) > float(long_raw.iloc[-2]) else 0

    return {'short': short_signal, 'long': long_signal}


# ---------------------------------------------------------------------------
# Scanner
# ---------------------------------------------------------------------------

def scan_stock(ticker):
    try:
        stock = yf.Ticker(ticker)
        df_w = stock.history(period="2y", interval="1wk")
        df_m = stock.history(period="5y", interval="1mo")

        if df_w.empty or df_m.empty or len(df_w) < 100 or len(df_m) < 24:
            return None

        info = stock.info
        name = info.get('longName', info.get('shortName', ticker))

        ind = {}

        # Fair Value Band: weekly & monthly, two period settings
        ind['FVB_W_100'] = calculate_fair_value_band(df_w, ha_len=100, ha_len2=100)
        ind['FVB_M_100'] = calculate_fair_value_band(df_m, ha_len=100, ha_len2=100)

        # BX-Trender: weekly & monthly
        bx_w = calculate_bx_trender(df_w)
        bx_m = calculate_bx_trender(df_m)
        ind['BX_W_Short'] = bx_w['short']
        ind['BX_W_Long']  = bx_w['long']
        ind['BX_M_Short'] = bx_m['short']
        ind['BX_M_Long']  = bx_m['long']

        total = sum(ind.values())
        price = round(float(df_w['Close'].iloc[-1]), 2)
        print(f'  {ticker}: {total}/6')
        return {
            'ticker': ticker,
            'name': name,
            'price': price,
            'indicators': ind,
            'total_score': total,
            'max_score': 6
        }
    except Exception as e:
        print(f'  {ticker}: SKIP ({e})')
        return None


def generate_html(results, timestamp):
    results.sort(key=lambda x: x["total_score"], reverse=True)
    rows = ""
    for s in results:
        sc = s["total_score"]
        cls = "score-high" if sc >= 5 else "score-med" if sc >= 3 else "score-low"
        ind = s["indicators"]
        def dot(v): return "bullish" if v else "bearish"
        rows += "<tr>"
        rows += f'<td class="ticker">{s["ticker"]}</td>'
        rows += f'<td>{s["name"]}</td>'
        rows += f'<td class="price">${s["price"]:.2f}</td>'
        rows += f'<td><span class="score-badge {cls}">{sc}/6</span></td>'
        for k in ["FVB_W_100","FVB_M_100","BX_W_Short","BX_W_Long","BX_M_Short","BX_M_Long"]:
            rows += f'<td><span class="indicator {dot(ind[k])}"></span></td>'
        rows += "</tr>\n"

    bull = len([r for r in results if r["total_score"] >= 5])
    neut = len([r for r in results if 3 <= r["total_score"] < 5])
    bear = len([r for r in results if r["total_score"] < 3])
    total = len(results)

    html = HTML_TEMPLATE.replace("{TIMESTAMP}", timestamp)
    html = html.replace("{TOTAL}", str(total))
    html = html.replace("{BULLISH}", str(bull))
    html = html.replace("{NEUTRAL}", str(neut))
    html = html.replace("{BEARISH}", str(bear))
    html = html.replace("{ROWS}", rows)
    return html


HTML_TEMPLATE = """<!DOCTYPE html>
<html><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Stock Scanner</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:sans-serif;background:#0a0e27;color:#e0e6f0;padding:20px}
.container{max-width:1400px;margin:0 auto}
header{text-align:center;margin-bottom:40px;padding:30px 0;border-bottom:2px solid #1a2332}
h1{font-size:2.5em;margin-bottom:10px;color:#64ffda}
.last-updated{color:#8892b0;font-size:0.9em}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:20px;margin-bottom:30px}
.stat-card{background:#1a2332;padding:20px;border-radius:10px;border:1px solid #2a3548}
.stat-label{color:#8892b0;font-size:0.85em;margin-bottom:8px}
.stat-value{font-size:2em;font-weight:bold;color:#64ffda}
table{width:100%;background:#1a2332;border-radius:10px;overflow:hidden;border-collapse:collapse}
thead{background:#0f1729}
th{padding:15px;text-align:left;font-weight:600;color:#64ffda;border-bottom:2px solid #2a3548}
td{padding:12px 15px;border-bottom:1px solid #2a3548}
tr:hover{background:#0f1729}
.score-badge{display:inline-block;padding:5px 12px;border-radius:20px;font-weight:bold;font-size:0.9em}
.score-high{background:#10b981;color:#000}
.score-med{background:#f59e0b;color:#000}
.score-low{background:#6b7280;color:#fff}
.indicator{display:inline-block;width:18px;height:18px;border-radius:3px}
.bullish{background:#10b981}
.bearish{background:#ef4444}
.ticker{font-weight:bold;color:#64ffda;font-size:1.05em}
.price{color:#8892b0}
</style>
</head>
<body>
<div class="container">
<header><h1>Stock Scanner Dashboard</h1>
<div class="last-updated">Last Updated: {TIMESTAMP}</div></header>
<div class="stats">
<div class="stat-card"><div class="stat-label">Total Scanned</div><div class="stat-value">{TOTAL}</div></div>
<div class="stat-card"><div class="stat-label">Bullish (5-6/6)</div><div class="stat-value">{BULLISH}</div></div>
<div class="stat-card"><div class="stat-label">Neutral (3-4/6)</div><div class="stat-value">{NEUTRAL}</div></div>
<div class="stat-card"><div class="stat-label">Bearish (0-2/6)</div><div class="stat-value">{BEARISH}</div></div>
</div>
<table><thead><tr>
<th>Ticker</th><th>Name</th><th>Price</th><th>Score</th>
<th>FVB W</th><th>FVB M</th><th>BX-S W</th><th>BX-L W</th><th>BX-S M</th><th>BX-L M</th>
</tr></thead>
<tbody>
{ROWS}
</tbody></table>
</div>
</body></html>"""


def main():
    print("Starting stock scanner...")
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"Timestamp: {ts}")

    tickers = get_stock_universe()
    print(f"Scanning {len(tickers)} stocks...")

    results = []
    for i, t in enumerate(tickers):
        if i % 50 == 0 and i > 0:
            print(f"Progress: {i}/{len(tickers)}")
        r = scan_stock(t)
        if r:
            results.append(r)

    print(f"Scanned {len(results)} stocks successfully")
    os.makedirs("docs", exist_ok=True)

    html = generate_html(results, ts)
    with open("docs/index.html", "w") as f:
        f.write(html)
    with open("docs/data.json", "w") as f:
        json.dump({"timestamp": ts, "total_scanned": len(results), "results": results}, f, indent=2)

    print("Dashboard generated: docs/index.html")
    bull = len([r for r in results if r["total_score"] >= 5])
    print(f"Bullish (5-6/6): {bull}")


if __name__ == "__main__":
    main()

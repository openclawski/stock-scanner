import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import json
import os

def get_stock_universe():
    sp500_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    try:
        sp500_table = pd.read_html(sp500_url)[0]
        sp500_tickers = sp500_table['Symbol'].str.replace('.', '-').tolist()
        return sp500_tickers[:500]
    except:
        return ['AAPL','MSFT','GOOGL','AMZN','NVDA','META','TSLA','V','JPM','WMT',
                'MA','PG','UNH','HD','DIS','BAC','ADBE','CRM','NFLX','CMCSA',
                'XOM','PFE','COST','ABBV','TMO','CSCO','ABT','ACN','NKE','LIN']

def calculate_fair_value_band(df, length=20):
    if len(df) < length:
        return 0
    basis = df['Close'].rolling(window=length).mean()
    last_close = float(df['Close'].iloc[-1])
    last_basis = float(basis.iloc[-1])
    return 1 if last_close > last_basis else 0

def calculate_bx_trender(df):
    if len(df) < 6:
        return 0
    momentum = df['Close'].diff(1).rolling(window=5).mean()
    curr = float(momentum.iloc[-1])
    prev = float(momentum.iloc[-2])
    return 1 if curr > prev else 0

def scan_stock(ticker):
    try:
        stock = yf.Ticker(ticker)
        df_w = stock.history(period="2y", interval="1wk")
        df_m = stock.history(period="5y", interval="1mo")
        if df_w.empty or df_m.empty or len(df_w) < 33 or len(df_m) < 24:
            return None
        info = stock.info
        name = info.get('longName', info.get('shortName', ticker))
        ind = {}
        ind['FVB_W_20'] = calculate_fair_value_band(df_w, 20)
        ind['FVB_W_33'] = calculate_fair_value_band(df_w, 33)
        ind['FVB_M_20'] = calculate_fair_value_band(df_m, 20)
        ind['FVB_M_33'] = calculate_fair_value_band(df_m, 33)
        ind['BX_W'] = calculate_bx_trender(df_w)
        ind['BX_M'] = calculate_bx_trender(df_m)
        total = sum(ind.values())
        price = round(float(df_w['Close'].iloc[-1]), 2)
        print(f'  {ticker}: {total}/6')
        return {'ticker': ticker, 'name': name, 'price': price, 'indicators': ind, 'total_score': total, 'max_score': 6}
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
        for k in ["FVB_W_20","FVB_W_33","FVB_M_20","FVB_M_33","BX_W","BX_M"]:
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
<th>FVB W20</th><th>FVB W33</th><th>FVB M20</th><th>FVB M33</th><th>BX W</th><th>BX M</th>
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

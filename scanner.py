import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import json
import os


# ─────────────────────────────────────────────────────────────
# STOCK UNIVERSE
# ─────────────────────────────────────────────────────────────

def get_stock_universe():
    """Get S&P 500 tickers from Wikipedia, fallback to top 30."""
    try:
        sp500 = pd.read_html(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        )[0]
        tickers = sp500["Symbol"].str.replace(".", "-", regex=False).tolist()
        return tickers[:500]
    except Exception as e:
        print(f"Wikipedia fetch failed: {e}")
        return [
            "AAPL","MSFT","GOOGL","AMZN","NVDA","META","TSLA","V","JPM","WMT",
            "MA","PG","UNH","HD","DIS","BAC","ADBE","CRM","NFLX","CMCSA",
            "XOM","PFE","COST","ABBV","TMO","CSCO","ABT","ACN","NKE","LIN",
        ]


# ─────────────────────────────────────────────────────────────
# FAIR VALUE BAND  (PineScript v6 — faithful port)
#
#  1. EMA-smooth OHLC with `period`
#  2. Build Heikin Ashi candles (recursive haopen)
#  3. EMA-smooth the HA values with `smoothing`
#  4. osc_bias = 100 * (smoothed_HA_close − smoothed_HA_open)
#  5. Bullish when osc_bias >= 0
# ─────────────────────────────────────────────────────────────

def calculate_fair_value_band(df, period=20, smoothing=7):
    """
    Returns 1 (bullish) or 0 (bearish) for the last bar.
    `df` must have OHLC columns.
    """
    n = max(period, smoothing) * 3  # need enough warmup bars
    if len(df) < n:
        return 0

    o = df["Open"].ewm(span=period, adjust=False).mean()
    c = df["Close"].ewm(span=period, adjust=False).mean()
    h = df["High"].ewm(span=period, adjust=False).mean()
    l = df["Low"].ewm(span=period, adjust=False).mean()

    # --- Heikin Ashi from smoothed OHLC ---
    ha_close = (o + h + l + c) / 4.0

    # Recursive haopen: haopen[0] = (o+c)/2, then (prev_haopen + prev_haclose)/2
    ha_open = pd.Series(np.nan, index=df.index)
    ha_open.iloc[0] = (o.iloc[0] + c.iloc[0]) / 2.0
    for i in range(1, len(df)):
        ha_open.iloc[i] = (ha_open.iloc[i - 1] + ha_close.iloc[i - 1]) / 2.0

    ha_high = pd.concat([h, ha_open, ha_close], axis=1).max(axis=1)
    ha_low = pd.concat([l, ha_open, ha_close], axis=1).min(axis=1)

    # --- Second smoothing of HA values ---
    o2 = ha_open.ewm(span=smoothing, adjust=False).mean()
    c2 = ha_close.ewm(span=smoothing, adjust=False).mean()

    # --- Oscillator bias ---
    osc_bias = 100.0 * (c2 - o2)

    last_val = float(osc_bias.iloc[-1])
    return 1 if last_val >= 0 else 0


# ─────────────────────────────────────────────────────────────
# B-XTRENDER  (PineScript v5 — faithful port)
#
# Short-term:  RSI( EMA(close,5) − EMA(close,20), 15 ) − 50
# Long-term:   RSI( EMA(close,20), 15 ) − 50
# T3 of short: 6× nested EMA with b=0.7
# Signal:      T3 line rising  →  bullish (lime)
#              T3 line falling →  bearish (red)
# ─────────────────────────────────────────────────────────────

def _rsi(series, period):
    """Wilder-style RSI matching TradingView's ta.rsi."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100.0 - 100.0 / (1.0 + rs)


def _t3(src, length, b=0.7):
    """Tillson T3 — 6× nested EMA with volume factor b."""
    e1 = src.ewm(span=length, adjust=False).mean()
    e2 = e1.ewm(span=length, adjust=False).mean()
    e3 = e2.ewm(span=length, adjust=False).mean()
    e4 = e3.ewm(span=length, adjust=False).mean()
    e5 = e4.ewm(span=length, adjust=False).mean()
    e6 = e5.ewm(span=length, adjust=False).mean()

    c1 = -(b ** 3)
    c2 = 3 * b * b + 3 * b * b * b
    c3 = -6 * b * b - 3 * b - 3 * b * b * b
    c4 = 1 + 3 * b + b * b * b + 3 * b * b
    return c1 * e6 + c2 * e5 + c3 * e4 + c4 * e3


def calculate_bx_trender(df, short_l1=5, short_l2=20, short_l3=15,
                         long_l1=20, long_l2=15):
    """
    Returns dict with 'short' and 'long' signals (each 1=bullish, 0=bearish).
    Short signal: T3 of short-term RSI is rising.
    Long signal:  long-term value is rising.
    """
    if len(df) < 60:
        return {"short": 0, "long": 0}

    close = df["Close"]

    # -- Short-term: RSI(EMA(close,5) - EMA(close,20), 15) - 50 --
    ema_s1 = close.ewm(span=short_l1, adjust=False).mean()
    ema_s2 = close.ewm(span=short_l2, adjust=False).mean()
    short_raw = _rsi(ema_s1 - ema_s2, short_l3) - 50.0

    # T3 of the short-term oscillator (length=5, b=0.7)
    ma_short = _t3(short_raw, 5, 0.7)

    # Short signal: T3 line rising
    short_signal = 1 if float(ma_short.iloc[-1]) > float(ma_short.iloc[-2]) else 0

    # -- Long-term: RSI(EMA(close,20), 15) - 50 --
    ema_l1 = close.ewm(span=long_l1, adjust=False).mean()
    long_raw = _rsi(ema_l1, long_l2) - 50.0

    # Long signal: value rising
    long_signal = 1 if float(long_raw.iloc[-1]) > float(long_raw.iloc[-2]) else 0

    return {"short": short_signal, "long": long_signal}


# ─────────────────────────────────────────────────────────────
# SCANNER
# ─────────────────────────────────────────────────────────────

def scan_stock(ticker):
    """Scan a single stock across weekly & monthly timeframes."""
    try:
        stock = yf.Ticker(ticker)
        df_w = stock.history(period="2y", interval="1wk")
        df_m = stock.history(period="10y", interval="1mo")

        if df_w.empty or df_m.empty or len(df_w) < 40 or len(df_m) < 40:
            return None

        info = stock.info
        name = info.get("longName", info.get("shortName", ticker))

        # Fair Value Band — 4 configs
        fvb_w_20 = calculate_fair_value_band(df_w, period=20, smoothing=7)
        fvb_w_33 = calculate_fair_value_band(df_w, period=33, smoothing=7)
        fvb_m_20 = calculate_fair_value_band(df_m, period=20, smoothing=7)
        fvb_m_33 = calculate_fair_value_band(df_m, period=33, smoothing=7)

        # BX-Trender — weekly & monthly (default params)
        bx_w = calculate_bx_trender(df_w)
        bx_m = calculate_bx_trender(df_m)

        indicators = {
            "FVB_W_20": fvb_w_20,
            "FVB_W_33": fvb_w_33,
            "FVB_M_20": fvb_m_20,
            "FVB_M_33": fvb_m_33,
            "BX_W_Short": bx_w["short"],
            "BX_W_Long": bx_w["long"],
            "BX_M_Short": bx_m["short"],
            "BX_M_Long": bx_m["long"],
        }
        total = sum(indicators.values())
        price = round(float(df_w["Close"].iloc[-1]), 2)

        print(f"  {ticker}: {total}/8")
        return {
            "ticker": ticker,
            "name": name,
            "price": price,
            "indicators": indicators,
            "total_score": total,
            "max_score": 8,
        }
    except Exception as e:
        print(f"  {ticker}: SKIP ({e})")
        return None


# ─────────────────────────────────────────────────────────────
# HTML GENERATION
# ─────────────────────────────────────────────────────────────

def generate_html(results, timestamp):
    results.sort(key=lambda x: x["total_score"], reverse=True)

    rows = ""
    for s in results:
        sc = s["total_score"]
        mx = s["max_score"]
        cls = "score-high" if sc >= 6 else "score-med" if sc >= 4 else "score-low"
        ind = s["indicators"]

        def dot(v):
            return "bullish" if v else "bearish"

        rows += "<tr>"
        rows += f'<td class="ticker">{s["ticker"]}</td>'
        rows += f'<td>{s["name"]}</td>'
        rows += f'<td class="price">${s["price"]:.2f}</td>'
        rows += f'<td><span class="score-badge {cls}">{sc}/{mx}</span></td>'
        for k in [
            "FVB_W_20","FVB_W_33","FVB_M_20","FVB_M_33",
            "BX_W_Short","BX_W_Long","BX_M_Short","BX_M_Long",
        ]:
            rows += f'<td><span class="indicator {dot(ind[k])}"></span></td>'
        rows += "</tr>\n"

    bull = len([r for r in results if r["total_score"] >= 6])
    neut = len([r for r in results if 4 <= r["total_score"] < 6])
    bear = len([r for r in results if r["total_score"] < 4])
    total = len(results)

    html = HTML_TEMPLATE
    html = html.replace("{TIMESTAMP}", timestamp)
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
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#0a0e27;color:#e0e6f0;padding:20px}
.container{max-width:1600px;margin:0 auto}
header{text-align:center;margin-bottom:40px;padding:30px 0;border-bottom:2px solid #1a2332}
h1{font-size:2.5em;margin-bottom:10px;color:#64ffda}
.last-updated{color:#8892b0;font-size:0.9em}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:20px;margin-bottom:30px}
.stat-card{background:#1a2332;padding:20px;border-radius:10px;border:1px solid #2a3548}
.stat-label{color:#8892b0;font-size:0.85em;margin-bottom:8px}
.stat-value{font-size:2em;font-weight:bold;color:#64ffda}
.table-wrap{overflow-x:auto}
table{width:100%;background:#1a2332;border-radius:10px;overflow:hidden;border-collapse:collapse;min-width:1100px}
thead{background:#0f1729}
th{padding:15px;text-align:left;font-weight:600;color:#64ffda;border-bottom:2px solid #2a3548;cursor:pointer;white-space:nowrap}
th:hover{background:#1a2548}
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
.controls{display:flex;gap:15px;margin-bottom:20px;flex-wrap:wrap;align-items:center}
.controls input,.controls select{background:#1a2332;color:#e0e6f0;border:1px solid #2a3548;padding:8px 12px;border-radius:6px;font-size:0.9em}
.controls label{color:#8892b0;font-size:0.85em}
.section-label{font-size:0.7em;color:#8892b0;display:block}
</style>
</head>
<body>
<div class="container">
<header><h1>Stock Scanner Dashboard</h1>
<p style="color:#8892b0;margin-bottom:5px">Fair Value Band + B-Xtrender | Weekly &amp; Monthly</p>
<div class="last-updated">Last Updated: {TIMESTAMP}</div></header>
<div class="stats">
<div class="stat-card"><div class="stat-label">Total Scanned</div><div class="stat-value">{TOTAL}</div></div>
<div class="stat-card"><div class="stat-label">Bullish (6-8)</div><div class="stat-value" style="color:#10b981">{BULLISH}</div></div>
<div class="stat-card"><div class="stat-label">Neutral (4-5)</div><div class="stat-value" style="color:#f59e0b">{NEUTRAL}</div></div>
<div class="stat-card"><div class="stat-label">Bearish (0-3)</div><div class="stat-value" style="color:#ef4444">{BEARISH}</div></div>
</div>
<div class="controls">
<div><label>Min Score</label><select id="minScore"><option value="0">All</option><option value="4">4+</option><option value="5">5+</option><option value="6">6+</option><option value="7">7+</option><option value="8">8/8</option></select></div>
<div><label>Search</label><input type="text" id="search" placeholder="Ticker or name..."></div>
</div>
<div class="table-wrap">
<table><thead><tr>
<th>Ticker</th><th>Name</th><th>Price</th><th>Score</th>
<th><span class="section-label">FVB</span>W 20</th>
<th><span class="section-label">FVB</span>W 33</th>
<th><span class="section-label">FVB</span>M 20</th>
<th><span class="section-label">FVB</span>M 33</th>
<th><span class="section-label">BX</span>W Short</th>
<th><span class="section-label">BX</span>W Long</th>
<th><span class="section-label">BX</span>M Short</th>
<th><span class="section-label">BX</span>M Long</th>
</tr></thead>
<tbody id="tbody">
{ROWS}
</tbody></table>
</div>
</div>
<script>
const rows=document.querySelectorAll('#tbody tr');
const minEl=document.getElementById('minScore');
const searchEl=document.getElementById('search');
function filter(){
  const min=parseInt(minEl.value);
  const q=searchEl.value.toLowerCase();
  rows.forEach(r=>{
    const score=parseInt(r.children[3].textContent);
    const text=r.children[0].textContent.toLowerCase()+' '+r.children[1].textContent.toLowerCase();
    r.style.display=(score>=min&&text.includes(q))?'':'none';
  });
}
minEl.addEventListener('change',filter);
searchEl.addEventListener('input',filter);
</script>
</body></html>"""


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

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
        json.dump(
            {"timestamp": ts, "total_scanned": len(results), "results": results},
            f,
            indent=2,
        )

    print("Dashboard generated: docs/index.html")
    bull = len([r for r in results if r["total_score"] >= 6])
    print(f"Bullish (6+/8): {bull}")


if __name__ == "__main__":
    main()

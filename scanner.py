import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import json

def get_stock_universe():
    sp500_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    try:
        sp500_table = pd.read_html(sp500_url)[0]
        sp500_tickers = sp500_table['Symbol'].str.replace('.', '-').tolist()
        return sp500_tickers[:100]
    except:
        return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'V', 'JPM', 'WMT']

def calculate_fair_value_band(df, length=20):
    if len(df) < length:
        return np.nan
    basis = df['Close'].rolling(window=length).mean()
    osc_bias = 1 if df['Close'].iloc[-1] > basis.iloc[-1] else -1
    return osc_bias

def calculate_bx_trender(df):
    if len(df) < 5:
        return False
    momentum = df['Close'].diff(1).rolling(window=5).mean()
    return momentum.iloc[-1] > momentum.iloc[-2] if len(momentum) > 1 else False

def scan_stock(ticker, results_list):
    try:
        stock = yf.Ticker(ticker)
        df_weekly = stock.history(period="2y", interval="1wk")
        df_monthly = stock.history(period="5y", interval="1mo")

        if df_weekly.empty or df_monthly.empty or len(df_weekly) < 33 or len(df_monthly) < 33:
            return

        scores = {
            'ticker': ticker,
            'name': stock.info.get('longName', ticker),
            'price': float(df_weekly['Close'].iloc[-1]),
            'indicators': {}
        }

        scores['indicators']['FVB_W_20'] = 1 if calculate_fair_value_band(df_weekly, 20) >= 0 else 0
        scores['indicators']['FVB_W_33'] = 1 if calculate_fair_value_band(df_weekly, 33) >= 0 else 0
        scores['indicators']['FVB_M_20'] = 1 if calculate_fair_value_band(df_monthly, 20) >= 0 else 0
        scores['indicators']['FVB_M_33'] = 1 if calculate_fair_value_band(df_monthly, 33) >= 0 else 0
        scores['indicators']['BX_W'] = 1 if calculate_bx_trender(df_weekly) else 0
        scores['indicators']['BX_M'] = 1 if calculate_bx_trender(df_monthly) else 0

        scores['total_score'] = sum(scores['indicators'].values())
        scores['max_score'] = 6

        results_list.append(scores)
        print(f"✓ {ticker}: {scores['total_score']}/6")
    except Exception as e:
        print(f"✗ {ticker}: {str(e)}")

def generate_html(results):
    results_sorted = sorted(results, key=lambda x: x['total_score'], reverse=True)
    rows = ""
    for stock in results_sorted:
        score = stock['total_score']
        score_class = 'score-high' if score >= 5 else 'score-med' if score >= 3 else 'score-low'
        ind = stock['indicators']
        rows += f'''<tr>
<td class="ticker">{stock['ticker']}</td>
<td>{stock['name']}</td>
<td class="price">${stock['price']:.2f}</td>
<td><span class="score-badge {score_class}">{score}/6</span></td>
<td><span class="indicator {'bullish' if ind['FVB_W_20'] else 'bearish'}"></span></td>
<td><span class="indicator {'bullish' if ind['FVB_W_33'] else 'bearish'}"></span></td>
<td><span class="indicator {'bullish' if ind['FVB_M_20'] else 'bearish'}"></span></td>
<td><span class="indicator {'bullish' if ind['FVB_M_33'] else 'bearish'}"></span></td>
<td><span class="indicator {'bullish' if ind['BX_W'] else 'bearish'}"></span></td>
<td><span class="indicator {'bullish' if ind['BX_M'] else 'bearish'}"></span></td>
</tr>
'''

    bullish_count = len([r for r in results_sorted if r['total_score'] >= 5])
    neutral_count = len([r for r in results_sorted if 3 <= r['total_score'] < 5])
    bearish_count = len([r for r in results_sorted if r['total_score'] < 3])

    html_template = open('/tmp/template.html', 'r').read()
    html = html_template.format(
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC"),
        total=len(results_sorted),
        bullish=bullish_count,
        neutral=neutral_count,
        bearish=bearish_count,
        rows=rows
    )
    return html

def main():
    print("🐳 Starting stock scanner...")
    print(f"Timestamp: {datetime.now()}")

    tickers = get_stock_universe()
    print(f"Scanning {len(tickers)} stocks...")

    results = []
    for ticker in tickers:
        scan_stock(ticker, results)

    print(f"✓ Scanned {len(results)} stocks successfully")

    html = generate_html(results)

    with open('docs/index.html', 'w') as f:
        f.write(html)

    with open('docs/data.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("✓ Dashboard generated!")

if __name__ == "__main__":
    main()

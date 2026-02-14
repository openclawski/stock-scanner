import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import json
import os


# ─────────────────────────────────────────────────────────────
# STOCK UNIVERSE — 1134 tickers
# S&P 500, FTSE 100, DAX 40, CAC 40, Nikkei 225, ASX 50,
# TSX 60, Euro Stoxx, HK/China, India, Korea
# ─────────────────────────────────────────────────────────────

GLOBAL_TICKERS = [
    "AAPL","ABBV","ABT","ACN","ADBE","ADI","ADM","ADP","ADSK","AEE",
    "AEP","AES","AFL","AIG","AIZ","AJG","AKAM","ALB","ALGN","ALK",
    "ALL","ALLE","AMAT","AMCR","AMD","AME","AMGN","AMP","AMT","AMZN",
    "ANET","ANSS","AON","AOS","APA","APD","APH","APTV","ARE","ATO",
    "ATVI","AVB","AVGO","AVY","AWK","AXP","AZO","BA","BAC","BAX",
    "BBWI","BBY","BDX","BEN","BF-B","BIO","BIIB","BK","BKNG","BKR",
    "BLK","BMY","BR","BRK-B","BRO","BSX","BWA","BXP","C","CAG",
    "CAH","CARR","CAT","CB","CBOE","CBRE","CCI","CCL","CDAY","CDNS",
    "CDW","CE","CEG","CF","CFG","CHD","CHRW","CHTR","CI","CINF",
    "CL","CLX","CMA","CMCSA","CME","CMG","CMI","CMS","CNC","CNP",
    "COF","COO","COP","COST","CPB","CPRT","CPT","CRL","CRM","CSCO",
    "CSGP","CSX","CTAS","CTLT","CTRA","CTSH","CTVA","CVS","CVX","CZR",
    "D","DAL","DD","DE","DFS","DG","DGX","DHI","DHR","DIS",
    "DISH","DLTR","DOV","DOW","DPZ","DRI","DTE","DUK","DVA","DVN",
    "DXC","DXCM","EA","EBAY","ECL","ED","EFX","EIX","EL","EMN",
    "EMR","ENPH","EOG","EPAM","EQIX","EQR","EQT","ES","ESS","ETN",
    "ETR","ETSY","EVRG","EW","EXC","EXPD","EXPE","EXR","F","FANG",
    "FAST","FBHS","FCX","FDS","FDX","FE","FFIV","FIS","FISV","FITB",
    "FLT","FMC","FOX","FOXA","FRC","FRT","FTNT","FTV","GD","GE",
    "GILD","GIS","GL","GLW","GM","GNRC","GOOG","GOOGL","GPC","GPN",
    "GRMN","GS","GWW","HAL","HAS","HBAN","HCA","HD","HOLX","HON",
    "HPE","HPQ","HRL","HSIC","HST","HSY","HUM","HWM","IBM","ICE",
    "IDXX","IEX","IFF","ILMN","INCY","INTC","INTU","INVH","IP","IPG",
    "IQV","IR","IRM","ISRG","IT","ITW","IVZ","J","JBHT","JCI",
    "JKHY","JNJ","JNPR","JPM","K","KDP","KEY","KEYS","KHC","KIM",
    "KLAC","KMB","KMI","KMX","KO","KR","L","LDOS","LEN","LH",
    "LHX","LIN","LKQ","LLY","LMT","LNC","LNT","LOW","LRCX","LUMN",
    "LUV","LVS","LW","LYB","LYV","MA","MAA","MAR","MAS","MCD",
    "MCHP","MCK","MCO","MDLZ","MDT","MET","META","MGM","MHK","MKC",
    "MKTX","MLM","MMC","MMM","MNST","MO","MOH","MOS","MPC","MPWR",
    "MRK","MRNA","MRO","MS","MSCI","MSFT","MSI","MTB","MTCH","MTD",
    "MU","NCLH","NDAQ","NDSN","NEE","NEM","NFLX","NI","NKE","NOC",
    "NOW","NRG","NSC","NTAP","NTRS","NUE","NVDA","NVR","NWL","NWS",
    "NWSA","NXPI","O","ODFL","OGN","OKE","OMC","ON","ORCL","ORLY",
    "OTIS","OXY","PARA","PAYC","PAYX","PCAR","PCG","PEAK","PEG","PEP",
    "PFE","PFG","PG","PGR","PH","PHM","PKG","PKI","PLD","PM",
    "PNC","PNR","PNW","POOL","PPG","PPL","PRU","PSA","PSX","PTC",
    "PVH","PWR","PXD","PYPL","QCOM","QRVO","RCL","RE","REG","REGN",
    "RF","RHI","RJF","RL","RMD","ROK","ROL","ROP","ROST","RSG",
    "RTX","RVTY","SBAC","SBNY","SBUX","SCHW","SEE","SHW","SIVB","SJM",
    "SLB","SNA","SNPS","SO","SPG","SPGI","SRE","STE","STT","STX",
    "STZ","SWK","SWKS","SYF","SYK","SYY","T","TAP","TDG","TDY",
    "TECH","TEL","TER","TFC","TFX","TGT","TMO","TMUS","TPR","TRGP",
    "TRMB","TROW","TRV","TSCO","TSLA","TSN","TT","TTWO","TXN","TXT",
    "TYL","UAL","UDR","UHS","ULTA","UNH","UNP","UPS","URI","USB",
    "V","VFC","VICI","VLO","VMC","VNO","VRSK","VRSN","VRTX","VTR",
    "VTRS","VZ","WAB","WAT","WBA","WBD","WDC","WEC","WELL","WFC",
    "WHR","WM","WMB","WMT","WRB","WRK","WST","WTW","WY","WYNN",
    "XEL","XOM","XRAY","XYL","YUM","ZBH","ZBRA","ZION","ZTS","AAF.L",
    "AAL.L","ABF.L","ADM.L","AHT.L","ANTO.L","AUTO.L","AV.L","AVST.L","AVV.L","AZN.L",
    "BA.L","BARC.L","BATS.L","BDEV.L","BKG.L","BLND.L","BMEB.L","BNZL.L","BP.L","BRBY.L",
    "BT-A.L","CCH.L","CNA.L","CPG.L","CRDA.L","CRH.L","CTEC.L","DCC.L","DGE.L","ENT.L",
    "EXPN.L","FCIT.L","FLTR.L","FRES.L","GLEN.L","GSK.L","HIK.L","HLMA.L","HLN.L","HSBA.L",
    "IAG.L","IHG.L","III.L","IMB.L","INF.L","ITRK.L","ITV.L","JD.L","KGF.L","LAND.L",
    "LGEN.L","LLOY.L","LSEG.L","MNG.L","MNDI.L","MRO.L","NG.L","NWG.L","NXT.L","OCDO.L",
    "PHNX.L","PRU.L","PSH.L","PSN.L","PSON.L","REL.L","RIO.L","RKT.L","RMV.L","RR.L",
    "RS1.L","RTO.L","SBRY.L","SDR.L","SGE.L","SGRO.L","SHEL.L","SKG.L","SMDS.L","SMIN.L",
    "SMT.L","SN.L","SPX.L","SSE.L","STAN.L","SVT.L","TSCO.L","TW.L","ULVR.L","UTG.L",
    "UU.L","VOD.L","WEIR.L","WPP.L","WTB.L","1COV.DE","ADS.DE","AIR.DE","ALV.DE","BAS.DE",
    "BAYN.DE","BEI.DE","BMW.DE","BNR.DE","CON.DE","DB1.DE","DBK.DE","DHL.DE","DTE.DE","DTG.DE",
    "EOAN.DE","FME.DE","FRE.DE","HEI.DE","HEN3.DE","HNR1.DE","IFX.DE","KGX.DE","LIN.DE","MBG.DE",
    "MRK.DE","MTX.DE","MUV2.DE","P911.DE","PAH3.DE","PUM.DE","QIA.DE","RHM.DE","RWE.DE","SAP.DE",
    "SHL.DE","SIE.DE","SRT3.DE","SY1.DE","VNA.DE","VOW3.DE","ZAL.DE","AC.PA","AI.PA","AIR.PA",
    "ALO.PA","ATO.PA","BN.PA","BNP.PA","CA.PA","CAP.PA","CS.PA","DG.PA","DSY.PA","EL.PA",
    "EN.PA","ENGI.PA","ERF.PA","GLE.PA","HO.PA","KER.PA","LR.PA","MC.PA","ML.PA","MT.PA",
    "OR.PA","ORA.PA","PUB.PA","RI.PA","RMS.PA","RNO.PA","SAF.PA","SAN.PA","SGO.PA","STM.PA",
    "STMPA.PA","SU.PA","TEP.PA","TTE.PA","URW.PA","VIE.PA","VIV.PA","WLN.PA","6501.T","6502.T",
    "6503.T","6504.T","6506.T","6508.T","6586.T","6594.T","6645.T","6701.T","6702.T","6703.T",
    "6723.T","6724.T","6752.T","6753.T","6758.T","6762.T","6770.T","6841.T","6857.T","6861.T",
    "6902.T","6952.T","6954.T","6963.T","6971.T","6976.T","7003.T","7004.T","7011.T","7012.T",
    "7013.T","7201.T","7202.T","7203.T","7205.T","7211.T","7261.T","7267.T","7269.T","7270.T",
    "7272.T","7731.T","7733.T","7735.T","7741.T","7751.T","7752.T","7762.T","7832.T","7911.T",
    "7912.T","7951.T","7974.T","8001.T","8002.T","8015.T","8028.T","8031.T","8035.T","8053.T",
    "8058.T","8233.T","8252.T","8253.T","8267.T","8303.T","8304.T","8306.T","8308.T","8309.T",
    "8316.T","8331.T","8354.T","8355.T","8411.T","8601.T","8604.T","8628.T","8630.T","8725.T",
    "8750.T","8766.T","8795.T","8801.T","8802.T","8830.T","9001.T","9005.T","9007.T","9008.T",
    "9009.T","9020.T","9021.T","9022.T","9064.T","9101.T","9104.T","9107.T","9201.T","9202.T",
    "9301.T","9432.T","9433.T","9434.T","9501.T","9502.T","9503.T","9531.T","9532.T","9602.T",
    "9613.T","9735.T","9766.T","9983.T","9984.T","4063.T","4452.T","4502.T","4503.T","4506.T",
    "4507.T","4519.T","4523.T","4543.T","4568.T","4578.T","4661.T","4689.T","4704.T","4751.T",
    "4755.T","4901.T","4902.T","4911.T","5019.T","5020.T","5101.T","5108.T","5201.T","5202.T",
    "5214.T","5232.T","5233.T","5301.T","5332.T","5333.T","5401.T","5406.T","5411.T","5541.T",
    "5703.T","5706.T","5707.T","5711.T","5713.T","5714.T","5801.T","5802.T","5803.T","5901.T",
    "6103.T","6113.T","6178.T","6301.T","6302.T","6305.T","6326.T","6361.T","6367.T","6471.T",
    "ANZ.AX","BHP.AX","CBA.AX","CSL.AX","FMG.AX","GMG.AX","IAG.AX","JHX.AX","MQG.AX","NAB.AX",
    "NCM.AX","QBE.AX","RIO.AX","STO.AX","TCL.AX","TLS.AX","WBC.AX","WDS.AX","WES.AX","WOW.AX",
    "ALL.AX","AMC.AX","APA.AX","APX.AX","ASX.AX","BXB.AX","CAR.AX","COH.AX","COL.AX","CPU.AX",
    "DXS.AX","EVN.AX","FPH.AX","GPT.AX","IEL.AX","JBH.AX","LLC.AX","MGR.AX","MIN.AX","MPL.AX",
    "ORI.AX","ORG.AX","QAN.AX","REA.AX","RHC.AX","SCG.AX","SEK.AX","SGP.AX","SHL.AX","SUN.AX",
    "RY.TO","TD.TO","BNS.TO","BMO.TO","CM.TO","ENB.TO","CNR.TO","CP.TO","TRP.TO","BCE.TO",
    "MFC.TO","SLF.TO","ABX.TO","NTR.TO","CSU.TO","ATD.TO","SU.TO","FNV.TO","WCN.TO","IFC.TO",
    "QSR.TO","GIB-A.TO","DOL.TO","CCO.TO","AEM.TO","SAP.TO","TRI.TO","SHOP.TO","BAM-A.TO","FTS.TO",
    "IMO.TO","GWO.TO","NA.TO","WPM.TO","PPL.TO","EMA.TO","H.TO","CNQ.TO","AQN.TO","RBA.TO",
    "POW.TO","IAG.TO","OTEX.TO","CCL-B.TO","MRU.TO","CTC-A.TO","WFG.TO","FFH.TO","GFL.TO","BIP-UN.TO",
    "ASML.AS","INGA.AS","PHIA.AS","REN.AS","UNA.AS","WKL.AS","HEIA.AS","RAND.AS","AD.AS","AKZA.AS",
    "SAN.MC","BBVA.MC","IBE.MC","ITX.MC","TEF.MC","REP.MC","AMS.MC","FER.MC","GRF.MC","ENG.MC",
    "ENI.MI","ISP.MI","UCG.MI","RACE.MI","STMMI.MI","ENEL.MI","TIT.MI","G.MI","PST.MI","AMP.MI",
    "NESN.SW","NOVN.SW","ROG.SW","UBSG.SW","ABBN.SW","CSGN.SW","SREN.SW","ZURN.SW","GIVN.SW","LONN.SW",
    "VOLV-B.ST","ERIC-B.ST","ASSA-B.ST","SAND.ST","ATCO-A.ST","SEB-A.ST","SWED-A.ST","HM-B.ST","INVE-B.ST","ALFA.ST",
    "NESTE.HE","FORTUM.HE","UPM.HE","NOKIA.HE","SAMPO.HE","EQNR.OL","DNB.OL","MOWI.OL","TEL.OL","ORK.OL",
    "YAR.OL","SALM.OL","STB.OL","AKRBP.OL","AKER.OL","0700.HK","9988.HK","0005.HK","1299.HK","0941.HK",
    "2318.HK","0388.HK","0011.HK","0001.HK","0016.HK","0003.HK","0002.HK","0006.HK","0012.HK","0027.HK",
    "0066.HK","0083.HK","0101.HK","0175.HK","0241.HK","0267.HK","0288.HK","0386.HK","0669.HK","0688.HK",
    "0762.HK","0823.HK","0857.HK","0883.HK","0939.HK","0960.HK","0968.HK","1038.HK","1044.HK","1088.HK",
    "1109.HK","1113.HK","1177.HK","1211.HK","1398.HK","1810.HK","1876.HK","1928.HK","1997.HK","2007.HK",
    "2020.HK","2269.HK","2313.HK","2319.HK","2382.HK","2388.HK","2628.HK","3328.HK","3690.HK","3968.HK",
    "3988.HK","6098.HK","6618.HK","6862.HK","9618.HK","9633.HK","9888.HK","9961.HK","9999.HK","RELIANCE.NS",
    "TCS.NS","HDFCBANK.NS","INFY.NS","ICICIBANK.NS","HINDUNILVR.NS","SBIN.NS","BHARTIARTL.NS","KOTAKBANK.NS","ITC.NS","LT.NS",
    "AXISBANK.NS","BAJFINANCE.NS","ASIANPAINT.NS","MARUTI.NS","TITAN.NS","SUNPHARMA.NS","NESTLEIND.NS","ULTRACEMCO.NS","WIPRO.NS","HCLTECH.NS",
    "BAJAJFINSV.NS","TATASTEEL.NS","NTPC.NS","POWERGRID.NS","ONGC.NS","JSWSTEEL.NS","TECHM.NS","DIVISLAB.NS","TATAMOTORS.NS","ADANIENT.NS",
    "ADANIPORTS.NS","COALINDIA.NS","GRASIM.NS","HINDALCO.NS","BPCL.NS","CIPLA.NS","DRREDDY.NS","EICHERMOT.NS","HEROMOTOCO.NS","INDUSINDBK.NS",
    "M-M.NS","SBILIFE.NS","TATACONSUM.NS","UPL.NS","005930.KS","000660.KS","005380.KS","051910.KS","006400.KS","035420.KS",
    "035720.KS","068270.KS","105560.KS","028260.KS","012330.KS","066570.KS","055550.KS","034730.KS","003550.KS","032830.KS",
    "096770.KS","011200.KS","015760.KS","033780.KS",
]


def get_stock_universe():
    """Return hardcoded list of ~1000+ top global stocks."""
    return GLOBAL_TICKERS


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
# Long-term:   RSI( EMA(close, long_l1), long_l2 ) − 50
# Signal:      Value rising from previous bar → bullish (increasing)
#              Value falling → bearish (decreasing)
#
# We only track the long-term component (histogram).
# "Increasing" = the histogram is rising vs previous bar,
# regardless of whether it's above or below zero (colour).
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


def calculate_bx_trender(df, long_l1=20, long_l2=15):
    """
    Returns 1 (increasing) or 0 (decreasing) for the long-term BX component.
    Bullish = histogram rising from previous bar, regardless of colour.
    """
    if len(df) < 60:
        return 0

    close = df["Close"]

    # -- Long-term: RSI(EMA(close, long_l1), long_l2) - 50 --
    ema_l1 = close.ewm(span=long_l1, adjust=False).mean()
    long_raw = _rsi(ema_l1, long_l2) - 50.0

    # Signal: value rising from previous bar
    return 1 if float(long_raw.iloc[-1]) > float(long_raw.iloc[-2]) else 0


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

        # BX-Trender — weekly & monthly (long-term only, increasing = bullish)
        bx_w = calculate_bx_trender(df_w)
        bx_m = calculate_bx_trender(df_m)

        indicators = {
            "FVB_W_20": fvb_w_20,
            "FVB_W_33": fvb_w_33,
            "FVB_M_20": fvb_m_20,
            "FVB_M_33": fvb_m_33,
            "BX_W_Long": bx_w,
            "BX_M_Long": bx_m,
        }
        total = sum(indicators.values())
        price = round(float(df_w["Close"].iloc[-1]), 2)

        print(f"  {ticker}: {total}/6")
        return {
            "ticker": ticker,
            "name": name,
            "price": price,
            "indicators": indicators,
            "total_score": total,
            "max_score": 6,
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
        cls = "score-high" if sc >= 5 else "score-med" if sc >= 3 else "score-low"
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
            "BX_W_Long","BX_M_Long",
        ]:
            rows += f'<td><span class="indicator {dot(ind[k])}"></span></td>'
        rows += "</tr>\n"

    bull = len([r for r in results if r["total_score"] >= 5])
    neut = len([r for r in results if 3 <= r["total_score"] < 5])
    bear = len([r for r in results if r["total_score"] < 3])
    total = len(results)

    html = HTML_TEMPLATE
    html = html.replace("{TIMESTAMP}", timestamp)
    html = html.replace("{TOTAL}", str(total))
    html = html.replace("{BULLISH}", str(bull))
    html = html.replace("{NEUTRAL}", str(neut))
    html = html.replace("{BEARISH}", str(bear))
    return html.replace("{ROWS}", rows)


HTML_TEMPLATE = """<!DOCTYPE html>
<html><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Stock Scanner</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#0a0e27;color:#e0e6f0;padding:20px}
.container{max-width:1400px;margin:0 auto}
header{text-align:center;margin-bottom:40px;padding:30px 0;border-bottom:2px solid #1a2332}
h1{font-size:2.5em;margin-bottom:10px;color:#64ffda}
.last-updated{color:#8892b0;font-size:0.9em}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:20px;margin-bottom:30px}
.stat-card{background:#1a2332;padding:20px;border-radius:10px;border:1px solid #2a3548}
.stat-label{color:#8892b0;font-size:0.85em;margin-bottom:8px}
.stat-value{font-size:2em;font-weight:bold;color:#64ffda}
.table-wrap{overflow-x:auto}
table{width:100%;background:#1a2332;border-radius:10px;overflow:hidden;border-collapse:collapse;min-width:900px}
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
<p style="color:#8892b0;margin-bottom:5px">Fair Value Band + B-Xtrender (Long) | Weekly &amp; Monthly</p>
<div class="last-updated">Last Updated: {TIMESTAMP}</div></header>
<div class="stats">
<div class="stat-card"><div class="stat-label">Total Scanned</div><div class="stat-value">{TOTAL}</div></div>
<div class="stat-card"><div class="stat-label">Bullish (5-6)</div><div class="stat-value" style="color:#10b981">{BULLISH}</div></div>
<div class="stat-card"><div class="stat-label">Neutral (3-4)</div><div class="stat-value" style="color:#f59e0b">{NEUTRAL}</div></div>
<div class="stat-card"><div class="stat-label">Bearish (0-2)</div><div class="stat-value" style="color:#ef4444">{BEARISH}</div></div>
</div>
<div class="controls">
<div><label>Min Score</label><select id="minScore"><option value="0">All</option><option value="3">3+</option><option value="4">4+</option><option value="5">5+</option><option value="6">6/6</option></select></div>
<div><label>Search</label><input type="text" id="search" placeholder="Ticker or name..."></div>
</div>
<div class="table-wrap">
<table><thead><tr>
<th>Ticker</th><th>Name</th><th>Price</th><th>Score</th>
<th><span class="section-label">FVB</span>W 20</th>
<th><span class="section-label">FVB</span>W 33</th>
<th><span class="section-label">FVB</span>M 20</th>
<th><span class="section-label">FVB</span>M 33</th>
<th><span class="section-label">BX</span>W Long</th>
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
    bull = len([r for r in results if r["total_score"] >= 5])
    print(f"Bullish (5+/6): {bull}")


if __name__ == "__main__":
    main()

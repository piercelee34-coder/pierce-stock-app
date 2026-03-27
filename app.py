import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import requests
import xml.etree.ElementTree as ET
import re
import json
import time
import os
import streamlit.components.v1 as components

# --- 0. 系統設定 ---
st.set_page_config(page_title="AI 實戰戰情室 V17.6.2 (箱底邏輯修正版)", layout="wide", page_icon="🎯")

# --- CSS 美化 ---
st.markdown("""
<style>
    .price-card {background-color: #1e1e1e; padding: 20px; border-radius: 10px; text-align: center; border: 1px solid #333; margin-bottom: 10px;}
    .ai-box {background-color: #333; padding: 15px; border-radius: 10px; border: 1px solid #555; text-align: center; height: 100%;}
    .news-card {background-color: #262730; padding: 12px; border-radius: 5px; border-left: 4px solid #555; margin-bottom: 10px; font-size: 14px; transition: transform 0.2s;}
    .news-card:hover {transform: translateX(5px);}
    .news-link {text-decoration: none; color: #e0e0e0; font-weight: bold; display: block;}
    .news-link:hover {color: #FFD700;}
    .news-date {color: #aaa; font-size: 12px; margin-right: 5px;}
    .news-src {background-color: #444; color: #eee; border: 1px solid #777; padding: 1px 5px; border-radius: 3px; font-size: 11px; margin-right: 5px;}
    
    .macro-alert {background-color: #3a1b1b; color: #ff6b6b; padding: 10px; border-radius: 5px; border: 1px solid #dc3545; margin-bottom: 10px; font-weight: bold;}
    
    /* 錨定區塊樣式優化 */
    .anchor-box {background-color: #1b3a4a; color: #00d4ff; padding: 12px; border-radius: 5px; border: 1px solid #00d4ff; margin-bottom: 10px; font-size: 13px; text-align: left;}
    .anchor-title-cn {color: #fff; font-weight: bold; font-size: 14px; margin-bottom: 4px;}
    .anchor-title-en {color: #aaa; font-size: 11px; font-style: italic;}
    
    /* 財報與引擎資訊樣式 */
    .earnings-tag {background-color: #2c2c2e; padding: 5px 10px; border-radius: 15px; font-size: 13px; margin-top: 10px; border: 1px solid #555; display: inline-block; margin-right: 8px;}
    .engine-tag {background-color: #1e3a8a; color: #38bdf8; padding: 5px 10px; border-radius: 15px; font-size: 13px; margin-top: 10px; border: 1px solid #38bdf8; display: inline-block; font-weight: bold;}
    .earn-beat {color: #4ade80; font-weight: bold;}
    .earn-miss {color: #ff6b6b; font-weight: bold;}
    .earn-warn {color: #ffaa00; font-weight: bold;}
    .earn-turn {color: #facc15; font-weight: bold;}
    
    /* 標籤系統 */
    .tag-sec {background-color: #003366; color: #00ffff; border: 1px solid #00ffff; padding: 2px 8px; border-radius: 4px; font-size: 12px; font-weight: bold; margin-right: 5px;}
    .tag-vip {background-color: #4a1b4a; color: #d8b4fe; border: 1px solid #a855f7; padding: 2px 6px; border-radius: 4px; font-size: 12px; margin-right: 5px;}
    .tag-hard {background-color: #1b3a1b; color: #4ade80; border: 1px solid #28a745; padding: 2px 6px; border-radius: 4px; font-size: 12px; margin-right: 5px;}
    .tag-div {background-color: #4a3b1b; color: #ffaa00; border: 1px solid #ffaa00; padding: 2px 6px; border-radius: 4px; font-size: 12px; margin-right: 5px;}
    .tag-risk {background-color: #3a1b1b; color: #ff6b6b; border: 1px solid #dc3545; padding: 2px 6px; border-radius: 4px; font-size: 12px; font-weight: bold;}
    .tag-chip {background-color: #555; color: #facc15; border: 1px solid #facc15; padding: 2px 6px; border-radius: 4px; font-size: 12px; font-weight: bold;}

    /* 戰略雷達信號燈 */
    .sig-green {background-color: #1b3a1b; color: #4ade80; border: 1px solid #28a745; padding: 2px 6px; border-radius: 4px; font-size: 12px;}
    .sig-red {background-color: #3a1b1b; color: #ff6b6b; border: 1px solid #dc3545; padding: 2px 6px; border-radius: 4px; font-size: 12px;}
    .sig-gray {background-color: #333; color: #ccc; border: 1px solid #666; padding: 2px 6px; border-radius: 4px; font-size: 12px;}
    .sig-orange {background-color: #4a3b1b; color: #ffaa00; border: 1px solid #ffaa00; padding: 2px 6px; border-radius: 4px; font-size: 12px;}
    .sig-blue {background-color: #1b3a4a; color: #4a9eff; border: 1px solid #00d4ff; padding: 2px 6px; border-radius: 4px; font-size: 12px;}
    .sig-purple {background-color: #4a1b4a; color: #d8b4fe; border: 1px solid #a855f7; padding: 2px 6px; border-radius: 4px; font-size: 12px;}
    .sig-cyan {background-color: #083344; color: #22d3ee; border: 1px solid #06b6d4; padding: 2px 6px; border-radius: 4px; font-size: 12px;}
</style>
""", unsafe_allow_html=True)

# --- 1. 自選股儲存系統 ---
WATCHLIST_FILE = "watchlist.json"
ANCHOR_FILE = "anchors.json"
DEFAULT_LIST = ['^IXIC', 'QQQ', 'NVDA', 'TSM', 'MU', 'ONDS', 'RXRX', 'CRCL', 'AAPL', 'TSLA', '0050.TW']

def load_watchlist():
    if os.path.exists(WATCHLIST_FILE):
        try:
            with open(WATCHLIST_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except: return DEFAULT_LIST
    return DEFAULT_LIST

def save_watchlist(watchlist):
    try:
        with open(WATCHLIST_FILE, "w", encoding="utf-8") as f:
            json.dump(watchlist, f)
        st.session_state.watchlist = watchlist
    except: pass

def load_anchors():
    if os.path.exists(ANCHOR_FILE):
        try:
            with open(ANCHOR_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except: return {}
    return {}

def save_anchor_data(data):
    try:
        with open(ANCHOR_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f)
    except: pass

def simple_translate(text):
    text_lower = text.lower()
    if "insider buy" in text_lower or "insider purchase" in text_lower: return "內部人買進"
    if "insider sale" in text_lower or "insider sell" in text_lower: return "內部人拋售"
    if "earnings beat" in text_lower or "tops estimates" in text_lower: return "財報優於預期"
    if "revenue growth" in text_lower: return "營收成長"
    if "options" in text_lower and "volume" in text_lower: return "期權成交異動"
    if "record high" in text_lower: return "創歷史新高"
    if "upgrade" in text_lower: return "機構升評"
    if "downgrade" in text_lower: return "機構降評"
    return "重大情報"

def update_anchor(ticker, news_title, score, news_date_str):
    anchors = load_anchors()
    summary_cn = simple_translate(news_title)
    anchors[ticker] = {"title": news_title, "summary": summary_cn, "score": score, "date": news_date_str, "saved_at": datetime.now().strftime("%Y-%m-%d")}
    save_anchor_data(anchors)

def get_valid_anchor(ticker):
    anchors = load_anchors()
    if ticker not in anchors: return None
    data = anchors[ticker]
    try:
        if (datetime.now() - datetime.strptime(data["saved_at"], "%Y-%m-%d")).days > 5:
            del anchors[ticker]
            save_anchor_data(anchors)
            return None
    except: return None
    return data

if 'watchlist' not in st.session_state:
    st.session_state.watchlist = load_watchlist()

# --- 2. 核心搜尋引擎 ---
def get_ticker_metadata(ticker):
    mapping = {
        'NVDA': {'name': '輝達', 'ceo': ['黃仁勳', 'Jensen', 'Huang'], 'key': ['Nvidia']},
        'TSLA': {'name': '特斯拉', 'ceo': ['馬斯克', 'Elon', 'Musk'], 'key': ['Tesla']},
        'AAPL': {'name': '蘋果', 'ceo': ['庫克', 'Tim', 'Cook'], 'key': ['Apple', 'iPhone']},
        'MU': {'name': '美光', 'ceo': ['Sanjay', 'Mehrotra'], 'key': ['Micron']},
        'TSM': {'name': '台積電', 'ceo': ['魏哲家'], 'key': ['TSMC']},
    }
    base = ticker.split('.')[0]
    return mapping.get(base, {'name': ticker, 'ceo': [], 'key': [ticker]})

def validate_news_relevance(title, ticker, info, strict_mode=True):
    t = title.lower()
    base_ticker = ticker.split('.')[0].lower()
    whitelist = [base_ticker] + [k.lower() for k in info['key']]
    if info['name'] != ticker: whitelist.append(info['name'].lower())
    for c in info['ceo']: whitelist.append(c.lower())
    hit = any(w in t for w in whitelist)
    if not strict_mode:
        if base_ticker in t and any(k in t for k in ['options', 'volume', 'shares', 'trading', '期權', '成交', '異動', '大漲', '大跌']):
            return True
    return hit

def fetch_deep_news(ticker, is_macro=False):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        is_tw_stock = ".TW" in ticker or ".TWO" in ticker
        items = []
        now = datetime.now()
        
        if is_macro:
            resp = requests.get("https://news.google.com/rss/search?q=聯準會+升息+通膨+鮑爾&hl=zh-TW&gl=TW&ceid=TW:zh-Hant", headers=headers, timeout=4)
            if resp.status_code == 200:
                for item in ET.fromstring(resp.content).findall('.//item')[:6]:
                    data = parse_rss_item(item, "macro", "Global")
                    if (now - data['dt']).days <= 30: items.append(data)
            return items

        ticker_info = get_ticker_metadata(ticker)
        cn_name = ticker_info['name']
        search_target = f"{ticker} OR {cn_name}" if cn_name else ticker
        is_small_cap = ticker.split('.')[0] in ['ONDS', 'RXRX', 'CRCL', 'SOUN', 'PLTR'] 
        
        if is_tw_stock:
            q_tw = f"{ticker.replace('.TW', '').replace('.TWO', '')}+(重訊+OR+重大訊息+OR+營收+OR+公告+OR+自結+OR+EPS+OR+配息+OR+法說)"
            try:
                resp = requests.get(f"https://news.google.com/rss/search?q={q_tw}&hl=zh-TW&gl=TW&ceid=TW:zh-Hant", headers=headers, timeout=4)
                if resp.status_code == 200:
                    for item in ET.fromstring(resp.content).findall('.//item')[:10]:
                        data = parse_rss_item(item, "tw_local", "台股重訊")
                        if (now - data['dt']).days <= 30 and validate_news_relevance(data['title'], ticker, ticker_info):
                            items.append(data)
            except: pass
        else:
            q_sec = f"{ticker}+stock+(SEC+Filing+OR+Form+4+OR+10-Q+OR+8-K+OR+Insider+Trading)"
            q_news = f"{ticker}+stock+(Options+OR+Volume+OR+Implied+Volatility+OR+期權+OR+成交+OR+異動+OR+財報)" if is_small_cap else f"{search_target}+stock+(財聯社+OR+鉅亨網+OR+營收+OR+訂單+OR+CEO+OR+財報)"

            for url, cat, src_name in [(f"https://news.google.com/rss/search?q={q_sec}&hl=en-US&gl=US&ceid=US:en", "us_sec", "🏛️ SEC"), (f"https://news.google.com/rss/search?q={q_news}&hl=zh-TW&gl=TW&ceid=TW:zh-Hant", "us_news", "📰 News")]:
                try:
                    resp = requests.get(url, headers=headers, timeout=4)
                    if resp.status_code == 200:
                        for item in ET.fromstring(resp.content).findall('.//item')[:8]:
                            data = parse_rss_item(item, cat, src_name)
                            if (now - data['dt']).days <= 30 and validate_news_relevance(data['title'], ticker, ticker_info, strict_mode=not is_small_cap):
                                items.append(data)
                except: pass
        items.sort(key=lambda x: x['dt'], reverse=True)
        return items
    except: return []

def parse_rss_item(item, category, source_name):
    title = re.sub('<[^<]+?>', '', item.find('title').text)
    link = item.find('link').text
    try: dt = datetime.strptime(item.find('pubDate').text[:16], '%a, %d %b %Y')
    except: dt = datetime.now()
    if "futunn" in link: source_name = "🐂 富途"
    elif "yahoo" in link: source_name = "🇺🇸 Yahoo"
    return {'title': title, 'link': link, 'date': dt.strftime('%m/%d'), 'dt': dt, 'cat': category, 'src': source_name}

def analyze_news_weight_strict(title, category):
    t = title.lower()
    score = 0; tag = ""; is_major = False; insider_penalty = 0.0
    if any(w in t for w in ['豪宅', '買房', '神操作', '財富自由', '名師', '懶人包']): return 0, "", False, 0
    for w in ['減持', '賣出', '劣於', '降評', '大跌', '崩盤', 'underweight', 'miss']:
        if w in t and 'form 4' not in t and 'insider' not in t: return -2.0, '<span class="tag-red">Risk</span>', False, 0

    if any(x in t for x in ['options', 'volume', '期權', '成交', '異動']):
        score, tag = (2.0, '<span class="tag-chip">🌊 籌碼異動</span>') if any(x in t for x in ['high', 'surge', 'active', '大增']) else (0.5, '<span class="tag-gray">📊 籌碼面</span>')
    elif any(x in t for x in ['營收', 'revenue', 'eps', 'profit', '獲利', '財報', 'earnings', '重訊']):
        if any(x in t for x in ['新高', 'beat', 'up', '增', '漲', '超預期']): score, tag, is_major = 4.0, '<span class="tag-filing">💎 財報/營運利多</span>', True
        elif any(x in t for x in ['miss', 'down', 'loss', 'cut', '減', '虧', '不如']): score, tag = -3.0, '<span class="tag-red">📉 財報利空</span>'
        else: score, tag = 1.5, '<span class="tag-div">📊 財務數據</span>'
    elif category == "us_sec" or 'form 4' in t or 'insider' in t:
        if any(x in t for x in ['buy', '買進', '增持']): score, tag, is_major = 4.0, '<span class="tag-vip">👑 VIP買進</span>', True
        elif any(x in t for x in ['sell', '賣出', '減持']):
            if any(p in t for p in ['cfo', '財務長']): score, insider_penalty, tag = -4.0, -4.0, '<span class="tag-risk">⚠️ CFO拋售</span>'
            elif any(p in t for p in ['ceo', '執行長']): score, insider_penalty, tag = -3.5, -3.5, '<span class="tag-risk">⚠️ CEO拋售</span>'
            else: score, insider_penalty, tag = -1.5, -1.5, '<span class="tag-gray">內部人賣出</span>'
    elif any(x in t for x in ['order', 'contract', '訂單']): score, tag = 3.0, '<span class="tag-hard">🔥 實質訂單</span>'

    if not tag:
        if score > 0: tag = '<span class="tag-hard">📈 利多</span>'
        elif score < 0: tag = '<span class="tag-red">📉 利空</span>'
    return score, tag, is_major, insider_penalty

def get_realtime_macro():
    news = fetch_deep_news("Macro", is_macro=True)
    news_score = sum(-1.5 if any(w in n['title'].lower() for w in ['hike', 'inflation', '升息', '鷹']) else 1 if any(w in n['title'].lower() for w in ['cut', 'pause', '降息', '鴿']) else 0 for n in news)
    txt, note, col, score = "宏觀穩健", "多頭支撐", "sig-green", 0
    if news_score <= -3: txt, note, col, score = "Fed 偏鷹", "系統風險高", "sig-red", -2
    elif news_score < 0: txt, note, col, score = "宏觀偏空", "震盪觀望", "sig-orange", -1
    try:
        hist = yf.Ticker("^IXIC").history(period="5d")
        if len(hist) >= 2:
            pct_chg = (hist.iloc[-1]['Close'] - hist.iloc[-2]['Close']) / hist.iloc[-2]['Close'] * 100
            if pct_chg < -1.5: txt, note, col, score = "市場恐慌", f"納指重挫 {pct_chg:.2f}%", "sig-red", -3
            elif pct_chg < -0.8: txt, note, col, score = "市場修正", f"納指下跌 {pct_chg:.2f}%", "sig-orange", -1.5
    except: pass
    return txt, note, col, score

# --- 3. 技術指標與圖表核心 ---
def calculate_volume_profile(df, bins=40, filter_mask=None):
    if df.empty: return pd.DataFrame({'Price': [], 'Volume': []})
    p_min, p_max = df['Low'].min(), df['High'].max()
    edges = np.linspace(p_min, p_max, bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    sub = df if filter_mask is None else df[filter_mask]
    if sub.empty: return pd.DataFrame({'Price': centers, 'Volume': np.zeros(bins)})
    idx = pd.cut(sub['Close'], bins=edges, labels=False, include_lowest=True)
    return pd.DataFrame({'Price': centers, 'Volume': sub.groupby(idx)['Volume'].sum().reindex(range(bins), fill_value=0).values})

def calculate_indicators(df):
    if len(df) < 50: return df
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_60'] = df['Close'].rolling(window=60).mean()
    df['Vol_SMA5'] = df['Volume'].rolling(window=5).mean()
    df['Std_Dev'] = df['Close'].rolling(window=20).std()
    
    df['Bollinger_Upper'] = df['SMA_20'] + (df['Std_Dev'] * 2)
    df['Bollinger_Lower'] = df['SMA_20'] - (df['Std_Dev'] * 2)
    df['KC_Upper'] = df['SMA_20'] + (df['SMA_20'] * 0.05) 
    df['KC_Lower'] = df['SMA_20'] - (df['SMA_20'] * 0.05)
    df['Squeeze_On'] = (df['Bollinger_Upper'] < df['KC_Upper']) & (df['Bollinger_Lower'] > df['KC_Lower'])
    
    tr = np.max(pd.concat([df['High']-df['Low'], np.abs(df['High']-df['Close'].shift()), np.abs(df['Low']-df['Close'].shift())], axis=1), axis=1)
    df['ATR'] = tr.rolling(14).mean()
    df['ATR_Pct'] = (df['ATR'] / df['Close']) * 100
    df['Vol_60D_Avg'] = df['ATR_Pct'].rolling(window=60).mean()
    df['ATR_Trailing_Stop'] = df['High'].rolling(22).max() - (df['ATR'] * 3)

    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    df['RSI'] = 100 - (100 / (1 + gain/loss))
    
    df['MACD'] = df['Close'].ewm(span=12, adjust=False).mean() - df['Close'].ewm(span=26, adjust=False).mean()
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['Signal_Line']
    
    clv = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'])
    df['AD_Line'] = (clv.fillna(0) * df['Volume']).cumsum()
    df['DMA_DDD'] = df['Close'].rolling(window=10).mean() - df['Close'].rolling(window=50).mean()
    df['DMA_AMA'] = df['DMA_DDD'].rolling(window=10).mean()
    
    close = df['Close'].values
    buy_seq, sell_seq = np.zeros(len(close), dtype=int), np.zeros(len(close), dtype=int)
    for i in range(4, len(close)):
        buy_seq[i] = buy_seq[i-1] + 1 if close[i] < close[i-4] else 0
        sell_seq[i] = sell_seq[i-1] + 1 if close[i] > close[i-4] else 0
    df['TD_Buy_9'] = np.where(buy_seq == 9, close, np.nan)
    df['TD_Sell_9'] = np.where(sell_seq == 9, close, np.nan)
    df['TD_Buy_Stop'] = np.nan
    for i in range(len(close)):
        if buy_seq[i] == 9: df.loc[df.index[i], 'TD_Buy_Stop'] = df['Low'].iloc[max(0, i-8):i+1].min()
    return df

# [V17.6.2] AI 歷史箱底探測器演算法 (修復：改找最近一次前波低點)
def find_structural_box_bottom(df, current_price):
    if len(df) < 60: 
        return current_price, df.index[0], df.index[-1], False
    
    p_data = df.tail(120).copy()
    
    # 尋找 V 型波谷 (前後 11 天內的最低點，這代表一個完整的兩週盤整底)
    p_data['Min10'] = p_data['Low'].rolling(window=5, center=True, min_periods=1).min()
    swing_lows = p_data[p_data['Low'] == p_data['Min10']]
    
    # 過濾條件：這個低點必須「低於現價的 98%」
    valid_lows = swing_lows[swing_lows['Low'] < current_price * 0.98]
    
    is_breaking_down = False
    
    if not valid_lows.empty:
        # 🛠️ 核心修正：找「距離現在最近」的那個波段低谷 (前波起漲點/頸線)，而不是半年前的歷史底！
        iron_bottom_row = valid_lows.iloc[-1]
        iron_price = iron_bottom_row['Low']
        
        iron_idx = np.where(p_data.index == iron_bottom_row.name)[0][0]
        start_idx = max(0, iron_idx - 15)
        end_idx = min(len(p_data) - 1, iron_idx + 15)
        
        return iron_price, p_data.index[start_idx], p_data.index[end_idx], False
    else:
        # ⚠️ 防呆機制啟動：找不到比現價更低的波谷，代表正在創 120 天新低！
        is_breaking_down = True
        return p_data['Low'].min(), p_data.index[0], p_data.index[-1], True

@st.cache_data(ttl=3600)
def get_stock_engine_mode(ticker, df_data):
    etf_list = ["QQQ", "SPY", "DIA", "IWM", "0050.TW", "0056.TW", "00878.TW"]
    if ticker.startswith("^") or any(etf in ticker for etf in etf_list): return "🏢 權值大盤 (強制 MA60 濾網)", "trend"
    try:
        mcap = yf.Ticker(ticker).info.get('marketCap', 0)
        vol = df_data['Vol_60D_Avg'].iloc[-1] if not df_data.empty else 3.0
        is_large = mcap >= (300e9 if ".TW" in ticker else 10e9) or (mcap == 0 and vol < 3.5)
        if is_large and vol < 4.0: return "🏢 權值穩健 (啟動 MA60 濾網)", "trend"
        elif is_large: return "🚀 巨型動能 (啟動 MA20 濾網)", "momentum"
        else: return "🎢 妖股轉折 (關閉均線濾網)", "reversal"
    except: return "🎢 動態模式 (預設)", "reversal"

def get_relative_strength(ticker, stock_df):
    try:
        bench = yf.download("^TWII" if ".TW" in ticker else "^GSPC", period="1mo", progress=False)['Close'].iloc[:, 0]
        aligned = stock_df['Close'].reindex(bench.index, method='ffill')
        if len(bench) > 20:
            diff = ((aligned.iloc[-1] - aligned.iloc[-20]) / aligned.iloc[-20]) - ((bench.iloc[-1] - bench.iloc[-20]) / bench.iloc[-20])
            if diff > 0.05: return "🦁 領頭羊 (強)", "sig-green"
            elif diff > 0: return "🐯 優於大盤", "sig-blue"
            else: return "🐶 落後股 (弱)", "sig-gray"
    except: pass
    return "⚖️ 跟隨大盤", "sig-gray"

def detect_smart_money_status(df):
    if len(df) < 10: return None
    latest = df.iloc[-1]
    if latest['Close'] < latest['Bollinger_Lower'] and latest['RSI'] < 30: return "⚡ 乖離抄底 (超賣)"
    if latest['RSI'] > 65 and latest['Volume'] > latest['Vol_SMA5'] * 1.3 and (latest['Close'] < latest['Open'] or (latest['High'] - max(latest['Open'], latest['Close']) > abs(latest['Close'] - latest['Open']) * 1.5)):
        return "🔴 主力調節 (爆量滯漲)"
    if latest['RSI'] < 30 and latest['Volume'] > latest['Vol_SMA5']: return "⚡ 恐慌殺盤"
    return None

def analyze_strategic_signals(df):
    if df.empty: return {}
    latest = df.iloc[-1]
    macd, signal = latest['MACD'], latest['Signal_Line']
    macd_text, macd_color = ("零軸上金叉", "sig-green") if macd > signal and macd > 0 else ("零軸下金叉", "sig-orange") if macd > signal else ("零軸上死叉", "sig-orange") if macd > 0 else ("零軸下死叉", "sig-red")
    vol, vol_ma = latest['Volume'], latest['Vol_SMA5']
    vol_text, vol_color = ("爆量", "sig-green") if vol > vol_ma * 1.5 else ("量增", "sig-green") if vol > vol_ma * 1.1 else ("量縮", "sig-gray")
    rsi = latest['RSI']
    rsi_text, rsi_color = (f"過熱 ({rsi:.0f})", "sig-red") if rsi > 70 else (f"超賣 ({rsi:.0f})", "sig-green") if rsi < 30 else (f"中性 ({rsi:.0f})", "sig-gray")
    
    summary, summary_color = "觀望", "sig-gray"
    if latest.get('Squeeze_On', False): summary, summary_color = "🌀 壓縮蓄力中", "sig-cyan"
    elif macd > signal: summary, summary_color = "📈 偏多震盪", "sig-green"
    else: summary, summary_color = "⛈️ 空頭走勢", "sig-red"
    status = detect_smart_money_status(df)
    if status: summary, summary_color = status, "sig-red" if "調節" in status else "sig-purple"
    
    return {"MACD_Text": macd_text, "MACD_Color": macd_color, "Vol_Text": vol_text, "Vol_Color": vol_color, "RSI_Text": rsi_text, "RSI_Color": rsi_color, "Summary": summary, "Summary_Color": summary_color}

def analyze_market_trend(df):
    price, ma20, ma60 = df['Close'].iloc[-1], df['SMA_20'].iloc[-1], df['SMA_60'].iloc[-1]
    if price > ma20 > ma60: return "🐂 牛市 (Bull)", "sig-green"
    elif price < ma20 < ma60: return "🐻 熊市 (Bear)", "sig-red"
    else: return "⚖️ 震盪 (Range)", "sig-orange"

def predict_target_and_rating(df):
    price, upper, recent_high_60 = df['Close'].iloc[-1], df['Bollinger_Upper'].iloc[-1], df['High'].tail(60).max()
    t_s = upper if price >= recent_high_60 else min(upper, recent_high_60)
    return t_s, max(recent_high_60 * 1.15, t_s * 1.1), "強勢" if price > df['SMA_20'].iloc[-1] else "持有"

def format_volume(num): return f"{num/1e9:.2f}B" if num >= 1e9 else f"{num/1e6:.2f}M" if num >= 1e6 else f"{num}"

@st.cache_data(ttl=3600)
def get_earnings_status(ticker):
    ignore_list = ["0050", "0056", "QQQ", "SPY", "DIA", "IWM", "^TWII", "^IXIC"]
    if any(x in ticker for x in ignore_list): return "", ""
    try:
        t = yf.Ticker(ticker); next_date = t.calendar['Earnings Date'][0].strftime('%Y-%m-%d') if t.calendar and 'Earnings Date' in t.calendar else "N/A"
        last_result = "⚪ 無數據"
        try:
            hist = t.earnings_dates
            if hist is not None and not hist.empty:
                last = hist[hist['Reported EPS'].notna()].iloc[0]
                act, est = last['Reported EPS'], last.get('EPS Estimate', np.nan)
                if pd.notna(act) and pd.notna(est):
                    if act > 0 and act >= est: last_result = f'<span class="earn-beat">🟢 獲利優於預期 (EPS: {act:.2f}|估:{est:.2f})</span>'
                    elif act <= 0 and act >= est: last_result = f'<span class="earn-turn">🟡 虧損優於預期 (EPS: {act:.2f}|估:{est:.2f})</span>'
                    elif act > 0 and act < est: last_result = f'<span class="earn-warn">🟠 獲利遜於預期 (EPS: {act:.2f}|估:{est:.2f})</span>'
                    else: last_result = f'<span class="earn-miss">🔴 虧損遜於預期 (EPS: {act:.2f}|估:{est:.2f})</span>'
        except: pass
        return f"📅 財報: {next_date}", last_result
    except: return "📅 財報: N/A", "⚪ 無數據"

# --- 6. 主介面 ---
with st.sidebar:
    st.title("🎛️ 控制台")
    st.header("📌 自選股清單")
    current_ticker = st.radio("選擇股票", st.session_state.watchlist)
    is_tw = ".TW" in current_ticker or ".TWO" in current_ticker

    c_up, c_down = st.columns(2)
    if c_up.button("⬆️ 上移") and current_ticker in st.session_state.watchlist:
        idx = st.session_state.watchlist.index(current_ticker)
        if idx > 0:
            st.session_state.watchlist[idx], st.session_state.watchlist[idx-1] = st.session_state.watchlist[idx-1], st.session_state.watchlist[idx]
            save_watchlist(st.session_state.watchlist); st.rerun()
    if c_down.button("⬇️ 下移") and current_ticker in st.session_state.watchlist:
        idx = st.session_state.watchlist.index(current_ticker)
        if idx < len(st.session_state.watchlist) - 1:
            st.session_state.watchlist[idx], st.session_state.watchlist[idx+1] = st.session_state.watchlist[idx+1], st.session_state.watchlist[idx]
            save_watchlist(st.session_state.watchlist); st.rerun()

    st.markdown("---")
    time_opt = st.radio("選擇週期", ["當沖 (分時)", "日線 (Daily)", "週線 (Weekly)", "月線 (長線)"], index=1)
    st.markdown("---")
    with st.expander("編輯清單"):
        new_t = st.text_input("代號", placeholder="MSTR").upper()
        if st.button("➕ 新增") and new_t and new_t not in st.session_state.watchlist:
            st.session_state.watchlist.append(new_t); save_watchlist(st.session_state.watchlist); st.rerun()
        if st.button("❌ 刪除") and current_ticker in st.session_state.watchlist:
            st.session_state.watchlist.remove(current_ticker); save_watchlist(st.session_state.watchlist); st.rerun()
    
    with st.expander("📖 雙防線系統解讀"):
        st.markdown("""
        🎯 **波段探底** ➔ 順勢支撐，適合大牛市回測接刀。
        🧱 **歷史箱底** ➔ 結構鐵板，大回檔時的絕佳防線/長多牛熊分界線。
        🕳️ **破底警告** ➔ 找不到歷史支撐，代表正在殺盤破底，嚴禁接刀！
        """)

st.title(f"📈 {current_ticker} 實戰戰情室 V17.6.2")

api_p, api_i = ("5d", "15m") if "當沖" in time_opt else ("6mo", "1d") if "日" in time_opt else ("2y", "1wk")
df = yf.download(current_ticker, period=api_p, interval=api_i, progress=False)
if df.empty: st.error("無數據"); st.stop()
if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
df = df.loc[:, ~df.columns.duplicated()]
df = calculate_indicators(df)

latest = df.iloc[-1]; prev = df.iloc[-2]
close_v = float(latest['Close']); prev_v = float(prev['Close'])
chg = (close_v - prev_v) / prev_v * 100
clr = "green" if chg >= 0 else "red"

sigs = analyze_strategic_signals(df)
trend_txt, trend_col = analyze_market_trend(df)
rs_txt, rs_col = get_relative_strength(current_ticker, df)
engine_label, engine_type = get_stock_engine_mode(current_ticker, df)
macro_txt, macro_note, macro_col, macro_score = get_realtime_macro()
t_s, t_l, rating = predict_target_and_rating(df)

# 🎯 第一防線：波段探底 (動態)
vp_60 = calculate_volume_profile(df.tail(60), bins=40)
vol_poc = vp_60.loc[vp_60['Volume'].idxmax(), 'Price'] if not vp_60.empty else close_v
if engine_type in ["trend", "momentum"]:
    wave_bottom = f"🎯 波段探底: ${max(latest.get('SMA_60', 0), vol_poc):.2f} (大戶防守/籌碼區)"
else:
    wave_bottom = f"🎯 恐慌探底: ${min(latest.get('Bollinger_Lower', 0), df['Low'].tail(20).min()):.2f} (極端超賣/嚴設停損)"

# 🧱 第二防線：AI 歷史箱底探測 (結構)
iron_price, box_start, box_end, is_breaking_down = find_structural_box_bottom(df, close_v)

if is_breaking_down:
    buy_hint = "⚠️ 警戒！恐慌殺盤，跌破所有支撐"
    iron_html = f'<div style="margin-bottom: 10px; color: #ff4500; font-weight: bold; font-size: 20px;">🕳️ 破底警告: ${iron_price:.2f} (創 120 天新低 / 無歷史支撐！)</div>'
else:
    buy_hint = "觀望，等待訊號" if sigs['Summary'] == "觀望" else sigs['Summary']
    iron_html = f'<div style="margin-bottom: 10px; color: #facc15; font-weight: bold; font-size: 20px;">🧱 歷史箱底: ${iron_price:.2f} (結構鐵板/前波起漲點)</div>'

ern_date, ern_res = get_earnings_status(current_ticker)
ern_html = f'<div class="earnings-tag">{ern_date} | {ern_res}</div>' if ern_date else ""

# 主面板顯示
st.markdown(f"""
<div class="price-card">
    <h1 style="margin:0; font-size: 50px;">${close_v:.2f}</h1>
    <h3 style="margin:0; color: {clr};">{chg:+.2f}%</h3>
    <p style="color: gray;">量: {format_volume(latest['Volume'])}</p>
    <div style="margin-bottom: 5px; font-size: 15px;">💡 操作提示: {buy_hint}</div>
    <div style="margin-bottom: 5px; color: #00ffff; font-weight: bold; font-size: 20px;">{wave_bottom}</div>
    {iron_html}
    <div class="engine-tag">⚙️ {engine_label}</div>
    {ern_html}
</div>
""", unsafe_allow_html=True)

r1, r2, r3 = st.columns(3)
with r1: st.markdown(f'<div class="ai-box"><h5 style="color:white; margin:0;">📡 綜合戰略</h5><div style="font-size:16px; margin-top:5px;" class="{sigs['Summary_Color']}">{sigs['Summary']}</div></div>', unsafe_allow_html=True)
with r2: st.markdown(f'<div class="ai-box"><h5 style="color:white; margin:0;">⚖️ 雙重格局</h5><div style="margin-top:5px;">個股: <span class="{trend_col}">{trend_txt}</span><br>宏觀: <span class="{macro_col}">{macro_txt}</span></div></div>', unsafe_allow_html=True)
with r3: st.markdown(f'<div class="ai-box" style="border: 1px solid #00d4ff;"><h5 style="color:white; margin:0;">🎯 AI 目標 & 強弱</h5><div style="margin-top:5px;">短: ${t_s:.2f} | 長: ${t_l:.2f}<br><span class="{rs_col}">{rs_txt}</span></div></div>', unsafe_allow_html=True)

# 繪圖區塊
p_data = df.tail(120) if "日" in time_opt else df.tail(60)
fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_width=[0.2, 0.2, 0.6])
fig.add_trace(go.Candlestick(x=p_data.index, open=p_data['Open'], high=p_data['High'], low=p_data['Low'], close=p_data['Close'], name="K線"), row=1, col=1)
fig.add_trace(go.Scatter(x=p_data.index, y=p_data['ATR_Trailing_Stop'], mode='lines', line=dict(color='#FF5F1F', width=1.5, dash='dot'), name='ATR 停損'), row=1, col=1)

# [V17.6.2] 繪製終極鐵板線 (青色) 與 歷史箱型區塊 (淡藍色)
if not is_breaking_down and iron_price > 0:
    fig.add_hline(y=iron_price, line_dash="dash", line_color="#20c997", annotation_text="🧱 終極鐵板", annotation_font_color="#20c997", row=1, col=1)
    fig.add_shape(type="rect", x0=box_start, y0=iron_price * 0.95, x1=box_end, y1=iron_price * 1.05, fillcolor="#00d4ff", opacity=0.15, layer="below", line_width=1, line_color="#00d4ff", line_dash="dot", row=1, col=1)

# 完全恢復所有圖表標籤 (吸籌/抄底/達標/吞噬/九轉/BUY/SELL)
for i in range(5, len(p_data)):
    curr = p_data.iloc[i]; prior = p_data.iloc[i-1]
    
    if (prior['Close'] < prior['Open']) and (curr['Close'] > curr['Open']) and (curr['Open'] <= prior['Close']) and (curr['Close'] >= prior['Open']):
        fig.add_annotation(x=p_data.index[i], y=curr['Low']*0.98, text="🕯️吞噬", showarrow=True, arrowhead=1, ay=40, row=1, col=1, font=dict(color="orange", size=8))

    if not np.isnan(curr.get('TD_Buy_9', np.nan)):
        fig.add_annotation(x=p_data.index[i], y=curr['Low'], text="9", showarrow=False, font=dict(color='#ff6b6b', size=12, weight="bold"), row=1, col=1)
    if not np.isnan(curr.get('TD_Sell_9', np.nan)):
        fig.add_annotation(x=p_data.index[i], y=curr['High'], text="9", showarrow=False, font=dict(color='#4a9eff', size=12, weight="bold"), row=1, col=1)

    status = detect_smart_money_status(df.iloc[:i+1])
    if status:
        if "吸籌" in status:
            fig.add_annotation(x=p_data.index[i], y=curr['Low']*0.98, text=f"🐳吸<br>${curr['Low']:.1f}", showarrow=True, arrowhead=1, ay=40, row=1, col=1, bgcolor="#6f42c1", font=dict(color="white", size=9))
        elif "抄底" in status:
            fig.add_annotation(x=p_data.index[i], y=curr['Low']*0.98, text=f"⚡抄底<br>${curr['Low']:.1f}", showarrow=True, arrowhead=1, ay=60, row=1, col=1, bgcolor="#9333ea", bordercolor="#ffffff", font=dict(color="white", size=10, weight="bold"))
        elif "調節" in status:
            fig.add_annotation(x=p_data.index[i], y=curr['High']*1.02, text=f"🔴調節<br>${curr['High']:.1f}", showarrow=True, arrowhead=1, ay=-60, row=1, col=1, bgcolor="#b91c1c", bordercolor="#ffffff", font=dict(color="white", size=10, weight="bold"))

    is_valid_buy = (curr['MACD'] > curr['Signal_Line']) and (prior['MACD'] <= prior['Signal_Line'])
    if is_valid_buy and engine_type == "trend" and curr['Close'] < curr.get('SMA_60', 0): is_valid_buy = False
    elif is_valid_buy and engine_type == "momentum" and curr['Close'] < curr.get('SMA_20', 0): is_valid_buy = False
    
    if is_valid_buy: fig.add_annotation(x=p_data.index[i], y=curr['Low']*0.98, text=f"BUY<br>${curr['Close']:.1f}", showarrow=True, arrowhead=1, ay=40, row=1, col=1, bgcolor="#28a745", font=dict(color="white", size=9))
    if (curr['MACD'] < curr['Signal_Line']) and (prior['MACD'] >= prior['Signal_Line']):
        fig.add_annotation(x=p_data.index[i], y=curr['High']*1.02, text=f"SELL<br>${curr['Close']:.1f}", showarrow=True, arrowhead=1, ay=-40, row=1, col=1, bgcolor="#dc3545", font=dict(color="white", size=9))

    hit_p, hit_r = curr['High'] >= t_s, curr['RSI'] > 75
    if (hit_p or hit_r) and not (prior['High'] >= t_s or prior['RSI'] > 75):
        fig.add_annotation(x=p_data.index[i], y=curr['High']*1.02, text=f"💰達標<br>${curr['Close']:.1f}" if hit_p else f"🔥過熱<br>${curr['Close']:.1f}", showarrow=True, arrowhead=1, ay=-60, row=1, col=1, bgcolor="#ffc107" if hit_p else "#ff4500", font=dict(color="black" if hit_p else "white", size=9))

macd_gold = (p_data['MACD'] > p_data['Signal_Line']) & (p_data['MACD'].shift(1) <= p_data['Signal_Line'].shift(1))
macd_dead = (p_data['MACD'] < p_data['Signal_Line']) & (p_data['MACD'].shift(1) >= p_data['Signal_Line'].shift(1))
if not p_data[macd_gold].empty: fig.add_trace(go.Scatter(x=p_data[macd_gold].index, y=p_data[macd_gold]['MACD'], mode='markers', marker=dict(symbol='triangle-up', size=10, color='#d8b4fe'), name='金叉'), row=2, col=1)
if not p_data[macd_dead].empty: fig.add_trace(go.Scatter(x=p_data[macd_dead].index, y=p_data[macd_dead]['MACD'], mode='markers', marker=dict(symbol='triangle-down', size=10, color='#facc15'), name='死叉'), row=2, col=1)

colors = ['green' if v >= 0 else 'red' for v in p_data['MACD_Hist']]
fig.add_trace(go.Bar(x=p_data.index, y=p_data['MACD_Hist'], marker_color=colors), row=2, col=1)
fig.add_trace(go.Scatter(x=p_data.index, y=p_data['MACD'], line=dict(color='white', width=1)), row=2, col=1)
fig.add_trace(go.Scatter(x=p_data.index, y=p_data['Signal_Line'], line=dict(color='yellow', width=1)), row=2, col=1)

fig.add_trace(go.Scatter(x=p_data.index, y=p_data['DMA_DDD'], line=dict(color='#d8b4fe', width=1)), row=3, col=1)
fig.add_trace(go.Scatter(x=p_data.index, y=p_data['DMA_AMA'], line=dict(color='#facc15', width=1)), row=3, col=1)

fig.update_layout(height=800, template="plotly_dark", xaxis_rangeslider_visible=False, showlegend=False, margin=dict(t=10, b=10, l=10, r=10))
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
engine_name = "🇹🇼 台股重訊模式" if is_tw else "🇺🇸 美股雙境獵手 (SEC+財聯社)"
with st.spinner(f"🕵️‍♂️ 啟動{engine_name}：正在掃描情報..."): items = fetch_deep_news(current_ticker, is_macro=False)
if items:
    for item in items[:8]:
        s, tag, m, p = analyze_news_weight_strict(item['title'], item['cat'])
        style = "border-left: 4px solid #00ffff; background-color: #003366;" if s >= 4 else "border-left: 4px solid #dc3545;" if s <= -2 else "border-left: 4px solid #4ade80;" if s > 0 else "border-left: 4px solid #555;"
        st.markdown(f'<div class="news-card" style="{style}"><a href="{item["link"]}" target="_blank" class="news-link"><span class="news-src">{item["src"]}</span> <span class="news-date">{item["date"]}</span> {tag} {item["title"]}</a></div>', unsafe_allow_html=True)
else: st.info("暫無 30 天內的重大情報")
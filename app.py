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
st.set_page_config(page_title="AI 實戰戰情室 V16.2 (台股即時版)", layout="wide", page_icon="📺")

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
    
    .header-legend {text-align: right; font-size: 13px; padding-top: 25px; color: #ccc;}
</style>
""", unsafe_allow_html=True)

# --- 1. 自選股儲存系統 ---
WATCHLIST_FILE = "watchlist.json"
ANCHOR_FILE = "anchors.json"
DEFAULT_LIST = ['NVDA', 'TSM', 'ONDS', 'RXRX', 'CRCL', 'AAPL', 'TSLA', '0050.TW', '2330.TW', '3535.TW']

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
    except Exception as e:
        st.error(f"儲存失敗: {e}")

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
    anchors[ticker] = {
        "title": news_title, 
        "summary": summary_cn, 
        "score": score, 
        "date": news_date_str, 
        "saved_at": datetime.now().strftime("%Y-%m-%d")
    }
    save_anchor_data(anchors)

def get_valid_anchor(ticker):
    anchors = load_anchors()
    if ticker not in anchors: return None
    data = anchors[ticker]
    try:
        saved_date = datetime.strptime(data["saved_at"], "%Y-%m-%d")
        if (datetime.now() - saved_date).days > 5:
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
        'ONDS': {'name': 'ONDS', 'ceo': [], 'key': ['Ondas', 'Networks']},
        'RXRX': {'name': 'RXRX', 'ceo': [], 'key': ['Recursion', 'Pharma']},
        'CRCL': {'name': 'CRCL', 'ceo': [], 'key': ['Circle']},
        'SOUN': {'name': 'SOUN', 'ceo': [], 'key': ['SoundHound']},
        'PLTR': {'name': 'PLTR', 'ceo': ['Karp'], 'key': ['Palantir']},
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
        small_cap_keywords = ['options', 'volume', 'shares', 'trading', '期權', '成交', '異動', '大漲', '大跌']
        if base_ticker in t and any(k in t for k in small_cap_keywords):
            return True
    return hit

def fetch_deep_news(ticker, is_macro=False):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        is_tw_stock = ".TW" in ticker or ".TWO" in ticker
        items = []
        now = datetime.now()
        
        if is_macro:
            url = f"https://news.google.com/rss/search?q=聯準會+升息+通膨+鮑爾&hl=zh-TW&gl=TW&ceid=TW:zh-Hant"
            resp = requests.get(url, headers=headers, timeout=4)
            if resp.status_code == 200:
                root = ET.fromstring(resp.content)
                for item in root.findall('.//item')[:6]:
                    data = parse_rss_item(item, "macro", "Global")
                    if (now - data['dt']).days <= 30: items.append(data)
            return items

        ticker_info = get_ticker_metadata(ticker)
        cn_name = ticker_info['name']
        search_target = f"{ticker} OR {cn_name}" if cn_name else ticker
        is_small_cap = ticker.split('.')[0] in ['ONDS', 'RXRX', 'CRCL', 'SOUN', 'PLTR'] 
        
        if is_tw_stock:
            clean_ticker = ticker.replace('.TW', '').replace('.TWO', '')
            q_tw = f"{clean_ticker}+(重訊+OR+重大訊息+OR+營收+OR+公告+OR+自結+OR+EPS+OR+配息+OR+法說)"
            url_tw = f"https://news.google.com/rss/search?q={q_tw}&hl=zh-TW&gl=TW&ceid=TW:zh-Hant"
            try:
                resp = requests.get(url_tw, headers=headers, timeout=4)
                if resp.status_code == 200:
                    root = ET.fromstring(resp.content)
                    for item in root.findall('.//item')[:10]:
                        data = parse_rss_item(item, "tw_local", "台股重訊")
                        if (now - data['dt']).days <= 30 and validate_news_relevance(data['title'], ticker, ticker_info):
                            items.append(data)
            except: pass
        else:
            q_sec = f"{ticker}+stock+(SEC+Filing+OR+Form+4+OR+10-Q+OR+8-K+OR+Insider+Trading)"
            if is_small_cap:
                q_news = f"{ticker}+stock+(Options+OR+Volume+OR+Implied+Volatility+OR+期權+OR+成交+OR+異動+OR+財報)"
            else:
                q_news = f"{search_target}+stock+(財聯社+OR+鉅亨網+OR+營收+OR+訂單+OR+CEO+OR+財報)"

            url_sec = f"https://news.google.com/rss/search?q={q_sec}&hl=en-US&gl=US&ceid=US:en"
            url_news = f"https://news.google.com/rss/search?q={q_news}&hl=zh-TW&gl=TW&ceid=TW:zh-Hant"
            
            sources = [(url_sec, "us_sec", "🏛️ SEC"), (url_news, "us_news", "📰 News")]
            for url, cat, src_name in sources:
                try:
                    resp = requests.get(url, headers=headers, timeout=4)
                    if resp.status_code == 200:
                        root = ET.fromstring(resp.content)
                        for item in root.findall('.//item')[:8]:
                            data = parse_rss_item(item, cat, src_name)
                            strict = not is_small_cap
                            is_relevant = validate_news_relevance(data['title'], ticker, ticker_info, strict_mode=strict)
                            if (now - data['dt']).days <= 30 and is_relevant:
                                items.append(data)
                except: pass

        items.sort(key=lambda x: x['dt'], reverse=True)
        return items
    except: return []

def parse_rss_item(item, category, source_name):
    title = re.sub('<[^<]+?>', '', item.find('title').text)
    link = item.find('link').text
    try:
        pub_date = item.find('pubDate').text
        dt = datetime.strptime(pub_date[:16], '%a, %d %b %Y')
        date_str = dt.strftime('%m/%d')
    except: 
        dt = datetime.now()
        date_str = "近期"
    
    if "futunn" in link: source_name = "🐂 富途"
    elif "yahoo" in link: source_name = "🇺🇸 Yahoo"
    return {'title': title, 'link': link, 'date': date_str, 'dt': dt, 'cat': category, 'src': source_name}

def analyze_news_weight_strict(title, category):
    t = title.lower()
    score = 0
    tag = ""
    is_major = False
    insider_penalty = 0.0
    
    farm_words = ['豪宅', '買房', '房貸', '藝人', '神操作', '心法', '財富自由', '被動收入', '開箱', '曬單', '退休', '名師', '股市名嘴', '怎麼買', '懶人包', 'motley fool', '後悔', '賓士', '人生', '笑談', '護身符']
    if any(w in t for w in farm_words): return 0, "", False, 0

    trap_words = ['減持', '賣出', '劣於', '降評', '損', '疲軟', '警告', '重挫', '砍單', '大跌', '崩盤', '利空', 'sell', 'underweight', 'miss', 'probe', 'lawsuit']
    for w in trap_words:
        if w in t: 
            if 'form 4' not in t and 'insider' not in t:
                return -2.0, '<span class="tag-red">Risk</span>', False, 0

    if any(x in t for x in ['options', 'volume', '期權', '成交', '異動', 'implied volatility']):
        if any(x in t for x in ['high', 'surge', 'active', 'jump', '活躍', '大增', '激增']):
            score = 2.0; tag = '<span class="tag-chip">🌊 籌碼異動</span>'
        else:
            score = 0.5; tag = '<span class="tag-gray">📊 籌碼面</span>'

    elif any(x in t for x in ['營收', 'revenue', 'eps', 'profit', '獲利', '自結', 'free cash flow', '財報', 'earnings', '8-k', '10-q', '重大訊息', '重訊']):
        if any(x in t for x in ['新高', 'record', 'beat', 'up', '增', '漲', '超預期']):
            score = 4.0; tag = '<span class="tag-filing">💎 財報/營運利多</span>'; is_major = True
        elif any(x in t for x in ['miss', 'down', 'loss', 'cut', '減', '虧', '不如']):
            score = -3.0; tag = '<span class="tag-red">📉 財報利空</span>'
        else:
            score = 1.5; tag = '<span class="tag-div">📊 財務數據</span>'

    elif category == "us_sec" or 'form 4' in t or '申報轉讓' in t or 'insider' in t:
        if any(x in t for x in ['buy', 'purchase', 'bought', '買進', '增持']):
            score = 4.0; tag = '<span class="tag-vip">👑 VIP買進</span>'; is_major = True
        elif any(x in t for x in ['sell', 'sold', 'dispose', '賣出', '減持']):
            if any(p in t for p in ['cfo', 'chief financial', 'controller', 'accounting', '財務長']):
                score = -4.0; insider_penalty = -4.0; tag = '<span class="tag-risk">⚠️ CFO拋售</span>'
            elif any(p in t for p in ['ceo', 'chief executive', '執行長']):
                score = -3.5; insider_penalty = -3.5; tag = '<span class="tag-risk">⚠️ CEO拋售</span>'
            else:
                score = -1.5; insider_penalty = -1.5; tag = '<span class="tag-gray">內部人賣出</span>'

    elif any(x in t for x in ['order', 'contract', '訂單', '簽約', 'backlog', '擴產']):
        score = 3.0; tag = '<span class="tag-hard">🔥 實質訂單</span>'
    elif any(x in t for x in ['musk', 'jensen', '黃仁勳', '馬斯克', '張忠謀', '魏哲家']):
        if any(v in t for v in ['說', '稱', '表示', '回應', '宣布', 'talks', 'says']): 
            score = 2.0; tag = '<span class="tag-vip">👑 VIP發言</span>'
        else: 
            score = 0.5; tag = '<span class="tag-gray">🗣️ VIP相關</span>'

    if not tag:
        if score > 0: tag = '<span class="tag-hard">📈 利多</span>'
        elif score < 0: tag = '<span class="tag-red">📉 利空</span>'
        
    return score, tag, is_major, insider_penalty

def get_realtime_macro():
    news = fetch_deep_news("Macro", is_macro=True)
    news_score = 0
    for n in news:
        t = n['title'].lower()
        if any(w in t for w in ['hike', 'inflation', '升息', '通膨', '鷹']): news_score -= 1.5
        if any(w in t for w in ['cut', 'pause', '降息', '鴿']): news_score += 1
    
    txt = "宏觀穩健"; note = "多頭支撐"; col = "sig-green"; score = 0
    if news_score <= -3: txt="Fed 偏鷹"; note="系統風險高"; col="sig-red"; score=-2
    elif news_score < 0: txt="宏觀偏空"; note="震盪觀望"; col="sig-orange"; score=-1

    try:
        nasdaq = yf.Ticker("^IXIC")
        hist = nasdaq.history(period="5d")
        if len(hist) >= 2:
            latest = hist.iloc[-1]['Close']
            prev = hist.iloc[-2]['Close']
            pct_chg = (latest - prev) / prev * 100
            if pct_chg < -1.5:
                txt = "市場恐慌"; note = f"納指重挫 {pct_chg:.2f}%"; col = "sig-red"; score = -3
            elif pct_chg < -0.8:
                txt = "市場修正"; note = f"納指下跌 {pct_chg:.2f}%"; col = "sig-orange"; score = -1.5
    except: pass
    
    return txt, note, col, score

# --- 3. 技術指標與圖表核心 ---

def calculate_volume_profile(df, bins=40, filter_mask=None):
    if df.empty: return pd.DataFrame({'Price': [], 'Volume': []})
    price_min = df['Low'].min(); price_max = df['High'].max()
    bin_edges = np.linspace(price_min, price_max, bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    target_df = df if filter_mask is None else df[filter_mask]
    if target_df.empty: return pd.DataFrame({'Price': bin_centers, 'Volume': np.zeros(bins)})
    bin_indices = pd.cut(target_df['Close'], bins=bin_edges, labels=False, include_lowest=True)
    profile = target_df.groupby(bin_indices)['Volume'].sum().reindex(range(bins), fill_value=0)
    return pd.DataFrame({'Price': bin_centers, 'Volume': profile.values})

def calculate_indicators(df):
    if len(df) < 50: return df
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_60'] = df['Close'].rolling(window=60).mean()
    df['Vol_SMA5'] = df['Volume'].rolling(window=5).mean()
    df['Std_Dev'] = df['Close'].rolling(window=20).std()
    
    # 布林通道 (標準差)
    df['Bollinger_Upper'] = df['SMA_20'] + (df['Std_Dev'] * 2)
    df['Bollinger_Lower'] = df['SMA_20'] - (df['Std_Dev'] * 2)
    
    # [V16.0] 肯特納通道 (Keltner Channels) for TTM Squeeze
    df['KC_Upper'] = df['SMA_20'] + (df['SMA_20'] * 0.05) # 簡化計算
    df['KC_Lower'] = df['SMA_20'] - (df['SMA_20'] * 0.05)
    df['Squeeze_On'] = (df['Bollinger_Upper'] < df['KC_Upper']) & (df['Bollinger_Lower'] > df['KC_Lower'])
    
    # [V15.4] ATR 計算 (14日)
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df['ATR'] = true_range.rolling(14).mean()
    df['ATR_Trailing_Stop'] = df['High'].rolling(22).max() - (df['ATR'] * 3)

    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    exp12 = df['Close'].ewm(span=12, adjust=False).mean()
    exp26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp12 - exp26
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['Signal_Line']
    clv = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'])
    df['AD_Line'] = (clv.fillna(0) * df['Volume']).cumsum()
    df['DMA_DDD'] = df['Close'].rolling(window=10).mean() - df['Close'].rolling(window=50).mean()
    df['DMA_AMA'] = df['DMA_DDD'].rolling(window=10).mean()
    close = df['Close'].values
    buy_seq = np.zeros(len(close), dtype=int)
    sell_seq = np.zeros(len(close), dtype=int)
    for i in range(4, len(close)):
        if close[i] < close[i-4]: buy_seq[i] = buy_seq[i-1] + 1
        else: buy_seq[i] = 0
        if close[i] > close[i-4]: sell_seq[i] = sell_seq[i-1] + 1
        else: sell_seq[i] = 0
    df['TD_Buy_9'] = np.where(buy_seq == 9, close, np.nan)
    df['TD_Sell_9'] = np.where(sell_seq == 9, close, np.nan)
    df['TD_Buy_Stop'] = np.nan
    for i in range(len(close)):
        if buy_seq[i] == 9:
            start_idx = max(0, i - 8)
            min_low = df['Low'].iloc[start_idx:i+1].min()
            df.loc[df.index[i], 'TD_Buy_Stop'] = min_low
    return df

def get_relative_strength(ticker, stock_df):
    try:
        benchmark_symbol = "^TWII" if (".TW" in ticker or ".TWO" in ticker) else "^GSPC"
        bench = yf.download(benchmark_symbol, period="1mo", progress=False)['Close']
        if isinstance(bench, pd.DataFrame): bench = bench.iloc[:, 0]
        aligned_stock = stock_df['Close'].reindex(bench.index, method='ffill')
        if len(bench) > 20:
            stock_perf = (aligned_stock.iloc[-1] - aligned_stock.iloc[-20]) / aligned_stock.iloc[-20]
            bench_perf = (bench.iloc[-1] - bench.iloc[-20]) / bench.iloc[-20]
            diff = stock_perf - bench_perf
            if diff > 0.05: return "🦁 領頭羊 (強)", "sig-green"
            elif diff > 0: return "🐯 優於大盤", "sig-blue"
            else: return "🐶 落後股 (弱)", "sig-gray"
    except: pass
    return "⚖️ 跟隨大盤", "sig-gray"

def find_support_levels(df, current_price):
    if df.empty or len(df) < 60: return current_price, current_price, "資料不足", "資料不足"
    s1 = df['Close'].rolling(window=20).mean().iloc[-1]
    s1_note = "股價在月線之上 (趨勢多)" if current_price > s1 else "已跌破月線 (趨勢轉弱)"
    recent_60 = df.tail(60)
    max_vol_idx = recent_60['Volume'].idxmax()
    key_low = df.loc[max_vol_idx]['Low']
    s2_note = f"最大量日({max_vol_idx.strftime('%m/%d')})低點"
    return s1, key_low, s1_note, s2_note

def detect_smart_money_status(df):
    if len(df) < 10: return None
    latest = df.iloc[-1]
    
    if latest['Close'] < latest['Bollinger_Lower'] and latest['RSI'] < 30: 
        return "⚡ 乖離抄底 (超賣)"
    
    price_now = latest['Close']; price_5d = df['Close'].iloc[-6]
    ad_now = latest['AD_Line']; ad_5d = df['AD_Line'].iloc[-6]
    rsi = latest['RSI']
    
    if price_now < price_5d * 0.98 and ad_now > ad_5d and rsi < 50:
        return "🎯 主力背離吸籌"
        
    if rsi > 65 and latest['Volume'] > latest['Vol_SMA5'] * 1.3:
        open_p = latest['Open']
        close_p = latest['Close']
        high_p = latest['High']
        is_black_body = close_p < open_p
        upper_shadow = high_p - max(open_p, close_p)
        body_len = abs(close_p - open_p)
        is_long_shadow = upper_shadow > body_len * 1.5
        if is_black_body or is_long_shadow:
            return "🔴 主力調節 (爆量滯漲)"
    
    if rsi < 30 and latest['Volume'] > latest['Vol_SMA5']: 
        return "⚡ 恐慌殺盤"
        
    return None

def analyze_strategic_signals(df):
    if df.empty: return {}
    latest = df.iloc[-1]
    macd = latest['MACD']; signal = latest['Signal_Line']
    if isinstance(macd, pd.Series): macd = macd.iloc[0]
    if isinstance(signal, pd.Series): signal = signal.iloc[0]
    
    if macd > signal:
        if macd > 0: macd_text, macd_color = "零軸上金叉 (多頭)", "sig-green"
        else: macd_text, macd_color = "零軸下金叉 (反彈)", "sig-orange"
    else:
        if macd > 0: macd_text, macd_color = "零軸上死叉 (修正)", "sig-orange"
        else: macd_text, macd_color = "零軸下死叉 (空頭)", "sig-red"
        
    vol = latest['Volume']; vol_ma = latest['Vol_SMA5']
    if vol > vol_ma * 1.5: vol_text, vol_color = "爆量", "sig-green"
    elif vol > vol_ma * 1.1: vol_text, vol_color = "量增", "sig-green"
    else: vol_text, vol_color = "量縮", "sig-gray"
    
    rsi = latest['RSI']
    if rsi > 70: rsi_text, rsi_color = f"過熱 ({rsi:.0f})", "sig-red"
    elif rsi < 30: rsi_text, rsi_color = f"超賣 ({rsi:.0f})", "sig-green"
    else: rsi_text, rsi_color = f"中性 ({rsi:.0f})", "sig-gray"
    
    summary = "觀望"; summary_color = "sig-gray"
    
    if latest.get('Squeeze_On', False):
        summary = "🌀 壓縮蓄力中 (變盤在即)"; summary_color = "sig-cyan"
    elif not np.isnan(latest.get('TD_Sell_9', np.nan)): summary, summary_color = "🔺 九轉賣點", "sig-red"
    elif not np.isnan(latest.get('TD_Buy_9', np.nan)): summary, summary_color = f"🔻 九轉買點", "sig-green"
    elif macd > signal: summary, summary_color = "📈 偏多震盪", "sig-green"
    else: summary, summary_color = "⛈️ 空頭走勢", "sig-red"
    
    status = detect_smart_money_status(df)
    if status: 
        summary = status
        if "調節" in status: summary_color = "sig-red"
        elif "乖離" in status: summary_color = "sig-purple"
        else: summary_color = "sig-blue"
    
    return {"MACD_Text": macd_text, "MACD_Color": macd_color, "Vol_Text": vol_text, "Vol_Color": vol_color, "RSI_Text": rsi_text, "RSI_Color": rsi_color, "Summary": summary, "Summary_Color": summary_color}

def analyze_market_trend(df):
    price = df['Close'].iloc[-1]; ma20 = df['SMA_20'].iloc[-1]; ma60 = df['SMA_60'].iloc[-1]
    if isinstance(price, pd.Series): price = price.iloc[0]
    if isinstance(ma20, pd.Series): ma20 = ma20.iloc[0]
    if isinstance(ma60, pd.Series): ma60 = ma60.iloc[0]

    if price > ma20 and ma20 > ma60: return "🐂 牛市 (Bull)", "多頭排列", "sig-green"
    elif price < ma20 and ma20 < ma60: return "🐻 熊市 (Bear)", "空頭排列", "sig-red"
    else: return "⚖️ 震盪 (Range)", "區間整理", "sig-orange"

def predict_target_and_rating(df):
    price = df['Close'].iloc[-1]; ma20 = df['SMA_20'].iloc[-1]
    upper = df['Bollinger_Upper'].iloc[-1]
    if isinstance(price, pd.Series): price = price.iloc[0]
    if isinstance(upper, pd.Series): upper = upper.iloc[0]
    
    rating = "持有"
    if price > df['SMA_20'].iloc[-1]: rating = "強勢"
    
    recent_high_60 = df['High'].tail(60).max()
    if price >= recent_high_60:
        target_short = upper
    else:
        target_short = min(upper, recent_high_60)

    target_long = max(recent_high_60 * 1.15, target_short * 1.1)
    return target_short, target_long, rating

def generate_buy_hint(df, current_price, s1, s2):
    latest = df.iloc[-1]
    
    if latest.get('Squeeze_On', False): return "🌀 波動壓縮中，等待方向突破"
    
    status = detect_smart_money_status(df)
    if status and "調節" in status: return f"⚠️ 警戒！{status}"
    
    if not np.isnan(latest.get('TD_Sell_9', np.nan)): return "🔺 藍色9 (上漲力竭)，注意獲利"
    if not np.isnan(latest.get('TD_Buy_9', np.nan)): return f"🔻 紅色9 (潛在買點)，SL:{latest.get('TD_Buy_Stop', 0):.2f}"
    
    if (latest['DMA_DDD'] > latest['DMA_AMA']) and (df['DMA_DDD'].iloc[-2] <= df['DMA_AMA'].iloc[-2]): 
        return "🚀 DMA 金叉，中線轉多"
        
    if status: return f"🚨 {status}"
    
    if abs(current_price - s1) / current_price < 0.015 and current_price > s1: return "回測月線有撐"
    return "觀望，等待訊號"

def format_volume(num):
    if num >= 1e9: return f"{num/1e9:.2f}B"
    elif num >= 1e6: return f"{num/1e6:.2f}M"
    else: return f"{num}"

# --- 5. TradingView Widget Integration (V16.2) ---
def render_tw_realtime_chart(ticker):
    """嵌入 TradingView 即時圖表 Widget (專為台股設計)"""
    
    # 轉換代號格式 (2330.TW -> TWSE:2330, 8069.TWO -> TPEX:8069)
    tv_symbol = ticker
    if ".TW" in ticker:
        tv_symbol = f"TWSE:{ticker.replace('.TW', '')}"
    elif ".TWO" in ticker:
        tv_symbol = f"TPEX:{ticker.replace('.TWO', '')}"
    else:
        # 美股或其他
        tv_symbol = ticker

    # HTML Embed Code
    html_code = f"""
    <div class="tradingview-widget-container">
      <div id="tradingview_chart"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
      <script type="text/javascript">
      new TradingView.widget(
      {{
        "width": "100%",
        "height": 500,
        "symbol": "{tv_symbol}",
        "interval": "D",
        "timezone": "Asia/Taipei",
        "theme": "dark",
        "style": "1",
        "locale": "zh_TW",
        "toolbar_bg": "#f1f3f6",
        "enable_publishing": false,
        "allow_symbol_change": true,
        "container_id": "tradingview_chart"
      }}
      );
      </script>
    </div>
    """
    components.html(html_code, height=500)

# --- 6. 主介面 ---
with st.sidebar:
    st.title("🎛️ 控制台")
    st.header("📌 自選股清單")
    selection = st.radio("選擇股票", st.session_state.watchlist)
    current_ticker = selection
    
    c_up, c_down = st.columns(2)
    if c_up.button("⬆️ 上移", key="up") and current_ticker in st.session_state.watchlist:
        idx = st.session_state.watchlist.index(current_ticker)
        if idx > 0:
            st.session_state.watchlist[idx], st.session_state.watchlist[idx-1] = st.session_state.watchlist[idx-1], st.session_state.watchlist[idx]
            save_watchlist(st.session_state.watchlist); st.rerun()
            
    if c_down.button("⬇️ 下移", key="down") and current_ticker in st.session_state.watchlist:
        idx = st.session_state.watchlist.index(current_ticker)
        if idx < len(st.session_state.watchlist) - 1:
            st.session_state.watchlist[idx], st.session_state.watchlist[idx+1] = st.session_state.watchlist[idx+1], st.session_state.watchlist[idx]
            save_watchlist(st.session_state.watchlist); st.rerun()
            
    c_top, c_bot = st.columns(2)
    if c_top.button("⏫ 置頂", key="top") and current_ticker in st.session_state.watchlist:
        st.session_state.watchlist.remove(current_ticker)
        st.session_state.watchlist.insert(0, current_ticker)
        save_watchlist(st.session_state.watchlist); st.rerun()
        
    if c_bot.button("⏬ 置底", key="bot") and current_ticker in st.session_state.watchlist:
        st.session_state.watchlist.remove(current_ticker)
        st.session_state.watchlist.append(current_ticker)
        save_watchlist(st.session_state.watchlist); st.rerun()

    st.markdown("---")
    time_opt = st.radio("選擇週期", ["當沖 (分時)", "日線 (Daily)", "週線 (Weekly)", "月線 (長線)"], index=1)
    st.markdown("---")
    with st.expander("編輯清單"):
        new_t = st.text_input("代號", placeholder="MSTR").upper()
        if st.button("➕", key="add") and new_t:
            if new_t not in st.session_state.watchlist:
                st.session_state.watchlist.append(new_t)
                save_watchlist(st.session_state.watchlist); st.rerun()
        if st.button("❌", key="del"):
            if current_ticker in st.session_state.watchlist:
                st.session_state.watchlist.remove(current_ticker)
                save_watchlist(st.session_state.watchlist); st.rerun()
    
    with st.expander("📖 訊號解讀指南"):
        st.markdown("""
        🌀 **波動壓縮** ➔ 出現 **【🌀蓄力中】** (布林縮口，準備大行情)
        🕯️ **K線型態** ➔ 出現 **【🕯️吞噬】** (多頭強力反轉)
        🛡️ **紫色階梯線** ➔ **【ATR停損線】** (跌破此線無條件離場)
        🦁 **相對強弱** ➔ **【🦁領頭羊】** (比大盤強) vs **【🐶落後股】** (比大盤弱)
        """)

st.title(f"📈 {current_ticker} 實戰戰情室 V16.2")

api_period = "1y"; api_int = "1d"; fmt = "%Y-%m-%d"
if "當沖" in time_opt: api_period = "5d"; api_int = "15m"; fmt = "%H:%M"
elif "日線" in time_opt: api_period = "6mo"; api_int = "1d"
elif "週線" in time_opt: api_period = "2y"; api_int = "1wk"
elif "月線" in time_opt: api_period = "2y"; api_int = "1wk"; fmt = "%Y-%m"

@st.cache_data(ttl=300)
def fetch_data(t, p, i):
    try: return yf.download(t, period=p, interval=i, progress=False)
    except: return pd.DataFrame()

# 🛡️ 核心防護罩 (Try-Catch)
try:
    df = fetch_data(current_ticker, api_period, api_int)
    if df.empty: st.error("無數據，請確認代號或網路"); st.stop()
    
    if isinstance(df.columns, pd.MultiIndex): 
        df.columns = df.columns.get_level_values(0)
    
    df = df.loc[:, ~df.columns.duplicated()]

    df = calculate_indicators(df)
    if 'DMA_DDD' not in df.columns: st.error("數據不足計算指標"); st.stop()
    
    latest = df.iloc[-1]; prev = df.iloc[-2]
    close_v = latest['Close'] if not isinstance(latest['Close'], pd.Series) else latest['Close'].iloc[0]
    prev_v = prev['Close'] if not isinstance(prev['Close'], pd.Series) else prev['Close'].iloc[0]
    
    chg = (close_v - prev_v) / prev_v * 100
    clr = "green" if chg >= 0 else "red"
    
    s1, s2, n1, n2 = find_support_levels(df, close_v)
    sigs = analyze_strategic_signals(df)
    trend_txt, trend_note, trend_col = analyze_market_trend(df)
    t_s, t_l, rating = predict_target_and_rating(df)
    buy_hint_text = generate_buy_hint(df, close_v, s1, s2)
    
    macro_txt, macro_note, macro_col, macro_score = get_realtime_macro()
    
    rs_txt, rs_col = get_relative_strength(current_ticker, df)

    # 頂部資訊
    st.markdown(f"""
    <div class="price-card">
        <h1 style="margin:0; font-size: 50px;">${close_v:.2f}</h1>
        <h3 style="margin:0; color: {clr};">{chg:+.2f}%</h3>
        <p style="color: gray;">量: {format_volume(latest['Volume'])}</p>
        <div class="buy-hint">💡 操作提示: {buy_hint_text}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # 2. 雙格局戰略雷達
    r1, r2, r3 = st.columns(3)
    with r1:
        st.markdown(f"""
        <div class="ai-box">
            <h5 style="color:white; margin:0; margin-bottom:5px;">📡 綜合戰略</h5>
            <div style="font-size:16px;" class="{sigs['Summary_Color']}">{sigs['Summary']}</div>
            <div class="radar-grid" style="margin-top:5px;">
                <div class="radar-item"><span>MACD</span><span class="{sigs['MACD_Color']}">{sigs['MACD_Text']}</span></div>
                <div class="radar-item"><span>RSI</span><span class="{sigs['RSI_Color']}">{sigs['RSI_Text']}</span></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with r2:
        st.markdown(f"""
        <div class="ai-box">
            <h5 style="color:white; margin:0;">⚖️ 雙重格局</h5>
            <div style="margin-top:5px;">
                <div>🏢 個股: <span class="{trend_col}">{trend_txt}</span></div>
                <div>🌍 宏觀: <span class="{macro_col}">{macro_txt}</span></div>
                <div style="font-size:11px; color:#aaa;">({macro_note})</div>
            </div>
        </div>""", unsafe_allow_html=True)
    with r3:
        st.markdown(f"""<div class="ai-box" style="border: 1px solid #00d4ff;"><h5 style="color:white; margin:0;">🎯 AI 目標 & 強弱</h5><div>短: ${t_s:.2f} | 長: ${t_l:.2f}</div><div style="margin-top:5px;"><span class="{rs_col}">{rs_txt}</span></div></div>""", unsafe_allow_html=True)

    # [V16.2] 雙模切換 (TradingView vs AI Chart)
    is_tw = ".TW" in current_ticker or ".TWO" in current_ticker
    
    # 建立頁籤
    tab1, tab2 = st.tabs(["📊 AI 戰略圖 (訊號)", "📺 即時看盤 (TradingView)"])
    
    with tab1:
        # 這是原本的 Plotly 圖表 (AI 訊號源)
        try:
            p_data = df.tail(150) if "週" in time_opt else (df.tail(120) if "日" in time_opt else df.tail(60))
            fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_width=[0.2, 0.2, 0.6])
            fig.add_trace(go.Candlestick(x=p_data.index, open=p_data['Open'], high=p_data['High'], low=p_data['Low'], close=p_data['Close'], name='Price'), row=1, col=1)
            
            # [V15.4] 繪製 ATR 停損線
            fig.add_trace(go.Scatter(x=p_data.index, y=p_data['ATR_Trailing_Stop'], mode='lines', line=dict(color='purple', width=1, dash='dot'), name='ATR Stop'), row=1, col=1)

            # 復刻標記 (含價格)
            for i in range(5, len(p_data)):
                curr = p_data.iloc[i]
                prior = p_data.iloc[i-1]
                
                # [V16.1] 關鍵K線型態：只保留多頭吞噬
                is_engulfing = (prior['Close'] < prior['Open']) and (curr['Close'] > curr['Open']) and (curr['Open'] <= prior['Close']) and (curr['Close'] >= prior['Open'])
                
                if is_engulfing:
                     fig.add_annotation(x=p_data.index[i], y=curr['Low']*0.98, text="🕯️吞噬", showarrow=True, arrowhead=1, ay=40, row=1, col=1, font=dict(color="orange", size=8))

                # 九轉
                if not np.isnan(curr.get('TD_Buy_9', np.nan)):
                     fig.add_annotation(x=p_data.index[i], y=curr['Low'], text="9", showarrow=False, font=dict(color='#ff6b6b', size=12, weight="bold"), row=1, col=1)
                if not np.isnan(curr.get('TD_Sell_9', np.nan)):
                     fig.add_annotation(x=p_data.index[i], y=curr['High'], text="9", showarrow=False, font=dict(color='#4a9eff', size=12, weight="bold"), row=1, col=1)
                     
                # [V15.1] 智能籌碼偵測 (吸籌/調節/抄底)
                status = detect_smart_money_status(df.iloc[:i+1])
                if status:
                    if "吸籌" in status:
                        fig.add_annotation(x=p_data.index[i], y=curr['Low']*0.98, text=f"🐳吸<br>${curr['Low']:.1f}", showarrow=True, arrowhead=1, ay=40, row=1, col=1, bgcolor="#6f42c1", font=dict(color="white", size=9))
                    elif "抄底" in status:
                        fig.add_annotation(x=p_data.index[i], y=curr['Low']*0.98, text=f"⚡抄底<br>${curr['Low']:.1f}", showarrow=True, arrowhead=1, ay=60, row=1, col=1, bgcolor="#9333ea", bordercolor="#ffffff", font=dict(color="white", size=10, weight="bold"))
                    elif "調節" in status:
                        fig.add_annotation(x=p_data.index[i], y=curr['High']*1.02, text=f"🔴調節<br>${curr['High']:.1f}", showarrow=True, arrowhead=1, ay=-60, row=1, col=1, bgcolor="#b91c1c", bordercolor="#ffffff", font=dict(color="white", size=10, weight="bold"))

                # 買入/賣出訊號
                macd_buy = (curr['MACD'] > curr['Signal_Line']) and (prior['MACD'] <= prior['Signal_Line'])
                macd_sell = (curr['MACD'] < curr['Signal_Line']) and (prior['MACD'] >= prior['Signal_Line'])
                
                if macd_buy:
                     fig.add_annotation(x=p_data.index[i], y=curr['Low']*0.98, text=f"BUY<br>${curr['Close']:.1f}", showarrow=True, arrowhead=1, ay=40, row=1, col=1, bgcolor="#28a745", font=dict(color="white", size=9))
                if macd_sell:
                     fig.add_annotation(x=p_data.index[i], y=curr['High']*1.02, text=f"SELL<br>${curr['Close']:.1f}", showarrow=True, arrowhead=1, ay=-40, row=1, col=1, bgcolor="#dc3545", font=dict(color="white", size=9))

                # [V15.3] 達標與過熱 (Visual Split)
                hit_price = curr['High'] >= t_s
                hit_rsi = curr['RSI'] > 75
                prev_hit = prior['High'] >= t_s or prior['RSI'] > 75
                
                if (hit_price or hit_rsi) and not prev_hit:
                     if hit_price:
                         fig.add_annotation(x=p_data.index[i], y=curr['High']*1.02, text=f"💰達標<br>${curr['Close']:.1f}", showarrow=True, arrowhead=1, ay=-60, row=1, col=1, bgcolor="#ffc107", font=dict(color="black", size=9))
                     else:
                         fig.add_annotation(x=p_data.index[i], y=curr['High']*1.02, text=f"🔥過熱<br>${curr['Close']:.1f}", showarrow=True, arrowhead=1, ay=-60, row=1, col=1, bgcolor="#ff4500", font=dict(color="white", size=9))

            # 金叉死叉
            macd_gold = (p_data['MACD'] > p_data['Signal_Line']) & (p_data['MACD'].shift(1) <= p_data['Signal_Line'].shift(1))
            macd_dead = (p_data['MACD'] < p_data['Signal_Line']) & (p_data['MACD'].shift(1) >= p_data['Signal_Line'].shift(1))
            
            valid_gold = p_data[macd_gold]
            valid_dead = p_data[macd_dead]
            
            if not valid_gold.empty:
                fig.add_trace(go.Scatter(x=valid_gold.index, y=valid_gold['MACD'], mode='markers', marker=dict(symbol='triangle-up', size=10, color='#d8b4fe'), name='金叉'), row=2, col=1)
            if not valid_dead.empty:
                fig.add_trace(go.Scatter(x=valid_dead.index, y=valid_dead['MACD'], mode='markers', marker=dict(symbol='triangle-down', size=10, color='#facc15'), name='死叉'), row=2, col=1)

            fig.add_hline(y=s1, line_dash="dash", line_color="#00d4ff", annotation_text="MA20", row=1, col=1)
            
            colors = ['green' if v >= 0 else 'red' for v in p_data['MACD_Hist']]
            fig.add_trace(go.Bar(x=p_data.index, y=p_data['MACD_Hist'], marker_color=colors), row=2, col=1)
            fig.add_trace(go.Scatter(x=p_data.index, y=p_data['MACD'], line=dict(color='white', width=1)), row=2, col=1)
            fig.add_trace(go.Scatter(x=p_data.index, y=p_data['Signal_Line'], line=dict(color='yellow', width=1)), row=2, col=1)
            
            fig.add_trace(go.Scatter(x=p_data.index, y=p_data['DMA_DDD'], line=dict(color='#d8b4fe', width=1)), row=3, col=1)
            fig.add_trace(go.Scatter(x=p_data.index, y=p_data['DMA_AMA'], line=dict(color='#facc15', width=1)), row=3, col=1)
            
            fig.update_layout(dragmode=False)
            fig.update_xaxes(tickformat=fmt)
            fig.update_layout(height=800, template="plotly_dark", xaxis_rangeslider_visible=False, showlegend=False, margin=dict(t=10, b=10, l=10, r=10))
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False, 'scrollZoom': False})
            
            if is_tw:
                st.caption("⚠️ 注意：此圖表數據可能有 15 分鐘延遲，AI 訊號僅供波段參考，當沖請切換至「即時看盤」分頁。")
                
        except Exception as e:
            st.error(f"圖表繪製發生錯誤 (可能是資料格式問題): {e}")

    with tab2:
        # [V16.2] 新增：TradingView 即時圖表
        if is_tw:
            st.info(f"正在顯示 {current_ticker} 的即時走勢 (來源：TradingView)")
            render_tw_realtime_chart(current_ticker)
        else:
            st.info("美股通常使用上述 AI 圖表即可 (盤中延遲較小)。若需 TradingView 介面也可在此查看。")
            render_tw_realtime_chart(current_ticker)

    # 籌碼與新聞區 (加裝 Try-Catch)
    try:
        c1, c2 = st.columns(2)
        mf = ((p_data['Close'] - p_data['Open']) / (p_data['High'] - p_data['Low'])) * p_data['Volume']
        mf = mf.fillna(0).cumsum()
        with c1:
            st.caption("主力資金流 (Money Flow)")
            fig_mf = go.Figure(go.Scatter(x=p_data.index, y=mf, fill='tozeroy', line=dict(color='#00d4ff')))
            
            # 主力動向標籤 (V13.20 回歸)
            if len(mf) > 5:
                trend = mf.iloc[-1] - mf.iloc[-5]
                if trend > 0:
                    fig_mf.add_annotation(x=p_data.index[-1], y=mf.iloc[-1], text="🟢 主力吸籌", showarrow=True, arrowhead=1, font=dict(color="#4ade80", size=12), bgcolor="#1b3a1b")
                else:
                    fig_mf.add_annotation(x=p_data.index[-1], y=mf.iloc[-1], text="🔴 主力出貨", showarrow=True, arrowhead=1, font=dict(color="#ff6b6b", size=12), bgcolor="#3a1b1b")

            fig_mf.update_layout(height=250, template="plotly_dark", margin=dict(t=10, b=10, l=10, r=10), dragmode=False)
            st.plotly_chart(fig_mf, use_container_width=True, config={'displayModeBar': False, 'scrollZoom': False})
            
        with c2:
            st.caption("籌碼分佈 (主力 vs 散戶)")
            # 雙層籌碼
            inst_mask = (p_data['Close'] > p_data['Open']) & (p_data['Volume'] > p_data['Vol_SMA5'])
            
            def calc_vp_layer(d, mask=None):
                if d.empty: return pd.DataFrame({'P':[], 'V':[]})
                p_min, p_max = d['Low'].min(), d['High'].max()
                edges = np.linspace(p_min, p_max, 41)
                centers = (edges[:-1] + edges[1:]) / 2
                sub = d if mask is None else d[mask]
                if sub.empty: return pd.DataFrame({'P': centers, 'V': np.zeros(40)})
                idx = pd.cut(sub['Close'], bins=edges, labels=False, include_lowest=True)
                v_sum = sub.groupby(idx)['Volume'].sum().reindex(range(40), fill_value=0)
                return pd.DataFrame({'P': centers, 'V': v_sum.values})

            vp_all = calc_vp_layer(p_data)
            vp_main = calc_vp_layer(p_data, inst_mask)

            fig_vp = go.Figure()
            fig_vp.add_trace(go.Scatter(x=vp_all['P'], y=vp_all['V'], fill='tozeroy', line=dict(color='#ffaa00', width=0), name='整體'))
            fig_vp.add_trace(go.Scatter(x=vp_main['P'], y=vp_main['V'], fill='tozeroy', line=dict(color='#00d4ff', width=2), name='主力'))
            
            # 標示現價
            fig_vp.add_vline(x=close_v, line_dash="dash", line_color="white", annotation_text="現價")
            fig_vp.update_layout(height=250, template="plotly_dark", margin=dict(t=10, b=10, l=10, r=10), showlegend=True, legend=dict(orientation="h", y=1.1), dragmode=False)
            st.plotly_chart(fig_vp, use_container_width=True, config={'displayModeBar': False, 'scrollZoom': False})
    except Exception as e:
        st.error(f"籌碼分析發生錯誤: {e}")

    st.markdown("---")

    # 搜尋與分析
    engine_name = "🇹🇼 台股重訊模式" if is_tw else "🇺🇸 美股雙境獵手 (SEC+財聯社)"
    
    with st.spinner(f"🕵️‍♂️ 啟動{engine_name}：正在掃描並過濾農場新聞..."):
        items = fetch_deep_news(current_ticker, is_macro=False)
    
    news_score = 0
    total_insider_penalty = 0
    valid_count = 0
    has_major = False
    processed = []
    
    anchor = get_valid_anchor(current_ticker)
    if anchor: news_score += anchor['score']
    
    for item in items:
        s, tag, major, penalty = analyze_news_weight_strict(item['title'], item['cat'])
        if s == 0 and not tag: continue
            
        news_score += s
        total_insider_penalty += penalty
        valid_count += 1
        if major: has_major = True
        
        # [V14.3] 傳入新聞日期
        if major and s > 0: update_anchor(current_ticker, item['title'], 3.0, item['date'])
        processed.append({'data': item, 'score': s, 'tag': tag})

    base_win_rate = 50.0
    win_rate = base_win_rate + (news_score * 5) + (macro_score * 5) # [V14.8] 納入宏觀分數
    if total_insider_penalty <= -3.0: win_rate -= 20 
    win_rate = max(10.0, min(95.0, win_rate))
    
    final_verdict = ""
    v_col = "gray"
    
    if total_insider_penalty <= -4.0:
        final_verdict = f"⚠️ 謹慎持有 (熔斷)！內部人/CFO 大量拋售"
        v_col = "#ffc107"
        m_disp = f'<div class="macro-alert" style="background-color:#3a1b1b; color:#ffc107; border:1px solid #ffc107;">⚡ 觸發內部人熔斷：高管拋售過大，強制降評</div>'
    elif has_major:
        final_verdict = f"🚀 強力看漲 (霸體)！重訊/財報利多 (+{news_score:.1f})"
        v_col = "#4ade80"
        m_disp = f'<div class="macro-alert" style="background-color:#1b3a1b; color:#4ade80; border:1px solid #28a745;">💎 偵測到重大訊息：已自動忽略宏觀風險 ({macro_txt})</div>'
    else:
        news_score += macro_score
        if macro_score < 0: m_disp = f'<div class="macro-alert">{macro_txt}：{macro_note}，評分已下修</div>'
        else: m_disp = f'<div style="color:orange;">{macro_txt}</div>'
        
        if news_score >= 3: final_verdict = "📈 偏多操作 (基本面支撐)"; v_col = "#ffc107"
        elif news_score <= -2: final_verdict = "📉 偏空看待 (利空罩頂)"; v_col = "#ff6b6b"
        else: final_verdict = "☁️ 觀望整理 (缺乏驅動力)"; v_col = "gray"

    nc1, nc2 = st.columns([0.4, 0.6])
    with nc1:
        st.markdown(m_disp, unsafe_allow_html=True)
        if anchor:
            st.markdown(f"""
            <div class="anchor-box">
                <div class="anchor-title-cn">⚓ 記憶錨定 ({anchor['date']})</div>
                <div style="color: #00ffff; margin-bottom:5px;">{anchor['summary']} (+{anchor['score']}分)</div>
                <div class="anchor-title-en">{anchor['title']}</div>
            </div>""", unsafe_allow_html=True)
            
        st.markdown(f"""
        <div class="ai-box" style="text-align:left;">
            <h3 style="color:white; margin:0;">🔮 AI 戰情推演</h3>
            <div style="font-size:18px; color:{v_col}; font-weight:bold; margin-top:10px;">{final_verdict}</div>
            <div style="font-size:24px; color:#00ffff; font-weight:bold; margin-top:5px;">📈 多方勝率：{win_rate:.1f}%</div>
            <hr style="border-color:#555;">
            <div style="font-size:14px; color:#ccc;">
                <b>基本面總分：</b> {news_score:.1f}<br>
                <b>內部人扣分：</b> {total_insider_penalty:.1f}<br>
                <b>有效情報數：</b> {valid_count} 則<br>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with nc2:
        st.caption(f"目前搜尋引擎：{engine_name} | ⏳ 時效：30天 | 🛡️ 農場文過濾：開啟")
        if processed:
            for p in processed:
                style = "border-left: 4px solid #555;"
                if p['score'] >= 4: style = "border-left: 4px solid #00ffff; background-color: #003366;"
                elif p['score'] <= -2: style = "border-left: 4px solid #dc3545;"
                elif p['score'] > 0: style = "border-left: 4px solid #4ade80;"
                
                st.markdown(f"""
                <div class="news-card" style="{style}">
                    <a href="{p['data']['link']}" target="_blank" class="news-link">
                        <span class="news-src">{p['data'].get('src', 'News')}</span> 
                        <span class="news-date">{p['data']['date']}</span>
                        {p['tag']} {p['data']['title']}
                    </a>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("暫無 30 天內的重大情報 (或 API 連線限制)")

except Exception as e:
    st.error(f"系統錯誤: {e}")
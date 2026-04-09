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
import os

# --- 0. 系統設定 ---
st.set_page_config(page_title="AI 實戰戰情室 V17.36 (破底翻右側買點版)", layout="wide", page_icon="🛡️")

# --- CSS 美化 ---
st.markdown("""
<style>
    .tactical-box {background-color: #1a1a1a; padding: 20px; border-radius: 12px; border-left: 10px solid; margin-bottom: 20px; border-right: 1px solid #333; border-top: 1px solid #333; border-bottom: 1px solid #333;}
    .price-card {background-color: #1e1e1e; padding: 20px; border-radius: 10px; text-align: center; border: 1px solid #333; margin-bottom: 10px;}
    .ai-box {background-color: #333; padding: 15px; border-radius: 10px; border: 1px solid #555; text-align: center; height: 100%;}
    .news-card {background-color: #262730; padding: 12px; border-radius: 5px; border-left: 4px solid #555; margin-bottom: 10px; font-size: 14px; transition: transform 0.2s;}
    .news-card:hover {transform: translateX(5px);}
    .news-link {text-decoration: none; color: #e0e0e0; font-weight: bold; display: block;}
    .news-link:hover {color: #FFD700;}
    .news-date {color: #aaa; font-size: 12px; margin-right: 5px;}
    .news-src {background-color: #444; color: #eee; border: 1px solid #777; padding: 1px 5px; border-radius: 3px; font-size: 11px; margin-right: 5px;}
    .macro-alert {background-color: #3a1b1b; color: #ff6b6b; padding: 10px; border-radius: 5px; border: 1px solid #dc3545; margin-bottom: 10px; font-weight: bold;}
    .anchor-box {background-color: #1b3a4a; color: #00d4ff; padding: 12px; border-radius: 5px; border: 1px solid #00d4ff; margin-bottom: 10px; font-size: 13px; text-align: left;}
    .anchor-title-cn {color: #fff; font-weight: bold; font-size: 14px; margin-bottom: 4px;}
    .anchor-title-en {color: #aaa; font-size: 11px; font-style: italic;}
    .earnings-tag {background-color: #2c2c2e; padding: 5px 10px; border-radius: 15px; font-size: 13px; margin-top: 10px; border: 1px solid #555; display: inline-block; margin-right: 8px;}
    .engine-tag {background-color: #1e3a8a; color: #38bdf8; padding: 5px 10px; border-radius: 15px; font-size: 13px; margin-top: 10px; border: 1px solid #38bdf8; display: inline-block; font-weight: bold;}
    .tag-hard {background-color: #1b3a1b; color: #4ade80; border: 1px solid #28a745; padding: 2px 6px; border-radius: 4px; font-size: 12px; margin-right: 5px;}
    .tag-risk {background-color: #3a1b1b; color: #ff6b6b; border: 1px solid #dc3545; padding: 2px 6px; border-radius: 4px; font-size: 12px; font-weight: bold;}
    .sig-green {background-color: #1b3a1b; color: #4ade80; border: 1px solid #28a745; padding: 2px 6px; border-radius: 4px; font-size: 12px;}
    .sig-red {background-color: #3a1b1b; color: #ff6b6b; border: 1px solid #dc3545; padding: 2px 6px; border-radius: 4px; font-size: 12px;}
    .sig-gray {background-color: #333; color: #ccc; border: 1px solid #666; padding: 2px 6px; border-radius: 4px; font-size: 12px;}
    .sig-orange {background-color: #4a3b1b; color: #ffaa00; border: 1px solid #ffaa00; padding: 2px 6px; border-radius: 4px; font-size: 12px;}
    .sig-blue {background-color: #1b3a4a; color: #4a9eff; border: 1px solid #00d4ff; padding: 2px 6px; border-radius: 4px; font-size: 12px;}
    .sig-purple {background-color: #4a1b4a; color: #d8b4fe; border: 1px solid #a855f7; padding: 2px 6px; border-radius: 4px; font-size: 12px;}
    .sig-cyan {background-color: #083344; color: #22d3ee; border: 1px solid #06b6d4; padding: 2px 6px; border-radius: 4px; font-size: 12px;}
</style>
""", unsafe_allow_html=True)

# --- 1. 資料系統 ---
WATCHLIST_FILE, ANCHOR_FILE, TW_NAMES_FILE = "watchlist.json", "anchors.json", "tw_names.json"
DEFAULT_WATCHLISTS = {"清單 A": ['^IXIC', 'QQQ', 'NVDA', 'TSM'], "清單 B": ['MU', 'AAPL', 'TSLA'], "清單 C": ['0050.TW', '6127.TWO'], "清單 D": ['ONDS', 'RXRX']}

def json_load(f_name, default):
    if os.path.exists(f_name):
        try:
            with open(f_name, "r", encoding="utf-8") as f: return json.load(f)
        except: pass
    return default

def json_save(f_name, data):
    try:
        with open(f_name, "w", encoding="utf-8") as f: json.dump(data, f)
    except: pass

def load_watchlists():
    data = json_load(WATCHLIST_FILE, DEFAULT_WATCHLISTS)
    if isinstance(data, list): return {"清單 A": data, **{k:v for k,v in DEFAULT_WATCHLISTS.items() if k != "清單 A"}}
    return data

def save_watchlists(data): json_save(WATCHLIST_FILE, data); st.session_state.watchlists = data
def load_anchors(): return json_load(ANCHOR_FILE, {})
def save_anchor_data(data): json_save(ANCHOR_FILE, data)

def get_stock_name(ticker):
    us_map = {'NVDA': '輝達', 'TSLA': '特斯拉', 'AAPL': '蘋果', 'MU': '美光', 'TSM': '台積電'}
    base = ticker.split('.')[0]
    if base in us_map and not (".TW" in ticker or ".TWO" in ticker): return us_map[base]
    if ".TW" in ticker or ".TWO" in ticker:
        local_map = json_load(TW_NAMES_FILE, {})
        bad_words = ["Yahoo", "股市", "走勢", "無符合", "找不到", "代碼或名稱", "html", "TW"]
        keys_to_delete = [k for k, v in local_map.items() if any(bad in v for bad in bad_words)]
        for k in keys_to_delete: del local_map[k]
        if ticker in local_map: return local_map[ticker]
        name = None
        prefix = "otc" if ".TWO" in ticker else "tse"
        try:
            res = requests.get(f"https://mis.twse.com.tw/stock/api/getStockInfo.jsp?ex_ch={prefix}_{base}.tw", timeout=3)
            if res.status_code == 200:
                data = res.json()
                if "msgArray" in data and len(data["msgArray"]) > 0:
                    n = data["msgArray"][0].get("n")
                    if n and n != "--": name = n.strip()
        except: pass
        if not name:
            try:
                res = requests.get(f"https://ws.api.cnyes.com/ws/api/v1/quote/quotes/TWS:{base}:STOCK", timeout=3)
                if res.status_code == 200:
                    data = res.json()
                    if "data" in data and len(data["data"]) > 0:
                        n = data["data"][0].get("name")
                        if n: name = n.strip()
            except: pass
        if not name and ".TWO" not in ticker:
            try:
                res = requests.get(f"https://www.twse.com.tw/zh/api/codeQuery?query={base}", timeout=2)
                if res.status_code == 200:
                    data = res.json()
                    if "suggestions" in data and len(data["suggestions"]) > 0:
                        sug = data["suggestions"][0]
                        if "無符合" not in sug: name = sug.replace(base, '').replace('\t', '').strip()
            except: pass
        if name and not any(bad in name for bad in bad_words):
            local_map[ticker] = name; json_save(TW_NAMES_FILE, local_map)
            return name
    return ticker

if 'watchlists' not in st.session_state: st.session_state.watchlists = load_watchlists()
if 'active_list' not in st.session_state: st.session_state.active_list = list(st.session_state.watchlists.keys())[0]
if 'user_opened_list' not in st.session_state: st.session_state.user_opened_list = None 
if 'current_ticker' not in st.session_state:
    st.session_state.current_ticker = "^IXIC"
    for wl_name, wl in st.session_state.watchlists.items():
        if wl: st.session_state.current_ticker = wl[0]; st.session_state.active_list = wl_name; break

# --- 2. 核心搜尋與新聞引擎 ---
def get_ticker_metadata(ticker):
    name = get_stock_name(ticker)
    base = ticker.split('.')[0]
    mapping = {'NVDA': {'ceo': ['黃仁勳', 'Jensen'], 'key': ['Nvidia']}, 'TSLA': {'ceo': ['馬斯克', 'Elon'], 'key': ['Tesla']}, 'AAPL': {'ceo': ['庫克', 'Tim'], 'key': ['Apple']}, 'MU': {'ceo': ['Sanjay'], 'key': ['Micron']}, 'TSM': {'ceo': ['魏哲家'], 'key': ['TSMC']}}
    meta = mapping.get(base, {'ceo': [], 'key': [base]}); meta['name'] = name
    return meta

def validate_news(title, ticker, info, strict=True):
    t = title.lower(); bt = ticker.split('.')[0].lower(); wl = [bt] + [k.lower() for k in info['key']] + [info['name'].lower()] + [c.lower() for c in info['ceo']]
    if any(w in t for w in wl): return True
    if not strict and bt in t and any(k in t for k in ['options', 'volume', 'shares', 'trading', '期權', '成交']): return True
    return False

def fetch_deep_news(ticker, is_macro=False):
    try:
        hdrs = {'User-Agent': 'Mozilla/5.0'}; items = []; now = datetime.now()
        if is_macro:
            resp = requests.get("https://news.google.com/rss/search?q=聯準會+升息+通膨+鮑爾&hl=zh-TW&gl=TW&ceid=TW:zh-Hant", headers=hdrs, timeout=4)
            if resp.status_code == 200:
                for item in ET.fromstring(resp.content).findall('.//item')[:6]:
                    d = parse_rss(item, "macro", "Global"); 
                    if (now - d['dt']).days <= 30: items.append(d)
            return items
        
        info = get_ticker_metadata(ticker); cn = info['name']; trg = f"{ticker} OR {cn}" if cn else ticker; small = ticker.split('.')[0] in ['ONDS', 'RXRX'] 
        if ".TW" in ticker or ".TWO" in ticker:
            base_tk = ticker.split('.')[0]
            q = f"({base_tk}+OR+{cn})+(重訊+OR+重大訊息+OR+營收+OR+公告+OR+自結)"
            try:
                resp = requests.get(f"https://news.google.com/rss/search?q={q}&hl=zh-TW&gl=TW&ceid=TW:zh-Hant", headers=hdrs, timeout=4)
                if resp.status_code == 200:
                    for item in ET.fromstring(resp.content).findall('.//item')[:10]:
                        d = parse_rss(item, "tw_local", "台股重訊")
                        if (now - d['dt']).days <= 30 and validate_news(d['title'], ticker, info): items.append(d)
            except: pass
        else:
            q_sec = f"{ticker}+stock+(SEC+Filing+OR+Form+4+OR+Insider)"
            q_news = f"{ticker}+stock+(Options+OR+Volume)" if small else f"{trg}+stock+(財聯社+OR+鉅亨網+OR+營收)"
            for url, cat, src in [(f"https://news.google.com/rss/search?q={q_sec}&hl=en-US&gl=US&ceid=US:en", "us_sec", "🏛️ SEC"), (f"https://news.google.com/rss/search?q={q_news}&hl=zh-TW&gl=TW&ceid=TW:zh-Hant", "us_news", "📰 News")]:
                try:
                    resp = requests.get(url, headers=hdrs, timeout=4)
                    if resp.status_code == 200:
                        for item in ET.fromstring(resp.content).findall('.//item')[:8]:
                            d = parse_rss(item, cat, src)
                            if (now - d['dt']).days <= 30 and validate_news(d['title'], ticker, info, not small): items.append(d)
                except: pass
        items.sort(key=lambda x: x['dt'], reverse=True); return items
    except: return []

def parse_rss(item, cat, src):
    title = re.sub('<[^<]+?>', '', item.find('title').text); link = item.find('link').text
    try: dt = datetime.strptime(item.find('pubDate').text[:16], '%a, %d %b %Y')
    except: dt = datetime.now()
    if "futunn" in link: src = "🐂 富途"
    elif "yahoo" in link: src = "🇺🇸 Yahoo"
    return {'title': title, 'link': link, 'date': dt.strftime('%m/%d'), 'dt': dt, 'cat': cat, 'src': src}

def analyze_news_strict(title, cat):
    t = title.lower(); s = 0; tag = ""; maj = False; pen = 0.0
    if any(w in t for w in ['豪宅', '買房', '神操作']): return 0, "", False, 0
    if any(w in t for w in ['減持', '賣出', '降評', '大跌', '崩盤', 'miss']) and 'form 4' not in t and 'insider' not in t: return -2.0, '<span class="tag-red">Risk</span>', False, 0
    if any(x in t for x in ['營收', 'eps', 'profit', '獲利', '財報', '重訊']):
        if any(x in t for x in ['新高', 'beat', '增', '漲', '超預期']): s, tag, maj = 4.0, '<span class="tag-hard">💎 利多</span>', True
        elif any(x in t for x in ['miss', 'down', 'loss', '虧']): s, tag = -3.0, '<span class="tag-red">📉 利空</span>'
        else: s, tag = 1.5, '<span class="tag-hard">📊 數據</span>'
    if not tag: tag = '<span class="tag-hard">📈 情報</span>' if s >= 0 else '<span class="tag-red">📉 利空</span>'
    return s, tag, maj, pen

def get_realtime_macro():
    news = fetch_deep_news("Macro", is_macro=True)
    ns = sum(-1.5 if any(w in n['title'].lower() for w in ['hike', 'inflation', '升息', '鷹']) else 1 if any(w in n['title'].lower() for w in ['cut', '降息', '鴿']) else 0 for n in news)
    txt, note, col, sc = ("宏觀穩健", "多頭支撐", "sig-green", 0) if ns == 0 else ("Fed 偏鷹", "系統風險高", "sig-red", -2) if ns <= -3 else ("宏觀偏空", "震盪觀望", "sig-orange", -1) if ns < 0 else ("宏觀利多", "資金寬鬆", "sig-green", 1)
    try:
        hist = yf.Ticker("^IXIC").history(period="5d")
        if len(hist) >= 2:
            chg = (hist.iloc[-1]['Close'] - hist.iloc[-2]['Close']) / hist.iloc[-2]['Close'] * 100
            if chg < -1.5: txt, note, col, sc = "市場恐慌", f"納指重挫 {chg:.2f}%", "sig-red", -3
    except: pass
    return txt, note, col, sc

def update_anchor(ticker, news_title, score, news_date_str):
    anchors = load_anchors()
    anchors[ticker] = {"title": news_title, "summary": "重大情報", "score": score, "date": news_date_str, "saved_at": datetime.now().strftime("%Y-%m-%d")}
    save_anchor_data(anchors)

def get_valid_anchor(ticker):
    anchors = load_anchors()
    if ticker not in anchors: return None
    data = anchors[ticker]
    try:
        if (datetime.now() - datetime.strptime(data["saved_at"], "%Y-%m-%d")).days > 5:
            del anchors[ticker]; save_anchor_data(anchors); return None
    except: return None
    return data

# --- 3. 技術指標與圖表演算法 ---
def calculate_volume_profile(df, bins=40, filter_mask=None):
    if df.empty: return pd.DataFrame({'Price': [], 'Volume': []})
    sub = df if filter_mask is None else df[filter_mask]; b = np.linspace(df['Low'].min(), df['High'].max(), 41); c = (b[:-1] + b[1:])/2
    if sub.empty: return pd.DataFrame({'Price': c, 'Volume': np.zeros(40)})
    return pd.DataFrame({'Price': c, 'Volume': sub.groupby(pd.cut(sub['Close'], b, labels=False, include_lowest=True))['Volume'].sum().reindex(range(40), fill_value=0).values})

def calculate_indicators(df):
    if len(df) < 50: return df
    df['SMA_20'] = df['Close'].rolling(20).mean(); df['SMA_60'] = df['Close'].rolling(60).mean(); df['Vol_SMA5'] = df['Volume'].rolling(5).mean()
    sd = df['Close'].rolling(20).std(); df['Bollinger_Upper'] = df['SMA_20'] + sd*2; df['Bollinger_Lower'] = df['SMA_20'] - sd*2
    df['KC_Upper'] = df['SMA_20'] + df['SMA_20']*0.05; df['KC_Lower'] = df['SMA_20'] - df['SMA_20']*0.05
    df['Squeeze_On'] = (df['Bollinger_Upper'] < df['KC_Upper']) & (df['Bollinger_Lower'] > df['KC_Lower'])
    tr = np.max(pd.concat([df['High']-df['Low'], np.abs(df['High']-df['Close'].shift()), np.abs(df['Low']-df['Close'].shift())], axis=1), axis=1)
    df['ATR'] = tr.rolling(14).mean(); df['ATR_Trailing_Stop'] = df['High'].rolling(22).max() - df['ATR']*3
    delta = df['Close'].diff(); df['RSI'] = 100 - (100 / (1 + (delta.where(delta>0,0).rolling(14).mean() / -delta.where(delta<0,0).rolling(14).mean())))
    df['MACD'] = df['Close'].ewm(span=12, adjust=False).mean() - df['Close'].ewm(span=26, adjust=False).mean()
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean(); df['MACD_Hist'] = df['MACD'] - df['Signal_Line']
    clv = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low']); df['AD_Line'] = (clv.fillna(0) * df['Volume']).cumsum()
    df['DMA_DDD'] = df['Close'].rolling(10).mean() - df['Close'].rolling(50).mean(); df['DMA_AMA'] = df['DMA_DDD'].rolling(10).mean()
    buy_seq = np.zeros(len(df), dtype=int); sell_seq = np.zeros(len(df), dtype=int)
    for i in range(4, len(df)):
        buy_seq[i] = buy_seq[i-1] + 1 if df['Close'].iloc[i] < df['Close'].iloc[i-4] else 0
        sell_seq[i] = sell_seq[i-1] + 1 if df['Close'].iloc[i] > df['Close'].iloc[i-4] else 0
    df['TD_Buy_9'] = np.where(buy_seq == 9, df['Close'], np.nan); df['TD_Sell_9'] = np.where(sell_seq == 9, df['Close'], np.nan)
    return df

def find_structural_box_bottom(df, current_price):
    if len(df) < 60: return current_price, df.index[0], df.index[-1], False
    p = df.tail(120).copy(); p['M'] = p['Low'].rolling(5, center=True, min_periods=1).min()
    v = p[(p['Low'] == p['M']) & (p['Low'] < current_price * 0.98)]
    if not v.empty:
        idx = np.where(p.index == v.iloc[-1].name)[0][0]
        return v.iloc[-1]['Low'], p.index[max(0, idx-15)], p.index[min(len(p)-1, idx+15)], False
    return p['Low'].min(), p.index[0], p.index[-1], True

# --- 🚀 劇本推演修正 (緩衝比例) ---
def generate_projection_points(df, trend_text, cur_p, iron_p, is_brk, t_s, t_l):
    last_d = df.index[-1]; f_d = []
    d = last_d
    while len(f_d) < 20:
        d += pd.Timedelta(days=1)
        if d.weekday() < 5: f_d.append(d)
    
    x = [last_d]; y = [cur_p]
    ma20 = df['SMA_20'].iloc[-1] if not pd.isna(df['SMA_20'].iloc[-1]) else cur_p
    recent_high_20 = df['High'].tail(20).max()
    scenario_name = ""

    if cur_p > t_s * 0.98:
        scenario_name = "🔄 突破後回測 (支撐驗證)"
        dip = max(t_s * 0.98, cur_p * 0.94) 
        rally = min(t_l, cur_p * 1.08, recent_high_20 * 1.1) 
        x.extend([f_d[2], f_d[8], f_d[15]]); y.extend([dip, cur_p * 1.02, rally])
    elif "牛市" in trend_text:
        scenario_name = "🐂 牛市 N 字突破"
        dip = max(cur_p * 0.96, ma20) if cur_p > ma20 else cur_p * 0.96
        rally = max(cur_p * 1.05, recent_high_20 * 1.02)
        x.extend([f_d[4], f_d[14]]); y.extend([dip, rally])
    elif "熊市" in trend_text:
        if is_brk:
            scenario_name = "🕳️ 熊市無底洞墜落"
            x.extend([f_d[4], f_d[14]]); y.extend([min(cur_p * 1.03, ma20), cur_p * 0.92])
        else:
            scenario_name = "🐻 熊市死貓反彈"
            x.extend([f_d[4], f_d[14]]); y.extend([min(cur_p * 1.06, ma20) if cur_p < ma20 else cur_p * 1.05, cur_p * 0.92])
    else:
        scenario_name = "⚖️ 區間震盪收斂"
        x.extend([f_d[5], f_d[12], f_d[18]]); y.extend([cur_p * 1.04, cur_p * 0.96, cur_p * 1.02])
        
    return x, y, scenario_name

def analyze_market_trend(df):
    c, m20, m60 = df['Close'].iloc[-1], df['SMA_20'].iloc[-1], df['SMA_60'].iloc[-1]
    if c > m20 > m60: return "🐂 牛市", "多頭排列", "sig-green"
    elif c < m20 < m60: return "🐻 熊市", "空頭排列", "sig-red"
    else: return "⚖️ 震盪", "區間整理", "sig-orange"

def get_stock_engine_mode(ticker):
    return "🏢 權值穩健", "trend"

def get_relative_strength(ticker, stock_df):
    try:
        b = yf.download("^TWII" if ".TW" in ticker else "^GSPC", period="1mo", progress=False)['Close'].iloc[:, 0]
        a = stock_df['Close'].reindex(b.index, method='ffill')
        if len(b) > 20:
            diff = ((a.iloc[-1] - a.iloc[-20])/a.iloc[-20]) - ((b.iloc[-1] - b.iloc[-20])/b.iloc[-20])
            if diff > 0.05: return "🦁 領頭羊 (強)", "sig-green"
            elif diff > 0: return "🐯 優於大盤", "sig-blue"
    except: pass
    return "⚖️ 跟隨大盤", "sig-gray"

def detect_smart_money_status(df):
    if len(df) < 10: return None
    latest = df.iloc[-1]
    if latest['Close'] < latest['Bollinger_Lower'] and latest['RSI'] < 30: return "⚡ 乖離抄底 (超賣)"
    price_now, price_5d = latest['Close'], df['Close'].iloc[-6]; ad_now, ad_5d = latest['AD_Line'], df['AD_Line'].iloc[-6]; rsi = latest['RSI']
    if price_now < price_5d * 0.98 and ad_now > ad_5d and rsi < 50: return "🎯 主力背離吸籌"
    if rsi > 65 and latest['Volume'] > latest['Vol_SMA5'] * 1.3 and (latest['Close'] < latest['Open']): return "🔴 主力調節 (爆量滯漲)"
    return None

def analyze_strategic_signals(df):
    if df.empty: return {}
    latest = df.iloc[-1]
    macd, signal = latest['MACD'], latest['Signal_Line']
    macd_text, macd_color = ("零軸上金叉", "sig-green") if macd > signal and macd > 0 else ("零軸下金叉", "sig-orange") if macd > signal else ("零軸上死叉", "sig-orange") if macd > 0 else ("零軸下死叉", "sig-red")
    vol, vol_ma = latest['Volume'], latest['Vol_SMA5']
    vol_text, vol_color = ("爆量", "sig-green") if vol > vol_ma * 1.5 else ("量縮", "sig-gray")
    rsi = latest['RSI']
    rsi_text, rsi_color = (f"過熱 ({rsi:.0f})", "sig-red") if rsi > 70 else (f"超賣 ({rsi:.0f})", "sig-green") if rsi < 30 else (f"中性 ({rsi:.0f})", "sig-gray")
    summary, summary_color = "觀望", "sig-gray"
    if latest.get('Squeeze_On', False): summary, summary_color = "🌀 壓縮蓄力中", "sig-cyan"
    elif macd > signal: summary, summary_color = "📈 偏多震盪", "sig-green"
    else: summary, summary_color = "⛈️ 空頭走勢", "sig-red"
    return {"MACD_Text": macd_text, "MACD_Color": macd_color, "Vol_Text": vol_text, "Vol_Color": vol_color, "RSI_Text": rsi_text, "RSI_Color": rsi_color, "Summary": summary, "Summary_Color": summary_color}

def predict_target_and_rating(df):
    price, upper, recent_high_60 = df['Close'].iloc[-1], df['Bollinger_Upper'].iloc[-1], df['High'].tail(60).max()
    t_s = upper if price >= recent_high_60 else min(upper, recent_high_60)
    return t_s, max(recent_high_60 * 1.15, t_s * 1.1), "強勢" if price > df['SMA_20'].iloc[-1] else "持有"

def format_volume(num): return f"{num/1e9:.2f}B" if num >= 1e9 else f"{num/1e6:.2f}M" if num >= 1e6 else f"{num}"

@st.cache_data(ttl=300)
def get_earnings_status(ticker):
    try:
        t = yf.Ticker(ticker); next_date = "N/A"
        try:
            cal = t.calendar
            if cal is not None and not cal.empty and 'Earnings Date' in cal: next_date = cal['Earnings Date'][0].strftime('%Y-%m-%d')
        except: pass
        return f"📅 財報: {next_date}", "⚪ 無數據"
    except: return "📅 財報: N/A", "⚪ 無數據"

# --- 🎯 戰術決策引擎 (加入黃金右側/破底翻 邏輯) ---
def get_tactical_advice(df, t_s, t_l):
    if len(df) < 5: return "無狀態", "gray", "資料不足", None
    
    latest = df.iloc[-1]; prev = df.iloc[-2]
    cp, op, hi, lo = latest['Close'], latest['Open'], latest['High'], latest['Low']
    vol, vol_ma = latest['Volume'], latest['Vol_SMA5']
    
    zone_upper = t_s * 1.015
    zone_lower = t_s * 0.985
    
    upper_shadow = (hi - max(op, cp)) / (hi - lo + 1e-9)
    lower_shadow = (min(op, cp) - lo) / (hi - lo + 1e-9)
    
    # 破底翻 (Spring) 右側買點邏輯
    recent_lows = df['Low'].iloc[-4:-1] # 過去3天
    dipped_recently = any(recent_lows < zone_lower)
    pierced_today = lo < zone_lower
    spring_A = dipped_recently and (cp > zone_lower) and (cp > op)
    spring_B = pierced_today and (cp > zone_lower) and (lower_shadow > 0.6)
    is_spring = (spring_A or spring_B) and (cp < zone_upper * 1.02)
    
    is_peak = (cp >= t_s * 0.99) and (op < prev['Close'] or upper_shadow > 0.6)
    is_bottom = (zone_lower <= cp <= zone_upper) and (vol < vol_ma * 0.8) and (cp > op or lower_shadow > 0.6)

    if is_peak: return "‼️ 減碼警示 (觸壓反轉)", "#dc2626", f"價格觸及目標 ${t_s:.2f} 且出現高檔反轉特徵。建議啟動獲利了結！", "PEAK"
    if is_spring: return "💎 破底翻確認 (右側買點)", "#eab308", f"股價假跌破 ${zone_lower:.2f} 後迅速站回，且出現強勢拒絕訊號(收紅或長下影線)。主力洗盤結束，為絕佳右側進場點！", "SPRING"
    if is_bottom: return "✅ 支撐接點確認 (量縮回測)", "#059669", f"價格回測 ${t_s:.2f} 且量縮守穩。實體紅K確認支撐有效，可伺機佈局。", "BOTTOM"
    if cp >= t_s * 1.04: return "🚀 乖離過大（超漲）", "#ef4444", "股價已大幅超越短期目標，隨時面臨獲利了結賣壓。禁止追高！", None
    if cp > t_s: return "🎯 達標警戒區 (突破)", "#f59e0b", "已站上短期目標價。目前強勢，但需留意隨時可能發動的向下回測。", None
    if zone_lower <= lo <= zone_upper:
        if vol < vol_ma * 0.8: return "📥 進入支撐區 (縮量)", "#3b82f6", "目前正落入目標支撐帶。量縮代表賣壓減輕，請密切觀察收盤是否站穩。", "IN_ZONE"
        else: return "⚔️ 支撐保衛戰 (帶量)", "#0ea5e9", "帶量測試關鍵支撐帶中，多空分歧大。建議多看 1-2 天，等待勝負揭曉。", "IN_ZONE"
    return "📈 趨勢發展中", "#10b981", f"結構穩健運行中。距離短線目標 ${t_s:.2f} 尚有空間，可依照原定方向觀望。", None

# --- 6. 主介面 Sidebar ---
with st.sidebar:
    st.title("🎛️ 控制台")
    st.header("📱 平台顯示設定")
    mobile_mode = st.toggle("啟用手機防卡死模式", value=False)
    st.markdown("---")

    st.header("📌 多維度自選股清單")
    cur_t = st.session_state.current_ticker; act_l = st.session_state.active_list
    
    for wl_name, tickers in st.session_state.watchlists.items():
        is_exp = (wl_name == st.session_state.user_opened_list)
        with st.expander(f"📁 {wl_name} ({len(tickers)}檔)", expanded=is_exp):
            for t in tickers:
                is_sel = (t == cur_t and wl_name == act_l)
                btn_t = "primary" if is_sel else "secondary"
                s_name = get_stock_name(t)
                disp_name = f"{s_name} ({t})" if s_name != t else t
                
                if st.button(f"{'👉 ' if is_sel else ''}{disp_name}", key=f"btn_{wl_name}_{t}", type=btn_t, use_container_width=True):
                    st.session_state.current_ticker = t; st.session_state.active_list = wl_name; st.session_state.user_opened_list = wl_name; st.rerun()

    st.markdown("---")
    c1, c2 = st.columns(2); c3, c4 = st.columns(2)
    lst = st.session_state.watchlists[act_l]
    idx = lst.index(cur_t) if cur_t in lst else -1
    if c1.button("⏫ 置頂") and idx > 0: lst.insert(0, lst.pop(idx)); save_watchlists(st.session_state.watchlists); st.rerun()
    if c2.button("⬆️ 上移") and idx > 0: lst[idx], lst[idx-1] = lst[idx-1], lst[idx]; save_watchlists(st.session_state.watchlists); st.rerun()
    if c3.button("⬇️ 下移") and 0 <= idx < len(lst)-1: lst[idx], lst[idx+1] = lst[idx+1], lst[idx]; save_watchlists(st.session_state.watchlists); st.rerun()
    if c4.button("⏬ 置底") and 0 <= idx < len(lst)-1: lst.append(lst.pop(idx)); save_watchlists(st.session_state.watchlists); st.rerun()

    st.markdown("---")
    time_opt = st.radio("選擇週期", ["當沖 (分時)", "日線 (Daily)", "週線 (Weekly)"], index=1)
    
    with st.expander("✏️ 編輯清單"):
        target_list = st.selectbox("加入抽屜：", list(st.session_state.watchlists.keys()), index=list(st.session_state.watchlists.keys()).index(act_l))
        new_t = st.text_input("代號", placeholder="2330.TW").upper()
        if st.button("➕ 新增", use_container_width=True) and new_t:
            if new_t not in st.session_state.watchlists[target_list]: st.session_state.watchlists[target_list].append(new_t); st.session_state.current_ticker = new_t; st.session_state.active_list = target_list; st.session_state.user_opened_list = target_list; save_watchlists(st.session_state.watchlists); st.rerun()
        if st.button("❌ 刪除股票", use_container_width=True) and cur_t in st.session_state.watchlists[act_l]:
            st.session_state.watchlists[act_l].remove(cur_t); save_watchlists(st.session_state.watchlists)
            if st.session_state.watchlists[act_l]: st.session_state.current_ticker = st.session_state.watchlists[act_l][0]
            else: st.session_state.current_ticker = "^IXIC"
            st.rerun()

# --- 主體資料載入 ---
main_title_name = get_stock_name(cur_t)
disp_main_title = f"{main_title_name} ({cur_t})" if main_title_name != cur_t else cur_t
st.title(f"📈 {disp_main_title} 實戰戰情室 V17.36")

api_p, api_i = ("5d", "15m") if "當沖" in time_opt else ("6mo", "1d") if "日" in time_opt else ("2y", "1wk")
df = yf.download(cur_t, period=api_p, interval=api_i, progress=False)
if df.empty: st.error("無數據"); st.stop()
if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
df = df.loc[:, ~df.columns.duplicated()]

df.index = pd.to_datetime(df.index)
if df.index.tz is not None: df.index = df.index.tz_localize(None)

df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['Open', 'High', 'Low', 'Close'])
if len(df) < 2: st.error("數據量不足"); st.stop()

df = calculate_indicators(df)
latest = df.iloc[-1]; close_v = float(latest['Close']); chg = (close_v - float(df.iloc[-2]['Close'])) / float(df.iloc[-2]['Close']) * 100
clr = "green" if chg >= 0 else "red"

sigs = analyze_strategic_signals(df)
trend_txt, trend_note, trend_col = analyze_market_trend(df)
rs_txt, rs_col = get_relative_strength(cur_t, df)
engine_label, engine_type = get_stock_engine_mode(cur_t)
macro_txt, macro_note, macro_col, macro_score = get_realtime_macro()
t_s, t_l, rating = predict_target_and_rating(df)

# --- 🚀 戰術提示盒 ---
tac_status, tac_color, tac_msg, tac_signal = get_tactical_advice(df, t_s, t_l)
st.markdown(f"""
<div class="tactical-box" style="border-left-color: {tac_color};">
    <div style="display: flex; align-items: center; margin-bottom: 10px;">
        <span style="font-size: 1.8rem; margin-right: 12px;">🚩</span>
        <h2 style="color: white; margin: 0; font-size: 1.7rem;">戰術建議：{tac_status}</h2>
    </div>
    <div style="background-color: rgba(255,255,255,0.05); padding: 12px; border-radius: 8px;">
        <p style="font-size: 1.2rem; color: #f0f2f6; margin: 0;"><b>💡 行動指南：</b> {tac_msg}</p>
    </div>
</div>
""", unsafe_allow_html=True)

iron_price, box_start, box_end, is_breaking = find_structural_box_bottom(df, close_v)

st.markdown(f"""
<div class="price-card">
    <h1 style="margin:0; font-size: 50px;">${close_v:.2f}</h1>
    <h3 style="margin:0; color: {clr};">{chg:+.2f}%</h3>
    <p style="color: gray;">量: {format_volume(latest['Volume'])}</p>
    <div style="margin-bottom: 5px; color: #00ffff; font-weight: bold; font-size: 20px;">🎯 波段探底: ${df['Bollinger_Lower'].iloc[-1]:.2f}</div>
    <div style="margin-bottom: 10px; color: #facc15; font-weight: bold; font-size: 20px;">🧱 歷史箱底: ${iron_price:.2f}</div>
</div>
""", unsafe_allow_html=True)

r1, r2, r3 = st.columns(3)
with r1: st.markdown(f'<div class="ai-box"><h5 style="color:white; margin:0; margin-bottom:5px;">📡 綜合戰略</h5><div style="font-size:16px;" class="{sigs["Summary_Color"]}">{sigs["Summary"]}</div><div style="margin-top:5px; font-size:14px;">MACD: <span class="{sigs["MACD_Color"]}">{sigs["MACD_Text"]}</span><br>RSI: <span class="{sigs["RSI_Color"]}">{sigs["RSI_Text"]}</span></div></div>', unsafe_allow_html=True)
with r2: st.markdown(f'<div class="ai-box"><h5 style="color:white; margin:0;">⚖️ 雙重格局</h5><div style="margin-top:5px; font-size:14px;">🏢 個股: <span class="{trend_col}">{trend_txt} ({trend_note})</span><br>🌍 宏觀: <span class="{macro_col}">{macro_txt}</span></div></div>', unsafe_allow_html=True)
with r3: st.markdown(f'<div class="ai-box" style="border: 1px solid #00d4ff;"><h5 style="color:white; margin:0;">🎯 AI 目標 & 強弱</h5><div style="margin-top:5px; font-size:14px;">短: ${t_s:.2f} | 長: ${t_l:.2f}<br><span class="{rs_col}">{rs_txt}</span></div></div>', unsafe_allow_html=True)

# --- 繪圖區 ---
p_data = df.tail(120) if "日" in time_opt else df.tail(60)
fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_width=[0.2, 0.2, 0.6])
fig.add_trace(go.Candlestick(x=p_data.index, open=p_data['Open'], high=p_data['High'], low=p_data['Low'], close=p_data['Close'], name="K線"), row=1, col=1)

# ATR 停損與鐵板防線
fig.add_trace(go.Scatter(x=p_data.index, y=p_data['ATR_Trailing_Stop'], mode='lines', line=dict(color='#FF5F1F', width=1.5, dash='dot'), name='ATR 停損'), row=1, col=1)
if not is_breaking and iron_price > 0:
    fig.add_hline(y=iron_price, line_dash="dash", line_color="#20c997", annotation_text=f"🧱 鐵板 ${iron_price:.2f}", annotation_font_color="#20c997", annotation_position="bottom right", row=1, col=1)

# 🟦 支撐/壓力轉換區塊 (Flip Zone) 淺藍色 & 置右 🟦
zone_upper = t_s * 1.015
zone_lower = t_s * 0.985
fig.add_hrect(
    y0=zone_lower, 
    y1=zone_upper, 
    line_width=0, 
    fillcolor="#BEEFFF", 
    opacity=0.15, 
    annotation_text=f"支撐/壓力帶 (突破: ${zone_upper:.2f} | 跌破: ${zone_lower:.2f})", 
    annotation_position="top right", 
    annotation_font_color="#f760eb", 
    row=1, col=1
)

# 自動推演 (每日校準)
px, py, sc_name = generate_projection_points(df, trend_txt, close_v, iron_price, is_breaking, t_s, t_l)
fig.add_trace(go.Scatter(x=px, y=py, mode='lines+markers', line=dict(color='#eab308', width=3, dash='dash'), marker=dict(size=8, symbol='diamond', color='#eab308'), name='🔮 AI 驗證式推演'), row=1, col=1)
for i in range(1, len(px)): fig.add_annotation(x=px[i], y=py[i], text=f"${py[i]:.2f}", showarrow=True, arrowhead=0, ay=-20 if py[i]>py[i-1] else 20, font=dict(color="#eab308", size=11), bgcolor="rgba(0,0,0,0.6)", row=1, col=1)
fig.add_annotation(x=0.01, y=0.98, xref="paper", yref="paper", text=f"🔮 目前 AI 推演：{sc_name}", showarrow=False, font=dict(color="white", size=14, weight="bold"), bgcolor="rgba(0, 0, 0, 0.6)", bordercolor="#eab308", borderwidth=1, borderpad=6)

last_d = p_data.index[-1]
for r in range(1, 4): fig.add_vline(x=last_d, line_dash="dash", line_color="#666", opacity=0.7, row=r, col=1)
fig.add_trace(go.Scatter(x=[last_d], y=[close_v], mode='markers', marker=dict(size=12, color='#00ffff', line=dict(color='white', width=2)), name="今日收盤"), row=1, col=1)
fig.add_annotation(x=last_d, y=p_data['High'].max(), text="🗓️ 今日", showarrow=False, yshift=20, font=dict(color="#aaa", size=11), row=1, col=1)

# 特徵點標記 (抓出那一刻)
if tac_signal == "PEAK":
    fig.add_annotation(x=last_d, y=p_data['High'].iloc[-1]*1.02, text="‼️ 減碼警示", showarrow=True, arrowhead=1, ay=-50, row=1, col=1, font=dict(color="white", size=14, weight="bold"), bgcolor="#dc2626", bordercolor="white")
elif tac_signal == "SPRING":
    fig.add_annotation(x=last_d, y=p_data['Low'].iloc[-1]*0.98, text="💎 破底翻買點", showarrow=True, arrowhead=1, ay=50, row=1, col=1, font=dict(color="black", size=14, weight="bold"), bgcolor="#eab308", bordercolor="white")
elif tac_signal == "BOTTOM":
    fig.add_annotation(x=last_d, y=p_data['Low'].iloc[-1]*0.98, text="✅ 支撐接點", showarrow=True, arrowhead=1, ay=50, row=1, col=1, font=dict(color="white", size=14, weight="bold"), bgcolor="#059669", bordercolor="white")
elif tac_signal == "IN_ZONE":
    fig.add_annotation(x=last_d, y=p_data['Low'].iloc[-1]*0.98, text="📥 進入支撐帶", showarrow=True, arrowhead=1, ay=50, row=1, col=1, font=dict(color="white", size=12, weight="bold"), bgcolor="#0ea5e9", bordercolor="white")

for i in range(5, len(p_data)):
    curr, prior = p_data.iloc[i], p_data.iloc[i-1]
    
    is_overheated = (curr['RSI'] > 72) and not (prior['RSI'] > 72)
    is_target_hit = (curr['High'] >= t_s) and not (prior['High'] >= t_s)
    
    if is_target_hit or is_overheated:
        lbl_txt = f"💰達標<br>${curr['High']:.1f}" if is_target_hit else f"🔥過熱<br>${curr['High']:.1f}"
        bg_c = "#ffc107" if is_target_hit else "#ff4500"
        txt_c = "black" if is_target_hit else "white"
        fig.add_annotation(x=p_data.index[i], y=curr['High']*1.03, text=lbl_txt, showarrow=True, arrowhead=1, ay=-85, row=1, col=1, bgcolor=bg_c, font=dict(color=txt_c, size=9))

    status = detect_smart_money_status(p_data.iloc[:i+1])
    if status:
        if "吸籌" in status: fig.add_annotation(x=p_data.index[i], y=curr['Low']*0.98, text=f"🐳吸<br>${curr['Low']:.1f}", showarrow=True, arrowhead=1, ay=40, row=1, col=1, bgcolor="#6f42c1", font=dict(color="white", size=9))
        elif "調節" in status: fig.add_annotation(x=p_data.index[i], y=curr['High']*1.02, text=f"🔴調節<br>${curr['High']:.1f}", showarrow=True, arrowhead=1, ay=-60, row=1, col=1, bgcolor="#b91c1c", font=dict(color="white", size=10, weight="bold"))

    macd_buy = (curr['MACD'] > curr['Signal_Line']) and (prior['MACD'] <= prior['Signal_Line'])
    macd_sell = (curr['MACD'] < curr['Signal_Line']) and (prior['MACD'] >= prior['Signal_Line'])
    if macd_buy: fig.add_annotation(x=p_data.index[i], y=curr['Low']*0.98, text=f"BUY<br>${curr['Close']:.1f}", showarrow=True, arrowhead=1, ay=40, row=1, col=1, bgcolor="#28a745", font=dict(color="white", size=9))
    if macd_sell: fig.add_annotation(x=p_data.index[i], y=curr['High']*1.02, text=f"SELL<br>${curr['Close']:.1f}", showarrow=True, arrowhead=1, ay=-40, row=1, col=1, bgcolor="#dc3545", font=dict(color="white", size=9))

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

fig.update_layout(dragmode=False if mobile_mode else 'zoom', height=800, template="plotly_dark", xaxis_rangeslider_visible=False, showlegend=False, margin=dict(t=10, b=10, l=10, r=10))
st.plotly_chart(fig, use_container_width=True)

# 雙籌碼分析圖
try:
    c1, c2 = st.columns(2); mf = ((p_data['Close'] - p_data['Open']) / (p_data['High'] - p_data['Low'])) * p_data['Volume']; mf = mf.fillna(0).cumsum()
    with c1:
        st.caption("主力資金流 (Money Flow)")
        fig_mf = go.Figure(go.Scatter(x=p_data.index, y=mf, fill='tozeroy', line=dict(color='#00d4ff')))
        fig_mf.update_layout(dragmode=False if mobile_mode else 'zoom', height=250, template="plotly_dark", margin=dict(t=10, b=10, l=10, r=10)); st.plotly_chart(fig_mf, use_container_width=True, config={'displayModeBar': False})
    with c2:
        st.caption("籌碼分佈 (主力 vs 散戶)")
        inst_mask = (p_data['Close'] > p_data['Open']) & (p_data['Volume'] > p_data['Vol_SMA5'])
        vp_all = calculate_volume_profile(p_data); vp_main = calculate_volume_profile(p_data, filter_mask=inst_mask)
        fig_vp = go.Figure()
        fig_vp.add_trace(go.Scatter(x=vp_all['Price'], y=vp_all['Volume'], fill='tozeroy', line=dict(color='#ffaa00', width=0), name='整體'))
        fig_vp.add_trace(go.Scatter(x=vp_main['Price'], y=vp_main['Volume'], fill='tozeroy', line=dict(color='#00d4ff', width=2), name='主力'))
        fig_vp.add_vline(x=close_v, line_dash="dash", line_color="white")
        fig_vp.update_layout(dragmode=False if mobile_mode else 'zoom', height=250, template="plotly_dark", margin=dict(t=10, b=10, l=10, r=10), showlegend=True, legend=dict(orientation="h", y=1.1)); st.plotly_chart(fig_vp, use_container_width=True, config={'displayModeBar': False})
except: pass

st.markdown("---")

# 新聞與 AI 勝率推演面板
engine_name = "🇹🇼 台股重訊模式" if ".TW" in cur_t or ".TWO" in cur_t else "🇺🇸 美股雙境獵手"
with st.spinner(f"🕵️‍♂️ 啟動{engine_name}：正在掃描並過濾新聞情報..."): items = fetch_deep_news(cur_t, is_macro=False)

news_score, total_insider_penalty, valid_count, has_major, processed = 0, 0, 0, False, []
anchor = get_valid_anchor(cur_t)
if anchor: news_score += anchor['score']

for item in items:
    s, tag, major, penalty = analyze_news_strict(item['title'], item['cat'])
    if s == 0 and not tag: continue
    news_score += s; total_insider_penalty += penalty; valid_count += 1
    if major: has_major = True
    if major and s > 0: update_anchor(cur_t, item['title'], 3.0, item['date'])
    processed.append({'data': item, 'score': s, 'tag': tag})

win_rate = max(10.0, min(95.0, 50.0 + (news_score * 5) + (macro_score * 5) - (20 if total_insider_penalty <= -3.0 else 0)))

if total_insider_penalty <= -4.0:
    final_verdict, v_col = f"⚠️ 謹慎持有 (熔斷)", "#ffc107"
    m_disp = f'<div class="macro-alert" style="background-color:#3a1b1b; color:#ffc107;">⚡ 觸發內部人熔斷</div>'
elif has_major:
    final_verdict, v_col = f"🚀 強力看漲 (霸體)！ (+{news_score:.1f})", "#4ade80"
    m_disp = f'<div class="macro-alert" style="background-color:#1b3a1b; color:#4ade80;">💎 偵測到重大訊息</div>'
else:
    news_score += macro_score
    m_disp = f'<div class="macro-alert">{macro_txt}：{macro_note}</div>' if macro_score < 0 else f'<div style="color:orange;">{macro_txt}</div>'
    if news_score >= 3: final_verdict, v_col = "📈 偏多操作", "#ffc107"
    elif news_score <= -2: final_verdict, v_col = "📉 偏空看待", "#ff6b6b"
    else: final_verdict, v_col = "☁️ 觀望整理", "gray"

nc1, nc2 = st.columns([0.4, 0.6])
with nc1:
    st.markdown(m_disp, unsafe_allow_html=True)
    if anchor: st.markdown(f'<div class="anchor-box"><div class="anchor-title-cn">⚓ 記憶錨定 ({anchor["date"]})</div><div style="color: #00ffff; margin-bottom:5px;">{anchor["summary"]} (+{anchor["score"]}分)</div><div class="anchor-title-en">{anchor["title"]}</div></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="ai-box" style="text-align:left;"><h3 style="color:white; margin:0;">🔮 AI 戰情推演</h3><div style="font-size:18px; color:{v_col}; font-weight:bold; margin-top:10px;">{final_verdict}</div><div style="font-size:24px; color:#00ffff; font-weight:bold; margin-top:5px;">📈 多方勝率：{win_rate:.1f}%</div><hr style="border-color:#555;"><div style="font-size:14px; color:#ccc;"><b>基本面總分：</b> {news_score:.1f}<br><b>內部人扣分：</b> {total_insider_penalty:.1f}<br><b>有效情報數：</b> {valid_count} 則<br></div></div>', unsafe_allow_html=True)
    
with nc2:
    st.caption(f"目前搜尋引擎：{engine_name} | ⏳ 時效：30天 | 🛡️ 農場文過濾：開啟")
    if processed:
        for p in processed:
            style = "border-left: 4px solid #00ffff; background-color: #003366;" if p['score'] >= 4 else "border-left: 4px solid #dc3545;" if p['score'] <= -2 else "border-left: 4px solid #4ade80;" if p['score'] > 0 else "border-left: 4px solid #555;"
            st.markdown(f'<div class="news-card" style="{style}"><a href="{p["data"]["link"]}" target="_blank" class="news-link"><span class="news-src">{p["data"].get("src", "News")}</span> <span class="news-date">{p["data"]["date"]}</span> {p["tag"]} {p["data"]["title"]}</a></div>', unsafe_allow_html=True)
    else: st.info("暫無 30 天內的重大情報")
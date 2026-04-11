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
st.set_page_config(page_title="AI 實戰戰情室 V17.46 (全景價格通道版)", layout="wide", page_icon="🛡️")

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
    .anchor-box {background-color: #1b3a4a; color: #00d4ff; padding: 12px; border-radius: 5px; border: 1px solid #00d4ff; margin-bottom: 10px; font-size: 13px; text-align: left;}
    .anchor-title-cn {color: #fff; font-weight: bold; font-size: 14px; margin-bottom: 4px;}
    .anchor-title-en {color: #aaa; font-size: 11px; font-style: italic;}
    .earnings-tag {background-color: #2c2c2e; padding: 5px 10px; border-radius: 15px; font-size: 13px; margin-top: 10px; border: 1px solid #555; display: inline-block; margin-right: 8px;}
    .engine-tag {background-color: #1e3a8a; color: #38bdf8; padding: 5px 10px; border-radius: 15px; font-size: 13px; margin-top: 10px; border: 1px solid #38bdf8; display: inline-block; font-weight: bold;}
    .earn-beat {color: #4ade80; font-weight: bold;}
    .earn-miss {color: #ff6b6b; font-weight: bold;}
    .earn-warn {color: #ffaa00; font-weight: bold;}
    .earn-turn {color: #facc15; font-weight: bold;}
    .tag-sec {background-color: #003366; color: #00ffff; border: 1px solid #00ffff; padding: 2px 8px; border-radius: 4px; font-size: 12px; font-weight: bold; margin-right: 5px;}
    .tag-vip {background-color: #4a1b4a; color: #d8b4fe; border: 1px solid #a855f7; padding: 2px 6px; border-radius: 4px; font-size: 12px; margin-right: 5px;}
    .tag-hard {background-color: #1b3a1b; color: #4ade80; border: 1px solid #28a745; padding: 2px 6px; border-radius: 4px; font-size: 12px; margin-right: 5px;}
    .tag-div {background-color: #4a3b1b; color: #ffaa00; border: 1px solid #ffaa00; padding: 2px 6px; border-radius: 4px; font-size: 12px; margin-right: 5px;}
    .tag-risk {background-color: #3a1b1b; color: #ff6b6b; border: 1px solid #dc3545; padding: 2px 6px; border-radius: 4px; font-size: 12px; font-weight: bold;}
    .tag-chip {background-color: #555; color: #facc15; border: 1px solid #facc15; padding: 2px 6px; border-radius: 4px; font-size: 12px; font-weight: bold;}
    .sig-green {background-color: #1b3a1b; color: #4ade80; border: 1px solid #28a745; padding: 2px 6px; border-radius: 4px; font-size: 12px;}
    .sig-red {background-color: #3a1b1b; color: #ff6b6b; border: 1px solid #dc3545; padding: 2px 6px; border-radius: 4px; font-size: 12px;}
    .sig-gray {background-color: #333; color: #ccc; border: 1px solid #666; padding: 2px 6px; border-radius: 4px; font-size: 12px;}
    .sig-orange {background-color: #4a3b1b; color: #ffaa00; border: 1px solid #ffaa00; padding: 2px 6px; border-radius: 4px; font-size: 12px;}
    .sig-blue {background-color: #1b3a4a; color: #4a9eff; border: 1px solid #00d4ff; padding: 2px 6px; border-radius: 4px; font-size: 12px;}
    .sig-purple {background-color: #4a1b4a; color: #d8b4fe; border: 1px solid #a855f7; padding: 2px 6px; border-radius: 4px; font-size: 12px;}
    .sig-cyan {background-color: #083344; color: #22d3ee; border: 1px solid #06b6d4; padding: 2px 6px; border-radius: 4px; font-size: 12px;}
    
    .tactical-box {
        background-color: #1a1a1c;
        padding: 18px 24px;
        border-radius: 8px;
        margin-bottom: 20px;
        border-left: 8px solid; 
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    .tactical-title {
        font-size: 22px;
        font-weight: bold;
        color: #ffffff;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
    }
    .tactical-body {
        font-size: 15px;
        color: #e5e7eb;
        line-height: 1.6;
        background-color: #262730;
        padding: 14px;
        border-radius: 6px;
        border: 1px solid #374151;
    }
</style>
""", unsafe_allow_html=True)

# --- 1. 資料系統 ---
WATCHLIST_FILE, ANCHOR_FILE, TW_NAMES_FILE = "watchlist.json", "anchors.json", "tw_names.json"
DEFAULT_WATCHLISTS = {"清單 A": ['^IXIC', 'QQQ', 'NVDA', 'TSM'], "清單 B": ['MU', 'AAPL', 'TSLA'], "清單 C": ['0050.TW', '6127.TWO'], "清單 D": ['ONDS', 'RXRX'], "清單 E": ['CRCL']}

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
    us_map = {'NVDA': '輝達', 'TSLA': '特斯拉', 'AAPL': '蘋果', 'MU': '美光', 'TSM': '台積電', 'GOOGL': '谷歌'}
    base = ticker.split('.')[0]
    
    if base in us_map and not (".TW" in ticker or ".TWO" in ticker):
        return us_map[base]
        
    if ".TW" in ticker or ".TWO" in ticker:
        local_map = json_load(TW_NAMES_FILE, {})
        bad_words = ["Yahoo", "股市", "走勢", "無符合", "找不到", "代碼或名稱", "html", "TW"]
        keys_to_delete = [k for k, v in local_map.items() if any(bad in v for bad in bad_words)]
        for k in keys_to_delete:
            del local_map[k]
            
        if ticker in local_map: 
            return local_map[ticker]
        
        name = None
        prefix = "otc" if ".TWO" in ticker else "tse"
        try:
            res = requests.get(f"https://mis.twse.com.tw/stock/api/getStockInfo.jsp?ex_ch={prefix}_{base}.tw", timeout=3)
            if res.status_code == 200:
                data = res.json()
                if "msgArray" in data and len(data["msgArray"]) > 0:
                    n = data["msgArray"][0].get("n")
                    if n and n != "--":
                        name = n.strip()
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
                        if "無符合" not in sug:
                            name = sug.replace(base, '').replace('\t', '').strip()
            except: pass
            
        if name and not any(bad in name for bad in bad_words):
            local_map[ticker] = name
            json_save(TW_NAMES_FILE, local_map)
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
    meta = mapping.get(base, {'ceo': [], 'key': [base]})
    meta['name'] = name
    return meta

def validate_news(title, ticker, info, strict=True):
    t = title.lower(); bt = ticker.split('.')[0].lower(); wl = [bt] + [k.lower() for k in info['key']] + [info['name'].lower()] + [c.lower() for c in info['ceo']]
    if any(w in t for w in wl): return True
    if not strict and bt in t and any(k in t for k in ['options', 'volume', 'shares', 'trading', '期權', '成交', '異動']): return True
    return False

def fetch_deep_news(ticker, is_macro=False):
    try:
        hdrs = {'User-Agent': 'Mozilla/5.0'}; items = []; now = datetime.now()
        if is_macro:
            resp = requests.get("https://news.google.com/rss/search?q=聯準會+升息+通膨+鮑爾&hl=zh-TW&gl=TW&ceid=TW:zh-Hant", headers=hdrs, timeout=4)
            if resp.status_code == 200:
                for item in ET.fromstring(resp.content).findall('.//item')[:6]:
                    d = parse_rss(item, "macro", "Global")
                    if (now - d['dt']).days <= 30: items.append(d)
            return items
        
        info = get_ticker_metadata(ticker); cn = info['name']; trg = f"{ticker} OR {cn}" if cn else ticker; small = ticker.split('.')[0] in ['ONDS', 'RXRX', 'CRCL', 'SOUN', 'PLTR'] 
        
        if ".TW" in ticker or ".TWO" in ticker:
            base_tk = ticker.split('.')[0]
            q = f"({base_tk}+OR+{cn})+(重訊+OR+重大訊息+OR+營收+OR+公告+OR+自結+OR+EPS+OR+配息+OR+法說)"
            try:
                resp = requests.get(f"https://news.google.com/rss/search?q={q}&hl=zh-TW&gl=TW&ceid=TW:zh-Hant", headers=hdrs, timeout=4)
                if resp.status_code == 200:
                    for item in ET.fromstring(resp.content).findall('.//item')[:10]:
                        d = parse_rss(item, "tw_local", "台股重訊")
                        if (now - d['dt']).days <= 30 and validate_news(d['title'], ticker, info): items.append(d)
            except: pass
        else:
            q_sec = f"{ticker}+stock+(SEC+Filing+OR+Form+4+OR+Insider)"
            q_news = f"{ticker}+stock+(Options+OR+Volume)" if small else f"{trg}+stock+(財聯社+OR+鉅亨網+OR+營收+OR+財報)"
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
    if any(x in t for x in ['options', 'volume', '期權', '異動']):
        s, tag = (2.0, '<span class="tag-chip">🌊 籌碼異動</span>') if any(x in t for x in ['high', 'surge', '大增']) else (0.5, '<span class="tag-gray">📊 籌碼面</span>')
    elif any(x in t for x in ['營收', 'eps', 'profit', '獲利', '財報', '重訊']):
        if any(x in t for x in ['新高', 'beat', '增', '漲', '超預期']): s, tag, maj = 4.0, '<span class="tag-filing">💎 財報/營運利多</span>', True
        elif any(x in t for x in ['miss', 'down', 'loss', '虧']): s, tag = -3.0, '<span class="tag-red">📉 財報利空</span>'
        else: s, tag = 1.5, '<span class="tag-div">📊 財務數據</span>'
    elif cat == "us_sec" or 'form 4' in t or 'insider' in t:
        if any(x in t for x in ['buy', '買進']): s, tag, maj = 4.0, '<span class="tag-vip">👑 VIP買進</span>', True
        elif any(x in t for x in ['sell', '賣出']):
            if any(p in t for p in ['cfo', '財務長']): s, pen, tag = -4.0, -4.0, '<span class="tag-risk">⚠️ CFO拋售</span>'
            elif any(p in t for p in ['ceo', '執行長']): s, pen, tag = -3.5, -3.5, '<span class="tag-risk">⚠️ CEO拋售</span>'
            else: s, pen, tag = -1.5, -1.5, '<span class="tag-gray">內部人賣出</span>'
    elif any(x in t for x in ['order', 'contract', '訂單']): s, tag = 3.0, '<span class="tag-hard">🔥 實質訂單</span>'
    if not tag: tag = '<span class="tag-hard">📈 利多</span>' if s > 0 else '<span class="tag-red">📉 利空</span>' if s < 0 else ""
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
            elif chg < -0.8: txt, note, col, sc = "市場修正", f"納指下跌 {chg:.2f}%", "sig-orange", -1.5
    except: pass
    return txt, note, col, sc

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
    df['ATR'] = tr.rolling(14).mean(); df['Vol_60D_Avg'] = ((df['ATR'] / df['Close']) * 100).rolling(60).mean(); df['ATR_Trailing_Stop'] = df['High'].rolling(22).max() - df['ATR']*3
    delta = df['Close'].diff(); df['RSI'] = 100 - (100 / (1 + (delta.where(delta>0,0).rolling(14).mean() / -delta.where(delta<0,0).rolling(14).mean())))
    df['MACD'] = df['Close'].ewm(span=12, adjust=False).mean() - df['Close'].ewm(span=26, adjust=False).mean()
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean(); df['MACD_Hist'] = df['MACD'] - df['Signal_Line']
    clv = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low']); df['AD_Line'] = (clv.fillna(0) * df['Volume']).cumsum()
    df['DMA_DDD'] = df['Close'].rolling(10).mean() - df['Close'].rolling(50).mean(); df['DMA_AMA'] = df['DMA_DDD'].rolling(10).mean()
    
    buy_seq = np.zeros(len(df), dtype=int); sell_seq = np.zeros(len(df), dtype=int)
    for i in range(4, len(df)):
        buy_seq[i] = buy_seq[i-1] + 1 if df['Close'].iloc[i] < df['Close'].iloc[i-4] else 0
        sell_seq[i] = sell_seq[i-1] + 1 if df['Close'].iloc[i] > df['Close'].iloc[i-4] else 0
    df['TD_Buy_Seq'] = buy_seq
    df['TD_Sell_Seq'] = sell_seq
    return df

def find_structural_box_bottom(df, current_price):
    if len(df) < 60: return current_price, df.index[0], df.index[-1], False
    p = df.tail(120).copy(); p['M'] = p['Low'].rolling(5, center=True, min_periods=1).min()
    v = p[(p['Low'] == p['M']) & (p['Low'] < current_price * 0.98)]
    if not v.empty:
        idx = np.where(p.index == v.iloc[-1].name)[0][0]
        return v.iloc[-1]['Low'], p.index[max(0, idx-15)], p.index[min(len(p)-1, idx+15)], False
    return p['Low'].min(), p.index[0], p.index[-1], True

# [V17.46 核心] 接收兩個參數：前高支撐 與 前高壓力，打造完整價格通道
def generate_projection_points(df, trend_text, cur_p, iron_p, is_brk, ph_support, ph_resist):
    last_d = df.index[-1]; f_d = []
    d = last_d
    while len(f_d) < 30:
        d += pd.Timedelta(days=1)
        if d.weekday() < 5: f_d.append(d)
    
    x = [last_d]; y = [cur_p]
    ma20 = df['SMA_20'].iloc[-1] if not pd.isna(df['SMA_20'].iloc[-1]) else cur_p
    rsi = df['RSI'].iloc[-1] if not pd.isna(df['RSI'].iloc[-1]) else 50
    td_sell_cur = df['TD_Sell_Seq'].iloc[-1] 
    recent_high_20 = df['High'].tail(20).max()
    is_relief_rally = (rsi < 60 and trend_text == "🐻 熊市") or (rsi > 70) 

    # 將 None 轉換為極值以便於計算
    support_level = ph_support if ph_support is not None else iron_p
    resist_level = ph_resist if ph_resist is not None else float('inf')

    scenario_name = ""

    if "牛市" in trend_text:
        scenario_name = "🐂 牛市 N 字突破 (支撐不破)"
        # 回測絕對不破下方最近的前高支撐
        dip = max(cur_p * 0.96, ma20, support_level)
        # 上攻挑戰上方壓力 (若無壓力則挑戰近期新高)
        rally = min(cur_p * 1.05, resist_level * 0.99) if resist_level != float('inf') else max(cur_p * 1.05, recent_high_20 * 1.02)
        x.extend([f_d[4], f_d[14]]); y.extend([dip, rally])
        
    elif "熊市" in trend_text:
        if is_brk:
            scenario_name = "🕳️ 熊市無底洞墜落"
            bounce = min(cur_p * 1.03, ma20, resist_level)
            drop = cur_p * 0.92
            x.extend([f_d[4], f_d[14]]); y.extend([bounce, drop])
        elif is_relief_rally:
            days_to_peak = (9 - td_sell_cur) if (0 < td_sell_cur < 9) else 3
            days_to_peak = max(1, days_to_peak)
            scenario_name = f"🐻 熊市打底：TD({td_sell_cur}➔9)動能延續 ➔ 遇壓測底" if (0 < td_sell_cur < 9) else "🐻 熊市打底：均線遇壓 ➔ 二次測底 (W底)"
            
            exhaustion_p = min(max(cur_p * 1.01, ma20), resist_level * 0.99)
            w_dip = max(iron_p, iron_p * 1.015, support_level, cur_p * 0.93) if iron_p > 0 else cur_p * 0.9
            breakout_p = max(exhaustion_p * 1.05, recent_high_20 * 1.02, cur_p * 1.1)
            
            x.extend([f_d[days_to_peak], f_d[days_to_peak + 5], f_d[days_to_peak + 15]])
            y.extend([exhaustion_p, w_dip, breakout_p])
        else:
            scenario_name = "🐻 熊市下降通道"
            bounce = min(cur_p * 1.06, ma20, resist_level)
            drop = max(iron_p, cur_p * 0.9)
            x.extend([f_d[4], f_d[14]]); y.extend([bounce, drop])
    else:
        if ph_support and cur_p > ph_support:
            scenario_name = "⚖️ 區間突破 ➔ 回測前高支撐"
            up_1 = min(cur_p * 1.04, resist_level * 0.99)
            dip = max(cur_p * 0.96, support_level) # 精準踩在前高支撐上
            up_2 = max(up_1, dip * 1.05)
            x.extend([f_d[5], f_d[12], f_d[18]]); y.extend([up_1, dip, up_2])
        else:
            scenario_name = "⚖️ 區間震盪 ➔ 挑戰前高壓力"
            up_1 = min(cur_p * 1.04, resist_level * 0.99) # 撞擊前高壓力
            dip = max(cur_p * 0.96, iron_p, support_level)
            up_2 = resist_level * 1.01 if resist_level != float('inf') else cur_p * 1.05
            x.extend([f_d[5], f_d[12], f_d[18]]); y.extend([up_1, dip, up_2])
            
    return x, y, scenario_name

def analyze_market_trend(df):
    c, m20, m60 = df['Close'].iloc[-1], df['SMA_20'].iloc[-1], df['SMA_60'].iloc[-1]
    if c > m20 > m60: return "🐂 牛市", "多頭排列", "sig-green"
    elif c < m20 < m60: return "🐻 熊市", "空頭排列", "sig-red"
    else: return "⚖️ 震盪", "區間整理", "sig-orange"

def get_stock_engine_mode(ticker, df_data):
    if ticker.startswith("^") or any(e in ticker for e in ["QQQ", "SPY", "DIA", "0050.TW"]): return "🏢 權值大盤", "trend"
    try:
        m = yf.Ticker(ticker).info.get('marketCap', 0); v = df_data['Vol_60D_Avg'].iloc[-1] if not df_data.empty else 3.0
        l = m >= (300e9 if ".TW" in ticker else 10e9) or (m==0 and v<3.5)
        if l and v < 4.0: return "🏢 權值穩健", "trend"
        elif l: return "🚀 巨型動能", "momentum"
        else: return "🎢 妖股轉折", "reversal"
    except: return "🎢 動態模式", "reversal"

def get_relative_strength(ticker, stock_df):
    try:
        b = yf.download("^TWII" if ".TW" in ticker else "^GSPC", period="1mo", progress=False)['Close'].iloc[:, 0]
        a = stock_df['Close'].reindex(b.index, method='ffill')
        if len(b) > 20:
            diff = ((a.iloc[-1] - a.iloc[-20])/a.iloc[-20]) - ((b.iloc[-1] - b.iloc[-20])/b.iloc[-20])
            if diff > 0.05: return "🦁 領頭羊 (強)", "sig-green"
            elif diff > 0: return "🐯 優於大盤", "sig-blue"
            else: return "🐶 落後股 (弱)", "sig-gray"
    except: pass
    return "⚖️ 跟隨大盤", "sig-gray"

def detect_smart_money_status(df):
    if len(df) < 10: return None
    latest = df.iloc[-1]
    price_now, price_5d = latest['Close'], df['Close'].iloc[-6]; ad_now, ad_5d = latest['AD_Line'], df['AD_Line'].iloc[-6]; rsi = latest['RSI']
    
    if price_now > latest['Open'] and latest['Low'] < latest['Bollinger_Lower'] and rsi < 40 and price_now > price_5d * 0.95: return "💎 破底翻買點"
    if latest['Close'] < latest['Bollinger_Lower'] and latest['RSI'] < 30: return "⚡ 乖離抄底 (超賣)"
    if price_now < price_5d * 0.98 and ad_now > ad_5d and rsi < 50: return "🎯 主力背離吸籌"
    if rsi > 65 and latest['Volume'] > latest['Vol_SMA5'] * 1.3 and (latest['Close'] < latest['Open'] or (latest['High'] - max(latest['Open'], latest['Close']) > abs(latest['Close'] - latest['Open']) * 1.5)): return "🔴 主力調節 (爆量滯漲)"
    if rsi < 30 and latest['Volume'] > latest['Vol_SMA5']: return "⚡ 恐慌殺盤"
    return None

def get_tactical_advice(df, cur_p, t_s, iron_p):
    if len(df) < 10: return "⌛ 數據不足", "等待更多 K 線資料寫入...", "#9ca3af"
    latest = df.iloc[-1]
    td_b = latest.get('TD_Buy_Seq', 0)
    td_s = latest.get('TD_Sell_Seq', 0)
    ma20 = latest.get('SMA_20', cur_p)
    rsi = latest.get('RSI', 50)
    
    if td_s >= 8 or rsi > 75:
        return "⚠️ 多頭力竭預警 (嚴禁追高)", f"上漲動能已達極限 (目前 TD {int(td_s)} 或是 RSI過熱)。隨時面臨獲利了結賣壓，強烈建議【嚴禁追高】，持有多單者請準備【分批出場】！", "#ef4444"
    
    if 1 <= td_s <= 7 and cur_p > ma20:
        return "🚀 主升歡樂帶 (抱緊處理)", f"目前上漲動能強勁 (TD {int(td_s)})，且穩居月線之上。正處於上升通道，請【沿著均線續抱】，享受獲利，不輕易下車！", "#22c55e"

    if td_b >= 8 or rsi < 30:
        return "💎 空頭力竭 (準備破底翻)", f"殺盤動能即將耗盡 (目前 TD 下跌 {int(td_b)} 或是 RSI超賣)，股價已進入絕佳的潛在支撐區。請密切關注反轉向上的【右側買點】！", "#facc15"
        
    if 1 <= td_b <= 7 and cur_p < ma20:
        return "🔪 恐慌殺盤中 (嚴禁接刀)", f"目前處於主跌段 (TD 下跌 {int(td_b)})，賣壓極度沉重。在底部紅字 8 或 9 出現之前，請綁好雙手【絕對不要進場買進】！", "#f97316"
        
    if cur_p > ma20:
        return "📈 多頭延續 (沿均線續抱)", "目前股價穩居月線 (MA20) 之上，多頭格局不變。建議沿均線續抱。", "#22c55e"
    else:
        return "⚖️ 震盪整理 (耐心觀望)", "目前股價處於區間震盪，方向尚未明確。建議耐心觀望，等待突破或測底訊號。", "#9ca3af"

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

def predict_target_and_rating(df):
    price, upper, recent_high_60 = df['Close'].iloc[-1], df['Bollinger_Upper'].iloc[-1], df['High'].tail(60).max()
    t_s = upper if price >= recent_high_60 else min(upper, recent_high_60)
    return t_s, max(recent_high_60 * 1.15, t_s * 1.1), "強勢" if price > df['SMA_20'].iloc[-1] else "持有"

def format_volume(num): return f"{num/1e9:.2f}B" if num >= 1e9 else f"{num/1e6:.2f}M" if num >= 1e6 else f"{num}"

# --- 6. 主介面 Sidebar ---
with st.sidebar:
    st.title("🎛️ 控制台")
    
    st.header("📱 平台顯示設定")
    mobile_mode = st.toggle("啟用手機防卡死模式", value=False, help="手機瀏覽網頁時請開啟此選項，鎖定 K 線圖滑動以防卡死。電腦版請保持關閉以獲得完整操作體驗。")
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
                disp_base = f"{s_name} ({t})" if s_name != t else t
                
                if st.button(f"{'👉 ' if is_sel else ''}{disp_base}", key=f"btn_{wl_name}_{t}", type=btn_t, use_container_width=True):
                    st.session_state.current_ticker = t; st.session_state.active_list = wl_name; st.session_state.user_opened_list = wl_name; st.rerun()

    st.markdown("---")
    st.markdown("<span style='color:gray; font-size:13px;'>排列目前代碼</span>", unsafe_allow_html=True)
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
st.title(f"📈 {disp_main_title} 實戰戰情室 V17.46")

api_p, api_i = ("5d", "15m") if "當沖" in time_opt else ("6mo", "1d") if "日" in time_opt else ("2y", "1wk")
df = yf.download(cur_t, period=api_p, interval=api_i, progress=False)
if df.empty: st.error("無數據"); st.stop()
if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
df = df.loc[:, ~df.columns.duplicated()]

df.index = pd.to_datetime(df.index)
if df.index.tz is not None:
    df.index = df.index.tz_localize(None)

df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=['Open', 'High', 'Low', 'Close'])
if len(df) < 2: 
    st.error("數據連線異常或資料量不足以進行 AI 分析，請稍後再試。")
    st.stop()

df = calculate_indicators(df)

latest = df.iloc[-1]; close_v = float(latest['Close']); chg = (close_v - float(df.iloc[-2]['Close'])) / float(df.iloc[-2]['Close']) * 100
clr = "green" if chg >= 0 else "red"
sigs = analyze_strategic_signals(df)
trend_txt, trend_note, trend_col = analyze_market_trend(df)
rs_txt, rs_col = get_relative_strength(cur_t, df)
engine_label, engine_type = get_stock_engine_mode(cur_t, df)
t_s, t_l, rating = predict_target_and_rating(df)

vp_60 = calculate_volume_profile(df.tail(60), bins=40)
vol_poc = vp_60.loc[vp_60['Volume'].idxmax(), 'Price'] if not vp_60.empty else close_v
iron_price, box_start, box_end, is_breaking = find_structural_box_bottom(df, close_v)

# --- 戰術建議盒 ---
tac_title, tac_body, tac_color = get_tactical_advice(df, close_v, t_s, iron_price)
st.markdown(f"""
<div class="tactical-box" style="border-left-color: {tac_color};">
    <div class="tactical-title">🚩 戰術建議： <span style="color:{tac_color}; margin-left: 8px;">{tac_title}</span></div>
    <div class="tactical-body">💡 <b>行動指南：</b> {tac_body}</div>
</div>
""", unsafe_allow_html=True)

# --- 主卡片與三大戰情方塊 ---
st.markdown(f"""
<div class="price-card">
    <h1 style="margin:0; font-size: 50px;">${close_v:.2f}</h1>
    <h3 style="margin:0; color: {clr};">{chg:+.2f}%</h3>
    <p style="color: gray;">量: {format_volume(latest['Volume'])}</p>
    <div class="engine-tag">⚙️ {engine_label}</div>
</div>
""", unsafe_allow_html=True)

r1, r2, r3 = st.columns(3)
with r1: st.markdown(f'<div class="ai-box"><h5 style="color:white; margin:0; margin-bottom:5px;">📡 綜合戰略</h5><div style="font-size:16px;" class="{sigs["Summary_Color"]}">{sigs["Summary"]}</div><div style="margin-top:5px; font-size:14px;">MACD: <span class="{sigs["MACD_Color"]}">{sigs["MACD_Text"]}</span><br>RSI: <span class="{sigs["RSI_Color"]}">{sigs["RSI_Text"]}</span></div></div>', unsafe_allow_html=True)
with r2: st.markdown(f'<div class="ai-box"><h5 style="color:white; margin:0;">⚖️ 雙重格局</h5><div style="margin-top:5px; font-size:14px;">🏢 個股: <span class="{trend_col}">{trend_txt} ({trend_note})</span></div></div>', unsafe_allow_html=True)
with r3: st.markdown(f'<div class="ai-box" style="border: 1px solid #00d4ff;"><h5 style="color:white; margin:0;">🎯 AI 目標 & 強弱</h5><div style="margin-top:5px; font-size:14px;">短: ${t_s:.2f} | 長: ${t_l:.2f}<br><span class="{rs_col}">{rs_txt}</span></div></div>', unsafe_allow_html=True)

# ==========================================
# 繪圖區 (嚴格圖層 Z-Index 管理)
# ==========================================
p_data = df.tail(120) if "日" in time_opt else df.tail(60)
fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_width=[0.2, 0.2, 0.6])

# [第 1 層] 繪製基礎 K 線與 ATR 停損線
fig.add_trace(go.Candlestick(x=p_data.index, open=p_data['Open'], high=p_data['High'], low=p_data['Low'], close=p_data['Close'], name="K線"), row=1, col=1)
fig.add_trace(go.Scatter(x=p_data.index, y=p_data['ATR_Trailing_Stop'], mode='lines', line=dict(color='#FF5F1F', width=1.5, dash='dot'), name='ATR 停損'), row=1, col=1)

# [第 2 層] 繪製籌碼背景區塊
zone_top = vol_poc * 1.025
zone_bottom = vol_poc * 0.975
fig.add_hrect(y0=zone_bottom, y1=zone_top, line_width=0, fillcolor="rgba(100, 100, 100, 0.2)", layer="below", row=1, col=1)

last_d = p_data.index[-1]
for r in range(1, 4): fig.add_vline(x=last_d, line_dash="dash", line_color="#666", opacity=0.7, layer="below", row=r, col=1)

# [第 3 層] 跑所有 K 線上的訊號標籤
for i in range(5, len(p_data)):
    curr, prior = p_data.iloc[i], p_data.iloc[i-1]
    
    td_b = curr['TD_Buy_Seq']
    if 0 < td_b <= 9:
        fig.add_annotation(x=p_data.index[i], y=curr['Low'], text=str(int(td_b)), showarrow=False, yshift=-12, font=dict(color='#ff6b6b', size=9 if td_b<9 else 14, weight="normal" if td_b<9 else "bold"), row=1, col=1)
    
    td_s = curr['TD_Sell_Seq']
    if 0 < td_s <= 9:
        fig.add_annotation(x=p_data.index[i], y=curr['High'], text=str(int(td_s)), showarrow=False, yshift=12, font=dict(color='#4a9eff', size=9 if td_s<9 else 14, weight="normal" if td_s<9 else "bold"), row=1, col=1)

    if prior['Close'] < prior['Open'] and curr['Close'] > curr['Open'] and curr['Open'] <= prior['Close'] and curr['Close'] >= prior['Open']:
        fig.add_annotation(x=p_data.index[i], y=curr['Low'], text="🕯️吞噬", showarrow=True, arrowhead=1, ax=0, ay=30, row=1, col=1, font=dict(color="orange", size=9))
    
    status = detect_smart_money_status(p_data.iloc[:i+1])
    if status:
        if "吸籌" in status: 
            fig.add_annotation(x=p_data.index[i], y=curr['Low'], text=f"🐳吸<br>${curr['Low']:.1f}", showarrow=True, arrowhead=1, ax=0, ay=35, row=1, col=1, bgcolor="rgba(111, 66, 193, 0.8)", font=dict(color="white", size=9))
        elif "破底翻" in status or "抄底" in status: 
            fig.add_annotation(x=p_data.index[i], y=curr['Low'], text=f"{status}<br>${curr['Low']:.1f}", showarrow=True, arrowhead=1, ax=0, ay=45, row=1, col=1, bgcolor="rgba(147, 51, 234, 0.8)", font=dict(color="white", size=10, weight="bold"))
        elif "調節" in status: 
            fig.add_annotation(x=p_data.index[i], y=curr['High'], text=f"🔴調節<br>${curr['High']:.1f}", showarrow=True, arrowhead=1, ax=0, ay=-40, row=1, col=1, bgcolor="rgba(185, 28, 28, 0.8)", font=dict(color="white", size=10, weight="bold"))

    macd_buy = (curr['MACD'] > curr['Signal_Line']) and (prior['MACD'] <= prior['Signal_Line'])
    macd_sell = (curr['MACD'] < curr['Signal_Line']) and (prior['MACD'] >= prior['Signal_Line'])
    if macd_buy and ((engine_type == "trend" and curr['Close'] < curr.get('SMA_60', 0)) or (engine_type == "momentum" and curr['Close'] < curr.get('SMA_20', 0))): macd_buy = False
    
    if macd_buy: 
        fig.add_annotation(x=p_data.index[i], y=curr['Low'], text=f"BUY<br>${curr['Close']:.1f}", showarrow=True, arrowhead=1, ax=0, ay=25, row=1, col=1, bgcolor="rgba(40, 167, 69, 0.8)", font=dict(color="white", size=9))
    if macd_sell: 
        fig.add_annotation(x=p_data.index[i], y=curr['High'], text=f"SELL<br>${curr['Close']:.1f}", showarrow=True, arrowhead=1, ax=0, ay=-25, row=1, col=1, bgcolor="rgba(220, 53, 69, 0.8)", font=dict(color="white", size=9))

    if (curr['High'] >= t_s or curr['RSI'] > 75) and not (prior['High'] >= t_s or prior['RSI'] > 75):
        fig.add_annotation(x=p_data.index[i], y=curr['High'], text=f"💰達標<br>${curr['Close']:.1f}" if curr['High'] >= t_s else f"🔥過熱<br>${curr['Close']:.1f}", showarrow=True, arrowhead=1, ax=0, ay=-45, row=1, col=1, bgcolor="rgba(255, 193, 7, 0.8)" if curr['High'] >= t_s else "rgba(255, 69, 0, 0.8)", font=dict(color="black" if curr['High'] >= t_s else "white", size=9))

# [V17.46 核心升級] 雙軌雷達：同時抓出上方壓力與下方支撐
abs_high = p_data['High'].max()
local_maxes = p_data['High'][(p_data['High'] == p_data['High'].rolling(9, center=True).max())].dropna()

# 過濾掉絕對高點
filtered_maxes = local_maxes[local_maxes < abs_high]

resist_line = None
support_line = None

if not filtered_maxes.empty:
    # 找尋大於現價的最近一個前高 (天花板/壓力)
    above_current = filtered_maxes[filtered_maxes > close_v]
    if not above_current.empty:
        resist_line = above_current.iloc[-1]

    # 找尋小於現價的最近一個前高 (地板/支撐)
    below_current = filtered_maxes[filtered_maxes < close_v]
    if not below_current.empty:
        support_line = below_current.iloc[-1]

# [第 4 層] 繪製預測折線 (餵入雙軌支撐壓力)
px, py, sc_name = generate_projection_points(df, trend_txt, close_v, iron_price, is_breaking, support_line, resist_line)
fig.add_trace(go.Scatter(x=px, y=py, mode='lines+markers', line=dict(color='#eab308', width=3, dash='dash'), marker=dict(size=8, symbol='diamond', color='#eab308'), name='🔮 AI 劇本推演'), row=1, col=1)
for i in range(1, len(px)): fig.add_annotation(x=px[i], y=py[i], text=f"${py[i]:.2f}", showarrow=True, arrowhead=0, ay=-20, font=dict(color="#eab308", size=11), bgcolor="rgba(0,0,0,0.6)", row=1, col=1)

# [第 5 層 - 終極霸體] 全景價格通道與 UI 標題
fig.add_hline(y=abs_high, line_dash="dot", line_color="#ef4444", annotation_text=f"🔴 波段最高<br>${abs_high:.2f}", annotation_font_color="#ef4444", annotation_position="top right", annotation_align="right", opacity=1.0, layer="above", row=1, col=1)

# 繪製天花板 (前高壓力)
if resist_line:
    fig.add_hline(y=resist_line, line_dash="dot", line_color="#f97316", annotation_text=f"🟠 前高壓力<br>${resist_line:.2f}", annotation_font_color="#f97316", annotation_position="top right", annotation_align="right", opacity=1.0, layer="above", row=1, col=1)

# 繪製地板 (前高支撐)
if support_line:
    fig.add_hline(y=support_line, line_dash="dot", line_color="#3b82f6", annotation_text=f"🔵 前高支撐<br>${support_line:.2f}", annotation_font_color="#3b82f6", annotation_position="bottom right", annotation_align="right", opacity=1.0, layer="above", row=1, col=1)

if not is_breaking and iron_price > 0:
    fig.add_hline(y=iron_price, line_dash="dash", line_color="#20c997", annotation_text=f"🧱 鐵板 ${iron_price:.2f}", annotation_font_color="#20c997", annotation_position="bottom right", opacity=1.0, layer="above", row=1, col=1)

# 絕對座標標題
fig.add_annotation(x=0.01, y=0.98, xref="paper", yref="paper", text=f"🔮 目前 AI 推演：{sc_name}", showarrow=False, font=dict(color="white", size=14, weight="bold"), bgcolor="rgba(0, 0, 0, 0.9)", bordercolor="#eab308", borderwidth=1, borderpad=6, xanchor="left", yanchor="top")
fig.add_annotation(x=0.5, y=0.98, xref="paper", yref="paper", text=f"🛡️ 籌碼攻防帶 (突破: ${zone_top:.2f} | 跌破: ${zone_bottom:.2f})", showarrow=False, font=dict(color="#d8b4fe", size=12, weight="bold"), bgcolor="rgba(10, 10, 10, 0.9)", borderpad=4, xanchor="center", yanchor="top")

# 今日收盤點
fig.add_trace(go.Scatter(x=[last_d], y=[close_v], mode='markers', marker=dict(size=12, color='#00ffff', line=dict(color='white', width=2)), name="今日收盤"), row=1, col=1)

# --- 副圖 MACD ---
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

# --- 雙籌碼分析圖 ---
try:
    c1, c2 = st.columns(2); mf = ((p_data['Close'] - p_data['Open']) / (p_data['High'] - p_data['Low'])) * p_data['Volume']; mf = mf.fillna(0).cumsum()
    with c1:
        st.caption("主力資金流 (Money Flow)")
        fig_mf = go.Figure(go.Scatter(x=p_data.index, y=mf, fill='tozeroy', line=dict(color='#00d4ff')))
        if len(mf) > 5:
            trend = mf.iloc[-1] - mf.iloc[-5]
            if trend > 0: fig_mf.add_annotation(x=p_data.index[-1], y=mf.iloc[-1], text="🟢 主力吸籌", showarrow=True, arrowhead=1, font=dict(color="#4ade80", size=12), bgcolor="#1b3a1b")
            else: fig_mf.add_annotation(x=p_data.index[-1], y=mf.iloc[-1], text="🔴 主力出貨", showarrow=True, arrowhead=1, font=dict(color="#ff6b6b", size=12), bgcolor="#3a1b1b")
        fig_mf.update_layout(dragmode=False if mobile_mode else 'zoom', height=250, template="plotly_dark", margin=dict(t=10, b=10, l=10, r=10)); st.plotly_chart(fig_mf, use_container_width=True, config={'displayModeBar': False})
    with c2:
        st.caption("籌碼分佈 (主力 vs 散戶)")
        inst_mask = (p_data['Close'] > p_data['Open']) & (p_data['Volume'] > p_data['Vol_SMA5'])
        vp_all = calculate_volume_profile(p_data); vp_main = calculate_volume_profile(p_data, filter_mask=inst_mask)
        fig_vp = go.Figure()
        fig_vp.add_trace(go.Scatter(x=vp_all['Price'], y=vp_all['Volume'], fill='tozeroy', line=dict(color='#ffaa00', width=0), name='整體'))
        fig_vp.add_trace(go.Scatter(x=vp_main['Price'], y=vp_main['Volume'], fill='tozeroy', line=dict(color='#00d4ff', width=2), name='主力'))
        fig_vp.add_vline(x=close_v, line_dash="dash", line_color="white", annotation_text="現價")
        fig_vp.update_layout(dragmode=False if mobile_mode else 'zoom', height=250, template="plotly_dark", margin=dict(t=10, b=10, l=10, r=10), showlegend=True, legend=dict(orientation="h", y=1.1)); st.plotly_chart(fig_vp, use_container_width=True, config={'displayModeBar': False})
except: pass
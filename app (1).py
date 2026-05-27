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

# [V17.66] 美股情緒雙核引擎
try:
    from sentiment_engine import render_us_sentiment_dashboard
    _SENTIMENT_AVAILABLE = True
except ImportError:
    _SENTIMENT_AVAILABLE = False

# [V17.67] 台股空頭危機距離指數引擎
try:
    from tw_sentiment_engine import render_tw_crisis_dashboard
    _TW_SENTIMENT_AVAILABLE = True
except ImportError:
    _TW_SENTIMENT_AVAILABLE = False

# --- 0. 系統設定 ---
st.set_page_config(page_title="AI 實戰戰情室 V17.67 (雙市場情緒版)", layout="wide", page_icon="🛡️")

# --- CSS 美化 ---
st.markdown("""
<style>
    .price-card {background-color: #1e1e1e; padding: 20px; border-radius: 10px; text-align: center; border: 1px solid #333; margin-bottom: 10px;}
    .ai-box {background-color: #333; padding: 15px; border-radius: 10px; border: 1px solid #555; text-align: center; height: 100%;}
    .macro-alert {background-color: #3a1b1b; color: #ff6b6b; padding: 10px; border-radius: 5px; border: 1px solid #dc3545; margin-bottom: 10px; font-weight: bold;}
    .anchor-box {background-color: #1b3a4a; color: #00d4ff; padding: 12px; border-radius: 5px; border: 1px solid #00d4ff; margin-bottom: 10px; font-size: 13px; text-align: left;}
    .anchor-title-cn {color: #fff; font-weight: bold; font-size: 14px; margin-bottom: 4px;}
    .anchor-title-en {color: #aaa; font-size: 11px; font-style: italic;}
    .earnings-tag {background-color: #2c2c2e; padding: 5px 12px; border-radius: 15px; font-size: 13px; margin-top: 10px; border: 1px solid #555; display: inline-block; margin-right: 8px;}
    .engine-tag {background-color: #1e3a8a; color: #38bdf8; padding: 5px 12px; border-radius: 15px; font-size: 13px; margin-top: 10px; border: 1px solid #38bdf8; display: inline-block; font-weight: bold;}
    .earn-beat {color: #4ade80; font-weight: bold;}
    .earn-miss {color: #ff6b6b; font-weight: bold;}
    .earn-warn {color: #ffaa00; font-weight: bold;}
    .earn-turn {color: #facc15; font-weight: bold;}
    
    .sig-green {background-color: #1b3a1b; color: #4ade80; border: 1px solid #28a745; padding: 3px 8px; border-radius: 6px; font-size: 13px; display: inline-block; line-height: 1.4;}
    .sig-red {background-color: #3a1b1b; color: #ff6b6b; border: 1px solid #dc3545; padding: 3px 8px; border-radius: 6px; font-size: 13px; display: inline-block; line-height: 1.4;}
    .sig-gray {background-color: #333; color: #ccc; border: 1px solid #666; padding: 3px 8px; border-radius: 6px; font-size: 13px; display: inline-block; line-height: 1.4;}
    .sig-orange {background-color: #4a3b1b; color: #ffaa00; border: 1px solid #ffaa00; padding: 3px 8px; border-radius: 6px; font-size: 13px; display: inline-block; line-height: 1.4;}
    .sig-blue {background-color: #1b3a4a; color: #4a9eff; border: 1px solid #00d4ff; padding: 3px 8px; border-radius: 6px; font-size: 13px; display: inline-block; line-height: 1.4;}
    .sig-purple {background-color: #4a1b4a; color: #d8b4fe; border: 1px solid #a855f7; padding: 3px 8px; border-radius: 6px; font-size: 13px; display: inline-block; line-height: 1.4;}
    .sig-cyan {background-color: #083344; color: #22d3ee; border: 1px solid #06b6d4; padding: 3px 8px; border-radius: 6px; font-size: 13px; display: inline-block; line-height: 1.4;}
    
    .tactical-box {background-color: #1a1a1c; padding: 18px 24px; border-radius: 8px; margin-bottom: 20px; border-left: 8px solid; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);}
    .tactical-title {font-size: 22px; font-weight: bold; color: #ffffff; margin-bottom: 12px; display: flex; align-items: center;}
    .tactical-body {font-size: 15px; color: #e5e7eb; line-height: 1.6; background-color: #262730; padding: 14px; border-radius: 6px; border: 1px solid #374151;}
</style>
""", unsafe_allow_html=True)

# --- 1. 資料系統 ---
WATCHLIST_FILE, ANCHOR_FILE, TW_NAMES_FILE = "watchlist.json", "anchors.json", "tw_names.json"

# [V17.65] 預設清單更新，加入清單 E 與您的專屬股票
DEFAULT_WATCHLISTS = {
    "清單 A": ['^IXIC', 'QQQ', 'NVDA', 'TSM'], 
    "清單 B": ['MU', 'AAPL', 'TSLA'], 
    "清單 C": ['0050.TW', '6127.TWO'],
    "清單 D": ['ONDS', 'RXRX', 'CRCL'],
    "清單 E": ['00878.TW', '2324.TW', '8215.TW', '00403A.TW', '4958.TW', '2344.TW', '2327.TW', '1815.TWO', '5347.TWO']
}

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

def load_watchlists(): return json_load(WATCHLIST_FILE, DEFAULT_WATCHLISTS)
def save_watchlists(data): json_save(WATCHLIST_FILE, data); st.session_state.watchlists = data
def load_anchors(): return json_load(ANCHOR_FILE, {})
def save_anchor_data(data): json_save(ANCHOR_FILE, data)

# [V17.64 延續] 終極台股正名引擎 (自動清除英文快取)
def get_stock_name(ticker):
    us_map = {'NVDA': '輝達', 'TSLA': '特斯拉', 'AAPL': '蘋果', 'MU': '美光', 'TSM': '台積電', 'GOOGL': '谷歌'}
    base = ticker.split('.')[0]
    if base in us_map and not (".TW" in ticker or ".TWO" in ticker): return us_map[base]
    
    if ".TW" in ticker or ".TWO" in ticker:
        local_map = json_load(TW_NAMES_FILE, {})
        bad_words = ["Yahoo", "股市", "走勢", "無符合", "找不到", "代碼或名稱", "html", "TW", "TWO", "INC", "CORP", "LTD", "COMPANY"]
        
        # 🧹 自動淨化協議：清除包含壞字，或是「完全沒有中文且長度大於6」的錯誤快取
        keys_to_delete = [k for k, v in local_map.items() if any(bad in v.upper() for bad in bad_words) or (not re.search(r'[\u4e00-\u9fff]', v) and len(v) > 6)]
        for k in keys_to_delete: 
            del local_map[k]
        
        if ticker in local_map: return local_map[ticker]
        
        name = None
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        
        # 1. 證交所 API (最準確的中文名，上市櫃代碼皆可查)
        try:
            res = requests.get(f"https://www.twse.com.tw/zh/api/codeQuery?query={base}", headers=headers, timeout=3)
            if res.status_code == 200:
                data = res.json()
                if "suggestions" in data and len(data["suggestions"]) > 0:
                    sug = data["suggestions"][0]
                    if "無符合" not in sug:
                        n = sug.replace(base, '').replace('\t', '').strip()
                        if re.search(r'[\u4e00-\u9fff]', n): # 嚴格檢查：必須包含中文字
                            name = n
        except: pass

        # 2. 鉅亨網 API (備用)
        if not name:
            try:
                cnyes_prefix = "OTC" if ".TWO" in ticker else "TWS"
                res = requests.get(f"https://ws.api.cnyes.com/ws/api/v1/quote/quotes/{cnyes_prefix}:{base}:STOCK", headers=headers, timeout=3)
                if res.status_code == 200:
                    data = res.json()
                    if "data" in data and len(data["data"]) > 0:
                        n = data["data"][0].get("name")
                        if n and re.search(r'[\u4e00-\u9fff]', n): name = n.strip()
            except: pass

        # 3. Yahoo Search API (最後備用)
        if not name:
            try:
                res = requests.get(f"https://query2.finance.yahoo.com/v1/finance/search?q={ticker}", headers=headers, timeout=3)
                if res.status_code == 200:
                    quotes = res.json().get('quotes', [])
                    if quotes:
                        n = quotes[0].get('shortname') or quotes[0].get('longname')
                        if n and not any(bad in n.upper() for bad in bad_words): name = n.strip()
            except: pass

        if name and not any(bad in name.upper() for bad in bad_words):
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
        if wl: st.session_state.current_ticker = wl[0]; st.session_state.active_list = wl_name; st.session_state.user_opened_list = wl_name; break

# --- 2. 核心搜尋與新聞引擎 ---
def fetch_deep_news(ticker, is_macro=False): return [] # 模擬爬蟲以節省資源
def get_realtime_macro(): return "宏觀穩健", "多頭支撐", "sig-green", 0
def get_valid_anchor(ticker): return None
def analyze_news_strict(title, cat): return 0, "", False, 0

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
    df['TD_Buy_Seq'] = buy_seq; df['TD_Sell_Seq'] = sell_seq
    return df

def find_structural_box_bottom(df, current_price):
    if len(df) < 60: return current_price, df.index[0], df.index[-1], False
    p = df.tail(120).copy(); p['M'] = p['Low'].rolling(5, center=True, min_periods=1).min()
    v = p[(p['Low'] == p['M']) & (p['Low'] < current_price * 0.98)]
    if not v.empty: return v.iloc[-1]['Low'], p.index[max(0, np.where(p.index == v.iloc[-1].name)[0][0]-15)], p.index[min(len(p)-1, np.where(p.index == v.iloc[-1].name)[0][0]+15)], False
    return p['Low'].min(), p.index[0], p.index[-1], True

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

    support_level = ph_support if ph_support is not None else iron_p
    resist_level = ph_resist if ph_resist is not None else float('inf')

    range_percent = (resist_level - support_level) / cur_p * 100 if resist_level != float('inf') and support_level > 0 else 100.0
    is_squeezed = True if range_percent < 5.0 else False
    scenario_name = ""

    if is_squeezed:
        if cur_p > ma20 or "牛市" in trend_text:
            scenario_name = f"🚀 極限壓縮<br>N字噴發"
            up_1 = max(cur_p * 1.03, resist_level * 1.025); dip = max(cur_p * 1.01, resist_level); up_2 = max(up_1 * 1.05, recent_high_20 * 1.03)   
            x.extend([f_d[4], f_d[9], f_d[18]]); y.extend([up_1, dip, up_2])
        else:
            scenario_name = f"💥 極限壓縮<br>倒N崩盤"
            down_1 = min(cur_p * 0.97, support_level * 0.975); bounce = min(cur_p * 0.99, support_level); down_2 = max(iron_p, down_1 * 0.92)               
            x.extend([f_d[4], f_d[9], f_d[18]]); y.extend([down_1, bounce, down_2])
    else:
        if "牛市" in trend_text:
            scenario_name = "🐂 牛市格局<br>N字突破"
            dip = max(cur_p * 0.96, ma20, support_level); rally = min(cur_p * 1.05, resist_level * 0.99) if resist_level != float('inf') else max(cur_p * 1.05, recent_high_20 * 1.02)
            x.extend([f_d[4], f_d[14]]); y.extend([dip, rally])
        elif "熊市" in trend_text:
            if is_brk:
                scenario_name = "🕳️ 熊市格局<br>無底洞墜落"
                bounce = min(cur_p * 1.03, ma20, resist_level); drop = cur_p * 0.92
                x.extend([f_d[4], f_d[14]]); y.extend([bounce, drop])
            elif is_relief_rally:
                days_to_peak = max(1, (9 - td_sell_cur) if (0 < td_sell_cur < 9) else 3)
                scenario_name = f"🐻 熊市打底<br>遇壓測底" if (0 < td_sell_cur < 9) else "🐻 熊市打底<br>二次測底"
                exhaustion_p = min(max(cur_p * 1.01, ma20), resist_level * 0.99 if resist_level != float('inf') else float('inf'))
                w_dip = max(iron_p, iron_p * 1.015, support_level, cur_p * 0.93) if iron_p > 0 else cur_p * 0.9
                breakout_p = max(exhaustion_p * 1.05, recent_high_20 * 1.02, cur_p * 1.1)
                x.extend([f_d[days_to_peak], f_d[days_to_peak + 5], f_d[days_to_peak + 15]]); y.extend([exhaustion_p, w_dip, breakout_p])
            else:
                scenario_name = "🐻 熊市格局<br>下降通道"
                bounce = min(cur_p * 1.06, ma20, resist_level); drop = max(iron_p, cur_p * 0.9)
                x.extend([f_d[4], f_d[14]]); y.extend([bounce, drop])
        else:
            if ph_support and cur_p > ph_support:
                scenario_name = "⚖️ 區間突破<br>回測支撐"
                up_1 = min(cur_p * 1.04, resist_level * 0.99 if resist_level != float('inf') else float('inf'))
                dip = max(cur_p * 0.96, support_level); up_2 = max(up_1, dip * 1.05)
                x.extend([f_d[5], f_d[12], f_d[18]]); y.extend([up_1, dip, up_2])
            else:
                scenario_name = "⚖️ 區間震盪<br>挑戰壓力"
                up_1 = min(cur_p * 1.04, resist_level * 0.99 if resist_level != float('inf') else float('inf')) 
                dip = max(cur_p * 0.96, iron_p, support_level); up_2 = resist_level * 1.01 if resist_level != float('inf') else cur_p * 1.05
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

def get_tactical_advice(df, cur_p, t_s, iron_p, ph_support, ph_resist):
    if len(df) < 10: return "⌛ 數據不足", "等待更多 K 線資料寫入...", "#9ca3af"
    latest = df.iloc[-1]
    td_b = latest.get('TD_Buy_Seq', 0); td_s = latest.get('TD_Sell_Seq', 0)
    ma20 = latest.get('SMA_20', cur_p); rsi = latest.get('RSI', 50)
    s_level = ph_support if ph_support is not None else iron_p
    r_level = ph_resist if ph_resist is not None else float('inf')
    
    if r_level != float('inf') and s_level > 0 and ((r_level - s_level) / cur_p * 100) < 5.0:
        return "🌪️ 即將引爆預警 (極限收斂)", f"支撐(${s_level:.2f})與壓力(${r_level:.2f})空間已被極度壓縮至 5% 內，猶如壓緊的彈簧！隨時面臨【暴力表態】，請密切關注突破方向，順勢操作！", "#d946ef" 
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

def get_compre_color_class(text):
    tl = text.lower()
    if any(w in tl for w in ['突破', '回測支撐', 'n字', '看漲']): return 'sig-green'
    if any(w in tl for w in ['跌破', '崩盤', '測底', '無底洞']): return 'sig-red'
    if any(w in tl for w in ['整理', '震盪', '挑戰壓力']): return 'sig-orange'
    return 'sig-gray'

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

# 財報引擎
@st.cache_data(ttl=300)
def get_earnings_status(ticker):
    ignore_list = ["0050", "0056", "00878", "QQQ", "SPY", "DIA", "IWM", "^TWII", "^IXIC"]
    if any(x in ticker for x in ignore_list) or "00403A" in ticker: return "", ""
    
    next_date = "N/A"
    last_result = "⚪ 無前季數據"
    
    try:
        t = yf.Ticker(ticker)
        hist = None
        try: hist = t.get_earnings_dates(limit=12)
        except:
            try: hist = t.earnings_dates
            except: pass
            
        if hist is not None and not hist.empty:
            now_dt = pd.Timestamp.now(tz=hist.index.tz) if hist.index.tz else pd.Timestamp.now()
            future_dates = hist[hist.index > now_dt]
            if not future_dates.empty:
                next_date = future_dates.index.min().strftime('%Y-%m-%d')
                
            try:
                past_dates = hist[hist['Reported EPS'].notna()]
                if not past_dates.empty:
                    last = past_dates.iloc[0]
                    act = last.get('Reported EPS', np.nan)
                    est = last.get('EPS Estimate', np.nan)
                    
                    if pd.notna(act) and pd.notna(est):
                        if act > 0 and act >= est: last_result = f'<span class="earn-beat">🟢 獲利優於預期 (EPS: {act:.2f}|估:{est:.2f})</span>'
                        elif act <= 0 and act >= est: last_result = f'<span class="earn-turn">🟡 虧損優於預期 (EPS: {act:.2f}|估:{est:.2f})</span>'
                        elif act > 0 and act < est: last_result = f'<span class="earn-warn">🟠 獲利遜於預期 (EPS: {act:.2f}|估:{est:.2f})</span>'
                        else: last_result = f'<span class="earn-miss">🔴 虧損遜於預期 (EPS: {act:.2f}|估:{est:.2f})</span>'
                    elif pd.notna(last.get('Surprise(%)')):
                        sur = last['Surprise(%)']
                        last_result = f'<span class="earn-beat">🟢 優於預期 (+{sur:.1f}%)</span>' if sur > 0 else f'<span class="earn-miss">🔴 遜於預期 ({sur:.1f}%)</span>'
            except: pass
        
        if next_date == "N/A":
            try:
                cal = t.calendar
                if isinstance(cal, dict) and 'Earnings Date' in cal and len(cal['Earnings Date']) > 0:
                    next_date = cal['Earnings Date'][0].strftime('%Y-%m-%d')
                elif isinstance(cal, pd.DataFrame) and 'Earnings Date' in cal.columns and not cal.empty:
                    next_date = pd.to_datetime(cal['Earnings Date'].iloc[0]).strftime('%Y-%m-%d')
            except: pass

        return f"📅 財報: {next_date}", last_result
    except: return "📅 財報: N/A", "⚪ 無數據"

# --- 6. 主介面 Sidebar ---
with st.sidebar:
    st.title("🎛️ 控制台")
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
st.title(f"📈 {disp_main_title} 實戰戰情室 V17.67")

api_p, api_i = ("5d", "15m") if "當沖" in time_opt else ("6mo", "1d") if "日" in time_opt else ("2y", "1wk")
df = yf.download(cur_t, period=api_p, interval=api_i, progress=False)
if df.empty: st.error("無數據"); st.stop()
if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
df = df.loc[:, ~df.columns.duplicated()]

df.index = pd.to_datetime(df.index)
if df.index.tz is not None: df.index = df.index.tz_localize(None)

df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=['Open', 'High', 'Low', 'Close'])
if len(df) < 2: st.error("數據連線異常或資料量不足以進行 AI 分析，請稍後再試。"); st.stop()

df = calculate_indicators(df)

latest = df.iloc[-1]; close_v = float(latest['Close']); chg = (close_v - float(df.iloc[-2]['Close'])) / float(df.iloc[-2]['Close']) * 100
clr = "#4ade80" if chg >= 0 else "#ff6b6b"
sigs = analyze_strategic_signals(df)
trend_txt, trend_note, trend_col = analyze_market_trend(df)
rs_txt, rs_col = get_relative_strength(cur_t, df)
engine_label, engine_type = get_stock_engine_mode(cur_t, df)
t_s, t_l, rating = predict_target_and_rating(df)

vp_60 = calculate_volume_profile(df.tail(60), bins=40)
vol_poc = vp_60.loc[vp_60['Volume'].idxmax(), 'Price'] if not vp_60.empty else close_v
iron_price, _box_start, _box_end, is_breaking = find_structural_box_bottom(df, close_v)

p_data = df.tail(120) if "日" in time_opt else df.tail(60)
abs_high = p_data['High'].max()
local_maxes = p_data['High'][(p_data['High'] == p_data['High'].rolling(9, center=True).max())].dropna()
filtered_maxes = local_maxes[local_maxes < abs_high]

resist_line = None; support_line = None
if not filtered_maxes.empty:
    above_current = filtered_maxes[filtered_maxes > close_v]
    if not above_current.empty: resist_line = above_current.iloc[-1]
    below_current = filtered_maxes[filtered_maxes < close_v]
    if not below_current.empty: support_line = below_current.iloc[-1]

# --- 戰術建議盒 ---
tac_title, tac_body, tac_color = get_tactical_advice(df, close_v, t_s, iron_price, support_line, resist_line)
st.markdown(f"""
<div class="tactical-box" style="border-left-color: {tac_color};">
    <div class="tactical-title">🚩 戰術建議： <span style="color:{tac_color}; margin-left: 8px;">{tac_title}</span></div>
    <div class="tactical-body">💡 <b>行動指南：</b> {tac_body}</div>
</div>
""", unsafe_allow_html=True)

# [V17.66] 美股情緒雙核儀表板（僅美股顯示）
_is_us_stock = not (".TW" in cur_t or ".TWO" in cur_t)
if _is_us_stock and _SENTIMENT_AVAILABLE:
    try:
        render_us_sentiment_dashboard()
    except Exception as _e:
        st.caption(f"⚠️ 美股情緒儀表板載入失敗：{_e}")

# [V17.67] 台股空頭危機距離指數儀表板（僅台股顯示）
if (not _is_us_stock) and _TW_SENTIMENT_AVAILABLE:
    try:
        render_tw_crisis_dashboard()
    except Exception as _e:
        st.caption(f"⚠️ 台股危機指數儀表板載入失敗：{_e}")

# ==========================================
# 一體化四大戰情方塊佈局 (1.3 : 1 : 1 : 1)
# ==========================================
ern_date, ern_res = get_earnings_status(cur_t)
px, py, sc_name = generate_projection_points(df, trend_txt, close_v, iron_price, is_breaking, support_line, resist_line)
sc_color_class = get_compre_color_class(sc_name)

c1, c2, c3, c4 = st.columns([1.3, 1, 1, 1])

with c1:
    st.markdown(f'''
    <div class="ai-box" style="border: 1px solid #4a9eff; background-color: #16202b; padding: 15px; display: flex; flex-direction: column; justify-content: center;">
        <h2 style="margin:0; font-size: 38px; font-weight: 900; line-height: 1.1;">${close_v:.2f}</h2>
        <div style="font-size: 18px; font-weight: bold; color: {clr}; margin-bottom: 8px;">{chg:+.2f}% <span style="color: #888; font-size: 13px; font-weight: normal;">(量: {format_volume(latest['Volume'])})</span></div>
        <div><span class="engine-tag" style="margin:0; padding: 3px 8px; font-size: 12px;">⚙️ {engine_label}</span></div>
        <div style="margin-top: 8px; font-size: 12px; line-height: 1.4; color: #ccc;">{ern_date}<br>{ern_res}</div>
    </div>
    ''', unsafe_allow_html=True)

with c2:
    st.markdown(f'''
    <div class="ai-box" style="display:flex; flex-direction:column; justify-content:center;">
        <h5 style="color:white; margin:0; margin-bottom:12px;">📡 綜合戰略</h5>
        <div><span class="{sigs["Summary_Color"]}" style="font-size:14px; font-weight:bold; padding:4px 10px; display:inline-block; line-height:1.4; border-radius:6px; margin-bottom:8px;">{sigs["Summary"]}</span></div>
        <div style="margin-top: 4px; padding-top: 8px; border-top: 1px dashed #555;">
            <span style="color:#aaa; font-size:12px;">🔮 未來推演:</span><br>
            <span class="{sc_color_class}" style="font-size:14px; font-weight:bold; line-height:1.3; border:0; background:transparent;">{sc_name}</span>
        </div>
    </div>
    ''', unsafe_allow_html=True)

with c3:
    st.markdown(f'''
    <div class="ai-box" style="display: flex; flex-direction: column; justify-content: center;">
        <h5 style="color:white; margin:0; margin-bottom:12px;">⚖️ 雙重格局</h5>
        <div style="font-size:14px; margin-bottom: 8px;">🏢 個股: <span class="{trend_col}" style="display: inline-block;">{trend_txt}</span></div>
        <div style="font-size:13px; color: #888;">({trend_note})</div>
    </div>
    ''', unsafe_allow_html=True)

with c4:
    st.markdown(f'''
    <div class="ai-box" style="border: 1px solid #00d4ff; display: flex; flex-direction: column; justify-content: center;">
        <h5 style="color:white; margin:0; margin-bottom:12px;">🎯 AI 目標 & 強弱</h5>
        <div style="font-size:14px; color: #ddd; margin-bottom: 8px;">短: ${t_s:.2f} | 長: ${t_l:.2f}</div>
        <div><span class="{rs_col}" style="display: inline-block;">{rs_txt}</span></div>
    </div>
    ''', unsafe_allow_html=True)

st.markdown("<div style='margin-bottom: 15px;'></div>", unsafe_allow_html=True)

# ==========================================
# 繪圖區
# ==========================================
fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_width=[0.2, 0.2, 0.6])

fig.add_trace(go.Candlestick(x=p_data.index, open=p_data['Open'], high=p_data['High'], low=p_data['Low'], close=p_data['Close'], name="K線"), row=1, col=1)
fig.add_trace(go.Scatter(x=p_data.index, y=p_data['ATR_Trailing_Stop'], mode='lines', line=dict(color='#FF5F1F', width=1.5, dash='dot'), name='ATR 停損'), row=1, col=1)

zone_top = vol_poc * 1.025; zone_bottom = vol_poc * 0.975
fig.add_hrect(y0=zone_bottom, y1=zone_top, line_width=0, fillcolor="rgba(100, 100, 100, 0.2)", layer="below", row=1, col=1)

last_d = p_data.index[-1]
for r in range(1, 4): fig.add_vline(x=last_d, line_dash="dash", line_color="#666", opacity=0.7, layer="below", row=r, col=1)

for i in range(5, len(p_data)):
    curr, prior = p_data.iloc[i], p_data.iloc[i-1]
    
    td_b = curr['TD_Buy_Seq']; td_s = curr['TD_Sell_Seq']
    if 0 < td_b <= 9: fig.add_annotation(x=p_data.index[i], y=curr['Low'], text=str(int(td_b)), showarrow=False, yshift=-12, font=dict(color='#ff6b6b', size=9 if td_b<9 else 14, weight="normal" if td_b<9 else "bold"), row=1, col=1)
    if 0 < td_s <= 9: fig.add_annotation(x=p_data.index[i], y=curr['High'], text=str(int(td_s)), showarrow=False, yshift=12, font=dict(color='#4a9eff', size=9 if td_s<9 else 14, weight="normal" if td_s<9 else "bold"), row=1, col=1)

    if prior['Close'] < prior['Open'] and curr['Close'] > curr['Open'] and curr['Open'] <= prior['Close'] and curr['Close'] >= prior['Open']:
        fig.add_annotation(x=p_data.index[i], y=curr['Low'], text="🕯️吞噬", showarrow=True, arrowhead=1, ax=0, ay=30, row=1, col=1, font=dict(color="orange", size=9))

    status = detect_smart_money_status(p_data.iloc[:i+1])
    if status:
        if "吸籌" in status: fig.add_annotation(x=p_data.index[i], y=curr['Low'], text=f"🐳吸<br>${curr['Low']:.1f}", showarrow=True, arrowhead=1, ax=0, ay=35, row=1, col=1, bgcolor="rgba(111, 66, 193, 0.8)", font=dict(color="white", size=9))
        elif "破底翻" in status or "抄底" in status: fig.add_annotation(x=p_data.index[i], y=curr['Low'], text=f"{status}<br>${curr['Low']:.1f}", showarrow=True, arrowhead=1, ax=0, ay=45, row=1, col=1, bgcolor="rgba(147, 51, 234, 0.8)", font=dict(color="white", size=10, weight="bold"))
        elif "調節" in status: fig.add_annotation(x=p_data.index[i], y=curr['High'], text=f"🔴調節<br>${curr['High']:.1f}", showarrow=True, arrowhead=1, ax=0, ay=-40, row=1, col=1, bgcolor="rgba(185, 28, 28, 0.8)", font=dict(color="white", size=10, weight="bold"))

    macd_buy = (curr['MACD'] > curr['Signal_Line']) and (prior['MACD'] <= prior['Signal_Line'])
    macd_sell = (curr['MACD'] < curr['Signal_Line']) and (prior['MACD'] >= prior['Signal_Line'])
    if macd_buy and ((engine_type == "trend" and curr['Close'] < curr.get('SMA_60', 0)) or (engine_type == "momentum" and curr['Close'] < curr.get('SMA_20', 0))): macd_buy = False
    
    if macd_buy: fig.add_annotation(x=p_data.index[i], y=curr['Low'], text=f"BUY<br>${curr['Close']:.1f}", showarrow=True, arrowhead=1, ax=0, ay=25, row=1, col=1, bgcolor="rgba(40, 167, 69, 0.8)", font=dict(color="white", size=9))
    if macd_sell: fig.add_annotation(x=p_data.index[i], y=curr['High'], text=f"SELL<br>${curr['Close']:.1f}", showarrow=True, arrowhead=1, ax=0, ay=-25, row=1, col=1, bgcolor="rgba(220, 53, 69, 0.8)", font=dict(color="white", size=9))
    if (curr['High'] >= t_s or curr['RSI'] > 75) and not (prior['High'] >= t_s or prior['RSI'] > 75): fig.add_annotation(x=p_data.index[i], y=curr['High'], text=f"💰達標<br>${curr['Close']:.1f}" if curr['High'] >= t_s else f"🔥過熱<br>${curr['Close']:.1f}", showarrow=True, arrowhead=1, ax=0, ay=-45, row=1, col=1, bgcolor="rgba(255, 193, 7, 0.8)" if curr['High'] >= t_s else "rgba(255, 69, 0, 0.8)", font=dict(color="black" if curr['High'] >= t_s else "white", size=9))

    is_near_support = False
    if support_line and abs(curr['Low'] - support_line) / support_line < 0.015: is_near_support = True
    if iron_price > 0 and abs(curr['Low'] - iron_price) / iron_price < 0.015: is_near_support = True
    
    vol_surge = False 
    if not pd.isna(p_data['Vol_SMA5'].iloc[i]): vol_surge = curr['Volume'] > p_data['Vol_SMA5'].iloc[i] * 1.2
        
    strong_reversal = False 
    if curr['Close'] > curr['Open']: strong_reversal = True
    elif ((min(curr['Close'], curr['Open']) - curr['Low']) > abs(curr['Close'] - curr['Open']) * 2.5): strong_reversal = True

    if is_near_support and vol_surge and strong_reversal:
        fig.add_annotation(x=p_data.index[i], y=curr['Low'], text="支撐", showarrow=True, arrowhead=1, ax=0, ay=30, row=1, col=1, font=dict(color="#00ffff", size=9, weight="bold"))

fig.add_trace(go.Scatter(x=px, y=py, mode='lines+markers', line=dict(color='#eab308', width=1, dash='dash'), marker=dict(size=8, symbol='diamond', color='#eab308'), name='🔮 AI 劇本推演'), row=1, col=1)
for i in range(1, len(px)): fig.add_annotation(x=px[i], y=py[i], text=f"${py[i]:.2f}", showarrow=True, arrowhead=0, ay=-20, font=dict(color="#eab308", size=11), bgcolor="rgba(0,0,0,0.6)", row=1, col=1)

fig.add_hline(y=abs_high, line_dash="dot", line_color="#ef4444", annotation_text=f"🔴 波段最高<br>${abs_high:.2f}", annotation_font_color="#ef4444", annotation_position="top right", annotation_align="right", opacity=1.0, layer="above", row=1, col=1)
if resist_line: fig.add_hline(y=resist_line, line_dash="dot", line_color="#f97316", annotation_text=f"🟠 前高壓力<br>${resist_line:.2f}", annotation_font_color="#f97316", annotation_position="top right", annotation_align="right", opacity=1.0, layer="above", row=1, col=1)
if support_line: fig.add_hline(y=support_line, line_dash="dot", line_color="#3b82f6", annotation_text=f"🔵 前高支撐<br>${support_line:.2f}", annotation_font_color="#3b82f6", annotation_position="bottom right", annotation_align="right", opacity=1.0, layer="above", row=1, col=1)
if not is_breaking and iron_price > 0: fig.add_hline(y=iron_price, line_dash="dash", line_color="#20c997", annotation_text=f"🧱 鐵板 ${iron_price:.2f}", annotation_font_color="#20c997", annotation_position="bottom right", opacity=1.0, layer="above", row=1, col=1)

fig.add_annotation(x=0.5, y=0.98, xref="paper", yref="paper", text=f"🛡️ 籌碼攻防帶 (突破: ${zone_top:.2f} | 跌破: ${zone_bottom:.2f})", showarrow=False, font=dict(color="#d8b4fe", size=12, weight="bold"), bgcolor="rgba(10, 10, 10, 0.9)", borderpad=4, xanchor="center", yanchor="top")
fig.add_trace(go.Scatter(x=[last_d], y=[close_v], mode='markers', marker=dict(size=6, color='rgba(100, 149, 237, 0.6)', line=dict(color='rgba(255, 255, 255, 0.5)', width=1)), name="今日收盤"), row=1, col=1)

# 副圖
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

st.markdown("---")

# ==========================================
# AI 勝率推演與宏觀面板
# ==========================================
engine_name = "🇹🇼 台股重訊模式" if ".TW" in cur_t or ".TWO" in cur_t else "🇺🇸 美股雙境獵手"
with st.spinner(f"🕵️‍♂️ 啟動{engine_name}：正在掃描並過濾情報..."): items = fetch_deep_news(cur_t, is_macro=False)

news_score, total_insider_penalty, valid_count, has_major = 0, 0, 0, False
anchor = get_valid_anchor(cur_t)
if anchor: news_score += anchor['score']

for item in items:
    s, tag, major, penalty = analyze_news_strict(item['title'], item['cat'])
    if s == 0 and not tag: continue
    news_score += s; total_insider_penalty += penalty; valid_count += 1
    if major: has_major = True
    if major and s > 0: update_anchor(cur_t, item['title'], 3.0, item['date'])

macro_txt, macro_note, macro_col, macro_score = get_realtime_macro()
win_rate = max(10.0, min(95.0, 50.0 + (news_score * 5) + (macro_score * 5) - (20 if total_insider_penalty <= -3.0 else 0)))

if total_insider_penalty <= -4.0:
    final_verdict, v_col = f"⚠️ 謹慎持有 (熔斷)！內部人/CFO 大量拋售", "#ffc107"
    m_disp = f'<div class="macro-alert" style="background-color:#3a1b1b; color:#ffc107; border:1px solid #ffc107;">⚡ 觸發內部人熔斷：高管拋售過大，強制降評</div>'
elif has_major:
    final_verdict, v_col = f"🚀 強力看漲 (霸體)！重訊/財報利多 (+{news_score:.1f})", "#4ade80"
    m_disp = f'<div class="macro-alert" style="background-color:#1b3a1b; color:#4ade80; border:1px solid #28a745;">💎 偵測到重大訊息：已自動忽略宏觀風險 ({macro_txt})</div>'
else:
    news_score += macro_score
    m_disp = f'<div class="macro-alert">{macro_txt}：{macro_note}，評分已下修</div>' if macro_score < 0 else f'<div style="color:orange;">{macro_txt}</div>'
    if news_score >= 3: final_verdict, v_col = "📈 偏多操作 (基本面支撐)", "#ffc107"
    elif news_score <= -2: final_verdict, v_col = "📉 偏空看待 (利空罩頂)", "#ff6b6b"
    else: final_verdict, v_col = "☁️ 觀望整理 (缺乏驅動力)", "gray"

nc1, nc2 = st.columns([0.5, 0.5])
with nc1:
    st.markdown(m_disp, unsafe_allow_html=True)
    if anchor: st.markdown(f'<div class="anchor-box"><div class="anchor-title-cn">⚓ 記憶錨定 ({anchor["date"]})</div><div style="color: #00ffff; margin-bottom:5px;">{anchor["summary"]} (+{anchor["score"]}分)</div><div class="anchor-title-en">{anchor["title"]}</div></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="ai-box" style="text-align:left;"><h3 style="color:white; margin:0;">🔮 AI 戰略勝率推演</h3><div style="font-size:18px; color:{v_col}; font-weight:bold; margin-top:10px;">{final_verdict}</div><div style="font-size:24px; color:#00ffff; font-weight:bold; margin-top:5px;">📈 多方勝率：{win_rate:.1f}%</div><hr style="border-color:#555;"><div style="font-size:14px; color:#ccc;"><b>基本面得分：</b> {news_score:.1f} | <b>內部人扣分：</b> {total_insider_penalty:.1f} | <b>情報數：</b> {valid_count}</div></div>', unsafe_allow_html=True)
    
with nc2:
    st.markdown(f'<div class="ai-box" style="text-align:left; background-color:#1e1e1e;"><h4 style="color:white; margin:0; margin-bottom:15px;">📰 AI 情報判讀結果 (已過濾雜訊)</h4>', unsafe_allow_html=True)
    if valid_count > 0:
        st.markdown(f'<div style="color:#4ade80; margin-bottom:10px;">✅ 系統已自動閱讀並評分 {valid_count} 則關鍵情報。</div>', unsafe_allow_html=True)
        st.markdown('<div style="color:#aaa; font-size:14px;">(為保持版面潔淨，已隱藏新聞連結列表。AI 引擎仍會在背景 24 小時監控重訊與籌碼異動)</div></div>', unsafe_allow_html=True)
    else:
        st.info("近 30 日無重大影響勝率之情報。")
        st.markdown('</div>', unsafe_allow_html=True)
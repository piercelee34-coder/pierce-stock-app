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

# --- 改用瀏覽器本地儲存 ---
from streamlit_local_storage import LocalStorage

# --- 0. 系統設定 ---
st.set_page_config(page_title="AI 實戰戰情室 V13.4 (雙軌錨定版)", layout="wide", page_icon="🧠")

# --- CSS 美化 ---
st.markdown("""
<style>
    .price-card {background-color: #1e1e1e; padding: 20px; border-radius: 10px; text-align: center; border: 1px solid #333; margin-bottom: 10px;}
    .ai-box {background-color: #333; padding: 10px; border-radius: 10px; border: 1px solid #555; text-align: center; height: 100%;}
    .news-card {background-color: #262730; padding: 10px; border-radius: 5px; border-left: 3px solid #FFD700; margin-bottom: 8px; font-size: 14px;}
    .news-link {text-decoration: none; color: #e0e0e0; font-weight: bold;}
    .news-link:hover {color: #FFD700;}
    .macro-alert {background-color: #3a1b1b; color: #ff6b6b; padding: 10px; border-radius: 5px; border: 1px solid #dc3545; margin-bottom: 10px; font-weight: bold;}
    .anchor-box {background-color: #1b3a4a; color: #00d4ff; padding: 10px; border-radius: 5px; border: 1px solid #00d4ff; margin-bottom: 10px; font-size: 13px;}
    
    .signal-tag {font-weight: bold; padding: 2px 8px; border-radius: 4px; font-size: 13px; display: inline-block;}
    .tag-red {background-color: #3a1b1b; color: #ff6b6b; border: 1px solid #dc3545;}
    .tag-green {background-color: #1b3a1b; color: #4ade80; border: 1px solid #28a745;}
    .tag-orange {background-color: #4a3b1b; color: #ffaa00; border: 1px solid #ffc107;}
    .tag-gray {background-color: #333; color: #ccc; border: 1px solid #666;}
    .tag-purple {background-color: #4a1b4a; color: #d8b4fe; border: 1px solid #a855f7;} 
    .tag-blue {background-color: #1b3a4a; color: #4a9eff; border: 1px solid #00d4ff;}
    
    .header-legend {text-align: right; font-size: 13px; padding-top: 25px; color: #ccc;}
</style>
""", unsafe_allow_html=True)

# --- 1. 自選股儲存系統 ---
DEFAULT_LIST = ['NVDA', 'TSM', 'AAPL', 'MU', 'TSLA', 'ONDS', 'QQQ', '^IXIC', '0050.TW', '2330.TW']

def init_storage(): return LocalStorage()

def load_watchlist(ls):
    try:
        stored = ls.getItem("my_watchlist")
        return stored if stored and isinstance(stored, list) else DEFAULT_LIST
    except: return DEFAULT_LIST

def save_watchlist(ls, watchlist):
    try: ls.setItem("my_watchlist", watchlist); st.toast("✅ 清單已更新", icon="💾")
    except: pass

# [V13.4 新增] 記憶錨定系統
def save_anchor(ls, ticker, news_title, score):
    """將 Level 3 重大新聞存入記憶"""
    try:
        key = f"anchor_{ticker}"
        data = {
            "title": news_title,
            "score": score,
            "date": datetime.now().strftime("%Y-%m-%d")
        }
        ls.setItem(key, data)
    except: pass

def get_valid_anchor(ls, ticker):
    """讀取 5 天內的有效記憶"""
    try:
        key = f"anchor_{ticker}"
        data = ls.getItem(key)
        if not data: return None
        
        # 檢查是否過期 (5天)
        saved_date = datetime.strptime(data["date"], "%Y-%m-%d")
        if (datetime.now() - saved_date).days > 5:
            ls.deleteItem(key) # 過期刪除
            return None
        return data
    except: return None

ls = init_storage()
if 'watchlist' not in st.session_state:
    st.session_state.watchlist = load_watchlist(ls)

# --- 2. 核心函數 ---

def fetch_google_news_rss(query, is_macro=False):
    """通用新聞抓取器"""
    try:
        # 如果是宏觀，強制搜全球財經
        search_query = query if is_macro else f"{query} stock"
        if ".TW" in query: search_query = f"{query.replace('.TW','')} 營收 訂單"
        
        url = f"https://news.google.com/rss/search?q={search_query}&hl=zh-TW&gl=TW&ceid=TW:zh-Hant"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=5)
        
        if response.status_code != 200: return []
        
        root = ET.fromstring(response.content)
        news_items = []
        for item in root.findall('.//item')[:6]:
            title = re.sub('<[^<]+?>', '', item.find('title').text)
            link = item.find('link').text
            try:
                pub_date = item.find('pubDate').text
                dt = datetime.strptime(pub_date[:16], '%a, %d %b %Y')
                date_str = dt.strftime('%m/%d')
            except: date_str = "近期"
            news_items.append({'title': title, 'link': link, 'date': date_str})
        return news_items
    except: return []

def analyze_news_weight(title):
    """
    [V13.4] 權重計分大腦
    回傳: (分數, 是否為Level 3證據)
    """
    t = title.lower()
    score = 0
    is_l3 = False
    
    # Level 3: 實質證據 (權重 3.0) - 錨定觸發點
    l3_words = ['訂單', '簽約', '奪下', '供不應求', '缺貨', '漲價', '擴產', '滿載', '認證', 
                'order', 'deal', 'contract', 'record', 'shortage', 'hike', 'capacity']
    for w in l3_words:
        if w in t: 
            score += 3.0
            is_l3 = True
            
    # Level 2: 財務事實 (權重 1.5)
    l2_words = ['營收', '獲利', 'eps', '毛利', '配息', '增長', '年增', '月增', 'beat', 'profit', 'revenue', 'growth']
    for w in l2_words:
        if w in t: score += 1.5
        
    # Level 1: 市場情緒 (權重 0.5)
    l1_words = ['看好', '樂觀', '喊進', '目標價', '飆', '噴', 'buy', 'bull', 'target', 'upgrade']
    for w in l1_words:
        if w in t: score += 0.5
        
    # Level -2: 實質利空/風險
    neg_words = ['下修', '砍單', '庫存', '不如', '虧損', '重挫', '衰退', 'cut', 'miss', 'loss', 'drop', 'inventory']
    for w in neg_words:
        if w in t: score -= 2.0
        
    # 謠言濾網 (分數減半)
    rumor_words = ['傳', '據悉', '消息人士', 'rumor', 'reportedly']
    for w in rumor_words:
        if w in t: score *= 0.5
        
    return score, is_l3

def get_macro_environment():
    """
    [V13.4] 宏觀風向球 (Fed/通膨)
    """
    # 搜尋聯準會相關新聞
    news = fetch_google_news_rss("Federal Reserve Interest Rate", is_macro=True)
    score = 0
    summary = "平穩"
    
    hawk_words = ['hike', 'raise', 'inflation', 'hawk', 'warn', '升息', '通膨', '鷹', '警告', '熱']
    dove_words = ['cut', 'pause', 'slow', 'dove', '降息', '暫停', '趨緩', '鴿']
    
    for n in news:
        t = n['title'].lower()
        for w in hawk_words: 
            if w in t: score -= 2 # 鷹派扣分
        for w in dove_words: 
            if w in t: score += 1 # 鴿派加分
            
    if score <= -3: return "⛈️ 系統風險高 (Fed偏鷹)", "bad"
    elif score < 0: return "☁️ 宏觀偏空 (觀望)", "fair"
    else: return "🌤️ 宏觀穩健", "good"

def calculate_volume_profile(df, bins=40, filter_mask=None):
    if df.empty: return pd.DataFrame({'Price': [], 'Volume': []})
    price_min = df['Low'].min(); price_max = df['High'].max()
    if price_min == price_max: return pd.DataFrame({'Price': [], 'Volume': []})
    bin_edges = np.linspace(price_min, price_max, bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    target_df = df if filter_mask is None else df[filter_mask]
    if target_df.empty: return pd.DataFrame({'Price': bin_centers, 'Volume': np.zeros(bins)})
    bin_indices = pd.cut(target_df['Close'], bins=bin_edges, labels=False, include_lowest=True)
    profile_series = target_df.groupby(bin_indices)['Volume'].sum()
    profile_series = profile_series.reindex(range(bins), fill_value=0)
    return pd.DataFrame({'Price': bin_centers, 'Volume': profile_series.values})

def calculate_indicators(df):
    if len(df) < 50: return df
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_60'] = df['Close'].rolling(window=60).mean()
    df['Vol_SMA5'] = df['Volume'].rolling(window=5).mean()
    df['Std_Dev'] = df['Close'].rolling(window=20).std()
    df['Bollinger_Upper'] = df['SMA_20'] + (df['Std_Dev'] * 2)
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
    clv = clv.fillna(0)
    df['AD_Line'] = (clv * df['Volume']).cumsum()
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

def find_support_levels(df, current_price):
    if df.empty or len(df) < 60: return current_price, current_price, "資料不足", "資料不足"
    s1 = df['Close'].rolling(window=20).mean().iloc[-1]
    if current_price > s1: s1_note = "股價在月線之上 (趨勢多)"
    else: s1_note = "已跌破月線 (趨勢轉弱)"
    recent_60 = df.tail(60)
    max_vol_date = recent_60['Volume'].idxmax()
    key_bar_low = df.loc[max_vol_date]['Low']
    s2_date_str = max_vol_date.strftime('%m/%d')
    if current_price < key_bar_low: s2 = recent_60['Low'].min(); s2_note = f"主力籌碼區({s2_date_str})已失守"
    else: s2 = key_bar_low; s2_note = f"最大量日({s2_date_str})低點"
    return s1, s2, s1_note, s2_note

def detect_smart_money_divergence(df):
    if len(df) < 10: return None
    price_now = df['Close'].iloc[-1]; price_5d = df['Close'].iloc[-6]
    ad_now = df['AD_Line'].iloc[-1]; ad_5d = df['AD_Line'].iloc[-6]
    rsi = df['RSI'].iloc[-1]
    price_drop = price_now < price_5d * 0.98; ad_rise = ad_now > ad_5d               
    if price_drop and ad_rise and rsi < 50: return "🎯 主力背離吸籌 (價跌量增)"
    if rsi < 30 and df['Volume'].iloc[-1] > df['Vol_SMA5'].iloc[-1]: return "⚡ 恐慌殺盤 (主力接刀)"
    return None

def analyze_strategic_signals(df):
    if df.empty: return {}
    latest = df.iloc[-1]; prev = df.iloc[-2] if len(df) > 1 else latest
    macd = latest['MACD']; signal = latest['Signal_Line']
    if macd > signal:
        if macd > 0: macd_text, macd_color = "零軸上金叉 (多頭)", "tag-green"
        else: macd_text, macd_color = "零軸下金叉 (反彈)", "tag-orange"
    else:
        if macd > 0: macd_text, macd_color = "零軸上死叉 (修正)", "tag-orange"
        else: macd_text, macd_color = "零軸下死叉 (空頭)", "tag-red"
    vol = latest['Volume']; vol_ma = latest['Vol_SMA5']
    if vol > vol_ma * 1.5: vol_text, vol_color = "爆量 (>1.5倍)", "tag-green"
    elif vol > vol_ma * 1.1: vol_text, vol_color = "量增", "tag-green"
    else: vol_text, vol_color = "量縮/平量", "tag-gray"
    rsi = latest['RSI']
    if rsi > 70: rsi_text, rsi_color = f"過熱 ({rsi:.0f})", "tag-red"
    elif rsi < 30: rsi_text, rsi_color = f"超賣 ({rsi:.0f})", "tag-green"
    else: rsi_text, rsi_color = f"中性 ({rsi:.0f})", "tag-gray"
    recent_20 = df.tail(20)
    box_range = (recent_20['High'].max() - recent_20['Low'].min()) / recent_20['Low'].min() * 100
    if box_range < 8: trend_text, trend_color = f"⚠️ 盤整 ({box_range:.1f}%)", "tag-orange"
    else: trend_text, trend_color = "🌊 趨勢盤", "tag-green"
    summary = "觀望"; summary_color = "tag-gray"
    td_buy_stop = latest.get('TD_Buy_Stop', np.nan)
    is_td_buy_9 = not np.isnan(latest.get('TD_Buy_9', np.nan))
    is_td_sell_9 = not np.isnan(latest.get('TD_Sell_9', np.nan))
    dma_gold = (latest['DMA_DDD'] > latest['DMA_AMA']) and (prev['DMA_DDD'] <= prev['DMA_AMA'])
    if is_td_sell_9: summary, summary_color = "🔺 九轉賣點 (藍9)", "tag-blue"
    elif is_td_buy_9: summary, summary_color = f"🔻 九轉買點 (紅9, SL:{td_buy_stop:.1f})", "tag-purple"
    elif dma_gold: summary, summary_color = "🚀 DMA 金叉翻多", "tag-green"
    elif macd > signal: summary, summary_color = "📈 偏多震盪", "tag-green"
    else: summary, summary_color = "⛈️ 空頭走勢", "tag-red"
    hunter_signal = detect_smart_money_divergence(df)
    if hunter_signal: summary = hunter_signal; summary_color = "tag-purple"
    return {"MACD_Text": macd_text, "MACD_Color": macd_color, "Vol_Text": vol_text, "Vol_Color": vol_color, "RSI_Text": rsi_text, "RSI_Color": rsi_color, "Trend_Text": trend_text, "Trend_Color": trend_color, "Summary": summary, "Summary_Color": summary_color}

def analyze_market_trend(df):
    price = df['Close'].iloc[-1]; ma20 = df['SMA_20'].iloc[-1]; ma60 = df['SMA_60'].iloc[-1]
    if price > ma20 and ma20 > ma60: return "🐂 牛市 (Bull)", "多頭排列，順勢操作"
    elif price < ma20 and ma20 < ma60: return "🐻 熊市 (Bear)", "空頭排列，保守為宜"
    else: return "⚖️ 震盪 (Range)", "區間整理，高出低進"

def predict_target_and_rating(df):
    price = df['Close'].iloc[-1]; ma20 = df['SMA_20'].iloc[-1]; macd = df['MACD'].iloc[-1]
    signal = df['Signal_Line'].iloc[-1]; rsi = df['RSI'].iloc[-1]; upper_band = df['Bollinger_Upper'].iloc[-1]
    dma_ddd = df['DMA_DDD'].iloc[-1]; dma_ama = df['DMA_AMA'].iloc[-1]
    score = 0
    if price > ma20: score += 1
    if macd > signal: score += 1
    if rsi < 70 and rsi > 40: score += 1
    if df['Volume'].iloc[-1] > df['Vol_SMA5'].iloc[-1]: score += 1
    if dma_ddd > dma_ama: score += 1
    if score >= 4: rating = "💪 強力買進"
    elif score >= 2: rating = "✊ 持有/續抱"
    else: rating = "✋ 觀望/賣出"
    target_short = price * 1.05 if price > upper_band else upper_band
    recent_60_high = df['High'].tail(60).max()
    target_long = max(recent_60_high * 1.15, target_short * 1.1) 
    return target_short, target_long, rating

def generate_buy_hint(df, current_price, s1, s2):
    if df.empty: return "無資料"
    latest = df.iloc[-1]; prev = df.iloc[-2]
    rsi = latest['RSI']; macd = latest['MACD']; signal = latest['Signal_Line']
    if not np.isnan(latest.get('TD_Sell_9', np.nan)): return "🔺 藍色9 (上漲力竭)，注意獲利"
    if not np.isnan(latest.get('TD_Buy_9', np.nan)): return f"🔻 紅色9 (潛在買點)，SL:{latest.get('TD_Buy_Stop', 0):.2f}"
    if (latest['DMA_DDD'] > latest['DMA_AMA']) and (prev['DMA_DDD'] <= prev['DMA_AMA']): return "🚀 DMA 金叉，中線轉多"
    divergence = detect_smart_money_divergence(df)
    if divergence: return f"🚨 {divergence}"
    hints = []
    if abs(current_price - s1) / current_price < 0.015 and current_price > s1: hints.append("回測月線有撐")
    if abs(current_price - s2) / current_price < 0.02: hints.append("近主力成本區")
    if rsi < 30: hints.append("RSI超賣反彈")
    if macd > signal and df['MACD'].iloc[-2] <= df['Signal_Line'].iloc[-2]: hints.append("MACD剛翻多")
    if not hints:
        if current_price > s1 * 1.1: return "乖離過大，勿追高"
        else: return "觀望，等待訊號"
    return " | ".join(hints)

def format_volume(num):
    if num >= 1_000_000_000: return f"{num/1_000_000_000:.2f}B"
    elif num >= 1_000_000: return f"{num/1_000_000:.2f}M"
    elif num >= 1_000: return f"{num/1_000:.2f}K"
    else: return f"{num}"

# --- 3. 側邊欄 ---
with st.sidebar:
    st.title("🎛️ 控制台")
    st.markdown("---")
    st.header("📌 自選股清單")
    selection = st.radio("選擇股票", st.session_state.watchlist)
    current_ticker = selection
    st.markdown("---")
    c_up, c_down = st.columns(2)
    if c_up.button("⬆️ 上移") and current_ticker in st.session_state.watchlist:
        idx = st.session_state.watchlist.index(current_ticker)
        if idx > 0:
            st.session_state.watchlist[idx], st.session_state.watchlist[idx-1] = st.session_state.watchlist[idx-1], st.session_state.watchlist[idx]
            save_watchlist(ls, st.session_state.watchlist); st.rerun()
    if c_down.button("⬇️ 下移") and current_ticker in st.session_state.watchlist:
        idx = st.session_state.watchlist.index(current_ticker)
        if idx < len(st.session_state.watchlist) - 1:
            st.session_state.watchlist[idx], st.session_state.watchlist[idx+1] = st.session_state.watchlist[idx+1], st.session_state.watchlist[idx]
            save_watchlist(ls, st.session_state.watchlist); st.rerun()
    
    c_top, c_bottom = st.columns(2)
    if c_top.button("⏫ 置頂") and current_ticker in st.session_state.watchlist:
        st.session_state.watchlist.remove(current_ticker)
        st.session_state.watchlist.insert(0, current_ticker)
        save_watchlist(ls, st.session_state.watchlist); st.rerun()
    if c_bottom.button("⏬ 置底") and current_ticker in st.session_state.watchlist:
        st.session_state.watchlist.remove(current_ticker)
        st.session_state.watchlist.append(current_ticker)
        save_watchlist(ls, st.session_state.watchlist); st.rerun()

    with st.expander("編輯清單"):
        new_t = st.text_input("輸入代號", placeholder="MSTR").upper()
        c1, c2 = st.columns(2)
        if c1.button("➕ 新增"):
            if new_t and new_t not in st.session_state.watchlist:
                st.session_state.watchlist.append(new_t)
                save_watchlist(ls, st.session_state.watchlist); st.rerun()
        if c2.button("❌ 刪除"):
            if current_ticker in st.session_state.watchlist:
                st.session_state.watchlist.remove(current_ticker)
                save_watchlist(ls, st.session_state.watchlist); st.rerun()

# --- 4. 主程式 ---
st.title(f"📈 {current_ticker} 實戰戰情室 V13.4")

@st.cache_data(ttl=300)
def fetch_main_data(ticker, period, interval):
    try: return yf.download(ticker, period=period, interval=interval, progress=False, threads=False)
    except Exception: return pd.DataFrame()

@st.cache_data(ttl=3600)
def fetch_fundamental_info(ticker):
    try: return yf.Ticker(ticker).info
    except Exception: return None

# [Section 1] 預留頂部容器
metrics_placeholder = st.container()

st.write("") 

# [Section 2] 控制選單 & 標題
t_col1, t_col2 = st.columns([0.65, 0.35])
with t_col1:
    st.subheader(f"📈 走勢圖 (含九轉/DMA)")
with t_col2:
    st.markdown("""
        <div class="header-legend">
            <span style="color:#6f42c1; font-weight:bold; margin-right:10px;">🐳 主力吸籌</span>
            <span style="color:#ff6b6b; font-weight:bold;">▼ 紅9買</span>
        </div>
    """, unsafe_allow_html=True)

# 手機圖表觸控開關
enable_touch = st.toggle("🖐️ 啟用圖表操作 (開啟後可縮放/平移，關閉後可滑動網頁)", value=False)

# 週期選單
time_opt = st.radio("選擇週期", ["當沖 (分時)", "日線 (Daily)", "週線 (Weekly)", "月線 (長線)"], 
                    index=1, horizontal=True, label_visibility="collapsed")

# [Section 3] 資料邏輯
api_period = "1y"; api_interval = "1d"; xaxis_format = "%Y-%m-%d"
if "當沖" in time_opt: api_period = "5d"; api_interval = "15m"; xaxis_format = "%H:%M" 
elif "日線" in time_opt: api_period = "6mo"; api_interval = "1d"; xaxis_format = "%m-%d" 
elif "週線" in time_opt: api_period = "2y"; api_interval = "1wk"; xaxis_format = "%Y-%m-%d"
elif "月線" in time_opt: api_period = "2y"; api_interval = "1wk"; xaxis_format = "%Y-%m"

try:
    df = fetch_main_data(current_ticker, api_period, api_interval)
    info = fetch_fundamental_info(current_ticker)
    
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    if df.empty: st.error("⚠️ 無法取得數據。"); st.stop()

    df = calculate_indicators(df)
    if 'DMA_DDD' not in df.columns or 'TD_Buy_9' not in df.columns:
         st.error("⚠️ 數據不足以計算高階指標 (DMA/九轉)，請選擇更長的週期。"); st.stop()

    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else latest
    s1, s2, s1_note, s2_note = find_support_levels(df, latest['Close'])
    buy_hint_text = generate_buy_hint(df, latest['Close'], s1, s2)
    strat_signals = analyze_strategic_signals(df)
    trend_icon, trend_desc = analyze_market_trend(df)
    target_short, target_long, ai_rating = predict_target_and_rating(df)
    pct_change = ((latest['Close'] - prev['Close']) / prev['Close']) * 100
    color_price = "green" if pct_change >= 0 else "red"

    # [Section 4] 回填頂部容器
    with metrics_placeholder:
        # 1. 現價卡片
        st.markdown(f"""
        <div class="price-card">
            <h1 style="margin:0; font-size: 50px;">${latest['Close']:.2f}</h1>
            <h3 style="margin:0; color: {color_price};">{pct_change:+.2f}%</h3>
            <p style="color: gray; margin-bottom: 5px;">最新成交量: {format_volume(latest['Volume'])}</p>
            <div class="buy-hint">💡 操作提示: {buy_hint_text}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # 2. 戰略雷達
        r_col1, r_col2, r_col3 = st.columns(3)
        with r_col1:
            st.markdown(f"""
            <div class="ai-box">
                <h5 style="color:white; margin:0; margin-bottom:5px;">📡 綜合戰略</h5>
                <div class="signal-tag {strat_signals['Summary_Color']}" style="font-size:16px;">{strat_signals['Summary']}</div>
                <div class="radar-grid">
                    <div class="radar-item"><span>1. MACD</span><span class="signal-tag {strat_signals['MACD_Color']}">{strat_signals['MACD_Text']}</span></div>
                    <div class="radar-item"><span>2. 成交量</span><span class="signal-tag {strat_signals['Vol_Color']}">{strat_signals['Vol_Text']}</span></div>
                    <div class="radar-item"><span>3. RSI</span><span class="signal-tag {strat_signals['RSI_Color']}">{strat_signals['RSI_Text']}</span></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        with r_col2:
            trend_color = "#ccc"
            if "牛市" in trend_icon: trend_color = "#4ade80" 
            elif "熊市" in trend_icon: trend_color = "#ff6b6b"
            elif "震盪" in trend_icon: trend_color = "#ffc107"
            
            st.markdown(f"""<div class="ai-box"><h5 style="color:white; margin:0;">⚖️ 格局&評級</h5><div style="font-size: 30px; margin-top:5px;">{trend_icon.split(' ')[0]} <span style="font-size:20px; color:#FFD700;">{ai_rating}</span></div><p style="font-size:12px; color:{trend_color}; font-weight:bold;">{trend_icon.split(' ')[1]} | {trend_desc}</p></div>""", unsafe_allow_html=True)
        with r_col3:
            st.markdown(f"""<div class="ai-box" style="border: 1px solid #00d4ff;"><h5 style="color:white; margin:0;">🎯 AI 目標價</h5><div style="margin-top:10px; text-align:left;"><div style="display:flex; justify-content:space-between; margin-bottom:5px;"><span style="color:#4ade80;">🚀 短線</span><span style="font-weight:bold; font-size:18px;">${target_short:.2f}</span></div><div style="display:flex; justify-content:space-between; border-top:1px solid #555; padding-top:5px;"><span style="color:#FFD700;">🌊 波段</span><span style="font-weight:bold; font-size:18px;">${target_long:.2f}</span></div></div></div>""", unsafe_allow_html=True)
        
        st.write("")
        
        # 3. 基本面與防守 (含重抓按鈕)
        f_header, f_btn = st.columns([0.8, 0.2])
        with f_header: st.caption("📊 基本面與結構防守")
        with f_btn: 
            if st.button("🔄 重抓"): fetch_fundamental_info.clear(); st.rerun()

        f_col1, f_col2, f_col3 = st.columns(3)
        if info is None: info = {}
        peg = info.get('pegRatio'); fwd_pe = info.get('forwardPE'); rev_growth = info.get('revenueGrowth')
        
        if fwd_pe is not None:
            p_val = f"{fwd_pe:.2f} (Fwd PE)"
            if peg is not None:
                peg_html = f'<div class="val-good">✨ 成長動能強</div>' if peg < 1.0 else '<div class="val-fair">⚖️ 估值合理</div>'
            else:
                peg_html = '<div class="val-good">💰 價格相對便宜</div>' if fwd_pe < 15 else '<div class="val-fair">⚖️ 估值合理區間</div>'
        else:
            p_val = "N/A"
            peg_html = '<div class="val-bad">資料不足</div>'
        
        with f_col1: 
            st.metric("預估本益比 (Fwd PE)", p_val)
            st.markdown(peg_html, unsafe_allow_html=True)
            
        with f_col2:
            if rev_growth is not None: 
                st.metric("成長率", f"{rev_growth*100:.2f}%")
                if rev_growth > 0.2:
                    st.markdown('<div class="val-good">🔥 高成長</div>', unsafe_allow_html=True)
                elif rev_growth > 0:
                    st.markdown('<div class="val-fair">📈 正成長</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="val-bad">📉 衰退中</div>', unsafe_allow_html=True)
            else: 
                st.metric("成長率", "N/A")
                st.caption("無資料")
        
        # FCF
        try:
            t_obj = yf.Ticker(current_ticker)
            cf = t_obj.cash_flow
            if not cf.empty and cf.shape[1] > 1:
                fcf_cur = cf.iloc[0, 0] if 'Free' in str(cf.index) else (cf.loc['Operating Cash Flow'].iloc[0] + cf.loc['Capital Expenditure'].iloc[0])
                fcf_prev = cf.iloc[0, 1] if 'Free' in str(cf.index) else (cf.loc['Operating Cash Flow'].iloc[1] + cf.loc['Capital Expenditure'].iloc[1])
                
                if fcf_prev != 0:
                    fcf_change = ((fcf_cur - fcf_prev) / abs(fcf_prev)) * 100
                    fcf_delta = f"{fcf_change:+.1f}% vs 去年"
                else:
                    fcf_delta = "N/A"
                with f_col3:
                    st.metric("自由現金流", f"${fcf_cur/1e9:.2f}B", fcf_delta)
            elif not cf.empty:
                 fcf_cur = cf.iloc[0, 0]
                 with f_col3:
                     st.metric("自由現金流", f"${fcf_cur/1e9:.2f}B", "無前期數據")
            else:
                with f_col3:
                    st.metric("自由現金流", "N/A")
        except:
            with f_col3:
                st.metric("自由現金流", "資料不足")
        
        st.markdown("---")
        
        # S1/S2 (V13.1 彩色化)
        s_col1, s_col2 = st.columns(2)
        s1_delta = "normal" if latest['Close'] >= s1 else "inverse"
        
        with s_col1: 
            st.metric("🛡️ S1 趨勢 (MA20)", f"${s1:.2f}", delta_color=s1_delta)
            s1_class = "val-good" if latest['Close'] >= s1 else "val-bad"
            st.markdown(f'<div class="{s1_class}">{s1_note}</div>', unsafe_allow_html=True)

        with s_col2: 
            st.metric("🛡️ S2 籌碼 (大量低)", f"${s2:.2f}")
            s2_class = "val-good" if latest['Close'] >= s2 else "val-bad"
            st.markdown(f'<div class="{s2_class}">{s2_note}</div>', unsafe_allow_html=True)

    # [Section 5] 繪圖
    plot_data = df
    if "當沖" in time_opt: plot_data = df.tail(50) 
    elif "日線" in time_opt: plot_data = df.tail(120) 
    elif "週線" in time_opt: plot_data = df.tail(150)

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_width=[0.2, 0.2, 0.6])
    fig.add_trace(go.Candlestick(x=plot_data.index, open=plot_data['Open'], high=plot_data['High'], low=plot_data['Low'], close=plot_data['Close'], name='Price'), row=1, col=1)
    
    # [V13.1 修改] 加入獲利價格與 AI 推演
    for i in range(5, len(plot_data)):
        curr = plot_data.iloc[i]
        prior = plot_data.iloc[i-1]
        prior5 = plot_data.iloc[i-5]
        
        # 1. 偵測主力吸籌
        price_drop = curr['Close'] <= prior5['Close'] * 0.99
        ad_rise = curr['AD_Line'] > prior5['AD_Line']
        is_whale = price_drop and ad_rise and (curr['RSI'] < 60)
        
        # 2. 偵測買入訊號
        is_macd_buy = (curr['MACD'] > curr['Signal_Line']) and (prior['MACD'] <= prior['Signal_Line'])
        is_td_buy_9 = not np.isnan(curr.get('TD_Buy_9', np.nan))
        
        # 3. 標記
        if is_whale:
            stop_loss = curr['Low'] * 0.98
            fig.add_annotation(x=plot_data.index[i], y=curr['Low']*0.97, text=f"🐳主力吸<br>SL:${stop_loss:.1f}", showarrow=True, arrowhead=1, ay=60, row=1, col=1, bgcolor="#6f42c1", bordercolor="white", font=dict(color="white", size=10))
        elif is_macd_buy or is_td_buy_9:
            fig.add_annotation(x=plot_data.index[i], y=curr['Low']*0.98, text=f"BUY<br>${curr['Close']:.1f}", showarrow=True, arrowhead=1, ay=40, row=1, col=1, bgcolor="#28a745", font=dict(color="white", size=9))
        
        # [V13.1] 獲利標籤加入價格
        if curr['High'] >= target_short and prior['High'] < target_short:
             fig.add_annotation(x=plot_data.index[i], y=curr['High']*1.02, text=f"💰達標<br>${curr['Close']:.2f}", showarrow=True, arrowhead=1, ay=-40, row=1, col=1, bgcolor="#ffc107", font=dict(color="black", size=10))
        
        is_macd_sell = (curr['MACD'] < curr['Signal_Line']) and (prior['MACD'] >= prior['Signal_Line']) and curr['MACD'] > 0
        is_td_sell_9 = not np.isnan(curr.get('TD_Sell_9', np.nan))
        if is_macd_sell or is_td_sell_9:
            fig.add_annotation(x=plot_data.index[i], y=curr['High']*1.02, text=f"SELL<br>${curr['Close']:.1f}", showarrow=True, arrowhead=1, ay=-40, row=1, col=1, bgcolor="#dc3545", font=dict(color="white", size=9))

    fig.add_hline(y=s1, line_dash="dash", line_color="#00d4ff", annotation_text=f"MA20支撐 ${s1:.1f}", row=1, col=1)
    fig.add_hline(y=target_short, line_dash="dashdot", line_color="#4ade80", annotation_text=f"目標價 ${target_short:.1f}", row=1, col=1)
    
    for idx, row in plot_data[~np.isnan(plot_data['TD_Buy_9'])].iterrows():
        fig.add_annotation(x=idx, y=row['Low'], text="9", showarrow=False, font=dict(color='#ff6b6b', size=12, weight="bold"), row=1, col=1)
    for idx, row in plot_data[~np.isnan(plot_data['TD_Sell_9'])].iterrows():
        fig.add_annotation(x=idx, y=row['High'], text="9", showarrow=False, font=dict(color='#4a9eff', size=12, weight="bold"), row=1, col=1)

    if 'MACD_Hist' in plot_data.columns:
        colors = ['green' if v >= 0 else 'red' for v in plot_data['MACD_Hist']]
        fig.add_trace(go.Bar(x=plot_data.index, y=plot_data['MACD_Hist'], marker_color=colors, name='MACD Hist'), row=2, col=1)
        fig.add_trace(go.Scatter(x=plot_data.index, y=plot_data['MACD'], line=dict(color='white', width=1), name='DIF'), row=2, col=1)
        fig.add_trace(go.Scatter(x=plot_data.index, y=plot_data['Signal_Line'], line=dict(color='yellow', width=1), name='DEM'), row=2, col=1)
        macd_gold = (plot_data['MACD'] > plot_data['Signal_Line']) & (plot_data['MACD'].shift(1) <= plot_data['Signal_Line'].shift(1))
        macd_dead = (plot_data['MACD'] < plot_data['Signal_Line']) & (plot_data['MACD'].shift(1) >= plot_data['Signal_Line'].shift(1))
        fig.add_trace(go.Scatter(x=plot_data[macd_gold].index, y=plot_data[macd_gold]['MACD'], mode='markers', marker=dict(symbol='triangle-up', size=8, color='#d8b4fe'), name='MACD金叉', showlegend=False), row=2, col=1)
        fig.add_trace(go.Scatter(x=plot_data[macd_dead].index, y=plot_data[macd_dead]['MACD'], mode='markers', marker=dict(symbol='triangle-down', size=8, color='#facc15'), name='MACD死叉', showlegend=False), row=2, col=1)

    if 'DMA_DDD' in plot_data.columns:
        fig.add_trace(go.Scatter(x=plot_data.index, y=plot_data['DMA_DDD'], line=dict(color='#d8b4fe', width=1.5), name='DMA (DDD)'), row=3, col=1)
        fig.add_trace(go.Scatter(x=plot_data.index, y=plot_data['DMA_AMA'], line=dict(color='#facc15', width=1.5), name='AMA (Avg)'), row=3, col=1)
        fig.add_trace(go.Scatter(x=plot_data.index, y=plot_data['DMA_DDD'], fill='tonexty', fillcolor='rgba(216, 180, 254, 0.1)', mode='none', showlegend=False), row=3, col=1)
        dma_gold = (plot_data['DMA_DDD'] > plot_data['DMA_AMA']) & (plot_data['DMA_DDD'].shift(1) <= plot_data['DMA_AMA'].shift(1))
        dma_dead = (plot_data['DMA_DDD'] < plot_data['DMA_AMA']) & (plot_data['DMA_DDD'].shift(1) >= plot_data['DMA_AMA'].shift(1))
        fig.add_trace(go.Scatter(x=plot_data[dma_gold].index, y=plot_data[dma_gold]['DMA_DDD'], mode='markers', marker=dict(symbol='triangle-up', size=8, color='#d8b4fe'), name='DMA金叉', showlegend=False), row=3, col=1)
        fig.add_trace(go.Scatter(x=plot_data[dma_dead].index, y=plot_data[dma_dead]['DMA_DDD'], mode='markers', marker=dict(symbol='triangle-down', size=8, color='#facc15'), name='DMA死叉', showlegend=False), row=3, col=1)

    fig.update_xaxes(tickformat=xaxis_format)
    chart_dragmode = 'pan' if enable_touch else False
    fig.update_layout(height=950, template="plotly_dark", xaxis_rangeslider_visible=False, margin=dict(t=30, b=10, l=10, r=10), dragmode=chart_dragmode, showlegend=False)
    st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': enable_touch, 'displayModeBar': enable_touch})

    st.subheader("🐳 籌碼與主力動向分析")
    chip_col1, chip_col2 = st.columns(2)
    mf = ((plot_data['Close'] - plot_data['Open']) / (plot_data['High'] - plot_data['Low'])) * plot_data['Volume']
    mf = mf.fillna(0); mf_cum = mf.cumsum()

    with chip_col1:
        st.markdown("##### 🏦 主力資金流向 (吸籌/出貨)")
        fig_mf = go.Figure()
        fig_mf.add_trace(go.Scatter(x=plot_data.index, y=mf_cum, fill='tozeroy', mode='lines', line=dict(color='#00d4ff', width=2), name='主力資金'))
        
        if len(mf_cum) > 5:
            trend = mf_cum.iloc[-1] - mf_cum.iloc[-5]
            if trend > 0:
                fig_mf.add_annotation(x=plot_data.index[-1], y=mf_cum.iloc[-1], text="🟢 主力吸籌", showarrow=True, arrowhead=1, bgcolor="#1b3a1b", font=dict(color="#4ade80"))
            else:
                fig_mf.add_annotation(x=plot_data.index[-1], y=mf_cum.iloc[-1], text="🔴 主力出貨", showarrow=True, arrowhead=1, bgcolor="#3a1b1b", font=dict(color="#ff6b6b"))

        fig_mf.update_layout(height=350, template="plotly_dark", margin=dict(l=10, r=10, t=30, b=10), showlegend=False)
        st.plotly_chart(fig_mf, use_container_width=True, config={'staticPlot': True})

    with chip_col2:
        st.markdown("##### 👥 主力 vs 散戶 持股成本")
        total_profile = calculate_volume_profile(plot_data, bins=40)
        inst_mask = mf > 0
        inst_profile = calculate_volume_profile(plot_data, bins=40, filter_mask=inst_mask)
        
        fig_vp = go.Figure()
        if not total_profile.empty:
            fig_vp.add_trace(go.Scatter(x=total_profile['Price'], y=total_profile['Volume'], fill='tozeroy', mode='lines', line=dict(color='#ffaa00', width=0), name='整體'))
        if not inst_profile.empty:
            fig_vp.add_trace(go.Scatter(x=inst_profile['Price'], y=inst_profile['Volume'], fill='tozeroy', mode='lines', line=dict(color='#00d4ff', width=2), name='主力'))
            
        fig_vp.add_vline(x=latest['Close'], line_dash="dash", line_color="white", annotation_text="現價")
        fig_vp.update_layout(height=350, template="plotly_dark", margin=dict(l=10, r=10, t=30, b=10), showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig_vp, use_container_width=True, config={'staticPlot': True})
        
    # --- [V13.4] 新聞情緒與未來推演 (雙軌錨定) ---
    st.markdown("---")
    st.subheader(f"📰 {current_ticker} 新聞情緒與未來推演")
    
    # 1. 獲取宏觀風向
    macro_text, macro_status = get_macro_environment()
    
    # 2. 獲取個股新聞與分數
    news_items = fetch_google_news_rss(current_ticker, is_macro=False)
    news_score = 0
    l3_found = False
    
    for n in news_items:
        s, is_l3 = analyze_news_weight(n['title'])
        news_score += s
        if is_l3: 
            l3_found = True
            save_anchor(ls, current_ticker, n['title'], 3.0) # 寫入記憶
    
    # 3. 讀取錨定記憶
    anchor_data = get_valid_anchor(ls, current_ticker)
    if anchor_data:
        news_score += anchor_data['score'] # 加上記憶分數
    
    # 4. 綜合推演邏輯
    prediction_text = "盤整觀望"
    pred_color = "gray"
    
    # 技術面得分
    tech_score = 0
    if strat_signals['MACD_Color'] == 'tag-green': tech_score += 1
    if strat_signals['Vol_Color'] == 'tag-green': tech_score += 1
    if latest['Close'] > s1: tech_score += 1
    
    # 宏觀折扣
    if macro_status == "bad": 
        news_score -= 2 # 強制扣分
        tech_score -= 1
        
    final_score = tech_score + news_score
    
    if final_score >= 4:
        prediction_text = f"🚀 強力看漲！實質利多支撐 (Level 3證據) + 技術面強勢，目標 ${target_long:.2f}"
        pred_color = "#4ade80"
    elif final_score >= 1.5:
        prediction_text = f"📈 偏多操作。基本面有撐，但留意宏觀風險，目標先看 ${target_short:.2f}"
        pred_color = "#ffc107"
    else:
        prediction_text = f"🌧️ 建議觀望。缺乏實質利多或受宏觀({macro_text})壓抑，勿追高。"
        pred_color = "#ff6b6b"

    n_col1, n_col2 = st.columns([0.4, 0.6])
    
    with n_col1:
        # 顯示宏觀警示
        if macro_status == "bad":
            st.markdown(f'<div class="macro-alert">{macro_text}：市場風險高，個股評價已下修</div>', unsafe_allow_html=True)
        elif macro_status == "fair":
            st.markdown(f'<div style="color:orange; margin-bottom:10px;">{macro_text}</div>', unsafe_allow_html=True)
            
        # 顯示記憶錨定
        if anchor_data:
            st.markdown(f"""
            <div class="anchor-box">
                <b>⚓ 記憶錨定 (有效)</b><br>
                {anchor_data['date']} 偵測到重大訊號：<br>
                "{anchor_data['title']}" (+3.0分)
            </div>
            """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="ai-box" style="text-align:left;">
            <h4 style="color:white; margin:0;">🔮 AI 綜合推演</h4>
            <div style="font-size:18px; color:{pred_color}; font-weight:bold; margin-top:10px;">{prediction_text}</div>
            <hr style="border-color:#555;">
            <div style="font-size:14px; color:#ccc;">
                <b>技術面得分：</b> {tech_score}/3<br>
                <b>基本面得分：</b> {news_score:.1f} (含記憶與權重)
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with n_col2:
        st.markdown("##### 🗞️ 實質證據快篩 (Google News)")
        if news_items:
            for n in news_items:
                s, is_l3 = analyze_news_weight(n['title'])
                score_icon = "🔥" if is_l3 else ("🔴" if s < 0 else ("🟢" if s > 0 else "⚪"))
                border_style = "border-left: 3px solid #00d4ff;" if is_l3 else "border-left: 3px solid #555;"
                st.markdown(f"""<div class="news-card" style="{border_style}"><a href="{n['link']}" target="_blank" class="news-link">{score_icon} <b>{n['date']}</b> | {n['title']}</a></div>""", unsafe_allow_html=True)
        else:
            st.info("暫無相關新聞資料 (可能是 API 連線限制)")

except Exception as e:
    st.error(f"系統錯誤 (請稍後再試或檢查網路): {e}")
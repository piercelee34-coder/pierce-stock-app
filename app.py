import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os

# --- 0. 系統設定 ---
st.set_page_config(page_title="AI 實戰戰情室 V11.9 (操控與視覺整合版)", layout="wide", page_icon="💎")

# --- CSS 美化 ---
st.markdown("""
<style>
    .price-card {background-color: #1e1e1e; padding: 20px; border-radius: 10px; text-align: center; border: 1px solid #333;}
    .ai-box {background-color: #333; padding: 10px; border-radius: 10px; border: 1px solid #555; text-align: center;}
    .val-good {color: #28a745; font-weight: bold; font-size: 14px;}
    .val-fair {color: #ffc107; font-weight: bold; font-size: 14px;}
    .val-bad {color: #dc3545; font-weight: bold; font-size: 14px;}
    .buy-hint {background-color: #1b3a1b; color: #4ade80; padding: 5px 10px; border-radius: 5px; font-size: 16px; margin-top: 10px; display: inline-block;}
    
    .radar-grid {display: grid; grid-template-columns: 1fr; gap: 8px; text-align: left; font-size: 14px; margin-top: 10px;}
    .radar-item {padding: 4px 0; border-bottom: 1px solid #444; display: flex; justify-content: space-between; align-items: center;}
    .signal-tag {font-weight: bold; padding: 2px 8px; border-radius: 4px; font-size: 13px; display: inline-block;}
    
    .tag-red {background-color: #3a1b1b; color: #ff6b6b; border: 1px solid #dc3545;}
    .tag-green {background-color: #1b3a1b; color: #4ade80; border: 1px solid #28a745;}
    .tag-orange {background-color: #4a3b1b; color: #ffaa00; border: 1px solid #ffc107;}
    .tag-gray {background-color: #333; color: #ccc; border: 1px solid #666;}
    .tag-purple {background-color: #4a1b4a; color: #d8b4fe; border: 1px solid #a855f7;} 
    .tag-blue {background-color: #1b3a4a; color: #4a9eff; border: 1px solid #00d4ff;}

    .stButton>button {width: 100%; border-radius: 5px;}
    .guide-box {background-color: #262730; padding: 15px; border-radius: 5px; border-left: 4px solid #00d4ff; font-size: 14px; line-height: 1.6;}
    
    /* 標題右側說明樣式 */
    .header-legend {
        text-align: right; 
        font-size: 14px; 
        padding-top: 20px;
    }
</style>
""", unsafe_allow_html=True)

# --- 1. 自選股儲存系統 ---
WATCHLIST_FILE = 'watchlist.json'

def load_watchlist():
    default_list = ['NVDA', 'TSM', 'AAPL', '^IXIC', '0050.TW', '2330.TW']
    if os.path.exists(WATCHLIST_FILE):
        try:
            with open(WATCHLIST_FILE, 'r') as f:
                saved_list = json.load(f)
                return saved_list if saved_list else default_list
        except: return default_list
    return default_list

def save_watchlist(watchlist):
    with open(WATCHLIST_FILE, 'w') as f:
        json.dump(watchlist, f)

if 'watchlist' not in st.session_state:
    st.session_state.watchlist = load_watchlist()

# --- 2. 核心函數 ---

def calculate_volume_profile(df, bins=40, filter_mask=None):
    if df.empty: return pd.DataFrame({'Price': [], 'Volume': []})
    price_min = df['Low'].min()
    price_max = df['High'].max()
    if price_min == price_max: return pd.DataFrame({'Price': [], 'Volume': []})
    
    bin_edges = np.linspace(price_min, price_max, bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    target_df = df if filter_mask is None else df[filter_mask]
    if target_df.empty: return pd.DataFrame({'Price': bin_centers, 'Volume': np.zeros(bins)})
    
    bin_indices = pd.cut(target_df['Close'], bins=bin_edges, labels=False, include_lowest=True)
    profile_series = target_df.groupby(bin_indices)['Volume'].sum()
    profile_series = profile_series.reindex(range(bins), fill_value=0)
    
    return pd.DataFrame({'Price': bin_centers, 'Volume': profile_series.values})

def calculate_dma(df, short_period=10, long_period=50, signal_period=10):
    df['DMA_DDD'] = df['Close'].rolling(window=short_period).mean() - df['Close'].rolling(window=long_period).mean()
    df['DMA_AMA'] = df['DMA_DDD'].rolling(window=signal_period).mean()
    return df

def calculate_td_sequential(df):
    if len(df) < 5: return df
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
    df = calculate_dma(df)
    df = calculate_td_sequential(df)
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
    if current_price < key_bar_low: s2 = recent_60['Low'].min(); s2_note = f"主力籌碼區({s2_date_str})已失守，退守地板"
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
        if macd > 0: macd_text, macd_color = "零軸上金叉 (多頭市場)", "tag-green"
        else: macd_text, macd_color = "零軸下金叉 (跌深反彈)", "tag-orange"
    else:
        if macd > 0: macd_text, macd_color = "零軸上死叉 (由多轉空)", "tag-orange"
        else: macd_text, macd_color = "零軸下死叉 (賣出訊號)", "tag-red"
    vol = latest['Volume']; vol_ma = latest['Vol_SMA5']
    if vol > vol_ma * 1.5: vol_text, vol_color = "爆量 (>1.5倍)", "tag-green"
    elif vol > vol_ma * 1.1: vol_text, vol_color = "量增 (>1.1倍)", "tag-green"
    else: vol_text, vol_color = "量縮/平量", "tag-gray"
    rsi = latest['RSI']
    if rsi > 70: rsi_text, rsi_color = f"過熱 ({rsi:.0f})", "tag-red"
    elif rsi < 30: rsi_text, rsi_color = f"超賣 ({rsi:.0f})", "tag-green"
    else: rsi_text, rsi_color = f"中性 ({rsi:.0f})", "tag-gray"
    recent_20 = df.tail(20)
    box_range = (recent_20['High'].max() - recent_20['Low'].min()) / recent_20['Low'].min() * 100
    if box_range < 8: trend_text, trend_color = f"⚠️ 盤整 (波幅{box_range:.1f}%)", "tag-orange"
    else: trend_text, trend_color = "🌊 趨勢盤", "tag-green"
    summary = "觀望"; summary_color = "tag-gray"
    td_buy_stop = latest.get('TD_Buy_Stop', np.nan)
    is_td_buy_9 = not np.isnan(latest.get('TD_Buy_9', np.nan))
    is_td_sell_9 = not np.isnan(latest.get('TD_Sell_9', np.nan))
    dma_gold = (latest['DMA_DDD'] > latest['DMA_AMA']) and (prev['DMA_DDD'] <= prev['DMA_AMA'])
    if is_td_sell_9: summary, summary_color = "🔺 九轉賣點 (藍9力竭)", "tag-blue"
    elif is_td_buy_9: summary, summary_color = f"🔻 九轉買點 (紅9力竭, 停損{td_buy_stop:.2f})", "tag-purple"
    elif dma_gold: summary, summary_color = "🚀 DMA 金叉翻多", "tag-green"
    elif "盤整" in trend_text: summary, summary_color = "盤整陷阱", "tag-orange"
    elif macd > signal:
        if "tag-green" in vol_color: summary, summary_color = "🚀 放量攻擊", "tag-green"
        else: summary, summary_color = "📈 偏多震盪", "tag-green"
    else:
        if rsi > 60: summary, summary_color = "🌧️ 拉回修正", "tag-orange"
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
    if not np.isnan(latest.get('TD_Sell_9', np.nan)): return "🔺 出現藍色9 (上漲力竭)，注意獲利了結"
    td_buy_stop = latest.get('TD_Buy_Stop', np.nan)
    if not np.isnan(latest.get('TD_Buy_9', np.nan)): return f"🔻 出現紅色9 (下跌力竭)，潛在買點，停損設 {td_buy_stop:.2f}"
    if (latest['DMA_DDD'] > latest['DMA_AMA']) and (prev['DMA_DDD'] <= prev['DMA_AMA']): return "🚀 DMA 黃金交叉，中線轉多訊號"
    divergence = detect_smart_money_divergence(df)
    if divergence: return f"🚨 {divergence}！機會難得"
    hints = []
    if abs(current_price - s1) / current_price < 0.015 and current_price > s1: hints.append("回測月線有撐")
    elif current_price < s1 and current_price > s2: hints.append("等待回測 S2")
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
    
    # 上移/下移
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
            
    # [V11.9 新增] 置頂/置底
    c_top, c_bottom = st.columns(2)
    if c_top.button("⏫ 置頂") and current_ticker in st.session_state.watchlist:
        st.session_state.watchlist.remove(current_ticker)
        st.session_state.watchlist.insert(0, current_ticker)
        save_watchlist(st.session_state.watchlist); st.rerun()
    if c_bottom.button("⏬ 置底") and current_ticker in st.session_state.watchlist:
        st.session_state.watchlist.remove(current_ticker)
        st.session_state.watchlist.append(current_ticker)
        save_watchlist(st.session_state.watchlist); st.rerun()

    with st.expander("編輯清單"):
        new_t = st.text_input("輸入代號", placeholder="MSTR").upper()
        c1, c2 = st.columns(2)
        if c1.button("➕ 新增"):
            if new_t and new_t not in st.session_state.watchlist:
                st.session_state.watchlist.append(new_t)
                save_watchlist(st.session_state.watchlist); st.rerun()
        if c2.button("❌ 刪除"):
            if current_ticker in st.session_state.watchlist:
                st.session_state.watchlist.remove(current_ticker)
                save_watchlist(st.session_state.watchlist); st.rerun()

# --- 4. 主程式 ---
# [V11.9 修改] 週期選單移至主標題右側
top_col1, top_col2 = st.columns([0.65, 0.35])

with top_col1:
    st.title(f"📈 {current_ticker} 實戰戰情室 V11.9")
    
with top_col2:
    st.write("") # Spacer
    st.write("") 
    # [V11.9] 週期選單
    time_opt = st.radio("週期", ["當沖 (分時)", "日線 (Daily)", "週線 (Weekly)", "月線 (長線)"], 
                        index=1, horizontal=True, label_visibility="collapsed")

api_period = "1y"; api_interval = "1d"; xaxis_format = "%Y-%m-%d"
if "當沖" in time_opt: api_period = "5d"; api_interval = "15m"; xaxis_format = "%H:%M" 
elif "日線" in time_opt: api_period = "6mo"; api_interval = "1d"; xaxis_format = "%m-%d" 
elif "週線" in time_opt: api_period = "2y"; api_interval = "1wk"; xaxis_format = "%Y-%m-%d"
elif "月線" in time_opt: api_period = "2y"; api_interval = "1wk"; xaxis_format = "%Y-%m"

@st.cache_data(ttl=300)
def fetch_main_data(ticker, period, interval):
    try: return yf.download(ticker, period=period, interval=interval, progress=False, threads=False)
    except Exception: return pd.DataFrame()

@st.cache_data(ttl=3600)
def fetch_fundamental_info(ticker):
    try:
        t = yf.Ticker(ticker)
        info = t.info
        if not info or len(info) < 5: return None
        return info
    except Exception: return None

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
    
    st.markdown(f"""
    <div class="price-card">
        <h1 style="margin:0; font-size: 50px;">${latest['Close']:.2f}</h1>
        <h3 style="margin:0; color: {color_price};">{pct_change:+.2f}%</h3>
        <p style="color: gray; margin-bottom: 5px;">最新成交量: {format_volume(latest['Volume'])}</p>
        <div class="buy-hint">💡 操作提示: {buy_hint_text}</div>
    </div>
    """, unsafe_allow_html=True)
    st.write("")

    st.subheader("🚀 戰略雷達與 AI 預測")
    m_col1, m_col2, m_col3 = st.columns(3)

    with m_col1:
        st.markdown(f"""
        <div class="ai-box">
            <h5 style="color:white; margin:0; margin-bottom:5px;">📡 綜合戰略雷達</h5>
            <div class="signal-tag {strat_signals['Summary_Color']}" style="font-size:16px;">{strat_signals['Summary']}</div>
            <div class="radar-grid">
                <div class="radar-item">
                    <span>1. MACD</span><span class="signal-tag {strat_signals['MACD_Color']}">{strat_signals['MACD_Text']}</span>
                </div>
                <div class="radar-item">
                    <span>2. 成交量</span><span class="signal-tag {strat_signals['Vol_Color']}">{strat_signals['Vol_Text']}</span>
                </div>
                <div class="radar-item">
                    <span>3. RSI</span><span class="signal-tag {strat_signals['RSI_Color']}">{strat_signals['RSI_Text']}</span>
                </div>
                <div class="radar-item">
                    <span>4. 盤整</span><span class="signal-tag {strat_signals['Trend_Color']}">{strat_signals['Trend_Text']}</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with m_col2:
        st.markdown(f"""
        <div class="ai-box">
            <h5 style="color:white; margin:0;">⚖️ 市場格局 & 評級</h5>
            <div style="font-size: 30px; margin-top:5px;">{trend_icon.split(' ')[0]} <span style="font-size:20px; color:#FFD700;">{ai_rating}</span></div>
            <p style="font-size:12px; color:#ccc;">{trend_icon.split(' ')[1]} | {trend_desc}</p>
        </div>
        """, unsafe_allow_html=True)

    with m_col3:
        st.markdown(f"""
        <div class="ai-box" style="border: 1px solid #00d4ff;">
            <h5 style="color:white; margin:0;">🎯 AI 雙軌目標價</h5>
            <div style="margin-top:10px; text-align:left;">
                <div style="display:flex; justify-content:space-between; margin-bottom:5px;">
                    <span style="color:#4ade80;">🚀 短線 (布林)</span><span style="font-weight:bold; font-size:18px;">${target_short:.2f}</span>
                </div>
                <div style="display:flex; justify-content:space-between; border-top:1px solid #555; padding-top:5px;">
                    <span style="color:#FFD700;">🌊 中長 (波段)</span><span style="font-weight:bold; font-size:18px;">${target_long:.2f}</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")

    col_header, col_btn = st.columns([0.85, 0.15])
    with col_header: st.subheader("📊 基本面與結構防守")
    with col_btn:
        if st.button("🔄 重抓基本面"):
            fetch_fundamental_info.clear()
            st.rerun()

    f_col1, f_col2, f_col3, f_col4, f_col5 = st.columns(5)
    
    if info is None: info = {}
    peg = info.get('pegRatio')
    fwd_pe = info.get('forwardPE')
    rev_growth = info.get('revenueGrowth') or info.get('quarterlyRevenueGrowth') or info.get('earningsGrowth')
    
    if fwd_pe is not None:
        p_val = f"{fwd_pe:.2f} (Fwd PE)"
        if peg is not None:
            if peg < 1.0: peg_html = f'<div class="val-good">✨ 成長動能強 (PEG < 1)</div>'
            elif peg > 1.5: peg_html = f'<div class="val-bad">⚠️ 溢價偏高 (PEG > 1.5)</div>'
            else: peg_html = f'<div class="val-fair">⚖️ 估值合理 (PEG 1~1.5)</div>'
        else:
            if fwd_pe < 15: peg_html = '<div class="val-good">💰 價格相對便宜</div>'
            elif fwd_pe > 30: peg_html = '<div class="val-bad">🔥 市場預期極高</div>'
            else: peg_html = '<div class="val-fair">⚖️ 估值合理區間</div>'
    elif peg is not None:
        p_val = f"{peg} (PEG)"
        peg_html = '<div class="val-fair">暫無 PE，僅參考 PEG</div>'
    else:
        p_val = "N/A"
        peg_html = '<div class="val-bad">資料不足</div>'
    
    with f_col1: st.metric("預估本益比 (Fwd PE)", p_val); st.markdown(peg_html, unsafe_allow_html=True)

    with f_col2:
        if rev_growth is not None:
            st.metric("成長率", f"{rev_growth*100:.2f}%")
            if rev_growth > 0.2: st.markdown('<div class="val-good">🔥 高成長</div>', unsafe_allow_html=True)
            else: st.markdown('<div class="val-fair">📈 正成長</div>', unsafe_allow_html=True)
        else: st.metric("成長率", "N/A"); st.caption("無資料")
    
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

    s1_delta = "normal"
    if latest['Close'] < s1: s1_delta = "inverse"
    with f_col4: st.metric("🛡️ S1 趨勢 (MA20)", f"${s1:.2f}", delta_color=s1_delta); st.caption(s1_note)
    with f_col5: st.metric("🛡️ S2 籌碼 (大量低)", f"${s2:.2f}"); st.caption(s2_note)

    # [V11.9 修改] 走勢圖標題與右側說明 (因為週期選單移至頂部，這裡只留標題)
    t_col1, t_col2 = st.columns([0.65, 0.35])
    with t_col1:
        st.subheader(f"📈 走勢圖 - {time_opt} (含九轉/DMA)")
    with t_col2:
        st.markdown("""
            <div class="header-legend">
                <span style="color:#ff6b6b; font-weight:bold; margin-right:15px;">▼ 紅9: 下跌力竭 (潛在買點)</span>
                <span style="color:#4a9eff; font-weight:bold;">▲ 藍9: 上漲力竭 (潛在賣點)</span>
            </div>
        """, unsafe_allow_html=True)

    plot_data = df
    if "當沖" in time_opt: plot_data = df.tail(50) 
    elif "日線" in time_opt: plot_data = df.tail(120) 
    elif "週線" in time_opt: plot_data = df.tail(150)

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_width=[0.2, 0.2, 0.6])
    fig.add_trace(go.Candlestick(x=plot_data.index, open=plot_data['Open'], high=plot_data['High'], low=plot_data['Low'], close=plot_data['Close'], name='Price'), row=1, col=1)
    
    for i in range(1, len(plot_data)):
        curr = plot_data.iloc[i]; prior = plot_data.iloc[i-1]
        date_str = plot_data.index[i].strftime('%m/%d')
        is_macd_buy = (curr['MACD'] > curr['Signal_Line']) and (prior['MACD'] <= prior['Signal_Line'])
        is_rsi_buy = (curr['RSI'] < 30) and (prior['RSI'] >= 30)
        is_td_buy_9 = not np.isnan(curr.get('TD_Buy_9', np.nan))
        
        if is_macd_buy or is_rsi_buy or is_td_buy_9:
            fig.add_annotation(x=plot_data.index[i], y=curr['Low']*0.98, text=f"BUY<br>{date_str}<br>${curr['Close']:.2f}", showarrow=True, arrowhead=1, ay=50, row=1, col=1, bgcolor="#28a745", font=dict(color="white", size=10))
        
        is_macd_sell = (curr['MACD'] < curr['Signal_Line']) and (prior['MACD'] >= prior['Signal_Line']) and curr['MACD'] > 0
        is_rsi_sell = (curr['RSI'] > 75) and (prior['RSI'] <= 75)
        is_td_sell_9 = not np.isnan(curr.get('TD_Sell_9', np.nan))
        
        if is_macd_sell or is_rsi_sell or is_td_sell_9:
            fig.add_annotation(x=plot_data.index[i], y=curr['High']*1.02, text=f"SELL<br>{date_str}<br>${curr['Close']:.2f}", showarrow=True, arrowhead=1, ay=-50, row=1, col=1, bgcolor="#dc3545", font=dict(color="white", size=10))

    fig.add_hline(y=s1, line_dash="dash", line_color="#00d4ff", annotation_text=f"S1 (MA20)", row=1, col=1)
    fig.add_hline(y=s2, line_dash="dot", line_color="orange", annotation_text=f"S2 (Key Bar)", row=1, col=1)
    fig.add_hline(y=target_short, line_dash="dashdot", line_color="#4ade80", annotation_text=f"Target 1", row=1, col=1)
    
    for idx, row in plot_data[~np.isnan(plot_data['TD_Buy_9'])].iterrows():
        date_str = idx.strftime('%m/%d')
        fig.add_annotation(x=idx, y=row['Low'], text=f"<b>紅9</b><br>{date_str}<br>${row['Close']:.2f}", showarrow=True, arrowhead=1, ay=70, arrowcolor='#ff6b6b', bgcolor="#3a1b1b", bordercolor="#ff6b6b", font=dict(color='#ff6b6b', size=11), row=1, col=1)

    for idx, row in plot_data[~np.isnan(plot_data['TD_Sell_9'])].iterrows():
        date_str = idx.strftime('%m/%d')
        fig.add_annotation(x=idx, y=row['High'], text=f"<b>藍9</b><br>{date_str}<br>${row['Close']:.2f}", showarrow=True, arrowhead=1, ay=-70, arrowcolor='#4a9eff', bgcolor="#1b3a4a", bordercolor="#4a9eff", font=dict(color='#4a9eff', size=11), row=1, col=1)

    if 'MACD_Hist' in plot_data.columns:
        colors = ['green' if v >= 0 else 'red' for v in plot_data['MACD_Hist']]
        fig.add_trace(go.Bar(x=plot_data.index, y=plot_data['MACD_Hist'], marker_color=colors, name='MACD Hist'), row=2, col=1)
        fig.add_trace(go.Scatter(x=plot_data.index, y=plot_data['MACD'], line=dict(color='white', width=1), name='DIF'), row=2, col=1)
        fig.add_trace(go.Scatter(x=plot_data.index, y=plot_data['Signal_Line'], line=dict(color='yellow', width=1), name='DEM'), row=2, col=1)
        
    if 'DMA_DDD' in plot_data.columns:
        fig.add_trace(go.Scatter(x=plot_data.index, y=plot_data['DMA_DDD'], line=dict(color='#d8b4fe', width=1.5), name='DMA (DDD)'), row=3, col=1)
        fig.add_trace(go.Scatter(x=plot_data.index, y=plot_data['DMA_AMA'], line=dict(color='#facc15', width=1.5), name='AMA (Avg)'), row=3, col=1)
        fig.add_trace(go.Scatter(x=plot_data.index, y=plot_data['DMA_DDD'], fill='tonexty', fillcolor='rgba(216, 180, 254, 0.1)', mode='none', showlegend=False), row=3, col=1)

    fig.update_xaxes(tickformat=xaxis_format)
    fig.update_layout(height=700, template="plotly_dark", xaxis_rangeslider_visible=False, margin=dict(t=30, b=10, r=220), dragmode='zoom')
    st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True})

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
        st.plotly_chart(fig_mf, use_container_width=True)

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
        st.plotly_chart(fig_vp, use_container_width=True)
        st.markdown("""<div class="guide-box"><b>🧐 說明：</b><br>🟡 黃色山峰 = 散戶套牢區<br>🔵 青色山峰 = 主力成本區<br>若現價 > 青色山峰 👉 主力獲利 (強支撐)<br>若現價 < 青色山峰 👉 主力套牢 (強壓力)</div>""", unsafe_allow_html=True)

except Exception as e:
    st.error(f"系統錯誤 (請稍後再試或檢查網路): {e}")
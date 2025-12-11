import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
import time

# --- 0. 系統設定 ---
st.set_page_config(page_title="AI 實戰戰情室 V10.1 (戰略訊號圖解版)", layout="wide", page_icon="💎")

# --- CSS 美化 ---
st.markdown("""
<style>
    .big-alert {padding: 15px; border-radius: 10px; margin-bottom: 20px; text-align: center; font-size: 20px; font-weight: bold; color: white;}
    .price-card {background-color: #1e1e1e; padding: 20px; border-radius: 10px; text-align: center; border: 1px solid #333;}
    .ai-box {background-color: #333; padding: 10px; border-radius: 10px; border: 1px solid #555; text-align: center;}
    .trend-box {background-color: #2b2b2b; padding: 15px; border-radius: 10px; border-left: 5px solid #FFD700; margin-top: 10px; margin-bottom: 10px;}
    .val-good {color: #28a745; font-weight: bold; font-size: 14px;}
    .val-fair {color: #ffc107; font-weight: bold; font-size: 14px;}
    .val-bad {color: #dc3545; font-weight: bold; font-size: 14px;}
    .buy-hint {background-color: #1b3a1b; color: #4ade80; padding: 5px 10px; border-radius: 5px; font-size: 16px; margin-top: 10px; display: inline-block;}
    .stButton>button {width: 100%; border-radius: 5px;}
    .guide-box {background-color: #262730; padding: 15px; border-radius: 5px; border-left: 4px solid #00d4ff; font-size: 14px; line-height: 1.6;}
    
    /* V10.1 戰略雷達樣式優化 */
    .radar-grid {display: grid; grid-template-columns: 1fr; gap: 8px; text-align: left; font-size: 14px; margin-top: 10px;}
    .radar-item {padding: 4px 0; border-bottom: 1px solid #444; display: flex; justify-content: space-between; align-items: center;}
    .signal-tag {font-weight: bold; padding: 2px 8px; border-radius: 4px; font-size: 13px; display: inline-block;}
    
    /* 燈號顏色定義 */
    .tag-red {background-color: #3a1b1b; color: #ff6b6b; border: 1px solid #dc3545;}     /* 危險/看空 */
    .tag-green {background-color: #1b3a1b; color: #4ade80; border: 1px solid #28a745;}   /* 安全/看多 */
    .tag-orange {background-color: #4a3b1b; color: #ffaa00; border: 1px solid #ffc107;}  /* 警戒/轉弱 */
    .tag-gray {background-color: #333; color: #ccc; border: 1px solid #666;}             /* 中性/觀望 */
</style>
""", unsafe_allow_html=True)

# --- 1. 自選股儲存系統 (JSON) ---
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
def calculate_indicators(df):
    if len(df) < 20: return df
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_60'] = df['Close'].rolling(window=60).mean()
    df['Vol_SMA5'] = df['Volume'].rolling(window=5).mean()
    
    # 布林通道 (用於目標價)
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
    return df

def find_support_levels(df, current_price):
    if df.empty or len(df) < 60:
        return current_price, current_price, "資料不足", "資料不足"

    s1 = df['Close'].rolling(window=20).mean().iloc[-1]
    if current_price > s1:
        dist = (current_price - s1) / s1 * 100
        s1_note = f"股價在月線之上 {dist:.1f}% (趨勢多)"
    else:
        dist = (s1 - current_price) / s1 * 100
        s1_note = f"已跌破月線 {dist:.1f}% (趨勢轉弱)"

    recent_60 = df.tail(60)
    max_vol_date = recent_60['Volume'].idxmax()
    key_bar_low = df.loc[max_vol_date]['Low']
    floor_price = recent_60['Low'].min()
    s2_date_str = max_vol_date.strftime('%m/%d')
    
    if current_price < key_bar_low:
        s2 = floor_price
        s2_note = f"主力籌碼區({s2_date_str})已失守，退守地板"
    else:
        s2 = key_bar_low
        dist_s2 = (current_price - s2) / current_price * 100
        s2_note = f"最大量日({s2_date_str})低點，距現價 {dist_s2:.1f}%"

    return s1, s2, s1_note, s2_note

# [V10.1 核心更新] 戰略訊號圖解化 (加入詳細註解與顏色)
def analyze_strategic_signals(df):
    if df.empty: return {}
    
    latest = df.iloc[-1]
    
    # 1. MACD 狀態 (加入易懂註解)
    macd = latest['MACD']
    signal = latest['Signal_Line']
    
    if macd > signal: # 金叉
        if macd > 0:
            macd_text = "零軸上金叉 (多頭市場)"
            macd_color = "tag-green"
        else:
            macd_text = "零軸下金叉 (跌深反彈)"
            macd_color = "tag-orange" # 反彈視為警戒或中性偏多
    else: # 死叉
        if macd > 0:
            macd_text = "零軸上死叉 (由多轉空)"
            macd_color = "tag-orange"
        else:
            macd_text = "零軸下死叉 (賣出訊號)"
            macd_color = "tag-red"
    
    # 2. 成交量狀態 (加入顏色)
    vol = latest['Volume']
    vol_ma = latest['Vol_SMA5']
    if vol > vol_ma * 1.5:
        vol_text = "爆量 (>1.5倍)"
        vol_color = "tag-green" # 動能強
    elif vol > vol_ma * 1.1:
        vol_text = "量增 (>1.1倍)"
        vol_color = "tag-green"
    else:
        vol_text = "量縮/平量"
        vol_color = "tag-gray"
    
    # 3. RSI 狀態 (加入顏色)
    rsi = latest['RSI']
    if rsi > 70:
        rsi_text = f"過熱 ({rsi:.0f})"
        rsi_color = "tag-red" # 警戒
    elif rsi < 30:
        rsi_text = f"超賣 ({rsi:.0f})"
        rsi_color = "tag-green" # 買點
    else:
        rsi_text = f"中性 ({rsi:.0f})"
        rsi_color = "tag-gray"
    
    # 4. 盤整期判斷
    recent_20 = df.tail(20)
    high_20 = recent_20['High'].max()
    low_20 = recent_20['Low'].min()
    box_range = (high_20 - low_20) / low_20 * 100
    
    is_consolidating = False
    trend_text = "🌊 趨勢盤"
    trend_color = "tag-green"
    
    if box_range < 8:
        is_consolidating = True
        trend_text = f"⚠️ 盤整 (波幅{box_range:.1f}%)"
        trend_color = "tag-orange"
    
    # 綜合評語
    summary = "觀望"
    summary_color = "tag-gray"
    
    if is_consolidating:
        summary = "盤整陷阱"
        summary_color = "tag-orange"
    elif macd > signal: # 金叉狀態
        if "tag-green" in vol_color:
            summary = "🚀 放量攻擊"
            summary_color = "tag-green"
        else:
            summary = "📈 偏多震盪"
            summary_color = "tag-green"
    else: # 死叉狀態
        if rsi > 60:
            summary = "🌧️ 拉回修正"
            summary_color = "tag-orange"
        else:
            summary = "⛈️ 空頭走勢"
            summary_color = "tag-red"

    return {
        "MACD_Text": macd_text, "MACD_Color": macd_color,
        "Vol_Text": vol_text, "Vol_Color": vol_color,
        "RSI_Text": rsi_text, "RSI_Color": rsi_color,
        "Trend_Text": trend_text, "Trend_Color": trend_color,
        "Summary": summary, "Summary_Color": summary_color
    }

def analyze_market_trend(df):
    price = df['Close'].iloc[-1]
    ma20 = df['SMA_20'].iloc[-1]
    ma60 = df['SMA_60'].iloc[-1]
    
    if price > ma20 and ma20 > ma60:
        return "🐂 牛市 (Bull)", "多頭排列，順勢操作"
    elif price < ma20 and ma20 < ma60:
        return "🐻 熊市 (Bear)", "空頭排列，保守為宜"
    else:
        return "⚖️ 震盪 (Range)", "區間整理，高出低進"

def predict_target_and_rating(df):
    price = df['Close'].iloc[-1]
    ma20 = df['SMA_20'].iloc[-1]
    macd = df['MACD'].iloc[-1]
    signal = df['Signal_Line'].iloc[-1]
    rsi = df['RSI'].iloc[-1]
    upper_band = df['Bollinger_Upper'].iloc[-1]
    
    score = 0
    if price > ma20: score += 1
    if macd > signal: score += 1
    if rsi < 70 and rsi > 40: score += 1
    if df['Volume'].iloc[-1] > df['Vol_SMA5'].iloc[-1]: score += 1
    
    if score >= 3: rating = "💪 強力買進"
    elif score == 2: rating = "✊ 持有/續抱"
    else: rating = "✋ 觀望/賣出"
    
    if price > upper_band:
        target = price * 1.05
    else:
        target = upper_band
        
    return target, rating

def generate_buy_hint(df, current_price, s1, s2):
    if df.empty: return "無資料"
    rsi = df['RSI'].iloc[-1]
    macd = df['MACD'].iloc[-1]
    signal = df['Signal_Line'].iloc[-1]
    
    hints = []
    if abs(current_price - s1) / current_price < 0.015 and current_price > s1:
        hints.append("回測月線有撐")
    elif current_price < s1 and current_price > s2:
        hints.append("等待回測 S2")
    if abs(current_price - s2) / current_price < 0.02:
        hints.append("近主力成本區")
    if rsi < 30: hints.append("RSI超賣反彈")
    if macd > signal and df['MACD'].iloc[-2] <= df['Signal_Line'].iloc[-2]:
        hints.append("MACD剛翻多")
        
    if not hints:
        if current_price > s1 * 1.1: return "乖離過大，勿追高"
        else: return "觀望，等待訊號"
    return " | ".join(hints)

def calculate_volume_profile(df, bins=40, filter_mask=None):
    price_min = df['Low'].min()
    price_max = df['High'].max()
    bin_edges = np.linspace(price_min, price_max, bins + 1)
    target_df = df if filter_mask is None else df[filter_mask]
    if target_df.empty: return pd.DataFrame({'Price': [], 'Volume': []})
    bin_indices = pd.cut(target_df['Close'], bins=bin_edges, labels=False, include_lowest=True)
    profile = target_df.groupby(bin_indices)['Volume'].sum().reset_index()
    profile['Price'] = [(bin_edges[int(i)] + bin_edges[int(i)+1])/2 for i in profile['Close']]
    return profile

def format_volume(num):
    if num >= 1_000_000_000: return f"{num/1_000_000_000:.2f}B"
    elif num >= 1_000_000: return f"{num/1_000_000:.2f}M"
    elif num >= 1_000: return f"{num/1_000:.2f}K"
    else: return f"{num}"

# --- 3. 側邊欄 (靜態版) ---
with st.sidebar:
    st.title("🎛️ 控制台")
    st.markdown("---")
    st.header("📌 自選股清單")
    st.caption("點選下方股票代號以開始分析 (不連線 Yahoo)。")
    
    selection = st.radio("選擇股票", st.session_state.watchlist)
    current_ticker = selection

    st.markdown("---")
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
    
    st.markdown("---")
    time_opt = st.radio("週期", ["當沖 (分時)", "日線 (Daily)", "3日 (短線)", "10日 (波段)", "月線 (長線)"], index=1)

# --- 4. 主程式 ---
st.title(f"📈 {current_ticker} 實戰戰情室 V10.1")

api_period = "1y"; api_interval = "1d"; xaxis_format = "%Y-%m-%d"
if "當沖" in time_opt: api_period = "5d"; api_interval = "15m"; xaxis_format = "%H:%M" 
elif "日線" in time_opt: api_period = "6mo"; api_interval = "1d"; xaxis_format = "%m-%d" 
elif "3日" in time_opt: api_period = "5d"; api_interval = "30m"; xaxis_format = "%m-%d %H:%M" 
elif "10日" in time_opt: api_period = "1mo"; api_interval = "60m"; xaxis_format = "%m-%d %H:%M"
elif "月線" in time_opt: api_period = "2y"; api_interval = "1wk"; xaxis_format = "%Y-%m"

@st.cache_data(ttl=300)
def fetch_main_data(ticker, period, interval):
    try:
        return yf.download(ticker, period=period, interval=interval, progress=False, threads=False)
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
    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else latest

    # 計算 V10.1 指標
    s1, s2, s1_note, s2_note = find_support_levels(df, latest['Close'])
    buy_hint_text = generate_buy_hint(df, latest['Close'], s1, s2)
    strat_signals = analyze_strategic_signals(df) # 包含顏色與註解的戰略數據
    trend_icon, trend_desc = analyze_market_trend(df)
    target_price, ai_rating = predict_target_and_rating(df)
    
    pct_change = ((latest['Close'] - prev['Close']) / prev['Close']) * 100
    color_price = "green" if pct_change >= 0 else "red"
    
    # Row 1: 價格
    st.markdown(f"""
    <div class="price-card">
        <h1 style="margin:0; font-size: 50px;">${latest['Close']:.2f}</h1>
        <h3 style="margin:0; color: {color_price};">{pct_change:+.2f}%</h3>
        <p style="color: gray; margin-bottom: 5px;">最新成交量: {format_volume(latest['Volume'])}</p>
        <div class="buy-hint">💡 操作提示: {buy_hint_text}</div>
    </div>
    """, unsafe_allow_html=True)
    st.write("")

    # --- Row 2: V10.1 戰略雷達 (四合一圖解版) ---
    st.subheader("🚀 戰略雷達與 AI 預測")
    m_col1, m_col2, m_col3 = st.columns(3)

    # 1. 綜合戰略雷達 (更新版)
    with m_col1:
        st.markdown(f"""
        <div class="ai-box">
            <h5 style="color:white; margin:0; margin-bottom:5px;">📡 綜合戰略雷達</h5>
            <div class="signal-tag {strat_signals['Summary_Color']}" style="font-size:16px;">{strat_signals['Summary']}</div>
            <div class="radar-grid">
                <div class="radar-item">
                    <span>1. MACD</span>
                    <span class="signal-tag {strat_signals['MACD_Color']}">{strat_signals['MACD_Text']}</span>
                </div>
                <div class="radar-item">
                    <span>2. 成交量</span>
                    <span class="signal-tag {strat_signals['Vol_Color']}">{strat_signals['Vol_Text']}</span>
                </div>
                <div class="radar-item">
                    <span>3. RSI</span>
                    <span class="signal-tag {strat_signals['RSI_Color']}">{strat_signals['RSI_Text']}</span>
                </div>
                <div class="radar-item">
                    <span>4. 盤整</span>
                    <span class="signal-tag {strat_signals['Trend_Color']}">{strat_signals['Trend_Text']}</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # 2. 市場格局
    with m_col2:
        st.markdown(f"""
        <div class="ai-box">
            <h5 style="color:white; margin:0;">⚖️ 市場格局 & 評級</h5>
            <div style="font-size: 30px; margin-top:5px;">{trend_icon.split(' ')[0]} <span style="font-size:20px; color:#FFD700;">{ai_rating}</span></div>
            <p style="font-size:12px; color:#ccc;">{trend_icon.split(' ')[1]} | {trend_desc}</p>
        </div>
        """, unsafe_allow_html=True)

    # 3. 目標價
    with m_col3:
        target_upside = (target_price - latest['Close']) / latest['Close'] * 100
        target_color = "#28a745" if target_upside > 0 else "#dc3545"
        st.markdown(f"""
        <div class="ai-box" style="border: 1px solid {target_color};">
            <h5 style="color:white; margin:0;">🎯 短線 AI 目標價</h5>
            <h2 style="color:{target_color}; margin:0;">${target_price:.2f}</h2>
            <p style="font-size:12px; color:#ccc;">潛在空間: {target_upside:+.2f}% (布林通道測幅)</p>
        </div>
        """, unsafe_allow_html=True)

    st.write("")

    # --- 基本面 (Row 3) ---
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
    
    if peg is not None:
        p_val = f"{peg}"; peg_html = f'<div class="val-good">PEG: {peg}</div>' if peg < 1 else f'<div class="val-fair">PEG: {peg}</div>'
    elif fwd_pe is not None:
        p_val = f"{fwd_pe:.2f} (PE)"; peg_html = '<div class="val-fair">參考 Fwd PE</div>'
    else:
        p_val = "N/A"; peg_html = '<div class="val-bad">請重抓</div>'
    
    with f_col1: st.metric("估值 (PEG/PE)", p_val); st.markdown(peg_html, unsafe_allow_html=True)

    with f_col2:
        if rev_growth is not None:
            st.metric("成長率", f"{rev_growth*100:.2f}%")
            if rev_growth > 0.2: st.markdown('<div class="val-good">🔥 高成長</div>', unsafe_allow_html=True)
            else: st.markdown('<div class="val-fair">📈 正成長</div>', unsafe_allow_html=True)
        else: st.metric("成長率", "N/A"); st.caption("無資料")
    
    try:
        t_obj = yf.Ticker(current_ticker)
        cf = t_obj.cash_flow
        if not cf.empty:
            fcf_cur = cf.iloc[0, 0] if 'Free' in str(cf.index) else (cf.loc['Operating Cash Flow'].iloc[0] + cf.loc['Capital Expenditure'].iloc[0])
            with f_col3: st.metric("自由現金流", f"${fcf_cur/1e9:.2f}B")
        else:
            with f_col3: st.metric("自由現金流", "N/A")
    except:
        with f_col3: st.metric("自由現金流", "資料不足")

    s1_delta = "normal"
    if latest['Close'] < s1: s1_delta = "inverse"
    
    with f_col4: st.metric("🛡️ S1 趨勢 (MA20)", f"${s1:.2f}", delta_color=s1_delta); st.caption(s1_note)
    with f_col5: st.metric("🛡️ S2 籌碼 (大量低)", f"${s2:.2f}"); st.caption(s2_note)

    # Chart
    st.subheader(f"📈 走勢圖 - {time_opt}")
    plot_data = df
    if "當沖" in time_opt: plot_data = df.tail(26) 
    elif "3日" in time_opt: plot_data = df.tail(78)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_width=[0.2, 0.7])
    fig.add_trace(go.Candlestick(x=plot_data.index, open=plot_data['Open'], high=plot_data['High'], low=plot_data['Low'], close=plot_data['Close'], name='Price'), row=1, col=1)

    for i in range(1, len(plot_data)):
        curr = plot_data.iloc[i]; prior = plot_data.iloc[i-1]
        is_buy = ((curr['MACD'] > curr['Signal_Line']) and (prior['MACD'] <= prior['Signal_Line'])) or ((curr['RSI'] < 30) and (prior['RSI'] >= 30))
        if is_buy:
            fig.add_annotation(x=plot_data.index[i], y=curr['Low']*0.99, text=f"BUY<br>${curr['Close']:.2f}", showarrow=True, arrowhead=1, row=1, col=1, bgcolor="#28a745", font=dict(color="white", size=10))

    fig.add_hline(y=s1, line_dash="dash", line_color="#00d4ff", annotation_text=f"S1 (MA20)", row=1, col=1)
    fig.add_hline(y=s2, line_dash="dot", line_color="orange", annotation_text=f"S2 (Key Bar)", row=1, col=1)
    fig.add_hline(y=target_price, line_dash="dashdot", line_color="#FFD700", annotation_text=f"🎯 Target: {target_price:.2f}", row=1, col=1)

    if len(plot_data) > 20: fig.add_trace(go.Scatter(x=plot_data.index, y=plot_data['SMA_20'], line=dict(color='#00d4ff', width=1), name='20 MA'), row=1, col=1)
    if 'MACD_Hist' in plot_data.columns:
        colors = ['green' if v >= 0 else 'red' for v in plot_data['MACD_Hist']]
        fig.add_trace(go.Bar(x=plot_data.index, y=plot_data['MACD_Hist'], marker_color=colors, name='MACD'), row=2, col=1)
        fig.add_trace(go.Scatter(x=plot_data.index, y=plot_data['MACD'], line=dict(color='white', width=1), name='DIF'), row=2, col=1)
        fig.add_trace(go.Scatter(x=plot_data.index, y=plot_data['Signal_Line'], line=dict(color='yellow', width=1), name='DEM'), row=2, col=1)

    fig.update_xaxes(tickformat=xaxis_format)
    fig.update_layout(height=600, template="plotly_dark", xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

    # 籌碼分析
    st.subheader("🐳 籌碼與主力動向分析")
    chip_col1, chip_col2 = st.columns(2)
    mf = ((plot_data['Close'] - plot_data['Open']) / (plot_data['High'] - plot_data['Low'])) * plot_data['Volume']
    mf = mf.fillna(0); mf_cum = mf.cumsum()

    with chip_col1:
        st.markdown("##### 🏦 主力資金流向 (吸籌/出貨)")
        fig_mf = go.Figure()
        
        # Area chart color logic
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
        fig_vp.add_trace(go.Scatter(x=total_profile['Price'], y=total_profile['Volume'], fill='tozeroy', mode='lines', line=dict(color='#ffaa00', width=0), name='整體'))
        fig_vp.add_trace(go.Scatter(x=inst_profile['Price'], y=inst_profile['Volume'], fill='tozeroy', mode='lines', line=dict(color='#00d4ff', width=2), name='主力'))
        fig_vp.add_vline(x=latest['Close'], line_dash="dash", line_color="white", annotation_text="現價")
        fig_vp.update_layout(height=350, template="plotly_dark", margin=dict(l=10, r=10, t=30, b=10), showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig_vp, use_container_width=True)

except Exception as e:
    st.error(f"系統錯誤 (請稍後再試或檢查網路): {e}")
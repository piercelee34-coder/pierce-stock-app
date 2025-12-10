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
st.set_page_config(page_title="AI 實戰戰情室 V9.8 (極速單兵版)", layout="wide", page_icon="💎")

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

# [V9.8 優化] S1/S2 提供更清楚的中文狀態提示
def find_support_levels(df, current_price):
    if df.empty or len(df) < 60:
        return current_price, current_price, "資料不足", "資料不足"

    # S1: MA20
    s1 = df['Close'].rolling(window=20).mean().iloc[-1]
    # 判斷狀態
    if current_price > s1:
        dist = (current_price - s1) / s1 * 100
        s1_note = f"股價在月線之上 {dist:.1f}% (趨勢多)"
    else:
        dist = (s1 - current_price) / s1 * 100
        s1_note = f"已跌破月線 {dist:.1f}% (趨勢轉弱)"

    # S2: Key Bar Low
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

# [V9.8 新增] 產生「最近買點」的文字提示
def generate_buy_hint(df, current_price, s1, s2):
    if df.empty: return "無資料"
    
    rsi = df['RSI'].iloc[-1]
    macd = df['MACD'].iloc[-1]
    signal = df['Signal_Line'].iloc[-1]
    
    hints = []
    
    # 1. 均線邏輯
    if abs(current_price - s1) / current_price < 0.015 and current_price > s1:
        hints.append("回測月線(MA20)有撐，可佈局")
    elif current_price < s1 and current_price > s2:
        hints.append("跌破月線，等待回測 S2 支撐")
        
    # 2. 籌碼邏輯
    if abs(current_price - s2) / current_price < 0.02:
        hints.append("接近主力成本區(S2)，勝率高")
        
    # 3. 指標邏輯
    if rsi < 30:
        hints.append("RSI 超賣(<30)，隨時反彈")
    if macd > signal and df['MACD'].iloc[-2] <= df['Signal_Line'].iloc[-2]:
        hints.append("MACD 黃金交叉，波段起漲")
        
    if not hints:
        if current_price > s1 * 1.1:
            return "股價乖離過大，勿追高，等回檔"
        else:
            return "目前觀望，等待明確訊號"
            
    return " | ".join(hints)

def run_backtest_analysis(df):
    signals = df[df['RSI'] < 30].index
    trades = []
    hold_days = 10
    for date in signals:
        try:
            buy_price = df.loc[date]['Close']
            idx = df.index.get_loc(date) + hold_days
            if idx < len(df):
                sell_price = df.iloc[idx]['Close']
                profit_pct = (sell_price - buy_price) / buy_price
                trades.append(profit_pct)
        except: pass
    return trades

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
    st.caption("點選下方股票代號以開始分析。")
    
    # [V9.8] 移除所有紅綠燈抓取邏輯，改為純靜態列表
    # 這樣完全不會在背景連線 Yahoo，速度最快且不被鎖
    
    # 顯示簡單列表
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
st.title(f"📈 {current_ticker} 實戰戰情室 V9.8")

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
    except Exception:
        return pd.DataFrame()

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
    if df.empty: st.error("⚠️ 無法取得數據，請檢查代號。"); st.stop()

    df = calculate_indicators(df)
    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else latest

    # 計算 S1/S2 與 買點提示
    s1, s2, s1_note, s2_note = find_support_levels(df, latest['Close'])
    buy_hint_text = generate_buy_hint(df, latest['Close'], s1, s2)

    pct_change = ((latest['Close'] - prev['Close']) / prev['Close']) * 100
    color_price = "green" if pct_change >= 0 else "red"
    
    # [V9.8] 顯示價格與買點提示
    st.markdown(f"""
    <div class="price-card">
        <h1 style="margin:0; font-size: 50px;">${latest['Close']:.2f}</h1>
        <h3 style="margin:0; color: {color_price};">{pct_change:+.2f}%</h3>
        <p style="color: gray; margin-bottom: 5px;">最新成交量: {format_volume(latest['Volume'])}</p>
        <div class="buy-hint">💡 操作提示: {buy_hint_text}</div>
    </div>
    """, unsafe_allow_html=True)
    st.write("")

    # --- 基本面與防守 (含強制刷新) ---
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
    trail_pe = info.get('trailingPE')
    rev_growth = info.get('revenueGrowth') or info.get('quarterlyRevenueGrowth') or info.get('earningsGrowth')
    
    # Col 1: 估值
    if peg is not None:
        p_val = f"{peg}"
        peg_html = f'<div class="val-good">PEG: {peg}</div>' if peg < 1 else f'<div class="val-fair">PEG: {peg}</div>'
    elif fwd_pe is not None:
        p_val = f"{fwd_pe:.2f} (PE)"
        peg_html = '<div class="val-fair">參考 Fwd PE</div>'
    elif trail_pe is not None:
        p_val = f"{trail_pe:.2f} (PE)"
        peg_html = '<div class="val-fair">參考 Trailing PE</div>'
    else:
        p_val = "N/A"
        peg_html = '<div class="val-bad">請重抓</div>'
    
    with f_col1: 
        st.metric("估值 (PEG/PE)", p_val)
        st.markdown(peg_html, unsafe_allow_html=True)

    # Col 2: 成長率
    with f_col2:
        if rev_growth is not None:
            st.metric("成長率", f"{rev_growth*100:.2f}%")
            if rev_growth > 0.2: st.markdown('<div class="val-good">🔥 高成長</div>', unsafe_allow_html=True)
            else: st.markdown('<div class="val-fair">📈 正成長</div>', unsafe_allow_html=True)
        else:
            st.metric("成長率", "N/A")
            st.caption("無資料")
    
    # Col 3: 現金流
    try:
        t_obj = yf.Ticker(current_ticker)
        cf = t_obj.cash_flow
        if not cf.empty:
            fcf_cur = cf.iloc[0, 0] if 'Free' in str(cf.index) else (cf.loc['Operating Cash Flow'].iloc[0] + cf.loc['Capital Expenditure'].iloc[0])
            fcf_prev = cf.iloc[0, 1] if 'Free' in str(cf.index) else (cf.loc['Operating Cash Flow'].iloc[1] + cf.loc['Capital Expenditure'].iloc[1])
            fcf_chg = ((fcf_cur - fcf_prev)/abs(fcf_prev))*100
            with f_col3: 
                st.metric("自由現金流", f"${fcf_cur/1e9:.2f}B", f"{fcf_chg:.1f}% vs 去年")
        else:
            with f_col3: st.metric("自由現金流", "N/A")
    except:
        with f_col3: st.metric("自由現金流", "資料不足")

    # Col 4 & 5: S1 / S2 (優化後的提示)
    s1_delta = "normal"
    if latest['Close'] < s1: s1_delta = "inverse"
    
    with f_col4: 
        st.metric("🛡️ S1 趨勢 (MA20)", f"${s1:.2f}", delta_color=s1_delta)
        st.caption(s1_note) # 顯示更詳細的中文提示
        
    with f_col5: 
        st.metric("🛡️ S2 籌碼 (大量低)", f"${s2:.2f}")
        st.caption(s2_note) # 顯示更詳細的中文提示

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

    fig.add_hline(y=s1, line_dash="dash", line_color="#00d4ff", annotation_text=f"S1 (MA20): {s1:.2f}", row=1, col=1)
    fig.add_hline(y=s2, line_dash="dot", line_color="orange", annotation_text=f"S2 (Key Bar): {s2:.2f}", row=1, col=1)

    if len(plot_data) > 20: fig.add_trace(go.Scatter(x=plot_data.index, y=plot_data['SMA_20'], line=dict(color='#00d4ff', width=1, dash='solid'), name='20 MA'), row=1, col=1)
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
        st.markdown("##### 🏦 主力資金流向")
        fig_mf = go.Figure()
        fig_mf.add_trace(go.Scatter(x=plot_data.index, y=mf_cum, fill='tozeroy', mode='lines', line=dict(color='#00d4ff', width=2), name='主力'))
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
        st.markdown("""<div class="guide-box"><b>🧐 說明：</b><br>🟡 黃色山峰 = 散戶套牢區<br>🔵 青色山峰 = 主力成本區<br>若現價 > 青色山峰 👉 主力獲利 (強支撐)<br>若現價 < 青色山峰 👉 主力套牢 (強壓力)</div>""", unsafe_allow_html=True)

except Exception as e:
    st.error(f"系統錯誤 (請稍後再試或檢查網路): {e}")
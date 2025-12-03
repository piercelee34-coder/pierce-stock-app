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
st.set_page_config(page_title="AI 實戰戰情室 V8.7", layout="wide", page_icon="🛡️")

# --- CSS 美化 ---
st.markdown("""
<style>
    .big-alert {padding: 15px; border-radius: 10px; margin-bottom: 20px; text-align: center; font-size: 20px; font-weight: bold; color: white;}
    .price-card {background-color: #1e1e1e; padding: 20px; border-radius: 10px; text-align: center; border: 1px solid #333;}
    .ai-box {background-color: #333; padding: 10px; border-radius: 10px; border: 1px solid #555; text-align: center;}
    .trend-box {background-color: #2b2b2b; padding: 15px; border-radius: 10px; border-left: 5px solid #FFD700; margin-top: 10px; margin-bottom: 10px;}
    .signal-box-green {background-color: #1b3a1b; padding: 10px; border-radius: 8px; border: 1px solid #28a745; text-align: center; height: 100%;}
    .signal-box-red {background-color: #3a1b1b; padding: 10px; border-radius: 8px; border: 1px solid #dc3545; text-align: center; height: 100%;}
    .signal-box-neutral {background-color: #333; padding: 10px; border-radius: 8px; border: 1px solid #6c757d; text-align: center; height: 100%;}
    .undervalued {background-color: #d4edda; color: #155724; padding: 5px; border-radius: 5px; font-weight: bold;}
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

def find_support_levels(df):
    recent_lows = df['Low'].tail(60)
    s1 = recent_lows.min()
    s2 = recent_lows[recent_lows > s1 * 1.02].min()
    if pd.isna(s2): s2 = s1 * 1.05
    return s1, s2

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

# [V8.7] 防當機版：獲取自選股顏色
@st.cache_data(ttl=600) # 延長緩存時間到 10 分鐘，減少對 Yahoo 的請求
def get_colored_labels(tickers):
    labels = []
    try:
        # 嘗試下載數據
        data = yf.download(tickers, period="2d", progress=False)['Close']
        
        # 檢查是否下載失敗 (Yahoo Rate Limit)
        if data.empty:
            return tickers # 如果失敗，直接回傳原名單，不要報錯

        for t in tickers:
            try:
                closes = data if len(tickers) == 1 else data[t]
                if len(closes) >= 2:
                    change = closes.iloc[-1] - closes.iloc[-2]
                    icon = "🟢" if change >= 0 else "🔴"
                    labels.append(f"{t} {icon}")
                else:
                    labels.append(t)
            except: labels.append(t)
    except Exception:
        # 萬一發生任何錯誤，就回傳原始名單，確保程式不當機
        return tickers
    return labels

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

# --- 3. 側邊欄 ---
with st.sidebar:
    st.title("🎛️ 控制台")
    if st.button("🔄 立即更新報價"):
        st.cache_data.clear()
        st.rerun()
    st.markdown("---")
    
    st.header("📌 自選股清單")
    
    # 獲取標籤 (含防錯機制)
    display_labels = get_colored_labels(st.session_state.watchlist)
    
    # 確保 label_map 長度一致
    if len(display_labels) != len(st.session_state.watchlist):
        display_labels = st.session_state.watchlist # fallback
    
    label_map = {label: ticker for label, ticker in zip(display_labels, st.session_state.watchlist)}
    
    selection = st.radio("選擇股票 (即時漲跌)", display_labels)
    current_ticker = label_map.get(selection, "NVDA") # 防止選單錯誤

    # 排序按鈕
    c_up, c_down = st.columns(2)
    if c_up.button("⬆️ 上移") and current_ticker in st.session_state.watchlist:
        idx = st.session_state.watchlist.index(current_ticker)
        if idx > 0:
            st.session_state.watchlist[idx], st.session_state.watchlist[idx-1] = st.session_state.watchlist[idx-1], st.session_state.watchlist[idx]
            save_watchlist(st.session_state.watchlist)
            st.rerun()
            
    if c_down.button("⬇️ 下移") and current_ticker in st.session_state.watchlist:
        idx = st.session_state.watchlist.index(current_ticker)
        if idx < len(st.session_state.watchlist) - 1:
            st.session_state.watchlist[idx], st.session_state.watchlist[idx+1] = st.session_state.watchlist[idx+1], st.session_state.watchlist[idx]
            save_watchlist(st.session_state.watchlist)
            st.rerun()

    with st.expander("編輯清單"):
        new_t = st.text_input("輸入代號", placeholder="MSTR").upper()
        c1, c2 = st.columns(2)
        if c1.button("➕ 新增"):
            if new_t and new_t not in st.session_state.watchlist:
                st.session_state.watchlist.append(new_t)
                save_watchlist(st.session_state.watchlist)
                st.rerun()
        if c2.button("❌ 刪除"):
            if current_ticker in st.session_state.watchlist:
                st.session_state.watchlist.remove(current_ticker)
                save_watchlist(st.session_state.watchlist)
                st.rerun()

    st.markdown("---")
    time_opt = st.radio("週期", ["當沖 (分時)", "日線 (Daily)", "3日 (短線)", "10日 (波段)", "月線 (長線)"], index=1)

# --- 4. 主程式 ---
st.title(f"📈 {current_ticker} 實戰戰情室 V8.7")

api_period = "1y"; api_interval = "1d"; xaxis_format = "%Y-%m-%d"
if "當沖" in time_opt: api_period = "5d"; api_interval = "15m"; xaxis_format = "%H:%M" 
elif "日線" in time_opt: api_period = "6mo"; api_interval = "1d"; xaxis_format = "%m-%d" 
elif "3日" in time_opt: api_period = "5d"; api_interval = "30m"; xaxis_format = "%m-%d %H:%M" 
elif "10日" in time_opt: api_period = "1mo"; api_interval = "60m"; xaxis_format = "%m-%d %H:%M"
elif "月線" in time_opt: api_period = "2y"; api_interval = "1wk"; xaxis_format = "%Y-%m"

try:
    df = yf.download(current_ticker, period=api_period, interval=api_interval, progress=False)
    t_obj = yf.Ticker(current_ticker)
    info = t_obj.info
    
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    if df.empty: 
        st.warning("⚠️ 數據暫時無法取得，可能是 Yahoo Rate Limit。請稍後再試。")
        st.stop()

    df = calculate_indicators(df)
    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else latest

    hist_data = yf.download(current_ticker, period="2y", progress=False)
    if isinstance(hist_data.columns, pd.MultiIndex): hist_data.columns = hist_data.columns.get_level_values(0)
    hist_data = calculate_indicators(hist_data)
    trades = run_backtest_analysis(hist_data)
    
    avg_profit = 0; target_sell_price = 0; win_rate = 0
    if trades:
        win_count = sum(1 for t in trades if t > 0)
        win_rate = (win_count / len(trades)) * 100
        profitable = [t for t in trades if t > 0]
        if profitable:
            avg_profit = np.mean(profitable) * 100
            target_sell_price = latest['Close'] * (1 + (avg_profit/100))

    pct_change = ((latest['Close'] - prev['Close']) / prev['Close']) * 100
    color_price = "green" if pct_change >= 0 else "red"
    st.markdown(f"""
    <div class="price-card">
        <h1 style="margin:0; font-size: 50px;">${latest['Close']:.2f}</h1>
        <h3 style="margin:0; color: {color_price};">{pct_change:+.2f}%</h3>
        <p style="color: gray;">最新成交量: {format_volume(latest['Volume'])}</p>
    </div>
    """, unsafe_allow_html=True)
    st.write("")

    # [V8.7] 語法修復: 分行處理
    st.subheader("📊 基本面透視")
    f_col1, f_col2, f_col3, f_col4 = st.columns(4)
    peg = info.get('pegRatio'); fwd_pe = info.get('forwardPE')
    
    if peg is not None: 
        p_val = f"{peg}"
        p_st = "✨ 被低估" if peg < 1.0 else "估值合理"
        p_cls = "undervalued" if peg < 1.0 else ""
    elif fwd_pe is not None: 
        p_val = f"{fwd_pe:.2f} (Fwd P/E)"
        p_st = "無 PEG"
        p_cls = ""
    else: 
        p_val = "N/A"
        p_st = "資料不足"
        p_cls = ""
    
    with f_col1: 
        st.metric("估值指標", p_val)
        st.markdown(f'<span class="{p_cls}">{p_st}</span>', unsafe_allow_html=True)
    
    try:
        cf = t_obj.cash_flow
        if not cf.empty:
            fcf_cur = cf.iloc[0, 0] if 'Free' in str(cf.index) else (cf.loc['Operating Cash Flow'].iloc[0] + cf.loc['Capital Expenditure'].iloc[0])
            fcf_prev = cf.iloc[0, 1] if 'Free' in str(cf.index) else (cf.loc['Operating Cash Flow'].iloc[1] + cf.loc['Capital Expenditure'].iloc[1])
            fcf_chg = ((fcf_cur - fcf_prev)/abs(fcf_prev))*100
            with f_col2: 
                st.metric("自由現金流", f"${fcf_cur/1e9:.2f}B", f"{fcf_chg:.1f}% vs 去年")
        else: 
            with f_col2: 
                st.metric("自由現金流", "N/A")
    except: 
        with f_col2: 
            st.metric("自由現金流", "資料不足")

    s1, s2 = find_support_levels(df)
    with f_col3: st.metric("🛡️ 第一支撐位", f"${s1:.2f}")
    with f_col4: st.metric("🛡️ 第二支撐位", f"${s2:.2f}")

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

    if target_sell_price > 0: fig.add_hline(y=target_sell_price, line_dash="dashdot", line_color="#FFD700", annotation_text=f"🎯 Target: {target_sell_price:.2f}", row=1, col=1)
    fig.add_hline(y=s1, line_dash="dash", line_color="green", annotation_text=f"Support 1: {s1:.2f}", row=1, col=1)

    if len(plot_data) > 20: fig.add_trace(go.Scatter(x=plot_data.index, y=plot_data['SMA_20'], line=dict(color='orange', width=1), name='20 MA'), row=1, col=1)
    if 'MACD_Hist' in plot_data.columns:
        colors = ['green' if v >= 0 else 'red' for v in plot_data['MACD_Hist']]
        fig.add_trace(go.Bar(x=plot_data.index, y=plot_data['MACD_Hist'], marker_color=colors, name='MACD'), row=2, col=1)
        fig.add_trace(go.Scatter(x=plot_data.index, y=plot_data['MACD'], line=dict(color='white', width=1), name='DIF'), row=2, col=1)
        fig.add_trace(go.Scatter(x=plot_data.index, y=plot_data['Signal_Line'], line=dict(color='yellow', width=1), name='DEM'), row=2, col=1)

    fig.update_xaxes(tickformat=xaxis_format)
    fig.update_layout(height=600, template="plotly_dark", xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

    # Backtest
    st.subheader("🧠 智能策略回測系統")
    bt_col1, bt_col2, bt_col3, bt_col4 = st.columns(4)
    if trades:
        with bt_col1: st.metric("歷史勝率", f"{win_rate:.1f}%")
        with bt_col2: st.metric("平均漲幅", f"+{avg_profit:.2f}%")
        sugg = "Strong Buy" if win_rate >= 60 else "Wait"
        sugg_color = "#28a745" if win_rate >= 60 else "#6c757d"
        with bt_col3: st.markdown(f'<div class="ai-box" style="border: 2px solid {sugg_color};"><h5 style="color:white; margin:0;">AI 建議</h5><h3 style="color:{sugg_color}; margin:0;">{sugg}</h3></div>', unsafe_allow_html=True)
        with bt_col4: st.markdown(f'<div class="ai-box" style="border: 1px solid #FFD700;"><h5 style="color:white; margin:0;">目標價</h5><h2 style="color:#FFD700; margin:0;">${target_sell_price:.2f}</h2></div>', unsafe_allow_html=True)
    else: st.info("無足夠數據計算回測。")

    # Trend Dashboard
    st.markdown('<div class="trend-box"><h3>🧭 整體趨勢 (Market Trend)</h3></div>', unsafe_allow_html=True)
    
    trend_bull = latest['Close'] > latest['SMA_20']
    trend_bear = latest['Close'] < latest['SMA_20']
    rsi_low = latest['RSI'] < 40; rsi_high = latest['RSI'] > 70
    macd_red = latest['MACD_Hist'] > 0; macd_green = latest['MACD_Hist'] < 0
    vol_up = False; vol_down = False
    if 'Vol_SMA5' in latest and not pd.isna(latest['Vol_SMA5']):
        vol_up = (latest['Volume'] > latest['Vol_SMA5'] * 1.1) and (latest['Close'] > prev['Close'])
        vol_down = (latest['Volume'] > latest['Vol_SMA5'] * 1.1) and (latest['Close'] < prev['Close'])
    
    score_buy = sum([trend_bull, rsi_low, macd_red, vol_up])
    score_sell = sum([trend_bear, rsi_high, macd_green, vol_down])
    
    trend_title = "⚖️ 震盪整理 (Neutral)"; title_color = "#f0ad4e"
    if score_buy >= 3: trend_title = "🐂 牛市格局 (Bullish)"; title_color = "#28a745"
    elif score_sell >= 3: trend_title = "🐻 熊市格局 (Bearish)"; title_color = "#dc3545"
        
    st.markdown(f"<h2 style='text-align: center; color: {title_color};'>{trend_title}</h2>", unsafe_allow_html=True)
    st.write("")

    sig_col1, sig_col2, sig_col3, sig_col4 = st.columns(4)
    with sig_col1:
        if trend_bull: st.markdown('<div class="signal-box-green">📈 均線多頭<br>(價 > 20MA)</div>', unsafe_allow_html=True)
        elif trend_bear: st.markdown('<div class="signal-box-red">📉 均線空頭<br>(價 < 20MA)</div>', unsafe_allow_html=True)
        else: st.markdown('<div class="signal-box-neutral">⚖️ 均線糾結</div>', unsafe_allow_html=True)
    with sig_col2:
        if rsi_low: st.markdown(f'<div class="signal-box-green">💎 RSI 低檔<br>({latest["RSI"]:.1f} < 40)</div>', unsafe_allow_html=True)
        elif rsi_high: st.markdown(f'<div class="signal-box-red">🔥 RSI 過熱<br>({latest["RSI"]:.1f} > 70)</div>', unsafe_allow_html=True)
        else: st.markdown(f'<div class="signal-box-neutral">⚪ RSI 中性<br>({latest["RSI"]:.1f})</div>', unsafe_allow_html=True)
    with sig_col3:
        if macd_red: st.markdown('<div class="signal-box-green">🚀 MACD 翻紅<br>(多方動能)</div>', unsafe_allow_html=True)
        elif macd_green: st.markdown('<div class="signal-box-red">🔻 MACD 翻綠<br>(空方動能)</div>', unsafe_allow_html=True)
        else: st.markdown('<div class="signal-box-neutral">🟡 MACD 黏合</div>', unsafe_allow_html=True)
    with sig_col4:
        if vol_up: st.markdown('<div class="signal-box-green">📢 爆量上漲<br>(量增價漲)</div>', unsafe_allow_html=True)
        elif vol_down: st.markdown('<div class="signal-box-red">💥 爆量下跌<br>(量增價跌)</div>', unsafe_allow_html=True)
        else: st.markdown('<div class="signal-box-neutral">💤 量能溫和</div>', unsafe_allow_html=True)

    st.write("")
    v_col1, v_col2 = st.columns(2)
    with v_col1:
        st.caption("🟡 近 5 日微觀量能")
        last_5_vol = df['Volume'].tail(5)
        fig_v5 = go.Figure(go.Scatter(y=last_5_vol, fill='tozeroy', line=dict(color='yellow', width=2)))
        fig_v5.update_layout(height=100, margin=dict(l=0, r=0, t=0, b=0), showlegend=False, xaxis=dict(visible=False), yaxis=dict(visible=False), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_v5, use_container_width=True)
        st.metric("當日量", format_volume(latest['Volume']))
    with v_col2:
        st.caption("🔵 近 30 日巨觀量能")
        last_30_vol = df['Volume'].tail(30)
        fig_v30 = go.Figure(go.Scatter(y=last_30_vol, fill='tozeroy', line=dict(color='#00d4ff', width=2)))
        fig_v30.update_layout(height=100, margin=dict(l=0, r=0, t=0, b=0), showlegend=False, xaxis=dict(visible=False), yaxis=dict(visible=False), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_v30, use_container_width=True)
        st.metric("30日均量", format_volume(df['Volume'].tail(30).mean()))

    st.markdown("---")

    # Chip
    st.subheader("🐳 籌碼與主力動向分析")
    chip_col1, chip_col2 = st.columns(2)
    mf = ((plot_data['Close'] - plot_data['Open']) / (plot_data['High'] - plot_data['Low'])) * plot_data['Volume']
    mf = mf.fillna(0); mf_cum = mf.cumsum()

    with chip_col1:
        st.markdown("##### 🏦 主力資金流向")
        fig_mf = go.Figure()
        fig_mf.add_trace(go.Scatter(x=plot_data.index, y=mf_cum, fill='tozeroy', mode='lines', line=dict(color='#00d4ff', width=2), name='主力'))
        if len(mf_cum) > 1 and mf_cum.iloc[-1] < mf_cum.iloc[-2]:
            fig_mf.add_annotation(x=plot_data.index[-1], y=mf_cum.iloc[-1], text="⚠️ 主力出貨", showarrow=True, arrowhead=1, bgcolor="red", font=dict(color="white"))
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
    # V8.7: 錯誤處理更溫柔，不會直接當機，而是顯示警告
    st.error(f"系統暫時繁忙，請稍後再試: {e}")
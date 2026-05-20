import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import requests
import json
import os
import re

# 空頭距離指數引擎（新增）
try:
    import crisis_engine
    _CRISIS_AVAILABLE = True
except ImportError:
    _CRISIS_AVAILABLE = False

# SEC Form 4 內部人賣壓引擎（新增）
try:
    import insider_sentiment
    _INSIDER_AVAILABLE = True
except ImportError:
    _INSIDER_AVAILABLE = False

# --- 0. 系統設定 ---
st.set_page_config(page_title="AI 實戰戰情室 V26.04", layout="wide", page_icon="🚨")

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

    .aaii-bar-container { width: 100%; background-color: #444; border-radius: 4px; margin-bottom: 5px; height: 18px; overflow: hidden; }
    .aaii-bar-bull { height: 100%; background-color: #4ade80; float: left; text-align: center; color: black; font-size: 11px; font-weight: bold; line-height: 18px;}
    .aaii-bar-bear { height: 100%; background-color: #ff6b6b; float: left; text-align: center; color: white; font-size: 11px; font-weight: bold; line-height: 18px;}
    .aaii-bar-neu { height: 100%; background-color: #aaa; float: left; text-align: center; color: black; font-size: 11px; font-weight: bold; line-height: 18px;}
</style>
""", unsafe_allow_html=True)

# --- 1. 資料系統 ---
WATCHLIST_FILE, ANCHOR_FILE, TW_NAMES_FILE = "watchlist.json", "anchors.json", "tw_names.json"
DEFAULT_WATCHLISTS = {
    "🇺🇸 美持股": ['^NDX', 'NVDA', 'TSM', 'AAPL', 'PLTR'],
    "🔭 美觀察": ['AMD', 'TSLA', 'GOOG', 'META', 'AMZN'],
    "🔭 觀察股2": ['COIN', 'SOFI', 'NKE', 'NFLX'],
    "🇹🇼 台持股": ['2330.TW', '0050.TW', '00878.TW'],
    "🔭 台觀察": ['8150.TW', '5534.TW', '3535.TW', '6830.TW'],
}

# [修復 #3 v2] 拿掉 hardcoded fallback；抓不到環境變數時回空字串並印警告
def _get_secret(name, default=""):
    """同時支援 Streamlit secrets 與環境變數，皆無則回 default"""
    try:
        if hasattr(st, "secrets") and name in st.secrets:
            return str(st.secrets[name]).strip()
    except Exception:
        pass
    return os.environ.get(name, default).strip()

# ──────────────────────────────────────────────────────
# 固定時間錨點（給快取用）
# ──────────────────────────────────────────────────────
def get_cache_anchor():
    """回傳當前所屬的「固定時間錨點」字串。
    
    台灣時間 4 個錨點：
      05:00  美股收盤後 1 小時（美股最新數據就緒）
      08:00  台股開盤前 1 小時
      14:00  台股收盤後 30 分鐘（台股收盤資料就緒）
      20:00  美股盤前 1.5 小時
    
    用作 cache_data 的 key — anchor 變了才會重抓資料。
    """
    try:
        from datetime import datetime as _dt
        # 用 UTC + 8 算台灣時間（避免 pytz 套件依賴）
        utc_now = _dt.utcnow()
        tw_now = utc_now + timedelta(hours=8)
        hour = tw_now.hour
        anchors = [5, 8, 14, 20]
        # 找最近的「之前」錨點
        for a in reversed(anchors):
            if hour >= a:
                anchor_dt = tw_now.replace(hour=a, minute=0, second=0, microsecond=0)
                return anchor_dt.strftime("%Y%m%d_%H")
        # 比 05:00 還早 → 用昨天 20:00
        yesterday = tw_now - timedelta(days=1)
        anchor_dt = yesterday.replace(hour=20, minute=0, second=0, microsecond=0)
        return anchor_dt.strftime("%Y%m%d_%H")
    except Exception:
        # 失敗時 fallback 用 8 小時錨點
        return _dt.utcnow().strftime("%Y%m%d_%H")[:11]


FINMIND_TOKEN = _get_secret("FINMIND_TOKEN", "")


# ──────────────────────────────────────────────────────
# v27 起漲訊號 — 大盤過濾器 (SPY) + 7 維推演 context
# ──────────────────────────────────────────────────────
@st.cache_data(ttl=21600, show_spinner=False)
def get_market_filter_state(anchor):
    """抓 SPY 計算大盤過濾器：是否站上 SMA200 + SMA200 上升
    
    回傳 dict: {date: market_ok_bool}
    給「起漲訊號」當大盤過濾條件用
    """
    try:
        import yfinance as _yf
        spy = _yf.Ticker("SPY").history(period="2y", auto_adjust=False)
        if spy.empty:
            return {}
        spy["SMA_200"] = spy["Close"].rolling(200).mean()
        spy["above"] = spy["Close"] > spy["SMA_200"]
        spy["uptrend"] = spy["SMA_200"] > spy["SMA_200"].shift(5)
        spy["market_ok"] = spy["above"] & spy["uptrend"]
        # 轉成 dict: date -> bool
        result = {}
        for d, ok in spy["market_ok"].items():
            try:
                key = pd.Timestamp(d).normalize()
                result[key] = bool(ok) if not pd.isna(ok) else True
            except Exception:
                continue
        return result
    except Exception:
        return {}


def market_ok_at_date(market_state, date):
    """查指定日期的大盤狀態。若無資料則回傳 True（不過濾）"""
    if not market_state:
        return True
    try:
        key = pd.Timestamp(date).normalize()
        # 找最接近的日期（往前）
        sorted_keys = sorted(market_state.keys())
        for k in reversed(sorted_keys):
            if k <= key:
                return market_state[k]
        return True
    except Exception:
        return True


def get_engine_type_v27(ticker):
    """股票分類 (給 7 維 context 用)"""
    trend = {"NVDA", "MSFT", "AAPL", "GOOG", "GOOGL", "AMZN", "META", "TSM",
              "QQQ", "ORCL", "AVGO", "2330.TW", "2317.TW"}
    momentum = {"AMD", "TSLA", "MU", "INTC", "ASX"}
    growth = {"PLTR", "CRCL", "SOFI", "PYPL"}
    if ticker in trend: return "trend"
    if ticker in momentum: return "momentum"
    if ticker in growth: return "growth"
    return "reversal"


def compute_signal_context(p_data, idx_pos, ticker, market_state):
    """計算 7 維推演 context（給 AI 劇本用）
    
    Returns dict with: engine, market, position, vol ratios, momentum, obv_pct
    """
    try:
        if idx_pos < 20 or idx_pos >= len(p_data):
            return None
        curr = p_data.iloc[idx_pos]
        prior_20 = p_data.iloc[max(0, idx_pos - 20)]
        engine = get_engine_type_v27(ticker)
        mkt_ok = market_ok_at_date(market_state, curr.name if hasattr(curr, "name") else curr.get("Date"))
        
        # 位置
        high_52w_window = p_data.iloc[max(0, idx_pos - 252): idx_pos + 1]
        high_52w = high_52w_window["High"].max() if "High" in high_52w_window.columns else curr["Close"]
        sma60 = curr.get("SMA_60", curr["Close"])
        sma200 = curr.get("SMA_200", np.nan)
        dist_from_52w_high = (high_52w - curr["Close"]) / high_52w * 100 if high_52w > 0 else 0
        dist_from_sma60 = (curr["Close"] - sma60) / sma60 * 100 if sma60 > 0 else 0
        above_sma200 = (not pd.isna(sma200) and curr["Close"] > sma200)
        
        # 量能
        vol_5 = curr.get("Vol_SMA5", np.nan)
        vol_20 = curr.get("Vol_SMA20", np.nan)
        vol_60 = curr.get("Vol_SMA60", np.nan)
        short_vol = (vol_5 / vol_20) if (not pd.isna(vol_5) and not pd.isna(vol_20) and vol_20 > 0) else 1.0
        trend_vol = (vol_20 / vol_60) if (not pd.isna(vol_20) and not pd.isna(vol_60) and vol_60 > 0) else 1.0
        
        # 動能
        ret_20d = ((curr["Close"] / prior_20["Close"]) - 1) * 100 if prior_20["Close"] > 0 else 0
        
        # OBV
        obv_window = p_data.iloc[max(0, idx_pos - 60): idx_pos + 1].get("OBV", pd.Series())
        obv_now = curr.get("OBV", 0)
        if len(obv_window) > 1:
            obv_max = obv_window.max()
            obv_min = obv_window.min()
            obv_pct = ((obv_now - obv_min) / (obv_max - obv_min) * 100) if obv_max > obv_min else 50
        else:
            obv_pct = 50
        
        return {
            "engine": engine,
            "market_ok": mkt_ok,
            "dist_from_52w_high": round(dist_from_52w_high, 1),
            "dist_from_sma60": round(dist_from_sma60, 1),
            "above_sma200": above_sma200,
            "short_vol_ratio": round(short_vol, 2),
            "trend_vol_ratio": round(trend_vol, 2),
            "momentum_20d": round(ret_20d, 2),
            "obv_pct": round(obv_pct, 0),
        }
    except Exception:
        return None


def render_signal_context_panel(ctx, signal_label):
    """渲染 7 維 context 推演面板 (HTML)"""
    if not ctx:
        return ""
    mkt_icon = "🟢 大盤健康" if ctx["market_ok"] else "🔴 大盤偏空"
    mkt_color = "#22c55e" if ctx["market_ok"] else "#ef4444"
    
    # 機率推演
    base_win = 60  # 方案 A 基準勝率
    bonus = 0
    if ctx["market_ok"]: bonus += 5
    if ctx["short_vol_ratio"] > 1.2: bonus += 3
    if ctx["obv_pct"] > 80: bonus += 3
    if ctx["dist_from_52w_high"] > 15: bonus += 3  # 不在頂部
    if ctx["momentum_20d"] > 0 and ctx["momentum_20d"] < 15: bonus += 2  # 動能合理
    if ctx["dist_from_52w_high"] < 5: bonus -= 5   # 已在頂部
    
    win_rate = min(85, max(35, base_win + bonus))
    
    # 風險警示
    risks = []
    if ctx["dist_from_52w_high"] < 5:
        risks.append("⚠️ 距 52W 高點僅 {:.1f}%，逢高風險".format(ctx["dist_from_52w_high"]))
    if not ctx["market_ok"]:
        risks.append("⚠️ 大盤偏空，個股表現受拖累")
    if ctx["momentum_20d"] > 20:
        risks.append("⚠️ 20 日漲幅 +{:.1f}%，已偏熱".format(ctx["momentum_20d"]))
    if ctx["short_vol_ratio"] < 0.8:
        risks.append("⚠️ 近 5 日量縮，動能可能轉弱")
    
    risk_html = "".join(f'<div style="color:#facc15;font-size:12px;margin-top:3px;">{r}</div>' for r in risks)
    if not risks:
        risk_html = '<div style="color:#22c55e;font-size:12px;margin-top:3px;">✅ 無明顯警示</div>'
    
    # ── 各維度直白解讀 ──
    engine_hint_map = {
        "trend": "權值股/龍頭股，訊號最可靠",
        "momentum": "高波動成長股，訊號需驗證",
        "growth": "成長型新股，可信度中等",
        "reversal": "妖股/低流動性，訊號最雜訊",
    }
    engine_hint = engine_hint_map.get(ctx["engine"], "")

    # [V26.04] 提示分色 helper：危險=紅、中等=橙、機會=綠、中性=灰
    def _hc(hint: str) -> str:
        """根據提示文字前綴回傳對應 CSS 顏色。"""
        if hint.startswith("✅"):
            return "#22c55e"   # 綠：機會 / 正面
        if hint.startswith("⚠️"):
            _danger_kw = ("風險", "回檔", "出走", "轉弱", "警惕", "跌破", "偏弱", "清倉")
            return "#ef4444" if any(k in hint for k in _danger_kw) else "#f97316"
        return "#9ca3af"       # 灰：中性

    # 距 52W 高點解讀
    d52 = ctx["dist_from_52w_high"]
    if d52 < 5:    d52_hint = "⚠️ 在頂部，逢高風險"
    elif d52 < 15: d52_hint = "⚠️ 接近高點，注意過熱"
    elif d52 < 30: d52_hint = "✅ 健康位置，仍有空間"
    else:          d52_hint = "✅ 深度回檔/低位，反轉空間大"

    # 距 SMA60 解讀
    d60 = ctx["dist_from_sma60"]
    if d60 > 15:   d60_hint = "⚠️ 大幅偏離季線，可能回檔"
    elif d60 > 5:  d60_hint = "✅ 趨勢向上"
    elif d60 > -5: d60_hint = "貼近季線，整理中"
    else:          d60_hint = "⚠️ 跌破季線，仍偏弱"

    # 5 日量比解讀
    sv = ctx["short_vol_ratio"]
    if sv > 1.5:   sv_hint = "✅ 量增明顯，動能強"
    elif sv > 1.1: sv_hint = "✅ 輕微量增"
    elif sv > 0.9: sv_hint = "量平"
    else:          sv_hint = "⚠️ 量縮，動能轉弱"

    # 20 日動能解讀
    m20 = ctx["momentum_20d"]
    if m20 > 25:    m20_hint = "⚠️ 漲過頭，警惕回檔"
    elif m20 > 10:  m20_hint = "✅ 健康漲勢"
    elif m20 > 0:   m20_hint = "小漲，動能溫和"
    elif m20 > -10: m20_hint = "下跌中"
    else:           m20_hint = "深度回檔，可能築底"

    # OBV 解讀
    obv = ctx["obv_pct"]
    if obv > 80:   obv_hint = "✅ 資金強烈流入，籌碼集中"
    elif obv > 50: obv_hint = "中位偏高，籌碼穩定"
    elif obv > 20: obv_hint = "中位偏低，籌碼分散中"
    else:          obv_hint = "⚠️ 籌碼正在出走"
    
    return f'''
<div style="background:linear-gradient(135deg,#1a1a1c,#1a2a2c);
            border-left:4px solid #facc15; padding:12px 16px; margin:8px 0;
            border-radius:6px; font-size:13px;">
  <div style="color:#facc15; font-weight:bold; margin-bottom:8px;">
    🎯 {signal_label} — AI 推演（7 維 context）
  </div>
  <div style="display:grid; grid-template-columns:1fr 1fr; gap:10px; color:#ddd;">
    <div>
      <div>個股性質：<b>{ctx["engine"]}</b> <span style="color:{_hc(engine_hint)};font-size:11px;">（{engine_hint}）</span></div>
      <div>大盤狀態：<span style="color:{mkt_color};font-weight:bold;">{mkt_icon}</span></div>
      <div>距 52W 高點：<b>{ctx["dist_from_52w_high"]:.1f}%</b> <span style="color:{_hc(d52_hint)};font-size:11px;">（{d52_hint}）</span></div>
      <div>距 SMA60：<b>{ctx["dist_from_sma60"]:+.1f}%</b> <span style="color:{_hc(d60_hint)};font-size:11px;">（{d60_hint}）</span></div>
    </div>
    <div>
      <div>5 日量比：<b>{ctx["short_vol_ratio"]:.2f}x</b> <span style="color:{_hc(sv_hint)};font-size:11px;">（{sv_hint}）</span></div>
      <div>20 日趨勢量：<b>{ctx["trend_vol_ratio"]:.2f}x</b></div>
      <div>20 日動能：<b>{ctx["momentum_20d"]:+.2f}%</b> <span style="color:{_hc(m20_hint)};font-size:11px;">（{m20_hint}）</span></div>
      <div>OBV 累積位置：<b>{ctx["obv_pct"]:.0f}%</b> <span style="color:{_hc(obv_hint)};font-size:11px;">（{obv_hint}）</span></div>
    </div>
  </div>
  <div style="margin-top:10px; padding-top:8px; border-top:1px solid #2a2a2c;">
    <span style="color:#aaa;">🎯 10 日上漲機率推演：</span>
    <span style="color:#22c55e; font-weight:bold; font-size:18px;">{win_rate}%</span>
    <span style="color:#888; font-size:11px;"> （基準 60% ± 修正）</span>
  </div>
  <div style="margin-top:6px;">
    {risk_html}
  </div>
</div>
'''

if not FINMIND_TOKEN:
    print("⚠️  警告：未設定 FINMIND_TOKEN，台股籌碼功能將無法使用。")

# 新增：FRED API key（空頭距離指數 — 殖利率倒掛、HY 信用利差需要）
FRED_KEY = _get_secret("FRED_API_KEY", "")
if not FRED_KEY:
    print("⚠️  警告：未設定 FRED_API_KEY，空頭距離指數的殖利率/信用利差將無法使用。")

def json_load(f_name, default):
    if os.path.exists(f_name):
        try:
            with open(f_name, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return default

def json_save(f_name, data):
    try:
        with open(f_name, "w", encoding="utf-8") as f:
            json.dump(data, f)
    except Exception:
        pass

def load_watchlists():
    return json_load(WATCHLIST_FILE, DEFAULT_WATCHLISTS)

def save_watchlists(data):
    json_save(WATCHLIST_FILE, data)
    st.session_state.watchlists = data


# [修復 #7 v2] cache 只包住 API 網路查詢，file I/O 移到外層 get_stock_name 處理
_TW_NAME_BAD_WORDS = ["Yahoo", "股市", "走勢", "無符合", "找不到", "代碼或名稱", "html", "TW", "TWO", "INC", "CORP", "LTD", "COMPANY"]

@st.cache_data(ttl=86400)
def _query_tw_name_api(ticker: str) -> str | None:
    """純 API 查詢，不碰任何 file I/O；結果由外層決定是否存檔。"""
    base = ticker.split('.')[0]
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    name = None
    try:
        res = requests.get(f"https://www.twse.com.tw/zh/api/codeQuery?query={base}", headers=headers, timeout=3)
        if res.status_code == 200:
            data = res.json()
            if "suggestions" in data and len(data["suggestions"]) > 0:
                sug = data["suggestions"][0]
                if "無符合" not in sug:
                    n = sug.replace(base, '').replace('\t', '').strip()
                    if re.search(r'[\u4e00-\u9fff]', n):
                        name = n
    except Exception:
        pass
    if not name:
        try:
            cnyes_prefix = "OTC" if ".TWO" in ticker else "TWS"
            res = requests.get(f"https://ws.api.cnyes.com/ws/api/v1/quote/quotes/{cnyes_prefix}:{base}:STOCK", headers=headers, timeout=3)
            if res.status_code == 200:
                data = res.json()
                if "data" in data and len(data["data"]) > 0:
                    n = data["data"][0].get("name")
                    if n and re.search(r'[\u4e00-\u9fff]', n):
                        name = n.strip()
        except Exception:
            pass
    if not name:
        try:
            res = requests.get(f"https://query2.finance.yahoo.com/v1/finance/search?q={ticker}", headers=headers, timeout=3)
            if res.status_code == 200:
                quotes = res.json().get('quotes', [])
                if quotes:
                    n = quotes[0].get('shortname') or quotes[0].get('longname')
                    if n and not any(bad in n.upper() for bad in _TW_NAME_BAD_WORDS):
                        name = n.strip()
        except Exception:
            pass
    if name and not any(bad in name.upper() for bad in _TW_NAME_BAD_WORDS):
        return name
    return None


def get_stock_name(ticker: str) -> str:
    """外層負責 file cache 讀寫；API 查詢交給有 @st.cache_data 的 _query_tw_name_api。"""
    us_map = {'NVDA': '輝達', 'TSLA': '特斯拉', 'AAPL': '蘋果', 'MU': '美光', 'TSM': '台積電', 'GOOGL': '谷歌'}
    base = ticker.split('.')[0]
    if base in us_map and not (".TW" in ticker or ".TWO" in ticker):
        return us_map[base]
    if ".TW" in ticker or ".TWO" in ticker:
        # --- 1. 讀 file cache，順便清理髒資料 ---
        local_map = json_load(TW_NAMES_FILE, {})
        keys_to_delete = [
            k for k, v in local_map.items()
            if any(bad in v.upper() for bad in _TW_NAME_BAD_WORDS)
            or (not re.search(r'[\u4e00-\u9fff]', v) and len(v) > 6)
        ]
        for k in keys_to_delete:
            local_map.pop(k, None)
        # --- 2. file cache 命中 → 直接回傳，不碰 API ---
        if ticker in local_map:
            return local_map[ticker]
        # --- 3. file cache 未命中 → 走 API（有 st.cache_data 保護）---
        name = _query_tw_name_api(ticker)
        # --- 4. 查到名稱才存檔，存檔與 API cache 完全分離 ---
        if name:
            local_map[ticker] = name
            json_save(TW_NAMES_FILE, local_map)
            return name
    return ticker


if 'watchlists' not in st.session_state:
    st.session_state['watchlists'] = load_watchlists()
if 'active_list' not in st.session_state:
    st.session_state['active_list'] = list(st.session_state['watchlists'].keys())[0]
if 'user_opened_list' not in st.session_state:
    st.session_state['user_opened_list'] = None
if 'current_ticker' not in st.session_state:
    st.session_state['current_ticker'] = "^IXIC"
    for wl_name, wl in st.session_state['watchlists'].items():
        if wl:
            st.session_state['current_ticker'] = wl[0]
            st.session_state['active_list'] = wl_name
            st.session_state['user_opened_list'] = wl_name
            break

# --- 2. 總經指標 API 引擎 ---
# =============================================================
# [v25 重構] NAAIM / AAII 真實資料抓取（含限流與快取）
# 設計原則：
#   1. NAAIM 每週四發布、AAII 每週四發布
#   2. 當日嘗試 2~5 次（每 30 分一次），抓到後快取整週
#   3. 抓不到 → 退回最後一次成功的快取，再退回示範資料
#   4. 主源 macromicro，後備 NAAIM 官方 XLSX
# =============================================================
import time as _time

_SENT_CACHE_DIR = ".sentiment_cache"
_NAAIM_CACHE_FILE = os.path.join(_SENT_CACHE_DIR, "naaim.json")
_AAII_CACHE_FILE = os.path.join(_SENT_CACHE_DIR, "aaii.json")
_ATTEMPT_LOG_FILE = os.path.join(_SENT_CACHE_DIR, "attempts.json")
_MAX_ATTEMPTS_PER_DAY = 5
_MIN_ATTEMPT_GAP_MIN = 30
_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"


def _ensure_sent_cache_dir():
    if not os.path.exists(_SENT_CACHE_DIR):
        try:
            os.makedirs(_SENT_CACHE_DIR, exist_ok=True)
        except Exception:
            pass


def _sent_load(path, default):
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return default


def _sent_save(path, data):
    _ensure_sent_cache_dir()
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, default=str)
    except Exception:
        pass


def _can_fetch(source_key: str) -> bool:
    log = _sent_load(_ATTEMPT_LOG_FILE, {})
    today = datetime.now().strftime("%Y-%m-%d")
    today_log = log.get(today, {}).get(source_key, {"count": 0, "last_ts": 0, "has_success": False})
    if today_log.get("has_success", False):
        return False
    if today_log.get("count", 0) >= _MAX_ATTEMPTS_PER_DAY:
        return False
    last_ts = today_log.get("last_ts", 0)
    if last_ts and (_time.time() - last_ts) < _MIN_ATTEMPT_GAP_MIN * 60:
        return False
    return True


def _log_fetch(source_key: str, success: bool):
    log = _sent_load(_ATTEMPT_LOG_FILE, {})
    today = datetime.now().strftime("%Y-%m-%d")
    if today not in log:
        log[today] = {}
    if source_key not in log[today]:
        log[today][source_key] = {"count": 0, "last_ts": 0, "has_success": False}
    log[today][source_key]["count"] += 1
    log[today][source_key]["last_ts"] = _time.time()
    if success:
        log[today][source_key]["has_success"] = True
    cutoff = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
    log = {k: v for k, v in log.items() if k >= cutoff}
    _sent_save(_ATTEMPT_LOG_FILE, log)


def _fetch_naaim_macromicro():
    try:
        url = "https://www.macromicro.me/charts/data/46198"
        headers = {
            "User-Agent": _UA,
            "Accept": "application/json, text/javascript, */*; q=0.01",
            "Referer": "https://www.macromicro.me/charts/46198/naaim-exposure-index",
            "X-Requested-With": "XMLHttpRequest",
        }
        r = requests.get(url, headers=headers, timeout=8)
        if r.status_code != 200:
            return None
        data = r.json()
        series = None
        if isinstance(data, dict):
            if "data" in data and isinstance(data["data"], dict):
                for k, v in data["data"].items():
                    if isinstance(v, dict) and "series" in v:
                        series = v["series"]
                        break
            elif "series" in data:
                series = data["series"]
        if not series:
            return None
        out = []
        for item in series:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                ts, val = item[0], item[1]
                try:
                    if ts > 1e10:
                        dt = datetime.fromtimestamp(ts / 1000)
                    else:
                        dt = datetime.fromtimestamp(ts)
                    out.append({"date": dt.strftime("%Y-%m-%d"), "value": float(val)})
                except Exception:
                    continue
        return out if len(out) >= 4 else None
    except Exception:
        return None


def _fetch_naaim_official():
    try:
        url = "https://naaim.org/wp-content/uploads/2014/03/NAAIM-Exposure-Index-Data.xlsx"
        r = requests.get(url, headers={"User-Agent": _UA}, timeout=10)
        if r.status_code != 200:
            return None
        try:
            import io as _io
            df_x = pd.read_excel(_io.BytesIO(r.content))
        except Exception:
            return None
        date_col, val_col = None, None
        for c in df_x.columns:
            cl = str(c).lower()
            if "date" in cl and date_col is None:
                date_col = c
            elif ("naaim" in cl or "exposure" in cl or "mean" in cl or "average" in cl) and val_col is None:
                val_col = c
        if not date_col or not val_col:
            return None
        df_x = df_x[[date_col, val_col]].dropna()
        df_x.columns = ["date", "value"]
        df_x["date"] = pd.to_datetime(df_x["date"], errors="coerce")
        df_x = df_x.dropna(subset=["date"]).sort_values("date").tail(60)
        out = [{"date": rr["date"].strftime("%Y-%m-%d"), "value": float(rr["value"])}
               for _, rr in df_x.iterrows()]
        return out if len(out) >= 4 else None
    except Exception:
        return None


@st.cache_data(ttl=1800)  # 30 分鐘記憶體 cache,真正限流靠下面 disk 機制
def get_naaim_data():
    """
    回傳 (DataFrame[Date, Exposure], status_str)
    status_str: "real" / "cached" / "demo"
    """
    cache = _sent_load(_NAAIM_CACHE_FILE, None)

    # 1. 檔案 cache 有資料且 <6 天 → 直接用
    if cache:
        try:
            last_dt = datetime.strptime(cache.get("last_update", ""), "%Y-%m-%d %H:%M:%S")
            if (datetime.now() - last_dt).days < 6:
                df = pd.DataFrame(cache["data"])
                df["Date"] = pd.to_datetime(df["date"])
                df["Exposure"] = pd.to_numeric(df["value"], errors="coerce")
                return df[["Date", "Exposure"]].dropna(), "cached"
        except Exception:
            pass

    # 2. 嘗試抓取（限流）
    data = None
    if _can_fetch("naaim"):
        data = _fetch_naaim_macromicro()
        if not data:
            data = _fetch_naaim_official()
        _log_fetch("naaim", success=(data is not None))

    if data:
        _sent_save(_NAAIM_CACHE_FILE, {
            "last_update": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "data": data,
        })
        df = pd.DataFrame(data)
        df["Date"] = pd.to_datetime(df["date"])
        df["Exposure"] = pd.to_numeric(df["value"], errors="coerce")
        return df[["Date", "Exposure"]].dropna(), "real"

    # 3. 退回 cache（即便過期也用）
    if cache:
        df = pd.DataFrame(cache.get("data", []))
        if not df.empty:
            df["Date"] = pd.to_datetime(df["date"])
            df["Exposure"] = pd.to_numeric(df["value"], errors="coerce")
            return df[["Date", "Exposure"]].dropna(), "cached"

    # 4. 完全沒資料 → 示範
    now = datetime.now()
    dates = [now - timedelta(weeks=51 - i) for i in range(52)]
    rng = np.random.default_rng(seed=42)
    values = np.clip(rng.integers(40, 100, size=52).astype(float), 20, 110).round(1)
    return pd.DataFrame({"Date": dates, "Exposure": values}), "demo"


def _fetch_aaii_macromicro():
    try:
        url = "https://www.macromicro.me/charts/data/20828"
        headers = {
            "User-Agent": _UA,
            "Accept": "application/json, text/javascript, */*; q=0.01",
            "Referer": "https://www.macromicro.me/charts/20828/us-aaii-sentimentsurvey",
            "X-Requested-With": "XMLHttpRequest",
        }
        r = requests.get(url, headers=headers, timeout=8)
        if r.status_code != 200:
            return None
        data = r.json()
        series_dict = {}
        if isinstance(data, dict) and "data" in data and isinstance(data["data"], dict):
            for k, v in data["data"].items():
                if isinstance(v, dict) and "series" in v:
                    name = str(v.get("name", k)).lower()
                    series_dict[name] = v["series"]

        def _pick_latest(keys):
            for kw in keys:
                for name, s in series_dict.items():
                    if kw in name and s:
                        try:
                            return float(s[-1][1])
                        except Exception:
                            continue
            return None

        bull = _pick_latest(["bull", "看多"])
        neu = _pick_latest(["neutral", "中立"])
        bear = _pick_latest(["bear", "看空"])
        if bull is not None and bear is not None:
            if neu is None:
                neu = max(0, 100 - bull - bear)
            return (round(bull, 1), round(neu, 1), round(bear, 1))
        return None
    except Exception:
        return None


@st.cache_data(ttl=1800)
def get_aaii_data():
    """
    回傳 ((bull, neu, bear), status_str)
    """
    cache = _sent_load(_AAII_CACHE_FILE, None)
    if cache:
        try:
            last_dt = datetime.strptime(cache.get("last_update", ""), "%Y-%m-%d %H:%M:%S")
            if (datetime.now() - last_dt).days < 6:
                d = cache.get("data", {})
                return (d.get("bull", 0), d.get("neu", 0), d.get("bear", 0)), "cached"
        except Exception:
            pass

    result = None
    if _can_fetch("aaii"):
        result = _fetch_aaii_macromicro()
        _log_fetch("aaii", success=(result is not None))

    if result:
        bull, neu, bear = result
        _sent_save(_AAII_CACHE_FILE, {
            "last_update": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "data": {"bull": bull, "neu": neu, "bear": bear},
        })
        return (bull, neu, bear), "real"

    if cache:
        d = cache.get("data", {})
        return (d.get("bull", 0), d.get("neu", 0), d.get("bear", 0)), "cached"

    return (51.5, 25.1, 23.4), "demo"


# =============================================================
# [v25 新增] 大盤崩跌警示模組
# =============================================================
_INDEX_TICKERS = {
    "^IXIC", "^GSPC", "^DJI", "^SOX", "^TWII", "^TW50",
    "QQQ", "SPY", "DIA", "IWM",
}


def is_index_or_etf(ticker: str) -> bool:
    """判定是否為大盤類標的（指數 / ETF）。"""
    if not ticker:
        return False
    if ticker.startswith("^"):
        return True
    base = ticker.upper()
    if base in _INDEX_TICKERS:
        return True
    etf_keywords = ["QQQ", "SPY", "DIA", "IWM", "VOO", "VTI",
                    "00878", "0050", "0056", "00713", "00919", "00929", "00940", "00713"]
    return any(k in base for k in etf_keywords)


def detect_market_crash_signals(df, naaim_value=None, naaim_prev_max=None,
                                 aaii_bull=None, aaii_bear=None):
    """
    偵測大盤崩跌前兆（多維度加權打分）。
    
    分數規則（總分越高 = 風險越高）：
      1. 趨勢崩壞（跌破季線 + 月線死叉）：+2
      2. RSI 高位轉折（前 5 日過熱 → 急墜）：+2
      3. MACD 高位死叉（零軸上方死叉）：+1
      4. 爆量黑 K（量增 1.5x + 跌 2%）：+2
      5. 上漲乏力（連 3 紅但量縮）：+1
      6. 布林反轉下殺：+2
      7. ATR 飆升（波動 1.5x）：+1
      8. 跌破 120 日新低：+3
      9. NAAIM 從高點回落：+2
     10. AAII 散戶極度樂觀：+1
    
    等級：
      0-2：⚪ 安全
      3-4：🟡 注意
      5-7：🟠 警戒
      8+ ：🔴 崩跌警報
    """
    result = {
        "score": 0, "level": "safe",
        "level_label": "⚪ 大盤穩健", "level_color": "#22c55e",
        "signals": [], "summary": "目前大盤未出現崩跌前兆，可正常操作。",
    }
    if df is None or len(df) < 60:
        result["summary"] = "資料不足，無法判定大盤狀態。"
        return result

    latest = df.iloc[-1]
    prev = df.iloc[-2]
    score = 0
    sigs = []

    # 1. 趨勢崩壞
    try:
        ma20 = latest.get("SMA_20", np.nan)
        ma60 = latest.get("SMA_60", np.nan)
        ma20_p = prev.get("SMA_20", np.nan)
        ma60_p = prev.get("SMA_60", np.nan)
        if (not pd.isna(ma60) and latest["Close"] < ma60 and
                not pd.isna(ma20) and ma20 < ma60):
            score += 2
            sigs.append("⚠️ 趨勢崩壞（跌破季線 + 月線死叉季線）")
        elif (not pd.isna(ma20) and not pd.isna(ma20_p) and
              not pd.isna(ma60) and not pd.isna(ma60_p) and
              ma20 < ma60 and ma20_p >= ma60_p):
            score += 1
            sigs.append("⚠️ 月線剛死叉季線（趨勢轉空）")
    except Exception:
        pass

    # 2. RSI 高位轉折
    try:
        rsi_now = latest.get("RSI", 50)
        rsi_max5 = df["RSI"].iloc[-6:-1].max() if len(df) >= 6 else 50
        if not pd.isna(rsi_max5) and rsi_max5 > 75 and rsi_now < 50:
            score += 2
            sigs.append(f"⚠️ RSI 高位轉折（5日內過熱 {rsi_max5:.0f} → 急墜 {rsi_now:.0f}）")
        elif rsi_now > 75:
            score += 1
            sigs.append(f"⚠️ RSI 過熱 ({rsi_now:.0f})")
    except Exception:
        pass

    # 3. MACD 高位死叉
    try:
        macd, sig_l = latest.get("MACD", 0), latest.get("Signal_Line", 0)
        macd_p, sig_p = prev.get("MACD", 0), prev.get("Signal_Line", 0)
        if macd_p >= sig_p and macd < sig_l and macd > 0:
            score += 1
            sigs.append("⚠️ MACD 零軸上方死叉")
    except Exception:
        pass

    # 4. 爆量黑 K
    try:
        vol_5ma = latest.get("Vol_SMA5", latest["Volume"])
        if (latest["Volume"] > vol_5ma * 1.5 and
                latest["Close"] < latest["Open"] and
                latest["Open"] != 0 and
                (latest["Close"] - latest["Open"]) / latest["Open"] < -0.02):
            score += 2
            chg = (latest["Close"] - latest["Open"]) / latest["Open"] * 100
            sigs.append(f"🔴 爆量黑 K（量 {latest['Volume']/vol_5ma:.1f}x，跌 {chg:.1f}%）")
    except Exception:
        pass

    # 5. 上漲乏力
    try:
        if len(df) >= 4:
            last3 = df.iloc[-3:]
            all_red = (last3["Close"] > last3["Open"]).all()
            vol_dec = (last3["Volume"].iloc[0] > last3["Volume"].iloc[1] >
                       last3["Volume"].iloc[2])
            if all_red and vol_dec:
                score += 1
                sigs.append("⚠️ 上漲乏力（連 3 紅但量逐步萎縮）")
    except Exception:
        pass

    # 6. 布林反轉下殺
    try:
        last3 = df.iloc[-4:-1]
        bb_upper = last3["Bollinger_Upper"]
        touched = ((last3["High"] >= bb_upper * 0.99) & bb_upper.notna()).any()
        ma20 = latest.get("SMA_20", latest["Close"])
        if touched and not pd.isna(ma20) and latest["Close"] < ma20:
            score += 2
            sigs.append("🔴 布林反轉下殺（觸上軌後 3 日內跌破月線）")
    except Exception:
        pass

    # 7. ATR 飆升
    try:
        atr_now = latest.get("ATR", np.nan)
        atr_avg14 = df["ATR"].iloc[-15:-1].mean() if len(df) >= 15 else atr_now
        if not pd.isna(atr_now) and not pd.isna(atr_avg14) and atr_avg14 > 0:
            if atr_now > atr_avg14 * 1.5:
                score += 1
                sigs.append(f"⚠️ ATR 飆升（波動放大 {atr_now/atr_avg14:.1f}x）")
    except Exception:
        pass

    # 8. 跌破 120 日新低
    try:
        if len(df) >= 120:
            low_120 = df["Low"].iloc[-120:].min()
            if latest["Close"] < low_120 * 1.01:
                score += 3
                sigs.append("🚨 跌破 120 日新低（大級別支撐失守）")
    except Exception:
        pass

    # 9. NAAIM 高點回落
    if naaim_value is not None and naaim_prev_max is not None:
        if naaim_prev_max > 90 and naaim_value < 60:
            score += 2
            sigs.append(f"⚠️ 大戶減倉（NAAIM 從 {naaim_prev_max:.0f} 回落到 {naaim_value:.0f}）")
        elif naaim_value > 95:
            score += 1
            sigs.append(f"⚠️ NAAIM 過熱 ({naaim_value:.0f})")

    # 10. AAII 散戶極度樂觀
    if aaii_bull is not None and aaii_bear is not None:
        if aaii_bull > 50 and aaii_bear < 25:
            score += 1
            sigs.append(f"⚠️ AAII 散戶極度樂觀（多 {aaii_bull:.0f}% / 空 {aaii_bear:.0f}%）")

    # 等級判定
    if score >= 8:
        result.update({
            "level": "danger",
            "level_label": "🔴 大盤崩跌警報",
            "level_color": "#dc2626",
            "summary": f"觸發 {len(sigs)} 個訊號（總分 {score}），強烈建議【立即減碼避險】。",
        })
    elif score >= 5:
        result.update({
            "level": "warn",
            "level_label": "🟠 大盤警戒",
            "level_color": "#f97316",
            "summary": f"觸發 {len(sigs)} 個風險訊號（總分 {score}），建議【降低槓桿、分批出場】。",
        })
    elif score >= 3:
        result.update({
            "level": "watch",
            "level_label": "🟡 大盤注意",
            "level_color": "#facc15",
            "summary": f"出現 {len(sigs)} 個值得注意的訊號（總分 {score}），尚未崩跌但需提高警覺。",
        })

    result["score"] = score
    result["signals"] = sigs
    return result


# [修復 #1 & #2] 移除重複定義，統一放在頂層並加上 @st.cache_data
@st.cache_data(ttl=60)
def get_tw_all_macro():
    end_date = datetime.now()
    start_date = (end_date - timedelta(days=200)).strftime('%Y-%m-%d')
    url = "https://api.finmindtrade.com/api/v4/data"
    headers = {'User-Agent': 'Mozilla/5.0'}

    results = {
        "success": False, "oi_df": pd.DataFrame(), "retail_ratio": None,
        "current_tx_oi": None, "pc_ratio": None, "spot_df": pd.DataFrame(), "msg": ""
    }
    err_msgs = []

    def fetch(dataset, data_id=None):
        params = {"dataset": dataset, "start_date": start_date, "token": FINMIND_TOKEN}
        if data_id:
            params["data_id"] = data_id
        try:
            r = requests.get(url, params=params, headers=headers, timeout=10).json()
            if r.get('msg') == 'success' and r.get('data'):
                return pd.DataFrame(r['data'])
        except Exception as e:
            err_msgs.append(f"{dataset}: {str(e)[:15]}")
        return pd.DataFrame()

    def find_col(df, keywords):
        for c in df.columns:
            if any(k.lower() in str(c).lower() for k in keywords):
                return c
        return None

    def get_net_oi(df_subset):
        # [修復 #12] 計算 temp_net 時直接寫回原始 df_subset（呼叫方已 .copy()）
        net_col = find_col(df_subset, ['open_interest_net', 'net_oi', '未平仓淨'])
        if net_col:
            return net_col
        lc = find_col(df_subset, ['long_open_interest', '多方未平仓'])
        sc = find_col(df_subset, ['short_open_interest', '空方未平仓'])
        if lc and sc:
            df_subset['temp_net'] = (
                pd.to_numeric(df_subset[lc], errors='coerce') -
                pd.to_numeric(df_subset[sc], errors='coerce')
            )
            return 'temp_net'
        return None

    # 1. 大台
    df_tx = fetch("TaiwanFuturesInstitutionalInvestors", "TX")
    if not df_tx.empty:
        name_col = find_col(df_tx, ['name', 'institutional', 'investor', '法人', '名稱'])
        if name_col:
            df_f = df_tx[df_tx[name_col].astype(str).str.contains('外資|Foreign', case=False, na=False)].copy()
            if not df_f.empty:
                date_col = find_col(df_f, ['date', '日期', 'time'])
                net_col = get_net_oi(df_f)
                if date_col and net_col:
                    df_f['Date'] = pd.to_datetime(df_f[date_col])
                    df_f['Net_OI'] = pd.to_numeric(df_f[net_col], errors='coerce')
                    oi_df = df_f[['Date', 'Net_OI']].dropna().sort_values('Date').tail(120)
                    oi_df['OI_5MA'] = oi_df['Net_OI'].rolling(window=5).mean()
                    results["oi_df"] = oi_df
                    results["current_tx_oi"] = oi_df['Net_OI'].iloc[-1]
                    results["success"] = True

    # 2. 小台
    df_mtx = fetch("TaiwanFuturesInstitutionalInvestors", "MTX")
    df_tot = fetch("TaiwanFuturesDaily", "MTX")
    if not df_mtx.empty:
        dc_mtx = find_col(df_mtx, ['date', '日期'])
        nc_mtx = get_net_oi(df_mtx)
        if dc_mtx and nc_mtx:
            latest_d = df_mtx[dc_mtx].max()
            inst_net = pd.to_numeric(df_mtx[df_mtx[dc_mtx] == latest_d][nc_mtx], errors='coerce').sum()
            total_oi = 100000
            if not df_tot.empty:
                dc_tot = find_col(df_tot, ['date', '日期'])
                oc_tot = find_col(df_tot, ['open_interest', 'oi', '未平仓'])
                if dc_tot and oc_tot:
                    tot_row = df_tot[df_tot[dc_tot] == latest_d]
                    if not tot_row.empty:
                        val = pd.to_numeric(tot_row[oc_tot], errors='coerce').max()
                        if not pd.isna(val) and val > 0:
                            total_oi = val
            results["retail_ratio"] = round((-inst_net / total_oi) * 100, 2)

    # 3. 現貨
    df_spot = fetch("TaiwanStockTotalInstitutionalInvestors")
    if not df_spot.empty:
        name_col_s = find_col(df_spot, ['name', 'institutional', '法人'])
        if name_col_s:
            df_spot_f = df_spot[df_spot[name_col_s].astype(str).str.contains('外資|Foreign', case=False, na=False)].copy()
            date_col_s = find_col(df_spot_f, ['date', 'time'])
            buy_col = find_col(df_spot_f, ['buy', '買'])
            sell_col = find_col(df_spot_f, ['sell', '賣'])
            net_col_s = find_col(df_spot_f, ['net', '買賣超'])

            if date_col_s:
                df_spot_f['Date'] = pd.to_datetime(df_spot_f[date_col_s])
                if net_col_s:
                    df_spot_f['Net_Buy'] = pd.to_numeric(df_spot_f[net_col_s], errors='coerce') / 1e8
                elif buy_col and sell_col:
                    df_spot_f['Net_Buy'] = (
                        pd.to_numeric(df_spot_f[buy_col], errors='coerce') -
                        pd.to_numeric(df_spot_f[sell_col], errors='coerce')
                    ) / 1e8
                else:
                    df_spot_f['Net_Buy'] = 0
                results["spot_df"] = (
                    df_spot_f[['Date', 'Net_Buy']].dropna()
                    .groupby('Date').sum().reset_index()
                    .sort_values('Date').tail(120)
                )

    results["msg"] = " | ".join(err_msgs) if err_msgs else ""
    return results


# --- 3. 技術分析 ---
def calculate_volume_profile(df, bins=40, filter_mask=None):
    if df.empty:
        return pd.DataFrame({'Price': [], 'Volume': []})
    sub = df if filter_mask is None else df[filter_mask]
    b = np.linspace(df['Low'].min(), df['High'].max(), 41)
    c = (b[:-1] + b[1:]) / 2
    if sub.empty:
        return pd.DataFrame({'Price': c, 'Volume': np.zeros(40)})
    return pd.DataFrame({
        'Price': c,
        'Volume': sub.groupby(pd.cut(sub['Close'], b, labels=False, include_lowest=True))['Volume']
                     .sum().reindex(range(40), fill_value=0).values
    })


def calculate_indicators(df):
    if len(df) < 50:
        return df
    df['SMA_5'] = df['Close'].rolling(5).mean()
    df['SMA_10'] = df['Close'].rolling(10).mean()
    df['SMA_20'] = df['Close'].rolling(20).mean()
    df['SMA_60'] = df['Close'].rolling(60).mean()
    df['SMA_200'] = df['Close'].rolling(200).mean()
    df['Vol_SMA5'] = df['Volume'].rolling(5).mean()
    df['Vol_SMA20'] = df['Volume'].rolling(20).mean()
    df['Vol_SMA60'] = df['Volume'].rolling(60).mean()
    # [v27] OBV (用於 7 維 context)
    df['_price_diff'] = df['Close'].diff()
    df['_obv_change'] = np.where(df['_price_diff'] > 0, df['Volume'],
                                   np.where(df['_price_diff'] < 0, -df['Volume'], 0))
    df['OBV'] = df['_obv_change'].cumsum()

    sd = df['Close'].rolling(20).std()
    df['Bollinger_Upper'] = df['SMA_20'] + sd * 2
    df['Bollinger_Lower'] = df['SMA_20'] - sd * 2

    df['KC_Upper'] = df['SMA_20'] + df['SMA_20'] * 0.05
    df['KC_Lower'] = df['SMA_20'] - df['SMA_20'] * 0.05
    df['Squeeze_On'] = (df['Bollinger_Upper'] < df['KC_Upper']) & (df['Bollinger_Lower'] > df['KC_Lower'])

    tr = np.max(pd.concat([
        df['High'] - df['Low'],
        np.abs(df['High'] - df['Close'].shift()),
        np.abs(df['Low'] - df['Close'].shift())
    ], axis=1), axis=1)
    df['ATR'] = tr.rolling(14).mean()
    df['Vol_60D_Avg'] = ((df['ATR'] / df['Close']) * 100).rolling(60).mean()
    # [修復 #8] ATR 追蹤停損改用 Close.rolling.max，避免急跌時停損位永遠貼著天花板
    df['ATR_Trailing_Stop'] = df['Close'].rolling(22).max() - df['ATR'] * 3

    # [修復 #6] RSI 加上除零保護，極端行情（全漲/全跌）不產生 NaN
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    loss_safe = loss.replace(0, np.nan)          # 0 → NaN，避免除零
    rs = gain / loss_safe
    df['RSI'] = 100 - (100 / (1 + rs))
    # fillna 不接受 ndarray，需包成 pd.Series
    rsi_fill = pd.Series(
        np.where(gain.isna(), 50, np.where(loss == 0, 100, 0)),
        index=df.index
    )
    df['RSI'] = df['RSI'].fillna(rsi_fill)

    df['MACD'] = df['Close'].ewm(span=12, adjust=False).mean() - df['Close'].ewm(span=26, adjust=False).mean()
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['Signal_Line']

    clv = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'])
    df['AD_Line'] = (clv.fillna(0) * df['Volume']).cumsum()

    df['DMA_DDD'] = df['Close'].rolling(10).mean() - df['Close'].rolling(50).mean()
    df['DMA_AMA'] = df['DMA_DDD'].rolling(10).mean()

    buy_seq = np.zeros(len(df), dtype=int)
    sell_seq = np.zeros(len(df), dtype=int)
    for i in range(4, len(df)):
        buy_seq[i] = buy_seq[i - 1] + 1 if df['Close'].iloc[i] < df['Close'].iloc[i - 4] else 0
        sell_seq[i] = sell_seq[i - 1] + 1 if df['Close'].iloc[i] > df['Close'].iloc[i - 4] else 0
    df['TD_Buy_Seq'] = buy_seq
    df['TD_Sell_Seq'] = sell_seq
    return df


def find_structural_box_bottom(df, current_price):
    if len(df) < 60:
        return current_price, df.index[0], df.index[-1], False
    p = df.tail(120).copy()
    p['M'] = p['Low'].rolling(5, center=True, min_periods=1).min()
    v = p[(p['Low'] == p['M']) & (p['Low'] < current_price * 0.98)]
    if not v.empty:
        last_pivot = v.iloc[-1]
        # [修復 #9] 改用 get_loc 避免重複 index 造成的錯誤
        try:
            loc = p.index.get_loc(last_pivot.name)
            if isinstance(loc, slice):
                loc = loc.start
            elif isinstance(loc, np.ndarray):
                loc = int(np.where(loc)[0][0])
        except Exception:
            loc = 0
        start_loc = max(0, loc - 15)
        end_loc = min(len(p) - 1, loc + 15)
        return last_pivot['Low'], p.index[start_loc], p.index[end_loc], False
    return p['Low'].min(), p.index[0], p.index[-1], True


def generate_projection_points(df, trend_text, cur_p, iron_p, is_brk, ph_support, ph_resist):
    last_d = df.index[-1]
    f_d = []
    d = last_d
    while len(f_d) < 30:
        d += pd.Timedelta(days=1)
        if d.weekday() < 5:
            f_d.append(d)
    x = [last_d]
    y = [cur_p]
    ma20 = df['SMA_20'].iloc[-1] if not pd.isna(df['SMA_20'].iloc[-1]) else cur_p
    rsi = df['RSI'].iloc[-1] if not pd.isna(df['RSI'].iloc[-1]) else 50
    td_sell_cur = df['TD_Sell_Seq'].iloc[-1]
    recent_high_20 = df['High'].tail(20).max()
    is_relief_rally = (rsi < 60 and trend_text == "🐻 熊市") or (rsi > 70)
    support_level = ph_support if ph_support is not None else iron_p
    resist_level = ph_resist if ph_resist is not None else float('inf')

    range_percent = (resist_level - support_level) / cur_p * 100 if resist_level != float('inf') and support_level > 0 else 100.0
    is_squeezed = range_percent < 5.0
    scenario_name = ""

    if is_squeezed:
        if cur_p > ma20 or "牛市" in trend_text:
            scenario_name = "🚀 極限壓縮<br>N字噴發"
            up_1 = max(cur_p * 1.03, resist_level * 1.025)
            dip = max(cur_p * 1.01, resist_level)
            up_2 = max(up_1 * 1.05, recent_high_20 * 1.03)
            x.extend([f_d[4], f_d[9], f_d[18]])
            y.extend([up_1, dip, up_2])
        else:
            scenario_name = "💥 極限壓縮<br>倒N崩盤"
            down_1 = min(cur_p * 0.97, support_level * 0.975)
            bounce = min(cur_p * 0.99, support_level)
            down_2 = max(iron_p, down_1 * 0.92)
            x.extend([f_d[4], f_d[9], f_d[18]])
            y.extend([down_1, bounce, down_2])
    else:
        if "牛市" in trend_text:
            scenario_name = "🐂 牛市格局<br>N字突破"
            dip = max(cur_p * 0.96, ma20, support_level)
            rally = min(cur_p * 1.05, resist_level * 0.99) if resist_level != float('inf') else max(cur_p * 1.05, recent_high_20 * 1.02)
            x.extend([f_d[4], f_d[14]])
            y.extend([dip, rally])
        elif "熊市" in trend_text:
            if is_brk:
                scenario_name = "🕳️ 熊市格局<br>無底洞墜落"
                bounce = min(cur_p * 1.03, ma20, resist_level)
                drop = cur_p * 0.92
                x.extend([f_d[4], f_d[14]])
                y.extend([bounce, drop])
            elif is_relief_rally:
                days_to_peak = max(1, (9 - td_sell_cur) if (0 < td_sell_cur < 9) else 3)
                scenario_name = f"🐻 熊市打底<br>遇壓測底" if (0 < td_sell_cur < 9) else "🐻 熊市打底<br>二次測底"
                exhaustion_p = min(max(cur_p * 1.01, ma20), resist_level * 0.99 if resist_level != float('inf') else float('inf'))
                w_dip = max(iron_p, iron_p * 1.015, support_level, cur_p * 0.93) if iron_p > 0 else cur_p * 0.9
                breakout_p = max(exhaustion_p * 1.05, recent_high_20 * 1.02, cur_p * 1.1)
                x.extend([f_d[days_to_peak], f_d[days_to_peak + 5], f_d[days_to_peak + 15]])
                y.extend([exhaustion_p, w_dip, breakout_p])
            else:
                scenario_name = "🐻 熊市格局<br>下降通道"
                bounce = min(cur_p * 1.06, ma20, resist_level)
                drop = max(iron_p, cur_p * 0.9)
                x.extend([f_d[4], f_d[14]])
                y.extend([bounce, drop])
        else:
            if ph_support and cur_p > ph_support:
                scenario_name = "⚖️ 區間突破<br>回測支撐"
                up_1 = min(cur_p * 1.04, resist_level * 0.99 if resist_level != float('inf') else float('inf'))
                dip = max(cur_p * 0.96, support_level)
                up_2 = max(up_1, dip * 1.05)
                x.extend([f_d[5], f_d[12], f_d[18]])
                y.extend([up_1, dip, up_2])
            else:
                scenario_name = "⚖️ 區間震盪<br>挑戰壓力"
                up_1 = min(cur_p * 1.04, resist_level * 0.99 if resist_level != float('inf') else float('inf'))
                dip = max(cur_p * 0.96, iron_p, support_level)
                up_2 = resist_level * 1.01 if resist_level != float('inf') else cur_p * 1.05
                x.extend([f_d[5], f_d[12], f_d[18]])
                y.extend([up_1, dip, up_2])
    return x, y, scenario_name


def analyze_market_trend(df):
    c, m20, m60 = df['Close'].iloc[-1], df['SMA_20'].iloc[-1], df['SMA_60'].iloc[-1]
    if c > m20 > m60:
        return "🐂 牛市", "多頭排列", "sig-green"
    elif c < m20 < m60:
        return "🐻 熊市", "空頭排列", "sig-red"
    else:
        return "⚖️ 震盪", "區間整理", "sig-orange"


def get_stock_engine_mode(ticker, df_data):
    if ticker.startswith("^") or any(e in ticker for e in ["QQQ", "SPY", "DIA", "0050.TW"]):
        return "🏢 權值大盤", "trend"
    try:
        m = yf.Ticker(ticker).info.get('marketCap', 0)
        v = df_data['Vol_60D_Avg'].iloc[-1] if not df_data.empty else 3.0
        l = m >= (300e9 if ".TW" in ticker else 10e9) or (m == 0 and v < 3.5)
        if l and v < 4.0:
            return "🏢 權值穩健", "trend"
        elif l:
            return "🚀 巨型動能", "momentum"
        else:
            return "🎢 妖股轉折", "reversal"
    except Exception:
        return "🎢 動態模式", "reversal"


def get_relative_strength(ticker, stock_df):
    try:
        b = yf.download("^TWII" if ".TW" in ticker else "^GSPC", period="1mo", progress=False)['Close'].iloc[:, 0]
        a = stock_df['Close'].reindex(b.index, method='ffill')
        if len(b) > 20:
            diff = ((a.iloc[-1] - a.iloc[-20]) / a.iloc[-20]) - ((b.iloc[-1] - b.iloc[-20]) / b.iloc[-20])
            if diff > 0.05:
                return "🦁 領頭羊 (強)", "sig-green"
            elif diff > 0:
                return "🐯 優於大盤", "sig-blue"
            else:
                return "🐶 落後股 (弱)", "sig-gray"
    except Exception:
        pass
    return "⚖️ 跟隨大盤", "sig-gray"


def detect_smart_money_status(df):
    if len(df) < 10:
        return None
    latest = df.iloc[-1]
    price_now = latest['Close']
    price_5d = df['Close'].iloc[-6]
    ad_now = latest['AD_Line']
    ad_5d = df['AD_Line'].iloc[-6]
    rsi = latest['RSI']
    if price_now > latest['Open'] and latest['Low'] < latest['Bollinger_Lower'] and rsi < 40 and price_now > price_5d * 0.95:
        return "💎 破底翻買點"
    if latest['Close'] < latest['Bollinger_Lower'] and latest['RSI'] < 30:
        return "⚡ 乖離抄底 (超賣)"
    if price_now < price_5d * 0.98 and ad_now > ad_5d and rsi < 50:
        return "🎯 主力背離吸籌"
    if (rsi > 65 and latest['Volume'] > latest['Vol_SMA5'] * 1.3 and
            (latest['Close'] < latest['Open'] or
             (latest['High'] - max(latest['Open'], latest['Close']) > abs(latest['Close'] - latest['Open']) * 1.5))):
        return "🔴 主力調節 (爆量滯漲)"
    if rsi < 30 and latest['Volume'] > latest['Vol_SMA5']:
        return "⚡ 恐慌殺盤"
    return None


def get_compre_color_class(text):
    tl = text.lower()
    if any(w in tl for w in ['突破', '回測支撐', 'n字', '看漲']):
        return 'sig-green'
    if any(w in tl for w in ['跌破', '崩盤', '測底', '無底洞']):
        return 'sig-red'
    if any(w in tl for w in ['整理', '震盪', '挑戰壓力']):
        return 'sig-orange'
    return 'sig-gray'


def analyze_strategic_signals(df):
    if df.empty:
        return {}
    latest = df.iloc[-1]
    macd, signal = latest['MACD'], latest['Signal_Line']
    if macd > signal and macd > 0:
        macd_text, macd_color = "零軸上金叉", "sig-green"
    elif macd > signal:
        macd_text, macd_color = "零軸下金叉", "sig-orange"
    elif macd > 0:
        macd_text, macd_color = "零軸上死叉", "sig-orange"
    else:
        macd_text, macd_color = "零軸下死叉", "sig-red"

    vol, vol_ma = latest['Volume'], latest['Vol_SMA5']
    if vol > vol_ma * 1.5:
        vol_text, vol_color = "爆量", "sig-green"
    elif vol > vol_ma * 1.1:
        vol_text, vol_color = "量增", "sig-green"
    else:
        vol_text, vol_color = "量縮", "sig-gray"

    rsi = latest['RSI']
    if rsi > 70:
        rsi_text, rsi_color = f"{rsi:.0f} 過熱", "sig-red"
    elif rsi < 30:
        rsi_text, rsi_color = f"{rsi:.0f} 超賣", "sig-green"
    else:
        rsi_text, rsi_color = f"{rsi:.0f}", "sig-gray"

    summary, summary_color = "觀望", "sig-gray"
    if latest.get('Squeeze_On', False):
        summary, summary_color = "🌀 壓縮蓄力中", "sig-cyan"
    elif macd > signal:
        summary, summary_color = "📈 偏多震盪", "sig-green"
    else:
        summary, summary_color = "⛈️ 空頭走勢", "sig-red"

    status = detect_smart_money_status(df)
    if status:
        summary = status
        summary_color = "sig-red" if "調節" in status else "sig-purple"

    return {
        "MACD_Text": macd_text, "MACD_Color": macd_color,
        "Vol_Text": vol_text, "Vol_Color": vol_color,
        "RSI_Text": rsi_text, "RSI_Color": rsi_color,
        "Summary": summary, "Summary_Color": summary_color
    }


def predict_target_and_rating(df, mc_result=None):
    """
    短/長期目標 + 強弱評等。
    優先用蒙地卡羅 p50/p90 反映當下趨勢，無 MC 結果時退回布林+60日高點（舊邏輯）。
    """
    price = df['Close'].iloc[-1]
    
    if mc_result is not None and mc_result.get("p50_final") is not None:
        # 新邏輯：用 MC 30 日後 p50 當短期、p90 當長期
        t_s = float(mc_result["p50_final"])
        t_l = float(mc_result["p90_final"])
    else:
        # 退回舊邏輯（MC 失敗時的 fallback）
        upper = df['Bollinger_Upper'].iloc[-1]
        recent_high_60 = df['High'].tail(60).max()
        t_s = upper if price >= recent_high_60 else min(upper, recent_high_60)
        t_l = max(recent_high_60 * 1.15, t_s * 1.1)
    
    rating = "強勢" if price > df['SMA_20'].iloc[-1] else "持有"
    return t_s, t_l, rating


def get_technical_target_threshold(df):
    """技術面「達標」閾值（給走勢圖標籤用，與 AI 目標 t_s 分離）
    
    這個閾值用於判斷「K 線是否觸及短期壓力上緣」，與蒙地卡羅 p50 中位數預期不同。
    永遠回傳一個「真實會被觸及」的價位（布林上軌 / 60 日高點 / 季布林上軌）。
    """
    price = float(df['Close'].iloc[-1])
    upper = float(df['Bollinger_Upper'].iloc[-1])
    recent_high_60 = float(df['High'].tail(60).max())
    
    # 若當前股價已逼近或突破 60 日高，用布林上軌
    # 否則用 min(布林上軌, 60日高)，這保留「先碰小目標再衝大目標」的層次感
    if price >= recent_high_60 * 0.98:
        threshold = upper
    else:
        threshold = min(upper, recent_high_60)
    
    return threshold


def format_volume(num):
    if num >= 1e9:
        return f"{num / 1e9:.2f}B"
    elif num >= 1e6:
        return f"{num / 1e6:.2f}M"
    return f"{num}"


def detect_launch_points(df: pd.DataFrame) -> pd.Series:
    """
    偵測「起漲點」：MA 黃金交叉 + 近期曾超賣（RSI 20根最低 < 45）。
    只有兩個條件同時成立才亮燈，避免假訊號。
    回傳 bool Series，index 與 df 相同。
    """
    if len(df) < 61:
        return pd.Series(False, index=df.index)
    ma_gold = (
        (df['SMA_20'] > df['SMA_60']) &
        (df['SMA_20'].shift(1) <= df['SMA_60'].shift(1))
    )
    rsi_was_oversold = df['RSI'].rolling(20, min_periods=1).min() < 45
    return ma_gold & rsi_was_oversold


def get_launch_macd_points(df: pd.DataFrame) -> pd.Series:
    """
    MACD 金叉 + 同時滿足起漲條件（RSI 剛從低位回升、收盤在 SMA_20 附近以內）。
    用來在 MACD 副圖的金叉三角上疊加特殊 ⭐ 起漲標記。
    """
    if len(df) < 30:
        return pd.Series(False, index=df.index)
    macd_gold = (
        (df['MACD'] > df['Signal_Line']) &
        (df['MACD'].shift(1) <= df['Signal_Line'].shift(1))
    )
    rsi_low = df['RSI'].rolling(10, min_periods=1).min() < 50
    price_low = df['Close'] <= df['SMA_60'] * 1.05   # 仍在季線附近（剛起漲）
    return macd_gold & rsi_low & price_low



    if num >= 1e9:
        return f"{num / 1e9:.2f}B"
    elif num >= 1e6:
        return f"{num / 1e6:.2f}M"
    return f"{num}"


@st.cache_data(ttl=300)
def get_earnings_status(ticker):
    ignore_list = ["0050", "0056", "00878", "QQQ", "SPY", "DIA", "IWM", "^TWII", "^IXIC"]
    if any(x in ticker for x in ignore_list) or "00403A" in ticker:
        return "", ""
    next_date = "N/A"
    last_result = "⚪ 無前季數據"
    try:
        t = yf.Ticker(ticker)
        hist = None
        try:
            hist = t.get_earnings_dates(limit=12)
        except Exception:
            try:
                hist = t.earnings_dates
            except Exception:
                pass
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
                        if act > 0 and act >= est:
                            last_result = f'<span class="earn-beat">🟢 獲利優於預期 (EPS: {act:.2f}|估:{est:.2f})</span>'
                        elif act <= 0 and act >= est:
                            last_result = f'<span class="earn-turn">🟡 虧損優於預期 (EPS: {act:.2f}|估:{est:.2f})</span>'
                        elif act > 0 and act < est:
                            last_result = f'<span class="earn-warn">🟠 獲利遜於預期 (EPS: {act:.2f}|估:{est:.2f})</span>'
                        else:
                            last_result = f'<span class="earn-miss">🔴 虧損遜於預期 (EPS: {act:.2f}|估:{est:.2f})</span>'
                    elif pd.notna(last.get('Surprise(%)')):
                        sur = last['Surprise(%)']
                        if sur > 0:
                            last_result = f'<span class="earn-beat">🟢 優於預期 (+{sur:.1f}%)</span>'
                        else:
                            last_result = f'<span class="earn-miss">🔴 遜於預期 ({sur:.1f}%)</span>'
            except Exception:
                pass
        if next_date == "N/A":
            try:
                cal = t.calendar
                if isinstance(cal, dict) and 'Earnings Date' in cal and len(cal['Earnings Date']) > 0:
                    next_date = cal['Earnings Date'][0].strftime('%Y-%m-%d')
                elif isinstance(cal, pd.DataFrame) and 'Earnings Date' in cal.columns and not cal.empty:
                    next_date = pd.to_datetime(cal['Earnings Date'].iloc[0]).strftime('%Y-%m-%d')
            except Exception:
                pass
        return f"📅 財報: {next_date}", last_result
    except Exception:
        return "📅 財報: N/A", "⚪ 無數據"


def get_tactical_advice(df, cur_p, t_s, iron_p, ph_support, ph_resist, bias_ma5, bias_limit, engine_type):
    if len(df) < 10:
        return "⌛ 數據不足", "等待更多 K 線資料寫入...", "#9ca3af"
    latest = df.iloc[-1]
    td_b = latest.get('TD_Buy_Seq', 0)
    td_s = latest.get('TD_Sell_Seq', 0)
    ma20 = latest.get('SMA_20', cur_p)
    rsi = latest.get('RSI', 50)
    s_level = ph_support if ph_support is not None else iron_p
    r_level = ph_resist if ph_resist is not None else float('inf')
    b_upper = latest.get('Bollinger_Upper', float('inf'))
    b_lower = latest.get('Bollinger_Lower', 0)

    if cur_p > b_upper and bias_ma5 >= bias_limit:
        return "🚨 雙重鎖定 (極限逃頂)", f"【強烈警告】股價已飛出布林上軌，且 MA5 正乖離達 {bias_ma5:.1f}% (超過該股性極限 {bias_limit}%)！橡皮筋瀕臨斷裂，主力倒貨風險極高，請立即【逢高收網、入袋為安】！", "#ef4444"
    if cur_p < b_lower and bias_ma5 <= -bias_limit:
        return "💎 雙重鎖定 (極度恐慌底)", f"【極度超賣】股價已跌穿布林下軌，且 MA5 負乖離達 {bias_ma5:.1f}%！市場出現非理性殺盤，這通常是阿呆谷，請隨時準備迎接『破底翻』報復性反彈！", "#10b981"
    if cur_p > b_upper * 1.02:
        return "🔥 軌道噴發 (高度警戒)", "目前股價已飛出布林上軌，雖然動能極強，但隨時面臨暴力回檔均線的風險。嚴禁追高！", "#ef4444"
    if bias_ma5 >= bias_limit * 0.8:
        return "⚠️ 短線超買 (乖離偏高)", f"MA5 正乖離達 {bias_ma5:.1f}%，已逼近該股性極限。短線追高動能耗盡，請勿追高！", "#f97316"
    if r_level != float('inf') and s_level > 0 and ((r_level - s_level) / cur_p * 100) < 5.0:
        return "🌪️ 即將引爆預警 (極限收斂)", f"支撐(${s_level:.2f})與壓力(${r_level:.2f})空間已被極度壓縮至 5% 內，猶如壓緊的彈簧！隨時面臨【暴力表態】，請密切關注突破方向，順勢操作！", "#d946ef"
    if td_s >= 8:
        return "⚠️ 動能倒數 (留意轉折)", f"上漲動能已達極限 (目前已亮出 TD {int(td_s)})。即將面臨時間轉折賣壓，強烈建議【嚴禁追高】，持有多單者請準備【分批出場】！", "#ef4444"

    rsi_limit = 85 if engine_type in ["momentum", "reversal"] else 75
    if rsi > rsi_limit:
        return "🔥 極度超買 (指標高檔)", f"RSI 已達 {rsi:.0f} 的極端高位 (該股性危險值為 {rsi_limit})。雖然技術面強勢，但隨時可能面臨劇烈洗盤，空手者請勿進場當最後一隻老鼠！", "#ef4444"
    if 1 <= td_s <= 7 and cur_p > ma20:
        return "🚀 主升歡樂帶 (抱緊處理)", f"目前上漲動能強勁 (TD {int(td_s)})，且穩居月線之上。正處於上升通道，請【沿著均線續抱】，享受獲利，不輕易下車！", "#22c55e"
    if td_b >= 8 or rsi < 30:
        return "💎 空頭力竭 (準備破底翻)", f"殺盤動能即將耗盡 (目前 TD 下跌 {int(td_b)} 或是 RSI超賣)，股價已進入絕佳的潛在支撐區。請密切關注反轉向上的【右側買點】！", "#facc15"
    if 1 <= td_b <= 7 and cur_p < ma20:
        return "🔪 恐慌殺盤中 (嚴禁接刀)", f"目前處於主跌段 (TD 下跌 {int(td_b)})，賣壓極度沉重。在底部紅字 8 或 9 出現之前，請綁好雙手【絕對不要進場買進】！", "#f97316"
    if cur_p > ma20:
        return "📈 多頭延續 (沿均線續抱)", "目前股價穩居月線 (MA20) 之上，多頭格局不變。建議沿均線續抱。", "#22c55e"
    return "⚖️ 震盪整理 (耐心觀望)", "目前股價處於區間震盪，方向尚未明確。建議耐心觀望，等待突破或測底訊號。", "#9ca3af"


# --- 4. 側邊控制台 ---
with st.sidebar:
    st.title("🎛️ 控制台")
    mobile_mode = st.toggle("啟用手機防卡死模式", value=False)
    st.markdown("---")

    st.header("📌 多維度自選股清單")
    cur_t = st.session_state.get('current_ticker', "^NDX")
    act_l = st.session_state.get('active_list')
    wls = st.session_state['watchlists']

    # ── 多選模式 toggle ─────────────────────────────────
    multi_mode = st.toggle(
        "☑️ 多選編輯模式",
        value=st.session_state.get('multi_mode', False),
        key='multi_mode_toggle',
        help="開啟後可勾選多檔股票，批次移動、複製、刪除或排序",
    )
    st.session_state['multi_mode'] = multi_mode

    # 初始化選取集合
    if 'selected_stocks' not in st.session_state:
        st.session_state['selected_stocks'] = set()  # {(wl_name, ticker)}

    # ── 多選模式 UI ─────────────────────────────────────
    if multi_mode:
        st.caption("💡 勾選股票 → 下方選擇批次動作")
        # 列出所有清單與股票（含 checkbox）
        for wl_name, tickers in wls.items():
            with st.expander(
                f"{wl_name} ({len(tickers)})",
                expanded=(wl_name == st.session_state.get('user_opened_list')),
            ):
                # 全選/取消全選 按鈕（小巧）
                sel_col1, sel_col2 = st.columns(2)
                if sel_col1.button(f"☑️ 全選", key=f"sel_all_{wl_name}",
                                     use_container_width=True):
                    for t in tickers:
                        st.session_state['selected_stocks'].add((wl_name, t))
                    st.rerun()
                if sel_col2.button(f"☐ 全消", key=f"sel_none_{wl_name}",
                                     use_container_width=True):
                    for t in tickers:
                        st.session_state['selected_stocks'].discard((wl_name, t))
                    st.rerun()
                # 每檔股票的 checkbox
                for t in tickers:
                    key = (wl_name, t)
                    s_name = get_stock_name(t)
                    disp = f"{s_name} ({t})" if s_name != t else t
                    is_checked = key in st.session_state['selected_stocks']
                    new_val = st.checkbox(disp, value=is_checked,
                                            key=f"cb_{wl_name}_{t}")
                    if new_val and not is_checked:
                        st.session_state['selected_stocks'].add(key)
                    elif not new_val and is_checked:
                        st.session_state['selected_stocks'].discard(key)

        # ── 批次動作面板 ────────────────────────────────
        st.markdown("---")
        sel_count = len(st.session_state['selected_stocks'])
        if sel_count == 0:
            st.info("👆 請先勾選股票")
        else:
            st.markdown(f"**已選 {sel_count} 檔**")

            # 批次移動 / 複製
            target_list = st.selectbox(
                "📦 目標清單",
                list(wls.keys()),
                key="batch_target_list",
            )
            bc1, bc2 = st.columns(2)
            if bc1.button(f"✂️ 移動到「{target_list}」", use_container_width=True):
                moved = 0
                for (src_wl, t) in list(st.session_state['selected_stocks']):
                    if src_wl != target_list and t in wls.get(src_wl, []):
                        wls[src_wl].remove(t)
                        if t not in wls[target_list]:
                            wls[target_list].append(t)
                        moved += 1
                if moved:
                    save_watchlists(wls)
                    st.session_state['selected_stocks'].clear()
                    st.success(f"✅ 移動 {moved} 檔到「{target_list}」")
                    st.rerun()
            if bc2.button(f"📋 複製到「{target_list}」", use_container_width=True):
                copied = 0
                for (src_wl, t) in st.session_state['selected_stocks']:
                    if t not in wls.get(target_list, []):
                        wls[target_list].append(t)
                        copied += 1
                if copied:
                    save_watchlists(wls)
                    st.success(f"✅ 複製 {copied} 檔到「{target_list}」")
                    st.rerun()

            st.markdown("---")
            # 批次排序（同清單內）
            st.markdown("**📦 批次排序（同清單內）**")
            sc1, sc2, sc3, sc4 = st.columns(4)
            def _batch_reorder(direction):
                """direction: top/up/down/bottom"""
                changed = False
                # 依清單分組
                from collections import defaultdict
                by_list = defaultdict(list)
                for (wl, t) in st.session_state['selected_stocks']:
                    by_list[wl].append(t)
                for wl, ts in by_list.items():
                    lst = wls.get(wl, [])
                    # 取得這些選中股票在清單中的索引（保持原順序）
                    idx_list = sorted([lst.index(t) for t in ts if t in lst])
                    if not idx_list:
                        continue
                    # 取出選中的股票（保持原順序），其餘照舊
                    selected_in_order = [lst[i] for i in idx_list]
                    others = [t for i, t in enumerate(lst) if i not in idx_list]
                    if direction == "top":
                        new_lst = selected_in_order + others
                    elif direction == "bottom":
                        new_lst = others + selected_in_order
                    elif direction == "up":
                        # 每個選中的整體上移一格（保持相對順序）
                        new_lst = lst[:]
                        first_idx = idx_list[0]
                        if first_idx > 0:
                            # 上面那檔下移到選中段之後
                            above = new_lst[first_idx - 1]
                            del new_lst[first_idx - 1]
                            insert_pos = first_idx - 1 + len(idx_list)
                            new_lst.insert(insert_pos, above)
                    elif direction == "down":
                        new_lst = lst[:]
                        last_idx = idx_list[-1]
                        if last_idx < len(new_lst) - 1:
                            below = new_lst[last_idx + 1]
                            del new_lst[last_idx + 1]
                            new_lst.insert(idx_list[0], below)
                    else:
                        continue
                    wls[wl] = new_lst
                    changed = True
                return changed

            if sc1.button("⏫ 置頂", use_container_width=True):
                if _batch_reorder("top"):
                    save_watchlists(wls); st.rerun()
            if sc2.button("⬆️ 上移", use_container_width=True):
                if _batch_reorder("up"):
                    save_watchlists(wls); st.rerun()
            if sc3.button("⬇️ 下移", use_container_width=True):
                if _batch_reorder("down"):
                    save_watchlists(wls); st.rerun()
            if sc4.button("⏬ 置底", use_container_width=True):
                if _batch_reorder("bottom"):
                    save_watchlists(wls); st.rerun()

            st.markdown("---")
            # 危險區
            if st.button(f"🗑️ 批次刪除 {sel_count} 檔",
                          type="primary", use_container_width=True):
                for (wl, t) in list(st.session_state['selected_stocks']):
                    if t in wls.get(wl, []):
                        wls[wl].remove(t)
                save_watchlists(wls)
                # 如果目前看的股票被刪了，切換
                if (act_l, cur_t) in st.session_state['selected_stocks']:
                    if wls.get(act_l):
                        st.session_state['current_ticker'] = wls[act_l][0]
                    else:
                        st.session_state['current_ticker'] = "^NDX"
                st.session_state['selected_stocks'].clear()
                st.rerun()

    else:
        # ── 點按模式（既有 UI） ─────────────────────────
        for wl_name, tickers in wls.items():
            is_exp = (wl_name == st.session_state.get('user_opened_list'))
            with st.expander(f"{wl_name} ({len(tickers)})", expanded=is_exp):
                for t in tickers:
                    is_sel = (t == cur_t and wl_name == act_l)
                    s_name = get_stock_name(t)
                    disp = f"{s_name} ({t})" if s_name != t else t
                    if st.button(
                        f"{'▶ ' if is_sel else ''}{disp}",
                        key=f"btn_{wl_name}_{t}",
                        type="primary" if is_sel else "secondary",
                        use_container_width=True,
                    ):
                        st.session_state['current_ticker'] = t
                        st.session_state['active_list'] = wl_name
                        st.session_state['user_opened_list'] = wl_name
                        st.rerun()

    st.markdown("---")
    # ── 排列按鈕（單一股票，非多選模式時可用） ──────
    if not multi_mode:
        st.markdown("<span style='color:gray; font-size:12px;'>📦 排列目前代碼</span>",
                    unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        lst = wls.get(act_l, [])
        idx = lst.index(cur_t) if cur_t in lst else -1
        if c1.button("⏫", help="置頂") and idx > 0:
            lst.insert(0, lst.pop(idx)); save_watchlists(wls); st.rerun()
        if c2.button("⬆️", help="上移") and idx > 0:
            lst[idx], lst[idx-1] = lst[idx-1], lst[idx]; save_watchlists(wls); st.rerun()
        if c3.button("⬇️", help="下移") and 0 <= idx < len(lst)-1:
            lst[idx], lst[idx+1] = lst[idx+1], lst[idx]; save_watchlists(wls); st.rerun()
        if c4.button("⏬", help="置底") and 0 <= idx < len(lst)-1:
            lst.append(lst.pop(idx)); save_watchlists(wls); st.rerun()

    st.markdown("---")
    time_opt = st.radio("選擇週期", ["當沖 (分時)", "日線 (Daily)", "週線 (Weekly)"], index=1)

    # ── 編輯清單（展開） ────────────────────────────────
    with st.expander("✏️ 編輯清單"):
        st.markdown("**➕ 新增單一股票**")
        col_inp, col_tgt = st.columns([1, 1])
        new_t = col_inp.text_input("代號", placeholder="2330.TW",
                                    label_visibility="collapsed").upper().strip()
        target_list = col_tgt.selectbox(
            "加入清單", list(wls.keys()),
            index=list(wls.keys()).index(act_l) if act_l in wls else 0,
            label_visibility="collapsed",
        )
        if st.button("➕ 新增", use_container_width=True) and new_t:
            if new_t not in wls[target_list]:
                wls[target_list].append(new_t)
                st.session_state['current_ticker'] = new_t
                st.session_state['active_list'] = target_list
                st.session_state['user_opened_list'] = target_list
                save_watchlists(wls)
                st.rerun()

        st.markdown("---")
        st.markdown("**📋 批量匯入（換行或逗號分隔）**")
        bulk_raw = st.text_area(
            "批量貼上代號", placeholder="NVDA\nAMD\nTSLA\n或 NVDA, AMD, TSLA",
            height=80, label_visibility="collapsed",
        )
        bulk_target = st.selectbox(
            "匯入到", list(wls.keys()),
            index=list(wls.keys()).index(act_l) if act_l in wls else 0,
            key="bulk_target",
        )
        if st.button("📥 批量匯入", use_container_width=True) and bulk_raw.strip():
            raw_tickers = [
                t.strip().upper()
                for t in bulk_raw.replace(",", "\n").splitlines()
                if t.strip()
            ]
            added = []
            for t in raw_tickers:
                if t and t not in wls[bulk_target]:
                    wls[bulk_target].append(t)
                    added.append(t)
            if added:
                save_watchlists(wls)
                st.session_state['active_list'] = bulk_target
                st.session_state['user_opened_list'] = bulk_target
                st.success(f"已匯入 {len(added)} 檔：{', '.join(added[:5])}{'...' if len(added)>5 else ''}")
                st.rerun()

        st.markdown("---")
        st.markdown(f"**🔀 移動「{cur_t}」到其他清單**")
        move_target = st.selectbox(
            "移到", [k for k in wls.keys() if k != act_l],
            key="move_target",
        )
        col_mv1, col_mv2 = st.columns(2)
        if col_mv1.button("✂️ 移動", use_container_width=True) and cur_t in lst:
            lst.remove(cur_t)
            if cur_t not in wls[move_target]:
                wls[move_target].append(cur_t)
            save_watchlists(wls)
            st.session_state['active_list'] = move_target
            st.session_state['user_opened_list'] = move_target
            st.rerun()
        if col_mv2.button("📋 複製", use_container_width=True):
            if cur_t not in wls[move_target]:
                wls[move_target].append(cur_t)
                save_watchlists(wls)
                st.success(f"已複製到 {move_target}")
                st.rerun()

        st.markdown("---")
        st.markdown("**🗑️ 刪除 / 重命名**")
        if st.button("❌ 從清單移除目前股票", use_container_width=True) and cur_t in lst:
            lst.remove(cur_t)
            save_watchlists(wls)
            st.session_state['current_ticker'] = lst[0] if lst else "^NDX"
            st.rerun()

        st.markdown("---")
        st.markdown("**🏷️ 重命名清單**")
        rename_src = st.selectbox("選擇清單", list(wls.keys()), key="rename_src")
        new_name = st.text_input("新名稱", placeholder="例如：🇺🇸 美持股",
                                  key="rename_name")
        if st.button("✏️ 確認重命名", use_container_width=True) and new_name.strip():
            new_name = new_name.strip()
            if new_name not in wls:
                # 保留順序的重命名
                new_wls = {}
                for k, v in wls.items():
                    new_wls[new_name if k == rename_src else k] = v
                st.session_state['watchlists'] = new_wls
                if act_l == rename_src:
                    st.session_state['active_list'] = new_name
                    st.session_state['user_opened_list'] = new_name
                save_watchlists(new_wls)
                st.rerun()
            else:
                st.warning("名稱已存在")

# --- 5. 主體資料載入 ---
main_title_name = get_stock_name(cur_t)
disp_main_title = f"{main_title_name} ({cur_t})" if main_title_name != cur_t else cur_t
st.title(f"📈 {disp_main_title} 實戰戰情室 V26.04")

api_p, api_i = ("5d", "15m") if "當沖" in time_opt else ("6mo", "1d") if "日" in time_opt else ("2y", "1wk")
df = yf.download(cur_t, period=api_p, interval=api_i, progress=False)
if df.empty:
    st.error("無數據")
    st.stop()
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)
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

latest = df.iloc[-1]
close_v = float(latest['Close'])
# [修復 #13] 前一日收盤為 0 時避免產生 inf
prev_close = float(df.iloc[-2]['Close'])
chg = ((close_v - prev_close) / prev_close * 100) if prev_close != 0 else 0.0
clr = "#4ade80" if chg >= 0 else "#ff6b6b"
sigs = analyze_strategic_signals(df)
trend_txt, trend_note, trend_col = analyze_market_trend(df)
rs_txt, rs_col = get_relative_strength(cur_t, df)
engine_label, engine_type = get_stock_engine_mode(cur_t, df)

# [v28] 提前生成 p_data + 跑蒙地卡羅，讓 AI 目標卡片能用 MC 結果
p_data = df.tail(120) if "日" in time_opt else df.tail(60)

# ── 蒙地卡羅計算（在繪圖前先算完，c4 卡片 + 圖上雲帶 + 推演摘要共用結果）──
_mc = None
_mc_status = {"ran": False, "reason": "", "n_points": 0}
try:
    import reversal_scanner as _rev_mc
    _MC_AVAILABLE = True
except ImportError as _imp_e:
    _MC_AVAILABLE = False
    _MC_IMPORT_ERR = str(_imp_e)

if _MC_AVAILABLE:
    # 計算漂移調整（context 失敗只代表漂移=0，不影響 MC 跑）
    _drift_adj = 0.0
    try:
        _market_state_mc = st.session_state.get('_market_state_v27', {})
        if len(p_data) >= 20:
            _ctx_mc = compute_signal_context(p_data, len(p_data) - 1, cur_t, _market_state_mc)
            if _ctx_mc:
                if _ctx_mc['market']['market_ok']:
                    _drift_adj += 0.0005
                if _ctx_mc['obv_pct_of_60d'] > 80:
                    _drift_adj += 0.0005
                elif _ctx_mc['obv_pct_of_60d'] < 30:
                    _drift_adj -= 0.0003
                if _ctx_mc['short_vol_ratio'] > 1.2:
                    _drift_adj += 0.0003
                elif _ctx_mc['short_vol_ratio'] < 0.8:
                    _drift_adj -= 0.0002
                if 0 < _ctx_mc['momentum_20d'] < 15:
                    _drift_adj += 0.0002
                elif _ctx_mc['momentum_20d'] > 25:
                    _drift_adj -= 0.0003
                if _ctx_mc['position']['dist_from_52w_high_pct'] < 5:
                    _drift_adj -= 0.0005
    except Exception as _ctx_e:
        _mc_status["reason"] = f"7 維 context 計算失敗（漂移=0）: {_ctx_e}"

    if len(p_data) < 60:
        _mc_status["reason"] = f"p_data 只有 {len(p_data)} 筆，MC 需 60+ 筆"
    else:
        try:
            _mc = _rev_mc.generate_monte_carlo_bands(
                p_data, days=30, n_simulations=1000, drift_adjust=_drift_adj
            )
            if _mc is None:
                _mc_status["reason"] = "MC 函式回傳 None（內部例外，已被 reversal_scanner 吞掉）"
            else:
                _mc_status["ran"] = True
                _mc_status["n_points"] = len(_mc.get('dates', []))
                st.session_state['_mc_result_v27'] = {
                    'p10_final': _mc['p10'][-1],
                    'p50_final': _mc['p50'][-1],
                    'p90_final': _mc['p90'][-1],
                    'sigma': _mc['sigma'],
                    'drift_adjust': _drift_adj,
                }
        except Exception as _mc_e:
            _mc_status["reason"] = f"MC 呼叫拋例外: {type(_mc_e).__name__}: {_mc_e}"
else:
    _mc_status["reason"] = f"reversal_scanner 模組未匯入"

st.session_state['_mc_status_v27'] = _mc_status

# AI 目標：優先使用 MC 結果（反映當下趨勢），無 MC 時退回布林+60日高點
_mc_for_target = st.session_state.get('_mc_result_v27') if _mc_status["ran"] else None
t_s, t_l, rating = predict_target_and_rating(df, _mc_for_target)
# 技術面「達標」閾值（與 AI 目標 t_s 分離，避免 MC p50 誤判過去 K 線）
_target_threshold = get_technical_target_threshold(df)

vp_60 = calculate_volume_profile(df.tail(60), bins=40)
vol_poc = vp_60.loc[vp_60['Volume'].idxmax(), 'Price'] if not vp_60.empty else close_v
iron_price, _box_start, _box_end, is_breaking = find_structural_box_bottom(df, close_v)

abs_high = p_data['High'].max()
local_maxes = p_data['High'][(p_data['High'] == p_data['High'].rolling(9, center=True).max())].dropna()
filtered_maxes = local_maxes[local_maxes < abs_high]

resist_line = None
support_line = None
if not filtered_maxes.empty:
    above_current = filtered_maxes[filtered_maxes > close_v]
    if not above_current.empty:
        resist_line = above_current.iloc[-1]
    below_current = filtered_maxes[filtered_maxes < close_v]
    if not below_current.empty:
        support_line = below_current.iloc[-1]

# [修復 #7] SMA_5 為 NaN 時 if cur_sma5 不攔截，改用 pd.isna 保護
cur_sma5 = latest.get('SMA_5', close_v)
bias_ma5 = ((close_v - cur_sma5) / cur_sma5) * 100 if (not pd.isna(cur_sma5) and cur_sma5 != 0) else 0.0
bias_limit = 5.0 if engine_type == "trend" else 10.0 if engine_type == "momentum" else 20.0

tac_title, tac_body, tac_color = get_tactical_advice(df, close_v, t_s, iron_price, support_line, resist_line, bias_ma5, bias_limit, engine_type)
st.markdown(
    f'<div class="tactical-box" style="border-left-color: {tac_color};">'
    f'<div class="tactical-title">🚩 戰術建議： <span style="color:{tac_color}; margin-left: 8px;">{tac_title}</span></div>'
    f'<div class="tactical-body">💡 <b>行動指南：</b> {tac_body}</div></div>',
    unsafe_allow_html=True
)

# ==========================================
# [v25 新增] 大盤崩跌警示欄位（只對指數/ETF顯示）
# ==========================================
if is_index_or_etf(cur_t):
    # 抓取 NAAIM / AAII 作為輔助訊號
    _naaim_v, _naaim_max4w, _aaii_b, _aaii_br = None, None, None, None
    try:
        _naaim_df, _ns = get_naaim_data()
        if not _naaim_df.empty and _ns != "demo":
            _naaim_v = float(_naaim_df["Exposure"].iloc[-1])
            _naaim_max4w = float(_naaim_df["Exposure"].iloc[-5:-1].max()) if len(_naaim_df) >= 5 else None
    except Exception:
        pass
    try:
        (_b, _n, _br), _as = get_aaii_data()
        if _as != "demo":
            _aaii_b, _aaii_br = float(_b), float(_br)
    except Exception:
        pass

    crash = detect_market_crash_signals(
        df,
        naaim_value=_naaim_v, naaim_prev_max=_naaim_max4w,
        aaii_bull=_aaii_b, aaii_bear=_aaii_br
    )

    # 只在 score >= 3 顯示完整警示框；否則顯示精簡綠燈
    if crash["score"] >= 3:
        sig_html = "".join(
            f'<li style="margin-bottom:4px;">{s}</li>' for s in crash["signals"]
        )
        st.markdown(
            f'<div style="background:#1a1a1c; padding:18px 24px; border-radius:8px; '
            f'margin-bottom:20px; border-left:8px solid {crash["level_color"]}; '
            f'box-shadow:0 4px 6px rgba(0,0,0,0.3);">'
            f'<div style="font-size:20px; font-weight:bold; color:#fff; margin-bottom:10px;">'
            f'🚨 大盤崩跌警示 '
            f'<span style="color:{crash["level_color"]}; margin-left:10px;">{crash["level_label"]}</span>'
            f'<span style="font-size:13px; color:#aaa; font-weight:normal; margin-left:10px;">'
            f'(風險分數：{crash["score"]} / 17)</span></div>'
            f'<div style="background:#262730; padding:14px; border-radius:6px; '
            f'border:1px solid {crash["level_color"]}; color:#e5e7eb; font-size:14px;">'
            f'<b style="color:{crash["level_color"]};">📋 行動指南：</b> {crash["summary"]}'
            f'<ul style="margin-top:10px; margin-bottom:0; padding-left:20px; color:#ddd; font-size:13px;">'
            f'{sig_html}</ul></div></div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'<div style="background:#1a2a1a; padding:10px 18px; border-radius:6px; '
            f'margin-bottom:15px; border-left:5px solid #22c55e; font-size:13px; color:#aaffaa;">'
            f'🛡️ <b>大盤崩跌警示：</b>{crash["level_label"]}（風險分數：{crash["score"]} / 17）。'
            f'{crash["summary"]}</div>',
            unsafe_allow_html=True
        )

# ==========================================
# 一體化四大戰情方塊佈局
# ==========================================
ern_date, ern_res = get_earnings_status(cur_t)

# [v28] 砍掉規則式劇本，改用 MC 漂移方向作為「未來推演」標籤
_drift_for_label = st.session_state.get('_mc_result_v27', {}).get('drift_adjust', 0.0)
if not _mc_status.get("ran"):
    sc_name = "⚠️ MC 未計算<br>無法推演"
    sc_color_class = "sig-orange"
elif _drift_for_label >= 0.0008:
    sc_name = "📈 偏多漂移<br>傾向上行"
    sc_color_class = "sig-green"
elif _drift_for_label >= 0.0003:
    sc_name = "↗ 輕微偏多<br>溫和上行"
    sc_color_class = "sig-green"
elif _drift_for_label > -0.0003:
    sc_name = "↔ 中性漂移<br>區間震盪"
    sc_color_class = "sig-orange"
elif _drift_for_label > -0.0008:
    sc_name = "↘ 輕微偏空<br>溫和下行"
    sc_color_class = "sig-red"
else:
    sc_name = "📉 偏空漂移<br>傾向下行"
    sc_color_class = "sig-red"

c1, c2, c3, c4 = st.columns([1.3, 1, 1, 1])

with c1:
    c1_html = (
        f'<div class="ai-box" style="border: 1px solid #4a9eff; background-color: #16202b; padding: 15px; display: flex; flex-direction: column; justify-content: center;">'
        f'<h2 style="margin:0; font-size: 38px; font-weight: 900; line-height: 1.1;">${close_v:.2f}</h2>'
        f'<div style="font-size: 18px; font-weight: bold; color: {clr}; margin-bottom: 8px;">{chg:+.2f}% <span style="color: #888; font-size: 13px; font-weight: normal;">(量: {format_volume(latest["Volume"])})</span></div>'
        f'<div><span class="engine-tag" style="margin:0; padding: 3px 8px; font-size: 12px;">⚙️ {engine_label}</span></div>'
        f'<div style="margin-top: 8px; font-size: 12px; line-height: 1.4; color: #ccc;">{ern_date}<br>{ern_res}</div>'
        f'</div>'
    )
    st.markdown(c1_html, unsafe_allow_html=True)

with c2:
    c2_html = (
        '<div class="ai-box" style="display:flex; flex-direction:column; justify-content:center;">'
        '<h5 style="color:white; margin:0; margin-bottom:12px;">📡 綜合戰略</h5>'
        f'<div><span class="{sigs["Summary_Color"]}" style="font-size:14px; font-weight:bold; padding:4px 10px; display:inline-block; line-height:1.4; border-radius:6px; margin-bottom:8px;">{sigs["Summary"]}</span></div>'
        '<div style="margin-top: 4px; padding-top: 8px; border-top: 1px dashed #555;">'
        '<span style="color:#aaa; font-size:12px;">🔮 未來推演:</span><br>'
        f'<span class="{sc_color_class}" style="font-size:14px; font-weight:bold; line-height:1.3; border:0; background:transparent;">{sc_name}</span>'
        '</div></div>'
    )
    st.markdown(c2_html, unsafe_allow_html=True)

with c3:
    bias_color = "sig-red" if bias_ma5 >= bias_limit * 0.8 else "sig-green" if bias_ma5 <= -bias_limit * 0.8 else "sig-gray"
    c3_html = (
        '<div class="ai-box" style="display: flex; flex-direction: column; justify-content: center;">'
        '<h5 style="color:white; margin:0; margin-bottom:10px;">⚖️ 技術與格局</h5>'
        f'<div style="font-size:14px; margin-bottom: 8px;">🏢 個股: <span class="{trend_col}" style="display: inline-block;">{trend_txt}</span></div>'
        f'<div style="display:flex; justify-content:center; gap:6px; flex-wrap:wrap; margin-top:2px;">'
        f'<span class="{bias_color}" style="font-size:11px; padding:2px 6px; border-radius:4px;">乖離 {bias_ma5:+.1f}%</span>'
        f'<span class="{sigs["RSI_Color"]}" style="font-size:11px; padding:2px 6px; border-radius:4px;">RSI {sigs["RSI_Text"]}</span>'
        f'<span class="{sigs["MACD_Color"]}" style="font-size:11px; padding:2px 6px; border-radius:4px;">{sigs["MACD_Text"]}</span>'
        f'<span class="{sigs["Vol_Color"]}" style="font-size:11px; padding:2px 6px; border-radius:4px;">{sigs["Vol_Text"]}</span>'
        f'</div>'
        '</div>'
    )
    st.markdown(c3_html, unsafe_allow_html=True)

with c4:
    # MC 預期方向（漲跌幅 vs 現價）+ 來源標籤
    _t_s_pct = (t_s / close_v - 1) * 100 if close_v > 0 else 0
    _t_l_pct = (t_l / close_v - 1) * 100 if close_v > 0 else 0
    _t_s_clr = "#22c55e" if _t_s_pct >= 0 else "#ef4444"
    _t_l_clr = "#22c55e" if _t_l_pct >= 0 else "#ef4444"
    _mc_ran = st.session_state.get('_mc_status_v27', {}).get('ran', False)
    _source_label = ("🔮 MC p50/p90" if _mc_ran else "📐 布林/60日高")
    
    c4_html = (
        '<div class="ai-box" style="border: 1px solid #00d4ff; display: flex; flex-direction: column; justify-content: center;">'
        '<h5 style="color:white; margin:0; margin-bottom:8px;">🎯 AI 目標 & 強弱</h5>'
        f'<div style="font-size:13px; color: #ddd; margin-bottom: 4px;">'
        f'短: <b>${t_s:.2f}</b> <span style="color:{_t_s_clr};font-size:11px;">({_t_s_pct:+.1f}%)</span><br>'
        f'長: <b>${t_l:.2f}</b> <span style="color:{_t_l_clr};font-size:11px;">({_t_l_pct:+.1f}%)</span>'
        f'</div>'
        f'<div style="font-size:10px; color:#888; margin-bottom: 6px;">{_source_label}（30 日推演）</div>'
        f'<div><span class="{rs_col}" style="display: inline-block;">{rs_txt}</span></div>'
        '</div>'
    )
    st.markdown(c4_html, unsafe_allow_html=True)

st.markdown("<div style='margin-bottom: 15px;'></div>", unsafe_allow_html=True)

# ==========================================
# 繪圖區
# ==========================================
fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_width=[0.2, 0.2, 0.6])

if 'Bollinger_Upper' in p_data.columns and 'Bollinger_Lower' in p_data.columns:
    fig.add_trace(go.Scatter(
        x=p_data.index, y=p_data['Bollinger_Upper'], mode='lines',
        line=dict(color='rgba(253, 224, 71, 0.5)', width=1, dash='dot'),
        name='布林上軌 (壓力)', hoverinfo='skip'
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=p_data.index, y=p_data['Bollinger_Lower'], mode='lines',
        line=dict(color='rgba(253, 224, 71, 0.5)', width=1, dash='dot'),
        fill='tonexty', fillcolor='rgba(253, 224, 71, 0.05)',
        name='布林下軌 (支撐)', hoverinfo='skip'
    ), row=1, col=1)

fig.add_trace(go.Candlestick(
    x=p_data.index, open=p_data['Open'], high=p_data['High'],
    low=p_data['Low'], close=p_data['Close'], name="K線"
), row=1, col=1)
fig.add_trace(go.Scatter(
    x=p_data.index, y=p_data['ATR_Trailing_Stop'], mode='lines',
    line=dict(color='#FF5F1F', width=1.5, dash='dot'), name='ATR 停損'
), row=1, col=1)

for col_name, color, width, label in [
    ('SMA_5', '#f97316', 1.5, 'MA5 (周線)'),
    ('SMA_10', '#38bdf8', 1.5, 'MA10 (雙周)'),
    ('SMA_20', '#d946ef', 2, 'MA20 (月線)'),
    ('SMA_60', '#2563eb', 2, 'MA60 (季線)'),
]:
    if col_name in p_data.columns:
        fig.add_trace(go.Scatter(
            x=p_data.index, y=p_data[col_name], mode='lines',
            line=dict(color=color, width=width), name=label
        ), row=1, col=1)

if 'SMA_20' in p_data.columns and 'SMA_60' in p_data.columns:
    ma_gold = (p_data['SMA_20'] > p_data['SMA_60']) & (p_data['SMA_20'].shift(1) <= p_data['SMA_60'].shift(1))
    ma_dead = (p_data['SMA_20'] < p_data['SMA_60']) & (p_data['SMA_20'].shift(1) >= p_data['SMA_60'].shift(1))
    launch_pts  = detect_launch_points(p_data)   # ★ 起漲點遮罩

    # [v27 方案 A] 預先載入大盤狀態（給主圖起漲點過濾用）
    if '_market_state_v27' not in st.session_state:
        st.session_state['_market_state_v27'] = get_market_filter_state(get_cache_anchor())
    _market_state_main = st.session_state.get('_market_state_v27', {})

    for date in p_data[ma_gold].index:
        y_val = p_data.loc[date, 'SMA_20']
        if not pd.isna(y_val):
            # [v27 方案 A] 起漲點需通過大盤過濾 + MA60 上升
            idx_pos_ma = p_data.index.get_loc(date)
            sma60_curr = p_data.iloc[idx_pos_ma].get('SMA_60', np.nan)
            sma60_prior5 = p_data.iloc[max(0, idx_pos_ma - 5)].get('SMA_60', np.nan)
            ma60_up = (not pd.isna(sma60_curr) and not pd.isna(sma60_prior5) and sma60_curr > sma60_prior5)
            mkt_ok = market_ok_at_date(_market_state_main, date)
            plan_a_pass_main = ma60_up and mkt_ok

            if launch_pts.get(date, False) and plan_a_pass_main:
                # ★ 黃金交叉 + 起漲條件 + Plan A → 強化標注
                fig.add_annotation(
                    x=date, y=p_data.loc[date, 'Low'],
                    text="🚀 起漲點", showarrow=True, arrowhead=2,
                    arrowcolor="#00ff88", ax=0, ay=55, row=1, col=1,
                    bgcolor="rgba(0, 200, 100, 0.85)",
                    font=dict(color="black", size=11, weight="bold")
                )
                fig.add_annotation(
                    x=date, y=y_val,
                    text="🌕 黃金交叉<br>⭐起漲確認", showarrow=True, arrowhead=1,
                    ax=0, ay=35, row=1, col=1,
                    bgcolor="rgba(234, 179, 8, 0.9)",
                    font=dict(color="black", size=10, weight="bold")
                )
                # 收集進 v27 訊號清單供 AI 推演
                if '_launch_signals_v27' not in st.session_state:
                    st.session_state['_launch_signals_v27'] = []
                st.session_state['_launch_signals_v27'].append({
                    "date": date,
                    "idx_pos": idx_pos_ma,
                    "close": float(p_data.iloc[idx_pos_ma]['Close']),
                    "trigger": "MA金叉+RSI超賣",
                    "ma60_uptrend": ma60_up,
                    "market_ok": mkt_ok,
                })
            elif launch_pts.get(date, False) and not plan_a_pass_main:
                # 起漲條件成立但 Plan A 未過 → 標弱化版「黃金交叉」（不發起漲點）
                reason = []
                if not ma60_up: reason.append("MA60↓")
                if not mkt_ok: reason.append("大盤偏空")
                reason_str = " · ".join(reason)
                fig.add_annotation(
                    x=date, y=y_val,
                    text=f"🌕 黃金交叉<br><span style='font-size:9px'>⚠️{reason_str}</span>",
                    showarrow=True, arrowhead=1, ax=0, ay=35, row=1, col=1,
                    bgcolor="rgba(234, 179, 8, 0.6)",
                    font=dict(color="black", size=10)
                )
            else:
                fig.add_annotation(x=date, y=y_val, text="🌕 黃金交叉", showarrow=True, arrowhead=1, ax=0, ay=35, row=1, col=1, bgcolor="rgba(234, 179, 8, 0.8)", font=dict(color="black", size=10, weight="bold"))
    for date in p_data[ma_dead].index:
        y_val = p_data.loc[date, 'SMA_20']
        if not pd.isna(y_val):
            fig.add_annotation(x=date, y=y_val, text="☠️ 死亡交叉", showarrow=True, arrowhead=1, ax=0, ay=-35, row=1, col=1, bgcolor="rgba(239, 68, 68, 0.8)", font=dict(color="white", size=10, weight="bold"))

zone_top = vol_poc * 1.025
zone_bottom = vol_poc * 0.975
fig.add_hrect(y0=zone_bottom, y1=zone_top, line_width=0, fillcolor="rgba(200, 200, 200, 0.15)", layer="below", row=1, col=1)

last_d = p_data.index[-1]
for r in range(1, 4):
    fig.add_vline(x=last_d, line_dash="dash", line_color="#666", opacity=0.7, layer="below", row=r, col=1)

# [修復 #6] 預先計算所有 smart money status，避免 O(n²) 重複計算
precomputed_status = []
for i in range(len(p_data)):
    if i < 9:
        precomputed_status.append(None)
    else:
        precomputed_status.append(detect_smart_money_status(p_data.iloc[max(0, i - 9):i + 1]))

# [v27] 重置進場訊號清單（每次重跑都清空）
st.session_state['_entry_signals_v27'] = []

for i in range(5, len(p_data)):
    curr = p_data.iloc[i]
    prior = p_data.iloc[i - 1]

    td_b = curr['TD_Buy_Seq']
    td_s = curr['TD_Sell_Seq']
    if 0 < td_b <= 9:
        fig.add_annotation(x=p_data.index[i], y=curr['Low'], text=str(int(td_b)), showarrow=False, yshift=-12, font=dict(color='#ff6b6b', size=9 if td_b < 9 else 14, weight="normal" if td_b < 9 else "bold"), row=1, col=1)
    if 0 < td_s <= 9:
        fig.add_annotation(x=p_data.index[i], y=curr['High'], text=str(int(td_s)), showarrow=False, yshift=12, font=dict(color='#4a9eff', size=9 if td_s < 9 else 14, weight="normal" if td_s < 9 else "bold"), row=1, col=1)

    if (prior['Close'] < prior['Open'] and curr['Close'] > curr['Open'] and
            curr['Open'] <= prior['Close'] and curr['Close'] >= prior['Open']):
        fig.add_annotation(x=p_data.index[i], y=curr['Low'], text="🕯️吞噬", showarrow=True, arrowhead=1, ax=0, ay=30, row=1, col=1, font=dict(color="orange", size=9))

    status = precomputed_status[i]
    if status:
        if "吸籌" in status:
            fig.add_annotation(x=p_data.index[i], y=curr['Low'], text=f"🐳吸<br>${curr['Low']:.1f}", showarrow=True, arrowhead=1, ax=0, ay=35, row=1, col=1, bgcolor="rgba(111, 66, 193, 0.8)", font=dict(color="white", size=9))
            # [v27] 收集吸籌訊號（給 AI 推演用）
            if '_entry_signals_v27' not in st.session_state:
                st.session_state['_entry_signals_v27'] = []
            st.session_state['_entry_signals_v27'].append({
                "type": "🤫 吸籌", "date": p_data.index[i], "idx_pos": i,
                "close": float(curr['Close']), "stage": 1,
            })
        elif "破底翻" in status or "抄底" in status:
            fig.add_annotation(x=p_data.index[i], y=curr['Low'], text=f"{status}<br>${curr['Low']:.1f}", showarrow=True, arrowhead=1, ax=0, ay=45, row=1, col=1, bgcolor="rgba(147, 51, 234, 0.8)", font=dict(color="white", size=10, weight="bold"))
            # [v27] 收集乖離抄底/破底翻訊號
            if '_entry_signals_v27' not in st.session_state:
                st.session_state['_entry_signals_v27'] = []
            sig_type = "💎 破底翻買點" if "破底翻" in status else "💎 乖離抄底"
            st.session_state['_entry_signals_v27'].append({
                "type": sig_type, "date": p_data.index[i], "idx_pos": i,
                "close": float(curr['Close']), "stage": 2,
            })
        elif "調節" in status:
            fig.add_annotation(x=p_data.index[i], y=curr['High'], text=f"🔴調節<br>${curr['High']:.1f}", showarrow=True, arrowhead=1, ax=0, ay=-40, row=1, col=1, bgcolor="rgba(185, 28, 28, 0.8)", font=dict(color="white", size=10, weight="bold"))

    macd_buy = (curr['MACD'] > curr['Signal_Line']) and (prior['MACD'] <= prior['Signal_Line'])
    macd_sell = (curr['MACD'] < curr['Signal_Line']) and (prior['MACD'] >= prior['Signal_Line'])
    if macd_buy and ((engine_type == "trend" and curr['Close'] < curr.get('SMA_60', 0)) or
                     (engine_type == "momentum" and curr['Close'] < curr.get('SMA_20', 0))):
        macd_buy = False

    if macd_buy:
        fig.add_annotation(x=p_data.index[i], y=curr['Low'], text=f"BUY<br>${curr['Close']:.1f}", showarrow=True, arrowhead=1, ax=0, ay=25, row=1, col=1, bgcolor="rgba(40, 167, 69, 0.8)", font=dict(color="white", size=9))
        # [v27] 收集 BUY 訊號（stage 3）
        if '_entry_signals_v27' not in st.session_state:
            st.session_state['_entry_signals_v27'] = []
        st.session_state['_entry_signals_v27'].append({
            "type": "🟢 MACD BUY", "date": p_data.index[i], "idx_pos": i,
            "close": float(curr['Close']), "stage": 3,
        })
    if macd_sell:
        fig.add_annotation(x=p_data.index[i], y=curr['High'], text=f"SELL<br>${curr['Close']:.1f}", showarrow=True, arrowhead=1, ax=0, ay=-25, row=1, col=1, bgcolor="rgba(220, 53, 69, 0.8)", font=dict(color="white", size=9))
    if (curr['High'] >= _target_threshold or curr['RSI'] > 75) and not (prior['High'] >= _target_threshold or prior['RSI'] > 75):
        fig.add_annotation(
            x=p_data.index[i], y=curr['High'],
            text=f"💰達標<br>${curr['Close']:.1f}" if curr['High'] >= _target_threshold else f"🔥過熱<br>${curr['Close']:.1f}",
            showarrow=True, arrowhead=1, ax=0, ay=-45, row=1, col=1,
            bgcolor="rgba(255, 193, 7, 0.8)" if curr['High'] >= _target_threshold else "rgba(255, 69, 0, 0.8)",
            font=dict(color="black" if curr['High'] >= _target_threshold else "white", size=9)
        )

    is_near_support = False
    if support_line and abs(curr['Low'] - support_line) / support_line < 0.015:
        is_near_support = True
    if iron_price > 0 and abs(curr['Low'] - iron_price) / iron_price < 0.015:
        is_near_support = True

    vol_surge = False
    if not pd.isna(p_data['Vol_SMA5'].iloc[i]):
        vol_surge = curr['Volume'] > p_data['Vol_SMA5'].iloc[i] * 1.2

    strong_reversal = False
    if curr['Close'] > curr['Open']:
        strong_reversal = True
    elif (min(curr['Close'], curr['Open']) - curr['Low']) > abs(curr['Close'] - curr['Open']) * 2.5:
        strong_reversal = True

    if is_near_support and vol_surge and strong_reversal:
        fig.add_annotation(x=p_data.index[i], y=curr['Low'], text="支撐", showarrow=True, arrowhead=1, ax=0, ay=30, row=1, col=1, font=dict(color="#00ffff", size=9, weight="bold"))

# ──────────────────────────────────────────────────────
# [v28] 蒙地卡羅雲帶繪圖（計算已在卡片區之前完成，這裡只負責畫到 fig）
# ──────────────────────────────────────────────────────
if _mc is not None:
    # 80% 信心帶（p10-p90）
    fig.add_trace(go.Scatter(
        x=list(_mc['dates']) + list(reversed(_mc['dates'])),
        y=list(_mc['p90']) + list(reversed(_mc['p10'])),
        fill='toself',
        fillcolor='rgba(234, 179, 8, 0.10)',
        line=dict(width=0),
        hoverinfo='skip',
        showlegend=True,
        name='🔮 MC 80% 信心區'
    ), row=1, col=1)
    # 50% 信心帶（p25-p75）
    fig.add_trace(go.Scatter(
        x=list(_mc['dates']) + list(reversed(_mc['dates'])),
        y=list(_mc['p75']) + list(reversed(_mc['p25'])),
        fill='toself',
        fillcolor='rgba(234, 179, 8, 0.20)',
        line=dict(width=0),
        hoverinfo='skip',
        showlegend=True,
        name='🔮 MC 50% 信心區'
    ), row=1, col=1)
    # 中位數路徑
    fig.add_trace(go.Scatter(
        x=_mc['dates'],
        y=_mc['p50'],
        mode='lines',
        line=dict(color='#eab308', width=2, dash='solid'),
        name='🔮 MC 中位數路徑',
        hovertemplate='中位數: $%{y:.2f}<extra></extra>'
    ), row=1, col=1)

# [v28] 規則式劇本（黃色虛線+菱形）已移除，預測完全交給蒙地卡羅雲帶

# ──────────────────────────────────────────────────────
# [v29] 事件節點垂直虛線（財報日 + FOMC/CPI/PPI/非農）
# ──────────────────────────────────────────────────────
_ev_status = {"drawn_count": 0, "reason": ""}
try:
    import event_nodes as _ev
    _EV_AVAILABLE = True
except ImportError as _ev_imp_e:
    _EV_AVAILABLE = False
    _ev_status["reason"] = f"event_nodes 模組未匯入：{_ev_imp_e}"

if not _EV_AVAILABLE:
    pass  # reason 已記錄
elif not _ev.is_us_stock(cur_t):
    _ev_status["reason"] = f"當前股票 {cur_t} 非美股，事件節點僅標記美股事件"
else:
    try:
        # 範圍：圖表左邊（p_data 起點）到 MC 雲帶終點（未來 30 交易日）
        _ev_start = p_data.index[0] if len(p_data) > 0 else pd.Timestamp.now()
        _ev_end = _mc['dates'][-1] if (_mc is not None) else (pd.Timestamp.now() + pd.Timedelta(days=45))

        # 從 ern_date 字串解析財報日（格式：「📅 財報: 2026-05-28」或「📅 財報: N/A」）
        _ern_date_only = None
        if ern_date and "N/A" not in ern_date:
            try:
                _ern_date_only = ern_date.split(":")[-1].strip()
                pd.to_datetime(_ern_date_only)  # 驗證可解析
            except Exception:
                _ern_date_only = None

        events_for_chart = _ev.get_all_events_for_chart(
            ticker=cur_t,
            earnings_date=_ern_date_only,
            start_date=_ev_start,
            end_date=_ev_end,
        )

        if not events_for_chart:
            _ev_status["reason"] = (
                f"範圍內無事件（{pd.to_datetime(_ev_start).strftime('%Y-%m-%d')} ~ "
                f"{pd.to_datetime(_ev_end).strftime('%Y-%m-%d')}）"
            )
        else:
            # 把每個事件畫成垂直虛線 + 上方標籤
            # 避免相同日期重疊：用 dict 收集再合併標籤
            events_by_date = {}
            for ev in events_for_chart:
                key = ev["date"].strftime("%Y-%m-%d")
                if key not in events_by_date:
                    events_by_date[key] = []
                events_by_date[key].append(ev)

            # 標籤要畫在 K 線範圍內（避免超出 subplot 被切）
            # 用主圖 y 軸的「波段最高」上方一點，這是已知會落在可見範圍內的位置
            _ev_label_y = abs_high * 1.015  # 波段最高 + 1.5%

            for date_key, evs in events_by_date.items():
                # 同一天多事件 → 取 priority 最高的當主色，標籤合併
                primary = max(evs, key=lambda x: x["priority"])
                if len(evs) == 1:
                    label = primary["label"]
                else:
                    other_labels = " + ".join(e["label"].split()[0] for e in evs if e is not primary)
                    label = f'{primary["label"]} + {other_labels}'

                # 用 add_vline（與 last_d 灰線同模式，已驗證可運作）
                fig.add_vline(
                    x=primary["date"],
                    line_dash=primary["dash"],
                    line_color=primary["color"],
                    opacity=0.7,
                    layer="below",
                    row=1, col=1,
                )
                # 用絕對 y 值畫標籤（避免 y domain 被 subplot 切掉）
                fig.add_annotation(
                    x=primary["date"],
                    y=_ev_label_y,
                    text=label,
                    showarrow=False,
                    font=dict(color=primary["color"], size=9, weight="bold"),
                    bgcolor="rgba(0,0,0,0.6)",
                    bordercolor=primary["color"],
                    borderwidth=1,
                    borderpad=2,
                    xanchor="center",
                    yanchor="bottom",
                    row=1, col=1,
                )
                _ev_status["drawn_count"] += 1
    except Exception as _ev_e:
        _ev_status["reason"] = f"事件節點繪製失敗：{type(_ev_e).__name__}: {_ev_e}"

# 把狀態存入 session 給圖下方面板顯示
st.session_state['_ev_status_v29'] = _ev_status
# [V26.04] 也把事件列表存起來，讓燈號上方面板可以讀取
if 'events_for_chart' not in dir():
    events_for_chart = []
st.session_state['_events_for_chart_v29'] = events_for_chart if events_for_chart else []


fig.add_hline(y=abs_high, line_dash="dot", line_color="#ef4444", annotation_text=f"🔴 波段最高<br>${abs_high:.2f}", annotation_font_color="#ef4444", annotation_position="top right", annotation_align="right", opacity=1.0, layer="above", row=1, col=1)
if resist_line:
    fig.add_hline(y=resist_line, line_dash="dot", line_color="#f97316", annotation_text=f"🟠 前高壓力<br>${resist_line:.2f}", annotation_font_color="#f97316", annotation_position="top right", annotation_align="right", opacity=1.0, layer="above", row=1, col=1)
if support_line:
    fig.add_hline(y=support_line, line_dash="dot", line_color="#3b82f6", annotation_text=f"🔵 前高支撐<br>${support_line:.2f}", annotation_font_color="#3b82f6", annotation_position="bottom right", annotation_align="right", opacity=1.0, layer="above", row=1, col=1)
if not is_breaking and iron_price > 0:
    fig.add_hline(y=iron_price, line_dash="dash", line_color="#20c997", annotation_text=f"🧱 鐵板 ${iron_price:.2f}", annotation_font_color="#20c997", annotation_position="bottom right", opacity=1.0, layer="above", row=1, col=1)

fig.add_annotation(
    x=0.5, y=0.98, xref="paper", yref="paper",
    text=f"🛡️ 籌碼攻防帶 (突破: ${zone_top:.2f} | 跌破: ${zone_bottom:.2f})",
    showarrow=False, font=dict(color="#d8b4fe", size=12, weight="bold"),
    bgcolor="rgba(10, 10, 10, 0.9)", borderpad=4, xanchor="center", yanchor="top"
)
fig.add_trace(go.Scatter(
    x=[last_d], y=[close_v], mode='markers',
    marker=dict(size=6, color='rgba(100, 149, 237, 0.6)', line=dict(color='rgba(255, 255, 255, 0.5)', width=1)),
    name="今日收盤"
), row=1, col=1)

macd_gold = (p_data['MACD'] > p_data['Signal_Line']) & (p_data['MACD'].shift(1) <= p_data['Signal_Line'].shift(1))
macd_dead = (p_data['MACD'] < p_data['Signal_Line']) & (p_data['MACD'].shift(1) >= p_data['Signal_Line'].shift(1))

# --- [v20] Bar + Line 先畫，確保 markers/星星在最上層 ---
colors = ['green' if v >= 0 else 'red' for v in p_data['MACD_Hist']]
fig.add_trace(go.Bar(x=p_data.index, y=p_data['MACD_Hist'], marker_color=colors), row=2, col=1)
fig.add_trace(go.Scatter(x=p_data.index, y=p_data['MACD'], line=dict(color='white', width=1)), row=2, col=1)
fig.add_trace(go.Scatter(x=p_data.index, y=p_data['Signal_Line'], line=dict(color='yellow', width=1)), row=2, col=1)

# --- [v20] 起漲確認邏輯 ---
# =============================================================
# [v26 規則總覽] 起漲點 ⭐ 給星規則（依股票分類自動適用差異化條件）
# -------------------------------------------------------------
# 四類股票特性與規則對應：
#   (a) 權值股／指數類（NVDA, TSM, AAPL, AMZN, GOOG, QQQ, META, MSFT）
#       特徵：趨勢綿長、波動溫和。MACD 多在零軸附近寬區震盪。
#       主要起漲：正值區域金叉、死叉超賣後深度翻身。
#       誤判：META/MSFT 紅圈是死貓彈，需 rsi_already_recovering 過濾。
#
#   (b) 大型科技／週期股（AMD, TSLA, MU, INTC）
#       特徵：週期性強，下殺幅度深。
#       主要起漲：死叉 + 負值金叉 + RSI 曾深度超賣。
#       注意：不能全擋死叉，會把真正底部反轉一起濾掉。
#
#   (c) 成長／中型股（PLTR, CRCL, SOFI, ASX）
#       特徵：高波動，RSI 容易短暫跌破 25 但 MACD 動能極弱。
#       主要過濾：macd_significant 動能門檻擋掉微小金叉。
#
#   (d) 妖股／高波動（RXRX, SNDK, COIN, BE）
#       特徵：MACD 在零軸附近快速震盪、金叉訊號密集。
#       主要過濾：差異化叢集窗口（正值區3根、其他7根）。
#       BE/COIN/SNDK 的綠圈：正值區域金叉直接給星。
#
# 條件與股票類別對照表：
#   (1) 正值區域金叉 (DIF>0, DEA>0)        → 4 類都直接給星（趨勢健康）
#   (2) 零軸突破 (非死叉, 從負穿正)         → 4 類都直接給星（動能確立）
#   (3) 死叉 + 負值金叉                     → 需 RSI 曾深度超賣 + 已在回升
#   (4) MACD 動能極弱 (|DIF|<股價×0.1%)    → 死叉期間封鎖（針對成長/妖股）
#   (5) 防叢集（前 N 根已給星）             → 正值區 3 根、其他 7 根
#
# [v26 新增] 超賣門檻差異化：
#   - trend / momentum 股（NVDA, GOOG, TSLA 等）→ RSI < 35
#     理由：大型權值股波動溫和，死叉期間 RSI 鮮少跌破 25；
#           若維持 25 門檻，底部反轉訊號會被大量過濾，
#           導致 DMA 上穿有形態但 MACD 面板無星號的視覺落差。
#   - reversal / 妖股類 → 維持 RSI < 25（避免假訊號過多）
#
# 衝突解決：
#   - 衝突 1：正值區多次金叉 vs 防叢集 → 用差異化窗口
#   - 衝突 2：MSFT/META 死貓彈 vs 深度超賣例外 → 加 rsi_already_recovering
#   - 衝突 3：PLTR 微小金叉 vs RSI 偶發超賣 → 加 macd_significant
#   - 衝突 4 [v26]：trend/momentum 門檻放寬 vs 死貓彈風險
#     → 仍需 rsi_already_recovering（RSI 比 5 根前高）雙重確認
# =============================================================
# 先取出主圖已判定的起漲點日期集合（MA 黃金交叉版 + Plan A）
# [v27 BUG修復] 主圖起漲點也需要過 Plan A，否則 RXRX 等趨勢下行股會誤標
main_launch_dates = set()
if 'launch_pts' in dir():
    # 預載大盤狀態
    if '_market_state_v27' not in st.session_state:
        st.session_state['_market_state_v27'] = get_market_filter_state(get_cache_anchor())
    _ms_main = st.session_state.get('_market_state_v27', {})
    for _d in p_data[launch_pts].index:
        _idx_pos = p_data.index.get_loc(_d)
        _sma60_curr = p_data.iloc[_idx_pos].get('SMA_60', np.nan)
        _sma60_p5 = p_data.iloc[max(0, _idx_pos - 5)].get('SMA_60', np.nan)
        _ma60_up = (not pd.isna(_sma60_curr) and not pd.isna(_sma60_p5)
                     and _sma60_curr > _sma60_p5)
        _mkt_ok = market_ok_at_date(_ms_main, _d)
        if _ma60_up and _mkt_ok:
            main_launch_dates.add(_d)

# [v27] 載入大盤狀態 + 重置起漲訊號清單（給 AI 推演用）
if '_market_state_v27' not in st.session_state:
    st.session_state['_market_state_v27'] = get_market_filter_state(get_cache_anchor())
st.session_state['_launch_signals_v27'] = []  # 每次重跑都清空

normal_macd_x, normal_macd_y = [], []
star_macd_x, star_macd_y = [], []
already_starred = set()  # 避免同一根 K 棒貼兩次星星

for date in p_data[macd_gold].index:
    idx_pos = p_data.index.get_loc(date)
    curr = p_data.iloc[idx_pos]
    prior = p_data.iloc[idx_pos - 1] if idx_pos > 0 else curr

    # [v20] RSI 回暖：改比 2 根前，避免單根偶發波動卡死
    prior2 = p_data.iloc[max(0, idx_pos - 2)]
    rsi_recovering = curr['RSI'] > prior2['RSI']

    ma60 = curr.get('SMA_60', 0)
    near_ma60 = (not pd.isna(ma60) and ma60 > 0 and abs(curr['Close'] - ma60) / ma60 <= 0.05)
    deep_reversal = curr['MACD'] < 0
    zero_cross = (curr['MACD'] >= 0 and prior['MACD'] < 0)

    # [v22] MA 趨勢方向
    ma20_val = curr.get('SMA_20', float('nan'))
    death_cross_active = (
        not pd.isna(ma20_val) and not pd.isna(ma60) and ma60 > 0
        and ma20_val < ma60
    )
    golden_cross_active = (
        not pd.isna(ma20_val) and not pd.isna(ma60) and ma60 > 0
        and ma20_val > ma60
    )

    # [v26] 共用計算
    # 超賣門檻差異化：trend/momentum 股（NVDA/GOOG/TSLA等）波動溫和，
    # RSI 不易跌破 25，故放寬至 35；妖股/高波動維持嚴格 25。
    rsi_w20 = p_data.iloc[max(0, idx_pos - 20): idx_pos + 1]['RSI']
    _oversold_threshold = 35 if engine_type in ("trend", "momentum") else 25
    is_deeply_oversold = rsi_w20.min() < _oversold_threshold

    # [v24] RSI 金叉時需已在回升（比5根前高），避免死貓彈途中給星
    prior5 = p_data.iloc[max(0, idx_pos - 5)]
    rsi_already_recovering = curr['RSI'] > prior5['RSI']

    # [v24] MACD 動能門檻：DIF 絕對值需 > 股價 × 0.1%，過濾高波動股的微小假金叉
    macd_significant = abs(curr['MACD']) > curr['Close'] * 0.001

    # [v24] 正值區域金叉：DIF > 0 且 DEA > 0（兩線均在零軸以上，趨勢健康）
    positive_zone_cross = (
        curr['MACD'] > 0 and
        not pd.isna(curr.get('Signal_Line', float('nan'))) and
        curr['Signal_Line'] > 0
    )

    # ── 死叉期間保護 ──────────────────────────────────────────
    if death_cross_active:
        if not macd_significant:
            # DIF動能太弱，不論任何條件都不給星
            normal_macd_x.append(date)
            normal_macd_y.append(curr['MACD'])
            continue

        # 死叉期間（含負值金叉 & 零軸突破）：需深度超賣 + RSI已在回升
        if not (is_deeply_oversold and rsi_already_recovering):
            normal_macd_x.append(date)
            normal_macd_y.append(curr['MACD'])
            continue

    # ── 防叢集 ────────────────────────────────────────────────
    # 正值區域金叉（健康趨勢）→ 防叢集 3 根；其他 → 7 根
    cluster_window = 3 if positive_zone_cross else 7
    has_nearby_star = any(
        0 < idx_pos - p_data.index.get_loc(sd) <= cluster_window
        for sd in already_starred
        if sd in p_data.index
    )
    if has_nearby_star:
        normal_macd_x.append(date)
        normal_macd_y.append(curr['MACD'])
        continue

    # ── [v27 方案 A] 給星條件 ────────────────────────────
    # 已砍：深度反轉(MACD<0)、近MA60+RSI回暖（命中率<50%）
    # 強制：MA60 上升 + 大盤過濾器（SPY 站 SMA200 且 SMA200 上升）
    uptrend_zero_breakout = (not death_cross_active) and zero_cross
    strong_positive = (not death_cross_active) and positive_zone_cross
    
    # 新規則：MA60 上升（強制要求）
    sma60_prior5 = p_data.iloc[max(0, idx_pos - 5)].get('SMA_60', np.nan)
    ma60_uptrend = (not pd.isna(ma60) and not pd.isna(sma60_prior5) and ma60 > sma60_prior5)
    
    # 新規則：大盤過濾器
    market_state_v27 = st.session_state.get('_market_state_v27', {})
    market_ok = market_ok_at_date(market_state_v27, date)
    
    # 方案 A: 只保留 strong_positive 和 uptrend_zero_breakout
    # （砍掉「近MA60+RSI回暖」「深度反轉」這 2 個爛條件）
    base_trigger = strong_positive or uptrend_zero_breakout
    plan_a_passes = base_trigger and ma60_uptrend and market_ok
    
    if plan_a_passes:
        star_macd_x.append(date)
        star_macd_y.append(curr['MACD'])
        already_starred.add(date)
        # 標註訊號類型供後續推演用
        trigger_label = "正值區金叉" if strong_positive else "零軸突破"
        fig.add_annotation(
            x=date, y=curr['MACD'], text="⭐起漲確認",
            showarrow=False, yshift=20, row=2, col=1,
            font=dict(color="#facc15", size=10, weight="bold")
        )
        # 收集起漲訊號詳情（供 AI 推演用）
        if '_launch_signals_v27' not in st.session_state:
            st.session_state['_launch_signals_v27'] = []
        st.session_state['_launch_signals_v27'].append({
            "date": date,
            "idx_pos": idx_pos,
            "close": float(curr['Close']),
            "trigger": trigger_label,
            "ma60_uptrend": ma60_uptrend,
            "market_ok": market_ok,
        })
    else:
        normal_macd_x.append(date)
        normal_macd_y.append(curr['MACD'])

# --- [v20 修改 / v27 BUG修復] 主圖起漲點同步到副圖：MA 金叉起漲點若尚未打星，補上星星 ---
# 注意：上面已過 Plan A 的 main_launch_dates 才會進來這裡，所以這裡直接補星是 OK 的
for date in main_launch_dates:
    if date in already_starred or date not in p_data.index:
        continue
    macd_val = p_data.loc[date, 'MACD']
    if pd.isna(macd_val):
        continue
    star_macd_x.append(date)
    star_macd_y.append(macd_val)
    already_starred.add(date)
    fig.add_annotation(
        x=date, y=macd_val, text="⭐起漲確認",
        showarrow=False, yshift=20, row=2, col=1,
        font=dict(color="#facc15", size=10, weight="bold")
    )
    # [v27] 收集主圖起漲點訊號（供 AI 推演用）
    if '_launch_signals_v27' not in st.session_state:
        st.session_state['_launch_signals_v27'] = []
    _idx_p = p_data.index.get_loc(date)
    st.session_state['_launch_signals_v27'].append({
        "date": date,
        "idx_pos": _idx_p,
        "close": float(p_data.iloc[_idx_p]['Close']),
        "trigger": "MA金叉+RSI超賣",
        "ma60_uptrend": True,  # 已過 Plan A 過濾
        "market_ok": True,
    })

if normal_macd_x:
    fig.add_trace(go.Scatter(
        x=normal_macd_x, y=normal_macd_y, mode='markers',
        marker=dict(symbol='triangle-up', size=10, color='#d8b4fe'), name='金叉'
    ), row=2, col=1)
if star_macd_x:
    fig.add_trace(go.Scatter(
        x=star_macd_x, y=star_macd_y, mode='markers',
        marker=dict(symbol='star', size=18, color='#facc15', line=dict(color='white', width=1.5)),
        name='⭐起漲確認'
    ), row=2, col=1)
if not p_data[macd_dead].empty:
    fig.add_trace(go.Scatter(
        x=p_data[macd_dead].index, y=p_data[macd_dead]['MACD'], mode='markers',
        marker=dict(symbol='triangle-down', size=10, color='#facc15'), name='死叉'
    ), row=2, col=1)
fig.add_trace(go.Scatter(x=p_data.index, y=p_data['DMA_DDD'], line=dict(color='#d8b4fe', width=1)), row=3, col=1)
fig.add_trace(go.Scatter(x=p_data.index, y=p_data['DMA_AMA'], line=dict(color='#facc15', width=1)), row=3, col=1)

fig.update_layout(
    dragmode=False if mobile_mode else 'zoom',
    height=800, template="plotly_dark",
    xaxis_rangeslider_visible=False, showlegend=False,
    margin=dict(t=10, b=10, l=10, r=10)
)
st.plotly_chart(fig, use_container_width=True)

# ──────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────
# [V26.04] 事件節點註解面板（走勢圖外、進場燈號上方）
# ──────────────────────────────────────────────────────
_ev_panel_events = st.session_state.get('_events_for_chart_v29', [])
_ev_panel_drawn  = st.session_state.get('_ev_status_v29', {}).get("drawn_count", 0)

# 每種事件的靜態說明（顏色跟走勢圖上一致）
_EV_META = {
    "CPI":  {
        "icon": "📈", "color": "#eab308",
        "title": "CPI（消費者物價指數）",
        "impact": "高影響",
        "impact_color": "#ef4444",
        "desc": "衡量通膨水位。高於預期 → 聯準會更鷹 → 股市偏空壓；低於預期 → 降息預期升 → 股市偏多衝。發布前 1-2 天常出現「盤整壓縮」，發布後 30 分鐘是最大波動窗口。",
        "strategy": "避免在 CPI 前一天加碼。若本倉有獲利，可考慮先鎖利一半等結果出來再決定。",
    },
    "PPI":  {
        "icon": "📉", "color": "#9ca3af",
        "title": "PPI（生產者物價指數）",
        "impact": "中影響",
        "impact_color": "#f97316",
        "desc": "CPI 的前導指標。PPI 高 → 成本壓力轉嫁 → 未來 CPI 可能跟上。對個股影響較小，但若 PPI 大幅超預期，往往拉升整體市場緊張情緒。",
        "strategy": "PPI 單獨影響較小，但若同週有 CPI / FOMC，要合併評估。若在 Stage 1-2 的吸籌期，PPI 造成的下殺反而是很好的加碼時機。",
    },
    "FOMC": {
        "icon": "🏛️", "color": "#f97316",
        "title": "FOMC（聯準會利率決策）",
        "impact": "極高影響",
        "impact_color": "#ef4444",
        "desc": "一年 8 次，每次都可能改變整體市場方向。不只看「升/降/不動」，更看 Dot Plot 和 Powell 記者會的措辭。即使「不動」但措辭偏鷹也會跌。「降息」但措辭猶豫也可能先漲後跌。",
        "strategy": "⚠️ FOMC 當天不建議開新倉。現有持倉若在高獲利位，可縮小倉位避開噪音。FOMC 後 1-3 天市場才能消化，那個方向才是真正趨勢。",
    },
    "非農": {
        "icon": "💼", "color": "#3b82f6",
        "title": "非農就業（NFP）",
        "impact": "高影響",
        "impact_color": "#ef4444",
        "desc": "每月第一個週五發布。就業強勁 → 聯準會沒理由急降息 → 偏空（對成長股）；就業疲弱 → 降息預期升 → 偏多。近年「壞消息 = 好消息」邏輯依然存在於成長股。",
        "strategy": "科技 / 成長股在非農弱數據時往往反彈。若走勢圖在 Stage 1-2（吸籌乖離底），非農後的急殺有時是最佳進場窗口。",
    },
    "財報": {
        "icon": "📊", "color": "#ef4444",
        "title": "財報日（Earnings）",
        "impact": "個股極高影響",
        "impact_color": "#ef4444",
        "desc": "個股最大波動催化劑。財報前隱含波動率（IV）攀升，財報後 IV crush（IV 崩塌），即使方向正確也可能因 IV crush 使選擇權盈虧縮水。財報後跳空缺口常是新趨勢的起點。",
        "strategy": "⚠️ 持有到財報 = 賭注，不是交易。若你的倉位基於技術面訊號，財報前應評估是否減倉。財報後若跳空站上前高 → 可能進入新一輪主升段（Stage 4 確認）。",
    },
}

# 只顯示這支股票圖上真正有的事件種類（不顯示空洞的通則）
if _ev_panel_events and _ev_panel_drawn > 0:
    # 收集圖上出現的事件 type（去重 + 保留原始日期）
    _seen_types: dict[str, list] = {}
    _today = pd.Timestamp.now(tz="UTC").normalize()
    for ev in _ev_panel_events:
        ev_type = ev.get("type", "")
        # 事件 type 可能是 "CPI", "PPI", "FOMC", "非農", "財報" 等
        # 也可能帶有額外文字（如 "財報 (MU)"）→ 取第一個詞
        ev_key = ev_type.split()[0] if ev_type else ""
        if ev_key not in _EV_META:
            ev_key = next((k for k in _EV_META if k in ev_type), None)
        if not ev_key:
            continue
        if ev_key not in _seen_types:
            _seen_types[ev_key] = []
        _seen_types[ev_key].append(ev.get("date", None))

    if _seen_types:
        # 把事件依日期排序（最近的先）
        def _days_to_event(dates):
            valid = [d for d in dates if d is not None]
            if not valid:
                return 9999
            deltas = [abs((pd.Timestamp(d).tz_localize(None) - _today.tz_localize(None)).days)
                      if pd.Timestamp(d).tzinfo is None
                      else abs((pd.Timestamp(d).tz_localize(None) - _today.tz_localize(None)).days)
                      for d in valid]
            return min(deltas)

        _sorted_types = sorted(_seen_types.keys(), key=lambda k: _days_to_event(_seen_types[k]))

        with st.expander(
            f"🔔 走勢圖上的事件節點說明（{len(_seen_types)} 種事件，點開看交易含義）",
            expanded=True,
        ):
            _cols = st.columns(min(len(_sorted_types), 3))
            for ci, ev_key in enumerate(_sorted_types):
                meta = _EV_META[ev_key]
                # 找最近一個日期（未來 or 今天）
                _upcoming = []
                _past = []
                for d in _seen_types[ev_key]:
                    if d is None:
                        continue
                    d_ts = pd.Timestamp(d)
                    if d_ts.tzinfo is not None:
                        d_ts = d_ts.tz_localize(None)
                    if d_ts >= _today.tz_localize(None):
                        _upcoming.append(d_ts)
                    else:
                        _past.append(d_ts)
                _upcoming.sort()
                _past.sort(reverse=True)

                if _upcoming:
                    _next_d = _upcoming[0]
                    _days_left = (_next_d - _today.tz_localize(None)).days
                    _date_badge = (
                        f'<span style="background:#22c55e22;color:#22c55e;'
                        f'border:1px solid #22c55e;padding:2px 7px;border-radius:12px;font-size:11px;">'
                        f'📅 {_next_d.strftime("%m/%d")} · {_days_left} 天後</span>'
                    )
                elif _past:
                    _last_d = _past[0]
                    _days_ago = (_today.tz_localize(None) - _last_d).days
                    _date_badge = (
                        f'<span style="background:#44444422;color:#888;'
                        f'border:1px solid #555;padding:2px 7px;border-radius:12px;font-size:11px;">'
                        f'📅 {_last_d.strftime("%m/%d")} · {_days_ago} 天前</span>'
                    )
                else:
                    _date_badge = ""

                with _cols[ci % len(_cols)]:
                    st.markdown(
                        f'<div style="background:#141416;border:1px solid {meta["color"]}44;'
                        f'border-left:3px solid {meta["color"]};border-radius:8px;padding:12px;'
                        f'margin-bottom:8px;">'
                        f'<div style="display:flex;justify-content:space-between;align-items:flex-start;'
                        f'flex-wrap:wrap;gap:4px;margin-bottom:8px;">'
                        f'<span style="color:{meta["color"]};font-weight:bold;font-size:14px;">'
                        f'{meta["icon"]} {meta["title"]}</span>'
                        f'<span style="background:{meta["impact_color"]}22;color:{meta["impact_color"]};'
                        f'border:1px solid {meta["impact_color"]};padding:1px 7px;border-radius:10px;'
                        f'font-size:10px;white-space:nowrap;">{meta["impact"]}</span>'
                        f'</div>'
                        f'{_date_badge}<br>' if _date_badge else ''
                        f'<div style="color:#ccc;font-size:12px;line-height:1.6;margin-top:6px;">'
                        f'{meta["desc"]}</div>'
                        f'<div style="background:{meta["color"]}11;border-radius:5px;padding:7px 9px;'
                        f'margin-top:8px;color:#aaa;font-size:11px;">'
                        f'<span style="color:{meta["color"]};">💡 交易策略</span><br>{meta["strategy"]}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

# ──────────────────────────────────────────────────────
# [v27] 進場節奏燈號（依最近訊號顯示目前位置）
# ──────────────────────────────────────────────────────
_entry_signals = st.session_state.get('_entry_signals_v27', [])
_launch_signals = st.session_state.get('_launch_signals_v27', [])

# 整理最近 30 根 K 棒內出現過的訊號 stage（含 launch=stage4）
recent_stages = set()
recent_30d_cutoff = len(p_data) - 30
for s in _entry_signals:
    if s["idx_pos"] >= recent_30d_cutoff:
        recent_stages.add(s["stage"])
# launch (stage4) 從 _launch_signals 來
for s in _launch_signals:
    if s.get("idx_pos", 0) >= recent_30d_cutoff:
        recent_stages.add(4)

# 找最後一個出現的 stage 作為「當前位置」
all_recent_with_stage = []
for s in _entry_signals:
    if s["idx_pos"] >= recent_30d_cutoff:
        all_recent_with_stage.append((s["idx_pos"], s["stage"], s["type"]))
for s in _launch_signals:
    if s.get("idx_pos", 0) >= recent_30d_cutoff:
        all_recent_with_stage.append((s["idx_pos"], 4, "⭐ MACD起漲確認"))
all_recent_with_stage.sort(key=lambda x: x[0])
current_stage_label = ""
if all_recent_with_stage:
    last_idx, last_stage, last_type = all_recent_with_stage[-1]
    days_ago = len(p_data) - 1 - last_idx
    current_stage_label = f"目前位置：**{last_type}**（{days_ago} 根 K 棒前）"

# 渲染 5 段燈號
def _stage_html(num, emoji, label, hit):
    bg = "linear-gradient(135deg,#22c55e,#16a34a)" if hit else "#1a1a1c"
    border = "#22c55e" if hit else "#3a3a3c"
    text_col = "white" if hit else "#666"
    return f'''
    <div style="background:{bg};border:2px solid {border};border-radius:8px;
                padding:10px 8px;text-align:center;color:{text_col};min-width:100px;">
      <div style="font-size:20px;">{emoji}</div>
      <div style="font-size:11px;font-weight:bold;margin-top:3px;">{label}</div>
      <div style="font-size:10px;opacity:0.8;">Stage {num}</div>
    </div>'''

st.markdown("### 🎯 進場節奏燈號（最近 30 根 K 棒）")
if current_stage_label:
    st.caption(current_stage_label)
else:
    st.caption("最近 30 根 K 棒內無進場訊號")

stages_html = '<div style="display:flex;gap:8px;align-items:center;margin:12px 0;flex-wrap:wrap;">'
stages_html += _stage_html(1, "🤫", "吸籌", 1 in recent_stages)
stages_html += '<div style="color:#666;font-size:18px;">→</div>'
stages_html += _stage_html(2, "💎", "乖離抄底", 2 in recent_stages)
stages_html += '<div style="color:#666;font-size:18px;">→</div>'
stages_html += _stage_html(3, "🟢", "BUY", 3 in recent_stages)
stages_html += '<div style="color:#666;font-size:18px;">→</div>'
stages_html += _stage_html(4, "⭐", "起漲確認", 4 in recent_stages)
stages_html += '</div>'
st.markdown(stages_html, unsafe_allow_html=True)
st.caption(
    "💡 **進場策略**："
    "Stage 1-2 是**最佳介入時機**（你的「吸籌 / 乖離抄底」進場習慣）；"
    "Stage 3 BUY 是**確認進場**；"
    "Stage 4 ⭐起漲確認 是**最後機會**（已漲一段，追高風險高，建議只用來確認手中持股繼續持有）。"
)

# ──────────────────────────────────────────────────────
# [v27] AI 推演面板（顯示最近的起漲訊號 + 7 維 context）
# ──────────────────────────────────────────────────────
_launch_signals = st.session_state.get('_launch_signals_v27', [])

# 📊 蒙地卡羅推演結果摘要
_mc_result = st.session_state.get('_mc_result_v27', None)
_mc_status_disp = st.session_state.get('_mc_status_v27', {})

if _mc_result:
    st.markdown("### 🔮 蒙地卡羅推演（1000 次模擬 + 7 維 context 動態調整）")
    _last_close = float(p_data['Close'].iloc[-1])
    _p10 = _mc_result['p10_final']
    _p50 = _mc_result['p50_final']
    _p90 = _mc_result['p90_final']
    _ret_p10 = (_p10 / _last_close - 1) * 100
    _ret_p50 = (_p50 / _last_close - 1) * 100
    _ret_p90 = (_p90 / _last_close - 1) * 100
    _drift_str = (f"<span style='color:#22c55e'>+{_mc_result['drift_adjust']*100:.3f}%/日加分</span>"
                   if _mc_result['drift_adjust'] > 0
                   else f"<span style='color:#ef4444'>{_mc_result['drift_adjust']*100:.3f}%/日減分</span>"
                   if _mc_result['drift_adjust'] < 0
                   else "<span style='color:#888'>±0 中性</span>")
    
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#1a1a1c,#2a1a2c);
                border-left:4px solid #eab308; padding:14px 18px; margin:10px 0;
                border-radius:8px;">
      <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:12px;font-size:13px;">
        <div>
          <div style="color:#888;font-size:11px;">悲觀情境 (p10)</div>
          <div style="color:#ef4444;font-weight:bold;font-size:18px;">${_p10:.2f}</div>
          <div style="color:#ef4444;font-size:12px;">{_ret_p10:+.2f}%</div>
        </div>
        <div>
          <div style="color:#888;font-size:11px;">中位數路徑 (p50)</div>
          <div style="color:#facc15;font-weight:bold;font-size:22px;">${_p50:.2f}</div>
          <div style="color:#facc15;font-size:12px;">{_ret_p50:+.2f}%</div>
        </div>
        <div>
          <div style="color:#888;font-size:11px;">樂觀情境 (p90)</div>
          <div style="color:#22c55e;font-weight:bold;font-size:18px;">${_p90:.2f}</div>
          <div style="color:#22c55e;font-size:12px;">{_ret_p90:+.2f}%</div>
        </div>
        <div>
          <div style="color:#888;font-size:11px;">歷史日波動率</div>
          <div style="color:#ddd;font-weight:bold;font-size:18px;">{_mc_result['sigma']*100:.2f}%</div>
          <div style="font-size:11px;">7 維調整：{_drift_str}</div>
        </div>
      </div>
      <div style="margin-top:10px;color:#aaa;font-size:12px;">
        💡 <b>解讀</b>：30 個交易日後，80% 機率股價落在 <b style="color:#ddd;">${_p10:.2f} ~ ${_p90:.2f}</b> 之間，
        中位數預期 <b style="color:#facc15;">${_p50:.2f}</b>。
        圖上「淺黃帶」= 80% 信心區、「深黃帶」= 50% 信心區、「黃實線」= 中位數路徑。<br>
        🔔 <b>垂直虛線</b>＝會發生大波動的關鍵日：
        <span style="color:#ef4444;">📊 財報</span> ·
        <span style="color:#f97316;">🏛️ FOMC</span> ·
        <span style="color:#eab308;">📈 CPI</span> ·
        <span style="color:#3b82f6;">💼 非農</span> ·
        <span style="color:#9ca3af;">📉 PPI</span>
        （只標日期、不預測方向）
      </div>
    </div>
    """, unsafe_allow_html=True)
elif _mc_status_disp and not _mc_status_disp.get("ran", False):
    # MC 沒跑成功，明確告訴使用者原因（Rule 12: fail loud）
    _reason = _mc_status_disp.get("reason", "未知原因")
    st.warning(
        f"⚠️ **蒙地卡羅模擬未產出**：{_reason}\n\n"
        f"資料筆數：{len(p_data)} 筆（MC 需要 60+ 筆 K 線資料）"
    )

# 🔔 事件節點狀態（Rule 12: fail loud）
_ev_status_disp = st.session_state.get('_ev_status_v29', {})
if _ev_status_disp:
    _drawn = _ev_status_disp.get("drawn_count", 0)
    _ev_reason = _ev_status_disp.get("reason", "")
    if _drawn > 0:
        st.caption(f"🔔 已標記 **{_drawn}** 個事件節點（垂直虛線）於走勢圖上")
    elif _ev_reason:
        st.caption(f"🔔 事件節點未顯示：{_ev_reason}")

if _launch_signals:
    # 取最近 3 個訊號（最新的在最後）
    recent_signals = _launch_signals[-3:]
    st.markdown("### 🎯 AI 起漲推演（最近訊號分析）")
    st.caption(
        f"目前共有 **{len(_launch_signals)}** 個方案 A 起漲訊號通過 "
        f"（已套用大盤過濾器 + MA60 上升）。下方顯示最近 {len(recent_signals)} 個訊號的 AI 7 維推演。"
    )
    _market_state = st.session_state.get('_market_state_v27', {})
    for sig in reversed(recent_signals):  # 最新的先顯示
        try:
            ctx = compute_signal_context(p_data, sig["idx_pos"], cur_t, _market_state)
            if ctx:
                date_str = pd.Timestamp(sig["date"]).strftime("%Y-%m-%d")
                label = f"{date_str} ⭐ 起漲確認（{sig['trigger']}）@ ${sig['close']:.2f}"
                st.markdown(render_signal_context_panel(ctx, label), unsafe_allow_html=True)
        except Exception as _e:
            pass
else:
    # 無訊號時的輔助提示
    _market_state_chk = st.session_state.get('_market_state_v27', {})
    if _market_state_chk:
        try:
            latest_date = p_data.index[-1]
            mkt_ok_now = market_ok_at_date(_market_state_chk, latest_date)
            mkt_icon = "🟢 大盤健康" if mkt_ok_now else "🔴 大盤偏空"
            mkt_msg = (
                "目前大盤健康，但此股近期未觸發方案 A 起漲訊號。可關注：💎 乖離抄底 / 🤫 吸籌 等前哨訊號。"
                if mkt_ok_now else
                "目前大盤偏空（SPY 跌破 SMA200 或 SMA200 下行），方案 A 暫不發任何起漲訊號以避免逆勢進場。"
            )
            st.info(f"**{mkt_icon}** — {mkt_msg}")
        except Exception:
            pass

# ──────────────────────────────────────────────────────
# [v27] 前哨訊號 AI 推演（吸籌 / 乖離抄底 / BUY 也享有 7 維 context）
# ──────────────────────────────────────────────────────
if _entry_signals:
    # 取最近 3 個進場訊號（依 idx_pos 排序，最新的在最後）
    sorted_entries = sorted(_entry_signals, key=lambda s: s["idx_pos"])
    recent_entries = sorted_entries[-3:]
    if recent_entries:
        with st.expander(f"🔍 前哨訊號 AI 推演（最近 {len(recent_entries)} 個 — 你的真實進場點）", expanded=False):
            st.caption(
                "**吸籌 / 乖離抄底 / BUY** 是比 ⭐起漲確認更早的進場機會。"
                "以下顯示每個訊號的 7 維 context 推演 — 幫你判斷「**該不該進場**」。"
            )
            _market_state = st.session_state.get('_market_state_v27', {})
            for sig in reversed(recent_entries):
                try:
                    ctx = compute_signal_context(p_data, sig["idx_pos"], cur_t, _market_state)
                    if ctx:
                        date_str = pd.Timestamp(sig["date"]).strftime("%Y-%m-%d")
                        stage_name = {1: "前哨警示", 2: "抄底機會", 3: "MACD 確認"}[sig["stage"]]
                        label = f"{date_str} {sig['type']}（{stage_name}）@ ${sig['close']:.2f}"
                        st.markdown(render_signal_context_panel(ctx, label), unsafe_allow_html=True)
                except Exception:
                    pass

# 雙籌碼分析圖
try:
    c1, c2 = st.columns(2)
    mf = ((p_data['Close'] - p_data['Open']) / (p_data['High'] - p_data['Low'])) * p_data['Volume']
    mf = mf.fillna(0).cumsum()
    with c1:
        st.caption("主力資金流 (Money Flow)")
        fig_mf = go.Figure(go.Scatter(x=p_data.index, y=mf, fill='tozeroy', line=dict(color='#00d4ff')))
        if len(mf) > 5:
            trend = mf.iloc[-1] - mf.iloc[-5]
            if trend > 0:
                fig_mf.add_annotation(x=p_data.index[-1], y=mf.iloc[-1], text="🟢 主力吸籌", showarrow=True, arrowhead=1, font=dict(color="#4ade80", size=12), bgcolor="#1b3a1b")
            else:
                fig_mf.add_annotation(x=p_data.index[-1], y=mf.iloc[-1], text="🔴 主力出貨", showarrow=True, arrowhead=1, font=dict(color="#ff6b6b", size=12), bgcolor="#3a1b1b")
        fig_mf.update_layout(dragmode=False if mobile_mode else 'zoom', height=250, template="plotly_dark", margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig_mf, use_container_width=True, config={'displayModeBar': False})
    with c2:
        st.caption("籌碼分佈 (主力 vs 散戶)")
        inst_mask = (p_data['Close'] > p_data['Open']) & (p_data['Volume'] > p_data['Vol_SMA5'])
        vp_all = calculate_volume_profile(p_data)
        vp_main = calculate_volume_profile(p_data, filter_mask=inst_mask)
        fig_vp = go.Figure()
        fig_vp.add_trace(go.Scatter(x=vp_all['Price'], y=vp_all['Volume'], fill='tozeroy', line=dict(color='#ffaa00', width=0), name='整體'))
        fig_vp.add_trace(go.Scatter(x=vp_main['Price'], y=vp_main['Volume'], fill='tozeroy', line=dict(color='#00d4ff', width=2), name='主力'))
        fig_vp.add_vline(x=close_v, line_dash="dash", line_color="white", annotation_text="現價")
        fig_vp.update_layout(dragmode=False if mobile_mode else 'zoom', height=250, template="plotly_dark", margin=dict(t=10, b=10, l=10, r=10), showlegend=True, legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig_vp, use_container_width=True, config={'displayModeBar': False})
except Exception:
    pass

# ==========================================
# 🚨 空頭距離指數（US + TW 雙核）
# ==========================================
if _CRISIS_AVAILABLE:
    st.markdown("---")
    st.header("🚨 空頭距離指數 (Crisis Distance Index)")
    st.caption(
        "整合 13 個資料源 → 兩個指數，告訴你距離空頭崩盤多遠。"
        "**↑ 越高越接近頂部 ｜ ↓ 越低越接近底部（反向買進機會）**　　"
        "85+ 強制清倉（系統性風險） / 75-85 高度危險 / 60-75 警戒 / "
        "40-60 中性 / 20-40 機會浮現 / **0-20 極度恐慌（逆勢買進）**"
    )
    
    # 顯示下次更新時間
    _anchor = get_cache_anchor()
    _anchors = [5, 8, 14, 20]
    try:
        _tw_now = datetime.utcnow() + timedelta(hours=8)
        _curr_hr = _tw_now.hour
        _next_anchor_hr = next((h for h in _anchors if h > _curr_hr), 5)
        _next_anchor_day = _tw_now if _next_anchor_hr > _curr_hr else _tw_now + timedelta(days=1)
        _next_str = _next_anchor_day.replace(hour=_next_anchor_hr, minute=0, second=0).strftime("%m-%d %H:%M")
        st.caption(f"🕒 下次自動更新：**台灣時間 {_next_str}** （每天 05:00 / 08:00 / 14:00 / 20:00 固定刷新）")
    except Exception:
        pass

    @st.cache_data(ttl=21600, show_spinner="抓取空頭距離指數資料...")
    def _cached_crisis(fred_key, finmind_token, anchor):
        # anchor 是固定時間錨點（台灣時間 05/08/14/20），不同就會重抓
        return crisis_engine.get_crisis_indices(fred_key, finmind_token)

    try:
        crisis = _cached_crisis(FRED_KEY, FINMIND_TOKEN, get_cache_anchor())
    except Exception as e:
        st.error(f"空頭距離指數計算失敗：{e}")
        crisis = None

    if crisis:
        # ─── 兩個大數字並排 ───
        col_us, col_tw = st.columns(2)

        def _render_market_card(col, market_data, market_label, flag):
            score = market_data["score"]
            level_text, level_color = market_data["level"]
            with col:
                if score is None:
                    st.markdown(
                        f'<div style="background:#1a1a1c;padding:18px;border-radius:10px;'
                        f'border:1px solid #333;text-align:center;">'
                        f'<div style="color:#aaa;font-size:14px;">{flag} {market_label}</div>'
                        f'<div style="color:#666;font-size:36px;margin:10px 0;">N/A</div>'
                        f'<div style="color:#888;font-size:13px;">{market_data.get("reason","資料不足")}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                    return

                st.markdown(
                    f'<div style="background:linear-gradient(135deg,#1a1a1c 0%,#2a1a2c 100%);'
                    f'padding:20px;border-radius:10px;border:2px solid {level_color};'
                    f'text-align:center;">'
                    f'<div style="color:#aaa;font-size:14px;margin-bottom:8px;">{flag} {market_label}</div>'
                    f'<div style="color:{level_color};font-size:56px;font-weight:bold;line-height:1.1;">'
                    f'{score:.1f}'
                    f'<span style="font-size:20px;color:#888;">/100</span>'
                    f'</div>'
                    f'<div style="background:{level_color}22;color:{level_color};'
                    f'border:1px solid {level_color};padding:6px 14px;border-radius:20px;'
                    f'display:inline-block;margin-top:10px;font-weight:bold;">'
                    f'{level_text}'
                    f'</div>'
                    f'<div style="color:#666;font-size:11px;margin-top:8px;">'
                    f'更新於 {market_data.get("latest_date","—")}'
                    f'</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        _render_market_card(col_us, crisis["us"], "美股空頭距離", "🇺🇸")
        _render_market_card(col_tw, crisis["tw"], "台股空頭距離", "🇹🇼")

        # ─── 子指標分解 ───
        st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)
        bd_us, bd_tw = st.columns(2)

        def _render_components(col, components, market_name):
            with col:
                st.markdown(f"**{market_name} 子指標分解**")
                if not components:
                    st.caption("（無資料）")
                    return
                for c in components:
                    name = c["name"]
                    score = c["score"]
                    weight = c["weight"] * 100
                    if score is None:
                        st.markdown(
                            f'<div style="display:flex;justify-content:space-between;'
                            f'padding:6px 10px;margin-bottom:4px;background:#262730;'
                            f'border-radius:4px;color:#666;">'
                            f'<span>{name} <small>({weight:.0f}%)</small></span>'
                            f'<span>N/A</span></div>',
                            unsafe_allow_html=True,
                        )
                        continue
                    # 色帶
                    if score >= 75:   bar_color = "#dc2626"
                    elif score >= 60: bar_color = "#f97316"
                    elif score >= 40: bar_color = "#facc15"
                    elif score >= 20: bar_color = "#84cc16"
                    else:             bar_color = "#22c55e"
                    bar_width = max(2, min(100, score))
                    st.markdown(
                        f'<div style="margin-bottom:6px;">'
                        f'<div style="display:flex;justify-content:space-between;'
                        f'font-size:13px;margin-bottom:2px;">'
                        f'<span style="color:#ddd;">{name} '
                        f'<small style="color:#888;">({weight:.0f}%)</small></span>'
                        f'<span style="color:{bar_color};font-weight:bold;">{score:.1f}</span>'
                        f'</div>'
                        f'<div style="background:#262730;height:8px;border-radius:4px;'
                        f'overflow:hidden;">'
                        f'<div style="width:{bar_width}%;height:100%;background:{bar_color};">'
                        f'</div></div></div>',
                        unsafe_allow_html=True,
                    )

        _render_components(bd_us, crisis["us"]["components"], "🇺🇸 美股")
        _render_components(bd_tw, crisis["tw"]["components"], "🇹🇼 台股")

        # ─── 歷史走勢圖 ───
        st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)
        with st.expander("📈 近 90 日指數歷史走勢", expanded=False):
            fig_crisis = make_subplots(specs=[[{"secondary_y": False}]])
            us_hist = crisis["us"].get("history")
            tw_hist = crisis["tw"].get("history")
            if us_hist is not None and not us_hist.empty:
                fig_crisis.add_trace(go.Scatter(
                    x=us_hist["Date"], y=us_hist["Composite"],
                    name="🇺🇸 美股", line=dict(color="#38bdf8", width=2),
                    hovertemplate="%{x|%Y-%m-%d}<br>美股指數: %{y:.1f}<extra></extra>",
                ))
            if tw_hist is not None and not tw_hist.empty:
                fig_crisis.add_trace(go.Scatter(
                    x=tw_hist["Date"], y=tw_hist["Composite"],
                    name="🇹🇼 台股", line=dict(color="#facc15", width=2),
                    hovertemplate="%{x|%Y-%m-%d}<br>台股指數: %{y:.1f}<extra></extra>",
                ))
            # 加門檻線
            for thr, label, color in [(85, "強制清倉", "#dc2626"),
                                        (75, "高度危險", "#f97316"),
                                        (60, "警戒區", "#facc15"),
                                        (40, "中性", "#9ca3af")]:
                fig_crisis.add_hline(y=thr, line=dict(color=color, width=1, dash="dot"),
                                      annotation_text=label,
                                      annotation_position="right",
                                      annotation_font=dict(size=10, color=color))
            fig_crisis.update_layout(
                height=380, template="plotly_dark",
                margin=dict(t=20, b=30, l=40, r=80),
                yaxis=dict(range=[0, 100], title="危機指數"),
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            st.plotly_chart(fig_crisis, use_container_width=True,
                             config={"displayModeBar": False})

        # ─── 資料源狀態 ───
        with st.expander("🔌 資料源狀態（可單獨刷新失敗源）", expanded=False):
            ds = crisis.get("data_status", {})
            ok_count = sum(1 for v in ds.values() if v)
            total = len(ds)
            failed_count = total - ok_count

            # 三個動作按鈕
            sc1, sc2, sc3 = st.columns([2, 1, 1])
            sc1.caption(f"目前 {ok_count}/{total} 個資料源可用")
            if sc2.button("🔁 重抓失敗源", help="只清除失敗源的快取，重新抓取",
                            disabled=(failed_count == 0)):
                # 從 crisis_engine 的快取目錄刪掉失敗源
                import shutil, os as _os
                # 名稱對應到 crisis_engine 的 cache key
                name_to_key = {
                    "SPX": "yf_idx_GSPC", "台灣加權": "yf_idx_TWII",
                    "VIX": "yf_idx_VIX", "VIX3M": "yf_idx_VIX3M",
                    "TSM": "yf_TSM", "2330.TW": "yf_2330.TW",
                    "TWD匯率": "yf_TWD_X",
                    "殖利率倒掛": "fred_T10Y2Y",
                    "HY信用利差": "fred_BAMLH0A0HYM2",
                    "NAAIM": "naaim", "AAII": "aaii",
                    "台股融資": "finmind_total_margin",
                    "2330 PER": "finmind_per_2330",
                }
                for name, ok in ds.items():
                    if not ok and name in name_to_key:
                        cache_file = f".crisis_cache/{name_to_key[name]}.json"
                        try:
                            if _os.path.exists(cache_file):
                                _os.remove(cache_file)
                        except Exception:
                            pass
                st.cache_data.clear()
                st.success(f"已清除 {failed_count} 個失敗源的快取，重抓中...")
                st.rerun()
            if sc3.button("💥 全部重抓", help="清除所有快取（含成功的），完整重抓"):
                import shutil
                st.cache_data.clear()
                try:
                    shutil.rmtree(".crisis_cache", ignore_errors=True)
                except Exception:
                    pass
                st.rerun()

            # 顯示快取時間資訊
            import os as _os
            cache_dir = ".crisis_cache"
            if _os.path.isdir(cache_dir):
                try:
                    mtimes = [_os.path.getmtime(_os.path.join(cache_dir, f))
                                for f in _os.listdir(cache_dir)
                                if f.endswith(".json")]
                    if mtimes:
                        oldest = datetime.fromtimestamp(min(mtimes)).strftime("%m-%d %H:%M")
                        newest = datetime.fromtimestamp(max(mtimes)).strftime("%m-%d %H:%M")
                        st.caption(f"📁 快取時間範圍：{oldest} ~ {newest}")
                except Exception:
                    pass

            cols = st.columns(4)
            for i, (name, ok) in enumerate(ds.items()):
                with cols[i % 4]:
                    icon = "✅" if ok else "❌"
                    color = "#22c55e" if ok else "#ef4444"
                    st.markdown(
                        f'<div style="padding:6px 10px;margin-bottom:4px;'
                        f'background:#262730;border-radius:4px;border-left:3px solid {color};">'
                        f'{icon} <span style="color:#ddd;font-size:13px;">{name}</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

# ==========================================
# 🕵️ SEC Form 4 內部人賣壓
# ==========================================
if _INSIDER_AVAILABLE:
    st.markdown("---")
    # [V26.01] 標題列加強制刷新按鈕
    ins_hdr_col1, ins_hdr_col2 = st.columns([5, 1])
    ins_hdr_col1.header("🕵️ 內部人賣壓指數 (SEC Form 4)")
    _force_insider_refresh = ins_hdr_col2.button(
        "🔄 強制刷新", key="insider_force_refresh",
        help="清除 cache 重新抓 SEC 資料（約 2-3 分鐘）",
    )

    _ins_anchor = get_cache_anchor()

    def _fmt_anchor(a):
        """anchor "20260520_14" → "2026-05-20 14:00"""
        try:
            return f"{a[:4]}-{a[4:6]}-{a[6:8]} {a[9:]}:00"
        except Exception:
            return a

    st.caption(
        "追蹤 S&P 100 大型權值股近 30 日的 CEO/CFO/Director 開放市場交易。"
        "賣壓比例（賣出金額 ÷ 總交易金額）越高表示內部人越看空。"
        f"**每天 05:00 / 08:00 / 14:00 / 20:00 (台灣時間) 固定刷新** ｜ "
        f"當前錨點：`{_ins_anchor}`"
    )

    # [V26.01] 把 anchor 傳進 insider_sentiment 模組，
    # 讓它的 disk cache key 也跟 anchor 連動（解決多裝置不同步問題）
    @st.cache_data(ttl=21600, show_spinner="🕵️ 正在抓取 SEC Form 4（首次約 2-3 分鐘）...")
    def _cached_insider(anchor):
        return insider_sentiment.get_insider_pressure_index(anchor=anchor)

    if _force_insider_refresh:
        # 清 Streamlit cache + 強制 module 端 force_refresh（會跳過 disk cache 重抓）
        st.cache_data.clear()
        try:
            insider = insider_sentiment.get_insider_pressure_index(
                force_refresh=True, anchor=_ins_anchor
            )
        except Exception as e:
            st.error(f"強制刷新失敗：{e}")
            insider = None
        else:
            st.success("✅ 已重新抓取 SEC Form 4 資料")
    else:
        try:
            insider = _cached_insider(_ins_anchor)
        except Exception as e:
            st.error(f"內部人賣壓計算失敗：{e}")
            insider = None

    if insider and insider.get("data_status"):
        score = insider["score"]
        level_text, level_color = insider["level"]
        stats = insider["stats"]

        # 主卡片
        ins_main_col1, ins_main_col2, ins_main_col3 = st.columns([2, 1, 1])
        with ins_main_col1:
            # [V26.01] 用 anchor 時間取代 updated_at（多裝置一致）
            _module_stamp = insider.get("updated_at", "—")
            st.markdown(
                f'<div style="background:linear-gradient(135deg,#1a1a1c 0%,#2a1c1c 100%);'
                f'padding:20px;border-radius:10px;border:2px solid {level_color};'
                f'text-align:center;">'
                f'<div style="color:#aaa;font-size:14px;">🇺🇸 內部人賣壓比例</div>'
                f'<div style="color:{level_color};font-size:48px;font-weight:bold;line-height:1.1;">'
                f'{score:.1f}<span style="font-size:20px;color:#888;">/100</span>'
                f'</div>'
                f'<div style="background:{level_color}22;color:{level_color};'
                f'border:1px solid {level_color};padding:5px 12px;border-radius:18px;'
                f'display:inline-block;margin-top:8px;font-weight:bold;">'
                f'{level_text}'
                f'</div>'
                f'<div style="color:#aaa;font-size:11px;margin-top:8px;">'
                f'資料快照：{_fmt_anchor(_ins_anchor)}'
                f'</div>'
                f'<div style="color:#555;font-size:10px;margin-top:2px;">'
                f'(模組執行時間：{_module_stamp})'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with ins_main_col2:
            st.markdown(
                f'<div style="background:#1a1a1c;padding:14px;border-radius:8px;'
                f'border:1px solid #2a2a2c;height:100%;">'
                f'<div style="color:#aaa;font-size:12px;">📈 賣出金額（30日）</div>'
                f'<div style="color:#ef4444;font-size:22px;font-weight:bold;margin-top:4px;">'
                f'${stats["sell_value"]/1e6:.1f}M</div>'
                f'<div style="color:#666;font-size:11px;">{stats["sell_count"]} 筆交易</div>'
                f'<div style="color:#aaa;font-size:12px;margin-top:10px;">📉 買進金額</div>'
                f'<div style="color:#22c55e;font-size:22px;font-weight:bold;margin-top:4px;">'
                f'${stats["buy_value"]/1e6:.1f}M</div>'
                f'<div style="color:#666;font-size:11px;">{stats["buy_count"]} 筆交易</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with ins_main_col3:
            ratio = stats["sell_ratio"]
            st.markdown(
                f'<div style="background:#1a1a1c;padding:14px;border-radius:8px;'
                f'border:1px solid #2a2a2c;height:100%;">'
                f'<div style="color:#aaa;font-size:12px;">📊 賣壓比例</div>'
                f'<div style="color:{level_color};font-size:32px;font-weight:bold;margin-top:6px;">'
                f'{ratio*100:.1f}%</div>'
                f'<div style="background:#262730;height:8px;border-radius:4px;margin-top:8px;overflow:hidden;">'
                f'<div style="width:{ratio*100}%;height:100%;background:{level_color};"></div>'
                f'</div>'
                f'<div style="color:#666;font-size:11px;margin-top:8px;">'
                f'掃描 {stats["companies_scanned"]} 家公司'
                f'</div>'
                f'<div style="color:#888;font-size:11px;margin-top:6px;">'
                f'>70% 警戒｜>85% 危險'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        # 賣最多 / 買最多 表格
        ts_col1, ts_col2 = st.columns(2)
        with ts_col1:
            st.markdown("##### 📉 賣最多 TOP 5")
            if insider["top_sellers"]:
                for i, s in enumerate(insider["top_sellers"], 1):
                    insiders_str = ", ".join(s["insiders"][:2]) if s["insiders"] else "—"
                    medal = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"][i-1]
                    st.markdown(
                        f'<div style="background:#1a1a1c;border-left:3px solid #ef4444;'
                        f'border-radius:6px;padding:8px 12px;margin-bottom:6px;">'
                        f'<div style="display:flex;justify-content:space-between;">'
                        f'<span style="font-weight:bold;color:#fff;">{medal} {s["ticker"]}</span>'
                        f'<span style="color:#ef4444;font-weight:bold;">'
                        f'${s["value"]/1e6:.2f}M</span></div>'
                        f'<div style="color:#888;font-size:11px;">'
                        f'{s["count"]} 筆 ｜ {insiders_str}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
            else:
                st.caption("無賣出資料")

        with ts_col2:
            st.markdown("##### 📈 買最多 TOP 5")
            if insider["top_buyers"]:
                for i, b in enumerate(insider["top_buyers"], 1):
                    insiders_str = ", ".join(b["insiders"][:2]) if b["insiders"] else "—"
                    medal = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"][i-1]
                    st.markdown(
                        f'<div style="background:#1a1a1c;border-left:3px solid #22c55e;'
                        f'border-radius:6px;padding:8px 12px;margin-bottom:6px;">'
                        f'<div style="display:flex;justify-content:space-between;">'
                        f'<span style="font-weight:bold;color:#fff;">{medal} {b["ticker"]}</span>'
                        f'<span style="color:#22c55e;font-weight:bold;">'
                        f'${b["value"]/1e6:.2f}M</span></div>'
                        f'<div style="color:#888;font-size:11px;">'
                        f'{b["count"]} 筆 ｜ {insiders_str}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
            else:
                st.caption("無買進資料")

        with st.expander("💡 怎麼看這個指標？", expanded=False):
            st.markdown("""
            **內部人賣壓比例** = 賣出金額 ÷ (賣出金額 + 買進金額)

            - **<40%**：內部人在加碼，極強看多訊號
            - **40-60%**：平衡，中性
            - **60-75%**：賣方占優，留意減碼
            - **>75%**：高度賣壓，內部人集中出貨
            - **>85%**：危險區，**歷史經驗常是中期頂部前 2-4 週**

            **資料來源**：SEC EDGAR Form 4（內部人法定 2 個工作天內申報）

            **追蹤範圍**：S&P 100 大型權值股，CEO/CFO/Director 在開放市場的真實買賣（已排除員工配股、選擇權行權、贈與等非市場行為）
            """)

    elif insider:
        st.warning(f"📊 內部人資料暫時無法取得：{insider.get('reason', '未知原因')}")

# ==========================================
# 💼 資金面綜合風險指數（結合空頭距離 + 內部人賣壓）
# ==========================================
if _CRISIS_AVAILABLE and _INSIDER_AVAILABLE:
    # 兩個來源都要有資料才計算
    try:
        _us_score = crisis["us"].get("score") if crisis else None
        _tw_score = crisis["tw"].get("score") if crisis else None
        _ins_score = insider.get("score") if insider and insider.get("data_status") else None
    except Exception:
        _us_score = _tw_score = _ins_score = None

    if _us_score is not None and _ins_score is not None:
        st.markdown("---")
        st.header("💼 資金面綜合風險指數")
        st.caption(
            "結合 **空頭距離（60%）+ 內部人賣壓（40%）** 給出單一綜合風險分數。"
            "**離場比例** = 建議從股市撤出的部位百分比。"
        )

        # 計算綜合風險
        combined_us = round(0.6 * _us_score + 0.4 * _ins_score, 1)
        # TW 沒有 SEC Form 4，只用空頭距離 + 美股共振（已內含）
        combined_tw = round(_tw_score, 1) if _tw_score is not None else None

        # 建議離場比例：分數越高離場越多
        def _exit_pct(score):
            if score is None:
                return None
            if score >= 85:   return 100   # 強制清倉
            elif score >= 75: return 70    # 高度危險
            elif score >= 60: return 40    # 警戒區
            elif score >= 40: return 15    # 中性
            elif score >= 20: return 5     # 機會浮現
            else:             return 0     # 極度恐慌 → 不離場（反向買進）

        def _level_color(score):
            if score is None: return ("❓ N/A", "#666")
            if score >= 85:   return ("🔴 強制清倉", "#dc2626")
            elif score >= 75: return ("🟠 高度危險", "#f97316")
            elif score >= 60: return ("🟡 警戒區", "#facc15")
            elif score >= 40: return ("⚖️ 中性區", "#9ca3af")
            elif score >= 20: return ("🟢 機會浮現", "#84cc16")
            else:             return ("💚 極度恐慌（逆勢買進）", "#22c55e")

        c_us, c_tw = st.columns(2)

        def _render_combined(col, score, market_label, flag, has_insider=True):
            with col:
                if score is None:
                    st.info("資料不足")
                    return
                exit_pct = _exit_pct(score)
                level_text, level_color = _level_color(score)
                stay_pct = 100 - exit_pct
                # 不離場（極度恐慌）特例
                buy_hint = " （考慮逢低加碼）" if score < 20 else ""

                st.markdown(
                    f'<div style="background:linear-gradient(135deg,#1a1a1c 0%,#1c2a2c 100%);'
                    f'padding:22px;border-radius:12px;border:2px solid {level_color};">'
                    f'<div style="display:flex;justify-content:space-between;align-items:center;">'
                    f'<span style="color:#ddd;font-size:15px;font-weight:bold;">'
                    f'{flag} {market_label}</span>'
                    f'<span style="color:#888;font-size:11px;">'
                    f'{"空頭+內部人" if has_insider else "僅空頭距離"}</span>'
                    f'</div>'
                    # 大數字
                    f'<div style="text-align:center;margin:10px 0 6px;">'
                    f'<span style="color:{level_color};font-size:54px;font-weight:bold;line-height:1;">'
                    f'{score:.1f}'
                    f'</span><span style="color:#888;font-size:18px;">/100</span>'
                    f'</div>'
                    f'<div style="text-align:center;margin-bottom:14px;">'
                    f'<span style="background:{level_color}22;color:{level_color};'
                    f'border:1px solid {level_color};padding:4px 12px;border-radius:16px;'
                    f'font-weight:bold;font-size:13px;">{level_text}</span>'
                    f'</div>'
                    # 離場/留場比例
                    f'<div style="background:#111;padding:12px;border-radius:8px;">'
                    f'<div style="display:flex;justify-content:space-between;font-size:13px;'
                    f'margin-bottom:6px;">'
                    f'<span style="color:#ef4444;">📤 建議離場 {exit_pct}%{buy_hint}</span>'
                    f'<span style="color:#22c55e;">📥 維持持倉 {stay_pct}%</span>'
                    f'</div>'
                    f'<div style="display:flex;height:10px;border-radius:5px;overflow:hidden;'
                    f'background:#262730;">'
                    f'<div style="width:{exit_pct}%;background:#ef4444;"></div>'
                    f'<div style="width:{stay_pct}%;background:#22c55e;"></div>'
                    f'</div>'
                    f'</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        _render_combined(c_us, combined_us, "美股綜合風險", "🇺🇸", has_insider=True)
        _render_combined(c_tw, combined_tw, "台股綜合風險", "🇹🇼", has_insider=False)

        with st.expander("💡 綜合風險指數怎麼算？", expanded=False):
            st.markdown(f"""
            #### 計算公式
            
            **美股**：`綜合風險 = 0.6 × 空頭距離 + 0.4 × 內部人賣壓`
            
            目前：`0.6 × {_us_score:.1f} + 0.4 × {_ins_score:.1f} = {combined_us:.1f}`
            
            **台股**：直接用空頭距離分數（台股無 SEC Form 4 資料）
            
            目前：`{_tw_score:.1f}` 
            
            #### 離場比例對應表
            
            | 風險分數 | 建議離場 | 含義 |
            |---|---|---|
            | 0-20 | **0%**（反向買進）| 極度恐慌、撿便宜 |
            | 20-40 | 5% | 機會浮現、保守進場 |
            | 40-60 | 15% | 中性、維持配置 |
            | 60-75 | 40% | 警戒、減碼一部分 |
            | 75-85 | 70% | 高度危險、大幅減碼 |
            | 85+ | 100% | 強制清倉、系統性風險 |
            
            #### 為什麼這樣配權重？
            
            - **空頭距離 60%**：整合 13 個資料源，覆蓋面廣（總經 + 情緒 + 技術）
            - **內部人賣壓 40%**：只有 1 個指標但**領先性強**（內部人 2 個工作天內必須申報，2-4 週領先性）
            
            兩者**互補**：空頭距離看「市場現在多熱」，內部人賣壓看「最了解公司的人現在在做什麼」。
            """)


# ==========================================
# 🤫 悄悄吸籌探測器（找尚未大漲但有徵兆的股票）
# ==========================================
try:
    import accumulation_screener as _accum
    _ACCUM_AVAILABLE = True
except ImportError:
    _ACCUM_AVAILABLE = False

if _ACCUM_AVAILABLE:
    st.markdown("---")
    accum_col1, accum_col2 = st.columns([5, 1])
    accum_col1.header("🤫 悄悄吸籌探測器")
    if accum_col2.button("🔄 強制刷新", key="accum_force_refresh",
                          help="清除快取重新掃描（約 1-2 分鐘）"):
        st.cache_data.clear()
        import shutil as _sh
        try:
            _sh.rmtree(".accumulation_cache", ignore_errors=True)
        except Exception:
            pass
        st.rerun()
    st.caption(
        "找出「**尚未大漲但有資金悄悄流入**」的股票。"
        "**前提**：股價 30 日變化在 -10% ~ +8%（不算進場 + 沒崩盤）。"
        "再依 4 個量價訊號計分。"
    )

    accum_market_col, accum_min_col = st.columns([1, 1])
    accum_market = accum_market_col.radio(
        "市場", ["🇺🇸 美股", "🇹🇼 台股"],
        horizontal=True, key="accum_market", label_visibility="collapsed",
    )
    accum_min = accum_min_col.radio(
        "最低訊號數", ["3/4 強訊號", "4/4 極強訊號", "2/4 觀察名單"],
        horizontal=True, key="accum_min", label_visibility="collapsed",
    )
    market_key = "us" if "美" in accum_market else "tw"
    min_sig_map = {"3/4 強訊號": 3, "4/4 極強訊號": 4, "2/4 觀察名單": 2}
    min_sig = min_sig_map[accum_min]

    @st.cache_data(ttl=21600, show_spinner="🔍 正在掃描吸籌訊號（首次約 1-2 分鐘）...")
    def _cached_accum(market, min_signals, anchor):
        return _accum.get_accumulation_signals(market=market, min_signals=min_signals)

    try:
        accum_result = _cached_accum(market_key, min_sig, get_cache_anchor())
    except Exception as e:
        st.error(f"掃描失敗：{e}")
        accum_result = {"accumulation": [], "edge": []}

    # 拆兩塊：吸籌 / 邊緣
    accum_list = accum_result.get("accumulation", []) if isinstance(accum_result, dict) else []
    edge_list = accum_result.get("edge", []) if isinstance(accum_result, dict) else []

    def _render_stock_card(stock):
        ticker = stock["ticker"]
        n_hit = stock["n_hit"]
        level_text, level_color = stock["level"]
        themes = stock["themes"]
        signals = stock["signals"]
        m = stock["metrics"]
        close = stock["last_close"]

        signal_dots = "".join(
            f'<span style="color:{"#22c55e" if s else "#444"};font-size:14px;">●</span>'
            for s in signals
        )
        theme_html = " ".join(
            f'<span style="background:#2a2a2c;color:#aaa;font-size:11px;'
            f'padding:2px 7px;border-radius:4px;margin-right:4px;">{t}</span>'
            for t in themes[:2]
        )
        name = get_stock_name(ticker)
        disp = f"{name} ({ticker})" if name != ticker else ticker

        st.markdown(
            f'<div style="background:#1a1a1c;border:1px solid #2a2a2c;'
            f'border-left:4px solid {level_color};'
            f'border-radius:8px;padding:14px 18px;margin-bottom:10px;">'
            f'<div style="display:flex;justify-content:space-between;align-items:center;">'
            f'<div>'
            f'<span style="font-size:16px;font-weight:bold;color:#fff;">{disp}</span>'
            f'<span style="color:#888;margin-left:8px;font-size:13px;">${close:.2f}</span>'
            f'</div>'
            f'<span style="color:{level_color};font-weight:bold;font-size:14px;">'
            f'{level_text} ({n_hit}/4)</span>'
            f'</div>'
            f'<div style="margin:6px 0;">{signal_dots}'
            f'<span style="color:#888;font-size:11px;margin-left:8px;">'
            f'價:{m["price_chg_30d"]:+.1f}% ｜ 量:{m["vol_ratio"]:.1f}x ｜ '
            f'OBV:{m["obv_pct_of_60d_max"]:.0f}% ｜ 漲跌量:{m["up_down_vol_ratio"]:.1f}x ｜ '
            f'距高:{m["dist_from_52w_high_pct"]:.0f}%'
            f'</span></div>'
            f'<div>{theme_html}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    if accum_list or edge_list:
        # 主榜：悄悄吸籌（沒大漲）
        if accum_list:
            st.markdown(f"#### 🤫 悄悄吸籌（30 日 < +8%）— 找到 **{len(accum_list)}** 檔")
            for stock in accum_list[:15]:
                _render_stock_card(stock)
        else:
            st.info("🤫 目前沒有符合「悄悄吸籌」條件的股票（市場可能過熱）")

        # 副榜：邊緣候選（已起漲但未飆）
        if edge_list:
            st.markdown(
                f"#### 📊 邊緣候選（30 日漲幅 8~15%）— 找到 **{len(edge_list)}** 檔"
            )
            st.caption(
                "已起漲、有量、有 OBV 訊號 — 介於吸籌與動能之間，"
                "**可能是吸籌完成正在突破，也可能是要回檔的尾巴**，需搭配技術面判斷。"
            )
            for stock in edge_list[:10]:
                _render_stock_card(stock)

        with st.expander("💡 訊號邏輯說明", expanded=False):
            st.markdown("""
            **悄悄吸籌的學術依據**：機構建倉時會避免推高股價，所以特徵是**量增價不漲 + 資金累積流入**。

            #### 🚫 第一道篩選
            - **30 日均量 > 50 萬股**（流動性篩選，排除冷門小型股）
            - **30 日股價變化 > -10%**（排除崩盤股）
            - **30 日股價變化 < +15%**（已飆漲超過 15% 就跳出榜外）

            通過後依 4 個訊號計分：

            | 訊號 | 含義 |
            |---|---|
            | **量能放大**（30日均 > 60日均 1.5x） | 有資金在進場 |
            | **OBV 新高** | On-Balance Volume 創 60 日新高 = 累積派發指標看好 |
            | **上漲日量 > 下跌日量 1.3x** | 買盤強過賣盤 |
            | **站上 50MA 距高 > 10%** | 在中性偏多區、不在頂部 |

            #### 🏆 兩類榜單
            - **🤫 悄悄吸籌**（30日 < +8%）：機構正在累積、價格還沒推高 → **最佳介入時機**
            - **📊 邊緣候選**（30日 8~15%）：剛起漲、可能突破中 → 看技術面決定要追或等回檔

            #### 等級分類
            - **2/4 中** 🟡 觀察名單
            - **3/4 中** 🟠 強訊號（高機率機構吸籌）
            - **4/4 中** 🔴 極強訊號（教科書級，罕見）

            ⚠️ 注意：吸籌訊號不保證一定漲，只是說明「特徵相符」。建議搭配個股戰情室的技術面再做決策。
            """)
    else:
        st.info(f"目前沒有符合 {accum_min} 的股票（市場可能太熱或太冷，導致吸籌訊號不明顯）")


# ==========================================
# 🔬 訊號變動診斷工具（為什麼今天訊號變了？）
# ==========================================
try:
    import signal_diagnostic as _sigdiag
    _SIGDIAG_AVAILABLE = True
except ImportError:
    _SIGDIAG_AVAILABLE = False

if _SIGDIAG_AVAILABLE:
    st.markdown("---")
    st.header("🔬 訊號變動診斷工具")
    st.caption(
        "**痛點解法**：「為什麼昨天命中 3/4，今天變 2/4？」"
        "這個工具會回溯過去 14 天，告訴你**哪個維度**從 ✅ 變 ❌，以及**為什麼**。"
    )

    diag_col1, diag_col2 = st.columns([3, 1])
    with diag_col1:
        diag_ticker = st.text_input(
            "輸入股票代碼（如 2354.TW、NVDA、TSLA）",
            value="",
            placeholder="例如：2354.TW",
            key="diag_ticker_input"
        ).strip().upper()
    with diag_col2:
        st.caption("")  # 空白對齊
        diag_btn = st.button("🔬 診斷訊號軌跡", key="diag_run_btn", use_container_width=True)

    if diag_btn and diag_ticker:
        with st.spinner(f"診斷 {diag_ticker} 過去 14 天訊號軌跡..."):
            try:
                # 抓 6 個月歷史資料（足夠算 14 天回溯 + 70 天基底）
                diag_t = yf.Ticker(diag_ticker)
                diag_df = diag_t.history(period="6mo", auto_adjust=False)
                if diag_df.empty:
                    st.error(f"找不到 {diag_ticker} 的資料，請確認代碼是否正確。")
                else:
                    diag_df = diag_df.reset_index()
                    diag_df["Date"] = pd.to_datetime(diag_df["Date"]).dt.tz_localize(None)
                    diag_result = _sigdiag.diagnose_signal_history(diag_df, days_back=14)

                    if diag_result is None:
                        st.warning(
                            f"⚠️ 資料不足無法診斷（需 84+ 個交易日，目前 {len(diag_df)} 天）"
                        )
                    else:
                        # ── 標題 + 總結 ──
                        summary = _sigdiag.get_summary_text(diag_result)
                        n_t = diag_result["n_hit_today"]
                        n_y = diag_result["n_hit_yesterday"]
                        change = diag_result["n_hit_change"]

                        c1, c2, c3 = st.columns(3)
                        c1.metric("昨天", f"{n_y}/4 訊號")
                        c2.metric("今天", f"{n_t}/4 訊號",
                                   f"{'+' if change > 0 else ''}{change}" if change != 0 else "無變動")
                        c3.metric("變動數", f"{len(diag_result['diff'])} 個維度")

                        # ── 今天 vs 昨天差異 ──
                        if diag_result["diff"]:
                            st.subheader("🎯 變動的維度（這就是答案）")
                            for d in diag_result["diff"]:
                                icon = "📉" if d["direction"] == "lost" else "📈"
                                color = "#ef4444" if d["direction"] == "lost" else "#22c55e"
                                action = "從 ✅ 變 ❌" if d["direction"] == "lost" else "從 ❌ 變 ✅"
                                st.markdown(
                                    f'<div style="background:#1a1a1c; border-left:4px solid {color}; '
                                    f'padding:12px 16px; margin:8px 0; border-radius:6px;">'
                                    f'<div style="font-weight:bold; color:{color};">{icon} 訊號 {d["signal_id"]}：{d["full_name"]} {action}</div>'
                                    f'<div style="margin-top:6px; color:#ccc; font-size:13px;">'
                                    f'💡 原因：{d["reason"]}'
                                    f'</div></div>',
                                    unsafe_allow_html=True
                                )
                        else:
                            st.info("✅ 今天的 4 個訊號與昨天完全相同，沒有任何維度發生變動。")

                        # ── 14 天命中數變化圖 ──
                        st.subheader("📊 過去 14 天命中數軌跡")
                        hist = diag_result["history"]
                        trace_dates = [h["date"] for h in hist]
                        trace_hits = [h["n_hit"] for h in hist]
                        trace_close = [h["close"] for h in hist]
                        hover_text = []
                        for h in hist:
                            sig_str = "".join(
                                ("✅" if h["signals"][s] else "❌") + f"S{s} "
                                for s in [2, 3, 4, 5]
                            )
                            hover_text.append(
                                f"{h['date'].strftime('%Y-%m-%d')}<br>"
                                f"命中：{h['n_hit']}/4<br>"
                                f"{sig_str}<br>"
                                f"收盤：${h['close']:.2f}"
                            )

                        fig_diag = go.Figure()
                        fig_diag.add_trace(go.Scatter(
                            x=trace_dates, y=trace_hits,
                            mode='lines+markers',
                            line=dict(color='#facc15', width=2),
                            marker=dict(size=10, color=['#22c55e' if h >= 3 else '#facc15' if h >= 2 else '#888'
                                                          for h in trace_hits]),
                            hovertext=hover_text,
                            hoverinfo='text',
                            name='命中數'
                        ))
                        fig_diag.add_hline(y=3, line_dash="dash", line_color="#22c55e",
                                            annotation_text="強訊號門檻 (3/4)",
                                            annotation_position="right",
                                            annotation_font_color="#22c55e")
                        fig_diag.update_layout(
                            height=250, template="plotly_dark",
                            margin=dict(t=20, b=20, l=10, r=10),
                            yaxis=dict(range=[-0.5, 4.5], dtick=1, title="命中數"),
                            showlegend=False,
                        )
                        st.plotly_chart(fig_diag, use_container_width=True, config={'displayModeBar': False})

                        # ── 14 天詳細表格 ──
                        with st.expander("📋 完整 14 天詳細紀錄", expanded=False):
                            table_data = []
                            for h in reversed(hist):  # 最新的在上
                                sigs = h["signals"]
                                m = h["metrics"]
                                table_data.append({
                                    "日期": h["date"].strftime("%Y-%m-%d"),
                                    "收盤": f"${h['close']:.2f}",
                                    "命中": f"{h['n_hit']}/4",
                                    "S2 量能放大": "✅" if sigs[2] else "❌",
                                    "S3 OBV 新高": "✅" if sigs[3] else "❌",
                                    "S4 買盤強勢": "✅" if sigs[4] else "❌",
                                    "S5 位置合適": "✅" if sigs[5] else "❌",
                                    "30日漲幅": f"{m['price_chg_30d']:+.1f}%",
                                    "量比": f"{m['vol_ratio']:.2f}x",
                                    "OBV%": f"{m['obv_pct']:.0f}%",
                                    "漲跌量比": f"{m['up_down_vol_ratio']:.2f}x",
                                    "距 52w 高": f"{m['dist_from_52w_high_pct']:.1f}%",
                                })
                            st.dataframe(table_data, hide_index=True, use_container_width=True)
                            st.caption(
                                "💡 **看板說明**：\n"
                                "- **S2 量能放大**：30 日均量比 60 日均量 ×1.5 以上\n"
                                "- **S3 OBV 新高**：OBV 達 60 日範圍 99% 以上\n"
                                "- **S4 買盤強勢**：30 日內上漲日量是下跌日量 ×1.3 以上\n"
                                "- **S5 位置合適**：站上 50MA 且距 52 週高點 > 10%\n"
                                "- 訊號每天會自然進出，看軌跡比看單日重要"
                            )
            except Exception as _de:
                st.error(f"診斷失敗：{type(_de).__name__}: {_de}")


# ==========================================
# 💡 反轉預警掃描器（v6 限定 S&P 100 成長股 + 訊號追蹤）
# ==========================================
try:
    import reversal_scanner as _rev
    _REV_AVAILABLE = True
except ImportError:
    _REV_AVAILABLE = False

if _REV_AVAILABLE:
    st.markdown("---")
    rev_col1, rev_col2 = st.columns([5, 1])
    rev_col1.header("💡 反轉預警掃描器（S&P 100）")
    if rev_col2.button("🔄 強制刷新", key="rev_force_refresh",
                        help="清除快取重新掃描（約 2-3 分鐘）"):
        st.cache_data.clear()

    st.caption(
        "**v6 反轉預警**：RSI 從 < 30 深度反轉到 > 45 + MACD 近零軸金叉 + OBV > 70%。"
        "**比 ⭐ 起漲確認早 ~22 天出現**。"
        "**回測勝率 81.8%（11 個樣本，需累積 25+ 樣本才正式採用）**。"
        "目前處於「**實戰觀察期**」— 訊號自動追蹤，5/10/20 天後自動評估。"
    )

    @st.cache_data(ttl=21600, show_spinner="🔍 正在掃描 S&P 100 反轉預警訊號（首次約 2-3 分鐘）...")
    def _cached_rev_scan(anchor):
        return _rev.scan_reversal_signals(lookback_days=5)

    try:
        scan_result = _cached_rev_scan(get_cache_anchor())
        tracking = _rev.update_and_get_tracking(scan_result)
    except Exception as e:
        st.error(f"反轉預警掃描失敗：{e}")
        scan_result = {"signals": [], "scan_time": "", "total_scanned": 0}
        tracking = {"total_signals": 0, "evaluated_count": 0, "pending_count": 0,
                    "win_rate_10d": 0, "avg_return_10d": 0, "all_signals": [], "can_decide": False}

    # ─── 統計卡片 ───
    track_col1, track_col2, track_col3, track_col4 = st.columns(4)
    with track_col1:
        st.metric("本次掃描", f"{len(scan_result.get('signals', []))} 個",
                    f"近 5 天內 / {scan_result.get('total_scanned', 0)} 支")
    with track_col2:
        st.metric("累積樣本", f"{tracking['total_signals']} 個",
                    f"{tracking['evaluated_count']} 已評估")
    with track_col3:
        win_color = "🟢" if tracking['win_rate_10d'] >= 60 else "🟡" if tracking['win_rate_10d'] >= 50 else "🔴"
        st.metric("實戰勝率", f"{tracking['win_rate_10d']}%",
                    f"{win_color} {'達標' if tracking['win_rate_10d'] >= 60 else '觀察中'}")
    with track_col4:
        st.metric("平均報酬", f"{tracking['avg_return_10d']:+.2f}%",
                    "10 日後")

    # 採用建議
    if tracking['can_decide']:
        if tracking['win_rate_10d'] >= 60:
            st.success(f"✅ **可正式採用**：實戰勝率 {tracking['win_rate_10d']}%（{tracking['evaluated_count']} 個樣本）達標")
        else:
            st.error(f"❌ **建議放棄**：實戰勝率 {tracking['win_rate_10d']}% 未達 60%")
    else:
        st.info(f"⏳ 樣本不足（需 25 個已評估）— 還缺 {max(0, 25 - tracking['evaluated_count'])} 個")

    # ─── 本次掃描到的訊號 ───
    current_sigs = scan_result.get("signals", [])
    if current_sigs:
        st.subheader(f"🎯 本次掃描出 {len(current_sigs)} 個新鮮訊號")
        rev_data = []
        for s in current_sigs:
            rev_data.append({
                "股票": s["ticker"],
                "訊號日": s["date"],
                "天數": f"{s.get('days_ago', 0)} 天前",
                "RSI 反轉": f"{s['rsi_was']} → {s['rsi_now']}",
                "OBV": f"{s['obv_pct']}%",
                "距 SMA60": f"{s['dist_sma60']:+.1f}%",
                "20 日動能": f"{s['momentum_20d']:+.2f}%",
                "進場價": f"${s['close']:.2f}",
            })
        st.dataframe(rev_data, hide_index=True, use_container_width=True)
    else:
        st.info("💤 本次掃描未發現新訊號（市場可能在強勢區或弱勢區，反轉條件不易觸發）")

    # ─── 歷史訊號追蹤紀錄 ───
    with st.expander(f"📋 歷史訊號追蹤紀錄（共 {tracking['total_signals']} 筆）", expanded=False):
        if tracking['all_signals']:
            history_data = []
            # 由新到舊
            sorted_sigs = sorted(tracking['all_signals'],
                                  key=lambda x: x.get('date', ''), reverse=True)
            for s in sorted_sigs[:50]:  # 顯示最近 50 筆
                def fmt(v):
                    if v is None: return "—"
                    return f"{v:+.2f}%"
                
                # 分類標記
                r10 = s.get('actual_10d')
                if r10 is None:
                    badge = "⏳ 待評估"
                elif r10 >= 5:
                    badge = "✅ 優秀"
                elif r10 >= 0:
                    badge = "👍 有效"
                elif r10 >= -5:
                    badge = "😟 不佳"
                else:
                    badge = "❌ 失敗"
                
                history_data.append({
                    "股票": s['ticker'],
                    "訊號日": s['date'],
                    "RSI 反轉": f"{s['rsi_was']}→{s['rsi_now']}",
                    "OBV": f"{s['obv_pct']}%",
                    "進場價": f"${s['close']:.2f}",
                    "+5日": fmt(s.get('actual_5d')),
                    "+10日": fmt(s.get('actual_10d')),
                    "+20日": fmt(s.get('actual_20d')),
                    "結果": badge,
                })
            st.dataframe(history_data, hide_index=True, use_container_width=True)
            st.caption(
                f"💡 系統每天自動追蹤這些訊號的實際表現。**5 天後**填入「+5日」、"
                f"**10 天後**填入「+10日」、**20 天後**填入「+20日」。"
                f"累積到 25 個已評估樣本後，會給你「採用 / 放棄」建議。"
            )
        else:
            st.caption("尚無歷史訊號紀錄（每次掃描到新訊號會自動加入）")


# ==========================================
# 🎯 [V26.02] AI 目標掃描器（可切換股票池：S&P 100 核心 / 擴大熱門 ~200）
# ==========================================
if _REV_AVAILABLE:  # 共用 reversal_scanner 的 generate_monte_carlo_bands
    st.markdown("---")
    tgt_hdr_col1, tgt_hdr_col2 = st.columns([5, 1])
    tgt_hdr_col1.header("🎯 AI 目標掃描器")
    if tgt_hdr_col2.button("🔄 強制刷新", key="target_force_refresh",
                            help="清除本掃描器的快取重新跑"):
        # [V26.02] 只清掉本掃描器的 cache，不動其他掃描器
        try:
            _cached_target_scan.clear()
        except Exception:
            st.cache_data.clear()
        st.rerun()

    _tgt_anchor = get_cache_anchor()

    # ── 股票池定義 ──
    # S&P 100 核心（與 insider_sentiment 的 SP100_TICKERS 對齊）
    _SP100_CORE_TICKERS = [
        "NVDA", "MSFT", "AAPL", "GOOG", "GOOGL", "AMZN", "META", "TSLA", "BRK-B",
        "AVGO", "JPM", "V", "WMT", "LLY", "MA", "ORCL", "XOM", "JNJ", "HD", "ABBV",
        "PG", "BAC", "COST", "KO", "TMUS", "PLTR", "CVX", "CSCO", "NFLX", "ABT",
        "WFC", "PEP", "CRM", "MRK", "ACN", "TMO", "AMD", "MCD", "GE", "ADBE",
        "DIS", "AXP", "LIN", "IBM", "CAT", "PM", "QCOM", "MS", "VZ", "GS",
        "INTU", "T", "ISRG", "RTX", "TXN", "NEE", "BX", "AMGN", "BKNG", "PFE",
        "C", "SCHW", "UPS", "LOW", "BLK", "DHR", "NOW", "UNH", "HON", "ELV",
        "SPGI", "ADP", "TJX", "SYK", "ETN", "DE", "GILD", "VRTX", "MMC", "PGR",
        "MDT", "PANW", "REGN", "MU", "ADI", "SBUX", "LMT", "BSX", "CMCSA", "BMY",
        "MO", "PYPL", "FI", "ICE", "DUK", "AMAT", "TGT", "MDLZ", "INTC", "USB",
    ]

    # [V26.02] 額外熱門股票（補齊 SP100 之外的散戶熱門股，含 SNDK）
    _EXTRA_POPULAR_TICKERS = [
        # 半導體 / 記憶體 / AI 硬體
        "SNDK", "WDC", "MRVL", "ON", "MCHP", "KLAC", "LRCX", "ASML", "TSM", "ARM",
        "NXPI", "SMCI", "ANET", "ALAB", "ENTG", "SWKS", "QRVO", "TER", "MPWR",
        # 雲端 / SaaS
        "SNOW", "NET", "DDOG", "MDB", "CRWD", "ZS", "OKTA", "ZM", "TEAM", "WDAY",
        "SHOP", "TWLO", "NTNX", "ESTC", "HUBS", "DOCU", "DBX", "ASAN", "MNDY", "GTLB",
        # AI / 量子 / 數據
        "AI", "PATH", "SOUN", "IONQ", "RGTI", "QBTS", "BBAI",
        # 消費 / 網路 / 媒體
        "ROKU", "PINS", "SPOT", "ABNB", "DASH", "U", "RBLX", "EA", "TTWO", "ETSY",
        "EBAY", "W", "CHWY", "DKNG", "BMBL", "MTCH", "WBD", "PARA",
        # 金融科技
        "COIN", "HOOD", "AFRM", "UPST", "SOFI", "NU", "MELI",
        # 電動車 / 汽車
        "RIVN", "LCID", "NIO", "XPEV", "LI", "F", "GM",
        # 中概股
        "BABA", "JD", "PDD", "BIDU", "BILI", "TME", "VIPS",
        # 醫療 / 生技
        "MRNA", "BNTX", "BIIB", "ILMN", "DXCM", "EW", "ZTS", "ALGN", "IDXX", "GH",
        # 能源
        "SLB", "OXY", "EOG", "MPC", "VLO", "PSX", "DVN", "FANG", "COP",
        # 旅遊 / 航空 / 服務
        "UBER", "LYFT", "DAL", "UAL", "AAL", "CCL", "RCL", "MAR", "HLT",
        # 其他熱門
        "SE", "GRAB",
    ]

    # 去重（SP100 + extra，保留順序）
    _EXTENDED_TGT_TICKERS = list(dict.fromkeys(_SP100_CORE_TICKERS + _EXTRA_POPULAR_TICKERS))

    _TGT_UNIVERSE_MAP = {
        f"🎯 S&P 100 核心（{len(_SP100_CORE_TICKERS)} 檔，~3-5 分鐘）": ("core", _SP100_CORE_TICKERS),
        f"🚀 擴大熱門（{len(_EXTENDED_TGT_TICKERS)} 檔，~7-12 分鐘）": ("extended", _EXTENDED_TGT_TICKERS),
    }

    # ── 範圍選擇器（一定要在 scan 前）──
    universe_label = st.radio(
        "📊 掃描範圍",
        list(_TGT_UNIVERSE_MAP.keys()),
        horizontal=True,
        key="tgt_universe",
        help="切換不會立刻重跑 — 各範圍快取獨立，已掃過的會秒回。",
    )
    universe_key, selected_tickers = _TGT_UNIVERSE_MAP[universe_label]

    st.caption(
        "用蒙地卡羅 30 日推演的 **p50 中位數（短）/ p90 樂觀（長）** 當目標價，"
        "計算每支股票的「目標 vs 現價」上漲空間。**演算法跟個股卡片右上「AI 目標 & 強弱」一致。**"
        f" 每日 05:00 / 08:00 / 14:00 / 20:00 (台灣時間) 固定刷新 ｜ "
        f"當前範圍：**{len(selected_tickers)} 檔** ｜ 錨點：`{_tgt_anchor}`"
    )

    @st.cache_data(ttl=21600, show_spinner="🎯 正在掃描 AI 目標（首次需數分鐘，視範圍而定）...")
    def _cached_target_scan(anchor, universe_key, tickers_tuple):
        """批次跑指定股票池的 MC + 目標價計算
        重用既有 calculate_indicators() + predict_target_and_rating() + generate_monte_carlo_bands()
        cache key 跟 anchor + universe_key 綁定，不同範圍快取獨立。
        """
        tickers = list(tickers_tuple)
        results = []
        # 批次下載提升效率（yf 一次抓很多檔比逐檔快）
        try:
            batch = yf.download(
                tickers, period="1y",
                auto_adjust=False, group_by="ticker",
                progress=False, threads=True,
            )
        except Exception:
            batch = None

        for tk in tickers:
            try:
                # 從 batch 取出單股 df；若 batch 失敗則退回單檔抓
                if batch is not None and isinstance(batch.columns, pd.MultiIndex) \
                        and tk in batch.columns.get_level_values(0):
                    hist = batch[tk].dropna(how="all").copy()
                else:
                    hist = yf.Ticker(tk).history(period="1y", auto_adjust=False)

                if hist is None or hist.empty or len(hist) < 80:
                    continue

                hist = calculate_indicators(hist)
                if "Bollinger_Upper" not in hist.columns:
                    continue
                if pd.isna(hist["Bollinger_Upper"].iloc[-1]) or pd.isna(hist["SMA_20"].iloc[-1]):
                    continue

                price = float(hist["Close"].iloc[-1])
                if price <= 0:
                    continue

                p_data = hist.tail(120)
                if len(p_data) < 60:
                    continue

                # 跑 MC（drift_adjust=0，batch 掃描不算 7 維 context 漂移，
                # 跟個股卡片接近但不完全相同；數字差約 1-3%，方向與順序一致）
                mc = _rev.generate_monte_carlo_bands(
                    p_data, days=30, n_simulations=1000, drift_adjust=0.0
                )
                if mc is None:
                    continue

                p10_final = float(mc["p10"][-1])
                p50_final = float(mc["p50"][-1])
                p90_final = float(mc["p90"][-1])

                # 用既有函式算目標 + 評等（跟個股卡片完全一致）
                _mc_for_target = {"p50_final": p50_final, "p90_final": p90_final}
                t_s, t_l, rating = predict_target_and_rating(hist, _mc_for_target)

                upside_s = (t_s - price) / price * 100
                upside_l = (t_l - price) / price * 100
                downside = (p10_final - price) / price * 100

                results.append({
                    "ticker": tk,
                    "price": price,
                    "target_s": t_s,
                    "target_l": t_l,
                    "p10": p10_final,
                    "upside_s": upside_s,
                    "upside_l": upside_l,
                    "downside": downside,
                    "rating": rating,
                })
            except Exception:
                continue
        return {"results": results, "scanned": len(tickers), "ok": len(results)}

    try:
        target_scan = _cached_target_scan(_tgt_anchor, universe_key, tuple(selected_tickers))
    except Exception as e:
        st.error(f"AI 目標掃描失敗：{e}")
        target_scan = {"results": [], "scanned": 0, "ok": 0}

    res_list = target_scan.get("results", [])

    # ── 統計摘要 ──
    if res_list:
        avg_up_s = float(np.mean([r["upside_s"] for r in res_list]))
        avg_up_l = float(np.mean([r["upside_l"] for r in res_list]))
        positive_count = sum(1 for r in res_list if r["upside_s"] > 0)
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("掃描成功", f"{target_scan['ok']} / {target_scan['scanned']}",
                  f"{target_scan['ok']/max(target_scan['scanned'],1)*100:.0f}%")
        s2.metric("平均短期空間", f"{avg_up_s:+.2f}%")
        s3.metric("平均長期空間", f"{avg_up_l:+.2f}%")
        s4.metric("MC 看多比例", f"{positive_count}/{target_scan['ok']}",
                  f"{positive_count/max(target_scan['ok'],1)*100:.0f}% 短期上漲")

    # ── 控制列 ──
    tgt_ctrl1, tgt_ctrl2, tgt_ctrl3 = st.columns([2, 1.2, 1])
    sort_mode = tgt_ctrl1.radio(
        "排序依據",
        ["📈 短期上漲空間 (MC p50)", "🚀 長期上漲空間 (MC p90)", "📉 下跌風險 (MC p10)"],
        horizontal=True, key="tgt_sort_mode",
    )
    top_n_tgt = tgt_ctrl2.radio("顯示 Top N", [10, 20, 30, 50],
                                 horizontal=True, index=2, key="tgt_topn")
    only_uptrend = tgt_ctrl3.checkbox("僅顯示強勢股", value=False, key="tgt_uptrend_only",
                                       help="只顯示現價 > SMA_20 的股票")

    # ── 篩選 + 排序 ──
    filtered = list(res_list)
    if only_uptrend:
        filtered = [r for r in filtered if r["rating"] == "強勢"]

    if "短期" in sort_mode:
        filtered = sorted(filtered, key=lambda r: r["upside_s"], reverse=True)
        sort_key_label = "短期空間"
    elif "長期" in sort_mode:
        filtered = sorted(filtered, key=lambda r: r["upside_l"], reverse=True)
        sort_key_label = "長期空間"
    else:
        filtered = sorted(filtered, key=lambda r: r["downside"])  # 越負（下跌風險越大）越前面
        sort_key_label = "下跌風險"

    # ── 結果表 ──
    if filtered:
        st.subheader(f"🎯 排序結果 TOP {min(top_n_tgt, len(filtered))}（依「{sort_key_label}」）")
        rows = []
        for i, r in enumerate(filtered[:top_n_tgt], 1):
            name = get_stock_name(r["ticker"])
            disp = f"{name} ({r['ticker']})" if name and name != r["ticker"] else r["ticker"]
            rating_disp = "🦁 強勢" if r["rating"] == "強勢" else "🐢 持有"
            rows.append({
                "#": i,
                "股票": disp,
                "現價": f"${r['price']:.2f}",
                "短期目標 p50": f"${r['target_s']:.2f}",
                "短期空間": f"{r['upside_s']:+.2f}%",
                "長期目標 p90": f"${r['target_l']:.2f}",
                "長期空間": f"{r['upside_l']:+.2f}%",
                "下跌風險 p10": f"{r['downside']:+.2f}%",
                "強弱": rating_disp,
            })
        st.dataframe(rows, hide_index=True, use_container_width=True)
    elif res_list:
        st.info("⏳ 沒有符合篩選條件的股票。試試取消「僅顯示強勢股」勾選。")
    else:
        st.info("⏳ 掃描中或失敗。請等待或按「強制刷新」。")

    # ── 說明 ──
    with st.expander("💡 演算法與使用方式", expanded=False):
        st.markdown("""
**演算法**：跟個股卡片右上「AI 目標 & 強弱」用一樣的蒙地卡羅 30 日推演。

- **短期目標 (p50)**：1000 次模擬的中位數，代表「最可能達到的價格」
- **長期目標 (p90)**：1000 次模擬的第 90 百分位，代表「樂觀情境上緣」
- **下跌風險 (p10)**：第 10 百分位，代表「悲觀情境下緣」

**三種排序怎麼用**：
- **短期上漲空間** → 找 MC 推演認為「現價偏低、最有機會回升」的股 → 偏短打
- **長期上漲空間** → 找 p90 比現價高最多的「飆股潛力股」→ 偏長放
- **下跌風險** → 看 p10 最差的股（可能避開，或拿來做空研究）

**注意**：
- 跟個股卡片數字會有 1-3% 的差異 — batch 掃描沒套用 7 維 context 的漂移調整，但方向跟順序一致。
- 蒙地卡羅是統計推演，不保證實際走勢。應搭配「反轉預警」「悄悄吸籌」「內部人賣壓」綜合判斷。
- 100 檔掃完約 3-5 分鐘，結果快取 6 小時。
""")


# ==========================================
# 📊 類股動能榜（原火箭類股探測器）
# ==========================================
try:
    import rocket_screener as _rocket
    _ROCKET_AVAILABLE = True
except ImportError:
    _ROCKET_AVAILABLE = False

if _ROCKET_AVAILABLE:
    st.markdown("---")
    st.header("📊 類股動能榜")
    st.caption("看「現在哪些主題在熱」— 注意：高動能 ≠ 該買，可能是漲到尾巴。建議搭配「悄悄吸籌探測器」一起看")

    rkt_col1, rkt_col2, rkt_col3 = st.columns([1, 1, 1])
    rkt_market = rkt_col1.radio("市場", ["🇺🇸 美股", "🇹🇼 台股"],
                                 horizontal=True, label_visibility="collapsed")
    rkt_period = rkt_col2.radio("週期", ["1w 本週", "1m 本月", "3m 本季"],
                                 horizontal=True, label_visibility="collapsed")
    rkt_force = rkt_col3.button("🔄 強制刷新", help="清除快取重新掃描（約1-2分鐘）")

    market_key = "us" if "美" in rkt_market else "tw"
    period_key = rkt_period.split()[0]

    @st.cache_data(ttl=21600, show_spinner="🔍 正在掃描類股漲幅（首次約1-2分鐘）...")
    def _cached_rocket(market, period, anchor):
        return _rocket.get_top_themes(market=market, top_n=5, period=period)

    if rkt_force:
        st.cache_data.clear()

    try:
        rocket_data = _cached_rocket(market_key, period_key, get_cache_anchor())
    except Exception as e:
        st.error(f"掃描失敗：{e}")
        rocket_data = []

    if rocket_data:
        period_label = {"1w": "本週", "1m": "本月", "3m": "本季"}.get(period_key, "")
        st.markdown(f"#### 🏆 {period_label}漲幅 TOP {len(rocket_data)} 主題")

        for rank, item in enumerate(rocket_data, 1):
            avg = item["avg_ret"]
            med = item["median_ret"]
            hot = item["hot_count"]
            total = item["total_count"]
            tops = item["top_stocks"]

            # 顏色依漲幅
            if avg >= 10:  bar_c = "#dc2626"
            elif avg >= 5: bar_c = "#f97316"
            elif avg >= 2: bar_c = "#facc15"
            elif avg >= 0: bar_c = "#4ade80"
            else:          bar_c = "#9ca3af"

            medal = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"][rank-1]

            # 代表股票 HTML
            top_html = " ".join(
                f'<span style="background:#2a2a2c;padding:2px 7px;border-radius:4px;'
                f'font-size:12px;color:{"#4ade80" if r>0 else "#ff6b6b"};">'
                f'{t} {r:+.1f}%</span>'
                for t, r in tops
            )

            st.markdown(
                f'<div style="background:#1a1a1c;border:1px solid #2a2a2c;border-left:4px solid {bar_c};'
                f'border-radius:8px;padding:14px 18px;margin-bottom:10px;">'
                f'<div style="display:flex;justify-content:space-between;align-items:center;">'
                f'<span style="font-size:17px;font-weight:bold;">{medal} {item["theme"]}</span>'
                f'<span style="color:{bar_c};font-size:22px;font-weight:bold;">{avg:+.1f}%</span>'
                f'</div>'
                f'<div style="color:#888;font-size:12px;margin:4px 0 8px;">'
                f'中位數 {med:+.1f}% ｜ {hot}/{total} 檔上漲'
                f'</div>'
                f'<div>{top_html}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        # 更新時間
        if rocket_data:
            upd = rocket_data[0].get("updated_at", "")
            st.caption(f"⏱ 資料時間：{upd}（快取 4 小時）｜ 點「強制刷新」可立即更新")
    else:
        st.info("正在準備資料，請稍候或點「強制刷新」")

# ==========================================
# 籌碼全面指揮中心
# ==========================================
st.markdown("---")
is_tw = ".TW" in cur_t or ".TWO" in cur_t

if is_tw:
    st.header("🏛️ 籌碼全面指揮中心 (FinMind 真實數據)")

    def get_retail_stage(ratio):
        if ratio is None:
            return "資料讀取中", "#666"
        if ratio >= 25:
            return "💥 極度瘋狂 (大崩盤警戒)", "#dc2626"
        elif ratio >= 10:
            return "⚠️ 過度樂觀 (反向偏空)", "#f97316"
        elif ratio > -10:
            return "⚖️ 多空膠著 (無明顯方向)", "#9ca3af"
        elif ratio > -25:
            return "🍀 過度悲觀 (反向偏多)", "#84cc16"
        else:
            return "🚀 極度恐慌 (絕佳軋空底)", "#22c55e"

    m_data = get_tw_all_macro()
    retail_ratio = m_data.get("retail_ratio")
    oi_df = m_data.get("oi_df", pd.DataFrame())
    spot_df = m_data.get("spot_df", pd.DataFrame())
    current_tx_oi = m_data.get("current_tx_oi")
    is_api_success = m_data.get("success", False)
    err_msg = m_data.get("msg", "")

    if err_msg:
        st.warning(f"⚠️ API 狀態回報: {err_msg}")

    mc1, mc2, mc3 = st.columns(3)
    with mc1:
        if retail_ratio is not None:
            st.metric("小台散戶多空比", f"{retail_ratio}%", delta="反指標高風險" if retail_ratio > 15 else "反指標安全" if retail_ratio < -15 else "觀望", delta_color="inverse" if retail_ratio > 15 else "normal")
        else:
            st.metric("小台散戶多空比", "N/A")
    with mc2:
        if current_tx_oi is not None:
            st.metric("外資大台淨部位", f"{current_tx_oi:,.0f} 口", delta="偏多" if current_tx_oi > 0 else "偏空", delta_color="normal" if current_tx_oi > 0 else "inverse")
        else:
            st.metric("外資大台淨部位", "N/A")
    with mc3:
        st.metric("FinMind API 狀態", "VIP 連網中" if is_api_success else "斷線/限制中", "正常" if is_api_success else "-異常")

    bc1, bc2 = st.columns([0.6, 0.4])

    with bc1:
        st.markdown('<div class="ai-box" style="text-align:left;"><h4 style="color:white; margin:0; margin-bottom:10px;">📈 指標一：外資大台期貨淨未平倉 (動態基準版)</h4>', unsafe_allow_html=True)
        if is_api_success and not oi_df.empty:
            fig_oi = go.Figure()
            oi_mean = oi_df['Net_OI'].mean()
            fig_oi.add_trace(go.Scatter(
                x=oi_df['Date'], y=oi_df['Net_OI'], mode='lines',
                line=dict(color='rgba(250, 204, 21, 0.4)', width=1.5),
                name='單日部位', hovertemplate='<b>日期</b>: %{x|%Y-%m-%d}<br><b>單日淨部位</b>: %{y:,.0f} 口<extra></extra>'
            ))
            if 'OI_5MA' in oi_df.columns:
                fig_oi.add_trace(go.Scatter(
                    x=oi_df['Date'], y=oi_df['OI_5MA'], mode='lines',
                    line=dict(color='#ffffff', width=3),
                    name='5日趨勢', hovertemplate='<b>日期</b>: %{x|%Y-%m-%d}<br><b>5日均線</b>: %{y:,.0f} 口<extra></extra>'
                ))
            fig_oi.add_hline(y=oi_mean, line_dash="dash", line_color="#888", opacity=0.8, annotation_text=f"半年平均水位 ({oi_mean:,.0f})", annotation_position="top left")
            fig_oi.update_layout(
                height=320, template="plotly_dark", margin=dict(t=10, b=10, l=10, r=10), showlegend=False,
                xaxis=dict(tickformat="%Y-%m", dtick="M1", rangeslider=dict(visible=True, thickness=0.08))
            )
            st.plotly_chart(fig_oi, use_container_width=True, config={'displayModeBar': False})
            st.caption("💡 歷史高點的『結構性避險』讓 0 軸失去意義。請觀察【白線】是否高於【灰線(半年均值)】來判斷真實多空。")
        else:
            st.markdown('<div style="height:280px; display:flex; align-items:center; justify-content:center; color:#666; font-size:18px; border:1px dashed #444; border-radius:5px;">N/A (資料讀取失敗)</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with bc2:
        if retail_ratio is not None:
            stage_text_label, stage_color = get_retail_stage(retail_ratio)
            tw_html = (
                f'<div class="ai-box" style="text-align:left; height:100%;">'
                f'<h4 style="color:white; margin:0; margin-bottom:15px;">🧠 小台指散戶多空比</h4>'
                f'<p style="color:#aaa; font-size:13px; margin-bottom:10px;">由期交所真實未平倉數據反推，為極具參考價值的「反指標」。</p>'
                f'<div style="text-align:center; padding: 10px 0;">'
                f'<div style="font-size: 14px; color: #ccc;">最新散戶多空比</div>'
                f'<div style="font-size: 48px; font-weight: 900; color: {stage_color};">{"+" if retail_ratio > 0 else ""}{retail_ratio}%</div>'
                f'<div style="font-size: 18px; color: {stage_color}; font-weight: bold; margin-top: 0px;">{stage_text_label}</div>'
                f'<div style="margin-top: 15px; font-size: 12px; color: #aaa; background-color: #222; padding: 8px; border-radius: 5px;">'
                f'<b>⚖️ 警戒標尺：</b><br>'
                f'<span style="color:#dc2626">極危 (≥+25%)</span> | <span style="color:#f97316">偏空 (≥+10%)</span> | <span style="color:#84cc16">偏多 (≤-10%)</span> | <span style="color:#22c55e">極安 (≤-25%)</span>'
                f'</div></div>'
                f'<div style="margin-top: 15px; padding:10px; background-color:#262730; border-radius:5px; border:1px solid #444; font-size:13px; color:#ddd; line-height: 1.6;">'
                f'<b>💡 實戰運用：</b><br>'
                f'散戶通常是市場的反指標。當上方數值出現紅色（散戶瘋狂做多），代表大戶正在倒貨，請居高思危；當數值出現綠色（散戶恐慌放空），代表大戶正在吸籌，容易出現軋空行情。'
                f'</div></div>'
            )
        else:
            tw_html = '<div class="ai-box" style="text-align:left; height:100%;"><h4 style="color:white; margin:0; margin-bottom:15px;">🧠 小台指散戶多空比</h4><div style="text-align:center; padding: 40px 0;"><div style="font-size: 36px; font-weight: 900; color: #666;">N/A</div><div style="font-size: 14px; color: #666; font-weight: bold; margin-top: 5px;">(API 暫無資料)</div></div></div>'
        st.markdown(tw_html, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="ai-box" style="text-align:left;"><h4 style="color:white; margin:0; margin-bottom:10px;">📊 指標二：外資現貨買賣超 (全寬大局版)</h4>', unsafe_allow_html=True)
    if not spot_df.empty:
        fig_spot = go.Figure(go.Bar(
            x=spot_df['Date'], y=spot_df['Net_Buy'],
            marker_color=['#4ade80' if v > 0 else '#ff6b6b' for v in spot_df['Net_Buy']],
            hovertemplate='<b>日期</b>: %{x|%Y-%m-%d}<br><b>買賣超</b>: %{y:,.1f} 億<extra></extra>'
        ))
        fig_spot.add_hline(y=150, line_dash="dot", line_color="#22c55e", annotation_text="大買 (>150億)", annotation_font_color="#22c55e", opacity=0.5)
        fig_spot.add_hline(y=-150, line_dash="dot", line_color="#ef4444", annotation_text="大賣 (<-150億)", annotation_font_color="#ef4444", annotation_position="bottom left", opacity=0.5)
        fig_spot.update_layout(height=250, template="plotly_dark", margin=dict(t=10, b=10, l=10, r=10), xaxis=dict(tickformat="%m-%d", dtick="M1"))
        st.plotly_chart(fig_spot, use_container_width=True, config={'displayModeBar': False})
        st.caption("💡 搭配上方大台圖表，若出現『期貨大空單 + 現貨大買』即為假避險真拉抬。")
    else:
        st.markdown('<div style="height:180px; display:flex; align-items:center; justify-content:center; color:#666;">N/A (資料讀取中)</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

else:
    naaim_df, naaim_status = get_naaim_data()
    (aaii_bull, aaii_neu, aaii_bear), aaii_status = get_aaii_data()

    # 動態顯示資料來源狀態
    status_map = {
        "real":   ("🟢 即時抓取", "#22c55e"),
        "cached": ("🟡 快取資料", "#facc15"),
        "demo":   ("⚠️ 示範數據", "#ef4444"),
    }
    naaim_label, naaim_color = status_map.get(naaim_status, status_map["demo"])
    aaii_label, aaii_color = status_map.get(aaii_status, status_map["demo"])

    if naaim_status == "demo" or aaii_status == "demo":
        st.header("🏛️ 美股專屬：總經情緒雙核觀測站")
        st.caption(
            "ℹ️ 部分指標為示範數據（網站抓取失敗時自動退回，每日嘗試最多 5 次後使用快取）。"
            "**NAAIM 與 AAII 每週四發布新數據**，狀態標籤顯示是否為即時/快取/示範資料。"
        )
    else:
        st.header("🏛️ 美股專屬：總經情緒雙核觀測站")
        st.caption(
            "**NAAIM 大戶曝險 vs AAII 散戶情緒**：兩者每週四同日發布，量化方式不同（大戶用 % 曝險、散戶用看多/看空比例）。"
            "**通常會方向一致，但敏感度不同** — NAAIM 反應較快、AAII 慣性較強。"
        )

    bc1, bc2 = st.columns([0.6, 0.4])

    with bc1:
        st.markdown(
            f'<div class="ai-box" style="text-align:left;">'
            f'<h4 style="color:white; margin:0; margin-bottom:10px;">'
            f'📈 NAAIM 主動經理人曝險指數 '
            f'<span style="font-size:11px; padding:2px 8px; border-radius:4px; '
            f'background:{naaim_color}22; color:{naaim_color}; border:1px solid {naaim_color}; '
            f'margin-left:8px;">{naaim_label}</span></h4>',
            unsafe_allow_html=True
        )
        fig_naaim = go.Figure()
        fig_naaim.add_trace(go.Scatter(
            x=naaim_df['Date'], y=naaim_df['Exposure'],
            fill='tozeroy', mode='lines+markers',
            line=dict(color='#38bdf8', width=2), marker=dict(size=6, color='#38bdf8')
        ))
        fig_naaim.add_hline(y=100, line_dash="dash", line_color="#ef4444",
                            annotation_text="過熱區 (大戶滿倉)", annotation_position="top left",
                            annotation_font_color="#ef4444")
        fig_naaim.add_hline(y=40, line_dash="dash", line_color="#22c55e",
                            annotation_text="恐慌區 (大戶減倉)", annotation_position="bottom left",
                            annotation_font_color="#22c55e")
        fig_naaim.update_layout(height=320, template="plotly_dark",
                                margin=dict(t=10, b=10, l=10, r=10),
                                yaxis=dict(range=[20, 110]),
                                xaxis=dict(rangeslider=dict(visible=True, thickness=0.1)))
        st.plotly_chart(fig_naaim, use_container_width=True, config={'displayModeBar': False})
        if naaim_status == "real":
            latest_val = naaim_df['Exposure'].iloc[-1] if not naaim_df.empty else None
            if latest_val is not None:
                st.caption(f"💡 最新值：{latest_val:.1f}（{naaim_df['Date'].iloc[-1].strftime('%Y-%m-%d')}）。資料源：MacroMicro / NAAIM 官網。")
        st.markdown('</div>', unsafe_allow_html=True)

    with bc2:
        aaii_html = (
            f'<div class="ai-box" style="text-align:left; height:100%;">'
            f'<h4 style="color:white; margin:0; margin-bottom:15px;">'
            f'🧠 AAII 散戶情緒調查 '
            f'<span style="font-size:11px; padding:2px 8px; border-radius:4px; '
            f'background:{aaii_color}22; color:{aaii_color}; border:1px solid {aaii_color};'
            f'margin-left:8px;">{aaii_label}</span></h4>'
            f'<p style="color:#aaa; font-size:13px; margin-bottom:10px;">代表美國散戶對未來六個月的股市看法。通常作為反指標使用。</p>'
            f'<div style="margin-bottom: 20px;"><div style="display:flex; justify-content:space-between; margin-bottom:4px;"><span style="color:#4ade80; font-weight:bold;">看多 (Bullish)</span><span style="color:#4ade80; font-weight:bold;">{aaii_bull}%</span></div><div class="aaii-bar-container"><div class="aaii-bar-bull" style="width: {aaii_bull}%;"></div></div></div>'
            f'<div style="margin-bottom: 20px;"><div style="display:flex; justify-content:space-between; margin-bottom:4px;"><span style="color:#aaa; font-weight:bold;">中立 (Neutral)</span><span style="color:#aaa; font-weight:bold;">{aaii_neu}%</span></div><div class="aaii-bar-container"><div class="aaii-bar-neu" style="width: {aaii_neu}%;"></div></div></div>'
            f'<div style="margin-bottom: 20px;"><div style="display:flex; justify-content:space-between; margin-bottom:4px;"><span style="color:#ff6b6b; font-weight:bold;">看空 (Bearish)</span><span style="color:#ff6b6b; font-weight:bold;">{aaii_bear}%</span></div><div class="aaii-bar-container"><div class="aaii-bar-bear" style="width: {aaii_bear}%;"></div></div></div>'
            f'<div style="margin-top: 25px; padding:10px; background-color:#262730; border-radius:5px; border:1px solid #444; font-size:13px; color:#ddd; line-height: 1.6;">'
            f'<b>💡 核心反轉邏輯：</b><br>• 當看多 &gt; 50% 且 看空 &lt; 25%：觸發逃頂警報<br>• 當看多 &lt; 25% 或 看空 &gt; 48%：觸發抄底警報'
            f'</div></div>'
        )
        st.markdown(aaii_html, unsafe_allow_html=True)
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

# --- 0. 系統設定 ---
st.set_page_config(page_title="AI 實戰戰情室 V26", layout="wide", page_icon="🚨")

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
    "美股": ['^IXIC', 'QQQ', 'NVDA', 'TSM'],
    "台股": ['00878.TW', '2324.TW', '8215.TW', '00403A.TW', '4958.TW', '2344.TW', '2327.TW', '1815.TWO', '5347.TWO']
}

# [修復 #3 v2] 拿掉 hardcoded fallback；抓不到環境變數時回空字串並印警告
FINMIND_TOKEN = os.environ.get("FINMIND_TOKEN", "")
if not FINMIND_TOKEN:
    print("⚠️  警告：未設定 FINMIND_TOKEN 環境變數，台股籌碼功能將無法使用。")

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
    df['Vol_SMA5'] = df['Volume'].rolling(5).mean()

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


def predict_target_and_rating(df):
    price = df['Close'].iloc[-1]
    upper = df['Bollinger_Upper'].iloc[-1]
    recent_high_60 = df['High'].tail(60).max()
    t_s = upper if price >= recent_high_60 else min(upper, recent_high_60)
    return t_s, max(recent_high_60 * 1.15, t_s * 1.1), "強勢" if price > df['SMA_20'].iloc[-1] else "持有"


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
    cur_t = st.session_state.get('current_ticker', "^IXIC")
    act_l = st.session_state.get('active_list')

    for wl_name, tickers in st.session_state['watchlists'].items():
        is_exp = (wl_name == st.session_state.get('user_opened_list'))
        with st.expander(f"📁 {wl_name} ({len(tickers)}檔)", expanded=is_exp):
            for t in tickers:
                is_sel = (t == cur_t and wl_name == act_l)
                btn_t = "primary" if is_sel else "secondary"
                s_name = get_stock_name(t)
                disp_base = f"{s_name} ({t})" if s_name != t else t
                if st.button(f"{'👉 ' if is_sel else ''}{disp_base}", key=f"btn_{wl_name}_{t}", type=btn_t, use_container_width=True):
                    st.session_state['current_ticker'] = t
                    st.session_state['active_list'] = wl_name
                    st.session_state['user_opened_list'] = wl_name
                    st.rerun()

    st.markdown("---")
    st.markdown("<span style='color:gray; font-size:13px;'>排列目前代碼</span>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    c3, c4 = st.columns(2)
    lst = st.session_state['watchlists'][act_l]
    idx = lst.index(cur_t) if cur_t in lst else -1
    if c1.button("⏫ 置頂") and idx > 0:
        lst.insert(0, lst.pop(idx)); save_watchlists(st.session_state['watchlists']); st.rerun()
    if c2.button("⬆️ 上移") and idx > 0:
        lst[idx], lst[idx - 1] = lst[idx - 1], lst[idx]; save_watchlists(st.session_state['watchlists']); st.rerun()
    if c3.button("⬇️ 下移") and 0 <= idx < len(lst) - 1:
        lst[idx], lst[idx + 1] = lst[idx + 1], lst[idx]; save_watchlists(st.session_state['watchlists']); st.rerun()
    if c4.button("⏬ 置底") and 0 <= idx < len(lst) - 1:
        lst.append(lst.pop(idx)); save_watchlists(st.session_state['watchlists']); st.rerun()

    st.markdown("---")
    time_opt = st.radio("選擇週期", ["當沖 (分時)", "日線 (Daily)", "週線 (Weekly)"], index=1)

    with st.expander("✏️ 編輯清單"):
        target_list = st.selectbox("加入抽屜：", list(st.session_state['watchlists'].keys()), index=list(st.session_state['watchlists'].keys()).index(act_l))
        new_t = st.text_input("代號", placeholder="2330.TW").upper()
        if st.button("➕ 新增", use_container_width=True) and new_t:
            if new_t not in st.session_state['watchlists'][target_list]:
                st.session_state['watchlists'][target_list].append(new_t)
                st.session_state['current_ticker'] = new_t
                st.session_state['active_list'] = target_list
                st.session_state['user_opened_list'] = target_list
                save_watchlists(st.session_state['watchlists'])
                st.rerun()
        if st.button("❌ 刪除股票", use_container_width=True) and cur_t in st.session_state['watchlists'][act_l]:
            st.session_state['watchlists'][act_l].remove(cur_t)
            save_watchlists(st.session_state['watchlists'])
            if st.session_state['watchlists'][act_l]:
                st.session_state['current_ticker'] = st.session_state['watchlists'][act_l][0]
            else:
                st.session_state['current_ticker'] = "^IXIC"
            st.rerun()

# --- 5. 主體資料載入 ---
main_title_name = get_stock_name(cur_t)
disp_main_title = f"{main_title_name} ({cur_t})" if main_title_name != cur_t else cur_t
st.title(f"📈 {disp_main_title} 實戰戰情室 V26")

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
t_s, t_l, rating = predict_target_and_rating(df)

vp_60 = calculate_volume_profile(df.tail(60), bins=40)
vol_poc = vp_60.loc[vp_60['Volume'].idxmax(), 'Price'] if not vp_60.empty else close_v
iron_price, _box_start, _box_end, is_breaking = find_structural_box_bottom(df, close_v)

p_data = df.tail(120) if "日" in time_opt else df.tail(60)
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
px, py, sc_name = generate_projection_points(df, trend_txt, close_v, iron_price, is_breaking, support_line, resist_line)
sc_color_class = get_compre_color_class(sc_name)

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
    c4_html = (
        '<div class="ai-box" style="border: 1px solid #00d4ff; display: flex; flex-direction: column; justify-content: center;">'
        '<h5 style="color:white; margin:0; margin-bottom:12px;">🎯 AI 目標 & 強弱</h5>'
        f'<div style="font-size:14px; color: #ddd; margin-bottom: 8px;">短: ${t_s:.2f} | 長: ${t_l:.2f}</div>'
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
    for date in p_data[ma_gold].index:
        y_val = p_data.loc[date, 'SMA_20']
        if not pd.isna(y_val):
            if launch_pts.get(date, False):
                # ★ 黃金交叉 + 起漲條件 → 特別強化標注
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
        elif "破底翻" in status or "抄底" in status:
            fig.add_annotation(x=p_data.index[i], y=curr['Low'], text=f"{status}<br>${curr['Low']:.1f}", showarrow=True, arrowhead=1, ax=0, ay=45, row=1, col=1, bgcolor="rgba(147, 51, 234, 0.8)", font=dict(color="white", size=10, weight="bold"))
        elif "調節" in status:
            fig.add_annotation(x=p_data.index[i], y=curr['High'], text=f"🔴調節<br>${curr['High']:.1f}", showarrow=True, arrowhead=1, ax=0, ay=-40, row=1, col=1, bgcolor="rgba(185, 28, 28, 0.8)", font=dict(color="white", size=10, weight="bold"))

    macd_buy = (curr['MACD'] > curr['Signal_Line']) and (prior['MACD'] <= prior['Signal_Line'])
    macd_sell = (curr['MACD'] < curr['Signal_Line']) and (prior['MACD'] >= prior['Signal_Line'])
    if macd_buy and ((engine_type == "trend" and curr['Close'] < curr.get('SMA_60', 0)) or
                     (engine_type == "momentum" and curr['Close'] < curr.get('SMA_20', 0))):
        macd_buy = False

    if macd_buy:
        fig.add_annotation(x=p_data.index[i], y=curr['Low'], text=f"BUY<br>${curr['Close']:.1f}", showarrow=True, arrowhead=1, ax=0, ay=25, row=1, col=1, bgcolor="rgba(40, 167, 69, 0.8)", font=dict(color="white", size=9))
    if macd_sell:
        fig.add_annotation(x=p_data.index[i], y=curr['High'], text=f"SELL<br>${curr['Close']:.1f}", showarrow=True, arrowhead=1, ax=0, ay=-25, row=1, col=1, bgcolor="rgba(220, 53, 69, 0.8)", font=dict(color="white", size=9))
    if (curr['High'] >= t_s or curr['RSI'] > 75) and not (prior['High'] >= t_s or prior['RSI'] > 75):
        fig.add_annotation(
            x=p_data.index[i], y=curr['High'],
            text=f"💰達標<br>${curr['Close']:.1f}" if curr['High'] >= t_s else f"🔥過熱<br>${curr['Close']:.1f}",
            showarrow=True, arrowhead=1, ax=0, ay=-45, row=1, col=1,
            bgcolor="rgba(255, 193, 7, 0.8)" if curr['High'] >= t_s else "rgba(255, 69, 0, 0.8)",
            font=dict(color="black" if curr['High'] >= t_s else "white", size=9)
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

fig.add_trace(go.Scatter(
    x=px, y=py, mode='lines+markers',
    line=dict(color='#eab308', width=1, dash='dash'),
    marker=dict(size=8, symbol='diamond', color='#eab308'),
    name='🔮 AI 劇本推演'
), row=1, col=1)
for i in range(1, len(px)):
    fig.add_annotation(x=px[i], y=py[i], text=f"${py[i]:.2f}", showarrow=True, arrowhead=0, ay=-20, font=dict(color="#eab308", size=11), bgcolor="rgba(0,0,0,0.6)", row=1, col=1)

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
# 先取出主圖已判定的起漲點日期集合（MA 黃金交叉版）
main_launch_dates = set(p_data[launch_pts].index) if 'launch_pts' in dir() else set()

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

    # ── 給星條件 ──────────────────────────────────────────────
    # A. 正值區域金叉（非死叉）→ 直接給星（趨勢健康，不需 RSI 超賣）
    # B. 零軸突破（非死叉）→ 直接給星
    # C. 其他條件（近MA60 / MACD仍負 / RSI回暖）→ 傳統條件
    uptrend_zero_breakout = (not death_cross_active) and zero_cross
    strong_positive = (not death_cross_active) and positive_zone_cross

    if strong_positive or uptrend_zero_breakout or (rsi_recovering and (near_ma60 or deep_reversal or zero_cross)):
        star_macd_x.append(date)
        star_macd_y.append(curr['MACD'])
        already_starred.add(date)
        fig.add_annotation(
            x=date, y=curr['MACD'], text="⭐起漲確認",
            showarrow=False, yshift=20, row=2, col=1,
            font=dict(color="#facc15", size=10, weight="bold")
        )
    else:
        normal_macd_x.append(date)
        normal_macd_y.append(curr['MACD'])

# --- [v20 新增] 主圖起漲點同步到副圖：MA 金叉起漲點若尚未打星，補上星星 ---
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
        st.caption("ℹ️ 部分指標為示範數據（網站抓取失敗時自動退回，每日嘗試最多 5 次後使用快取）")
    else:
        st.header("🏛️ 美股專屬：總經情緒雙核觀測站")

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
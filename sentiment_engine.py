# =============================================================
# sentiment_engine.py — 美股情緒雙核引擎 (v2)
# NAAIM (主動經理人曝險) + AAII (散戶情緒)
#
# 三層 Fallback 設計：
#   A. 官方原始檔 (AAII XLS / NAAIM HTML)
#   B. Nasdaq Data Link 公開資料集 (需免費 API key, 可選)
#   C. 解析網頁文字 / xlsx 路徑探測
#
# 全部都掛掉時：用上次成功的快取
# 連快取都沒有時：給示範資料 (status="demo") + 提示
#
# 設計原則：
#   1. NAAIM/AAII 都是「週四美東時間」公布，所以一週只主動嘗試一次
#   2. 已抓到本週資料 → 直接讀快取，零網路請求
#   3. 限流：同一天最多嘗試 3 次，最少間隔 60 分鐘
#
# 對外函式：
#   - get_naaim_data()                 → (df, status)
#   - get_aaii_data()                  → ((b,n,br), df_hist, status)
#   - render_us_sentiment_dashboard()  → Streamlit UI 元件
# =============================================================

import os
import io
import json
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ─── 設定 ──────────────────────────────────────────────
CACHE_DIR    = ".sentiment_cache"
NAAIM_CACHE  = os.path.join(CACHE_DIR, "naaim.json")
AAII_CACHE   = os.path.join(CACHE_DIR, "aaii.json")
ATTEMPT_LOG  = os.path.join(CACHE_DIR, "attempts.json")

MAX_TRIES_PER_DAY = 3
MIN_GAP_MINUTES   = 60

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
      "AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/126.0.0.0 Safari/537.36")


# ─── 工具函式 ───────────────────────────────────────────
def _ensure_dir():
    if not os.path.exists(CACHE_DIR):
        try: os.makedirs(CACHE_DIR, exist_ok=True)
        except: pass

def _load(path, default=None):
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except: pass
    return default

def _save(path, data):
    _ensure_dir()
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, default=str, indent=2)
    except: pass

def _et_now():
    """近似美東時間 (UTC-5)，差 1 小時可接受"""
    return datetime.utcnow() - timedelta(hours=5)

def _this_weeks_thursday_et():
    now = _et_now()
    days_since_thu = (now.weekday() - 3) % 7
    return (now - timedelta(days=days_since_thu)).replace(
        hour=0, minute=0, second=0, microsecond=0)

def _is_after_thursday_release():
    """美東週四中午 12:00 後算「本週已發布」"""
    now = _et_now()
    if now.weekday() < 3: return False
    if now.weekday() == 3 and now.hour < 12: return False
    return True

def _can_try(key: str) -> bool:
    log = _load(ATTEMPT_LOG, {})
    today = _et_now().strftime("%Y-%m-%d")
    rec = log.get(today, {}).get(key, {})
    if rec.get("count", 0) >= MAX_TRIES_PER_DAY: return False
    last_ts = rec.get("last_ts", 0)
    if last_ts and (time.time() - last_ts) < MIN_GAP_MINUTES * 60: return False
    return True

def _log_try(key: str, success: bool):
    log = _load(ATTEMPT_LOG, {})
    today = _et_now().strftime("%Y-%m-%d")
    log.setdefault(today, {}).setdefault(key, {"count":0,"last_ts":0,"has_success":False})
    log[today][key]["count"] += 1
    log[today][key]["last_ts"] = time.time()
    if success: log[today][key]["has_success"] = True
    cutoff = (_et_now() - timedelta(days=7)).strftime("%Y-%m-%d")
    log = {k: v for k, v in log.items() if k >= cutoff}
    _save(ATTEMPT_LOG, log)

def _get_nasdaq_api_key():
    """從 env 或 Streamlit secrets 取得 Nasdaq Data Link API key"""
    key = os.environ.get("xMQVLs5ussHBFoRAZzt8", "").strip()
    if key: return key
    try:
        import streamlit as st
        return str(st.secrets.get("NASDAQ_DATA_LINK_API_KEY", "")).strip()
    except Exception:
        return ""


# ═════════════════════════════════════════════════════
#                  AAII  抓取邏輯
# ═════════════════════════════════════════════════════

def _fetch_aaii_official_xls():
    """A 層：AAII 官方 sentiment.xls"""
    try:
        url = "https://www.aaii.com/files/surveys/sentiment.xls"
        headers = {"User-Agent": UA, "Accept": "*/*",
                   "Referer": "https://www.aaii.com/sentimentsurvey"}
        r = requests.get(url, headers=headers, timeout=15)
        if r.status_code != 200 or len(r.content) < 1000:
            return None

        df = None
        for engine in ("openpyxl", "xlrd"):
            try:
                df = pd.read_excel(io.BytesIO(r.content), engine=engine,
                                   sheet_name=0, header=None)
                break
            except Exception:
                continue
        if df is None or df.empty: return None

        header_row = None
        for i in range(min(10, len(df))):
            row_str = " ".join(str(x).lower() for x in df.iloc[i].values)
            if "bull" in row_str and "bear" in row_str and \
               ("date" in row_str or "week" in row_str):
                header_row = i
                break
        if header_row is None: return None

        df.columns = [str(c).strip() for c in df.iloc[header_row].values]
        df = df.iloc[header_row + 1:].reset_index(drop=True)

        date_c = bull_c = neu_c = bear_c = None
        for c in df.columns:
            cl = str(c).lower()
            if date_c is None and ("date" in cl or "week" in cl or "reported" in cl): date_c = c
            elif bull_c is None and "bull" in cl: bull_c = c
            elif neu_c is None and "neutral" in cl: neu_c = c
            elif bear_c is None and "bear" in cl: bear_c = c

        if not all([date_c, bull_c, bear_c]): return None

        sub = df[[date_c, bull_c] + ([neu_c] if neu_c else []) + [bear_c]].copy()
        sub.columns = ["Date", "Bullish"] + (["Neutral"] if neu_c else []) + ["Bearish"]
        sub["Date"] = pd.to_datetime(sub["Date"], errors="coerce")
        for col in ("Bullish", "Bearish"):
            sub[col] = pd.to_numeric(sub[col], errors="coerce")
        if "Neutral" in sub.columns:
            sub["Neutral"] = pd.to_numeric(sub["Neutral"], errors="coerce")
        else:
            sub["Neutral"] = (1 - sub["Bullish"] - sub["Bearish"]).clip(lower=0)

        sub = sub.dropna(subset=["Date", "Bullish", "Bearish"]).sort_values("Date")
        if sub["Bullish"].max() <= 1.5:
            sub["Bullish"]  *= 100
            sub["Neutral"]  *= 100
            sub["Bearish"]  *= 100

        sub = sub.tail(104)
        records = [{"date": r["Date"].strftime("%Y-%m-%d"),
                    "bull": float(r["Bullish"]),
                    "neu":  float(r["Neutral"]),
                    "bear": float(r["Bearish"])} for _, r in sub.iterrows()]
        return records if len(records) >= 4 else None
    except Exception:
        return None


def _fetch_aaii_nasdaq():
    """B 層：Nasdaq Data Link 公開 dataset AAII/AAII_SENTIMENT"""
    try:
        api_key = _get_nasdaq_api_key()
        params = {}
        if api_key: params["api_key"] = api_key
        url = "https://data.nasdaq.com/api/v3/datasets/AAII/AAII_SENTIMENT/data.json"
        r = requests.get(url, params=params, timeout=12, headers={"User-Agent": UA})
        if r.status_code != 200: return None
        data = r.json().get("dataset_data", {})
        cols = [c.lower() for c in data.get("column_names", [])]
        rows = data.get("data", [])
        if not rows: return None

        date_i = next((i for i, c in enumerate(cols) if "date" in c), 0)
        bull_i = next((i for i, c in enumerate(cols) if "bull" in c), None)
        neu_i  = next((i for i, c in enumerate(cols) if "neu"  in c), None)
        bear_i = next((i for i, c in enumerate(cols) if "bear" in c), None)
        if bull_i is None or bear_i is None: return None

        out = []
        for row in rows[:104]:
            try:
                d  = str(row[date_i])[:10]
                b  = float(row[bull_i])
                br = float(row[bear_i])
                n  = float(row[neu_i]) if neu_i is not None else max(0, 1 - b - br)
                if b <= 1.5: b *= 100; n *= 100; br *= 100
                out.append({"date": d, "bull": b, "neu": n, "bear": br})
            except Exception:
                continue
        out.sort(key=lambda x: x["date"])
        return out if len(out) >= 4 else None
    except Exception:
        return None


def _fetch_aaii_webpage():
    """C 層：解析 AAII 網頁文字"""
    try:
        import re
        url = "https://www.aaii.com/sentimentsurvey"
        r = requests.get(url, headers={"User-Agent": UA}, timeout=10)
        if r.status_code != 200: return None
        html = r.text
        pats = {
            "bull": r"[Bb]ullish[^0-9]{0,40}([0-9]+\.[0-9]+)\s*%",
            "neu":  r"[Nn]eutral[^0-9]{0,40}([0-9]+\.[0-9]+)\s*%",
            "bear": r"[Bb]earish[^0-9]{0,40}([0-9]+\.[0-9]+)\s*%",
        }
        out = {}
        for k, p in pats.items():
            m = re.search(p, html)
            if m: out[k] = float(m.group(1))
        if "bull" in out and "bear" in out:
            if "neu" not in out: out["neu"] = max(0, 100 - out["bull"] - out["bear"])
            return [{"date": _et_now().strftime("%Y-%m-%d"), **out}]
        return None
    except Exception:
        return None


def get_aaii_data():
    """回傳 ((bull, neu, bear), history_df, status)"""
    cache = _load(AAII_CACHE)

    if cache and cache.get("records"):
        try:
            last_dt = datetime.strptime(cache["records"][-1]["date"], "%Y-%m-%d")
            if last_dt >= _this_weeks_thursday_et() - timedelta(days=1):
                return _aaii_pack(cache["records"], "cached")
        except Exception: pass

    if not _is_after_thursday_release() and cache:
        return _aaii_pack(cache.get("records", []), "cached")

    records = None
    if _can_try("aaii"):
        for fetcher in (_fetch_aaii_official_xls, _fetch_aaii_nasdaq, _fetch_aaii_webpage):
            records = fetcher()
            if records: break
        _log_try("aaii", success=(records is not None))

    if records:
        if cache and cache.get("records"):
            existing = {r["date"]: r for r in cache["records"]}
            for r in records: existing[r["date"]] = r
            records = sorted(existing.values(), key=lambda x: x["date"])[-104:]
        _save(AAII_CACHE, {"last_update": _et_now().strftime("%Y-%m-%d %H:%M:%S"),
                            "records": records})
        return _aaii_pack(records, "fresh")

    if cache and cache.get("records"):
        return _aaii_pack(cache["records"], "stale")

    return ((38.0, 31.5, 30.5), pd.DataFrame(), "demo")


def _aaii_pack(records, status):
    if not records: return ((0,0,0), pd.DataFrame(), "demo")
    df = pd.DataFrame(records)
    df["Date"] = pd.to_datetime(df["date"])
    df = df.sort_values("Date").reset_index(drop=True)
    last = records[-1]
    return ((float(last.get("bull",0)), float(last.get("neu",0)), float(last.get("bear",0))),
            df, status)


# ═════════════════════════════════════════════════════
#                  NAAIM  抓取邏輯
# ═════════════════════════════════════════════════════

def _fetch_naaim_official_html():
    """A 層：NAAIM 官網 HTML 表格"""
    try:
        url = "https://www.naaim.org/programs/naaim-exposure-index/"
        r = requests.get(url, headers={"User-Agent": UA}, timeout=12)
        if r.status_code != 200: return None
        try:
            tables = pd.read_html(io.StringIO(r.text))
        except Exception:
            return None
        for tbl in tables:
            cols_lower = [str(c).lower() for c in tbl.columns]
            joined = " ".join(cols_lower)
            if ("date" in joined or "week" in joined) and \
               any(k in joined for k in ("exposure", "naaim", "mean", "average")):
                date_col = next((c for c in tbl.columns
                                 if "date" in str(c).lower() or "week" in str(c).lower()), None)
                val_col  = next((c for c in tbl.columns
                                 if any(k in str(c).lower() for k in
                                        ("mean", "average", "exposure", "naaim"))), None)
                if not date_col or not val_col: continue
                sub = tbl[[date_col, val_col]].dropna().copy()
                sub.columns = ["date", "value"]
                sub["date"] = pd.to_datetime(sub["date"], errors="coerce")
                sub["value"] = pd.to_numeric(sub["value"], errors="coerce")
                sub = sub.dropna().sort_values("date").tail(104)
                records = [{"date": r["date"].strftime("%Y-%m-%d"),
                            "value": float(r["value"])} for _, r in sub.iterrows()]
                if len(records) >= 4: return records
        return None
    except Exception:
        return None


def _fetch_naaim_nasdaq():
    """B 層：Nasdaq Data Link 公開 dataset NAAIM/NAAIM"""
    try:
        api_key = _get_nasdaq_api_key()
        params = {}
        if api_key: params["api_key"] = api_key
        url = "https://data.nasdaq.com/api/v3/datasets/NAAIM/NAAIM/data.json"
        r = requests.get(url, params=params, timeout=12, headers={"User-Agent": UA})
        if r.status_code != 200: return None
        data = r.json().get("dataset_data", {})
        cols = [c.lower() for c in data.get("column_names", [])]
        rows = data.get("data", [])
        if not rows: return None

        date_i = next((i for i, c in enumerate(cols) if "date" in c), 0)
        val_i  = next((i for i, c in enumerate(cols)
                       if any(k in c for k in ("mean", "average", "exposure", "naaim"))), 1)

        out = []
        for row in rows[:104]:
            try:
                out.append({"date": str(row[date_i])[:10], "value": float(row[val_i])})
            except Exception:
                continue
        out.sort(key=lambda x: x["date"])
        return out if len(out) >= 4 else None
    except Exception:
        return None


def _fetch_naaim_xlsx_attempts():
    """C 層：探測常見 wp-content xlsx 路徑"""
    candidates = [
        "https://www.naaim.org/wp-content/uploads/2014/03/NAAIM-Exposure-Index-Data.xlsx",
        "https://www.naaim.org/wp-content/uploads/NAAIM-Exposure-Index-Data.xlsx",
        "https://naaim.org/wp-content/uploads/2014/03/NAAIM-Exposure-Index-Data.xlsx",
    ]
    for url in candidates:
        try:
            r = requests.get(url, headers={"User-Agent": UA}, timeout=10)
            if r.status_code != 200 or len(r.content) < 500: continue
            try:
                df = pd.read_excel(io.BytesIO(r.content))
            except Exception:
                continue
            date_col = val_col = None
            for c in df.columns:
                cl = str(c).lower()
                if date_col is None and ("date" in cl or "week" in cl): date_col = c
                elif val_col is None and any(k in cl for k in
                                              ("mean", "average", "exposure", "naaim")): val_col = c
            if not date_col or not val_col: continue
            sub = df[[date_col, val_col]].dropna().copy()
            sub.columns = ["date", "value"]
            sub["date"] = pd.to_datetime(sub["date"], errors="coerce")
            sub["value"] = pd.to_numeric(sub["value"], errors="coerce")
            sub = sub.dropna().sort_values("date").tail(104)
            records = [{"date": r["date"].strftime("%Y-%m-%d"),
                        "value": float(r["value"])} for _, r in sub.iterrows()]
            if len(records) >= 4: return records
        except Exception:
            continue
    return None


def get_naaim_data():
    """回傳 (df[Date,Exposure], status)"""
    cache = _load(NAAIM_CACHE)

    if cache and cache.get("records"):
        try:
            last_dt = datetime.strptime(cache["records"][-1]["date"], "%Y-%m-%d")
            if last_dt >= _this_weeks_thursday_et() - timedelta(days=1):
                return _naaim_pack(cache["records"], "cached")
        except Exception: pass

    if not _is_after_thursday_release() and cache:
        return _naaim_pack(cache.get("records", []), "cached")

    records = None
    if _can_try("naaim"):
        for fetcher in (_fetch_naaim_official_html, _fetch_naaim_nasdaq, _fetch_naaim_xlsx_attempts):
            records = fetcher()
            if records: break
        _log_try("naaim", success=(records is not None))

    if records:
        if cache and cache.get("records"):
            existing = {r["date"]: r for r in cache["records"]}
            for r in records: existing[r["date"]] = r
            records = sorted(existing.values(), key=lambda x: x["date"])[-104:]
        _save(NAAIM_CACHE, {"last_update": _et_now().strftime("%Y-%m-%d %H:%M:%S"),
                             "records": records})
        return _naaim_pack(records, "fresh")

    if cache and cache.get("records"):
        return _naaim_pack(cache["records"], "stale")

    now = _et_now()
    dates = [now - timedelta(weeks=51-i) for i in range(52)]
    rng = np.random.default_rng(seed=42)
    vals = np.clip(rng.integers(40, 100, size=52).astype(float), 20, 110).round(1)
    return pd.DataFrame({"Date": dates, "Exposure": vals}), "demo"


def _naaim_pack(records, status):
    if not records: return pd.DataFrame(), "demo"
    df = pd.DataFrame(records)
    df["Date"] = pd.to_datetime(df["date"])
    df["Exposure"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["Date","Exposure"]).sort_values("Date").reset_index(drop=True)
    return df[["Date","Exposure"]], status


# ═════════════════════════════════════════════════════
#                 訊號判讀邏輯
# ═════════════════════════════════════════════════════

def _naaim_verdict(value):
    if value is None: return "—", "#888"
    if value >= 95:   return "🔴 經理人槓桿做多 (反向警訊)", "#ff6b6b"
    if value >= 80:   return "🟠 經理人積極做多", "#ffaa00"
    if value >= 60:   return "🟡 經理人中性偏多", "#facc15"
    if value >= 40:   return "⚖️ 經理人中性", "#aaa"
    if value >= 20:   return "🔵 經理人轉趨保守", "#4a9eff"
    return "🟢 經理人大幅減碼 (反向買進機會)", "#4ade80"


def _aaii_verdict(bull, bear):
    spread = bull - bear
    if bull >= 50 and bear <= 25:  return "🔴 散戶極度樂觀 (反向看空)", "#ff6b6b"
    if bear >= 50 and bull <= 25:  return "🟢 散戶極度恐慌 (反向看多)", "#4ade80"
    if spread >= 15:               return "🟠 散戶偏樂觀", "#ffaa00"
    if spread <= -15:              return "🔵 散戶偏悲觀", "#4a9eff"
    return "⚖️ 散戶情緒中性", "#aaa"


def _status_badge(status):
    badges = {
        "fresh":  ("✅", "本週新資料", "#4ade80"),
        "cached": ("📦", "本週快取",   "#4a9eff"),
        "stale":  ("⚠️", "過期快取",   "#ffaa00"),
        "demo":   ("🧪", "示範資料",   "#888"),
    }
    return badges.get(status, badges["demo"])


# ═════════════════════════════════════════════════════
#              Streamlit 渲染元件
# ═════════════════════════════════════════════════════

def render_us_sentiment_dashboard():
    """在主程式呼叫此函式即可顯示美股情緒雙核儀表板"""
    try:
        import streamlit as st
        import plotly.graph_objects as go
    except ImportError:
        return

    df_n, st_n = get_naaim_data()
    (a_bull, a_neu, a_bear), df_a, st_a = get_aaii_data()

    naaim_val = float(df_n["Exposure"].iloc[-1]) if not df_n.empty else None
    naaim_date = df_n["Date"].iloc[-1].strftime("%m/%d") if not df_n.empty else "—"
    aaii_date = df_a["Date"].iloc[-1].strftime("%m/%d") if not df_a.empty else "—"

    n_verdict, n_color = _naaim_verdict(naaim_val)
    a_verdict, a_color = _aaii_verdict(a_bull, a_bear)
    n_icon, n_label, n_bcolor = _status_badge(st_n)
    a_icon, a_label, a_bcolor = _status_badge(st_a)

    st.markdown("""
    <style>
    .sent-card {background-color: #1a1a1c; border-radius: 8px; padding: 14px 18px;
                margin-bottom: 8px; border: 1px solid #333;}
    .sent-title {font-size: 14px; color: #aaa; margin-bottom: 6px;}
    .sent-val   {font-size: 32px; font-weight: 900; line-height: 1.0; margin: 4px 0;}
    .sent-verdict {font-size: 14px; font-weight: bold; margin-top: 6px;}
    .sent-meta  {font-size: 11px; color: #888; margin-top: 4px;}
    .sent-badge {display:inline-block; padding:2px 8px; border-radius:10px;
                 font-size: 10px; font-weight: bold; margin-left:6px;}
    .sent-bar-wrap {display:flex; height:14px; border-radius:3px; overflow:hidden;
                    margin-top:8px; background:#444;}
    .sent-bar-seg {height:100%; text-align:center; color:#000;
                   font-size:9px; font-weight:bold; line-height:14px;}
    </style>
    """, unsafe_allow_html=True)

    st.markdown("### 🇺🇸 美股情緒雙核觀測站 "
                "<span style='font-size:12px;color:#888;'>(週四美東更新)</span>",
                unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    # ─── NAAIM 卡片 ──────────────────────────
    with col1:
        naaim_disp = f"{naaim_val:.1f}" if naaim_val is not None else "—"
        st.markdown(f"""
        <div class="sent-card" style="border-left: 4px solid {n_color};">
            <div class="sent-title">📊 NAAIM 主動經理人曝險指數
                <span class="sent-badge" style="background:{n_bcolor}30;color:{n_bcolor};">
                {n_icon} {n_label}</span>
            </div>
            <div class="sent-val" style="color:{n_color};">{naaim_disp}</div>
            <div class="sent-verdict" style="color:{n_color};">{n_verdict}</div>
            <div class="sent-meta">最新: {naaim_date} | 範圍: 0 (空頭) ↔ 200 (槓桿多)</div>
        </div>
        """, unsafe_allow_html=True)

        if len(df_n) >= 4:
            tail = df_n.tail(26)
            try:
                rr = int(n_color[1:3], 16); gg = int(n_color[3:5], 16); bb = int(n_color[5:7], 16)
                fill_c = f"rgba({rr},{gg},{bb},0.15)"
            except Exception:
                fill_c = "rgba(150,150,150,0.15)"
            fig = go.Figure(go.Scatter(
                x=tail["Date"], y=tail["Exposure"],
                mode="lines+markers", line=dict(color=n_color, width=2),
                marker=dict(size=4), fill="tozeroy", fillcolor=fill_c))
            fig.add_hline(y=95, line_dash="dot", line_color="#ff6b6b", opacity=0.5,
                          annotation_text="過熱", annotation_position="top right",
                          annotation_font_size=9, annotation_font_color="#ff6b6b")
            fig.add_hline(y=30, line_dash="dot", line_color="#4ade80", opacity=0.5,
                          annotation_text="超賣", annotation_position="bottom right",
                          annotation_font_size=9, annotation_font_color="#4ade80")
            fig.update_layout(height=130, template="plotly_dark",
                              margin=dict(t=5,b=5,l=5,r=5), showlegend=False,
                              xaxis=dict(showgrid=False), yaxis=dict(showgrid=False))
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # ─── AAII 卡片 ─────────────────────────
    with col2:
        total = max(1.0, a_bull + a_neu + a_bear)
        bull_pct = a_bull / total * 100
        neu_pct  = a_neu  / total * 100
        bear_pct = a_bear / total * 100
        spread = a_bull - a_bear
        spread_color = "#4ade80" if spread >= 0 else "#ff6b6b"

        st.markdown(f"""
        <div class="sent-card" style="border-left: 4px solid {a_color};">
            <div class="sent-title">📊 AAII 散戶情緒調查
                <span class="sent-badge" style="background:{a_bcolor}30;color:{a_bcolor};">
                {a_icon} {a_label}</span>
            </div>
            <div style="display:flex; align-items:baseline; gap:10px;">
                <div class="sent-val" style="color:#4ade80; font-size:24px;">多 {a_bull:.1f}%</div>
                <div class="sent-val" style="color:#aaa; font-size:24px;">中 {a_neu:.1f}%</div>
                <div class="sent-val" style="color:#ff6b6b; font-size:24px;">空 {a_bear:.1f}%</div>
            </div>
            <div class="sent-bar-wrap">
                <div class="sent-bar-seg" style="width:{bull_pct}%; background:#4ade80;">{f'{a_bull:.0f}' if bull_pct > 12 else ''}</div>
                <div class="sent-bar-seg" style="width:{neu_pct}%; background:#888;">{f'{a_neu:.0f}' if neu_pct > 12 else ''}</div>
                <div class="sent-bar-seg" style="width:{bear_pct}%; background:#ff6b6b;">{f'{a_bear:.0f}' if bear_pct > 12 else ''}</div>
            </div>
            <div class="sent-verdict" style="color:{a_color};">{a_verdict}
                <span style="color:{spread_color}; font-size:12px; font-weight:normal; margin-left:8px;">
                (多空差: {spread:+.1f})</span>
            </div>
            <div class="sent-meta">最新: {aaii_date} | 歷史均值 多37%/中31%/空31%</div>
        </div>
        """, unsafe_allow_html=True)

        if len(df_a) >= 4:
            tail = df_a.tail(26).copy()
            tail["Spread"] = tail["bull"] - tail["bear"]
            colors = ["#4ade80" if v >= 0 else "#ff6b6b" for v in tail["Spread"]]
            fig = go.Figure(go.Bar(x=tail["Date"], y=tail["Spread"], marker_color=colors))
            fig.add_hline(y=0, line_color="#666", line_width=1)
            fig.update_layout(height=130, template="plotly_dark",
                              margin=dict(t=5,b=5,l=5,r=5), showlegend=False,
                              xaxis=dict(showgrid=False), yaxis=dict(showgrid=False))
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # 警示燈
    alerts = []
    if naaim_val is not None:
        if naaim_val >= 95: alerts.append("🚨 NAAIM 經理人槓桿做多 → 反向警訊")
        if naaim_val <= 20: alerts.append("💎 NAAIM 經理人棄守 → 反向買進機會")
        if len(df_n) >= 5:
            prev_max = df_n["Exposure"].iloc[-5:-1].max()
            if prev_max > 90 and naaim_val < 60:
                alerts.append(f"⚠️ NAAIM 大戶減倉中（從 {prev_max:.0f} → {naaim_val:.0f}）")
    if a_bull >= 50 and a_bear <= 25:
        alerts.append("🚨 AAII 散戶極度樂觀 → 反向看空")
    if a_bear >= 50 and a_bull <= 25:
        alerts.append("💎 AAII 散戶極度恐慌 → 反向看多")

    if alerts:
        st.markdown(f"""
        <div style="background:#3a1b1b; border-left:4px solid #ff6b6b;
                    padding:10px 14px; border-radius:6px; margin-top:8px;">
            <div style="color:#ff6b6b; font-weight:bold; margin-bottom:4px;">
                ⚡ 情緒極值警示
            </div>
            <div style="color:#ffaa88; font-size:13px;">{' ｜ '.join(alerts)}</div>
        </div>
        """, unsafe_allow_html=True)

    if st_n == "demo" or st_a == "demo":
        st.warning("⚠️ 目前無真實資料（可能尚未首次抓取成功）。"
                   "建議至 [data.nasdaq.com](https://data.nasdaq.com) 註冊免費 API key，"
                   "在 `.streamlit/secrets.toml` 加入："
                   "`NASDAQ_DATA_LINK_API_KEY = \"你的key\"`")

    st.markdown("<div style='margin-bottom:15px;'></div>", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════
# CLI 自我測試
# ═════════════════════════════════════════════════════
if __name__ == "__main__":
    print(f"美東時間: {_et_now()}")
    print(f"已過本週四發布: {_is_after_thursday_release()}")
    print(f"Nasdaq API key 存在: {bool(_get_nasdaq_api_key())}")
    print()
    print("[NAAIM]")
    df_n, st_n = get_naaim_data()
    print(f"  status: {st_n}, rows: {len(df_n)}")
    if not df_n.empty: print(f"  最新: {df_n.iloc[-1].to_dict()}")
    print("[AAII]")
    (b,n,br), df_a, st_a = get_aaii_data()
    print(f"  status: {st_a}, 多/中/空: {b:.1f}/{n:.1f}/{br:.1f}")
    print(f"  歷史筆數: {len(df_a)}")

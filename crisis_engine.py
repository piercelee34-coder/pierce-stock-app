"""
crisis_engine.py — 空頭距離指數引擎

提供兩個主要函式：
  get_us_crisis_index() → dict（US 危機指數 + 子指標分解）
  get_tw_crisis_index() → dict（TW 危機指數 + 子指標分解）

設計原則：
  1. 每個資料源有 3 層 fallback（即時 / 快取 / N/A）
  2. 快取自動分層：價格 12 小時，FRED/FinMind 24 小時，AAII/NAAIM 7 天
  3. 缺資料源時自動跳過，剩下指標按 total_weight 正規化
  4. 百分位校準 252 日 + Top-3 加成

對 Streamlit App：
  import crisis_engine
  result = crisis_engine.get_us_crisis_index(fred_key, nasdaq_key)
  # result = {
  #   "score": 73.4,
  #   "level": ("🟡 警戒區", "#facc15"),
  #   "components": [{"name": "VIX期限結構", "score": 65.0, "raw": 0.95}, ...],
  #   "data_status": {"VIX": True, "FRED 10Y2Y": True, ...},
  #   "history": pd.DataFrame  # 過去 90 日的指數歷史
  # }
"""

import os
import io
import json
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ──────────────────────────────────────────────────────
# 設定
# ──────────────────────────────────────────────────────
HISTORY_DAYS = 365  # 抓多少歷史用來算百分位
SHOW_HISTORY_DAYS = 90  # 介面顯示多少天歷史
PERCENTILE_WINDOW = 252  # 百分位 lookback
CACHE_DIR = ".crisis_cache"
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"

LEVELS = [
    (85, "🔴 強制清倉",   "#dc2626"),
    (75, "🟠 高度危險",   "#f97316"),
    (60, "🟡 警戒區",     "#facc15"),
    (40, "⚖️ 中性區",     "#9ca3af"),
    (20, "🟢 機會浮現",   "#84cc16"),
    (0,  "💚 極度恐慌",   "#22c55e"),
]


def level_of(score):
    if score is None or pd.isna(score):
        return ("❓ N/A", "#666")
    for thr, label, color in LEVELS:
        if score >= thr:
            return (label, color)
    return (LEVELS[-1][1], LEVELS[-1][2])


# ──────────────────────────────────────────────────────
# 快取工具
# ──────────────────────────────────────────────────────
def _cache_path(key):
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, f"{key}.json")


def cache_get(key, max_age_hours=24):
    path = _cache_path(key)
    if not os.path.exists(path):
        return None
    age = (time.time() - os.path.getmtime(path)) / 3600
    if age > max_age_hours:
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def cache_set(key, data):
    try:
        with open(_cache_path(key), "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, default=str)
    except Exception:
        pass


# ──────────────────────────────────────────────────────
# 資料抓取
# ──────────────────────────────────────────────────────
def _df_from_cache_records(records):
    df = pd.DataFrame(records)
    if not df.empty:
        df["Date"] = pd.to_datetime(df["Date"])
    return df


def fetch_yf(ticker, days=HISTORY_DAYS, cache_hours=12):
    """yfinance 抓歷史價格"""
    import yfinance as yf
    cache_key = f"yf_{ticker.replace('^', 'idx_').replace('=', '_')}"
    cached = cache_get(cache_key, cache_hours)
    if cached:
        return _df_from_cache_records(cached)

    try:
        start = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        df = yf.Ticker(ticker).history(start=start, auto_adjust=False)
        if df.empty:
            return pd.DataFrame()
        df = df.reset_index()
        df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
        df = df[["Date", "Open", "High", "Low", "Close", "Volume"]].copy()
        records = [{
            "Date": str(r["Date"].date()),
            "Open": float(r["Open"]) if pd.notna(r["Open"]) else None,
            "High": float(r["High"]) if pd.notna(r["High"]) else None,
            "Low":  float(r["Low"])  if pd.notna(r["Low"])  else None,
            "Close": float(r["Close"]) if pd.notna(r["Close"]) else None,
            "Volume": float(r["Volume"]) if pd.notna(r["Volume"]) else 0.0,
        } for _, r in df.iterrows()]
        cache_set(cache_key, records)
        return df
    except Exception:
        return pd.DataFrame()


def fetch_fred(series_id, fred_key, cache_hours=24):
    """FRED 序列"""
    cache_key = f"fred_{series_id}"
    cached = cache_get(cache_key, cache_hours)
    if cached:
        return _df_from_cache_records(cached)
    if not fred_key:
        return pd.DataFrame()
    try:
        start = (datetime.now() - timedelta(days=HISTORY_DAYS)).strftime("%Y-%m-%d")
        r = requests.get(
            "https://api.stlouisfed.org/fred/series/observations",
            params={"series_id": series_id, "api_key": fred_key,
                    "file_type": "json", "observation_start": start},
            timeout=15,
        )
        if r.status_code != 200:
            return pd.DataFrame()
        rows = []
        for o in r.json().get("observations", []):
            if o["value"] != ".":
                try:
                    rows.append({"Date": o["date"], "Value": float(o["value"])})
                except Exception:
                    pass
        if not rows:
            return pd.DataFrame()
        cache_set(cache_key, rows)
        df = pd.DataFrame(rows)
        df["Date"] = pd.to_datetime(df["Date"])
        return df
    except Exception:
        return pd.DataFrame()


def fetch_naaim(cache_hours=24*7):
    """NAAIM 歷史 — 主頁面爬 xlsx 連結 → 直接 xlsx"""
    cache_key = "naaim"
    cached = cache_get(cache_key, cache_hours)
    if cached:
        return _df_from_cache_records(cached)

    headers = {"User-Agent": UA, "Accept": "*/*",
               "Referer": "https://www.naaim.org/programs/naaim-exposure-index/"}

    # 試 1: 從主頁找 xlsx 連結
    try:
        import re
        page = requests.get("https://www.naaim.org/programs/naaim-exposure-index/",
                            headers=headers, timeout=20)
        if page.status_code == 200:
            matches = re.findall(r'href="([^"]+\.xlsx?)"', page.text, re.I)
            for m in matches[:3]:
                if "naaim" in m.lower() or "exposure" in m.lower():
                    full = m if m.startswith("http") else f"https://www.naaim.org{m}"
                    try:
                        r = requests.get(full, headers=headers, timeout=20)
                        if r.status_code == 200 and len(r.content) > 1000:
                            result = _parse_naaim_xlsx(r.content)
                            if result is not None and not result.empty:
                                records = [{"Date": str(r2["Date"].date()),
                                            "Value": float(r2["Value"])}
                                            for _, r2 in result.iterrows()]
                                cache_set(cache_key, records)
                                return result
                    except Exception:
                        continue
    except Exception:
        pass
    return pd.DataFrame()


def _parse_naaim_xlsx(content):
    try:
        df_x = pd.read_excel(io.BytesIO(content))
        date_c = val_c = None
        for c in df_x.columns:
            cl = str(c).lower()
            if date_c is None and ("date" in cl or "week" in cl):
                date_c = c
            elif val_c is None and any(k in cl for k in
                                        ("mean", "average", "exposure", "naaim")):
                val_c = c
        if not date_c or not val_c:
            return None
        df_x = df_x[[date_c, val_c]].dropna().copy()
        df_x.columns = ["Date", "Value"]
        df_x["Date"] = pd.to_datetime(df_x["Date"], errors="coerce")
        df_x["Value"] = pd.to_numeric(df_x["Value"], errors="coerce")
        df_x = df_x.dropna().sort_values("Date")
        # 只保留 HISTORY_DAYS 內的
        cutoff = pd.to_datetime(datetime.now() - timedelta(days=HISTORY_DAYS))
        df_x = df_x[df_x["Date"] >= cutoff]
        return df_x
    except Exception:
        return None


def fetch_aaii(cache_hours=24*7):
    """AAII 歷史 — GitHub 鏡像優先"""
    cache_key = "aaii"
    cached = cache_get(cache_key, cache_hours)
    if cached:
        return _df_from_cache_records(cached)

    for url in (
        "https://raw.githubusercontent.com/psinopoli/AAII-Sentiment/master/AAII_SENTIMENT_CSV.csv",
        "https://raw.githubusercontent.com/psinopoli/AAII-Sentiment/main/AAII_SENTIMENT_CSV.csv",
    ):
        try:
            r = requests.get(url, headers={"User-Agent": UA}, timeout=15)
            if r.status_code != 200 or len(r.content) < 1000:
                continue
            df_csv = pd.read_csv(io.BytesIO(r.content))
            date_c = bull_c = bear_c = None
            for c in df_csv.columns:
                cl = str(c).lower().strip()
                if date_c is None and ("date" in cl or "week" in cl or "reported" in cl):
                    date_c = c
                elif bull_c is None and "bull" in cl and "8" not in cl:
                    bull_c = c
                elif bear_c is None and "bear" in cl and "8" not in cl:
                    bear_c = c
            if not all([date_c, bull_c, bear_c]):
                continue
            sub = df_csv[[date_c, bull_c, bear_c]].copy()
            sub.columns = ["Date", "Bullish", "Bearish"]
            sub["Date"] = pd.to_datetime(sub["Date"], errors="coerce")
            for col in ("Bullish", "Bearish"):
                sub[col] = (sub[col].astype(str)
                            .str.replace("%", "", regex=False).str.strip())
                sub[col] = pd.to_numeric(sub[col], errors="coerce")
            sub = sub.dropna().sort_values("Date")
            if sub["Bullish"].max() <= 1.5:
                sub["Bullish"] *= 100
                sub["Bearish"] *= 100
            cutoff = pd.to_datetime(datetime.now() - timedelta(days=HISTORY_DAYS))
            sub = sub[sub["Date"] >= cutoff]
            if sub.empty:
                continue
            records = [{"Date": str(r2["Date"].date()),
                        "Bullish": float(r2["Bullish"]),
                        "Bearish": float(r2["Bearish"])}
                       for _, r2 in sub.iterrows()]
            cache_set(cache_key, records)
            df = pd.DataFrame(records)
            df["Date"] = pd.to_datetime(df["Date"])
            return df
        except Exception:
            continue
    return pd.DataFrame()


def fetch_finmind_total_margin(token, cache_hours=24):
    """整體融資餘額（FinMind: TaiwanStockTotalMarginPurchaseShortSale 的 MarginPurchase 類別）"""
    cache_key = "finmind_total_margin"
    cached = cache_get(cache_key, cache_hours)
    if cached:
        return _df_from_cache_records(cached)
    if not token:
        return pd.DataFrame()
    try:
        start = (datetime.now() - timedelta(days=HISTORY_DAYS + 60)).strftime("%Y-%m-%d")
        end = datetime.now().strftime("%Y-%m-%d")
        r = requests.get(
            "https://api.finmindtrade.com/api/v4/data",
            params={"dataset": "TaiwanStockTotalMarginPurchaseShortSale",
                    "start_date": start, "end_date": end, "token": token},
            timeout=30,
        )
        if r.status_code != 200:
            return pd.DataFrame()
        j = r.json()
        if j.get("msg") != "success":
            return pd.DataFrame()
        rows = []
        for row in j.get("data", []):
            if row.get("name") == "MarginPurchase":
                rows.append({"Date": row["date"], "Value": float(row["TodayBalance"])})
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.drop_duplicates("Date").sort_values("Date").reset_index(drop=True)
        cache_set(cache_key, [{"Date": str(r2["Date"].date()),
                                "Value": r2["Value"]} for _, r2 in df.iterrows()])
        return df
    except Exception:
        return pd.DataFrame()


def fetch_finmind_per(token, cache_hours=24):
    """台積電 PER（FinMind: TaiwanStockPER, data_id=2330）"""
    cache_key = "finmind_per_2330"
    cached = cache_get(cache_key, cache_hours)
    if cached:
        return _df_from_cache_records(cached)
    if not token:
        return pd.DataFrame()
    try:
        start = (datetime.now() - timedelta(days=HISTORY_DAYS + 60)).strftime("%Y-%m-%d")
        end = datetime.now().strftime("%Y-%m-%d")
        r = requests.get(
            "https://api.finmindtrade.com/api/v4/data",
            params={"dataset": "TaiwanStockPER", "data_id": "2330",
                    "start_date": start, "end_date": end, "token": token},
            timeout=30,
        )
        if r.status_code != 200:
            return pd.DataFrame()
        j = r.json()
        if j.get("msg") != "success":
            return pd.DataFrame()
        rows = []
        for row in j.get("data", []):
            per = row.get("PER")
            if per is not None and per > 0:
                rows.append({"Date": row["date"], "Value": float(per)})
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.drop_duplicates("Date").sort_values("Date").reset_index(drop=True)
        cache_set(cache_key, [{"Date": str(r2["Date"].date()),
                                "Value": r2["Value"]} for _, r2 in df.iterrows()])
        return df
    except Exception:
        return pd.DataFrame()


# ──────────────────────────────────────────────────────
# 計分函式
# ──────────────────────────────────────────────────────
def _linear(x, x_low, x_high, score_low, score_high):
    if x_low == x_high:
        return (score_low + score_high) / 2
    t = max(0.0, min(1.0, (x - x_low) / (x_high - x_low)))
    return score_low + t * (score_high - score_low)


def score_vix_term(vix, vix3m):
    if vix is None or vix3m is None or vix3m <= 0:
        return None
    return round(_linear(vix / vix3m, 0.80, 1.20, 0, 100), 1)


def score_yield_curve(value, min_12m):
    if value is None:
        return None
    if   value < -0.5: cur = 90
    elif value <  0.0: cur = 75
    elif value <  0.5: cur = 55
    elif value <  1.0: cur = 40
    else:              cur = 25
    bonus = 0
    if min_12m is not None and min_12m < 0:
        bonus = min(15, abs(min_12m) * 15)
    return round(min(100, cur + bonus), 1)


def score_hy_spread(spread):
    if spread is None:
        return None
    return round(_linear(spread, 3.0, 8.0, 0, 100), 1)


def score_naaim(value, max_5w):
    if value is None:
        return None
    if   value >= 95: base = 75
    elif value >= 85: base = 55
    elif value >= 70: base = 40
    elif value >= 50: base = 35
    elif value >= 30: base = 45
    else:             base = 30
    bonus = 0
    if max_5w is not None and max_5w > 90 and value < 60:
        bonus = min(25, (max_5w - value) / 2)
    return round(min(100, base + bonus), 1)


def score_aaii(bull, bear):
    if bull is None or bear is None:
        return None
    return round(_linear(bull - bear, -30, 30, 15, 80), 1)


def score_adr_premium(premium_pct):
    if premium_pct is None:
        return None
    return round(_linear(-premium_pct, -5, 5, 25, 95), 1)


def score_twd_momentum(pct_20d):
    if pct_20d is None:
        return None
    return round(_linear(pct_20d, -2.0, 2.0, 20, 85), 1)


def score_market_breadth(close, sma200, dist_pct):
    if close is None:
        return None
    score = 50
    if sma200 is not None and close < sma200:
        score += 20
    if dist_pct is not None:
        if   dist_pct < -20: score += 30
        elif dist_pct < -10: score += 15
        elif dist_pct < -5:  score += 5
    return round(min(100, score), 1)


def score_percentile(value, history_series):
    if value is None or history_series is None:
        return None
    hist = history_series.dropna()
    if len(hist) < 30:
        return None
    return round((hist <= value).sum() / len(hist) * 100, 1)


# ──────────────────────────────────────────────────────
# 後處理：百分位校準 + Top-K 加成
# ──────────────────────────────────────────────────────
def _compress(pct):
    if pd.isna(pct):
        return np.nan
    if pct <= 50:
        return pct * 0.6
    elif pct <= 80:
        return 30 + (pct - 50) * (40/30)
    else:
        return 70 + (pct - 80) * (30/20)


def apply_percentile_and_topk(df, sub_cols, weights,
                                window=PERCENTILE_WINDOW,
                                top_k=3, top_k_weight=0.40):
    """對歷史 DataFrame 套用百分位校準 + Top-K 加成"""
    if df is None or df.empty:
        return df
    df = df.copy().reset_index(drop=True)

    calibrated = []
    for col in sub_cols:
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        raw_pct = s.rolling(window=window, min_periods=30).rank(pct=True) * 100
        df[col + "_pct"] = raw_pct.apply(_compress).round(1)
        calibrated.append((col, col + "_pct"))

    composites = []
    for _, row in df.iterrows():
        scores_w = []
        for (orig, pct_col), w in zip(calibrated, weights):
            v = row.get(pct_col)
            if pd.notna(v):
                scores_w.append((v, w))
        if not scores_w:
            composites.append(np.nan)
            continue
        total_w = sum(w for _, w in scores_w)
        weighted_avg = sum(s * w for s, w in scores_w) / total_w
        sorted_scores = sorted([s for s, _ in scores_w], reverse=True)
        k = min(top_k, len(sorted_scores))
        top_k_avg = sum(sorted_scores[:k]) / k
        composite = (1 - top_k_weight) * weighted_avg + top_k_weight * top_k_avg
        composites.append(round(composite, 1))
    df["Composite"] = composites

    # 把校準後的子分數放回原欄
    for orig, pct_col in calibrated:
        df[orig + "_raw"] = df[orig]
        df[orig] = df[pct_col].round(1)
        df = df.drop(columns=[pct_col])
    return df


# ──────────────────────────────────────────────────────
# 工具：asof
# ──────────────────────────────────────────────────────
def asof(df, date, col="Value"):
    if df is None or df.empty:
        return None
    sub = df[df["Date"] <= pd.to_datetime(date)]
    if sub.empty:
        return None
    v = sub.iloc[-1][col]
    return float(v) if pd.notna(v) else None


def rolling_min_before(df, date, days=365, col="Value"):
    if df is None or df.empty:
        return None
    d = pd.to_datetime(date)
    sub = df[(df["Date"] <= d) & (df["Date"] >= d - timedelta(days=days))]
    if sub.empty:
        return None
    return float(sub[col].min())


def rolling_max_before(df, date, weeks=5, col="Value"):
    if df is None or df.empty:
        return None
    d = pd.to_datetime(date)
    sub = df[(df["Date"] <= d) & (df["Date"] >= d - timedelta(weeks=weeks))]
    if sub.empty:
        return None
    return float(sub[col].max())


# ──────────────────────────────────────────────────────
# 主要對外函式
# ──────────────────────────────────────────────────────
def get_crisis_indices(fred_key="", finmind_token=""):
    """
    一次抓所有資料 + 計算 US/TW 兩個指數歷史
    回傳 dict:
      {
        "us": {
          "score": 73.4, "level": ("🟡 警戒區", "#facc15"),
          "components": [{"name": "VIX期限結構", "score": 65.0, "weight": 0.20}, ...],
          "history": pd.DataFrame[Date, Composite],
        },
        "tw": {...},
        "data_status": {"SPX": True, "VIX": True, ...},
      }
    """
    status = {}

    # ─── 抓資料 ───
    spx = fetch_yf("^GSPC");        status["SPX"]            = not spx.empty
    twii = fetch_yf("^TWII");       status["台灣加權"]       = not twii.empty
    vix = fetch_yf("^VIX");         status["VIX"]            = not vix.empty
    vix3m = fetch_yf("^VIX3M");     status["VIX3M"]          = not vix3m.empty
    tsm = fetch_yf("TSM");          status["TSM"]            = not tsm.empty
    t2330 = fetch_yf("2330.TW");    status["2330.TW"]        = not t2330.empty
    twd = fetch_yf("TWD=X");        status["TWD匯率"]        = not twd.empty
    t10y2y = fetch_fred("T10Y2Y", fred_key);          status["殖利率倒掛"] = not t10y2y.empty
    hy = fetch_fred("BAMLH0A0HYM2", fred_key);        status["HY信用利差"] = not hy.empty
    naaim = fetch_naaim();          status["NAAIM"]          = not naaim.empty
    aaii = fetch_aaii();            status["AAII"]           = not aaii.empty
    tw_margin = fetch_finmind_total_margin(finmind_token); status["台股融資"] = not tw_margin.empty
    tw_per = fetch_finmind_per(finmind_token);             status["2330 PER"] = not tw_per.empty

    # ─── 預處理 ───
    if not spx.empty:
        spx = spx.copy()
        spx["SMA200"] = spx["Close"].rolling(200).mean()
        spx["High52w"] = spx["High"].rolling(252).max()
        spx["DistFromHigh"] = (spx["Close"] / spx["High52w"] - 1) * 100

    if not twii.empty:
        twii = twii.copy()
        twii["SMA200"] = twii["Close"].rolling(200).mean()
        twii["High52w"] = twii["High"].rolling(252).max()
        twii["DistFromHigh"] = (twii["Close"] / twii["High52w"] - 1) * 100

    aaii_s = aaii[["Date", "Bullish", "Bearish"]].copy() if not aaii.empty else None
    tw_margin_s = (tw_margin.set_index("Date")["Value"].sort_index()
                    if not tw_margin.empty else None)
    tw_per_s = (tw_per.set_index("Date")["Value"].sort_index()
                 if not tw_per.empty else None)

    if spx.empty:
        # 沒 SPX 無法回測，回空
        return {
            "us": _empty_result("SPX 資料缺失"),
            "tw": _empty_result("SPX 資料缺失"),
            "data_status": status,
        }

    # ─── 逐日計算 ───
    us_rows = []
    tw_rows = []
    for _, row in spx.iterrows():
        date = row["Date"]

        # US 子分數
        vix_close = asof(vix, date, "Close")
        vix3m_close = asof(vix3m, date, "Close")
        s_vix = score_vix_term(vix_close, vix3m_close)

        t10_v = asof(t10y2y, date, "Value")
        t10_min = rolling_min_before(t10y2y, date, days=365, col="Value")
        s_yc = score_yield_curve(t10_v, t10_min)

        s_hy = score_hy_spread(asof(hy, date, "Value"))

        naaim_v = asof(naaim, date, "Value")
        naaim_max5w = rolling_max_before(naaim, date, weeks=5, col="Value")
        s_naaim = score_naaim(naaim_v, naaim_max5w)

        if aaii_s is not None:
            bull = asof(aaii_s, date, "Bullish")
            bear = asof(aaii_s, date, "Bearish")
        else:
            bull = bear = None
        s_aaii = score_aaii(bull, bear)

        s_breadth_us = score_market_breadth(
            row["Close"],
            row["SMA200"] if pd.notna(row.get("SMA200", np.nan)) else None,
            row["DistFromHigh"] if pd.notna(row.get("DistFromHigh", np.nan)) else None,
        )

        us_components = [
            ("VIX期限結構", s_vix, 0.20),
            ("殖利率倒掛",  s_yc,  0.20),
            ("HY信用利差",  s_hy,  0.20),
            ("NAAIM經理人", s_naaim, 0.15),
            ("AAII散戶",    s_aaii,  0.10),
            ("市場結構",    s_breadth_us, 0.15),
        ]
        tw_us = sum(w for _, s, w in us_components if s is not None)
        us_score_d = (sum(s * w for _, s, w in us_components if s is not None) / tw_us
                       if tw_us > 0 else None)

        us_rows.append({
            "Date": date, "Close": row["Close"],
            "VIX期限結構": s_vix, "殖利率倒掛": s_yc, "HY信用利差": s_hy,
            "NAAIM經理人": s_naaim, "AAII散戶": s_aaii, "市場結構": s_breadth_us,
            "Composite": round(us_score_d, 1) if us_score_d is not None else None,
        })

        # TW 子分數
        tsm_close = asof(tsm, date - timedelta(days=1), "Close")
        t2330_close = asof(t2330, date, "Close") if not t2330.empty else None
        twd_close = asof(twd, date, "Close")
        adr_premium = None
        if tsm_close and t2330_close and twd_close and t2330_close > 0:
            tsm_implied = tsm_close * twd_close / 5
            adr_premium = (tsm_implied / t2330_close - 1) * 100
        s_adr = score_adr_premium(adr_premium)

        twd_20d_ago = asof(twd, date - timedelta(days=28), "Close")
        twd_pct = None
        if twd_close and twd_20d_ago and twd_20d_ago > 0:
            twd_pct = (twd_close / twd_20d_ago - 1) * 100
        s_twd = score_twd_momentum(twd_pct)

        s_breadth_tw = None
        if not twii.empty:
            sub = twii[twii["Date"] <= date]
            if not sub.empty:
                last = sub.iloc[-1]
                s_breadth_tw = score_market_breadth(
                    last["Close"],
                    last["SMA200"] if pd.notna(last.get("SMA200", np.nan)) else None,
                    last["DistFromHigh"] if pd.notna(last.get("DistFromHigh", np.nan)) else None,
                )

        s_tw_margin = None
        if tw_margin_s is not None:
            past = tw_margin_s[tw_margin_s.index <= date]
            if len(past) >= 30:
                cur = past.iloc[-1]
                hist = past.iloc[-PERCENTILE_WINDOW:] if len(past) >= PERCENTILE_WINDOW else past
                s_tw_margin = score_percentile(cur, hist)

        s_tw_per = None
        if tw_per_s is not None:
            past = tw_per_s[tw_per_s.index <= date]
            if len(past) >= 30:
                cur = past.iloc[-1]
                hist = past.iloc[-PERCENTILE_WINDOW:] if len(past) >= PERCENTILE_WINDOW else past
                s_tw_per = score_percentile(cur, hist)

        tw_components = [
            ("TSM ADR溢價",  s_adr, 0.30),
            ("台幣20日變化", s_twd, 0.25),
            ("台股市場結構", s_breadth_tw, 0.15),
            ("美股共振",     us_score_d, 0.20),
            ("散戶融資餘額", s_tw_margin, 0.05),
            ("台積電PER",    s_tw_per, 0.05),
        ]
        tw_tw = sum(w for _, s, w in tw_components if s is not None)
        tw_score_d = (sum(s * w for _, s, w in tw_components if s is not None) / tw_tw
                       if tw_tw > 0 else None)

        tw_close_val = None
        if not twii.empty:
            sub = twii[twii["Date"] <= date]
            if not sub.empty:
                tw_close_val = float(sub.iloc[-1]["Close"])

        tw_rows.append({
            "Date": date, "Close": tw_close_val,
            "TSM ADR溢價": s_adr, "台幣20日變化": s_twd,
            "台股市場結構": s_breadth_tw, "美股共振": us_score_d,
            "散戶融資餘額": s_tw_margin, "台積電PER": s_tw_per,
            "Composite": round(tw_score_d, 1) if tw_score_d is not None else None,
        })

    us_df_raw = pd.DataFrame(us_rows)
    tw_df_raw = pd.DataFrame(tw_rows)

    # 後處理：百分位校準 + Top-K
    us_df = apply_percentile_and_topk(
        us_df_raw,
        sub_cols=["VIX期限結構", "殖利率倒掛", "HY信用利差",
                  "NAAIM經理人", "AAII散戶", "市場結構"],
        weights=[0.20, 0.20, 0.20, 0.15, 0.10, 0.15],
    )
    tw_df = apply_percentile_and_topk(
        tw_df_raw,
        sub_cols=["TSM ADR溢價", "台幣20日變化", "台股市場結構",
                  "美股共振", "散戶融資餘額", "台積電PER"],
        weights=[0.30, 0.25, 0.15, 0.20, 0.05, 0.05],
    )

    return {
        "us": _build_result(us_df, [
            ("VIX期限結構", 0.20), ("殖利率倒掛", 0.20), ("HY信用利差", 0.20),
            ("NAAIM經理人", 0.15), ("AAII散戶", 0.10), ("市場結構", 0.15),
        ]),
        "tw": _build_result(tw_df, [
            ("TSM ADR溢價", 0.30), ("台幣20日變化", 0.25), ("台股市場結構", 0.15),
            ("美股共振", 0.20), ("散戶融資餘額", 0.05), ("台積電PER", 0.05),
        ]),
        "data_status": status,
    }


def _empty_result(reason):
    return {
        "score": None, "level": ("❓ N/A", "#666"),
        "components": [], "history": pd.DataFrame(),
        "reason": reason,
    }


def _build_result(df, components_def):
    if df is None or df.empty:
        return _empty_result("無資料")
    valid = df.dropna(subset=["Composite"])
    if valid.empty:
        return _empty_result("樣本不足")
    latest = valid.iloc[-1]
    score = float(latest["Composite"])
    level = level_of(score)

    components = []
    for name, weight in components_def:
        if name in df.columns:
            val = latest.get(name)
            raw = latest.get(name + "_raw")
            components.append({
                "name": name,
                "score": float(val) if pd.notna(val) else None,
                "raw": float(raw) if pd.notna(raw) else None,
                "weight": weight,
            })

    # 取最近 SHOW_HISTORY_DAYS 天歷史
    cutoff = pd.to_datetime(datetime.now() - timedelta(days=SHOW_HISTORY_DAYS))
    history = df[df["Date"] >= cutoff][["Date", "Composite", "Close"]].copy()

    return {
        "score": score,
        "level": level,
        "components": components,
        "latest_date": str(latest["Date"].date()) if pd.notna(latest["Date"]) else None,
        "history": history,
    }

#!/usr/bin/env python3
"""
backtest_crisis_index.py — 空頭距離指數歷史回測

用法：
  1. 把下方「設定區」的兩組 key 貼好
  2. python backtest_crisis_index.py
  3. 打開 backtest_report.html 看結果

執行時間：
  首次跑：5~10 分鐘（抓全部歷史資料）
  之後跑：30 秒以內（吃快取）
  快取存在  .backtest_cache/ 目錄，要重抓刪掉即可

依賴：
  pip install requests pandas numpy yfinance plotly openpyxl xlrd
"""

import os
import io
import sys
import json
import time
import warnings
import traceback
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════
# ⚠️ 設定區 — 貼你的 key
# ═══════════════════════════════════════════════════════
FRED_KEY      = "43c0811da53d9355e6a308a2a18589c6"
FINMIND_TOKEN = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoia2Fpc2xpcGtub3QzNCIsImVtYWlsIjoicGllcmNlLmxlZS4zNEBnbWFpbC5jb20iLCJ0b2tlbl92ZXJzaW9uIjowfQ.9T3Y6zCiy5OlAlTyewu-p3THfao8Haz7jz1lj0o6J7Q"   # 沒有就留空（會跳過部分台股指標）

# ═══════════════════════════════════════════════════════
# 回測參數
# ═══════════════════════════════════════════════════════
START_DATE = "2019-06-01"   # 多抓半年方便算 rolling
END_DATE   = datetime.now().strftime("%Y-%m-%d")
CACHE_DIR  = ".backtest_cache"
OUTPUT_HTML = "backtest_report.html"

# 五大歷史事件（崩盤起點日 + 之後最低點跌幅）
EVENTS = [
    {"name": "COVID 全球崩盤",     "date": "2020-02-20", "drop_pct": -34.0, "market": "both"},
    {"name": "升息熊市起點",       "date": "2022-01-04", "drop_pct": -25.4, "market": "us"},
    {"name": "通膨爆表二次探底",   "date": "2022-09-13", "drop_pct": -16.0, "market": "us"},
    {"name": "日圓套利平倉",       "date": "2024-08-05", "drop_pct": -8.5,  "market": "both"},
    {"name": "美中關稅戰",         "date": "2025-04-02", "drop_pct": -12.0, "market": "both"},
]

# 等級門檻（依你的要求：85+ 強制清倉）
LEVELS = [
    (85, "🔴 強制清倉",   "#dc2626"),
    (75, "🟠 高度危險",   "#f97316"),
    (60, "🟡 警戒區",     "#facc15"),
    (40, "⚖️ 中性區",     "#9ca3af"),
    (20, "🟢 機會浮現",   "#84cc16"),
    (0,  "💚 極度恐慌",   "#22c55e"),
]


def level_of(score):
    for thr, label, color in LEVELS:
        if score >= thr:
            return label, color
    return LEVELS[-1][1], LEVELS[-1][2]


# 偵測設定區是否未填
if "貼這裡" in FRED_KEY:
    FRED_KEY = ""

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"


# ═══════════════════════════════════════════════════════
# 快取工具
# ═══════════════════════════════════════════════════════
def _ensure_cache():
    os.makedirs(CACHE_DIR, exist_ok=True)


def cache_get(key, max_age_hours=24*7):
    """讀快取，超過 max_age_hours 算過期"""
    path = os.path.join(CACHE_DIR, f"{key}.json")
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
    _ensure_cache()
    path = os.path.join(CACHE_DIR, f"{key}.json")
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, default=str)
    except Exception as e:
        print(f"   ⚠️ 快取寫入失敗 {key}: {e}")


def log(msg, level="info"):
    icon = {"info": "  ", "ok": "✅", "warn": "⚠️ ", "err": "❌", "step": "🔹"}[level]
    print(f"{icon} {msg}")


# ═══════════════════════════════════════════════════════
# 資料抓取層
# ═══════════════════════════════════════════════════════
def fetch_fred(series_id, cache_key=None):
    """FRED 抓單一序列，回傳 DataFrame[Date, Value]"""
    cache_key = cache_key or f"fred_{series_id}"
    cached = cache_get(cache_key, max_age_hours=24)
    if cached:
        df = pd.DataFrame(cached)
        df["Date"] = pd.to_datetime(df["Date"])
        return df

    if not FRED_KEY:
        return pd.DataFrame()

    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id, "api_key": FRED_KEY,
        "file_type": "json", "observation_start": START_DATE,
        "observation_end": END_DATE,
    }
    try:
        r = requests.get(url, params=params, timeout=30)
        if r.status_code != 200:
            log(f"FRED {series_id} HTTP {r.status_code}", "err")
            return pd.DataFrame()
        obs = r.json().get("observations", [])
        rows = []
        for o in obs:
            try:
                v = float(o["value"]) if o["value"] != "." else None
                if v is not None:
                    rows.append({"Date": o["date"], "Value": v})
            except Exception:
                continue
        df = pd.DataFrame(rows)
        if not df.empty:
            df["Date"] = pd.to_datetime(df["Date"])
            cache_set(cache_key, [{"Date": str(r["Date"].date()), "Value": r["Value"]}
                                  for _, r in df.iterrows()])
        return df
    except Exception as e:
        log(f"FRED {series_id} 失敗: {e}", "err")
        return pd.DataFrame()


def fetch_yf(ticker, cache_key=None):
    """yfinance 抓歷史價格"""
    import yfinance as yf
    cache_key = cache_key or f"yf_{ticker.replace('^','idx_').replace('=','_')}"
    cached = cache_get(cache_key, max_age_hours=12)
    if cached:
        df = pd.DataFrame(cached)
        df["Date"] = pd.to_datetime(df["Date"])
        return df

    try:
        t = yf.Ticker(ticker)
        df = t.history(start=START_DATE, end=END_DATE, auto_adjust=False)
        if df.empty:
            log(f"yfinance {ticker} 無資料", "warn")
            return pd.DataFrame()
        df = df.reset_index()
        df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
        df = df[["Date", "Open", "High", "Low", "Close", "Volume"]].copy()
        # 存快取
        cache_set(cache_key, [{
            "Date": str(r["Date"].date()),
            "Open": float(r["Open"]) if pd.notna(r["Open"]) else None,
            "High": float(r["High"]) if pd.notna(r["High"]) else None,
            "Low":  float(r["Low"])  if pd.notna(r["Low"])  else None,
            "Close":float(r["Close"])if pd.notna(r["Close"])else None,
            "Volume":float(r["Volume"]) if pd.notna(r["Volume"]) else 0.0,
        } for _, r in df.iterrows()])
        return df
    except Exception as e:
        log(f"yfinance {ticker} 失敗: {e}", "err")
        return pd.DataFrame()


def fetch_naaim_history():
    """從 NAAIM 官方 XLSX / HTML table 抓全歷史"""
    cached = cache_get("naaim_hist", max_age_hours=24*3)
    if cached:
        df = pd.DataFrame(cached)
        df["Date"] = pd.to_datetime(df["Date"])
        return df

    headers = {
        "User-Agent": UA,
        "Accept": "*/*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.naaim.org/programs/naaim-exposure-index/",
    }

    # 嘗試 1: 從主頁面爬出真正的 XLSX 連結
    try:
        page = requests.get("https://www.naaim.org/programs/naaim-exposure-index/",
                            headers=headers, timeout=20)
        if page.status_code == 200:
            import re
            matches = re.findall(r'href="([^"]+\.xlsx?)"', page.text, re.IGNORECASE)
            if matches:
                log(f"  從主頁面找到 {len(matches)} 個 xlsx 連結", "info")
                # 排除非 NAAIM 的（保險）
                for m in matches[:3]:
                    if "naaim" in m.lower() or "exposure" in m.lower():
                        full_url = m if m.startswith("http") else f"https://www.naaim.org{m}"
                        try:
                            r = requests.get(full_url, headers=headers, timeout=20)
                            if r.status_code == 200 and len(r.content) > 1000:
                                df_x = pd.read_excel(io.BytesIO(r.content))
                                result = _parse_naaim_xlsx(df_x)
                                if result is not None and not result.empty:
                                    return result
                        except Exception as e:
                            log(f"  下載 {full_url[:80]} 失敗: {e}", "info")
                            continue
    except Exception as e:
        log(f"  NAAIM 主頁面爬取失敗: {e}", "info")

    # 嘗試 2: 直接試已知的固定路徑
    urls = [
        "https://www.naaim.org/wp-content/uploads/2014/03/NAAIM-Exposure-Index-Data.xlsx",
        "https://naaim.org/wp-content/uploads/2014/03/NAAIM-Exposure-Index-Data.xlsx",
        "https://www.naaim.org/wp-content/uploads/2014/03/NAAIM-Exposure-Index.xlsx",
        "https://www.naaim.org/wp-content/uploads/NAAIM-Exposure-Index-Data.xlsx",
    ]
    for url in urls:
        try:
            r = requests.get(url, headers=headers, timeout=20)
            if r.status_code != 200:
                continue
            try:
                df_x = pd.read_excel(io.BytesIO(r.content))
            except Exception:
                continue
            result = _parse_naaim_xlsx(df_x)
            if result is not None and not result.empty:
                return result
        except Exception:
            continue

    log("NAAIM 官方下載失敗（嘗試了主頁爬取與固定路徑）", "err")
    return pd.DataFrame()


def _parse_naaim_xlsx(df_x):
    """解析 NAAIM xlsx 內容"""
    try:
        date_col = val_col = None
        for c in df_x.columns:
            cl = str(c).lower()
            if date_col is None and ("date" in cl or "week" in cl):
                date_col = c
            elif val_col is None and any(k in cl for k in
                                          ("mean", "average", "exposure", "naaim")):
                val_col = c
        if not date_col or not val_col:
            return None
        df_x = df_x[[date_col, val_col]].dropna().copy()
        df_x.columns = ["Date", "Value"]
        df_x["Date"] = pd.to_datetime(df_x["Date"], errors="coerce")
        df_x["Value"] = pd.to_numeric(df_x["Value"], errors="coerce")
        df_x = df_x.dropna().sort_values("Date")
        df_x = df_x[df_x["Date"] >= START_DATE]
        if df_x.empty:
            return None
        cache_set("naaim_hist", [{"Date": str(r["Date"].date()),
                                   "Value": float(r["Value"])}
                                  for _, r in df_x.iterrows()])
        return df_x
    except Exception:
        return None


def fetch_aaii_history():
    """從 GitHub 鏡像 / AAII 官方 XLS / macromicro 備援抓全歷史"""
    cached = cache_get("aaii_hist", max_age_hours=24*3)
    if cached:
        df = pd.DataFrame(cached)
        df["Date"] = pd.to_datetime(df["Date"])
        return df

    # 嘗試 0: GitHub 鏡像 CSV（最穩定，社群維護的 AAII 1987-now）
    github_urls = [
        "https://raw.githubusercontent.com/psinopoli/AAII-Sentiment/master/AAII_SENTIMENT_CSV.csv",
        "https://raw.githubusercontent.com/psinopoli/AAII-Sentiment/main/AAII_SENTIMENT_CSV.csv",
    ]
    for url in github_urls:
        try:
            r = requests.get(url, headers={"User-Agent": UA}, timeout=20)
            if r.status_code == 200 and len(r.content) > 1000:
                df_csv = pd.read_csv(io.BytesIO(r.content))
                # 通常欄位：Date / Bullish / Neutral / Bearish (有可能多/少幾欄)
                date_c = bull_c = neu_c = bear_c = None
                for c in df_csv.columns:
                    cl = str(c).lower().strip()
                    if date_c is None and ("date" in cl or "week" in cl or "reported" in cl):
                        date_c = c
                    elif bull_c is None and "bull" in cl and "8" not in cl:
                        bull_c = c
                    elif neu_c is None and "neutral" in cl and "8" not in cl:
                        neu_c = c
                    elif bear_c is None and "bear" in cl and "8" not in cl:
                        bear_c = c
                if not all([date_c, bull_c, bear_c]):
                    log(f"  AAII GitHub 欄位辨識失敗 cols={list(df_csv.columns)[:6]}", "info")
                    continue
                sub = df_csv[[date_c, bull_c] + ([neu_c] if neu_c else []) + [bear_c]].copy()
                cols = ["Date", "Bullish"] + (["Neutral"] if neu_c else []) + ["Bearish"]
                sub.columns = cols
                sub["Date"] = pd.to_datetime(sub["Date"], errors="coerce")
                for col in ("Bullish", "Bearish"):
                    sub[col] = (sub[col].astype(str)
                                .str.replace("%", "", regex=False).str.strip())
                    sub[col] = pd.to_numeric(sub[col], errors="coerce")
                if "Neutral" in sub.columns:
                    sub["Neutral"] = (sub["Neutral"].astype(str)
                                       .str.replace("%", "", regex=False).str.strip())
                    sub["Neutral"] = pd.to_numeric(sub["Neutral"], errors="coerce")
                else:
                    sub["Neutral"] = 0
                sub = sub.dropna(subset=["Date", "Bullish", "Bearish"]).sort_values("Date")
                if not sub.empty and sub["Bullish"].max() <= 1.5:
                    sub["Bullish"] *= 100
                    sub["Neutral"] *= 100
                    sub["Bearish"] *= 100
                sub = sub[sub["Date"] >= START_DATE]
                if sub.empty:
                    log(f"  AAII GitHub: 過濾日期後無資料", "info")
                    continue
                records = [{"Date": str(r["Date"].date()),
                            "Bullish": float(r["Bullish"]),
                            "Neutral": float(r["Neutral"]),
                            "Bearish": float(r["Bearish"])} for _, r in sub.iterrows()]
                cache_set("aaii_hist", records)
                log(f"  AAII GitHub 鏡像 抓到 {len(records)} 筆", "ok")
                return pd.DataFrame(records).assign(Date=lambda d: pd.to_datetime(d["Date"]))
        except Exception as e:
            log(f"  AAII GitHub 嘗試失敗: {str(e)[:80]}", "info")
            continue

    # 嘗試 1: AAII 官方 XLS（多個 URL + 完整 headers）
    aaii_urls = [
        "https://www.aaii.com/files/surveys/sentiment.xls",
        "https://www.aaii.com/files/surveys/sentiment.xlsx",
    ]
    aaii_headers = {
        "User-Agent": UA,
        "Accept": "application/vnd.ms-excel,application/octet-stream,*/*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.aaii.com/sentimentsurvey",
        "sec-ch-ua": '"Chromium";v="120", "Not?A_Brand";v="24"',
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": '"Windows"',
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "same-origin",
    }
    for url in aaii_urls:
        try:
            r = requests.get(url, headers=aaii_headers, timeout=20,
                             allow_redirects=True)
            if r.status_code == 200 and len(r.content) > 1000:
                result = _parse_aaii_xls(r.content)
                if result is not None and not result.empty:
                    log(f"  AAII 從官方 {url[-30:]} 成功", "ok")
                    return result
            else:
                log(f"  AAII {url[-30:]} HTTP {r.status_code}", "info")
        except Exception as e:
            log(f"  AAII {url[-30:]} 例外: {str(e)[:60]}", "info")

    # 嘗試 2: macromicro 備援
    try:
        mm_url = "https://www.macromicro.me/charts/data/20828"
        mm_headers = {
            "User-Agent": UA,
            "Accept": "application/json, text/plain, */*",
            "Referer": "https://www.macromicro.me/charts/20828/us-aaii-sentimentsurvey",
            "X-Requested-With": "XMLHttpRequest",
        }
        r = requests.get(mm_url, headers=mm_headers, timeout=20)
        if r.status_code == 200:
            j = r.json()
            # macromicro 的回傳結構：{"data": {"series_id": [[ts, value], ...]}}
            series = j.get("data") or j.get("series") or {}
            if isinstance(series, dict) and series:
                # 找 bullish / bearish 兩條序列
                bull_data = bear_data = None
                for k, v in series.items():
                    kl = k.lower()
                    if "bull" in kl and bull_data is None:
                        bull_data = v
                    elif "bear" in kl and bear_data is None:
                        bear_data = v
                if bull_data and bear_data:
                    # 對齊兩條序列
                    bd = {row[0]: row[1] for row in bull_data}
                    rd = {row[0]: row[1] for row in bear_data}
                    common = sorted(set(bd) & set(rd))
                    records = []
                    for ts in common:
                        try:
                            dt = pd.to_datetime(ts, unit="ms" if ts > 1e12 else "s")
                            if dt < pd.to_datetime(START_DATE):
                                continue
                            records.append({
                                "Date": str(dt.date()),
                                "Bullish": float(bd[ts]),
                                "Neutral": 0.0,
                                "Bearish": float(rd[ts]),
                            })
                        except Exception:
                            continue
                    if records:
                        log(f"  AAII 從 macromicro 抓到 {len(records)} 筆", "ok")
                        cache_set("aaii_hist", records)
                        df = pd.DataFrame(records)
                        df["Date"] = pd.to_datetime(df["Date"])
                        return df
    except Exception as e:
        log(f"  AAII macromicro 例外: {str(e)[:60]}", "info")

    log("AAII 全部來源失敗（已嘗試官方 XLS + macromicro）", "err")
    return pd.DataFrame()


def _parse_aaii_xls(content):
    """解析 AAII xls/xlsx 內容"""
    try:
        df_x = None
        for engine in ("openpyxl", "xlrd"):
            try:
                df_x = pd.read_excel(io.BytesIO(content), engine=engine,
                                     sheet_name=0, header=None)
                break
            except Exception:
                continue
        if df_x is None or df_x.empty:
            return None

        # 找表頭列
        header_row = None
        for i in range(min(15, len(df_x))):
            row_str = " ".join(str(x).lower() for x in df_x.iloc[i].values)
            if "bull" in row_str and "bear" in row_str:
                header_row = i
                break
        if header_row is None:
            return None
        df_x.columns = [str(c).strip() for c in df_x.iloc[header_row].values]
        df_x = df_x.iloc[header_row + 1:].reset_index(drop=True)

        date_c = bull_c = neu_c = bear_c = None
        for c in df_x.columns:
            cl = str(c).lower()
            if date_c is None and any(k in cl for k in ("date", "week", "reported")):
                date_c = c
            elif bull_c is None and "bull" in cl:
                bull_c = c
            elif neu_c is None and "neutral" in cl:
                neu_c = c
            elif bear_c is None and "bear" in cl:
                bear_c = c
        if not all([date_c, bull_c, bear_c]):
            return None

        sub = df_x[[date_c, bull_c] + ([neu_c] if neu_c else []) + [bear_c]].copy()
        cols = ["Date", "Bullish"] + (["Neutral"] if neu_c else []) + ["Bearish"]
        sub.columns = cols
        sub["Date"] = pd.to_datetime(sub["Date"], errors="coerce")
        for col in ("Bullish", "Bearish"):
            sub[col] = pd.to_numeric(sub[col], errors="coerce")
        if "Neutral" in sub.columns:
            sub["Neutral"] = pd.to_numeric(sub["Neutral"], errors="coerce")
        else:
            sub["Neutral"] = 0
        sub = sub.dropna(subset=["Date", "Bullish", "Bearish"]).sort_values("Date")
        if sub["Bullish"].max() <= 1.5:
            sub["Bullish"] *= 100
            sub["Neutral"] *= 100
            sub["Bearish"] *= 100
        sub = sub[sub["Date"] >= START_DATE]
        if sub.empty:
            return None
        records = [{"Date": str(r["Date"].date()),
                    "Bullish": float(r["Bullish"]),
                    "Neutral": float(r["Neutral"]),
                    "Bearish": float(r["Bearish"])} for _, r in sub.iterrows()]
        cache_set("aaii_hist", records)
        return pd.DataFrame(records).assign(Date=lambda d: pd.to_datetime(d["Date"]))
    except Exception:
        return None


def fetch_finmind_total_margin():
    """整體市場融資餘額（FinMind: TaiwanStockTotalMarginPurchaseShortSale）
    回傳 DataFrame[Date, Value]，Value = 融資今日餘額（張）"""
    cache_key = "finmind_total_margin"
    cached = cache_get(cache_key, max_age_hours=24)
    if cached:
        df = pd.DataFrame(cached)
        df["Date"] = pd.to_datetime(df["Date"])
        return df

    if not FINMIND_TOKEN:
        return pd.DataFrame()

    url = "https://api.finmindtrade.com/api/v4/data"
    # FinMind 單次抓取量可能受限，分年抓
    all_rows = []
    start_year = pd.to_datetime(START_DATE).year
    end_year = pd.to_datetime(END_DATE).year
    for year in range(start_year, end_year + 1):
        params = {
            "dataset": "TaiwanStockTotalMarginPurchaseShortSale",
            "start_date": f"{year}-01-01",
            "end_date": f"{year}-12-31",
            "token": FINMIND_TOKEN,
        }
        try:
            r = requests.get(url, params=params, timeout=30)
            if r.status_code != 200:
                continue
            j = r.json()
            if j.get("msg") != "success":
                continue
            data = j.get("data") or []
            # 整體 dataset 的 name 是「類別」：MarginPurchase=融資、
            # ShortSale=融券、MarginPurchaseMoney=融資金額
            # 我們要的是「融資餘額（張）」= MarginPurchase 的 TodayBalance
            for row in data:
                if row.get("name") == "MarginPurchase":
                    all_rows.append({
                        "Date": row["date"],
                        "Value": float(row["TodayBalance"]),
                    })
        except Exception as e:
            log(f"  FinMind 融資 {year} 失敗: {str(e)[:60]}", "info")
            continue
        time.sleep(0.5)  # 避免被限速

    if not all_rows:
        log("FinMind 融資餘額無資料（token 是否正確？）", "err")
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.drop_duplicates(subset="Date").sort_values("Date").reset_index(drop=True)
    cache_set(cache_key, [{"Date": str(r["Date"].date()),
                            "Value": r["Value"]} for _, r in df.iterrows()])
    return df


def fetch_finmind_per():
    """0050 ETF 的 PER 作為大盤估值代理（FinMind: TaiwanStockPER）"""
    cache_key = "finmind_per_0050"
    cached = cache_get(cache_key, max_age_hours=24)
    if cached:
        df = pd.DataFrame(cached)
        df["Date"] = pd.to_datetime(df["Date"])
        return df

    if not FINMIND_TOKEN:
        return pd.DataFrame()

    url = "https://api.finmindtrade.com/api/v4/data"
    all_rows = []
    start_year = pd.to_datetime(START_DATE).year
    end_year = pd.to_datetime(END_DATE).year

    # 用 2330 台積電的 PER 當大盤代理（權重 30%+，估值波動足夠反映大盤）
    for year in range(start_year, end_year + 1):
        params = {
            "dataset": "TaiwanStockPER",
            "data_id": "2330",
            "start_date": f"{year}-01-01",
            "end_date": f"{year}-12-31",
            "token": FINMIND_TOKEN,
        }
        try:
            r = requests.get(url, params=params, timeout=30)
            if r.status_code != 200:
                continue
            j = r.json()
            if j.get("msg") != "success":
                continue
            data = j.get("data") or []
            for row in data:
                per = row.get("PER")
                if per is not None and per > 0:
                    all_rows.append({
                        "Date": row["date"],
                        "Value": float(per),
                    })
        except Exception as e:
            log(f"  FinMind PER {year} 失敗: {str(e)[:60]}", "info")
            continue
        time.sleep(0.5)

    if not all_rows:
        log("FinMind PER 無資料", "err")
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.drop_duplicates(subset="Date").sort_values("Date").reset_index(drop=True)
    cache_set(cache_key, [{"Date": str(r["Date"].date()),
                            "Value": r["Value"]} for _, r in df.iterrows()])
    return df


# ═══════════════════════════════════════════════════════
# 子指標計分函式 (每個都回傳 0-100 的危機分數，越高=越危險)
# ═══════════════════════════════════════════════════════
def _linear(x, x_low, x_high, score_low, score_high):
    """線性插值，會被 clip 在 score_low~score_high 之間"""
    if x_low == x_high:
        return (score_low + score_high) / 2
    t = (x - x_low) / (x_high - x_low)
    t = max(0.0, min(1.0, t))
    return score_low + t * (score_high - score_low)


def score_vix_term(vix_close, vix3m_close):
    """VIX/VIX3M 比值：>1=倒掛=恐慌"""
    if vix_close is None or vix3m_close is None or vix3m_close <= 0:
        return None
    ratio = vix_close / vix3m_close
    # 0.80=平靜 → 0分；1.00=平 → 50分；1.20=極端倒掛 → 100分
    return round(_linear(ratio, 0.80, 1.20, 0, 100), 1)


def score_yield_curve(t10y2y_value, recent_min_12m):
    """殖利率倒掛打分；同時看當前值與近 12 月最低"""
    if t10y2y_value is None:
        return None
    # 當前值評分
    if   t10y2y_value < -0.5: cur = 90
    elif t10y2y_value <  0.0: cur = 75
    elif t10y2y_value <  0.5: cur = 55
    elif t10y2y_value <  1.0: cur = 40
    else:                     cur = 25
    # 加成：如果近 12 月有倒掛過，額外加分（餘悸效應）
    bonus = 0
    if recent_min_12m is not None and recent_min_12m < 0:
        bonus = min(15, abs(recent_min_12m) * 15)
    return round(min(100, cur + bonus), 1)


def score_hy_spread(spread_value):
    """High Yield 信用利差，越高代表信用市場越緊張"""
    if spread_value is None:
        return None
    # 3%=平靜→0；5%=警戒→60；8%=危機→100
    return round(_linear(spread_value, 3.0, 8.0, 0, 100), 1)


def score_naaim(naaim_value, recent_max_5w):
    """NAAIM 經理人曝險：高=過熱，急跌=大戶減倉"""
    if naaim_value is None:
        return None
    if   naaim_value >= 95: base = 75
    elif naaim_value >= 85: base = 55
    elif naaim_value >= 70: base = 40
    elif naaim_value >= 50: base = 35
    elif naaim_value >= 30: base = 45    # 低位也是警訊
    else:                   base = 30   # 太低反而是反向買進
    # 大戶減倉加成：近 5 週最高 > 90 但現在 < 60
    bonus = 0
    if recent_max_5w is not None and recent_max_5w > 90 and naaim_value < 60:
        bonus = min(25, (recent_max_5w - naaim_value) / 2)
    return round(min(100, base + bonus), 1)


def score_aaii(bull, bear):
    """AAII 多空差：極端樂觀=反向看空"""
    if bull is None or bear is None:
        return None
    spread = bull - bear
    # spread > 30 極樂觀 → 80
    # spread = 10  → 55
    # spread = -10 → 40
    # spread < -30 極悲觀 → 15
    return round(_linear(spread, -30, 30, 15, 80), 1)


def score_adr_premium(premium_pct):
    """TSM ADR 溢價：負值=外資搶逃"""
    if premium_pct is None:
        return None
    # -5%=外資搶逃→95；0%=持平→55；+5%=外資看好→25
    return round(_linear(-premium_pct, -5, 5, 25, 95), 1)


def score_twd_momentum(twd_pct_chg_20d):
    """台幣 20 日匯率變化：升值快=外資匯入；貶值快=匯出"""
    if twd_pct_chg_20d is None:
        return None
    # +2% (大幅貶值=資金流出) → 85
    # 0%  → 50
    # -2% (大幅升值=資金流入) → 20
    return round(_linear(twd_pct_chg_20d, -2.0, 2.0, 20, 85), 1)


def score_market_breadth(close_now, sma200, dist_from_high_pct):
    """市場結構：價位距高點 + 站上 200MA 與否"""
    if close_now is None:
        return None
    score = 50
    if sma200 is not None and close_now < sma200:
        score += 20
    if dist_from_high_pct is not None:
        if   dist_from_high_pct < -20: score += 30
        elif dist_from_high_pct < -10: score += 15
        elif dist_from_high_pct < -5:  score += 5
    return round(min(100, score), 1)


def score_tw_margin(margin_value, margin_history_series):
    """台股融資餘額 252 日百分位：餘額在 252 日高位=散戶槓桿過熱
    
    注意：這個函數回傳的是「原始分數」，會在 apply_percentile_and_topk
    再做一次 252 日校準。所以這裡只做粗略線性映射。
    """
    if margin_value is None or margin_history_series is None:
        return None
    # 用本身相對 252 日位置算原始分數
    hist = margin_history_series.dropna()
    if len(hist) < 30:
        return None
    rank = (hist <= margin_value).sum() / len(hist)
    # 直接把百分位轉成 0-100（後處理會再校準）
    return round(rank * 100, 1)


def score_tw_per(per_value, per_history_series):
    """台股 PER 252 日百分位：PE 高位=估值頂訊號"""
    if per_value is None or per_history_series is None:
        return None
    hist = per_history_series.dropna()
    if len(hist) < 30:
        return None
    rank = (hist <= per_value).sum() / len(hist)
    return round(rank * 100, 1)


# ═══════════════════════════════════════════════════════
# 取某日的最近值（用 asof 邏輯）
# ═══════════════════════════════════════════════════════
def asof(df, date, col="Value"):
    """回傳該日期之前或當日的最後一筆值"""
    if df is None or df.empty:
        return None
    d = pd.to_datetime(date)
    sub = df[df["Date"] <= d]
    if sub.empty:
        return None
    try:
        v = sub.iloc[-1][col]
        return float(v) if pd.notna(v) else None
    except Exception:
        return None


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


# ═══════════════════════════════════════════════════════
# 主流程：對每個交易日計算危機指數
# ═══════════════════════════════════════════════════════
def compute_daily_indices(data):
    """
    data 是個 dict，內含已抓好的所有歷史 DataFrame
    回傳：us_df, tw_df，欄位 = Date + 各子分數 + composite + level
    """
    # 用 SPX 的交易日當骨幹
    spx = data["spx"]
    if spx is None or spx.empty:
        log("SPX 資料缺，無法回測", "err")
        return pd.DataFrame(), pd.DataFrame()

    # 預先準備 SPX 的 200MA 與 52w 高
    spx = spx.copy()
    spx["SMA200"] = spx["Close"].rolling(200).mean()
    spx["High52w"] = spx["High"].rolling(252).max()
    spx["DistFromHigh"] = (spx["Close"] / spx["High52w"] - 1) * 100

    twii = data["twii"]
    if twii is not None and not twii.empty:
        twii = twii.copy()
        twii["SMA200"] = twii["Close"].rolling(200).mean()
        twii["High52w"] = twii["High"].rolling(252).max()
        twii["DistFromHigh"] = (twii["Close"] / twii["High52w"] - 1) * 100

    vix = data["vix"]
    vix3m = data["vix3m"]
    t10y2y = data["t10y2y"]
    hy = data["hy_spread"]
    naaim = data["naaim"]
    aaii = data["aaii"]
    tsm = data["tsm"]
    t2330 = data["t2330"]
    twd = data["twd"]
    tw_margin = data.get("tw_margin")
    tw_per = data.get("tw_per")

    # 預備 AAII spread 欄位（asof 用）
    if aaii is not None and not aaii.empty:
        aaii_s = aaii[["Date", "Bullish", "Bearish"]].copy()
    else:
        aaii_s = None

    # 預備融資/PER 的 series（用 Date 為 index 方便 rolling）
    tw_margin_s = None
    if tw_margin is not None and not tw_margin.empty:
        tw_margin_s = tw_margin.set_index("Date")["Value"].sort_index()

    tw_per_s = None
    if tw_per is not None and not tw_per.empty:
        tw_per_s = tw_per.set_index("Date")["Value"].sort_index()

    us_rows = []
    tw_rows = []

    log(f"開始逐日計算（共 {len(spx)} 個交易日）...", "step")

    for i, row in spx.iterrows():
        date = row["Date"]
        # ─── 美股子分數 ──────────────────────
        vix_close = float(row["Close"]) if False else (
            None if vix is None or vix.empty else asof(vix, date, "Close")
        )
        vix3m_close = None if vix3m is None or vix3m.empty else asof(vix3m, date, "Close")
        s_vix = score_vix_term(vix_close, vix3m_close)

        t10_v = asof(t10y2y, date, "Value")
        t10_min12m = rolling_min_before(t10y2y, date, days=365, col="Value")
        s_yc = score_yield_curve(t10_v, t10_min12m)

        hy_v = asof(hy, date, "Value")
        s_hy = score_hy_spread(hy_v)

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

        # 加權合成
        us_components = [
            ("VIX期限結構", s_vix, 0.20),
            ("殖利率倒掛",  s_yc,  0.20),
            ("HY信用利差",  s_hy,  0.20),
            ("NAAIM經理人", s_naaim, 0.15),
            ("AAII散戶",    s_aaii,  0.10),
            ("市場結構",    s_breadth_us, 0.15),
        ]
        total_w = sum(w for _, s, w in us_components if s is not None)
        us_score = None
        if total_w > 0:
            us_score = sum(s * w for _, s, w in us_components if s is not None) / total_w
            us_score = round(us_score, 1)

        us_rows.append({
            "Date": date, "Close": row["Close"],
            "VIX期限結構": s_vix, "殖利率倒掛": s_yc, "HY信用利差": s_hy,
            "NAAIM經理人": s_naaim, "AAII散戶": s_aaii, "市場結構": s_breadth_us,
            "Composite": us_score,
        })

        # ─── 台股子分數 ──────────────────────
        # TSM ADR 溢價（用 yesterday TSM vs today 2330）
        tsm_close = asof(tsm, date - timedelta(days=1), "Close")
        t2330_close = asof(t2330, date, "Close") if t2330 is not None else None
        twd_close = asof(twd, date, "Close")
        adr_premium = None
        if tsm_close and t2330_close and twd_close and t2330_close > 0:
            tsm_implied = tsm_close * twd_close / 5
            adr_premium = (tsm_implied / t2330_close - 1) * 100
        s_adr = score_adr_premium(adr_premium)

        # 台幣 20 日變化
        twd_20d_ago = asof(twd, date - timedelta(days=28), "Close")
        twd_pct = None
        if twd_close and twd_20d_ago and twd_20d_ago > 0:
            twd_pct = (twd_close / twd_20d_ago - 1) * 100
        s_twd = score_twd_momentum(twd_pct)

        # 台股市場結構
        s_breadth_tw = None
        if twii is not None and not twii.empty:
            tw_row = twii[twii["Date"] <= date]
            if not tw_row.empty:
                last = tw_row.iloc[-1]
                s_breadth_tw = score_market_breadth(
                    last["Close"],
                    last["SMA200"] if pd.notna(last.get("SMA200", np.nan)) else None,
                    last["DistFromHigh"] if pd.notna(last.get("DistFromHigh", np.nan)) else None,
                )

        # ★ 新增 ★ 台股融資餘額 252 日百分位（散戶槓桿）
        s_tw_margin = None
        if tw_margin_s is not None:
            # 找該日的融資餘額
            past = tw_margin_s[tw_margin_s.index <= date]
            if len(past) >= 30:
                cur = past.iloc[-1]
                hist = past.iloc[-252:] if len(past) >= 252 else past
                s_tw_margin = score_tw_margin(cur, hist)

        # ★ 新增 ★ 台積電 PER 252 日百分位（估值頂訊號）
        s_tw_per = None
        if tw_per_s is not None:
            past = tw_per_s[tw_per_s.index <= date]
            if len(past) >= 30:
                cur = past.iloc[-1]
                hist = past.iloc[-252:] if len(past) >= 252 else past
                s_tw_per = score_tw_per(cur, hist)

        # 加上新指標後重新分配權重（方案 A：弱訊號降權至 5%）：
        #   ADR 30%, 台幣 25%, 結構 15%, 美股共振 20%,
        #   融資 5%, PER 5%
        tw_components = [
            ("TSM ADR溢價",  s_adr, 0.30),
            ("台幣20日變化", s_twd, 0.25),
            ("台股市場結構", s_breadth_tw, 0.15),
            ("美股共振",     us_score, 0.20),
            ("散戶融資餘額", s_tw_margin, 0.05),
            ("台積電PER",    s_tw_per, 0.05),
        ]
        total_w_tw = sum(w for _, s, w in tw_components if s is not None)
        tw_score = None
        if total_w_tw > 0:
            tw_score = sum(s * w for _, s, w in tw_components if s is not None) / total_w_tw
            tw_score = round(tw_score, 1)

        tw_close_val = None
        if twii is not None and not twii.empty:
            sub = twii[twii["Date"] <= date]
            if not sub.empty:
                tw_close_val = float(sub.iloc[-1]["Close"])

        tw_rows.append({
            "Date": date, "Close": tw_close_val,
            "TSM ADR溢價": s_adr, "台幣20日變化": s_twd,
            "台股市場結構": s_breadth_tw, "美股共振": us_score,
            "散戶融資餘額": s_tw_margin, "台積電PER": s_tw_per,
            "Composite": tw_score,
        })

    us_df_raw = pd.DataFrame(us_rows)
    tw_df_raw = pd.DataFrame(tw_rows)

    # ─── 後處理：方向 A (百分位校準) + 方向 B (Top-K 加成) ───
    log("套用百分位校準 + 多重共振加成...", "step")
    us_df = apply_percentile_and_topk(
        us_df_raw,
        sub_cols=["VIX期限結構", "殖利率倒掛", "HY信用利差",
                  "NAAIM經理人", "AAII散戶", "市場結構"],
        weights=[0.20, 0.20, 0.20, 0.15, 0.10, 0.15],
        window=252, top_k=3, top_k_weight=0.40,
    )
    tw_df = apply_percentile_and_topk(
        tw_df_raw,
        sub_cols=["TSM ADR溢價", "台幣20日變化", "台股市場結構",
                  "美股共振", "散戶融資餘額", "台積電PER"],
        weights=[0.30, 0.25, 0.15, 0.20, 0.05, 0.05],
        window=252, top_k=3, top_k_weight=0.40,
    )

    return us_df, tw_df


# ═══════════════════════════════════════════════════════
# 百分位校準 + Top-K 加成（方向 A + B）
# ═══════════════════════════════════════════════════════
def apply_percentile_and_topk(df, sub_cols, weights, window=252,
                                top_k=3, top_k_weight=0.40):
    """
    對每個子指標：
      1. 用 rolling 252 日百分位重新校準到 0-100
      2. composite = (1-top_k_weight)*加權平均 + top_k_weight*Top-K 平均

    Args:
      df: 含子指標欄位 + Close + Date 的 DataFrame
      sub_cols: 要校準的子指標欄位名稱（依序）
      weights: 對應權重（同長度）
      window: rolling 百分位視窗（252 個交易日 = 1 年）
      top_k: 取前幾名加成
      top_k_weight: 加權平均 vs Top-K 平均的混合比例
    Returns:
      新 DataFrame（保留所有原欄位 + 多了 *_pct 校準欄）
    """
    if df is None or df.empty:
        return df
    df = df.copy().reset_index(drop=True)

    # 1. 每個子指標做 rolling 百分位 + 非線性壓縮
    #    為什麼壓縮：純百分位的中位數=50，平常日 composite 會卡在 50 附近
    #    解決方案：把 0~50 百分位壓到 0~30 分（壓低平常日）
    #              50~80 百分位映射到 30~70 分（過渡區）
    #              80~100 百分位映射到 70~100 分（保留警戒區敏感度）
    def _compress(pct):
        if pd.isna(pct):
            return np.nan
        if pct <= 50:
            return pct * 0.6              # 0~50 → 0~30
        elif pct <= 80:
            return 30 + (pct - 50) * (40/30)  # 50~80 → 30~70
        else:
            return 70 + (pct - 80) * (30/20)  # 80~100 → 70~100

    calibrated_cols = []
    for col in sub_cols:
        if col not in df.columns:
            continue
        pct_col = col + "_pct"
        # 使用 pandas 的 rolling().rank(pct=True) — 效率高、無迴圈
        s = pd.to_numeric(df[col], errors="coerce")
        raw_pct = (s.rolling(window=window, min_periods=30)
                    .rank(pct=True) * 100)
        df[pct_col] = raw_pct.apply(_compress).round(1)
        calibrated_cols.append((col, pct_col))

    # 2. 重新計算 Composite：加權平均 + Top-K 加成
    composites = []
    for idx, row in df.iterrows():
        scores_with_w = []
        for (orig, pct_col), w in zip(calibrated_cols, weights):
            v = row.get(pct_col)
            if pd.notna(v):
                scores_with_w.append((v, w))
        if not scores_with_w:
            composites.append(np.nan)
            continue

        # 加權平均
        total_w = sum(w for _, w in scores_with_w)
        weighted_avg = sum(s * w for s, w in scores_with_w) / total_w

        # Top-K 平均
        sorted_scores = sorted([s for s, _ in scores_with_w], reverse=True)
        k = min(top_k, len(sorted_scores))
        top_k_avg = sum(sorted_scores[:k]) / k

        # 混合
        composite = (1 - top_k_weight) * weighted_avg + top_k_weight * top_k_avg
        composites.append(round(composite, 1))

    df["Composite"] = composites

    # 把校準後的子分數覆蓋原欄（HTML 報告才會用校準後的數字顯示）
    for orig, pct_col in calibrated_cols:
        # 保留原始分數做 *_raw，校準後的給子指標欄
        df[orig + "_raw"] = df[orig]
        df[orig] = df[pct_col].round(1)
        df = df.drop(columns=[pct_col])

    return df


# ═══════════════════════════════════════════════════════
# 事件分析
# ═══════════════════════════════════════════════════════
def analyze_events(us_df, tw_df):
    """對每個歷史事件，找出指數最高點與大盤實際崩盤起點的時間差"""
    results = []
    for ev in EVENTS:
        crash_date = pd.to_datetime(ev["date"])
        window_start = crash_date - timedelta(days=45)
        window_end = crash_date + timedelta(days=10)

        rec = {"event": ev["name"], "crash_date": ev["date"],
               "drop_pct": ev["drop_pct"], "market": ev["market"]}

        for market_name, df in [("US", us_df), ("TW", tw_df)]:
            if df is None or df.empty:
                continue
            sub = df[(df["Date"] >= window_start) & (df["Date"] <= window_end)].copy()
            if sub.empty:
                continue
            pre = sub[sub["Date"] <= crash_date]
            if pre.empty or pre["Composite"].dropna().empty:
                continue
            peak_idx = pre["Composite"].idxmax()
            peak_row = pre.loc[peak_idx]
            peak_score = peak_row["Composite"]
            peak_date = peak_row["Date"]
            lead_days = (crash_date - peak_date).days

            # 找最大子分數貢獻
            sub_cols = [c for c in df.columns
                        if c not in ("Date", "Close", "Composite")]
            sub_scores = {c: peak_row.get(c) for c in sub_cols}
            sub_scores = {k: v for k, v in sub_scores.items() if pd.notna(v)}
            top_contrib = sorted(sub_scores.items(), key=lambda x: -x[1])[:3]

            rec[f"{market_name}_peak"] = round(peak_score, 1)
            rec[f"{market_name}_peak_date"] = str(peak_date.date())
            rec[f"{market_name}_lead_days"] = lead_days
            rec[f"{market_name}_top_contrib"] = top_contrib
            rec[f"{market_name}_signal"] = (
                "✅ 領先" if lead_days >= 3 and peak_score >= 60 else
                "⚠️ 部分" if peak_score >= 50 else
                "❌ 失準"
            )

        results.append(rec)
    return results


# ═══════════════════════════════════════════════════════
# 假警報分析（指數 >75 但 30 天內沒有崩跌）
# ═══════════════════════════════════════════════════════
def find_false_alarms(df, drop_threshold_pct=-5.0):
    """找出 composite >75 但之後 30 天 SPX/TWII 沒跌 5% 以上的點"""
    if df is None or df.empty or "Close" not in df.columns:
        return []
    df = df.copy().reset_index(drop=True)
    df["FwdMin30d"] = df["Close"].iloc[::-1].rolling(30, min_periods=1).min().iloc[::-1]
    df["FwdDrop"] = (df["FwdMin30d"] / df["Close"] - 1) * 100

    alerts = df[df["Composite"] >= 75].copy()
    if alerts.empty:
        return []
    # 過濾連續日，只保留每段警報的第一天
    alerts["GroupID"] = (alerts["Date"].diff().dt.days > 7).cumsum()
    grouped = alerts.groupby("GroupID").first().reset_index(drop=True)
    false = grouped[grouped["FwdDrop"] > drop_threshold_pct]
    return false[["Date", "Composite", "FwdDrop"]].to_dict("records")


# ═══════════════════════════════════════════════════════
# HTML 報告生成
# ═══════════════════════════════════════════════════════
def make_main_chart(us_df, tw_df, events):
    """主時間序列圖：US + TW composite + 事件標記"""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05,
        row_heights=[0.5, 0.5],
        subplot_titles=("🇺🇸 美股空頭距離指數", "🇹🇼 台股空頭距離指數"),
    )

    # 加上等級背景帶
    for thr, label, color in LEVELS:
        for r in (1, 2):
            fig.add_hrect(y0=thr, y1=thr + 100 if thr == 85 else
                          (LEVELS[max(0, LEVELS.index((thr, label, color)) - 1)][0]
                           if LEVELS.index((thr, label, color)) > 0 else 100),
                          fillcolor=color, opacity=0.06, line_width=0,
                          row=r, col=1)

    # US 線
    if us_df is not None and not us_df.empty:
        fig.add_trace(go.Scatter(
            x=us_df["Date"], y=us_df["Composite"],
            mode="lines", name="US Composite",
            line=dict(color="#38bdf8", width=2),
            hovertemplate="%{x|%Y-%m-%d}<br>分數: %{y:.1f}<extra></extra>",
        ), row=1, col=1)

    # TW 線
    if tw_df is not None and not tw_df.empty:
        fig.add_trace(go.Scatter(
            x=tw_df["Date"], y=tw_df["Composite"],
            mode="lines", name="TW Composite",
            line=dict(color="#facc15", width=2),
            hovertemplate="%{x|%Y-%m-%d}<br>分數: %{y:.1f}<extra></extra>",
        ), row=2, col=1)

    # 事件垂直線
    for ev in events:
        for r in (1, 2):
            fig.add_vline(x=pd.to_datetime(ev["date"]).timestamp() * 1000,
                          line=dict(color="#ff6b6b", width=1, dash="dash"),
                          row=r, col=1)
        fig.add_annotation(x=pd.to_datetime(ev["date"]), y=95,
                            text=f"⚡ {ev['name']}", showarrow=False,
                            font=dict(size=10, color="#ff6b6b"),
                            yref="y1", textangle=-90)

    # 門檻線
    for thr in (85, 75, 60):
        for r in (1, 2):
            fig.add_hline(y=thr, line=dict(color="#666", width=1, dash="dot"),
                          row=r, col=1)

    fig.update_layout(
        template="plotly_dark", height=700,
        margin=dict(t=50, b=40, l=50, r=20),
        showlegend=False, hovermode="x unified",
        plot_bgcolor="#1a1a1c", paper_bgcolor="#1a1a1c",
    )
    fig.update_yaxes(range=[0, 100], title="分數", gridcolor="#333")
    fig.update_xaxes(gridcolor="#333")
    return fig.to_html(full_html=False, include_plotlyjs=False, div_id="main_chart")


def make_event_zoom(df, event, market_label):
    """單一事件的近 90 天放大圖"""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    if df is None or df.empty:
        return ""
    crash = pd.to_datetime(event["date"])
    win_start = crash - timedelta(days=60)
    win_end = crash + timedelta(days=30)
    sub = df[(df["Date"] >= win_start) & (df["Date"] <= win_end)].copy()
    if sub.empty:
        return f"<p style='color:#888'>無資料</p>"

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(
        x=sub["Date"], y=sub["Composite"],
        name="危機指數", line=dict(color="#facc15", width=2.5),
        hovertemplate="%{x|%m/%d}<br>指數: %{y:.1f}<extra></extra>",
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=sub["Date"], y=sub["Close"],
        name="大盤指數", line=dict(color="#38bdf8", width=1.5, dash="dot"),
        hovertemplate="%{x|%m/%d}<br>大盤: %{y:.0f}<extra></extra>",
    ), secondary_y=True)
    fig.add_vline(x=crash.timestamp() * 1000,
                  line=dict(color="#ff6b6b", width=2, dash="dash"))
    fig.add_hline(y=85, line=dict(color="#dc2626", width=1, dash="dot"),
                  secondary_y=False)
    fig.add_hline(y=75, line=dict(color="#f97316", width=1, dash="dot"),
                  secondary_y=False)
    fig.add_hline(y=60, line=dict(color="#facc15", width=1, dash="dot"),
                  secondary_y=False)

    fig.update_layout(
        template="plotly_dark", height=280,
        margin=dict(t=30, b=30, l=40, r=40),
        title=dict(text=f"{event['name']} ({market_label})", font=dict(size=13)),
        showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=1.05),
        plot_bgcolor="#1a1a1c", paper_bgcolor="#1a1a1c",
    )
    fig.update_yaxes(range=[0, 100], title="危機指數", secondary_y=False, gridcolor="#333")
    fig.update_yaxes(title="大盤點數", secondary_y=True, gridcolor="#333")
    fig.update_xaxes(gridcolor="#333")
    return fig.to_html(full_html=False, include_plotlyjs=False,
                       div_id=f"zoom_{event['name']}_{market_label}".replace(" ", "_"))


def make_subindicator_chart(df, market_name):
    """子指標分數時間序列"""
    import plotly.graph_objects as go
    if df is None or df.empty:
        return ""
    sub_cols = [c for c in df.columns
                if c not in ("Date", "Close", "Composite")]
    palette = ["#ff6b6b", "#ffaa00", "#facc15", "#4ade80", "#38bdf8", "#a855f7", "#ec4899"]
    fig = go.Figure()
    for i, c in enumerate(sub_cols):
        fig.add_trace(go.Scatter(
            x=df["Date"], y=df[c], name=c,
            line=dict(color=palette[i % len(palette)], width=1.3),
            hovertemplate=c + "<br>%{x|%Y-%m-%d}: %{y:.1f}<extra></extra>",
        ))
    fig.update_layout(
        template="plotly_dark", height=320,
        margin=dict(t=30, b=30, l=50, r=20),
        title=f"{market_name} 子指標分數時間序列",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        plot_bgcolor="#1a1a1c", paper_bgcolor="#1a1a1c",
        hovermode="x unified",
    )
    fig.update_yaxes(range=[0, 100], gridcolor="#333", title="分數")
    fig.update_xaxes(gridcolor="#333")
    return fig.to_html(full_html=False, include_plotlyjs=False,
                       div_id=f"sub_{market_name}")


def render_event_card(rec):
    """單一事件的卡片 HTML"""
    market_blocks = []
    for prefix, label in [("US", "🇺🇸 美股"), ("TW", "🇹🇼 台股")]:
        if f"{prefix}_peak" not in rec:
            continue
        peak = rec[f"{prefix}_peak"]
        peak_date = rec[f"{prefix}_peak_date"]
        lead = rec[f"{prefix}_lead_days"]
        signal = rec[f"{prefix}_signal"]
        contrib = rec[f"{prefix}_top_contrib"]
        contrib_html = "".join(
            f'<span style="background:#2a2a2c;padding:2px 8px;border-radius:4px;'
            f'font-size:11px;margin-right:4px;">{name}: {score:.0f}</span>'
            for name, score in contrib
        )
        if "✅" in signal:
            verdict_color = "#4ade80"
        elif "❌" in signal:
            verdict_color = "#ff6b6b"
        else:
            verdict_color = "#facc15"
        market_blocks.append(f"""
        <div style="background:#222;padding:12px;border-radius:6px;margin-bottom:8px;">
          <div style="display:flex;justify-content:space-between;align-items:center;">
            <strong>{label}</strong>
            <span style="color:{verdict_color};font-weight:bold;">{signal}</span>
          </div>
          <div style="font-size:13px;color:#aaa;margin-top:4px;">
            指數峰值 <b style="color:#fff;">{peak:.1f}</b> @ {peak_date} ｜
            領先 <b style="color:#fff;">{lead}</b> 日
          </div>
          <div style="margin-top:6px;">{contrib_html}</div>
        </div>
        """)
    return f"""
    <div style="background:#1a1a1c;border:1px solid #333;border-radius:8px;
                padding:16px;margin-bottom:12px;">
      <div style="display:flex;justify-content:space-between;margin-bottom:8px;">
        <h3 style="margin:0;color:#fff;">⚡ {rec['event']}</h3>
        <span style="color:#888;font-size:13px;">
          崩盤起點 {rec['crash_date']} ｜ 跌幅 <b style="color:#ff6b6b;">{rec['drop_pct']:.1f}%</b>
        </span>
      </div>
      {''.join(market_blocks)}
    </div>
    """


def render_false_alarm_table(false_us, false_tw):
    def render(rows, market):
        if not rows:
            return f"<p style='color:#4ade80;'>✅ {market} 無假警報（指數 ≥75 後 30 日內都有跌 5% 以上）</p>"
        head = ("<table style='width:100%;border-collapse:collapse;font-size:13px;'>"
                "<tr style='background:#2a2a2c;'>"
                "<th style='padding:6px;text-align:left;'>日期</th>"
                "<th style='padding:6px;text-align:right;'>當日指數</th>"
                "<th style='padding:6px;text-align:right;'>後續 30 日最大跌幅</th>"
                "</tr>")
        body = "".join(
            f"<tr style='border-top:1px solid #333;'>"
            f"<td style='padding:6px;'>{str(r['Date'])[:10]}</td>"
            f"<td style='padding:6px;text-align:right;color:#ff6b6b;'>{r['Composite']:.1f}</td>"
            f"<td style='padding:6px;text-align:right;'>{r['FwdDrop']:+.1f}%</td>"
            f"</tr>" for r in rows
        )
        return f"<h4>{market} 假警報事件</h4>{head}{body}</table>"

    return render(false_us, "🇺🇸 美股") + "<br/>" + render(false_tw, "🇹🇼 台股")


def generate_html(us_df, tw_df, events_analysis, false_us, false_tw, data_status):
    """產出整個 HTML 報告"""
    main_chart_html = make_main_chart(us_df, tw_df, EVENTS)
    event_cards = "".join(render_event_card(r) for r in events_analysis)
    zoom_charts = ""
    for ev in EVENTS:
        if ev["market"] in ("us", "both"):
            zoom_charts += "<div style='margin-bottom:8px;'>"
            zoom_charts += make_event_zoom(us_df, ev, "🇺🇸 美股")
            zoom_charts += "</div>"
        if ev["market"] in ("tw", "both"):
            zoom_charts += "<div style='margin-bottom:8px;'>"
            zoom_charts += make_event_zoom(tw_df, ev, "🇹🇼 台股")
            zoom_charts += "</div>"

    sub_us = make_subindicator_chart(us_df, "🇺🇸 美股")
    sub_tw = make_subindicator_chart(tw_df, "🇹🇼 台股")
    false_html = render_false_alarm_table(false_us, false_tw)

    # 資料狀態列表
    status_html = "<table style='font-size:13px;'>"
    for k, v in data_status.items():
        icon = "✅" if v["ok"] else "❌"
        status_html += (f"<tr><td>{icon}</td><td style='padding:0 8px;'>{k}</td>"
                        f"<td style='color:#888;'>{v['msg']}</td></tr>")
    status_html += "</table>"

    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    html = f"""<!DOCTYPE html>
<html lang="zh-TW">
<head>
<meta charset="UTF-8">
<title>空頭距離指數歷史回測報告</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
  * {{ box-sizing: border-box; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang TC",
                 "Microsoft JhengHei", sans-serif;
    background: #0d0d0f; color: #e5e7eb; margin: 0; padding: 0;
  }}
  .container {{ max-width: 1400px; margin: 0 auto; padding: 24px; }}
  header {{
    background: linear-gradient(135deg, #1e3a8a 0%, #7c2d12 100%);
    padding: 30px; border-radius: 12px; margin-bottom: 24px;
  }}
  h1 {{ margin: 0; font-size: 28px; }}
  h2 {{ color: #38bdf8; border-left: 4px solid #38bdf8;
        padding-left: 12px; margin-top: 32px; }}
  h3, h4 {{ color: #facc15; }}
  .meta {{ color: #aaa; font-size: 14px; margin-top: 8px; }}
  .section {{ background: #1a1a1c; border-radius: 10px;
              padding: 20px; margin-bottom: 20px; border: 1px solid #2a2a2c; }}
  .grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
  @media (max-width: 1000px) {{ .grid-2 {{ grid-template-columns: 1fr; }} }}
  table {{ border-collapse: collapse; width: 100%; }}
  td, th {{ padding: 6px; }}
  .level-legend {{ display: flex; flex-wrap: wrap; gap: 8px; margin-top: 12px;
                   font-size: 12px; }}
  .level-tag {{ padding: 4px 10px; border-radius: 4px; color: #fff;
                font-weight: bold; }}
</style>
</head>
<body>
<div class="container">

<header>
  <h1>🚨 空頭距離指數歷史回測報告</h1>
  <div class="meta">
    生成時間 {now} ｜ 回測區間 {START_DATE} ~ {END_DATE}
  </div>
  <div class="level-legend">
    <span class="level-tag" style="background:#dc2626;">85+ 強制清倉</span>
    <span class="level-tag" style="background:#f97316;">75–85 高度危險</span>
    <span class="level-tag" style="background:#facc15;color:#000;">60–75 警戒區</span>
    <span class="level-tag" style="background:#9ca3af;color:#000;">40–60 中性</span>
    <span class="level-tag" style="background:#84cc16;color:#000;">20–40 機會浮現</span>
    <span class="level-tag" style="background:#22c55e;color:#000;">0–20 極度恐慌</span>
  </div>
</header>

<div class="section">
  <h2>📊 資料源狀態</h2>
  {status_html}
</div>

<div class="section">
  <h2>📈 危機指數歷史走勢</h2>
  <p style="color:#888;font-size:13px;">紅色虛線標註 5 大歷史事件的崩盤起點，橫向虛線為 60/75/85 三條門檻。</p>
  {main_chart_html}
</div>

<div class="section">
  <h2>⚡ 五大歷史事件分析</h2>
  <p style="color:#888;font-size:13px;">指標在崩盤前領先 ≥3 日且峰值 ≥60 視為有效訊號。</p>
  {event_cards}
</div>

<div class="section">
  <h2>🔍 事件近 90 日放大檢視</h2>
  <p style="color:#888;font-size:13px;">每張圖橫跨崩盤前 60 日到崩盤後 30 日；黃線=危機指數，藍虛線=大盤點數。</p>
  {zoom_charts}
</div>

<div class="section">
  <h2>📉 子指標分數時間序列</h2>
  {sub_us}
  <br/>
  {sub_tw}
</div>

<div class="section">
  <h2>🚨 假警報統計</h2>
  <p style="color:#888;font-size:13px;">指數 ≥75 但後續 30 日大盤沒跌超過 5% 的點。越少越好。</p>
  {false_html}
</div>

<div class="section">
  <h2>📝 結論</h2>
  <p>把上面的數字回報給我，我們一起檢視：</p>
  <ul>
    <li><b>領先性是否合格</b>：理想情況下每個事件都至少領先 3 個交易日</li>
    <li><b>哪些子指標貢獻最大</b>：低貢獻的可考慮剔除或降權</li>
    <li><b>假警報多不多</b>：太多代表門檻太敏感，要往上調</li>
  </ul>
  <p style="color:#888;">如果某些事件指標來不及反應、或假警報太多，下一輪我會調整權重與門檻。</p>
</div>

</div>
</body>
</html>
"""
    return html


# ═══════════════════════════════════════════════════════
# 主程式
# ═══════════════════════════════════════════════════════
def main():
    print("=" * 64)
    print("  空頭距離指數歷史回測 v1")
    print("=" * 64)
    print(f"  區間: {START_DATE} ~ {END_DATE}")
    print(f"  FRED key: {'✓' if FRED_KEY else '✗ 未設'}")
    print(f"  快取目錄: {CACHE_DIR}/")
    print("=" * 64)
    print()

    _ensure_cache()
    data = {}
    status = {}

    # ─── 抓資料 ───
    log("Step 1/4: 下載歷史資料...", "step")

    fetchers = [
        ("spx",        "S&P 500",              lambda: fetch_yf("^GSPC")),
        ("twii",       "台灣加權",             lambda: fetch_yf("^TWII")),
        ("vix",        "VIX",                  lambda: fetch_yf("^VIX")),
        ("vix3m",      "VIX3M",                lambda: fetch_yf("^VIX3M")),
        ("tsm",        "TSM ADR",              lambda: fetch_yf("TSM")),
        ("t2330",      "2330.TW",              lambda: fetch_yf("2330.TW")),
        ("twd",        "TWD=X",                lambda: fetch_yf("TWD=X")),
        ("t10y2y",     "FRED 10Y-2Y",          lambda: fetch_fred("T10Y2Y")),
        ("hy_spread",  "FRED HY 信用利差",     lambda: fetch_fred("BAMLH0A0HYM2")),
        ("naaim",      "NAAIM 歷史",           fetch_naaim_history),
        ("aaii",       "AAII 歷史",            fetch_aaii_history),
        ("tw_margin",  "台股融資餘額",         fetch_finmind_total_margin),
        ("tw_per",     "台積電 PER",           fetch_finmind_per),
    ]

    for key, label, fn in fetchers:
        try:
            df = fn()
            if df is None or df.empty:
                data[key] = pd.DataFrame()
                status[label] = {"ok": False, "msg": "無資料"}
                log(f"  {label}: 失敗", "warn")
            else:
                data[key] = df
                date_range = f"{df['Date'].min().date()} ~ {df['Date'].max().date()}"
                status[label] = {"ok": True, "msg": f"{len(df)} 筆 ({date_range})"}
                log(f"  {label}: {len(df)} 筆", "ok")
        except Exception as e:
            data[key] = pd.DataFrame()
            status[label] = {"ok": False, "msg": str(e)[:80]}
            log(f"  {label}: ERROR {e}", "err")
        time.sleep(0.3)

    # ─── 計算指數 ───
    log("Step 2/4: 計算每日指數...", "step")
    try:
        us_df, tw_df = compute_daily_indices(data)
    except Exception as e:
        log(f"指數計算失敗: {e}", "err")
        traceback.print_exc()
        return

    log(f"  US 指數: {len(us_df)} 筆 | TW 指數: {len(tw_df)} 筆", "ok")
    if not us_df.empty:
        valid = us_df.dropna(subset=["Composite"])
        if not valid.empty:
            log(f"  US 最新分數: {valid['Composite'].iloc[-1]:.1f} "
                f"({valid['Date'].iloc[-1].date()})", "ok")
    if not tw_df.empty:
        valid = tw_df.dropna(subset=["Composite"])
        if not valid.empty:
            log(f"  TW 最新分數: {valid['Composite'].iloc[-1]:.1f} "
                f"({valid['Date'].iloc[-1].date()})", "ok")

    # ─── 事件分析 ───
    log("Step 3/4: 分析五大歷史事件...", "step")
    events_analysis = analyze_events(us_df, tw_df)
    for r in events_analysis:
        line = f"  {r['event']} ({r['crash_date']})"
        if "US_peak" in r:
            line += f"  US: {r['US_peak']:.1f}/+{r['US_lead_days']}d {r['US_signal']}"
        if "TW_peak" in r:
            line += f"  TW: {r['TW_peak']:.1f}/+{r['TW_lead_days']}d {r['TW_signal']}"
        print(line)

    false_us = find_false_alarms(us_df)
    false_tw = find_false_alarms(tw_df)
    log(f"  US 假警報: {len(false_us)} 起 | TW 假警報: {len(false_tw)} 起", "ok")

    # ─── 生成 HTML ───
    log("Step 4/4: 產生 HTML 報告...", "step")
    try:
        html = generate_html(us_df, tw_df, events_analysis, false_us, false_tw, status)
        with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
            f.write(html)
        log(f"已輸出 {OUTPUT_HTML}", "ok")
    except Exception as e:
        log(f"HTML 生成失敗: {e}", "err")
        traceback.print_exc()
        return

    print()
    print("=" * 64)
    print(f"  ✅ 完成！打開 {OUTPUT_HTML} 看結果")
    print(f"     完整路徑: {os.path.abspath(OUTPUT_HTML)}")
    print("=" * 64)


if __name__ == "__main__":
    main()
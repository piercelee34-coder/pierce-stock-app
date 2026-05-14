"""
rocket_screener.py — 火箭類股探測器

功能：
  - 掃描主題字典內的所有股票，計算近期漲幅
  - 排出漲幅最強的 TOP 5 主題（週報酬 + 月報酬）
  - 每個主題顯示代表股票 + 平均漲幅

快取設計：
  - 每次抓完寫入 .rocket_cache/，TTL = 4 小時
  - 只在美股/台股盤後才更新

用法：
  import rocket_screener
  result = rocket_screener.get_top_themes(market="us", top_n=5, period="1w")
  # result = [
  #   {"theme": "⚡ AI 晶片", "avg_ret": 8.3, "median_ret": 7.1,
  #    "top_stocks": [("NVDA", 12.1), ("AMD", 8.4), ("AVGO", 6.2)],
  #    "hot_count": 4},
  #   ...
  # ]
"""

import os
import json
import time
import warnings
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

CACHE_DIR = ".rocket_cache"
THEMES_US_FILE = "themes_us.json"
THEMES_TW_FILE = "themes_tw.json"
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"

# 最低需要幾檔股票資料才算這個主題有效
MIN_STOCKS_PER_THEME = 2
# 每次批次抓幾檔（避免 yfinance rate limit）
BATCH_SIZE = 8
BATCH_SLEEP = 0.5


# ──────────────────────────────────────────────────────
# 快取工具
# ──────────────────────────────────────────────────────
def _cache_path(key):
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, f"{key}.json")


def cache_get(key, max_age_hours=4):
    path = _cache_path(key)
    if not os.path.exists(path):
        return None
    if (time.time() - os.path.getmtime(path)) / 3600 > max_age_hours:
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
# 主題字典載入
# ──────────────────────────────────────────────────────
def _load_themes(market="us"):
    fname = THEMES_US_FILE if market == "us" else THEMES_TW_FILE
    # 先找專案根目錄，再找模組同層目錄
    for base in (".", os.path.dirname(__file__)):
        path = os.path.join(base, fname)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    return {}


# ──────────────────────────────────────────────────────
# 股票報酬計算
# ──────────────────────────────────────────────────────
def _period_to_yf(period):
    """把 1w / 1m / 3m 轉換為 yfinance 的 period / interval"""
    mapping = {
        "1w": ("10d",  "1d"),
        "1m": ("35d",  "1d"),
        "3m": ("95d",  "1d"),
    }
    return mapping.get(period, ("10d", "1d"))


def _fetch_returns(tickers, period="1w"):
    """批次抓取多檔股票的報酬率，回傳 {ticker: pct_change}"""
    import yfinance as yf
    yf_period, yf_interval = _period_to_yf(period)
    returns = {}
    lookback = {"1w": 5, "1m": 20, "3m": 60}.get(period, 5)

    for i in range(0, len(tickers), BATCH_SIZE):
        batch = tickers[i:i+BATCH_SIZE]
        try:
            data = yf.download(
                batch, period=yf_period, interval=yf_interval,
                progress=False, auto_adjust=True,
            )
            if data.empty:
                continue
            # 處理單/多股票的欄位結構
            if isinstance(data.columns, pd.MultiIndex):
                close = data["Close"] if "Close" in data.columns.get_level_values(0) else None
            else:
                close = data[["Close"]] if "Close" in data.columns else None

            if close is None or close.empty:
                continue

            # 對每檔算報酬
            for t in batch:
                try:
                    if t in close.columns:
                        col = close[t].dropna()
                    elif isinstance(close, pd.DataFrame) and len(close.columns) == 1:
                        col = close.iloc[:, 0].dropna()
                    else:
                        continue
                    if len(col) < 2:
                        continue
                    # 取第一筆和最後一筆
                    start_price = col.iloc[0]
                    end_price = col.iloc[-1]
                    if start_price > 0:
                        ret = (end_price / start_price - 1) * 100
                        returns[t] = round(float(ret), 2)
                except Exception:
                    continue
        except Exception:
            pass
        if i + BATCH_SIZE < len(tickers):
            time.sleep(BATCH_SLEEP)

    return returns


# ──────────────────────────────────────────────────────
# 主函式
# ──────────────────────────────────────────────────────
def get_top_themes(market="us", top_n=5, period="1w", force_refresh=False):
    """
    掃描主題字典，回傳漲幅最強的 top_n 個主題。

    Args:
      market: "us" 或 "tw"
      top_n: 回傳幾個主題
      period: "1w"（週）/ "1m"（月）/ "3m"（季）
      force_refresh: 強制重抓（忽略快取）

    Returns:
      list of dict:
        theme: 主題名稱
        avg_ret: 主題平均報酬 %
        median_ret: 中位數報酬 %（去除極端值）
        hot_count: 正報酬股票數量
        total_count: 總股票數量
        top_stocks: [(ticker, ret%), ...] 排名前3
        worst_stocks: [(ticker, ret%), ...] 排名後1
    """
    cache_key = f"rocket_{market}_{period}"
    if not force_refresh:
        cached = cache_get(cache_key, max_age_hours=4)
        if cached:
            return cached

    themes = _load_themes(market)
    if not themes:
        return []

    # 收集所有唯一股票，一次批次抓取（節省時間）
    all_tickers = list({t for tickers in themes.values() for t in tickers})
    returns = _fetch_returns(all_tickers, period=period)

    results = []
    for theme_name, tickers in themes.items():
        # 只計算有資料的股票
        valid = {t: returns[t] for t in tickers if t in returns}
        if len(valid) < MIN_STOCKS_PER_THEME:
            continue

        rets = list(valid.values())
        avg_ret = round(np.mean(rets), 2)
        median_ret = round(np.median(rets), 2)
        hot_count = sum(1 for r in rets if r > 0)

        # 前3名 / 後1名
        sorted_stocks = sorted(valid.items(), key=lambda x: -x[1])
        top3 = sorted_stocks[:3]
        worst = sorted_stocks[-1:]

        results.append({
            "theme": theme_name,
            "avg_ret": avg_ret,
            "median_ret": median_ret,
            "hot_count": hot_count,
            "total_count": len(valid),
            "top_stocks": top3,
            "worst_stocks": worst,
            "period": period,
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        })

    # 依 median_ret 排序（對抗單一股拉抬平均）
    results.sort(key=lambda x: -x["median_ret"])
    top_results = results[:top_n]

    cache_set(cache_key, top_results)
    return top_results


def get_bottom_themes(market="us", bottom_n=3, period="1w"):
    """取跌最深的 bottom_n 個主題（危機警示）"""
    cache_key = f"rocket_{market}_{period}"
    cached = cache_get(cache_key, max_age_hours=4)
    if not cached:
        # 先跑完整掃描
        themes = _load_themes(market)
        all_tickers = list({t for tickers in themes.values() for t in tickers})
        returns = _fetch_returns(all_tickers, period=period)
        results = []
        for theme_name, tickers in themes.items():
            valid = {t: returns[t] for t in tickers if t in returns}
            if len(valid) < MIN_STOCKS_PER_THEME:
                continue
            rets = list(valid.values())
            results.append({
                "theme": theme_name,
                "avg_ret": round(np.mean(rets), 2),
                "median_ret": round(np.median(rets), 2),
                "hot_count": sum(1 for r in rets if r > 0),
                "total_count": len(valid),
                "top_stocks": sorted(valid.items(), key=lambda x: -x[1])[:3],
                "period": period,
            })
        cached = sorted(results, key=lambda x: -x["median_ret"])
        cache_set(cache_key, cached)

    return sorted(cached, key=lambda x: x["median_ret"])[:bottom_n]

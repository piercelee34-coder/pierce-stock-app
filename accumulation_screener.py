"""
accumulation_screener.py — 悄悄吸籌探測器

學術依據：accumulation/distribution 指標家族（Granville, Wyckoff, OBV）

5 個訊號（4/5 中算強訊號，5/5 中算極強訊號）：
  1. 過去 30 日股價變化 < +5%（沒大漲，還在低點蓄勢）
  2. 過去 30 日成交量 > 60 日均量 1.5 倍（量在增加）
  3. 過去 30 日 OBV 創 60 日新高（資金累積流入）
  4. 過去 30 日上漲日的成交量 > 下跌日的成交量（買盤強於賣盤）
  5. 股價站上 50MA 且離 52w 高還有 > 10% 空間（不在頂部）

對外函式：
  get_accumulation_signals(market="us", min_signals=4) → list of dict
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

CACHE_DIR = ".accumulation_cache"
THEMES_US_FILE = "themes_us.json"
THEMES_TW_FILE = "themes_tw.json"
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"

# 4 個計分訊號（訊號 1 已變為過濾條件 — 沒大漲必中）
SIGNAL_NAMES = [
    "量能放大 (30日均 > 60日均 1.5x)",
    "OBV 創 60 日新高",
    "上漲日量 > 下跌日量 1.3x",
    "站上 50MA 且距高點 > 10%",
]
# 過濾條件（必中才會進入計分）
FILTER_RULE = "股價 30 日變化在 -10% ~ +8% 之間（沒大漲、沒崩盤）"


# ──────────────────────────────────────────────────────
# 快取
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
# 主題字典
# ──────────────────────────────────────────────────────
def _load_themes(market="us"):
    fname = THEMES_US_FILE if market == "us" else THEMES_TW_FILE
    for base in (".", os.path.dirname(__file__) or "."):
        path = os.path.join(base, fname)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    return {}


def _stock_to_themes(themes):
    """反向字典：stock → [theme1, theme2]"""
    result = {}
    for theme, stocks in themes.items():
        for s in stocks:
            result.setdefault(s, []).append(theme)
    return result


# ──────────────────────────────────────────────────────
# 計算 5 個訊號
# ──────────────────────────────────────────────────────
def _analyze_stock(df):
    """對單一股票 DataFrame 計算吸籌訊號

    新規則（v2）：
      訊號 1（沒大漲）→ 過濾條件，沒中直接排除
      訊號 2-5 → 計分（4 個中 3 個 = 強訊號、4/4 = 極強訊號）

    df 需有: Date, Open, High, Low, Close, Volume，至少 70 個交易日
    """
    if df is None or len(df) < 70:
        return None
    df = df.copy().reset_index(drop=True)
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce")
    df = df.dropna(subset=["Close", "Volume"])
    if len(df) < 70:
        return None

    last_close = float(df["Close"].iloc[-1])

    # ── 流動性篩選（最低門檻）──
    vol_30d_avg = float(df["Volume"].iloc[-30:].mean())
    if vol_30d_avg < 500_000:
        return None  # 流動性不足，跳過

    # ── 分類：吸籌 vs 邊緣 vs 排除 ──
    price_30d_ago = float(df["Close"].iloc[-31]) if len(df) >= 31 else float(df["Close"].iloc[0])
    price_chg_pct = (last_close / price_30d_ago - 1) * 100 if price_30d_ago > 0 else 0

    if price_chg_pct <= -10.0:
        return None  # 暴跌，排除
    elif price_chg_pct < 8.0:
        category = "accumulation"   # 沒大漲，正規吸籌候選
    elif price_chg_pct < 15.0:
        category = "edge"           # 8~15% 之間，邊緣候選（剛起漲）
    else:
        return None  # 已大漲超過 15%，不再算吸籌

    # ── 訊號 2: 30 日均量 > 60 日均量 1.5x ──
    vol_60d = float(df["Volume"].iloc[-60:].mean())
    vol_ratio = vol_30d_avg / vol_60d if vol_60d > 0 else 1.0
    signal_2 = vol_ratio >= 1.5

    # ── 訊號 3: OBV 創 60 日新高 ──
    df["price_diff"] = df["Close"].diff()
    df["obv_change"] = np.where(df["price_diff"] > 0, df["Volume"],
                                  np.where(df["price_diff"] < 0, -df["Volume"], 0))
    df["OBV"] = df["obv_change"].cumsum()
    obv_now = float(df["OBV"].iloc[-1])
    obv_60d_max = float(df["OBV"].iloc[-60:].max())
    signal_3 = obv_now >= obv_60d_max * 0.99  # 允許小誤差

    # ── 訊號 4: 30 日上漲日量 > 下跌日量 ──
    last_30 = df.iloc[-30:].copy()
    up_days = last_30[last_30["price_diff"] > 0]
    down_days = last_30[last_30["price_diff"] < 0]
    up_vol_avg = float(up_days["Volume"].mean()) if len(up_days) > 0 else 0
    down_vol_avg = float(down_days["Volume"].mean()) if len(down_days) > 0 else 1
    up_down_ratio = up_vol_avg / down_vol_avg if down_vol_avg > 0 else 0
    signal_4 = up_down_ratio >= 1.3

    # ── 訊號 5: 站上 50MA 且距 52w 高 > 10% ──
    sma_50 = float(df["Close"].iloc[-50:].mean())
    high_52w = float(df["Close"].iloc[-252:].max()) if len(df) >= 252 else float(df["Close"].max())
    above_sma50 = last_close > sma_50
    dist_from_high = (high_52w - last_close) / high_52w * 100 if high_52w > 0 else 0
    signal_5 = above_sma50 and dist_from_high > 10

    # 計分（訊號 2-5，總共 4 個）
    scoring_signals = [signal_2, signal_3, signal_4, signal_5]
    n_hit = sum(scoring_signals)

    return {
        "n_hit": n_hit,                # 0-4
        "signals": scoring_signals,    # 4 個訊號
        "category": category,          # "accumulation" 或 "edge"
        "metrics": {
            "price_chg_30d": round(price_chg_pct, 2),
            "vol_ratio": round(vol_ratio, 2),
            "obv_pct_of_60d_max": round(obv_now / obv_60d_max * 100, 1) if obv_60d_max > 0 else 0,
            "up_down_vol_ratio": round(up_down_ratio, 2),
            "dist_from_52w_high_pct": round(dist_from_high, 1),
            "above_sma50": above_sma50,
            "vol_30d_avg": int(vol_30d_avg),
        },
        "last_close": last_close,
    }


# ──────────────────────────────────────────────────────
# 批次抓股票歷史價（70+ 個交易日）
# ──────────────────────────────────────────────────────
def _fetch_history_batch(tickers, batch_size=8):
    """批次抓多檔股票的 70 日歷史，回傳 {ticker: df}"""
    import yfinance as yf
    out = {}
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i+batch_size]
        try:
            data = yf.download(
                batch, period="6mo", interval="1d",
                progress=False, auto_adjust=True, threads=False,
            )
            if data.empty:
                continue
            if isinstance(data.columns, pd.MultiIndex):
                # 多檔
                for t in batch:
                    try:
                        sub = pd.DataFrame({
                            "Date": data.index,
                            "Open": data["Open"][t] if t in data["Open"].columns else pd.NA,
                            "High": data["High"][t] if t in data["High"].columns else pd.NA,
                            "Low":  data["Low"][t]  if t in data["Low"].columns  else pd.NA,
                            "Close": data["Close"][t] if t in data["Close"].columns else pd.NA,
                            "Volume": data["Volume"][t] if t in data["Volume"].columns else pd.NA,
                        }).dropna(subset=["Close"])
                        if len(sub) >= 70:
                            out[t] = sub
                    except Exception:
                        continue
            else:
                # 單檔
                if len(batch) == 1 and not data.empty:
                    sub = data.reset_index()
                    if len(sub) >= 70:
                        out[batch[0]] = sub
        except Exception:
            pass
        time.sleep(0.3)
    return out


# ──────────────────────────────────────────────────────
# 主函式
# ──────────────────────────────────────────────────────
def get_accumulation_signals(market="us", min_signals=3, force_refresh=False):
    """
    掃描主題字典裡的股票，找出潛在吸籌股票。

    Returns:
      dict: {
        "accumulation": [...],   # 沒大漲（30日 < +8%）的吸籌候選
        "edge":         [...],   # 邊緣候選（30日漲 8~15%，剛起漲）
      }
    每筆結構：
      {
        "ticker": "TER", "n_hit": 3, "level": (...), "themes": [...],
        "signals": [...], "metrics": {...}, "last_close": 89.50,
        "category": "accumulation" / "edge",
      }
    """
    cache_key = f"accumulation_v2_{market}"
    if not force_refresh:
        cached = cache_get(cache_key, max_age_hours=4)
        if cached:
            return cached

    themes = _load_themes(market)
    if not themes:
        return {"accumulation": [], "edge": []}

    stock_to_themes = _stock_to_themes(themes)
    all_tickers = list(stock_to_themes.keys())
    if not all_tickers:
        return {"accumulation": [], "edge": []}

    # 批次抓歷史
    history = _fetch_history_batch(all_tickers)

    # 分析每檔
    accumulation = []
    edge = []
    for ticker, df in history.items():
        result = _analyze_stock(df)
        if result is None:
            continue
        if result["n_hit"] < min_signals:
            continue
        # 等級（基於 4 個計分訊號）
        if result["n_hit"] == 4:
            level = ("🔴 極強訊號", "#dc2626")
        elif result["n_hit"] == 3:
            level = ("🟠 強訊號", "#f97316")
        else:  # 2
            level = ("🟡 觀察名單", "#facc15")

        record = {
            "ticker": ticker,
            "n_hit": result["n_hit"],
            "level": level,
            "themes": stock_to_themes.get(ticker, []),
            "signals": result["signals"],
            "metrics": result["metrics"],
            "last_close": result["last_close"],
            "category": result["category"],
        }

        if result["category"] == "accumulation":
            accumulation.append(record)
        else:
            edge.append(record)

    # 排序：先依 n_hit 倒序，相同則依量能放大倍數
    accumulation.sort(key=lambda x: (-x["n_hit"], -x["metrics"]["vol_ratio"]))
    edge.sort(key=lambda x: (-x["n_hit"], -x["metrics"]["vol_ratio"]))

    out = {"accumulation": accumulation, "edge": edge}
    cache_set(cache_key, out)
    return out

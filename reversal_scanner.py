"""
reversal_scanner.py — 💡 反轉預警 v6 掃描器 + 訊號追蹤

功能：
  1. 掃描 S&P 100 成長股，找最近 5 天內出現的反轉預警訊號
  2. 自動記錄每個訊號到 reversal_tracking.json
  3. 5/10/20 天後自動回填實際表現
  4. 累積樣本到 25+ 後可決定是否正式採用
"""

import os
import json
import time
import warnings
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import yfinance as yf

warnings.filterwarnings("ignore")

# ── S&P 100 成長股（已排除公用事業/必需消費等保守股）──
SP100_GROWTH = [
    # 大型科技 (22)
    "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "META", "NVDA", "TSLA", "AVGO", "ORCL",
    "ADBE", "CRM", "AMD", "INTC", "QCOM", "CSCO", "TXN", "INTU", "AMAT", "MU",
    "IBM", "ACN",
    # 金融 (15)
    "JPM", "BAC", "WFC", "GS", "MS", "BLK", "AXP", "C", "SCHW", "V", "MA", "PYPL",
    "USB", "SPGI", "MMC",
    # 醫療成長 (9)
    "UNH", "LLY", "TMO", "ABT", "DHR", "BMY", "AMGN", "GILD", "ISRG",
    # 消費循環 (9)
    "MCD", "SBUX", "DIS", "HD", "LOW", "TGT", "TJX", "NKE", "BKNG",
    # 工業/動能 (9)
    "BA", "CAT", "HON", "GE", "LMT", "RTX", "UPS", "FDX", "DE",
    # 通訊/媒體 (3)
    "CMCSA", "NFLX", "CHTR",
    # 其他高動能 (3)
    "BRK-B", "BX", "LIN",
]

_TRACK_FILE = "reversal_tracking.json"


# ──────────────────────────────────────────────────────
# 基礎資料抓取與指標
# ──────────────────────────────────────────────────────
def _fetch_stock(ticker, days=180):
    try:
        end = datetime.now()
        start = end - timedelta(days=days)
        df = yf.Ticker(ticker).history(start=start, end=end, auto_adjust=False)
        if df.empty: return None
        df = df.reset_index()
        df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
        return df
    except Exception:
        return None


def _compute_indicators(df):
    df = df.copy()
    close = df["Close"]
    df["SMA_60"] = close.rolling(60).mean()
    df["SMA_200"] = close.rolling(200).mean()
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["Signal_Line"] = df["MACD"].ewm(span=9, adjust=False).mean()
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["RSI"] = 100 - 100 / (1 + rs)
    df["price_diff"] = df["Close"].diff()
    df["obv_change"] = np.where(df["price_diff"] > 0, df["Volume"],
                                  np.where(df["price_diff"] < 0, -df["Volume"], 0))
    df["OBV"] = df["obv_change"].cumsum()
    return df


def _fetch_spy_state():
    spy = _fetch_stock("SPY")
    if spy is None: return None
    spy = _compute_indicators(spy)
    spy["above_sma200"] = spy["Close"] > spy["SMA_200"]
    return spy


def _market_ok_at(spy, date):
    if spy is None: return True
    sub = spy[pd.to_datetime(spy["Date"]) <= pd.to_datetime(date)]
    if sub.empty: return True
    return bool(sub.iloc[-1]["above_sma200"])


# ──────────────────────────────────────────────────────
# v6 反轉預警偵測（與回測腳本完全一致）
# ──────────────────────────────────────────────────────
def _detect_reversal_in_recent_days(df, ticker, spy, lookback_days=5):
    """偵測最近 N 天內的 v6 反轉預警"""
    df = df.copy().reset_index(drop=True)
    if len(df) < 60: return None

    df["macd_gold"] = ((df["MACD"] > df["Signal_Line"]) &
                       (df["MACD"].shift(1) <= df["Signal_Line"].shift(1)))

    # 只看最近 N 個交易日的金叉
    recent_start = len(df) - lookback_days
    candidates = df.index[df["macd_gold"] & (df.index >= recent_start)]
    if len(candidates) == 0: return None

    for idx in candidates:
        if idx < 20: continue
        curr = df.iloc[idx]
        if pd.isna(curr.get("RSI")) or pd.isna(curr.get("SMA_60")):
            continue

        # v6 條件（與 backtest_signals_v6.py 完全一致）
        macd_near_zero = abs(curr["MACD"]) < curr["Close"] * 0.005

        rsi_w10 = df.iloc[max(0, idx - 10): idx + 1]["RSI"]
        rsi_was_deep = rsi_w10.min() < 30 if not rsi_w10.empty else False
        rsi_now_up = curr["RSI"] > 45
        rsi_real_reversal = rsi_was_deep and rsi_now_up

        market_ok = _market_ok_at(spy, curr["Date"])

        sma60 = curr["SMA_60"]
        if sma60 <= 0: continue
        dist_from_sma60 = (curr["Close"] - sma60) / sma60 * 100
        in_range = -15 < dist_from_sma60 < 15

        if idx >= 20:
            prior_20 = df.iloc[idx - 20]
            momentum = (curr["Close"] / prior_20["Close"] - 1) * 100 if prior_20["Close"] > 0 else 0
        else:
            momentum = 0
        not_crashing = momentum > -15

        obv_60d = df.iloc[max(0, idx - 60): idx + 1]["OBV"]
        if not obv_60d.empty:
            omax = obv_60d.max(); omin = obv_60d.min()
            obv_pct = ((curr["OBV"] - omin) / (omax - omin) * 100) if omax > omin else 50
        else:
            obv_pct = 50

        if (macd_near_zero and rsi_real_reversal and market_ok
                and in_range and not_crashing and obv_pct > 70):
            # 找到了
            return {
                "ticker": ticker,
                "date": curr["Date"].strftime("%Y-%m-%d"),
                "close": float(curr["Close"]),
                "rsi_was": int(rsi_w10.min()) if not rsi_w10.empty else 0,
                "rsi_now": int(curr["RSI"]),
                "obv_pct": int(obv_pct),
                "dist_sma60": round(dist_from_sma60, 1),
                "momentum_20d": round(momentum, 2),
                "days_ago": int(len(df) - 1 - idx),
            }
    return None


# ──────────────────────────────────────────────────────
# 掃描主功能
# ──────────────────────────────────────────────────────
def scan_reversal_signals(lookback_days=5):
    """掃描 S&P 100 成長股，找最近 N 天內的反轉預警"""
    spy = _fetch_spy_state()
    results = []
    failed = []
    for ticker in SP100_GROWTH:
        df = _fetch_stock(ticker)
        if df is None or len(df) < 100:
            failed.append(ticker)
            continue
        df = _compute_indicators(df)
        sig = _detect_reversal_in_recent_days(df, ticker, spy, lookback_days)
        if sig:
            results.append(sig)
        time.sleep(0.1)

    # 排序：OBV 高 > RSI 反轉深 > 距 SMA60 小
    results.sort(key=lambda x: (
        -x["obv_pct"],
        -(x["rsi_now"] - x["rsi_was"]),
        abs(x["dist_sma60"])
    ))
    return {
        "signals": results,
        "failed_count": len(failed),
        "scan_time": datetime.now().isoformat(),
        "total_scanned": len(SP100_GROWTH),
    }


# ──────────────────────────────────────────────────────
# 訊號追蹤（保留 6 個月實戰觀察用）
# ──────────────────────────────────────────────────────
def _load_tracking():
    if os.path.exists(_TRACK_FILE):
        try:
            with open(_TRACK_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"signals": []}


def _save_tracking(data):
    try:
        with open(_TRACK_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def _fetch_actual_return(ticker, signal_date_str, days, signal_price):
    """取訊號日 +N 個交易日的實際漲跌幅"""
    try:
        signal_date = pd.to_datetime(signal_date_str)
        end = signal_date + timedelta(days=days + 14)
        start = signal_date - timedelta(days=1)
        df = yf.Ticker(ticker).history(start=start, end=end, auto_adjust=False)
        if df.empty: return None
        df = df.reset_index()
        df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
        on_or_after = df[df["Date"] >= signal_date]
        if len(on_or_after) <= days: return None
        future_price = float(on_or_after.iloc[days]["Close"])
        ret = (future_price / signal_price - 1) * 100
        return round(ret, 2)
    except Exception:
        return None


def update_and_get_tracking(scan_result):
    """新增掃描出的訊號 + 評估舊訊號"""
    data = _load_tracking()
    existing_keys = set((s["ticker"], s["date"]) for s in data["signals"])

    # 新增本次掃描到的訊號
    for sig in scan_result.get("signals", []):
        key = (sig["ticker"], sig["date"])
        if key not in existing_keys:
            data["signals"].append({
                **{k: v for k, v in sig.items() if k != "days_ago"},
                "added_at": datetime.now().isoformat(),
                "actual_5d": None,
                "actual_10d": None,
                "actual_20d": None,
            })

    # 評估舊訊號（5/10/20 天後是否漲）
    today = datetime.now()
    for sig in data["signals"]:
        signal_date = pd.to_datetime(sig["date"])
        days_passed = (today - signal_date).days

        if days_passed >= 7 and sig.get("actual_5d") is None:
            sig["actual_5d"] = _fetch_actual_return(sig["ticker"], sig["date"], 5, sig["close"])
        if days_passed >= 14 and sig.get("actual_10d") is None:
            sig["actual_10d"] = _fetch_actual_return(sig["ticker"], sig["date"], 10, sig["close"])
        if days_passed >= 28 and sig.get("actual_20d") is None:
            sig["actual_20d"] = _fetch_actual_return(sig["ticker"], sig["date"], 20, sig["close"])

    _save_tracking(data)

    # 統計
    signals = data["signals"]
    evaluated_10d = [s for s in signals if s.get("actual_10d") is not None]
    if evaluated_10d:
        wins = sum(1 for s in evaluated_10d if s["actual_10d"] > 0)
        win_rate = wins / len(evaluated_10d) * 100
        avg_ret = sum(s["actual_10d"] for s in evaluated_10d) / len(evaluated_10d)
    else:
        win_rate = 0
        avg_ret = 0

    return {
        "total_signals": len(signals),
        "evaluated_count": len(evaluated_10d),
        "pending_count": len(signals) - len(evaluated_10d),
        "win_rate_10d": round(win_rate, 1),
        "avg_return_10d": round(avg_ret, 2),
        "all_signals": signals,
        "can_decide": len(evaluated_10d) >= 25,  # 達 25 個樣本可決定採用
    }


# ──────────────────────────────────────────────────────
# 蒙地卡羅模擬
# ──────────────────────────────────────────────────────
def generate_monte_carlo_bands(df, days=30, n_simulations=1000, drift_adjust=0.0):
    """蒙地卡羅模擬未來 N 天的價格分布
    
    使用 Geometric Brownian Motion:
        S(t+1) = S(t) * exp((mu - 0.5*sigma^2) + sigma*Z)
    
    drift_adjust: 從 7 維 context 計算的漂移調整（-0.002 ~ +0.002）
    """
    try:
        close = df['Close'].dropna()
        if len(close) < 60: return None

        log_returns = np.log(close / close.shift(1)).dropna()
        mu = float(log_returns.mean()) + drift_adjust
        sigma = float(log_returns.std())
        last_price = float(close.iloc[-1])

        # 生成未來交易日
        last_d = df.index[-1] if hasattr(df.index[-1], 'weekday') else pd.to_datetime(df['Date'].iloc[-1])
        future_dates = []
        d = pd.to_datetime(last_d)
        while len(future_dates) < days:
            d += pd.Timedelta(days=1)
            if d.weekday() < 5:
                future_dates.append(d)

        # 跑模擬
        np.random.seed(42)  # 固定種子讓結果可重現
        paths = np.zeros((n_simulations, days))
        for i in range(n_simulations):
            prices = np.empty(days + 1)
            prices[0] = last_price
            for j in range(days):
                z = np.random.normal()
                prices[j + 1] = prices[j] * np.exp((mu - 0.5 * sigma ** 2) + sigma * z)
            paths[i] = prices[1:]

        return {
            "dates": future_dates,
            "p10": np.percentile(paths, 10, axis=0).tolist(),
            "p25": np.percentile(paths, 25, axis=0).tolist(),
            "p50": np.percentile(paths, 50, axis=0).tolist(),
            "p75": np.percentile(paths, 75, axis=0).tolist(),
            "p90": np.percentile(paths, 90, axis=0).tolist(),
            "mu": mu,
            "sigma": sigma,
            "last_price": last_price,
            "drift_adjust": drift_adjust,
        }
    except Exception:
        return None

#!/usr/bin/env python3
"""
backtest_signals_v2.py — 完整訊號回測 v2

新增：
  1. 涵蓋 3 種訊號：🌕 黃金交叉 / 🚀 起漲點 / ⭐ MACD 起漲確認
  2. 對比「現有規則 vs 新規則」（方案 A + MA60 上升 + 大盤過濾器）
  3. 找出失敗案例的真正訊號類型
"""
import os
import sys
import json
import warnings
import time
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

LOOKAHEAD_DAYS = [5, 10, 20]
HISTORY_MONTHS = 18
OUTPUT_FILE = "backtest_signals_v2_report.html"

# SPY 作為大盤過濾器
MARKET_TICKER = "SPY"

FAILURE_CASES = [
    # 假起漲
    {"ticker": "ONDS",  "date": "2026-05-01", "type": "false_launch"},
    {"ticker": "META",  "date": "2026-04-28", "type": "false_launch"},
    {"ticker": "META",  "date": "2026-03-04", "type": "false_launch"},
    {"ticker": "GLW",   "date": "2026-04-24", "type": "false_launch"},
    {"ticker": "QQQ",   "date": "2026-01-27", "type": "false_launch"},
    {"ticker": "QQQ",   "date": "2026-02-25", "type": "false_launch"},
    {"ticker": "CRCL",  "date": "2025-12-22", "type": "false_launch"},
    {"ticker": "TSLA",  "date": "2026-02-25", "type": "false_launch"},
    {"ticker": "TSLA",  "date": "2026-03-10", "type": "false_launch"},
    {"ticker": "RXRX",  "date": "2026-01-22", "type": "false_launch"},
    {"ticker": "RXRX",  "date": "2026-05-01", "type": "false_launch"},
    {"ticker": "GOOG",  "date": "2026-01-28", "type": "false_launch"},
    {"ticker": "GOOG",  "date": "2026-03-10", "type": "false_launch"},
    {"ticker": "MU",    "date": "2026-03-16", "type": "false_launch"},
    {"ticker": "MSFT",  "date": "2026-01-27", "type": "false_launch"},
    {"ticker": "MSFT",  "date": "2026-01-07", "type": "false_launch"},
    {"ticker": "MSFT",  "date": "2026-04-28", "type": "false_launch"},
    {"ticker": "SOFI",  "date": "2025-12-26", "type": "false_launch"},
    {"ticker": "COIN",  "date": "2026-01-06", "type": "false_launch"},
    {"ticker": "NKE",   "date": "2026-02-17", "type": "false_launch"},
    {"ticker": "NKE",   "date": "2026-03-31", "type": "false_launch"},
    {"ticker": "BE",    "date": "2026-02-25", "type": "false_launch"},
    {"ticker": "NFLX",  "date": "2026-04-02", "type": "false_launch"},
    {"ticker": "PLTR",  "date": "2026-04-08", "type": "false_launch"},
    {"ticker": "AMZN",  "date": "2026-01-28", "type": "false_launch"},
    # 漏標
    {"ticker": "META",  "date": "2026-04-07", "type": "missed_launch"},
    {"ticker": "TSLA",  "date": "2025-11-25", "type": "missed_launch"},
    {"ticker": "PLTR",  "date": "2025-12-01", "type": "missed_launch"},
    {"ticker": "SNDK",  "date": "2025-12-08", "type": "missed_launch"},
    {"ticker": "RKLB",  "date": "2026-04-06", "type": "missed_launch"},
]


def log(msg, kind="info"):
    icons = {"info": "🔹", "ok": "✅", "warn": "⚠️ ", "err": "❌", "step": "🔹"}
    print(f"{icons.get(kind, '  ')} {msg}")


# ──────────────────────────────────────────────────────
# 資料抓取 + 指標計算
# ──────────────────────────────────────────────────────
def fetch_stock(ticker, months=HISTORY_MONTHS):
    try:
        end = datetime.now()
        start = end - timedelta(days=months * 30)
        df = yf.Ticker(ticker).history(start=start, end=end, auto_adjust=False)
        if df.empty:
            return None
        df = df.reset_index()
        df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
        return df
    except Exception as e:
        log(f"抓 {ticker} 失敗: {e}", "err")
        return None


def compute_indicators(df):
    df = df.copy()
    close = df["Close"]
    df["SMA_20"] = close.rolling(20).mean()
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
    return df


# ──────────────────────────────────────────────────────
# 大盤狀態（給過濾器用）
# ──────────────────────────────────────────────────────
def compute_market_filter(market_df):
    """SPY 計算每日的市場狀態：是否在 SMA200 之上、SMA200 是否上升"""
    market_df = market_df.copy()
    market_df["above_sma200"] = market_df["Close"] > market_df["SMA_200"]
    # SMA200 上升 = 今天的 SMA200 > 5 根前的 SMA200
    market_df["sma200_uptrend"] = market_df["SMA_200"] > market_df["SMA_200"].shift(5)
    market_df["market_ok"] = market_df["above_sma200"] & market_df["sma200_uptrend"]
    return market_df.set_index("Date")[["market_ok", "above_sma200", "sma200_uptrend"]]


def market_ok_at(market_state, date):
    """查指定日期的大盤狀態"""
    if market_state is None or market_state.empty:
        return True  # 無資料時不過濾
    sub = market_state[market_state.index <= date]
    if sub.empty:
        return True
    return bool(sub.iloc[-1]["market_ok"])


# ──────────────────────────────────────────────────────
# 訊號偵測 — 涵蓋 3 種訊號
# ──────────────────────────────────────────────────────
def get_engine_type(ticker):
    trend = {"NVDA", "MSFT", "AAPL", "GOOG", "GOOGL", "AMZN", "META", "TSM",
              "QQQ", "ORCL", "AVGO"}
    momentum = {"AMD", "TSLA", "MU", "INTC", "ASX"}
    growth = {"PLTR", "CRCL", "SOFI", "PYPL"}
    if ticker in trend: return "trend"
    if ticker in momentum: return "momentum"
    if ticker in growth: return "growth"
    return "reversal"


def detect_all_signals(df, ticker, market_state=None):
    """偵測所有起漲訊號（3 種類型）"""
    engine_type = get_engine_type(ticker)
    df = df.copy().reset_index(drop=True)

    # ── 1. MA 金叉（黃金交叉 / 起漲點）──
    df["ma_gold"] = ((df["SMA_20"] > df["SMA_60"]) &
                     (df["SMA_20"].shift(1) <= df["SMA_60"].shift(1)))
    # RSI 20 根最低 < 45（起漲點額外條件）
    df["rsi_was_oversold_20"] = df["RSI"].rolling(20, min_periods=1).min() < 45

    # ── 2. MACD 金叉 ──
    df["macd_gold"] = ((df["MACD"] > df["Signal_Line"]) &
                       (df["MACD"].shift(1) <= df["Signal_Line"].shift(1)))

    all_signals = []
    already_starred_idx_macd = set()
    already_starred_idx_ma = set()

    # ──── 偵測 MA 金叉訊號（黃金交叉 / 起漲點）────
    for idx in df.index[df["ma_gold"]]:
        if idx < 60:
            continue
        curr = df.iloc[idx]
        prior5 = df.iloc[max(0, idx - 5)]
        # 是否同時 RSI 曾超賣 → 升級為「🚀 起漲點」
        is_launch_point = bool(curr["rsi_was_oversold_20"])
        signal_type = "🚀 起漲點" if is_launch_point else "🌕 黃金交叉"

        # 計算過濾條件
        ma60 = curr.get("SMA_60", 0)
        ma60_uptrend = (not pd.isna(ma60) and not pd.isna(prior5.get("SMA_60", np.nan))
                        and ma60 > prior5["SMA_60"])
        above_sma200 = (not pd.isna(curr.get("SMA_200"))
                        and curr["Close"] > curr["SMA_200"])
        market_ok = market_ok_at(market_state, curr["Date"]) if market_state is not None else True

        all_signals.append({
            "date": curr["Date"],
            "idx": idx,
            "close": curr["Close"],
            "signal_type": signal_type,
            "trigger": "MA金叉" + ("+RSI超賣" if is_launch_point else ""),
            "rsi": float(curr.get("RSI", 50)),
            "macd": float(curr.get("MACD", 0)),
            "death_cross": False,  # MA 金叉本身就不在死叉
            "positive_zone": False,  # 不適用
            "above_sma200": above_sma200,
            "ma60_uptrend": ma60_uptrend,
            "market_ok": market_ok,
            "engine_type": engine_type,
        })

    # ──── 偵測 MACD 金叉訊號（⭐ 起漲確認）────
    for idx in df.index[df["macd_gold"]]:
        if idx < 5:
            continue
        curr = df.iloc[idx]
        prior = df.iloc[idx - 1]
        prior2 = df.iloc[max(0, idx - 2)]
        prior5 = df.iloc[max(0, idx - 5)]

        rsi = curr.get("RSI", np.nan)
        if pd.isna(rsi):
            continue

        rsi_recovering = rsi > prior2.get("RSI", rsi)
        ma60 = curr.get("SMA_60", 0)
        near_ma60 = (not pd.isna(ma60) and ma60 > 0
                     and abs(curr["Close"] - ma60) / ma60 <= 0.05)
        deep_reversal = curr["MACD"] < 0
        zero_cross = (curr["MACD"] >= 0 and prior["MACD"] < 0)
        ma20_val = curr.get("SMA_20", np.nan)
        death_cross_active = (not pd.isna(ma20_val) and not pd.isna(ma60)
                              and ma60 > 0 and ma20_val < ma60)

        rsi_w20 = df.iloc[max(0, idx - 20): idx + 1]["RSI"]
        oversold_thr = 35 if engine_type in ("trend", "momentum") else 25
        is_deeply_oversold = rsi_w20.min() < oversold_thr if not rsi_w20.empty else False
        rsi_already_recovering = rsi > prior5.get("RSI", rsi)
        macd_significant = abs(curr["MACD"]) > curr["Close"] * 0.001
        positive_zone_cross = (curr["MACD"] > 0 and curr.get("Signal_Line", -1) > 0)

        if death_cross_active:
            if not macd_significant:
                continue
            if not (is_deeply_oversold and rsi_already_recovering):
                continue

        cluster_window = 3 if positive_zone_cross else 7
        has_nearby = any(0 < idx - prev_idx <= cluster_window
                          for prev_idx in already_starred_idx_macd)
        if has_nearby:
            continue

        uptrend_zero_breakout = (not death_cross_active) and zero_cross
        strong_positive = (not death_cross_active) and positive_zone_cross

        trigger = None
        if strong_positive:
            trigger = "正值區金叉"
        elif uptrend_zero_breakout:
            trigger = "零軸突破"
        elif rsi_recovering and near_ma60:
            trigger = "近MA60+RSI回暖"
        elif rsi_recovering and deep_reversal:
            trigger = "深度反轉(MACD<0)"
        elif rsi_recovering and zero_cross:
            trigger = "零軸+RSI回暖"

        if trigger:
            already_starred_idx_macd.add(idx)
            ma60_uptrend = (not pd.isna(ma60) and not pd.isna(prior5.get("SMA_60", np.nan))
                            and ma60 > prior5["SMA_60"])
            above_sma200 = (not pd.isna(curr.get("SMA_200"))
                            and curr["Close"] > curr["SMA_200"])
            market_ok = market_ok_at(market_state, curr["Date"]) if market_state is not None else True

            all_signals.append({
                "date": curr["Date"],
                "idx": idx,
                "close": curr["Close"],
                "signal_type": "⭐ MACD起漲確認",
                "trigger": trigger,
                "rsi": float(rsi),
                "macd": float(curr["MACD"]),
                "death_cross": death_cross_active,
                "positive_zone": positive_zone_cross,
                "above_sma200": above_sma200,
                "ma60_uptrend": ma60_uptrend,
                "market_ok": market_ok,
                "engine_type": engine_type,
            })

    # 排序 by date
    all_signals.sort(key=lambda s: s["date"])
    return df, all_signals


# ──────────────────────────────────────────────────────
# 新規則：方案 A + MA60 + 大盤過濾
# ──────────────────────────────────────────────────────
def passes_new_rules(signal):
    """方案 A + MA60 + 大盤過濾：判斷一個訊號在新規則下會不會被發出"""
    # ── 砍掉爛條件 ──
    if signal["trigger"] in ("深度反轉(MACD<0)", "近MA60+RSI回暖"):
        return False
    # ── 強制要求 MA60 上升 ──
    if not signal.get("ma60_uptrend"):
        return False
    # ── 大盤過濾器 ──
    if not signal.get("market_ok"):
        return False
    return True


# ──────────────────────────────────────────────────────
# 評估真實表現
# ──────────────────────────────────────────────────────
def evaluate_signal(df, signal):
    idx = signal["idx"]
    entry_price = signal["close"]
    result = {}
    for n in LOOKAHEAD_DAYS:
        future_idx = idx + n
        if future_idx >= len(df):
            result[f"ret_{n}d"] = None
            continue
        future = df.iloc[idx + 1: future_idx + 1]
        if future.empty:
            result[f"ret_{n}d"] = None
            continue
        final = future["Close"].iloc[-1]
        result[f"ret_{n}d"] = round((final / entry_price - 1) * 100, 2)
    return result


def classify(ret_10d):
    if ret_10d is None:
        return ("no_data", "資料不足")
    if ret_10d >= 5: return ("excellent", "✅ 優秀(+5%↑)")
    elif ret_10d >= 2: return ("good", "👍 不錯(+2~5%)")
    elif ret_10d >= -2: return ("flat", "⚖️ 平淡(±2%)")
    elif ret_10d >= -5: return ("poor", "😟 不佳(-2~-5%)")
    else: return ("bad", "❌ 失敗(-5%↓)")


# ──────────────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────────────
def main():
    log("=" * 60, "info")
    log("起漲訊號回測 v2 — 含主圖訊號 + 新規則對照", "info")
    log("=" * 60, "info")

    # 抓 SPY 當大盤過濾器
    log("抓取 SPY 作為大盤過濾器...", "step")
    spy_df = fetch_stock(MARKET_TICKER)
    if spy_df is None:
        log("SPY 抓取失敗，將跳過大盤過濾", "warn")
        market_state = None
    else:
        spy_df = compute_indicators(spy_df)
        market_state = compute_market_filter(spy_df)
        log(f"SPY 處理完成（{len(spy_df)} 筆）", "ok")

    case_tickers = list(set(c["ticker"] for c in FAILURE_CASES))
    extra = ["NVDA", "AAPL", "AMD", "TSM"]
    all_tickers = list(set(case_tickers + extra))
    log(f"將分析 {len(all_tickers)} 支股票", "step")

    all_results = {}
    for i, t in enumerate(all_tickers, 1):
        log(f"[{i}/{len(all_tickers)}] {t}", "info")
        df = fetch_stock(t)
        if df is None or len(df) < 100:
            log(f"  資料不足，跳過", "warn")
            continue
        df = compute_indicators(df)
        df, sigs = detect_all_signals(df, t, market_state)
        for s in sigs:
            s.update(evaluate_signal(df, s))
            s["category"], s["category_label"] = classify(s.get("ret_10d"))
            s["passes_new"] = passes_new_rules(s)
        all_results[t] = {"df": df, "signals": sigs}
        time.sleep(0.2)

    log("訊號偵測完成", "ok")

    # 統計
    all_signals = []
    for t, r in all_results.items():
        for s in r["signals"]:
            s["ticker"] = t
            all_signals.append(s)
    log(f"總訊號數: {len(all_signals)}", "ok")

    # 案例對應（找最近的訊號，不論類型）
    case_matches = []
    for case in FAILURE_CASES:
        t = case["ticker"]
        if t not in all_results:
            case_matches.append({**case, "matched_signal": None})
            continue
        target_date = pd.to_datetime(case["date"])
        matched = None
        best_diff = 10
        for s in all_results[t]["signals"]:
            sd = pd.to_datetime(s["date"])
            diff = abs((sd - target_date).days)
            if diff < best_diff:
                best_diff = diff
                matched = s
        case_matches.append({
            **case, "matched_signal": matched,
            "match_diff_days": best_diff if matched else None,
        })

    # 產報告
    log("產生 HTML 報告...", "step")
    html = generate_html(all_signals, case_matches, all_results)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(html)
    log(f"完成：{os.path.abspath(OUTPUT_FILE)}", "ok")


def calc_stats(signals):
    valid = [s for s in signals if s.get("ret_10d") is not None]
    if not valid:
        return {"count": 0, "win_rate": "—", "avg": "—", "median": "—", "worst": "—",
                "win_rate_num": 0}
    rets = [s["ret_10d"] for s in valid]
    wins = sum(1 for r in rets if r > 0)
    win_rate_num = wins / len(valid) * 100
    return {
        "count": len(valid),
        "win_rate": f"{win_rate_num:.1f}%",
        "win_rate_num": win_rate_num,
        "avg": f"{np.mean(rets):+.2f}%",
        "median": f"{np.median(rets):+.2f}%",
        "worst": f"{min(rets):+.2f}%",
    }


def generate_html(all_signals, case_matches, all_results):
    old_all = all_signals
    new_all = [s for s in all_signals if s["passes_new"]]

    old_stats = calc_stats(old_all)
    new_stats = calc_stats(new_all)

    # 各訊號類型的舊 vs 新
    by_type_html = ""
    for sig_type in ["🌕 黃金交叉", "🚀 起漲點", "⭐ MACD起漲確認"]:
        old_t = [s for s in old_all if s["signal_type"] == sig_type]
        new_t = [s for s in new_all if s["signal_type"] == sig_type]
        os_ = calc_stats(old_t)
        ns = calc_stats(new_t)
        # 顏色
        new_color = "#22c55e" if ns["win_rate_num"] >= 60 else "#facc15" if ns["win_rate_num"] >= 50 else "#ef4444"
        by_type_html += f"""
        <tr>
          <td><b>{sig_type}</b></td>
          <td>{os_['count']}</td>
          <td>{os_['win_rate']}</td>
          <td>{os_['avg']}</td>
          <td>→</td>
          <td>{ns['count']}</td>
          <td style="color:{new_color};font-weight:bold;">{ns['win_rate']}</td>
          <td>{ns['avg']}</td>
        </tr>"""

    # 各觸發條件
    triggers = {}
    for s in old_all:
        triggers.setdefault(s["trigger"], {"old": [], "new": []})["old"].append(s)
    for s in new_all:
        triggers.setdefault(s["trigger"], {"old": [], "new": []})["new"].append(s)

    trigger_html = ""
    for trig, d in sorted(triggers.items(), key=lambda x: -len(x[1]["old"])):
        os_ = calc_stats(d["old"])
        ns_ = calc_stats(d["new"])
        # 是否被新規則砍掉
        cut = trig in ("深度反轉(MACD<0)", "近MA60+RSI回暖")
        cut_label = " 🚫(已砍)" if cut else ""
        old_color = "#22c55e" if os_["win_rate_num"] >= 60 else "#facc15" if os_["win_rate_num"] >= 50 else "#ef4444"
        trigger_html += f"""
        <tr>
          <td><b>{trig}{cut_label}</b></td>
          <td>{os_['count']}</td>
          <td style="color:{old_color};">{os_['win_rate']}</td>
          <td>{os_['avg']}</td>
          <td>→</td>
          <td>{ns_['count']}</td>
          <td>{ns_['win_rate']}</td>
          <td>{ns_['avg']}</td>
        </tr>"""

    # 失敗案例對應（顯示是否會被新規則濾掉）
    case_rows = ""
    for c in case_matches:
        m = c.get("matched_signal")
        case_type = "🔴 假起漲" if c["type"] == "false_launch" else "🟡 漏標起漲"
        if m:
            sig_type = m["signal_type"]
            trig = m["trigger"]
            r5 = f"{m['ret_5d']:+.2f}%" if m.get("ret_5d") is not None else "—"
            r10 = f"{m['ret_10d']:+.2f}%" if m.get("ret_10d") is not None else "—"
            r20 = f"{m['ret_20d']:+.2f}%" if m.get("ret_20d") is not None else "—"
            cat = m.get("category_label", "—")
            new_pass = m.get("passes_new", False)
            new_pass_html = (
                '<span style="color:#ef4444;font-weight:bold;">❌ 新規則砍掉</span>'
                if c["type"] == "false_launch" and not new_pass else
                '<span style="color:#22c55e;font-weight:bold;">✅ 新規則保留</span>'
                if c["type"] == "missed_launch" and new_pass else
                '<span style="color:#888;">' + ("⚠️ 仍會發" if new_pass else "⚠️ 仍漏掉") + '</span>'
            )
            md = pd.to_datetime(m["date"]).strftime("%Y-%m-%d")
            case_rows += f"""
            <tr>
              <td>{c['ticker']}</td>
              <td>{c['date']}</td>
              <td>{case_type}</td>
              <td>{md}</td>
              <td>{sig_type}</td>
              <td>{trig}</td>
              <td>{r5}</td>
              <td>{r10}</td>
              <td>{r20}</td>
              <td>{cat}</td>
              <td>{new_pass_html}</td>
            </tr>"""
        else:
            case_rows += f"""
            <tr>
              <td>{c['ticker']}</td>
              <td>{c['date']}</td>
              <td>{case_type}</td>
              <td colspan="8" style="color:#888;">(無對應訊號 — 三種類型都沒找到)</td>
            </tr>"""

    # 分類分布對比
    dist_old = {}; dist_new = {}
    for s in old_all:
        dist_old[s["category_label"]] = dist_old.get(s["category_label"], 0) + 1
    for s in new_all:
        dist_new[s["category_label"]] = dist_new.get(s["category_label"], 0) + 1

    dist_html = ""
    cats = sorted(set(list(dist_old.keys()) + list(dist_new.keys())))
    for cat in cats:
        if cat == "資料不足":
            continue
        o = dist_old.get(cat, 0); n = dist_new.get(cat, 0)
        o_pct = o/len(old_all)*100 if old_all else 0
        n_pct = n/len(new_all)*100 if new_all else 0
        dist_html += f"""
        <tr><td>{cat}</td>
        <td>{o} ({o_pct:.1f}%)</td><td>→</td>
        <td>{n} ({n_pct:.1f}%)</td></tr>"""

    # 改善摘要
    diff_win = new_stats["win_rate_num"] - old_stats["win_rate_num"]
    diff_color = "#22c55e" if diff_win > 0 else "#ef4444"

    html = f"""<!DOCTYPE html>
<html lang="zh-Hant">
<head>
<meta charset="UTF-8">
<title>起漲訊號回測 v2 — 規則對照</title>
<style>
  body {{ background:#0e0e10; color:#e5e7eb; font-family:-apple-system,sans-serif;
         margin:0; padding:24px; max-width:1300px; margin:0 auto; }}
  h1, h2, h3 {{ color:#facc15; }}
  h1 {{ border-bottom:2px solid #facc15; padding-bottom:12px; }}
  table {{ width:100%; border-collapse:collapse; margin:12px 0;
           background:#1a1a1c; border-radius:8px; overflow:hidden; font-size:14px; }}
  th, td {{ padding:10px 12px; text-align:left; border-bottom:1px solid #2a2a2c; }}
  th {{ background:#2a2a2c; color:#facc15; font-weight:bold; }}
  tr:hover {{ background:#1f1f21; }}
  .compare-card {{ display:grid; grid-template-columns:1fr auto 1fr; gap:16px; }}
  .stat-old, .stat-new {{ background:#1a1a1c; padding:18px; border-radius:8px; }}
  .stat-old {{ border-left:4px solid #ef4444; }}
  .stat-new {{ border-left:4px solid #22c55e; }}
  .arrow {{ font-size:32px; align-self:center; text-align:center; }}
  .verdict {{ background:linear-gradient(135deg,#1a1a1c,#1a2a1c);
              padding:20px; border-radius:10px; border:2px solid {diff_color};
              margin:20px 0; }}
  .big {{ font-size:36px; font-weight:bold; color:#facc15; }}
</style>
</head>
<body>

<h1>🎯 起漲訊號回測 v2 — 新舊規則對照</h1>
<p style="color:#888;">產出時間：{datetime.now().strftime('%Y-%m-%d %H:%M')} | 股票數：{len(all_results)} | 含 SPY 大盤過濾器</p>

<div class="verdict">
  <h2 style="margin-top:0;">📊 新規則 vs 舊規則：總體表現</h2>
  <div class="compare-card">
    <div class="stat-old">
      <div style="color:#ef4444;font-weight:bold;">舊規則（現況）</div>
      <div style="margin-top:8px;">
        <div>訊號數：<b>{old_stats['count']}</b></div>
        <div>勝率：<span class="big" style="font-size:28px;">{old_stats['win_rate']}</span></div>
        <div>平均報酬：{old_stats['avg']}</div>
        <div>最差：<span style="color:#ef4444;">{old_stats['worst']}</span></div>
      </div>
    </div>
    <div class="arrow">→</div>
    <div class="stat-new">
      <div style="color:#22c55e;font-weight:bold;">新規則（方案 A）</div>
      <div style="margin-top:8px;">
        <div>訊號數：<b>{new_stats['count']}</b>（減少 {int((1-new_stats['count']/max(old_stats['count'],1))*100)}%）</div>
        <div>勝率：<span class="big" style="font-size:28px;color:{diff_color};">{new_stats['win_rate']}</span> <span style="color:{diff_color};">({diff_win:+.1f}%)</span></div>
        <div>平均報酬：{new_stats['avg']}</div>
        <div>最差：{new_stats['worst']}</div>
      </div>
    </div>
  </div>
  <p style="margin-top:14px;color:#aaa;">新規則：砍「深度反轉」「近MA60+RSI回暖」+ 強制要求 MA60 上升 + 大盤過濾器（SPY 站上 SMA200 且 SMA200 上升）</p>
</div>

<h2>🔍 各訊號類型對比</h2>
<table>
  <tr>
    <th>訊號類型</th>
    <th colspan="3" style="text-align:center;background:#3a1a1a;">舊規則</th>
    <th></th>
    <th colspan="3" style="text-align:center;background:#1a3a1a;">新規則</th>
  </tr>
  <tr>
    <th></th><th>數</th><th>勝率</th><th>平均</th><th></th>
    <th>數</th><th>勝率</th><th>平均</th>
  </tr>
  {by_type_html}
</table>

<h2>📈 各觸發條件對比</h2>
<table>
  <tr>
    <th>觸發條件</th>
    <th colspan="3" style="text-align:center;background:#3a1a1a;">舊規則</th>
    <th></th>
    <th colspan="3" style="text-align:center;background:#1a3a1a;">新規則</th>
  </tr>
  <tr>
    <th></th><th>數</th><th>勝率</th><th>平均</th><th></th>
    <th>數</th><th>勝率</th><th>平均</th>
  </tr>
  {trigger_html}
</table>

<h2>🎚️ 訊號品質分布對比</h2>
<table>
  <tr><th>分類</th><th>舊規則</th><th></th><th>新規則</th></tr>
  {dist_html}
</table>

<h2>📋 失敗案例對應（看新規則能否解決）</h2>
<p style="color:#888;">每個失敗案例後面標註「新規則砍掉」表示問題已修復 ✅</p>
<table>
  <tr>
    <th>股票</th><th>您日期</th><th>類型</th><th>對應日</th>
    <th>訊號類型</th><th>觸發</th>
    <th>+5日</th><th>+10日</th><th>+20日</th><th>分類</th>
    <th>新規則處理</th>
  </tr>
  {case_rows}
</table>

</body>
</html>
"""
    return html


if __name__ == "__main__":
    main()
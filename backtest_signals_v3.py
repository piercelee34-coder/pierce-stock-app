#!/usr/bin/env python3
"""backtest_signals_v3.py — 起漲 + 下車訊號雙向回測

新增 v3:
  1. 方案 A+（精準化）：近MA60+RSI回暖加嚴保留
  2. 下車訊號偵測（exit signals）
"""
import os, sys, json, warnings, time
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

LOOKAHEAD_DAYS = [5, 10, 20]
HISTORY_MONTHS = 18
OUTPUT_FILE = "backtest_signals_v3_report.html"
MARKET_TICKER = "SPY"

FAILURE_CASES = [
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
    {"ticker": "META",  "date": "2026-04-07", "type": "missed_launch"},
    {"ticker": "TSLA",  "date": "2025-11-25", "type": "missed_launch"},
    {"ticker": "PLTR",  "date": "2025-12-01", "type": "missed_launch"},
    {"ticker": "SNDK",  "date": "2025-12-08", "type": "missed_launch"},
    {"ticker": "RKLB",  "date": "2026-04-06", "type": "missed_launch"},
]


def log(msg, kind="info"):
    icons = {"info": "🔹", "ok": "✅", "warn": "⚠️ ", "err": "❌"}
    print(f"{icons.get(kind, '  ')} {msg}")


def fetch_stock(ticker, months=HISTORY_MONTHS):
    try:
        end = datetime.now()
        start = end - timedelta(days=months * 30)
        df = yf.Ticker(ticker).history(start=start, end=end, auto_adjust=False)
        if df.empty: return None
        df = df.reset_index()
        df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
        return df
    except Exception:
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
    df["Vol_20"] = df["Volume"].rolling(20).mean()
    df["Vol_60"] = df["Volume"].rolling(60).mean()
    df["Vol_5"] = df["Volume"].rolling(5).mean()
    return df


def compute_market_filter(market_df):
    market_df = market_df.copy()
    market_df["above_sma200"] = market_df["Close"] > market_df["SMA_200"]
    market_df["sma200_uptrend"] = market_df["SMA_200"] > market_df["SMA_200"].shift(5)
    market_df["market_ok"] = market_df["above_sma200"] & market_df["sma200_uptrend"]
    return market_df.set_index("Date")[["market_ok"]]


def market_ok_at(market_state, date):
    if market_state is None or market_state.empty: return True
    sub = market_state[market_state.index <= date]
    if sub.empty: return True
    return bool(sub.iloc[-1]["market_ok"])


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
    engine_type = get_engine_type(ticker)
    df = df.copy().reset_index(drop=True)
    df["ma_gold"] = ((df["SMA_20"] > df["SMA_60"]) &
                     (df["SMA_20"].shift(1) <= df["SMA_60"].shift(1)))
    df["rsi_was_oversold_20"] = df["RSI"].rolling(20, min_periods=1).min() < 45
    df["macd_gold"] = ((df["MACD"] > df["Signal_Line"]) &
                       (df["MACD"].shift(1) <= df["Signal_Line"].shift(1)))

    all_signals = []
    already_starred_idx_macd = set()

    for idx in df.index[df["ma_gold"]]:
        if idx < 60: continue
        curr = df.iloc[idx]
        prior5 = df.iloc[max(0, idx - 5)]
        is_launch_point = bool(curr["rsi_was_oversold_20"])
        signal_type = "🚀 起漲點" if is_launch_point else "🌕 黃金交叉"
        ma60 = curr.get("SMA_60", 0)
        ma60_uptrend = (not pd.isna(ma60) and not pd.isna(prior5.get("SMA_60", np.nan))
                        and ma60 > prior5["SMA_60"])
        above_sma200 = (not pd.isna(curr.get("SMA_200")) and curr["Close"] > curr["SMA_200"])
        market_ok = market_ok_at(market_state, curr["Date"]) if market_state is not None else True
        vol_increasing = (not pd.isna(curr.get("Vol_20")) and not pd.isna(curr.get("Vol_60"))
                          and curr["Vol_20"] > curr["Vol_60"])
        rsi_w20 = df.iloc[max(0, idx - 20): idx + 1]["RSI"]
        rsi_min_20d = rsi_w20.min() if not rsi_w20.empty else 50
        rsi_real_reversal = (rsi_min_20d <= 30 and curr["RSI"] >= 45)

        all_signals.append({
            "date": curr["Date"], "idx": idx, "close": curr["Close"],
            "signal_type": signal_type,
            "trigger": "MA金叉+RSI超賣" if is_launch_point else "MA金叉",
            "rsi": float(curr.get("RSI", 50)),
            "above_sma200": above_sma200, "ma60_uptrend": ma60_uptrend,
            "market_ok": market_ok, "engine_type": engine_type,
            "vol_increasing": vol_increasing,
            "rsi_real_reversal": rsi_real_reversal,
            "rsi_min_20d": float(rsi_min_20d),
        })

    for idx in df.index[df["macd_gold"]]:
        if idx < 5: continue
        curr = df.iloc[idx]
        prior = df.iloc[idx - 1]
        prior2 = df.iloc[max(0, idx - 2)]
        prior5 = df.iloc[max(0, idx - 5)]
        rsi = curr.get("RSI", np.nan)
        if pd.isna(rsi): continue
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
        rsi_min_20d = rsi_w20.min() if not rsi_w20.empty else 50
        oversold_thr = 35 if engine_type in ("trend", "momentum") else 25
        is_deeply_oversold = rsi_min_20d < oversold_thr
        rsi_already_recovering = rsi > prior5.get("RSI", rsi)
        macd_significant = abs(curr["MACD"]) > curr["Close"] * 0.001
        positive_zone_cross = (curr["MACD"] > 0 and curr.get("Signal_Line", -1) > 0)

        if death_cross_active:
            if not macd_significant: continue
            if not (is_deeply_oversold and rsi_already_recovering): continue

        cluster_window = 3 if positive_zone_cross else 7
        has_nearby = any(0 < idx - prev_idx <= cluster_window
                          for prev_idx in already_starred_idx_macd)
        if has_nearby: continue

        uptrend_zero_breakout = (not death_cross_active) and zero_cross
        strong_positive = (not death_cross_active) and positive_zone_cross
        trigger = None
        if strong_positive: trigger = "正值區金叉"
        elif uptrend_zero_breakout: trigger = "零軸突破"
        elif rsi_recovering and near_ma60: trigger = "近MA60+RSI回暖"
        elif rsi_recovering and deep_reversal: trigger = "深度反轉(MACD<0)"
        elif rsi_recovering and zero_cross: trigger = "零軸+RSI回暖"

        if trigger:
            already_starred_idx_macd.add(idx)
            ma60_uptrend = (not pd.isna(ma60) and not pd.isna(prior5.get("SMA_60", np.nan))
                            and ma60 > prior5["SMA_60"])
            above_sma200 = (not pd.isna(curr.get("SMA_200")) and curr["Close"] > curr["SMA_200"])
            market_ok = market_ok_at(market_state, curr["Date"]) if market_state is not None else True
            vol_increasing = (not pd.isna(curr.get("Vol_20")) and not pd.isna(curr.get("Vol_60"))
                              and curr["Vol_20"] > curr["Vol_60"])
            rsi_real_reversal = (rsi_min_20d <= 30 and rsi >= 45)

            all_signals.append({
                "date": curr["Date"], "idx": idx, "close": curr["Close"],
                "signal_type": "⭐ MACD起漲確認", "trigger": trigger,
                "rsi": float(rsi),
                "above_sma200": above_sma200, "ma60_uptrend": ma60_uptrend,
                "market_ok": market_ok, "engine_type": engine_type,
                "vol_increasing": vol_increasing,
                "rsi_real_reversal": rsi_real_reversal,
                "rsi_min_20d": float(rsi_min_20d),
            })

    all_signals.sort(key=lambda s: s["date"])
    return df, all_signals


def detect_exit_signals(df, ticker):
    """偵測下車訊號 4 條件"""
    df = df.copy().reset_index(drop=True)
    df["macd_dead"] = ((df["MACD"] < df["Signal_Line"]) &
                       (df["MACD"].shift(1) >= df["Signal_Line"].shift(1)))
    exit_signals = []
    already_marked = set()

    for idx in df.index[df["macd_dead"]]:
        if idx < 20: continue
        curr = df.iloc[idx]
        if pd.isna(curr.get("RSI")) or pd.isna(curr.get("SMA_20")):
            continue

        sig_1_macd_death = True
        rsi_w10 = df.iloc[max(0, idx - 10): idx + 1]["RSI"]
        was_overbought = rsi_w10.max() > 70 if not rsi_w10.empty else False
        sig_2_rsi_cooldown = was_overbought and curr["RSI"] < 60

        vol_5 = curr.get("Vol_5", np.nan)
        vol_20 = curr.get("Vol_20", np.nan)
        sig_3_vol_shrink = (not pd.isna(vol_5) and not pd.isna(vol_20)
                             and vol_5 < vol_20 * 0.8)
        sig_4_below_sma20 = curr["Close"] < curr["SMA_20"]

        signals_bool = [sig_1_macd_death, sig_2_rsi_cooldown,
                          sig_3_vol_shrink, sig_4_below_sma20]
        n_hit = sum(signals_bool)
        if n_hit < 2: continue

        has_nearby = any(0 < idx - p < 7 for p in already_marked)
        if has_nearby: continue
        already_marked.add(idx)

        if n_hit == 4: level = "🔴 強烈下車"
        elif n_hit == 3: level = "🟠 中度下車"
        else: level = "🟡 觀察下車"

        exit_signals.append({
            "date": curr["Date"], "idx": idx, "close": curr["Close"],
            "n_hit": n_hit, "level": level, "signals": signals_bool,
            "rsi": float(curr["RSI"]),
        })

    return exit_signals


def passes_new_rules_aplus(signal):
    if signal["trigger"] == "深度反轉(MACD<0)": return False
    if signal["trigger"] == "近MA60+RSI回暖":
        if not signal.get("rsi_real_reversal"): return False
        if not signal.get("vol_increasing"): return False
    if not signal.get("ma60_uptrend"): return False
    if not signal.get("market_ok"): return False
    return True


def passes_new_rules_a(signal):
    if signal["trigger"] in ("深度反轉(MACD<0)", "近MA60+RSI回暖"): return False
    if not signal.get("ma60_uptrend"): return False
    if not signal.get("market_ok"): return False
    return True


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


def classify_launch(ret_10d):
    if ret_10d is None: return ("no_data", "資料不足")
    if ret_10d >= 5: return ("excellent", "✅ 優秀(+5%↑)")
    elif ret_10d >= 2: return ("good", "👍 不錯(+2~5%)")
    elif ret_10d >= -2: return ("flat", "⚖️ 平淡(±2%)")
    elif ret_10d >= -5: return ("poor", "😟 不佳(-2~-5%)")
    else: return ("bad", "❌ 失敗(-5%↓)")


def classify_exit(ret_10d):
    if ret_10d is None: return ("no_data", "資料不足")
    if ret_10d <= -5: return ("excellent", "✅ 完美下車(-5%↓)")
    elif ret_10d <= -2: return ("good", "👍 好下車(-2~-5%)")
    elif ret_10d <= 2: return ("flat", "⚖️ 平淡(±2%)")
    elif ret_10d <= 5: return ("poor", "😟 假警報(+2~5%)")
    else: return ("bad", "❌ 大錯下車(+5%↑)")


def main():
    log("=" * 60)
    log("起漲 + 下車訊號雙向回測 v3")
    log("=" * 60)

    log("抓取 SPY 大盤過濾...", "info")
    spy_df = fetch_stock(MARKET_TICKER)
    if spy_df is None:
        log("SPY 失敗", "warn")
        market_state = None
    else:
        spy_df = compute_indicators(spy_df)
        market_state = compute_market_filter(spy_df)
        log(f"SPY 完成", "ok")

    case_tickers = list(set(c["ticker"] for c in FAILURE_CASES))
    extra = ["NVDA", "AAPL", "AMD", "TSM"]
    all_tickers = list(set(case_tickers + extra))
    log(f"分析 {len(all_tickers)} 支股票", "info")

    all_results = {}
    for i, t in enumerate(all_tickers, 1):
        log(f"[{i}/{len(all_tickers)}] {t}")
        df = fetch_stock(t)
        if df is None or len(df) < 100:
            continue
        df = compute_indicators(df)
        df, launch_sigs = detect_all_signals(df, t, market_state)
        for s in launch_sigs:
            s.update(evaluate_signal(df, s))
            s["category"], s["category_label"] = classify_launch(s.get("ret_10d"))
            s["passes_a"] = passes_new_rules_a(s)
            s["passes_aplus"] = passes_new_rules_aplus(s)
        exit_sigs = detect_exit_signals(df, t)
        for s in exit_sigs:
            s.update(evaluate_signal(df, s))
            s["category"], s["category_label"] = classify_exit(s.get("ret_10d"))
        all_results[t] = {"df": df, "launch": launch_sigs, "exit": exit_sigs}
        time.sleep(0.2)

    log("訊號偵測完成", "ok")

    all_launch = []
    all_exit = []
    for t, r in all_results.items():
        for s in r["launch"]:
            s["ticker"] = t
            all_launch.append(s)
        for s in r["exit"]:
            s["ticker"] = t
            all_exit.append(s)

    log(f"起漲訊號 {len(all_launch)} 個 / 下車訊號 {len(all_exit)} 個", "ok")

    case_matches = []
    for case in FAILURE_CASES:
        t = case["ticker"]
        if t not in all_results:
            case_matches.append({**case, "matched_signal": None})
            continue
        target_date = pd.to_datetime(case["date"])
        matched = None
        best_diff = 10
        for s in all_results[t]["launch"]:
            sd = pd.to_datetime(s["date"])
            diff = abs((sd - target_date).days)
            if diff < best_diff:
                best_diff = diff
                matched = s
        case_matches.append({
            **case, "matched_signal": matched,
            "match_diff_days": best_diff if matched else None,
        })

    log("產生 HTML 報告...", "info")
    html = generate_html(all_launch, all_exit, case_matches, all_results)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(html)
    log(f"完成：{os.path.abspath(OUTPUT_FILE)}", "ok")


def calc_stats(signals, win_dir="up"):
    valid = [s for s in signals if s.get("ret_10d") is not None]
    if not valid:
        return {"count": 0, "win_rate": "—", "avg": "—", "worst": "—", "win_num": 0}
    rets = [s["ret_10d"] for s in valid]
    if win_dir == "up":
        wins = sum(1 for r in rets if r > 0)
    else:
        wins = sum(1 for r in rets if r < 0)
    win_num = wins / len(valid) * 100
    return {
        "count": len(valid), "win_rate": f"{win_num:.1f}%", "win_num": win_num,
        "avg": f"{np.mean(rets):+.2f}%",
        "worst": f"{min(rets) if win_dir=='up' else max(rets):+.2f}%",
    }


def generate_html(all_launch, all_exit, case_matches, all_results):
    old_l = all_launch
    a_l = [s for s in all_launch if s["passes_a"]]
    aplus_l = [s for s in all_launch if s["passes_aplus"]]
    s_old = calc_stats(old_l)
    s_a = calc_stats(a_l)
    s_aplus = calc_stats(aplus_l)

    exit_44 = [s for s in all_exit if s["n_hit"] == 4]
    exit_33 = [s for s in all_exit if s["n_hit"] == 3]
    exit_22 = [s for s in all_exit if s["n_hit"] == 2]
    s_e44 = calc_stats(exit_44, win_dir="down")
    s_e33 = calc_stats(exit_33, win_dir="down")
    s_e22 = calc_stats(exit_22, win_dir="down")

    nm_old = [s for s in old_l if s["trigger"] == "近MA60+RSI回暖"]
    nm_aplus = [s for s in aplus_l if s["trigger"] == "近MA60+RSI回暖"]
    s_nm_old = calc_stats(nm_old)
    s_nm_aplus = calc_stats(nm_aplus)

    case_rows = ""
    for c in case_matches:
        m = c.get("matched_signal")
        case_type = "🔴 假起漲" if c["type"] == "false_launch" else "🟡 漏標"
        if m:
            r10 = f"{m['ret_10d']:+.2f}%" if m.get("ret_10d") is not None else "—"
            cat = m.get("category_label", "—")
            md = pd.to_datetime(m["date"]).strftime("%Y-%m-%d")
            a_pass = m.get("passes_a")
            aplus_pass = m.get("passes_aplus")
            is_false = c["type"] == "false_launch"
            def label(p, isf):
                if isf:
                    return '<span style="color:#22c55e;">✅ 砍掉</span>' if not p else '<span style="color:#facc15;">⚠️ 仍會發</span>'
                else:
                    return '<span style="color:#22c55e;">✅ 保留</span>' if p else '<span style="color:#ef4444;">❌ 仍漏</span>'
            case_rows += f"""<tr>
              <td>{c['ticker']}</td><td>{c['date']}</td><td>{case_type}</td>
              <td>{md}</td><td>{m['signal_type']}</td><td>{m['trigger']}</td>
              <td>{r10}</td><td>{cat}</td>
              <td>{label(a_pass, is_false)}</td><td>{label(aplus_pass, is_false)}</td></tr>"""
        else:
            case_rows += f"""<tr><td>{c['ticker']}</td><td>{c['date']}</td>
              <td>{case_type}</td><td colspan="7" style="color:#888;">(回測未抓到)</td></tr>"""

    exit_examples = sorted(all_exit, key=lambda x: -x["n_hit"])[:15]
    exit_rows = ""
    for s in exit_examples:
        r5 = f"{s.get('ret_5d', 0):+.2f}%" if s.get("ret_5d") is not None else "—"
        r10 = f"{s.get('ret_10d', 0):+.2f}%" if s.get("ret_10d") is not None else "—"
        r20 = f"{s.get('ret_20d', 0):+.2f}%" if s.get("ret_20d") is not None else "—"
        cat = s.get("category_label", "—")
        date_str = pd.to_datetime(s["date"]).strftime("%Y-%m-%d")
        exit_rows += f"""<tr><td>{s['ticker']}</td><td>{date_str}</td>
          <td>{s['level']}</td><td>{s['n_hit']}/4</td>
          <td>{r5}</td><td>{r10}</td><td>{r20}</td><td>{cat}</td></tr>"""

    html = f"""<!DOCTYPE html><html lang="zh-Hant"><head><meta charset="UTF-8">
<title>v3 雙向回測</title><style>
  body {{ background:#0e0e10; color:#e5e7eb; font-family:-apple-system,sans-serif;
         margin:0; padding:24px; max-width:1300px; margin:0 auto; }}
  h1, h2 {{ color:#facc15; }}
  h1 {{ border-bottom:2px solid #facc15; padding-bottom:12px; }}
  table {{ width:100%; border-collapse:collapse; margin:12px 0;
           background:#1a1a1c; border-radius:8px; overflow:hidden; font-size:14px; }}
  th, td {{ padding:10px 12px; text-align:left; border-bottom:1px solid #2a2a2c; }}
  th {{ background:#2a2a2c; color:#facc15; }}
  .card-row {{ display:grid; grid-template-columns:repeat(3,1fr); gap:14px; margin:16px 0; }}
  .card {{ background:#1a1a1c; padding:18px; border-radius:8px; border-left:4px solid #888; }}
  .card.old {{ border-left-color:#ef4444; }}
  .card.a {{ border-left-color:#facc15; }}
  .card.aplus {{ border-left-color:#22c55e; }}
  .big {{ font-size:32px; font-weight:bold; color:#facc15; }}
</style></head><body>

<h1>🎯 起漲 + 下車訊號雙向回測 v3</h1>
<p style="color:#888;">產出：{datetime.now().strftime('%Y-%m-%d %H:%M')} | 股票數：{len(all_results)} | 含 SPY 大盤過濾器</p>

<h2>🚀 起漲訊號：三方案大比拼</h2>
<div class="card-row">
  <div class="card old"><div style="color:#ef4444;font-weight:bold;">❌ 舊規則</div>
    <div style="margin-top:6px;">數：<b>{s_old['count']}</b></div>
    <div>勝率：<span class="big" style="font-size:24px;">{s_old['win_rate']}</span></div>
    <div>平均：{s_old['avg']}</div></div>
  <div class="card a"><div style="color:#facc15;font-weight:bold;">⚡ 方案 A</div>
    <div style="margin-top:6px;">數：<b>{s_a['count']}</b>（-{int((1-s_a['count']/max(s_old['count'],1))*100)}%）</div>
    <div>勝率：<span class="big" style="font-size:24px;">{s_a['win_rate']}</span></div>
    <div>平均：{s_a['avg']}</div></div>
  <div class="card aplus"><div style="color:#22c55e;font-weight:bold;">✨ 方案 A+（精準化）</div>
    <div style="margin-top:6px;">數：<b>{s_aplus['count']}</b>（-{int((1-s_aplus['count']/max(s_old['count'],1))*100)}%）</div>
    <div>勝率：<span class="big" style="font-size:24px;color:#22c55e;">{s_aplus['win_rate']}</span></div>
    <div>平均：{s_aplus['avg']}</div></div>
</div>
<p style="color:#aaa;">方案 A+：保留「近MA60+RSI回暖」但加嚴（RSI 真實反轉 30→45 + 量增 + MA60上升 + 大盤OK）</p>

<h2>🔬 「近MA60+RSI回暖」加嚴效果</h2>
<table>
  <tr><th>方案</th><th>訊號數</th><th>勝率</th><th>平均報酬</th></tr>
  <tr><td>舊規則（全保留）</td><td>{s_nm_old['count']}</td><td>{s_nm_old['win_rate']}</td><td>{s_nm_old['avg']}</td></tr>
  <tr><td>方案 A（砍掉）</td><td>0</td><td>—</td><td>—</td></tr>
  <tr><td>方案 A+（加嚴保留）</td><td>{s_nm_aplus['count']}</td><td><b style="color:#22c55e;">{s_nm_aplus['win_rate']}</b></td><td>{s_nm_aplus['avg']}</td></tr>
</table>

<h2>🔻 下車訊號回測</h2>
<p style="color:#888;">條件：MACD死叉 + RSI過熱回吐(>70→<60) + 量縮 + 跌破SMA20。**勝率=10日跌幅<0**</p>
<div class="card-row">
  <div class="card" style="border-left-color:#dc2626;">
    <div style="color:#dc2626;font-weight:bold;">🔴 強烈下車 (4/4)</div>
    <div style="margin-top:6px;">數：<b>{s_e44['count']}</b></div>
    <div>命中率：<span class="big" style="font-size:24px;color:#dc2626;">{s_e44['win_rate']}</span></div>
    <div>平均跌幅：{s_e44['avg']}</div></div>
  <div class="card" style="border-left-color:#f97316;">
    <div style="color:#f97316;font-weight:bold;">🟠 中度下車 (3/4)</div>
    <div style="margin-top:6px;">數：<b>{s_e33['count']}</b></div>
    <div>命中率：<span class="big" style="font-size:24px;color:#f97316;">{s_e33['win_rate']}</span></div>
    <div>平均跌幅：{s_e33['avg']}</div></div>
  <div class="card" style="border-left-color:#facc15;">
    <div style="color:#facc15;font-weight:bold;">🟡 觀察 (2/4)</div>
    <div style="margin-top:6px;">數：<b>{s_e22['count']}</b></div>
    <div>命中率：<span class="big" style="font-size:24px;">{s_e22['win_rate']}</span></div>
    <div>平均跌幅：{s_e22['avg']}</div></div>
</div>

<h3>📋 下車訊號代表案例</h3>
<table>
  <tr><th>股票</th><th>日期</th><th>等級</th><th>命中</th>
    <th>+5日</th><th>+10日</th><th>+20日</th><th>分類</th></tr>
  {exit_rows}
</table>

<h2>📋 失敗案例對應（A vs A+）</h2>
<table>
  <tr><th>股票</th><th>您日期</th><th>類型</th><th>對應日</th>
    <th>訊號類型</th><th>觸發</th><th>+10日</th><th>分類</th>
    <th>方案 A</th><th>方案 A+</th></tr>
  {case_rows}
</table>

</body></html>
"""
    return html


if __name__ == "__main__":
    main()
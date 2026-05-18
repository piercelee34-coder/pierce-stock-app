#!/usr/bin/env python3
"""backtest_signals_v4.py — 起漲 + 新下車訊號 + 7維推演"""
import os, sys, json, warnings, time
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

LOOKAHEAD_DAYS = [5, 10, 20]
HISTORY_MONTHS = 18
OUTPUT_FILE = "backtest_signals_v4_report.html"
MARKET_TICKER = "SPY"

FAILURE_CASES = [
    {"ticker": "ONDS", "date": "2026-05-01", "type": "false_launch"},
    {"ticker": "META", "date": "2026-04-28", "type": "false_launch"},
    {"ticker": "META", "date": "2026-03-04", "type": "false_launch"},
    {"ticker": "GLW",  "date": "2026-04-24", "type": "false_launch"},
    {"ticker": "QQQ",  "date": "2026-01-27", "type": "false_launch"},
    {"ticker": "QQQ",  "date": "2026-02-25", "type": "false_launch"},
    {"ticker": "CRCL", "date": "2025-12-22", "type": "false_launch"},
    {"ticker": "TSLA", "date": "2026-02-25", "type": "false_launch"},
    {"ticker": "TSLA", "date": "2026-03-10", "type": "false_launch"},
    {"ticker": "RXRX", "date": "2026-01-22", "type": "false_launch"},
    {"ticker": "RXRX", "date": "2026-05-01", "type": "false_launch"},
    {"ticker": "GOOG", "date": "2026-01-28", "type": "false_launch"},
    {"ticker": "GOOG", "date": "2026-03-10", "type": "false_launch"},
    {"ticker": "MU",   "date": "2026-03-16", "type": "false_launch"},
    {"ticker": "MSFT", "date": "2026-01-27", "type": "false_launch"},
    {"ticker": "MSFT", "date": "2026-01-07", "type": "false_launch"},
    {"ticker": "MSFT", "date": "2026-04-28", "type": "false_launch"},
    {"ticker": "SOFI", "date": "2025-12-26", "type": "false_launch"},
    {"ticker": "COIN", "date": "2026-01-06", "type": "false_launch"},
    {"ticker": "NKE",  "date": "2026-02-17", "type": "false_launch"},
    {"ticker": "NKE",  "date": "2026-03-31", "type": "false_launch"},
    {"ticker": "BE",   "date": "2026-02-25", "type": "false_launch"},
    {"ticker": "NFLX", "date": "2026-04-02", "type": "false_launch"},
    {"ticker": "PLTR", "date": "2026-04-08", "type": "false_launch"},
    {"ticker": "AMZN", "date": "2026-01-28", "type": "false_launch"},
    {"ticker": "META", "date": "2026-04-07", "type": "missed_launch"},
    {"ticker": "TSLA", "date": "2025-11-25", "type": "missed_launch"},
    {"ticker": "PLTR", "date": "2025-12-01", "type": "missed_launch"},
    {"ticker": "SNDK", "date": "2025-12-08", "type": "missed_launch"},
    {"ticker": "RKLB", "date": "2026-04-06", "type": "missed_launch"},
    {"ticker": "NVDA", "date": "2026-04-02", "type": "missed_launch"},
    {"ticker": "TSM",  "date": "2026-04-02", "type": "missed_launch"},
    {"ticker": "AMD",  "date": "2026-03-30", "type": "missed_launch"},
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
    df["Vol_5"] = df["Volume"].rolling(5).mean()
    df["Vol_20"] = df["Volume"].rolling(20).mean()
    df["Vol_60"] = df["Volume"].rolling(60).mean()
    df["price_diff"] = df["Close"].diff()
    df["obv_change"] = np.where(df["price_diff"] > 0, df["Volume"],
                                  np.where(df["price_diff"] < 0, -df["Volume"], 0))
    df["OBV"] = df["obv_change"].cumsum()
    return df


def compute_market_filter(market_df):
    market_df = market_df.copy()
    market_df["above_sma200"] = market_df["Close"] > market_df["SMA_200"]
    market_df["sma200_uptrend"] = market_df["SMA_200"] > market_df["SMA_200"].shift(5)
    market_df["market_ok"] = market_df["above_sma200"] & market_df["sma200_uptrend"]
    return market_df.set_index("Date")[["market_ok", "above_sma200", "sma200_uptrend"]]


def market_ok_at(market_state, date):
    if market_state is None or market_state.empty: return True
    sub = market_state[market_state.index <= date]
    if sub.empty: return True
    return bool(sub.iloc[-1]["market_ok"])


def market_status_at(market_state, date):
    if market_state is None or market_state.empty:
        return {"above_sma200": True, "sma200_uptrend": True, "market_ok": True}
    sub = market_state[market_state.index <= date]
    if sub.empty:
        return {"above_sma200": True, "sma200_uptrend": True, "market_ok": True}
    last = sub.iloc[-1]
    return {
        "above_sma200": bool(last["above_sma200"]),
        "sma200_uptrend": bool(last["sma200_uptrend"]),
        "market_ok": bool(last["market_ok"]),
    }


def get_engine_type(ticker):
    trend = {"NVDA", "MSFT", "AAPL", "GOOG", "GOOGL", "AMZN", "META", "TSM",
              "QQQ", "ORCL", "AVGO"}
    momentum = {"AMD", "TSLA", "MU", "INTC", "ASX"}
    growth = {"PLTR", "CRCL", "SOFI", "PYPL"}
    if ticker in trend: return "trend"
    if ticker in momentum: return "momentum"
    if ticker in growth: return "growth"
    return "reversal"


def compute_context(df, idx, ticker, market_state):
    if idx < 20 or idx >= len(df):
        return None
    curr = df.iloc[idx]
    prior_20 = df.iloc[max(0, idx - 20)]
    engine = get_engine_type(ticker)
    market = market_status_at(market_state, curr["Date"])
    high_52w = df.iloc[max(0, idx - 252): idx + 1]["High"].max()
    sma60 = curr.get("SMA_60", curr["Close"])
    sma200 = curr.get("SMA_200", curr["Close"])
    dist_from_52w_high = (high_52w - curr["Close"]) / high_52w * 100 if high_52w > 0 else 0
    dist_from_sma60 = (curr["Close"] - sma60) / sma60 * 100 if sma60 > 0 else 0
    above_sma200 = (not pd.isna(sma200) and curr["Close"] > sma200)
    vol_5 = curr.get("Vol_5", np.nan)
    vol_20 = curr.get("Vol_20", np.nan)
    vol_60 = curr.get("Vol_60", np.nan)
    short_vol_ratio = (vol_5 / vol_20 if (not pd.isna(vol_5) and not pd.isna(vol_20) and vol_20 > 0) else 1.0)
    trend_vol_ratio = (vol_20 / vol_60 if (not pd.isna(vol_20) and not pd.isna(vol_60) and vol_60 > 0) else 1.0)
    ret_20d = (curr["Close"] / prior_20["Close"] - 1) * 100 if prior_20["Close"] > 0 else 0
    obv_60d = df.iloc[max(0, idx - 60): idx + 1]["OBV"]
    obv_now = curr.get("OBV", 0)
    if not obv_60d.empty:
        obv_max = obv_60d.max()
        obv_min = obv_60d.min()
        obv_pct = ((obv_now - obv_min) / (obv_max - obv_min) * 100) if obv_max > obv_min else 50
    else:
        obv_pct = 50
    return {
        "engine_type": engine,
        "market": market,
        "position": {
            "dist_from_52w_high_pct": round(dist_from_52w_high, 1),
            "dist_from_sma60_pct": round(dist_from_sma60, 1),
            "above_sma200": above_sma200,
        },
        "short_vol_ratio": round(short_vol_ratio, 2),
        "trend_vol_ratio": round(trend_vol_ratio, 2),
        "momentum_20d": round(ret_20d, 2),
        "obv_pct_of_60d": round(obv_pct, 0),
    }


def detect_all_launch_signals(df, ticker, market_state=None):
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
        market_ok = market_ok_at(market_state, curr["Date"]) if market_state is not None else True
        all_signals.append({
            "date": curr["Date"], "idx": idx, "close": curr["Close"],
            "signal_type": signal_type,
            "trigger": "MA金叉+RSI超賣" if is_launch_point else "MA金叉",
            "ma60_uptrend": ma60_uptrend, "market_ok": market_ok,
            "engine_type": engine_type,
            "context": compute_context(df, idx, ticker, market_state),
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
        oversold_thr = 35 if engine_type in ("trend", "momentum") else 25
        is_deeply_oversold = rsi_w20.min() < oversold_thr if not rsi_w20.empty else False
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
            market_ok = market_ok_at(market_state, curr["Date"]) if market_state is not None else True
            all_signals.append({
                "date": curr["Date"], "idx": idx, "close": curr["Close"],
                "signal_type": "⭐ MACD起漲確認", "trigger": trigger,
                "ma60_uptrend": ma60_uptrend, "market_ok": market_ok,
                "engine_type": engine_type,
                "context": compute_context(df, idx, ticker, market_state),
            })

    all_signals.sort(key=lambda s: s["date"])
    return df, all_signals


def detect_exit_signals_v4(df, ticker, market_state=None):
    """v4 新版下車訊號：必要條件 + 4 計分"""
    df = df.copy().reset_index(drop=True)
    df["body"] = df["Close"] - df["Open"]
    df["is_red"] = df["body"] < 0

    exit_signals = []
    already_marked = set()

    for idx in range(60, len(df)):
        curr = df.iloc[idx]
        if (pd.isna(curr.get("RSI")) or pd.isna(curr.get("SMA_20"))
                or pd.isna(curr.get("SMA_60"))):
            continue

        # 必要條件 1: 距 SMA60 > +5%
        sma60 = curr["SMA_60"]
        if sma60 <= 0:
            continue
        dist_from_sma60 = (curr["Close"] - sma60) / sma60 * 100
        if dist_from_sma60 < 5:
            continue

        # 必要條件 2: 過去 20 日 RSI 曾 > 65
        rsi_w20 = df.iloc[max(0, idx - 20): idx + 1]["RSI"]
        if rsi_w20.empty or rsi_w20.max() < 65:
            continue

        # 計分訊號 1: RSI 負背離
        window = df.iloc[max(0, idx - 20): idx + 1]
        sig_1_rsi_divergence = False
        if len(window) >= 10:
            mid = len(window) // 2
            price_high_recent = window["Close"].iloc[mid:].max()
            price_high_old = window["Close"].iloc[:mid].max()
            rsi_high_recent = window["RSI"].iloc[mid:].max()
            rsi_high_old = window["RSI"].iloc[:mid].max()
            sig_1_rsi_divergence = (price_high_recent > price_high_old and
                                     rsi_high_recent < rsi_high_old)

        # 計分訊號 2: 量價背離（量縮但價漲）
        vol_5 = curr.get("Vol_5", np.nan)
        vol_20 = curr.get("Vol_20", np.nan)
        prior_5 = df.iloc[max(0, idx - 5)]
        price_rising = curr["Close"] > prior_5["Close"]
        vol_shrinking = (not pd.isna(vol_5) and not pd.isna(vol_20)
                          and vol_5 < vol_20 * 0.9)
        sig_2_vol_divergence = price_rising and vol_shrinking

        # 計分訊號 3: 跌破 SMA20 + RSI > 60
        sma20 = curr["SMA_20"]
        sig_3_break_sma20 = curr["Close"] < sma20 and curr["RSI"] > 60

        # 計分訊號 4: 連 5 漲後第一根放量黑 K
        last_5 = df.iloc[max(0, idx - 5): idx]
        sig_4_volume_red = False
        if len(last_5) >= 5:
            up_days = sum(1 for _, r in last_5.iterrows() if r["Close"] > r["Open"])
            sig_4_volume_red = (up_days >= 4 and curr["is_red"]
                                  and not pd.isna(vol_20)
                                  and curr["Volume"] > vol_20 * 1.2)

        signals_bool = [sig_1_rsi_divergence, sig_2_vol_divergence,
                          sig_3_break_sma20, sig_4_volume_red]
        n_hit = sum(signals_bool)
        if n_hit < 2: continue

        has_nearby = any(0 < idx - p < 10 for p in already_marked)
        if has_nearby: continue
        already_marked.add(idx)

        if n_hit == 4: level = "🔴 強烈下車"
        elif n_hit == 3: level = "🟠 中度下車"
        else: level = "🟡 觀察下車"

        exit_signals.append({
            "date": curr["Date"], "idx": idx, "close": curr["Close"],
            "n_hit": n_hit, "level": level, "signals": signals_bool,
            "rsi": float(curr["RSI"]),
            "context": compute_context(df, idx, ticker, market_state),
        })

    return exit_signals


def passes_new_rules_a(signal):
    if signal.get("trigger") in ("深度反轉(MACD<0)", "近MA60+RSI回暖"):
        return False
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
    log("v4 雙向回測 + 7維推演")
    log("=" * 60)

    spy_df = fetch_stock(MARKET_TICKER)
    market_state = None
    if spy_df is not None:
        spy_df = compute_indicators(spy_df)
        market_state = compute_market_filter(spy_df)
        log("SPY 完成", "ok")

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
        df, launch_sigs = detect_all_launch_signals(df, t, market_state)
        for s in launch_sigs:
            s.update(evaluate_signal(df, s))
            s["category"], s["category_label"] = classify_launch(s.get("ret_10d"))
            s["passes_a"] = passes_new_rules_a(s)
        exit_sigs = detect_exit_signals_v4(df, t, market_state)
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

    log(f"起漲 {len(all_launch)} 個 / 下車 {len(all_exit)} 個", "ok")

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
        case_matches.append({**case, "matched_signal": matched,
                              "match_diff_days": best_diff if matched else None})

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


def format_context(ctx):
    if not ctx:
        return "—"
    p = ctx["position"]
    m = ctx["market"]
    mkt_str = "🟢" if m["market_ok"] else "🔴"
    return (
        f'<span style="font-size:11px;color:#aaa;">'
        f'{ctx["engine_type"]} | {mkt_str} | '
        f'距高{p["dist_from_52w_high_pct"]:.0f}% | '
        f'量5d:{ctx["short_vol_ratio"]:.1f}x | '
        f'動能{ctx["momentum_20d"]:+.1f}% | '
        f'OBV{ctx["obv_pct_of_60d"]:.0f}%'
        f'</span>'
    )


def generate_html(all_launch, all_exit, case_matches, all_results):
    old_l = all_launch
    a_l = [s for s in all_launch if s["passes_a"]]
    s_old = calc_stats(old_l)
    s_a = calc_stats(a_l)

    exit_44 = [s for s in all_exit if s["n_hit"] == 4]
    exit_33 = [s for s in all_exit if s["n_hit"] == 3]
    exit_22 = [s for s in all_exit if s["n_hit"] == 2]
    s_e44 = calc_stats(exit_44, win_dir="down")
    s_e33 = calc_stats(exit_33, win_dir="down")
    s_e22 = calc_stats(exit_22, win_dir="down")

    case_rows = ""
    for c in case_matches:
        m = c.get("matched_signal")
        case_type = "🔴 假起漲" if c["type"] == "false_launch" else "🟡 漏標起漲"
        if m:
            r10 = f"{m['ret_10d']:+.2f}%" if m.get("ret_10d") is not None else "—"
            cat = m.get("category_label", "—")
            md = pd.to_datetime(m["date"]).strftime("%Y-%m-%d")
            a_pass = m.get("passes_a")
            is_false = c["type"] == "false_launch"
            if is_false:
                a_label = '<span style="color:#22c55e;">✅ 砍掉</span>' if not a_pass else '<span style="color:#facc15;">⚠️ 仍會發</span>'
            else:
                a_label = '<span style="color:#22c55e;">✅ 保留</span>' if a_pass else '<span style="color:#ef4444;">❌ 仍漏</span>'
            case_rows += f"""<tr>
              <td>{c['ticker']}</td><td>{c['date']}</td><td>{case_type}</td>
              <td>{md}</td><td>{m['signal_type']}</td><td>{m['trigger']}</td>
              <td>{r10}</td><td>{cat}</td><td>{a_label}</td></tr>"""
        else:
            case_rows += f"""<tr><td>{c['ticker']}</td><td>{c['date']}</td>
              <td>{case_type}</td><td colspan="6" style="color:#888;">(回測未抓到)</td></tr>"""

    exit_examples = sorted(all_exit, key=lambda x: -x["n_hit"])[:20]
    exit_rows = ""
    for s in exit_examples:
        r5 = f"{s.get('ret_5d', 0):+.2f}%" if s.get("ret_5d") is not None else "—"
        r10 = f"{s.get('ret_10d', 0):+.2f}%" if s.get("ret_10d") is not None else "—"
        r20 = f"{s.get('ret_20d', 0):+.2f}%" if s.get("ret_20d") is not None else "—"
        cat = s.get("category_label", "—")
        date_str = pd.to_datetime(s["date"]).strftime("%Y-%m-%d")
        ctx_str = format_context(s.get("context"))
        exit_rows += f"""<tr>
          <td>{s['ticker']}</td><td>{date_str}</td>
          <td>{s['level']}</td><td>{s['n_hit']}/4</td>
          <td>{r5}</td><td>{r10}</td><td>{r20}</td>
          <td>{cat}</td><td>{ctx_str}</td></tr>"""

    launch_examples = sorted(a_l, key=lambda x: -(x.get("ret_10d") or -999))[:15]
    launch_rows = ""
    for s in launch_examples:
        r10 = f"{s.get('ret_10d', 0):+.2f}%" if s.get("ret_10d") is not None else "—"
        cat = s.get("category_label", "—")
        date_str = pd.to_datetime(s["date"]).strftime("%Y-%m-%d")
        ctx_str = format_context(s.get("context"))
        launch_rows += f"""<tr>
          <td>{s['ticker']}</td><td>{date_str}</td>
          <td>{s['signal_type']}</td><td>{s['trigger']}</td>
          <td>{r10}</td><td>{cat}</td><td>{ctx_str}</td></tr>"""

    html = f"""<!DOCTYPE html><html lang="zh-Hant"><head><meta charset="UTF-8">
<title>v4 雙向回測 + 7維推演</title><style>
  body {{ background:#0e0e10; color:#e5e7eb; font-family:-apple-system,sans-serif;
         margin:0; padding:24px; max-width:1400px; margin:0 auto; }}
  h1, h2 {{ color:#facc15; }}
  h1 {{ border-bottom:2px solid #facc15; padding-bottom:12px; }}
  table {{ width:100%; border-collapse:collapse; margin:12px 0;
           background:#1a1a1c; border-radius:8px; overflow:hidden; font-size:13px; }}
  th, td {{ padding:8px 10px; text-align:left; border-bottom:1px solid #2a2a2c; }}
  th {{ background:#2a2a2c; color:#facc15; }}
  .card-row {{ display:grid; grid-template-columns:repeat(3,1fr); gap:14px; margin:16px 0; }}
  .card {{ background:#1a1a1c; padding:18px; border-radius:8px; border-left:4px solid #888; }}
  .big {{ font-size:28px; font-weight:bold; color:#facc15; }}
  .key-finding {{ background:#1a2a1c; border-left:4px solid #22c55e;
                  padding:14px 18px; margin:14px 0; border-radius:6px; }}
</style></head><body>

<h1>🎯 v4 雙向回測 — 新下車訊號 + 7 維推演 context</h1>
<p style="color:#888;">產出：{datetime.now().strftime('%Y-%m-%d %H:%M')} | 股票數：{len(all_results)}</p>

<h2>🚀 起漲訊號（方案 A）</h2>
<div class="card-row">
  <div class="card" style="border-left-color:#ef4444;">
    <div style="color:#ef4444;font-weight:bold;">❌ 舊規則</div>
    <div style="margin-top:6px;">數：<b>{s_old['count']}</b></div>
    <div>勝率：<span class="big">{s_old['win_rate']}</span></div>
    <div>平均：{s_old['avg']}</div>
  </div>
  <div class="card" style="border-left-color:#22c55e;">
    <div style="color:#22c55e;font-weight:bold;">✨ 方案 A</div>
    <div style="margin-top:6px;">數：<b>{s_a['count']}</b></div>
    <div>勝率：<span class="big" style="color:#22c55e;">{s_a['win_rate']}</span></div>
    <div>平均：{s_a['avg']}</div>
  </div>
  <div class="card" style="border-left-color:#facc15;">
    <div style="color:#facc15;font-weight:bold;">📐 訊號數變化</div>
    <div style="margin-top:6px;">減少：<b>{int((1-s_a['count']/max(s_old['count'],1))*100)}%</b></div>
    <div>勝率提升：<b style="color:#22c55e;">+{s_a['win_num']-s_old['win_num']:.1f}%</b></div>
  </div>
</div>

<h2>🔻 v4 新版下車訊號</h2>
<p style="color:#aaa;">
<b>必要條件</b>：距 SMA60 > +5% + 過去 20 日 RSI 曾 > 65<br>
<b>4 計分訊號</b>：RSI 負背離 / 量價背離 / 跌破 SMA20+RSI>60 / 連 5 漲後放量黑 K
</p>
<div class="card-row">
  <div class="card" style="border-left-color:#dc2626;">
    <div style="color:#dc2626;font-weight:bold;">🔴 強烈下車 (4/4)</div>
    <div style="margin-top:6px;">數：<b>{s_e44['count']}</b></div>
    <div>命中率：<span class="big" style="color:#dc2626;">{s_e44['win_rate']}</span></div>
    <div>平均跌幅：{s_e44['avg']}</div>
  </div>
  <div class="card" style="border-left-color:#f97316;">
    <div style="color:#f97316;font-weight:bold;">🟠 中度下車 (3/4)</div>
    <div style="margin-top:6px;">數：<b>{s_e33['count']}</b></div>
    <div>命中率：<span class="big" style="color:#f97316;">{s_e33['win_rate']}</span></div>
    <div>平均跌幅：{s_e33['avg']}</div>
  </div>
  <div class="card" style="border-left-color:#facc15;">
    <div style="color:#facc15;font-weight:bold;">🟡 觀察 (2/4)</div>
    <div style="margin-top:6px;">數：<b>{s_e22['count']}</b></div>
    <div>命中率：<span class="big">{s_e22['win_rate']}</span></div>
    <div>平均跌幅：{s_e22['avg']}</div>
  </div>
</div>

<h3>📋 下車訊號案例（含 7 維推演 context）</h3>
<table>
  <tr><th>股票</th><th>日期</th><th>等級</th><th>命中</th>
    <th>+5日</th><th>+10日</th><th>+20日</th><th>分類</th><th>7維 context</th></tr>
  {exit_rows}
</table>

<h3>🚀 起漲訊號代表案例（前 15 強，含 7 維 context）</h3>
<table>
  <tr><th>股票</th><th>日期</th><th>類型</th><th>觸發</th>
    <th>+10日</th><th>分類</th><th>7維 context</th></tr>
  {launch_rows}
</table>

<h2>📋 失敗案例對應（含 v4 新增 3 個漏標）</h2>
<table>
  <tr><th>股票</th><th>您日期</th><th>類型</th><th>對應日</th>
    <th>類型</th><th>觸發</th><th>+10日</th><th>分類</th><th>方案 A</th></tr>
  {case_rows}
</table>

<div class="key-finding">
  <b>💡 7 維推演 context</b><br>
  每個訊號附帶：個股性質 / 大盤狀態(🟢/🔴) / 距 52W 高 / 5日量比 / 20日動能 / OBV%<br>
  → 用於 AI 劇本給「機率推演 + 行動建議」
</div>

</body></html>
"""
    return html


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""backtest_signals_v6.py — 反轉預警 v6 加嚴版 + OBV 滿格反轉

v6 變更：
  1. 移除保守股（公用事業、必需消費品、電信、能源等低波動股）
  2. 反轉預警加嚴：RSI 從 < 30 → > 45 + OBV > 70 + 20 日動能 > -15%
  3. 新增「💰 OBV 滿格反轉」訊號（OBV > 90% + RSI 反轉）— 預期金訊號
"""
import os, sys, json, warnings, time
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

LOOKAHEAD_DAYS = [5, 10, 20]
HISTORY_MONTHS = 18
OUTPUT_FILE = "backtest_signals_v6_report.html"
MARKET_TICKER = "SPY"

# ── S&P 100 精選名單（去除保守股）──
# 排除：公用事業（DUK/SO/NEE）、必需消費品（PG/KO/PEP/CL/MDLZ/WMT/COST）、
#       電信（VZ/T）、能源（XOM/CVX/COP）、菸草（PM/MO）、健保穩健股（JNJ/PFE/MRK/ABBV）
# 保留：科技、金融、消費循環、半導體、雲端、生技創新等動能股
SP100_GROWTH = [
    # 大型科技
    "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "META", "NVDA", "TSLA", "AVGO", "ORCL",
    "ADBE", "CRM", "AMD", "INTC", "QCOM", "CSCO", "TXN", "INTU", "AMAT", "MU",
    "IBM", "ACN",
    # 金融（含成長型）
    "JPM", "BAC", "WFC", "GS", "MS", "BLK", "AXP", "C", "SCHW", "V", "MA", "PYPL",
    "USB", "SPGI", "MMC",
    # 醫療成長
    "UNH", "LLY", "TMO", "ABT", "DHR", "BMY", "AMGN", "GILD", "ISRG",
    # 消費循環
    "MCD", "SBUX", "DIS", "HD", "LOW", "TGT", "TJX", "NKE", "BKNG",
    # 工業/動能
    "BA", "CAT", "HON", "GE", "LMT", "RTX", "UPS", "FDX", "DE",
    # 通訊/媒體
    "CMCSA", "NFLX", "CHTR",
    # 其他高動能
    "BRK-B", "BX", "LIN",
]
SP100_GROWTH = list(set(SP100_GROWTH))


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
    df["price_diff"] = df["Close"].diff()
    df["obv_change"] = np.where(df["price_diff"] > 0, df["Volume"],
                                  np.where(df["price_diff"] < 0, -df["Volume"], 0))
    df["OBV"] = df["obv_change"].cumsum()
    return df


def compute_market_filter(market_df):
    market_df = market_df.copy()
    market_df["above_sma200"] = market_df["Close"] > market_df["SMA_200"]
    market_df["sma200_uptrend"] = market_df["SMA_200"] > market_df["SMA_200"].shift(5)
    market_df["market_ok_strict"] = market_df["above_sma200"] & market_df["sma200_uptrend"]
    market_df["market_ok_loose"] = market_df["above_sma200"]
    return market_df.set_index("Date")[["market_ok_strict", "market_ok_loose"]]


def market_state_at(market_state, date, key):
    if market_state is None or market_state.empty: return True
    sub = market_state[market_state.index <= date]
    if sub.empty: return True
    return bool(sub.iloc[-1][key])


def detect_plan_a_signals(df, ticker, market_state=None):
    """基準：方案 A 起漲訊號（不變）"""
    df = df.copy().reset_index(drop=True)
    df["ma_gold"] = ((df["SMA_20"] > df["SMA_60"]) &
                     (df["SMA_20"].shift(1) <= df["SMA_60"].shift(1)))
    df["rsi_was_oversold_20"] = df["RSI"].rolling(20, min_periods=1).min() < 45
    df["macd_gold"] = ((df["MACD"] > df["Signal_Line"]) &
                       (df["MACD"].shift(1) <= df["Signal_Line"].shift(1)))

    signals = []
    already_starred_idx = set()

    for idx in df.index[df["ma_gold"]]:
        if idx < 60: continue
        curr = df.iloc[idx]
        prior5 = df.iloc[max(0, idx - 5)]
        if not curr["rsi_was_oversold_20"]: continue
        ma60 = curr.get("SMA_60", 0)
        ma60_uptrend = (not pd.isna(ma60) and not pd.isna(prior5.get("SMA_60", np.nan))
                        and ma60 > prior5["SMA_60"])
        market_ok = market_state_at(market_state, curr["Date"], "market_ok_strict") if market_state is not None else True
        if ma60_uptrend and market_ok:
            signals.append({
                "date": curr["Date"], "idx": idx, "close": curr["Close"],
                "signal_type": "🚀 起漲點", "trigger": "MA金叉+RSI超賣",
            })

    for idx in df.index[df["macd_gold"]]:
        if idx < 5: continue
        curr = df.iloc[idx]
        prior = df.iloc[idx - 1]
        prior5 = df.iloc[max(0, idx - 5)]
        ma60 = curr.get("SMA_60", 0)
        ma20_val = curr.get("SMA_20", np.nan)
        death_cross_active = (not pd.isna(ma20_val) and not pd.isna(ma60)
                              and ma60 > 0 and ma20_val < ma60)
        zero_cross = (curr["MACD"] >= 0 and prior["MACD"] < 0)
        positive_zone_cross = (curr["MACD"] > 0 and curr.get("Signal_Line", -1) > 0)
        uptrend_zero_breakout = (not death_cross_active) and zero_cross
        strong_positive = (not death_cross_active) and positive_zone_cross
        base_trigger = strong_positive or uptrend_zero_breakout
        if not base_trigger: continue
        ma60_uptrend = (not pd.isna(ma60) and not pd.isna(prior5.get("SMA_60", np.nan))
                        and ma60 > prior5["SMA_60"])
        market_ok = market_state_at(market_state, curr["Date"], "market_ok_strict") if market_state is not None else True
        if not (ma60_uptrend and market_ok): continue
        has_nearby = any(0 < idx - p < 7 for p in already_starred_idx)
        if has_nearby: continue
        already_starred_idx.add(idx)
        trigger = "正值區金叉" if strong_positive else "零軸突破"
        signals.append({
            "date": curr["Date"], "idx": idx, "close": curr["Close"],
            "signal_type": "⭐ MACD起漲確認", "trigger": trigger,
        })

    signals.sort(key=lambda s: s["date"])
    return df, signals


def detect_reversal_v6(df, ticker, market_state=None):
    """v6 加嚴版反轉預警 + OBV 滿格反轉
    
    回傳兩種訊號：
      - 💡 反轉預警 v6（加嚴）
      - 💰 OBV 滿格反轉（金訊號）
    """
    df = df.copy().reset_index(drop=True)
    df["macd_gold"] = ((df["MACD"] > df["Signal_Line"]) &
                       (df["MACD"].shift(1) <= df["Signal_Line"].shift(1)))

    reversal_signals = []
    obv_gold_signals = []
    already_marked = set()

    for idx in df.index[df["macd_gold"]]:
        if idx < 20: continue
        curr = df.iloc[idx]
        if pd.isna(curr.get("RSI")) or pd.isna(curr.get("SMA_60")):
            continue

        # 條件 1: MACD 接近零軸
        macd_near_zero = abs(curr["MACD"]) < curr["Close"] * 0.005

        # 條件 2 加嚴: RSI 從 < 30 → > 45
        rsi_w10 = df.iloc[max(0, idx - 10): idx + 1]["RSI"]
        rsi_was_deep = rsi_w10.min() < 30 if not rsi_w10.empty else False
        rsi_now_up = curr["RSI"] > 45
        rsi_real_reversal = rsi_was_deep and rsi_now_up

        # 條件 3: 大盤寬鬆 OK
        market_ok_loose = market_state_at(market_state, curr["Date"], "market_ok_loose") if market_state is not None else True

        # 條件 4: 距 SMA60 -15% ~ +15%
        sma60 = curr["SMA_60"]
        if sma60 <= 0: continue
        dist_from_sma60 = (curr["Close"] - sma60) / sma60 * 100
        in_range = -15 < dist_from_sma60 < 15

        # 條件 5 新增: 20 日動能 > -15%（不在大跌中）
        if idx >= 20:
            prior_20 = df.iloc[idx - 20]
            momentum_20d = (curr["Close"] / prior_20["Close"] - 1) * 100 if prior_20["Close"] > 0 else 0
        else:
            momentum_20d = 0
        not_crashing = momentum_20d > -15

        # 條件 6: OBV 位置
        obv_60d = df.iloc[max(0, idx - 60): idx + 1]["OBV"]
        if not obv_60d.empty:
            obv_max = obv_60d.max()
            obv_min = obv_60d.min()
            obv_now = curr["OBV"]
            obv_pct = ((obv_now - obv_min) / (obv_max - obv_min) * 100) if obv_max > obv_min else 50
        else:
            obv_pct = 50

        # ── v6 加嚴反轉預警：OBV > 70% ──
        if (macd_near_zero and rsi_real_reversal and market_ok_loose
                and in_range and not_crashing and obv_pct > 70):
            has_nearby = any(0 < idx - p < 10 for p in already_marked)
            if not has_nearby:
                already_marked.add(idx)
                reversal_signals.append({
                    "date": curr["Date"], "idx": idx, "close": curr["Close"],
                    "signal_type": "💡 反轉預警(v6)", 
                    "trigger": f"RSI {rsi_w10.min():.0f}→{curr['RSI']:.0f}, OBV {obv_pct:.0f}%",
                    "rsi": float(curr["RSI"]),
                    "rsi_was_min": float(rsi_w10.min()) if not rsi_w10.empty else 0,
                    "obv_pct": round(obv_pct, 0),
                    "dist_from_sma60": round(dist_from_sma60, 1),
                    "momentum_20d": round(momentum_20d, 2),
                })

        # ── 💰 OBV 滿格反轉（金訊號）：條件更嚴 ──
        # 條件：OBV > 90% + RSI 反轉（從 < 35 → > 40）+ 大盤 OK + 距 SMA60 -10~+10
        obv_fullshot = obv_pct > 90
        rsi_reversal_loose = (rsi_w10.min() < 35) and curr["RSI"] > 40
        obv_in_range = -10 < dist_from_sma60 < 10
        if obv_fullshot and rsi_reversal_loose and market_ok_loose and obv_in_range and not_crashing:
            has_nearby_obv = any(0 < idx - p < 10 for p in already_marked)
            if not has_nearby_obv:
                already_marked.add(idx)
                obv_gold_signals.append({
                    "date": curr["Date"], "idx": idx, "close": curr["Close"],
                    "signal_type": "💰 OBV 滿格反轉",
                    "trigger": f"OBV {obv_pct:.0f}% + RSI {rsi_w10.min():.0f}→{curr['RSI']:.0f}",
                    "rsi": float(curr["RSI"]),
                    "rsi_was_min": float(rsi_w10.min()) if not rsi_w10.empty else 0,
                    "obv_pct": round(obv_pct, 0),
                    "dist_from_sma60": round(dist_from_sma60, 1),
                    "momentum_20d": round(momentum_20d, 2),
                })

    return reversal_signals, obv_gold_signals


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
    if ret_10d is None: return ("no_data", "資料不足")
    if ret_10d >= 5: return ("excellent", "✅ 優秀(+5%↑)")
    elif ret_10d >= 2: return ("good", "👍 不錯(+2~5%)")
    elif ret_10d >= -2: return ("flat", "⚖️ 平淡(±2%)")
    elif ret_10d >= -5: return ("poor", "😟 不佳(-2~-5%)")
    else: return ("bad", "❌ 失敗(-5%↓)")


def main():
    log("=" * 60)
    log("v6 反轉預警加嚴 + OBV 滿格反轉")
    log("=" * 60)

    log("抓取 SPY...", "info")
    spy_df = fetch_stock(MARKET_TICKER)
    market_state = None
    if spy_df is not None:
        spy_df = compute_indicators(spy_df)
        market_state = compute_market_filter(spy_df)
        log("SPY 完成", "ok")

    log(f"分析 S&P 100 成長股共 {len(SP100_GROWTH)} 支（已排除公用事業/必需消費等）", "info")

    all_results = {}
    failed = []
    for i, t in enumerate(SP100_GROWTH, 1):
        if i % 10 == 0:
            log(f"進度 {i}/{len(SP100_GROWTH)}", "info")
        df = fetch_stock(t)
        if df is None or len(df) < 100:
            failed.append(t)
            continue
        df = compute_indicators(df)
        df, plan_a_sigs = detect_plan_a_signals(df, t, market_state)
        for s in plan_a_sigs:
            s.update(evaluate_signal(df, s))
            s["category"], s["category_label"] = classify(s.get("ret_10d"))
        rev_sigs, obv_sigs = detect_reversal_v6(df, t, market_state)
        for s in rev_sigs:
            s.update(evaluate_signal(df, s))
            s["category"], s["category_label"] = classify(s.get("ret_10d"))
        for s in obv_sigs:
            s.update(evaluate_signal(df, s))
            s["category"], s["category_label"] = classify(s.get("ret_10d"))
        all_results[t] = {"df": df, "plan_a": plan_a_sigs,
                            "reversal": rev_sigs, "obv_gold": obv_sigs}
        time.sleep(0.15)

    log(f"完成（失敗 {len(failed)} 支）", "ok")

    all_plan_a = []; all_rev = []; all_obv = []
    for t, r in all_results.items():
        for s in r["plan_a"]:
            s["ticker"] = t
            all_plan_a.append(s)
        for s in r["reversal"]:
            s["ticker"] = t
            all_rev.append(s)
        for s in r["obv_gold"]:
            s["ticker"] = t
            all_obv.append(s)

    log(f"方案 A {len(all_plan_a)} | 反轉預警 v6 {len(all_rev)} | OBV 滿格 {len(all_obv)}", "ok")

    # 領先天數分析（反轉預警 vs 方案 A）
    lead_rev = []
    for s_a in all_plan_a:
        t = s_a["ticker"]; a_date = pd.to_datetime(s_a["date"])
        cands = [r for r in all_rev if r["ticker"] == t]
        best = None
        for r in cands:
            r_date = pd.to_datetime(r["date"])
            days = (a_date - r_date).days
            if 1 <= days <= 30:
                if best is None or days < best["days_lead"]:
                    best = {**r, "days_lead": days}
        if best:
            lead_rev.append({**best, "plan_a_date": s_a["date"], "plan_a_ret10": s_a.get("ret_10d")})

    # OBV 滿格的領先分析
    lead_obv = []
    for s_a in all_plan_a:
        t = s_a["ticker"]; a_date = pd.to_datetime(s_a["date"])
        cands = [r for r in all_obv if r["ticker"] == t]
        best = None
        for r in cands:
            r_date = pd.to_datetime(r["date"])
            days = (a_date - r_date).days
            if 1 <= days <= 30:
                if best is None or days < best["days_lead"]:
                    best = {**r, "days_lead": days}
        if best:
            lead_obv.append({**best, "plan_a_date": s_a["date"], "plan_a_ret10": s_a.get("ret_10d")})

    log("產生 HTML 報告...", "info")
    html = generate_html(all_plan_a, all_rev, all_obv, lead_rev, lead_obv, all_results, failed)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(html)
    log(f"完成：{os.path.abspath(OUTPUT_FILE)}", "ok")


def calc_stats(signals):
    valid = [s for s in signals if s.get("ret_10d") is not None]
    if not valid:
        return {"count": 0, "win_rate": "—", "avg": "—", "median": "—", "win_num": 0}
    rets = [s["ret_10d"] for s in valid]
    wins = sum(1 for r in rets if r > 0)
    return {
        "count": len(valid),
        "win_rate": f"{wins/len(valid)*100:.1f}%",
        "win_num": wins/len(valid)*100,
        "avg": f"{np.mean(rets):+.2f}%",
        "median": f"{np.median(rets):+.2f}%",
        "worst": f"{min(rets):+.2f}%",
    }


def generate_html(all_plan_a, all_rev, all_obv, lead_rev, lead_obv, all_results, failed):
    s_a = calc_stats(all_plan_a)
    s_r = calc_stats(all_rev)
    s_o = calc_stats(all_obv)

    avg_rev = np.mean([x["days_lead"] for x in lead_rev]) if lead_rev else 0
    avg_obv = np.mean([x["days_lead"] for x in lead_obv]) if lead_obv else 0

    def card_html(emoji, name, color, stats, lead_days=None, lead_count=None):
        win_col = "#22c55e" if stats["win_num"] >= 60 else "#facc15" if stats["win_num"] >= 50 else "#ef4444"
        lead_html = ""
        if lead_days is not None:
            lead_html = f"<div>領先方案 A：<b style='color:#22c55e'>{lead_days:.1f} 天</b>（{lead_count} 案例）</div>"
        return f"""
        <div class="card" style="border-left-color:{color};">
          <div style="color:{color};font-weight:bold;">{emoji} {name}</div>
          <div style="margin-top:6px;">數：<b>{stats['count']}</b></div>
          <div>勝率：<span class="big" style="color:{win_col};">{stats['win_rate']}</span></div>
          <div>平均：{stats['avg']}</div>
          <div>中位：{stats['median']}</div>
          {lead_html}
        </div>"""

    # 反轉預警案例
    rev_top = sorted(all_rev, key=lambda x: -(x.get("ret_10d") or -999))[:15]
    rev_rows = ""
    for s in rev_top:
        r5 = f"{s.get('ret_5d', 0):+.2f}%" if s.get("ret_5d") is not None else "—"
        r10 = f"{s.get('ret_10d', 0):+.2f}%" if s.get("ret_10d") is not None else "—"
        r20 = f"{s.get('ret_20d', 0):+.2f}%" if s.get("ret_20d") is not None else "—"
        cat = s.get("category_label", "—")
        date_str = pd.to_datetime(s["date"]).strftime("%Y-%m-%d")
        rev_rows += f"""<tr>
          <td>{s['ticker']}</td><td>{date_str}</td>
          <td>RSI {s.get('rsi_was_min',0):.0f}→{s.get('rsi',0):.0f}</td>
          <td>{s.get('obv_pct',0):.0f}%</td>
          <td>{s.get('momentum_20d',0):+.1f}%</td>
          <td>{r5}</td><td>{r10}</td><td>{r20}</td>
          <td>{cat}</td></tr>"""

    # OBV 滿格案例
    obv_top = sorted(all_obv, key=lambda x: -(x.get("ret_10d") or -999))[:15]
    obv_rows = ""
    for s in obv_top:
        r5 = f"{s.get('ret_5d', 0):+.2f}%" if s.get("ret_5d") is not None else "—"
        r10 = f"{s.get('ret_10d', 0):+.2f}%" if s.get("ret_10d") is not None else "—"
        r20 = f"{s.get('ret_20d', 0):+.2f}%" if s.get("ret_20d") is not None else "—"
        cat = s.get("category_label", "—")
        date_str = pd.to_datetime(s["date"]).strftime("%Y-%m-%d")
        obv_rows += f"""<tr>
          <td>{s['ticker']}</td><td>{date_str}</td>
          <td>RSI {s.get('rsi_was_min',0):.0f}→{s.get('rsi',0):.0f}</td>
          <td><b style="color:#22c55e">{s.get('obv_pct',0):.0f}%</b></td>
          <td>{s.get('momentum_20d',0):+.1f}%</td>
          <td>{r5}</td><td>{r10}</td><td>{r20}</td>
          <td>{cat}</td></tr>"""

    # 領先案例展示（OBV 滿格）
    obv_lead_rows = ""
    for la in sorted(lead_obv, key=lambda x: -(x.get("ret_10d") or -999))[:10]:
        plan_a_d = pd.to_datetime(la["plan_a_date"]).strftime("%Y-%m-%d")
        obv_d = pd.to_datetime(la["date"]).strftime("%Y-%m-%d")
        r10 = f"{la.get('ret_10d', 0):+.2f}%" if la.get("ret_10d") is not None else "—"
        a_r10 = f"{la.get('plan_a_ret10', 0):+.2f}%" if la.get("plan_a_ret10") is not None else "—"
        obv_lead_rows += f"""<tr>
          <td>{la['ticker']}</td>
          <td>{obv_d}</td>
          <td>{plan_a_d}</td>
          <td><b style="color:#22c55e;">{la['days_lead']} 天</b></td>
          <td>OBV {la.get('obv_pct', 0):.0f}%</td>
          <td>{r10}</td>
          <td>{a_r10}</td></tr>"""

    # 驗證結論
    rev_pass = s_r["win_num"] >= 60 and s_r["count"] >= 15
    obv_pass = s_o["win_num"] >= 65 and s_o["count"] >= 10

    html = f"""<!DOCTYPE html><html lang="zh-Hant"><head><meta charset="UTF-8">
<title>v6 反轉預警加嚴 + OBV 滿格反轉</title><style>
  body {{ background:#0e0e10; color:#e5e7eb; font-family:-apple-system,sans-serif;
         margin:0; padding:24px; max-width:1400px; margin:0 auto; }}
  h1, h2 {{ color:#facc15; }}
  h1 {{ border-bottom:2px solid #facc15; padding-bottom:12px; }}
  table {{ width:100%; border-collapse:collapse; margin:12px 0;
           background:#1a1a1c; border-radius:8px; overflow:hidden; font-size:13px; }}
  th, td {{ padding:9px 11px; text-align:left; border-bottom:1px solid #2a2a2c; }}
  th {{ background:#2a2a2c; color:#facc15; }}
  .card-row {{ display:grid; grid-template-columns:repeat(3,1fr); gap:14px; margin:16px 0; }}
  .card {{ background:#1a1a1c; padding:18px; border-radius:8px; border-left:4px solid #888; }}
  .big {{ font-size:30px; font-weight:bold; color:#facc15; }}
  .verdict {{ background:linear-gradient(135deg,#1a1a1c,#1a2a1c);
              padding:20px; border-radius:10px; border:2px solid #facc15;
              margin:20px 0; }}
  .ok {{ color:#22c55e;font-weight:bold; }}
  .bad {{ color:#ef4444;font-weight:bold; }}
</style></head><body>

<h1>🎯 v6 反轉預警加嚴 + OBV 滿格反轉</h1>
<p style="color:#888;">產出：{datetime.now().strftime('%Y-%m-%d %H:%M')} | 成功 {len(all_results)} / 失敗 {len(failed)} 支 | 已排除公用事業/必需消費等保守股</p>

<div class="verdict">
  <h2 style="margin-top:0;">📊 三方案大比拼</h2>
  <div class="card-row">
    {card_html("⚡", "方案 A 起漲訊號（基準）", "#facc15", s_a)}
    {card_html("💡", "反轉預警 v6（加嚴）", "#22c55e", s_r, avg_rev, len(lead_rev))}
    {card_html("💰", "OBV 滿格反轉（金訊號）", "#ec4899", s_o, avg_obv, len(lead_obv))}
  </div>
  
  <h3 style="margin-top:18px;">📋 採用結論</h3>
  <p>
    <b>反轉預警 v6</b>：勝率 {s_r['win_rate']} 
    <span class="{'ok' if s_r['win_num'] >= 60 else 'bad'}">
      {'✅ 達標（≥60%）' if s_r['win_num'] >= 60 else '❌ 未達標（&lt;60%）'}
    </span> 
    | 樣本 {s_r['count']} 
    <span class="{'ok' if s_r['count'] >= 15 else 'bad'}">
      {'✅ 充足' if s_r['count'] >= 15 else '⚠️ 偏少'}
    </span> 
    → <b>{('採用' if rev_pass else '不採用')}</b>
  </p>
  <p>
    <b>OBV 滿格反轉</b>：勝率 {s_o['win_rate']} 
    <span class="{'ok' if s_o['win_num'] >= 65 else 'bad'}">
      {'✅ 達標（≥65%）' if s_o['win_num'] >= 65 else '❌ 未達標（&lt;65%）'}
    </span> 
    | 樣本 {s_o['count']} 
    <span class="{'ok' if s_o['count'] >= 10 else 'bad'}">
      {'✅ 充足' if s_o['count'] >= 10 else '⚠️ 偏少'}
    </span>
    → <b>{('採用' if obv_pass else '不採用')}</b>
  </p>
</div>

<h2>💰 OBV 滿格反轉 — 代表案例（前 15 強）</h2>
<p style="color:#aaa;">這是預期的「金訊號」 — OBV 在 60 日內 > 90% + 深度 RSI 反轉。</p>
<table>
  <tr><th>股票</th><th>日期</th><th>RSI 反轉</th><th>OBV</th><th>20日動能</th>
    <th>+5日</th><th>+10日</th><th>+20日</th><th>分類</th></tr>
  {obv_rows if obv_rows else '<tr><td colspan="9" style="color:#888;">無資料</td></tr>'}
</table>

<h2>⏱️ OBV 滿格 vs 方案 A：領先案例（前 10）</h2>
<table>
  <tr><th>股票</th><th>OBV 滿格日</th><th>方案 A 日</th><th>領先天數</th>
    <th>OBV</th><th>OBV +10日</th><th>方案 A +10日</th></tr>
  {obv_lead_rows if obv_lead_rows else '<tr><td colspan="7" style="color:#888;">無領先案例</td></tr>'}
</table>

<h2>💡 反轉預警 v6 — 代表案例（前 15 強）</h2>
<table>
  <tr><th>股票</th><th>日期</th><th>RSI 反轉</th><th>OBV</th><th>20日動能</th>
    <th>+5日</th><th>+10日</th><th>+20日</th><th>分類</th></tr>
  {rev_rows if rev_rows else '<tr><td colspan="9" style="color:#888;">無資料</td></tr>'}
</table>

</body></html>
"""
    return html


if __name__ == "__main__":
    main()

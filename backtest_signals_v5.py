#!/usr/bin/env python3
"""backtest_signals_v5.py — 反轉預警訊號回測 (S&P 100 限定)

目標：抓「方案 A」沒抓到的真實反轉初期訊號
  - 比 ⭐ 起漲確認早 5-15 個交易日
  - 範圍：S&P 100（trend 類大型權值股）
  
反轉預警條件（需全部滿足）：
  1. 屬於 trend 類（S&P 100 大型股）
  2. MACD 金叉發生
  3. MACD 絕對值 < 股價 × 0.5%（接近零軸的金叉）
  4. RSI 從 < 35 回升到 > 40（真實深度反轉）
  5. 大盤 OK（SPY 站 SMA200，但不要求 SMA200 嚴格上升）
  6. 不要求 MA60 上升（反轉初期 MA60 通常未轉）
  7. 距 SMA60 < 15%（不在大跌深淵）
  8. OBV 在 60 日內位置 > 30%（仍有資金累積）
"""
import os, sys, json, warnings, time
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

LOOKAHEAD_DAYS = [5, 10, 20]
HISTORY_MONTHS = 18
OUTPUT_FILE = "backtest_signals_v5_report.html"
MARKET_TICKER = "SPY"

# ── S&P 100 名單（trend 類大型權值股）──
# 涵蓋科技/金融/醫療/消費等多元產業
SP100_TICKERS = [
    # 大型科技
    "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "META", "NVDA", "TSLA", "AVGO", "ORCL",
    "ADBE", "CRM", "AMD", "INTC", "QCOM", "CSCO", "TXN", "INTU", "AMAT", "MU",
    # 金融
    "JPM", "BAC", "WFC", "GS", "MS", "BLK", "AXP", "C", "SCHW", "V", "MA", "PYPL",
    # 醫療
    "UNH", "JNJ", "LLY", "ABBV", "MRK", "TMO", "ABT", "PFE", "DHR", "BMY", "AMGN",
    # 消費
    "WMT", "PG", "KO", "PEP", "COST", "MCD", "NKE", "SBUX", "DIS", "HD", "LOW",
    "TGT", "PM", "MO", "CL", "MDLZ",
    # 工業/能源
    "BA", "CAT", "HON", "GE", "LMT", "RTX", "UPS", "FDX", "DE", "XOM", "CVX", "COP",
    # 通訊/媒體
    "VZ", "T", "CMCSA", "NFLX", "CHTR",
    # 其他
    "BRK-B", "BX", "LIN", "ACN", "IBM", "DUK", "SO", "NEE", "GILD", "BKNG", "ISRG",
    "TJX", "SPGI", "MS", "MMC", "USB",
]
# 去除重複
SP100_TICKERS = list(set(SP100_TICKERS))


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
    # 嚴格大盤（給方案 A 用）
    market_df["market_ok_strict"] = market_df["above_sma200"] & market_df["sma200_uptrend"]
    # 寬鬆大盤（給反轉預警用）：只要在 SMA200 之上，不要求 SMA200 上升
    market_df["market_ok_loose"] = market_df["above_sma200"]
    return market_df.set_index("Date")[["market_ok_strict", "market_ok_loose"]]


def market_state_at(market_state, date, key):
    if market_state is None or market_state.empty: return True
    sub = market_state[market_state.index <= date]
    if sub.empty: return True
    return bool(sub.iloc[-1][key])


# ──────────────────────────────────────────────────────
# 偵測：方案 A 起漲訊號（基準）
# ──────────────────────────────────────────────────────
def detect_plan_a_signals(df, ticker, market_state=None):
    df = df.copy().reset_index(drop=True)
    df["ma_gold"] = ((df["SMA_20"] > df["SMA_60"]) &
                     (df["SMA_20"].shift(1) <= df["SMA_60"].shift(1)))
    df["rsi_was_oversold_20"] = df["RSI"].rolling(20, min_periods=1).min() < 45
    df["macd_gold"] = ((df["MACD"] > df["Signal_Line"]) &
                       (df["MACD"].shift(1) <= df["Signal_Line"].shift(1)))

    signals = []
    already_starred_idx = set()

    # MA 金叉起漲點
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

    # MACD 金叉 ⭐起漲確認
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

        # 防叢集
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


# ──────────────────────────────────────────────────────
# 偵測：💡 反轉預警訊號（新）
# ──────────────────────────────────────────────────────
def detect_reversal_early_signals(df, ticker, market_state=None):
    """💡 反轉預警 — 抓真實反轉初期
    
    比方案 A 早 5-15 個交易日的訊號，限 S&P 100 trend 類。
    """
    df = df.copy().reset_index(drop=True)
    df["macd_gold"] = ((df["MACD"] > df["Signal_Line"]) &
                       (df["MACD"].shift(1) <= df["Signal_Line"].shift(1)))

    signals = []
    already_marked = set()

    for idx in df.index[df["macd_gold"]]:
        if idx < 20: continue
        curr = df.iloc[idx]
        prior10 = df.iloc[max(0, idx - 10)]
        if pd.isna(curr.get("RSI")) or pd.isna(curr.get("SMA_60")):
            continue

        # 條件 3：MACD 接近零軸（絕對值 < 股價 × 0.5%）
        macd_near_zero = abs(curr["MACD"]) < curr["Close"] * 0.005

        # 條件 4：RSI 真實深度反轉（10 日內 RSI 曾 < 35，現在 > 40）
        rsi_w10 = df.iloc[max(0, idx - 10): idx + 1]["RSI"]
        rsi_was_deep = rsi_w10.min() < 35 if not rsi_w10.empty else False
        rsi_now_up = curr["RSI"] > 40
        rsi_real_reversal = rsi_was_deep and rsi_now_up

        # 條件 5：大盤寬鬆 OK
        market_ok_loose = market_state_at(market_state, curr["Date"], "market_ok_loose") if market_state is not None else True

        # 條件 7：距 SMA60 < 15%（不在大跌深淵）
        sma60 = curr["SMA_60"]
        if sma60 <= 0: continue
        dist_from_sma60 = (curr["Close"] - sma60) / sma60 * 100
        not_in_deep_abyss = dist_from_sma60 > -15  # 不能跌破 SMA60 超過 15%

        # 條件 8：OBV 累積位置 > 30%
        obv_60d = df.iloc[max(0, idx - 60): idx + 1]["OBV"]
        if not obv_60d.empty:
            obv_max = obv_60d.max()
            obv_min = obv_60d.min()
            obv_now = curr["OBV"]
            obv_pct = ((obv_now - obv_min) / (obv_max - obv_min) * 100) if obv_max > obv_min else 50
        else:
            obv_pct = 50
        obv_still_accumulating = obv_pct > 30

        # 全條件檢查
        if not (macd_near_zero and rsi_real_reversal and market_ok_loose
                and not_in_deep_abyss and obv_still_accumulating):
            continue

        # 防叢集（10 根內不重複）
        has_nearby = any(0 < idx - p < 10 for p in already_marked)
        if has_nearby: continue
        already_marked.add(idx)

        signals.append({
            "date": curr["Date"], "idx": idx, "close": curr["Close"],
            "signal_type": "💡 反轉預警", "trigger": "深度RSI反轉+MACD近零軸",
            "rsi": float(curr["RSI"]),
            "rsi_was_min": float(rsi_w10.min()) if not rsi_w10.empty else 0,
            "macd": float(curr["MACD"]),
            "obv_pct": round(obv_pct, 0),
            "dist_from_sma60": round(dist_from_sma60, 1),
        })

    return signals


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


def classify_signal(ret_10d):
    if ret_10d is None: return ("no_data", "資料不足")
    if ret_10d >= 5: return ("excellent", "✅ 優秀(+5%↑)")
    elif ret_10d >= 2: return ("good", "👍 不錯(+2~5%)")
    elif ret_10d >= -2: return ("flat", "⚖️ 平淡(±2%)")
    elif ret_10d >= -5: return ("poor", "😟 不佳(-2~-5%)")
    else: return ("bad", "❌ 失敗(-5%↓)")


def main():
    log("=" * 60)
    log("反轉預警訊號回測 v5 — S&P 100 限定")
    log("=" * 60)

    log("抓取 SPY 大盤過濾...", "info")
    spy_df = fetch_stock(MARKET_TICKER)
    market_state = None
    if spy_df is not None:
        spy_df = compute_indicators(spy_df)
        market_state = compute_market_filter(spy_df)
        log("SPY 完成", "ok")

    log(f"分析 S&P 100 共 {len(SP100_TICKERS)} 支股票", "info")

    all_results = {}
    failed_tickers = []
    for i, t in enumerate(SP100_TICKERS, 1):
        if i % 10 == 0:
            log(f"進度 {i}/{len(SP100_TICKERS)}", "info")
        df = fetch_stock(t)
        if df is None or len(df) < 100:
            failed_tickers.append(t)
            continue
        df = compute_indicators(df)
        # 方案 A 訊號（對照）
        df, plan_a_sigs = detect_plan_a_signals(df, t, market_state)
        for s in plan_a_sigs:
            s.update(evaluate_signal(df, s))
            s["category"], s["category_label"] = classify_signal(s.get("ret_10d"))
        # 反轉預警訊號（新）
        reversal_sigs = detect_reversal_early_signals(df, t, market_state)
        for s in reversal_sigs:
            s.update(evaluate_signal(df, s))
            s["category"], s["category_label"] = classify_signal(s.get("ret_10d"))
        all_results[t] = {"df": df, "plan_a": plan_a_sigs, "reversal": reversal_sigs}
        time.sleep(0.15)

    log(f"完成（失敗 {len(failed_tickers)} 支）", "ok")

    all_plan_a = []
    all_reversal = []
    for t, r in all_results.items():
        for s in r["plan_a"]:
            s["ticker"] = t
            all_plan_a.append(s)
        for s in r["reversal"]:
            s["ticker"] = t
            all_reversal.append(s)

    log(f"方案 A {len(all_plan_a)} 個 / 反轉預警 {len(all_reversal)} 個", "ok")

    # 計算「反轉預警」相對於「方案 A」的領先天數
    # 對每個方案 A 訊號，找該股票前 30 天內是否有反轉預警
    lead_analysis = []
    for s_a in all_plan_a:
        t = s_a["ticker"]
        a_date = pd.to_datetime(s_a["date"])
        # 找該股票的反轉預警，且日期在 a_date 之前 30 天內
        revs_for_t = [r for r in all_reversal if r["ticker"] == t]
        best_lead = None
        for r in revs_for_t:
            r_date = pd.to_datetime(r["date"])
            days_diff = (a_date - r_date).days
            if 1 <= days_diff <= 30:
                if best_lead is None or days_diff < best_lead["days_lead"]:
                    best_lead = {**r, "days_lead": days_diff}
        if best_lead:
            lead_analysis.append({
                "ticker": t,
                "plan_a_date": s_a["date"],
                "plan_a_trigger": s_a["trigger"],
                "plan_a_ret10": s_a.get("ret_10d"),
                "reversal_date": best_lead["date"],
                "reversal_ret10": best_lead.get("ret_10d"),
                "days_lead": best_lead["days_lead"],
            })

    log("產生 HTML 報告...", "info")
    html = generate_html(all_plan_a, all_reversal, lead_analysis, all_results, failed_tickers)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(html)
    log(f"完成：{os.path.abspath(OUTPUT_FILE)}", "ok")


def calc_stats(signals):
    valid = [s for s in signals if s.get("ret_10d") is not None]
    if not valid:
        return {"count": 0, "win_rate": "—", "avg": "—", "median": "—", "worst": "—",
                "win_num": 0}
    rets = [s["ret_10d"] for s in valid]
    wins = sum(1 for r in rets if r > 0)
    return {
        "count": len(valid),
        "win_rate": f"{wins/len(valid)*100:.1f}%",
        "win_num": wins / len(valid) * 100,
        "avg": f"{np.mean(rets):+.2f}%",
        "median": f"{np.median(rets):+.2f}%",
        "worst": f"{min(rets):+.2f}%",
    }


def generate_html(all_plan_a, all_reversal, lead_analysis, all_results, failed_tickers):
    s_a = calc_stats(all_plan_a)
    s_r = calc_stats(all_reversal)

    # 領先天數分布
    if lead_analysis:
        leads = [x["days_lead"] for x in lead_analysis]
        avg_lead = np.mean(leads)
        median_lead = np.median(leads)
        lead_count = len(lead_analysis)
    else:
        avg_lead = median_lead = lead_count = 0

    # 反轉預警案例（前 20 強）
    rev_examples = sorted(all_reversal, key=lambda x: -(x.get("ret_10d") or -999))[:20]
    rev_rows = ""
    for s in rev_examples:
        r5 = f"{s.get('ret_5d', 0):+.2f}%" if s.get("ret_5d") is not None else "—"
        r10 = f"{s.get('ret_10d', 0):+.2f}%" if s.get("ret_10d") is not None else "—"
        r20 = f"{s.get('ret_20d', 0):+.2f}%" if s.get("ret_20d") is not None else "—"
        cat = s.get("category_label", "—")
        date_str = pd.to_datetime(s["date"]).strftime("%Y-%m-%d")
        rev_rows += f"""<tr>
          <td>{s['ticker']}</td><td>{date_str}</td>
          <td>RSI 從 {s.get('rsi_was_min',0):.0f} → {s.get('rsi',0):.0f}</td>
          <td>OBV {s.get('obv_pct',0):.0f}%</td>
          <td>{s.get('dist_from_sma60',0):+.1f}%</td>
          <td>{r5}</td><td>{r10}</td><td>{r20}</td>
          <td>{cat}</td></tr>"""

    # 領先案例（top 15）
    lead_rows = ""
    for la in sorted(lead_analysis, key=lambda x: -(x.get("reversal_ret10") or -999))[:15]:
        plan_a_d = pd.to_datetime(la["plan_a_date"]).strftime("%Y-%m-%d")
        rev_d = pd.to_datetime(la["reversal_date"]).strftime("%Y-%m-%d")
        r10 = f"{la.get('reversal_ret10', 0):+.2f}%" if la.get("reversal_ret10") is not None else "—"
        a_r10 = f"{la.get('plan_a_ret10', 0):+.2f}%" if la.get("plan_a_ret10") is not None else "—"
        lead_rows += f"""<tr>
          <td>{la['ticker']}</td>
          <td>{rev_d}</td>
          <td>{plan_a_d}</td>
          <td><b style="color:#22c55e;">{la['days_lead']} 天</b></td>
          <td>{r10}</td>
          <td>{a_r10}</td></tr>"""

    # 比較分析
    diff_win = s_r["win_num"] - s_a["win_num"]
    diff_col = "#22c55e" if diff_win > 0 else "#ef4444"

    html = f"""<!DOCTYPE html><html lang="zh-Hant"><head><meta charset="UTF-8">
<title>v5 反轉預警回測（S&P 100）</title><style>
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
              padding:20px; border-radius:10px; border:2px solid {diff_col};
              margin:20px 0; }}
</style></head><body>

<h1>🎯 v5 反轉預警訊號回測（S&P 100）</h1>
<p style="color:#888;">產出：{datetime.now().strftime('%Y-%m-%d %H:%M')} | 成功 {len(all_results)} / 失敗 {len(failed_tickers)} 支</p>

<div class="verdict">
  <h2 style="margin-top:0;">📊 對照比較</h2>
  <div class="card-row">
    <div class="card" style="border-left-color:#facc15;">
      <div style="color:#facc15;font-weight:bold;">⚡ 方案 A 起漲訊號（基準）</div>
      <div style="margin-top:6px;">數：<b>{s_a['count']}</b></div>
      <div>勝率：<span class="big" style="font-size:24px;">{s_a['win_rate']}</span></div>
      <div>平均：{s_a['avg']}</div>
      <div>中位：{s_a['median']}</div>
    </div>
    <div class="card" style="border-left-color:#22c55e;">
      <div style="color:#22c55e;font-weight:bold;">💡 反轉預警（新）</div>
      <div style="margin-top:6px;">數：<b>{s_r['count']}</b></div>
      <div>勝率：<span class="big" style="font-size:24px;color:{diff_col};">{s_r['win_rate']}</span></div>
      <div>平均：{s_r['avg']}</div>
      <div>中位：{s_r['median']}</div>
    </div>
    <div class="card" style="border-left-color:#dc2626;">
      <div style="color:#dc2626;font-weight:bold;">⏱️ 領先優勢</div>
      <div style="margin-top:6px;">領先案例：<b>{lead_count}</b> 個</div>
      <div>平均領先：<span class="big" style="font-size:24px;">{avg_lead:.1f} 天</span></div>
      <div>中位領先：{median_lead:.0f} 天</div>
      <div style="color:#aaa;font-size:11px;margin-top:4px;">（反轉預警比方案 A 早多少天）</div>
    </div>
  </div>
  <p style="color:#aaa;margin-top:14px;">
    <b>採用條件</b>：勝率 ≥ 55% + 領先 ≥ 5 天 + 至少 30 個樣本 ✅<br>
    <b>實測結果</b>：勝率 {s_r['win_rate']} {' ✅ 通過' if s_r['win_num'] >= 55 else ' ❌ 未達標'} | 
    領先 {avg_lead:.1f} 天 {' ✅ 通過' if avg_lead >= 5 else ' ❌ 未達標'} | 
    樣本 {s_r['count']} {' ✅ 充足' if s_r['count'] >= 30 else ' ⚠️ 偏少'}
  </p>
</div>

<h2>⏱️ 領先案例展示（反轉預警 vs 方案 A）</h2>
<p style="color:#888;">同一支股票，反轉預警比方案 A 早出現了幾天，看後續實際表現對照。</p>
<table>
  <tr><th>股票</th><th>反轉預警日</th><th>方案 A 日</th><th>領先天數</th>
    <th>反轉預警 +10日</th><th>方案 A +10日</th></tr>
  {lead_rows if lead_rows else '<tr><td colspan="6" style="color:#888;">無領先案例</td></tr>'}
</table>

<h2>📋 反轉預警代表案例（前 20 強）</h2>
<table>
  <tr><th>股票</th><th>日期</th><th>RSI 反轉</th><th>OBV</th><th>距 SMA60</th>
    <th>+5日</th><th>+10日</th><th>+20日</th><th>分類</th></tr>
  {rev_rows if rev_rows else '<tr><td colspan="9" style="color:#888;">無資料</td></tr>'}
</table>

</body></html>
"""
    return html


if __name__ == "__main__":
    main()

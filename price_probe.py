# -*- coding: utf-8 -*-
"""
price_probe.py — 診斷「掃描價錯誤（ETF 尤其誇張）」
在本機跑：  python price_probe.py
把整段輸出貼回給 Claude。

目的：分辨是 (A) yfinance 回的資料本身就錯，還是 (B) app 的清理步驟弄壞。
做法：對同一檔股票，用三種抓法取現價 + 走一遍 app 掃描路徑的清理，並列比對。
"""
import sys

import numpy as np
import pandas as pd
import yfinance as yf

try:
    import requests
except Exception:
    requests = None

TICKERS = [
    "0050.TW",      # ETF，回報 44518.21（可疑）
    "00878.TW",     # ETF，回報 43800.52（可疑）
    "00403A.TW",    # ETF
    "2330.TW",      # 個股對照組（個股頁顯示 2350，正確）
    "3037.TW",      # 個股對照組
]

MIS_URL = "https://mis.twse.com.tw/stock/api/getStockInfo.jsp"


def mis_price(tk):
    """證交所 MIS 即時價，當作「真值」對照。"""
    if requests is None:
        return "requests 未安裝"
    try:
        ex = ("otc_" if ".TWO" in tk else "tse_") + tk.split(".")[0] + ".tw"
        r = requests.get(
            MIS_URL,
            params={"ex_ch": ex, "json": "1", "delay": "0"},
            headers={"User-Agent": "Mozilla/5.0",
                     "Referer": "https://mis.twse.com.tw/stock/index.jsp"},
            timeout=5,
        )
        arr = r.json().get("msgArray") or []
        if not arr:
            return "無資料"
        d = arr[0]
        return f"成交z={d.get('z')} 昨收y={d.get('y')} 名稱={d.get('n')}"
    except Exception as e:
        return f"失敗 {type(e).__name__}: {e}"


def app_clean(d):
    """完全複製 app 掃描路徑的清理步驟（scan_watchlist_icons）。"""
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = d.columns.get_level_values(0)
    d = d.loc[:, ~d.columns.duplicated()]
    d = d.replace([np.inf, -np.inf], np.nan)
    d = d.dropna(subset=["Open", "High", "Low", "Close"])
    return d


print("=" * 70)
print("python  :", sys.version.split()[0])
print("yfinance:", getattr(yf, "__version__", "unknown"))
print("pandas  :", pd.__version__)
print("=" * 70)

for tk in TICKERS:
    print("\n" + "#" * 70)
    print("#", tk)
    print("#" * 70)

    # 真值對照
    print("[MIS 即時]", mis_price(tk))

    # --- 1) 完全比照 app 掃描的抓法 ---
    try:
        raw = yf.download(tk, period="1y", interval="1d",
                          auto_adjust=False, progress=False)
        print("\n[A] yf.download(auto_adjust=False)  ← app 掃描用這個")
        print("    shape:", raw.shape)
        print("    columns(raw):", list(raw.columns)[:8])
        print("    MultiIndex:", isinstance(raw.columns, pd.MultiIndex))
        if not raw.empty:
            d = app_clean(raw.copy())
            print("    清理後 columns:", list(d.columns))
            print("    清理後 shape  :", d.shape)
            if "Close" in d.columns:
                cl = d["Close"]
                print("    Close 型別    :", type(cl).__name__,
                      "(若是 DataFrame 就有問題)")
                print("    最後3筆 Close :", [round(float(x), 2) for x in cl.iloc[-3:]])
                print("    >>> app 會存的現價 =", round(float(cl.iloc[-1]), 2))
            # 各欄最後一筆，看是不是欄位錯位
            print("    最後一列各欄  :")
            for c in d.columns[:8]:
                try:
                    print(f"        {c:<12} = {float(d[c].iloc[-1]):,.2f}")
                except Exception as e:
                    print(f"        {c:<12} = <{type(e).__name__}>")
    except Exception as e:
        print("    [A] 例外:", type(e).__name__, e)

    # --- 2) auto_adjust=True 對照 ---
    try:
        r2 = yf.download(tk, period="5d", interval="1d",
                         auto_adjust=True, progress=False)
        if not r2.empty:
            d2 = app_clean(r2.copy())
            print("\n[B] yf.download(auto_adjust=True) 最後 Close =",
                  round(float(d2["Close"].iloc[-1]), 2))
    except Exception as e:
        print("\n[B] 例外:", type(e).__name__, e)

    # --- 3) Ticker.history() 對照 ---
    try:
        h = yf.Ticker(tk).history(period="5d")
        if not h.empty:
            print("[C] Ticker.history() 最後 Close =",
                  round(float(h["Close"].iloc[-1]), 2))
        else:
            print("[C] Ticker.history() 空資料")
    except Exception as e:
        print("[C] 例外:", type(e).__name__, e)

print("\n" + "=" * 70)
print("跑完了。請把上面全部輸出貼回給 Claude。")
print("=" * 70)

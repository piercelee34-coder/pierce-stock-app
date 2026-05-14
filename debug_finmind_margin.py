#!/usr/bin/env python3
"""
debug_finmind_margin_v2.py — 診斷 FinMind 融資餘額抓取問題

用法（建議用環境變數，不要寫進檔案）：

  PowerShell:
    $env:FINMIND_TOKEN="你的token"
    python debug_finmind_margin_v2.py

  或直接編輯這個檔案，只改下面那一行 TOKEN_HERE：
"""
import os
import requests

# ⚠️ 只有一個地方要改 — 把你的 token 貼在引號之間
TOKEN_HERE = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoia2Fpc2xpcGtub3QzNCIsImVtYWlsIjoicGllcmNlLmxlZS4zNEBnbWFpbC5jb20iLCJ0b2tlbl92ZXJzaW9uIjowfQ.9T3Y6zCiy5OlAlTyewu-p3THfao8Haz7jz1lj0o6J7Q"

# ─────────────────────────────────────────
FINMIND_TOKEN = os.environ.get("FINMIND_TOKEN", TOKEN_HERE).strip()

if not FINMIND_TOKEN:
    print("❌ 請先設定 FINMIND_TOKEN（環境變數或填到 TOKEN_HERE）")
    raise SystemExit(1)

print(f"Token: {FINMIND_TOKEN[:8]}...{FINMIND_TOKEN[-8:]} ({len(FINMIND_TOKEN)} 字元)")
print()

url = "https://api.finmindtrade.com/api/v4/data"


def run_test(name, params=None, headers=None):
    print("=" * 60)
    print(f"  {name}")
    print("=" * 60)
    try:
        r = requests.get(url, params=params or {}, headers=headers or {}, timeout=30)
        print(f"HTTP {r.status_code}")
        try:
            j = r.json()
        except Exception:
            print(f"非 JSON 回應，前 500 字: {r.text[:500]}")
            print()
            return
        print(f"msg: {j.get('msg')}")
        data = j.get("data", [])
        print(f"資料筆數: {len(data)}")
        if data:
            print(f"前 3 筆:")
            for row in data[:3]:
                print(f"  {row}")
            names = set(row.get("name") for row in data if isinstance(row, dict))
            if names and {None} != names:
                print(f"所有 name 值: {sorted(n for n in names if n)}")
        else:
            # 看 raw response 是不是有錯誤訊息
            raw_keys = list(j.keys())
            print(f"回應結構: {raw_keys}")
            if "status" in j:
                print(f"status: {j['status']}")
    except Exception as e:
        print(f"例外: {e}")
    print()


# ─── 測試 1: 整體融資餘額 + token 在 params ───
run_test("1. TaiwanStockTotalMarginPurchaseShortSale (token in params)",
         params={
             "dataset": "TaiwanStockTotalMarginPurchaseShortSale",
             "start_date": "2024-01-01",
             "end_date": "2024-01-31",
             "token": FINMIND_TOKEN,
         })

# ─── 測試 2: 整體融資餘額 + Bearer header ───
run_test("2. 整體融資 (token in Authorization header)",
         params={
             "dataset": "TaiwanStockTotalMarginPurchaseShortSale",
             "start_date": "2024-01-01",
             "end_date": "2024-01-15",
         },
         headers={"Authorization": f"Bearer {FINMIND_TOKEN}"})

# ─── 測試 3: 個股版（2330）─── 應該一定有資料
run_test("3. 個股 2330 融資（驗證 token 可用）",
         params={
             "dataset": "TaiwanStockMarginPurchaseShortSale",
             "data_id": "2330",
             "start_date": "2024-01-01",
             "end_date": "2024-01-15",
             "token": FINMIND_TOKEN,
         })

# ─── 測試 4: 三大法人總計（另一個整體 dataset）─── 確認 token 用法
run_test("4. 三大法人總計 (整體 dataset 驗證)",
         params={
             "dataset": "TaiwanStockTotalInstitutionalInvestors",
             "start_date": "2024-01-01",
             "end_date": "2024-01-15",
             "token": FINMIND_TOKEN,
         })

# ─── 測試 5: 抓的時間用 2025-01（最近）─── 看是否舊資料才失敗
run_test("5. 整體融資 2025-01 (近期資料)",
         params={
             "dataset": "TaiwanStockTotalMarginPurchaseShortSale",
             "start_date": "2025-01-01",
             "end_date": "2025-01-15",
             "token": FINMIND_TOKEN,
         })

print("=" * 60)
print("  完成，把以上輸出貼回給 Claude")
print("=" * 60)
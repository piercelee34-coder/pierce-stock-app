"""
FRED API 診斷工具 — 不修改 app，直接從 secrets.toml 讀 key、打 FRED、印詳細結果。
用法：在專案根目錄 (與 .streamlit/ 同層) 執行：
    python diagnose_fred.py
"""
import os
import sys

# 從 .streamlit/secrets.toml 讀 key
SECRETS_PATH = os.path.join(".streamlit", "secrets.toml")
if not os.path.exists(SECRETS_PATH):
    print(f"❌ 找不到 {SECRETS_PATH}")
    sys.exit(1)

fred_key = ""
with open(SECRETS_PATH, encoding="utf-8") as f:
    for line in f:
        if line.strip().startswith("FRED_API_KEY"):
            # 形如 FRED_API_KEY = "ed944..."
            fred_key = line.split("=", 1)[1].strip().strip('"').strip("'")
            break

if not fred_key:
    print("❌ secrets.toml 裡找不到 FRED_API_KEY")
    sys.exit(1)

print(f"✅ 讀到 FRED_API_KEY，長度 {len(fred_key)}、前 6 碼 {fred_key[:6]}、末 4 碼 {fred_key[-4:]}")
if " " in fred_key or "\t" in fred_key:
    print("⚠️  key 內含空白字元！這會導致 401 — 請去 secrets.toml 移除空白")

# 打 FRED
try:
    import requests
except ImportError:
    print("❌ 沒裝 requests，跑 pip install requests")
    sys.exit(1)

URL = "https://api.stlouisfed.org/fred/series/observations"
for series_id in ["T10Y2Y", "BAMLH0A0HYM2"]:
    print(f"\n--- 測試 {series_id} ---")
    try:
        r = requests.get(URL, params={
            "series_id": series_id,
            "api_key": fred_key,
            "file_type": "json",
            "limit": 1,
        }, timeout=15)
        print(f"HTTP status: {r.status_code}")
        if r.status_code == 200:
            obs = r.json().get("observations", [])
            print(f"✅ 成功，取到 {len(obs)} 筆觀測；最後一筆：{obs[-1] if obs else 'N/A'}")
        else:
            # 失敗時 FRED 會回 JSON 含 error_message
            try:
                err = r.json()
                print(f"❌ FRED 回錯誤：{err}")
            except Exception:
                print(f"❌ 回應內文（前 200 字）：{r.text[:200]}")
    except requests.exceptions.Timeout:
        print("❌ 逾時（FRED 伺服器 15 秒內無回應）")
    except requests.exceptions.ConnectionError as e:
        print(f"❌ 連線失敗：{e}")
    except Exception as e:
        print(f"❌ 例外：{type(e).__name__}: {e}")

print("\n常見錯誤對照：")
print("  status 400 + 'api_key is not registered' → key 沒效，去 FRED 重新申請")
print("  status 400 + 'Bad Request' → key 格式錯（含空白？前後引號？）")
print("  status 429 → 短時間打太多次（等 60 秒）")
print("  Timeout / ConnectionError → 網路問題或防火牆擋 stlouisfed.org")

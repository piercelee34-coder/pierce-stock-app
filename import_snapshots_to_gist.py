"""
一次性匯入工具：把本機歷史快照 Excel 灌進 Gist 的 snapshots.json
讓校準推演線今天就有資料、不用等 6 天累積。

用法（在專案根目錄，與 .streamlit/ 同層執行）：
    python import_snapshots_to_gist.py

它會：
  1. 從 .streamlit/secrets.toml 讀 GIST_TOKEN / GIST_ID（token 不外流）
  2. 掃描目前資料夾 + snapshots/ 子資料夾裡所有 scenario_snapshot_*.xlsx
  3. 合併進 Gist 既有的 snapshots.json（同日期覆蓋，保留最近 120 天）
  4. 上傳，並印出結果

安全：token 只在你本機讀取使用，不寫進任何檔案、不上傳到別處。
"""
import os
import sys
import glob
import json

# ── 1. 讀 secrets ──
SECRETS = os.path.join(".streamlit", "secrets.toml")
if not os.path.exists(SECRETS):
    print(f"❌ 找不到 {SECRETS}，請在專案根目錄執行")
    sys.exit(1)

token = gist_id = ""
with open(SECRETS, encoding="utf-8") as f:
    for line in f:
        s = line.strip()
        if s.startswith("GIST_TOKEN"):
            token = s.split("=", 1)[1].strip().strip('"').strip("'")
        elif s.startswith("GIST_ID"):
            gist_id = s.split("=", 1)[1].strip().strip('"').strip("'")

if not (token and gist_id):
    print("❌ secrets.toml 裡找不到 GIST_TOKEN / GIST_ID")
    sys.exit(1)
print(f"✅ 讀到 Gist 認證（ID 末4碼 {gist_id[-4:]}）")

try:
    import pandas as pd
    import requests
except ImportError as e:
    print(f"❌ 缺套件：{e}，請 pip install pandas requests openpyxl")
    sys.exit(1)

API = "https://api.github.com/gists"
HEADERS = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}
SNAPSHOT_FILE = "snapshots.json"

# ── 2. 掃描本機快照 Excel ──
patterns = [
    "scenario_snapshot_*.xlsx",
    os.path.join("snapshots", "*", "scenario_snapshot_*.xlsx"),
    os.path.join("snapshots", "scenario_snapshot_*.xlsx"),
]
files = []
for p in patterns:
    files.extend(glob.glob(p))
files = sorted(set(files))
if not files:
    print("❌ 找不到任何 scenario_snapshot_*.xlsx")
    sys.exit(1)
print(f"✅ 找到 {len(files)} 個快照檔")

# ── 3. 讀 Gist 現有歷史 ──
try:
    r = requests.get(f"{API}/{gist_id}", headers=HEADERS, timeout=15)
    r.raise_for_status()
    existing = r.json().get("files", {}).get(SNAPSHOT_FILE)
    if existing and "content" in existing:
        if existing.get("truncated") and existing.get("raw_url"):
            hist = json.loads(requests.get(existing["raw_url"], headers=HEADERS, timeout=15).text)
        else:
            hist = json.loads(existing["content"])
    else:
        hist = {}
    print(f"✅ Gist 既有 {len(hist)} 天歷史，準備合併")
except Exception as e:
    print(f"❌ 讀 Gist 失敗：{e}")
    sys.exit(1)

# ── 4. 合併每個 Excel ──
added = 0
for f in files:
    try:
        df = pd.read_excel(f, sheet_name="劇本快照")
        if "快照日期" not in df.columns or df.empty:
            continue
        d = str(pd.to_datetime(df["快照日期"].iloc[0]).date())
        recs = json.loads(df.to_json(orient="records", date_format="iso"))
        hist[d] = recs
        added += 1
        print(f"  + {d}（{len(recs)} 檔）")
    except Exception as e:
        print(f"  ⚠️ 跳過 {os.path.basename(f)}：{str(e)[:40]}")

# 保留最近 120 天
if len(hist) > 120:
    for k in sorted(hist.keys())[:-120]:
        del hist[k]

# ── 5. 上傳 ──
try:
    body = {"files": {SNAPSHOT_FILE: {"content": json.dumps(hist, ensure_ascii=False)}}}
    r = requests.patch(f"{API}/{gist_id}", headers=HEADERS, json=body, timeout=20)
    if r.status_code == 200:
        print(f"\n🎉 完成！已匯入 {added} 天，Gist 現共 {len(hist)} 天歷史快照")
        print("   現在打開 app，校準推演線就有資料可畫了。")
    else:
        print(f"❌ 上傳失敗 HTTP {r.status_code}：{r.text[:120]}")
except Exception as e:
    print(f"❌ 上傳例外：{e}")

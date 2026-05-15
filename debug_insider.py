#!/usr/bin/env python3
"""
debug_insider.py — 診斷 SEC Form 4 抓取為什麼是 0

用法：python debug_insider.py
"""
import os
import sys
import requests
import time

# 從 secrets.toml 讀 user agent
SEC_USER_AGENT = ""
try:
    import tomllib
    with open(".streamlit/secrets.toml", "rb") as f:
        s = tomllib.load(f)
    SEC_USER_AGENT = s.get("SEC_USER_AGENT", "")
except Exception:
    pass

if not SEC_USER_AGENT:
    SEC_USER_AGENT = "Stock App Test test@example.com"

print(f"User-Agent: {SEC_USER_AGENT}")
print()

# 1. 測試 NVDA (CIK = 1045810) 抓 Form 4 清單
print("━━ 步驟 1: 抓 NVDA 申報清單 ━━")
cik = "0001045810"  # NVDA
url = f"https://data.sec.gov/submissions/CIK{cik}.json"
headers = {"User-Agent": SEC_USER_AGENT, "Accept": "application/json", "Host": "data.sec.gov"}
r = requests.get(url, headers=headers, timeout=15)
print(f"HTTP {r.status_code}")

if r.status_code != 200:
    print(f"❌ 失敗: {r.text[:300]}")
    sys.exit(1)

j = r.json()
recent = j.get("filings", {}).get("recent", {})
forms = recent.get("form", [])
form4_indices = [i for i, f in enumerate(forms) if f == "4"]
print(f"✅ 取得 {len(forms)} 個申報，其中 Form 4 共 {len(form4_indices)} 件")

if not form4_indices:
    print("❌ 沒有 Form 4")
    sys.exit(1)

# 取最近一件 Form 4
i = form4_indices[0]
accession = recent["accessionNumber"][i]
filing_date = recent["filingDate"][i]
primary_doc = recent["primaryDocument"][i]
print(f"   最近一件: {filing_date}, accession={accession}, doc={primary_doc}")

# 2. 嘗試抓 index.json 找出所有檔案
print()
print("━━ 步驟 2: 找 Form 4 XML 檔名 ━━")
time.sleep(0.2)
acc_clean = accession.replace("-", "")
# index.json 列出該 filing 的所有檔案
index_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_clean}/index.json"
r2 = requests.get(index_url,
                   headers={"User-Agent": SEC_USER_AGENT, "Host": "www.sec.gov",
                             "Accept": "*/*"},
                   timeout=15)
print(f"HTTP {r2.status_code}")
if r2.status_code != 200:
    print(f"❌ index.json 失敗: {r2.text[:200]}")
    print("試試直接從 primaryDocument 推導")
else:
    idx = r2.json()
    items = idx.get("directory", {}).get("item", [])
    print(f"該 filing 含 {len(items)} 個檔案：")
    for item in items:
        print(f"   - {item.get('name')} ({item.get('type')})")

# 3. 試幾種可能的 XML URL
print()
print("━━ 步驟 3: 試 XML URL ━━")
candidates = [
    primary_doc,
    primary_doc.replace(".html", ".xml").replace(".htm", ".xml"),
    primary_doc.rsplit(".", 1)[0] + ".xml",
    "primary_doc.xml",
    f"xslF345X05/{primary_doc}",
]
for cand in candidates:
    time.sleep(0.2)
    test_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_clean}/{cand}"
    r3 = requests.get(test_url,
                       headers={"User-Agent": SEC_USER_AGENT, "Host": "www.sec.gov",
                                 "Accept": "*/*"},
                       timeout=10)
    ok = r3.status_code == 200
    is_xml = "xml" in r3.headers.get("Content-Type", "").lower() if ok else False
    print(f"   {cand}: HTTP {r3.status_code}{'  (XML)' if is_xml else ''}")
    if ok and is_xml and len(r3.content) > 500:
        print(f"   ✅ 找到 XML！前 500 字：")
        print(f"   {r3.content[:500].decode('utf-8', errors='ignore')}")
        # 嘗試解析
        import xml.etree.ElementTree as ET
        try:
            root = ET.fromstring(r3.content)
            txs = root.findall(".//nonDerivativeTransaction")
            print(f"   解析到 {len(txs)} 個 nonDerivativeTransaction")
            for tx in txs[:2]:
                code_e = tx.find(".//transactionCode")
                shares_e = tx.find(".//transactionShares/value")
                price_e = tx.find(".//transactionPricePerShare/value")
                code = code_e.text if code_e is not None else "?"
                shares = shares_e.text if shares_e is not None else "?"
                price = price_e.text if price_e is not None else "?"
                print(f"     code={code}, shares={shares}, price={price}")
        except Exception as e:
            print(f"   ❌ XML 解析失敗: {e}")
        break
else:
    print("❌ 所有候選 URL 都失敗")
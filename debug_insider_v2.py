#!/usr/bin/env python3
"""debug_insider_v2.py — 找出 Form 4 純 XML 的路徑"""
import os, requests, time, sys

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

print(f"User-Agent: {SEC_USER_AGENT}\n")

# 從 NVDA 找一件最近的 Form 4
cik = "0001045810"
r = requests.get(
    f"https://data.sec.gov/submissions/CIK{cik}.json",
    headers={"User-Agent": SEC_USER_AGENT, "Accept": "application/json",
              "Host": "data.sec.gov"}, timeout=15,
)
j = r.json()
recent = j["filings"]["recent"]
forms = recent["form"]
i = next(idx for idx, f in enumerate(forms) if f == "4")
accession = recent["accessionNumber"][i]
primary_doc = recent["primaryDocument"][i]
filing_date = recent["filingDate"][i]
print(f"Form 4: {filing_date}, doc='{primary_doc}'\n")

acc_clean = accession.replace("-", "")

# primary_doc 是 'xslF345X06/wk-form4_xxx.xml'
# 真正的純 XML 應該是 'wk-form4_xxx.xml'（拿掉 XSL 前綴）
# 因為 XSL 路徑只是用來「轉成可閱讀的 HTML」，純 XML 才有結構化資料

# 找出檔名部分
filename = primary_doc.split("/")[-1]
print(f"純 XML 檔名應該是: {filename}\n")

time.sleep(0.5)

# 試直接抓純 XML
print("━━ 試 1: 直接抓 純 XML（拿掉 xsl 前綴） ━━")
url1 = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_clean}/{filename}"
r1 = requests.get(url1, headers={"User-Agent": SEC_USER_AGENT, "Host": "www.sec.gov"}, timeout=10)
print(f"URL: {url1}")
print(f"HTTP {r1.status_code}, Content-Type: {r1.headers.get('Content-Type','?')}, 長度: {len(r1.content)}")
if r1.status_code == 200 and len(r1.content) > 200:
    head = r1.content[:300].decode("utf-8", errors="ignore")
    if "<ownershipDocument>" in head or "<?xml" in head:
        print("✅ 是純 XML！")
        # 解析
        import xml.etree.ElementTree as ET
        try:
            root = ET.fromstring(r1.content)
            txs_non = root.findall(".//nonDerivativeTransaction")
            print(f"   nonDerivativeTransaction: {len(txs_non)}")
            for tx in txs_non:
                code_e = tx.find(".//transactionCode")
                shares_e = tx.find(".//transactionShares/value")
                price_e = tx.find(".//transactionPricePerShare/value")
                code = code_e.text if code_e is not None else "?"
                shares = shares_e.text if shares_e is not None else "?"
                price = price_e.text if price_e is not None else "?"
                print(f"     code={code}, shares={shares}, price={price}")

            rel = root.find(".//reportingOwnerRelationship")
            if rel is not None:
                for tag in ("isOfficer", "isDirector", "isTenPercentOwner"):
                    e = rel.find(tag)
                    if e is not None and e.text:
                        print(f"   {tag} = {e.text}")
                t = rel.find("officerTitle")
                if t is not None and t.text:
                    print(f"   officerTitle = {t.text}")
        except Exception as e:
            print(f"❌ XML 解析失敗: {e}")
            print(f"前 500 字: {head}")
    else:
        print(f"⚠️ 200 但內容不像 XML，前 300 字:\n   {head}")
else:
    print(f"❌ 失敗")

# 也測試 .txt 完整檔案，做 fallback
print()
print("━━ 試 2: .txt 完整檔（含所有元件）━━")
time.sleep(0.5)
txt_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_clean}/{accession}.txt"
r2 = requests.get(txt_url, headers={"User-Agent": SEC_USER_AGENT, "Host": "www.sec.gov"}, timeout=10)
print(f"HTTP {r2.status_code}, 長度: {len(r2.content)}")
if r2.status_code == 200 and "<ownershipDocument>" in r2.text:
    print("✅ .txt 含完整 XML，可用 regex 抽出")
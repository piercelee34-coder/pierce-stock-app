# -*- coding: utf-8 -*-
"""
name_probe.py — 驗證「台股名稱查錯人」假設
在本機跑：  python name_probe.py
把輸出貼回給 Claude。

背景：app 用 TWSE codeQuery（模糊搜尋）取 suggestions[0] 當名稱，
      但沒驗證回來的代碼是否等於查詢代碼 → 可能掛錯名字。
"""
import re

import requests

# 代碼 → 正確名稱（人工對照，用來自動判定對錯）
EXPECT = {
    "0050.TW": "元大台灣50",
    "00878.TW": "國泰永續高股息",
    "00403A.TW": "主動統一升級50",
    "2330.TW": "台積電",
    "3037.TW": "欣興",
    "5534.TW": "長虹",
    "3189.TW": "景碩",
    "2337.TW": "旺宏",
    "3455.TWO": "3455",
    "7610.TW": "聯友金屬",
}

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
BAD = ["Yahoo", "股市", "走勢", "無符合", "找不到", "代碼或名稱",
       "html", "TW", "TWO", "INC", "CORP", "LTD", "COMPANY"]


def twse(ticker):
    """完全複製 app 的第一順位查法。"""
    base = ticker.split(".")[0]
    try:
        r = requests.get(
            f"https://www.twse.com.tw/zh/api/codeQuery?query={base}",
            headers=HEADERS, timeout=5)
        if r.status_code != 200:
            return None, f"HTTP {r.status_code}", []
        data = r.json()
        sugs = data.get("suggestions", [])
        if not sugs:
            return None, "無 suggestions", []
        sug = sugs[0]
        if "無符合" in sug:
            return None, "無符合", sugs[:5]
        n = sug.replace(base, "").replace("\t", "").strip()
        if not re.search(r"[\u4e00-\u9fff]", n):
            return None, f"無中文字: {n!r}", sugs[:5]
        return n, "ok", sugs[:5]
    except Exception as e:
        return None, f"{type(e).__name__}: {e}", []


def cnyes(ticker):
    base = ticker.split(".")[0]
    pre = "OTC" if ".TWO" in ticker else "TWS"
    try:
        r = requests.get(
            f"https://ws.api.cnyes.com/ws/api/v1/quote/quotes/{pre}:{base}:STOCK",
            headers=HEADERS, timeout=5)
        if r.status_code != 200:
            return f"HTTP {r.status_code}"
        d = r.json().get("data") or []
        return (d[0].get("name") or "").strip() if d else "無資料"
    except Exception as e:
        return f"{type(e).__name__}"


def yahoo(ticker):
    try:
        r = requests.get(
            f"https://query2.finance.yahoo.com/v1/finance/search?q={ticker}",
            headers=HEADERS, timeout=5)
        if r.status_code != 200:
            return f"HTTP {r.status_code}"
        q = r.json().get("quotes", [])
        if not q:
            return "無資料"
        n = q[0].get("shortname") or q[0].get("longname") or ""
        return n.strip() if n else "無名稱"
    except Exception as e:
        return f"{type(e).__name__}"


print("=" * 78)
print(f"{'代碼':<12}{'應為':<18}{'TWSE(app用)':<20}{'判定':<8}")
print("=" * 78)

bad_rows = []
for tk, exp in EXPECT.items():
    got, status, sugs = twse(tk)
    if got is None:
        verdict = f"查無({status})"
    elif got == exp or exp in got or got in exp:
        verdict = "✅ 正確"
    else:
        verdict = "❌ 錯誤"
        bad_rows.append((tk, exp, got, sugs))
    print(f"{tk:<12}{exp:<18}{str(got):<20}{verdict:<8}")

print("\n" + "=" * 78)
print("錯誤明細（含 TWSE 回傳的前 5 筆 suggestions，看是不是模糊搜尋抓錯）")
print("=" * 78)
if not bad_rows:
    print("（無錯誤 → 名稱假設不成立，問題在價格那條路）")
else:
    for tk, exp, got, sugs in bad_rows:
        print(f"\n{tk}  應為 {exp}  但得到 {got}")
        for i, s in enumerate(sugs):
            print(f"    suggestions[{i}] = {s!r}")

print("\n" + "=" * 78)
print("備援來源對照（TWSE 失敗時 app 會走這兩個）")
print("=" * 78)
for tk, exp in EXPECT.items():
    print(f"{tk:<12} 應為 {exp:<18} cnyes={cnyes(tk):<18} yahoo={yahoo(tk)}")

print("\n跑完了，請把整段輸出貼回給 Claude。")

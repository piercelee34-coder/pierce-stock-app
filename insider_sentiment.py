"""
insider_sentiment.py — SEC Form 4 內部人賣壓指數

直接從 SEC EDGAR 公開 API（免費、不用 key）抓取 Form 4 申報資料。
SEC 要求：
  1. User-Agent 必須含 email
  2. Rate limit: 10 req/sec（程式內 throttle 到 8 req/sec）

策略：
  追蹤 S&P 100 大型權值股（市值最大、insider 動向最有代表性）
  抓近 30 日的 Form 4 → 統計 sell vs buy 金額比例
  排名 → 252 日百分位 → 0-100 分

對外函式：
  get_insider_pressure_index() → dict
    回傳結構：
    {
      "score": 73.2,                   # 0-100 危機分數（賣壓比例百分位）
      "level": ("🟠 高度危險", "#f97316"),
      "stats": {
        "sell_value": 45_000_000,      # 近 30 日總賣出金額
        "buy_value":  8_000_000,       # 近 30 日總買進金額
        "sell_ratio": 0.85,            # sell / (sell+buy)
        "sell_count": 152,             # 賣出筆數
        "buy_count":  18,              # 買進筆數
      },
      "top_sellers": [                 # 賣最多的 top 5 公司
        {"ticker": "NVDA", "value": 12_000_000, "insiders": ["CEO", "CFO"]},
        ...
      ],
      "top_buyers":  [...],            # 買最多的 top 5
      "data_status": True / False
    }
"""

import os
import json
import time
import requests
import pandas as pd
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

CACHE_DIR = ".insider_cache"
LOOKBACK_DAYS = 30
PERCENTILE_WINDOW = 252  # 1 年百分位

# SEC 要求 User-Agent 包含可聯絡 email
SEC_USER_AGENT = os.environ.get(
    "SEC_USER_AGENT",
    "Stock App Research contact@example.com",  # 改成你的 email
)
SEC_HEADERS = {
    "User-Agent": SEC_USER_AGENT,
    "Accept": "application/json",
    "Host": "data.sec.gov",
}

# S&P 100 主要權值股（代表性高、流動性好）
SP100_TICKERS = [
    "NVDA", "MSFT", "AAPL", "GOOG", "GOOGL", "AMZN", "META", "TSLA", "BRK.B",
    "AVGO", "JPM", "V", "WMT", "LLY", "MA", "ORCL", "XOM", "JNJ", "HD", "ABBV",
    "PG", "BAC", "COST", "KO", "TMUS", "PLTR", "CVX", "CSCO", "NFLX", "ABT",
    "WFC", "PEP", "CRM", "MRK", "ACN", "TMO", "AMD", "MCD", "GE", "ADBE",
    "DIS", "AXP", "LIN", "IBM", "CAT", "PM", "QCOM", "MS", "VZ", "GS",
    "INTU", "T", "ISRG", "RTX", "TXN", "NEE", "BX", "AMGN", "BKNG", "PFE",
    "C", "SCHW", "UPS", "LOW", "BLK", "DHR", "NOW", "UNH", "HON", "ELV",
    "SPGI", "ADP", "TJX", "SYK", "ETN", "DE", "GILD", "VRTX", "MMC", "PGR",
    "MDT", "PANW", "REGN", "MU", "ADI", "SBUX", "LMT", "BSX", "CMCSA", "BMY",
    "MO", "PYPL", "FI", "ICE", "DUK", "AMAT", "TGT", "MDLZ", "INTC", "USB",
]


# ──────────────────────────────────────────────────────
# 快取工具
# ──────────────────────────────────────────────────────
def _cache_path(key):
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, f"{key}.json")


def cache_get(key, max_age_hours=24):
    path = _cache_path(key)
    if not os.path.exists(path):
        return None
    if (time.time() - os.path.getmtime(path)) / 3600 > max_age_hours:
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def cache_set(key, data):
    try:
        with open(_cache_path(key), "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, default=str)
    except Exception:
        pass


# ──────────────────────────────────────────────────────
# SEC CIK 對照表（ticker → CIK）
# ──────────────────────────────────────────────────────
def get_cik_map():
    """從 SEC 抓 ticker → CIK 對照表，快取 7 天"""
    cached = cache_get("cik_map", max_age_hours=24 * 7)
    if cached:
        return cached
    try:
        r = requests.get(
            "https://www.sec.gov/files/company_tickers.json",
            headers={"User-Agent": SEC_USER_AGENT},
            timeout=15,
        )
        if r.status_code != 200:
            return {}
        data = r.json()
        # 結構：{"0": {"cik_str": 320193, "ticker": "AAPL", "title": "..."}, ...}
        cik_map = {}
        for entry in data.values():
            t = entry["ticker"].upper()
            cik = str(entry["cik_str"]).zfill(10)
            cik_map[t] = cik
        cache_set("cik_map", cik_map)
        return cik_map
    except Exception:
        return {}


# ──────────────────────────────────────────────────────
# 抓單一公司的 Form 4
# ──────────────────────────────────────────────────────
def _fetch_company_filings(cik, days=LOOKBACK_DAYS):
    """抓單一公司近 N 日的 Form 4 申報清單"""
    try:
        url = f"https://data.sec.gov/submissions/CIK{cik}.json"
        r = requests.get(url, headers=SEC_HEADERS, timeout=15)
        if r.status_code != 200:
            return []
        data = r.json()
        recent = data.get("filings", {}).get("recent", {})
        if not recent:
            return []
        forms = recent.get("form", [])
        accession_nos = recent.get("accessionNumber", [])
        filing_dates = recent.get("filingDate", [])
        primary_docs = recent.get("primaryDocument", [])

        cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        results = []
        for i, form in enumerate(forms):
            if form != "4":
                continue
            if filing_dates[i] < cutoff:
                break  # 已超出日期範圍（清單是時間倒序）
            results.append({
                "cik": cik,
                "accession": accession_nos[i],
                "date": filing_dates[i],
                "doc": primary_docs[i],
            })
        return results
    except Exception:
        return []


def _parse_form4_xml(cik, accession, doc):
    """從 SEC EDGAR 抓 Form 4 純 XML 並解析交易明細
    
    關鍵：primary_doc 通常是 'xslF345X06/wk-form4_xxx.xml' 這種帶 XSL 前綴的，
    但那是經過 XSL 轉換的 HTML 版。純 XML 在拿掉 xsl 前綴後的同名檔。
    """
    try:
        acc_no_clean = accession.replace("-", "")
        # 拿掉 XSL 前綴（xslF345X05/, xslF345X06/, 等等）
        filename = doc.split("/")[-1]  # 取最後一段檔名
        if not filename.endswith(".xml"):
            # 偶爾 primary_doc 是 .html，要強制改副檔名
            filename = filename.rsplit(".", 1)[0] + ".xml"

        url = (
            f"https://www.sec.gov/Archives/edgar/data/"
            f"{int(cik)}/{acc_no_clean}/{filename}"
        )
        r = requests.get(
            url,
            headers={"User-Agent": SEC_USER_AGENT,
                      "Host": "www.sec.gov", "Accept": "*/*"},
            timeout=15,
        )
        if r.status_code != 200 or len(r.content) < 200:
            return []
        # 確認是 XML（檢查內容前幾個字元）
        head = r.content[:200].decode("utf-8", errors="ignore")
        if "<ownershipDocument>" not in head and "<?xml" not in head:
            return []

        import xml.etree.ElementTree as ET
        try:
            root = ET.fromstring(r.content)
        except Exception:
            return []

        # 抓申報人身份
        owner_relationship = {
            "isOfficer": False, "isDirector": False, "isTenPercentOwner": False,
            "title": "",
        }
        rel = root.find(".//reportingOwnerRelationship")
        if rel is not None:
            for tag in ("isOfficer", "isDirector", "isTenPercentOwner"):
                elem = rel.find(tag)
                if elem is not None and elem.text and elem.text.strip() in ("1", "true"):
                    owner_relationship[tag] = True
            title_elem = rel.find("officerTitle")
            if title_elem is not None and title_elem.text:
                owner_relationship["title"] = title_elem.text.strip()

        # 抓所有非衍生交易（直接買賣）
        transactions = []
        for tx in root.findall(".//nonDerivativeTransaction"):
            code_elem = tx.find(".//transactionCode")
            shares_elem = tx.find(".//transactionShares/value")
            price_elem = tx.find(".//transactionPricePerShare/value")
            acq_disp_elem = tx.find(".//transactionAcquiredDisposedCode/value")

            if code_elem is None or shares_elem is None:
                continue

            code = code_elem.text.strip() if code_elem.text else ""
            try:
                shares = float(shares_elem.text)
                price = float(price_elem.text) if (price_elem is not None and price_elem.text) else 0
            except (ValueError, TypeError):
                continue

            # 只算 P (open market purchase) 和 S (open market sale)
            # 排除 A (award), M (option exercise), G (gift) 等非市場交易
            if code not in ("P", "S"):
                continue

            acq_disp = acq_disp_elem.text.strip() if (acq_disp_elem is not None and acq_disp_elem.text) else ""
            transactions.append({
                "code": code,
                "shares": shares,
                "price": price,
                "value": shares * price,
                "acquired": acq_disp == "A",
                "officer": owner_relationship["isOfficer"],
                "director": owner_relationship["isDirector"],
                "ten_pct": owner_relationship["isTenPercentOwner"],
                "title": owner_relationship["title"],
            })

        return transactions
    except Exception:
        return []


def _process_ticker(ticker, cik):
    """處理單一公司：抓申報 → 解析 → 彙總"""
    filings = _fetch_company_filings(cik)
    if not filings:
        return ticker, {"buy_value": 0, "sell_value": 0,
                         "buy_count": 0, "sell_count": 0,
                         "insiders": [], "filings_count": 0}

    total_buy = total_sell = 0
    buy_count = sell_count = 0
    insider_titles = set()

    for f in filings:
        time.sleep(0.13)  # 嚴格 ~8 req/sec
        txs = _parse_form4_xml(f["cik"], f["accession"], f["doc"])
        for tx in txs:
            if tx["code"] == "P":  # 買進
                total_buy += tx["value"]
                buy_count += 1
            elif tx["code"] == "S":  # 賣出
                total_sell += tx["value"]
                sell_count += 1
            # 記錄職位：優先用 title，沒 title 就用 director/officer 旗標
            if tx["title"]:
                title_l = tx["title"].lower()
                if any(k in title_l for k in
                        ("ceo", "cfo", "coo", "cto", "chief", "president", "director")):
                    insider_titles.add(tx["title"])
            elif tx["director"]:
                insider_titles.add("Director")
            elif tx["officer"]:
                insider_titles.add("Officer")
            elif tx["ten_pct"]:
                insider_titles.add("10% Owner")

    return ticker, {
        "buy_value": total_buy,
        "sell_value": total_sell,
        "buy_count": buy_count,
        "sell_count": sell_count,
        "insiders": sorted(insider_titles)[:5],
        "filings_count": len(filings),
    }


# ──────────────────────────────────────────────────────
# 主函式
# ──────────────────────────────────────────────────────
def get_insider_pressure_index(force_refresh=False, top_n=100):
    """
    回傳：
    {
      "score": 0-100,
      "level": (label, color),
      "stats": {...},
      "top_sellers": [...],
      "top_buyers":  [...],
      "by_ticker":   [...],  # 全部明細
      "data_status": bool,
      "updated_at":  str,
    }
    """
    cache_key = "insider_pressure"
    if not force_refresh:
        cached = cache_get(cache_key, max_age_hours=24)
        if cached:
            return cached

    cik_map = get_cik_map()
    if not cik_map:
        return _empty_result("無法取得 SEC CIK 對照表")

    # 篩選 SP100 中有 CIK 的
    target = [(t, cik_map[t]) for t in SP100_TICKERS[:top_n] if t in cik_map]
    if not target:
        return _empty_result("CIK 對照表為空")

    by_ticker = {}
    # 用 ThreadPoolExecutor 但限制 worker 數，避免超出 SEC rate limit
    with ThreadPoolExecutor(max_workers=4) as ex:
        futures = {ex.submit(_process_ticker, t, c): t for t, c in target}
        for fut in as_completed(futures):
            try:
                ticker, data = fut.result(timeout=120)
                by_ticker[ticker] = data
            except Exception:
                pass

    # 彙總
    total_buy = sum(v["buy_value"] for v in by_ticker.values())
    total_sell = sum(v["sell_value"] for v in by_ticker.values())
    total_buy_count = sum(v["buy_count"] for v in by_ticker.values())
    total_sell_count = sum(v["sell_count"] for v in by_ticker.values())

    # 計算賣壓比例 = sell / (sell + buy)
    if total_buy + total_sell == 0:
        sell_ratio = 0.5
    else:
        sell_ratio = total_sell / (total_buy + total_sell)

    # 用百分位校準（簡化版：用固定區間映射）
    # 歷史經驗：
    #   sell_ratio < 0.40 → 內部人在買 → 20 分（看多）
    #   sell_ratio = 0.50 → 平衡 → 40 分
    #   sell_ratio = 0.70 → 賣 > 買 → 60 分（警戒）
    #   sell_ratio = 0.85 → 集中賣出 → 80 分（危險）
    #   sell_ratio > 0.95 → 大規模賣壓 → 95 分
    if sell_ratio < 0.40:
        score = 15 + sell_ratio * 25  # 0-0.4 → 15-25
    elif sell_ratio < 0.50:
        score = 25 + (sell_ratio - 0.40) * 150  # 0.4-0.5 → 25-40
    elif sell_ratio < 0.70:
        score = 40 + (sell_ratio - 0.50) * 100  # 0.5-0.7 → 40-60
    elif sell_ratio < 0.85:
        score = 60 + (sell_ratio - 0.70) * 133.3  # 0.7-0.85 → 60-80
    else:
        score = 80 + (sell_ratio - 0.85) * 100  # 0.85-1.0 → 80-95
    score = round(min(95, score), 1)

    # 等級
    if   score >= 85: level = ("🔴 強制清倉", "#dc2626")
    elif score >= 75: level = ("🟠 高度危險", "#f97316")
    elif score >= 60: level = ("🟡 警戒區", "#facc15")
    elif score >= 40: level = ("⚖️ 中性區", "#9ca3af")
    elif score >= 20: level = ("🟢 機會浮現", "#84cc16")
    else:             level = ("💚 內部人加碼", "#22c55e")

    # 找賣最多/買最多的
    sellers = sorted(
        [(t, v) for t, v in by_ticker.items() if v["sell_value"] > 0],
        key=lambda x: -x[1]["sell_value"],
    )[:5]
    buyers = sorted(
        [(t, v) for t, v in by_ticker.items() if v["buy_value"] > 0],
        key=lambda x: -x[1]["buy_value"],
    )[:5]

    result = {
        "score": score,
        "level": level,
        "stats": {
            "sell_value": total_sell,
            "buy_value": total_buy,
            "sell_ratio": round(sell_ratio, 3),
            "sell_count": total_sell_count,
            "buy_count": total_buy_count,
            "companies_scanned": len(by_ticker),
        },
        "top_sellers": [
            {
                "ticker": t,
                "value": v["sell_value"],
                "count": v["sell_count"],
                "insiders": v["insiders"],
            }
            for t, v in sellers
        ],
        "top_buyers": [
            {
                "ticker": t,
                "value": v["buy_value"],
                "count": v["buy_count"],
                "insiders": v["insiders"],
            }
            for t, v in buyers
        ],
        "data_status": True,
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    cache_set(cache_key, result)
    return result


def _empty_result(reason):
    return {
        "score": None,
        "level": ("❓ N/A", "#666"),
        "stats": {},
        "top_sellers": [],
        "top_buyers": [],
        "data_status": False,
        "reason": reason,
    }

"""
event_nodes.py — 美股總經事件節點 + 個股財報日

提供未來會有大波動的關鍵日期，在走勢圖上以垂直虛線標記。
不預測方向，只標「會有事發生的日子」 — 100% 客觀資訊。

事件類型：
  📊 個股財報   → yfinance 抓（每股獨立）
  🏛️ FOMC 會議  → 寫死（2026 全年）
  📈 CPI 發布   → 寫死
  💼 非農就業   → 寫死
  📉 PPI 發布   → 寫死
"""

from datetime import datetime, date
import pandas as pd

# ──────────────────────────────────────────────────────
# 2026 美國總經事件日曆（公開政府行程，可確定）
# 資料來源：
#   - FOMC: https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm
#   - CPI/PPI/非農: https://www.bls.gov/schedule/news_release/
# 注意：日期可能因政府公告調整，使用者可在新年初手動更新
# ──────────────────────────────────────────────────────

# 2026 年 FOMC 會議（8 場，已含 2025 年底以便回溯顯示）
FOMC_DATES_2026 = [
    "2026-01-28",  # 1月會議
    "2026-03-18",  # 3月會議（含經濟預測）
    "2026-04-29",  # 4月會議
    "2026-06-17",  # 6月會議（含經濟預測）
    "2026-07-29",  # 7月會議
    "2026-09-16",  # 9月會議（含經濟預測）
    "2026-10-28",  # 10月會議
    "2026-12-09",  # 12月會議（含經濟預測）
]

# 2026 CPI 發布日（每月第二或第三個週二，08:30 ET）
CPI_DATES_2026 = [
    "2026-01-14", "2026-02-11", "2026-03-11", "2026-04-15",
    "2026-05-13", "2026-06-10", "2026-07-15", "2026-08-12",
    "2026-09-10", "2026-10-15", "2026-11-12", "2026-12-10",
]

# 2026 PPI 發布日（通常 CPI 前一天或後一天）
PPI_DATES_2026 = [
    "2026-01-13", "2026-02-12", "2026-03-12", "2026-04-14",
    "2026-05-14", "2026-06-11", "2026-07-14", "2026-08-13",
    "2026-09-11", "2026-10-14", "2026-11-13", "2026-12-11",
]

# 2026 非農就業（每月第一個週五 08:30 ET）
NFP_DATES_2026 = [
    "2026-01-02", "2026-02-06", "2026-03-06", "2026-04-03",
    "2026-05-01", "2026-06-05", "2026-07-03", "2026-08-07",
    "2026-09-04", "2026-10-02", "2026-11-06", "2026-12-04",
]

# 事件樣式設定
EVENT_STYLES = {
    "earnings": {
        "label": "📊 財報",
        "color": "#ef4444",   # 紅色（個股最重要）
        "dash": "dash",
        "width": 2,
        "priority": 10,        # 最高優先級
    },
    "fomc": {
        "label": "🏛️ FOMC",
        "color": "#f97316",   # 橘色
        "dash": "dash",
        "width": 2,
        "priority": 9,
    },
    "cpi": {
        "label": "📈 CPI",
        "color": "#eab308",   # 黃色
        "dash": "dot",
        "width": 1.5,
        "priority": 8,
    },
    "nfp": {
        "label": "💼 非農",
        "color": "#3b82f6",   # 藍色
        "dash": "dot",
        "width": 1.5,
        "priority": 7,
    },
    "ppi": {
        "label": "📉 PPI",
        "color": "#9ca3af",   # 灰色（次要）
        "dash": "dot",
        "width": 1,
        "priority": 5,
    },
}


def _parse_dates(date_strings):
    """轉成 datetime 物件清單"""
    return [pd.to_datetime(d) for d in date_strings]


def get_macro_events(start_date=None, end_date=None):
    """取得指定時間範圍內的所有總經事件
    
    Args:
        start_date: pd.Timestamp 或 None（None = 不過濾下限）
        end_date: pd.Timestamp 或 None
    
    Returns:
        list of dict: [{date, type, label, color, dash, width, priority}, ...]
    """
    events = []
    
    sources = [
        ("fomc", FOMC_DATES_2026),
        ("cpi", CPI_DATES_2026),
        ("ppi", PPI_DATES_2026),
        ("nfp", NFP_DATES_2026),
    ]
    
    for event_type, date_list in sources:
        for d_str in date_list:
            d = pd.to_datetime(d_str)
            if start_date is not None and d < pd.to_datetime(start_date):
                continue
            if end_date is not None and d > pd.to_datetime(end_date):
                continue
            style = EVENT_STYLES[event_type]
            events.append({
                "date": d,
                "type": event_type,
                "label": style["label"],
                "color": style["color"],
                "dash": style["dash"],
                "width": style["width"],
                "priority": style["priority"],
            })
    
    return events


def get_earnings_event(ticker, earnings_date):
    """單一個股財報事件
    
    Args:
        ticker: 股票代碼
        earnings_date: 財報日（str 或 datetime）
    
    Returns:
        dict 或 None
    """
    if earnings_date is None or earnings_date == "":
        return None
    try:
        d = pd.to_datetime(earnings_date)
        style = EVENT_STYLES["earnings"]
        return {
            "date": d,
            "type": "earnings",
            "label": f"📊 {ticker} 財報",
            "color": style["color"],
            "dash": style["dash"],
            "width": style["width"],
            "priority": style["priority"],
        }
    except Exception:
        return None


def get_all_events_for_chart(ticker=None, earnings_date=None, 
                                start_date=None, end_date=None):
    """整合所有事件（總經 + 個股財報），用於走勢圖標記
    
    Args:
        ticker: 個股代碼（None = 不抓財報）
        earnings_date: 個股財報日
        start_date / end_date: 時間範圍過濾
    
    Returns:
        list of events sorted by date
    """
    events = get_macro_events(start_date, end_date)
    
    if ticker and earnings_date:
        ern = get_earnings_event(ticker, earnings_date)
        if ern:
            # 檢查財報日是否在範圍內
            if start_date is None or ern["date"] >= pd.to_datetime(start_date):
                if end_date is None or ern["date"] <= pd.to_datetime(end_date):
                    events.append(ern)
    
    events.sort(key=lambda e: e["date"])
    return events


def is_us_stock(ticker):
    """判斷是否為美股（不含 .TW 等後綴）"""
    if not ticker:
        return False
    return "." not in ticker  # 簡化判斷：含 . 表示帶交易所代碼
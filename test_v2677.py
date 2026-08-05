# -*- coding: utf-8 -*-
"""V26.77 合成測試 — 驗證「為什麼」而不只是「做了什麼」。

covered:
  A. 8/4 事故回歸：整批慢一天必須被抓到（本功能存在的理由）。
  B. 就緒界線（美 08:00 / 台 14:00）：資料還沒該到的時段不誤報。
  C. 假日免疫：用交易日曆而非「平日」近似 —— 國定假日不誤報（採日曆庫的理由）。
  D. 反消音：整批一致的落後仍要報。「全部一樣=休市」的推論會把 8/4 消音，
     此測試存在的目的是讓未來任何人加這種推論時測試立刻紅掉。
  E. 市場歸屬沿用既有 ".TW" 慣例（含 .TWO）。
  F. asof 必須是字串（date 物件會讓 json.dump 在 except pass 裡靜默死掉）。
  G. 管線六觸點齊全（產出/pop/session/寫檔/讀檔/清除）—— 漏一點就靜默斷鏈。
"""
import json
import re
import sys
from datetime import datetime, timedelta

SRC = open("app_v18.py", encoding="utf-8", newline="").read().replace("\r\n", "\n")

# 從 app 原始碼挖出本輪新增的區塊執行，確保測的是真的那份程式碼
_m = re.search(r"\n# ─+\n# \[V26\.77\] 資料日基準.*?\n\nFINMIND_TOKEN", SRC, re.S)
assert _m, "抓不到 V26.77 資料日區塊 — 補丁沒套用或結構變了"
_block = _m.group(0).rsplit("\nFINMIND_TOKEN", 1)[0]


class _StubSt:  # get_expected_trade_dates 掛了 @st.cache_data，exec 需要替身
    @staticmethod
    def cache_data(**kw):
        return lambda f: f


_ns = {"st": _StubSt(), "timedelta": timedelta}
exec(_block, _ns)
pick = _ns["_pick_expected_trade_date"]
find_stale = _ns["_find_stale_asof"]

fails = []


def check(desc, got, want):
    if got != want:
        fails.append(f"  ✗ {desc}\n      got={got!r}\n      want={want!r}")


# 2026-07-27(一)~08-05(三)，NYSE/XTAI 實際皆無假日（沙箱以真日曆核對過）
DAYS = ["2026-07-27", "2026-07-28", "2026-07-29", "2026-07-30",
        "2026-07-31", "2026-08-03", "2026-08-04", "2026-08-05"]

# ── A. 8/4 事故回歸 ──
_now_incident = datetime(2026, 8, 4, 12, 16)  # TODO 記錄的實際時刻（台北）
check("A1: 8/4 12:16 美股預期日 = 08-03（週一已收盤逾 8 小時）",
      pick(DAYS, _now_incident, "US"), "2026-08-03")
_stale = find_stale({"SOFI": "2026-07-31", "^IXIC": "2026-07-31"},
                    {"US": "2026-08-03", "TW": "2026-08-04"},
                    ["SOFI", "^IXIC"])
check("A2: SOFI/^IXIC 資料日 07-31 → 兩檔都判落後",
      sorted(_stale), ["SOFI", "^IXIC"])
check("A3: 落後值帶 (資料日, 預期日) 供畫面顯示",
      _stale["SOFI"], ("2026-07-31", "2026-08-03"))

# ── B. 就緒界線 ──
check("B1: 8/4 07:00（美股資料未必就緒）→ 預期日退回 07-31，不誤報",
      pick(DAYS, datetime(2026, 8, 4, 7, 0), "US"), "2026-07-31")
check("B2: 8/4 08:00 整點起 → 預期日推進到 08-03",
      pick(DAYS, datetime(2026, 8, 4, 8, 0), "US"), "2026-08-03")
check("B3: 台股 8/4 13:59（未收盤+30分）→ 預期日 08-03",
      pick(DAYS, datetime(2026, 8, 4, 13, 59), "TW"), "2026-08-03")
check("B4: 台股 8/4 14:00 → 預期日 08-04",
      pick(DAYS, datetime(2026, 8, 4, 14, 0), "TW"), "2026-08-04")
check("B5: 週日晚間 → 兩市場預期日皆為上週五",
      (pick(DAYS, datetime(2026, 8, 2, 20, 0), "US"),
       pick(DAYS, datetime(2026, 8, 2, 20, 0), "TW")),
      ("2026-07-31", "2026-07-31"))

# ── C. 假日免疫（採日曆庫而非平日近似的理由）──
# 假想 08-04（週二）是國定假日：日曆裡沒有它。平日近似會要求 08-04 → 誤報；
# 日曆法預期日停在 08-03 → 資料日 08-03 判定正常。
DAYS_HOLIDAY = [d for d in DAYS if d != "2026-08-04"]
_exp_h = pick(DAYS_HOLIDAY, datetime(2026, 8, 5, 12, 0), "TW")
check("C1: 週間假日 → 預期日跳過假日停在 08-03", _exp_h, "2026-08-03")
check("C2: 假日隔天資料日 08-03 → 不誤報",
      find_stale({"2330.TW": "2026-08-03"}, {"TW": _exp_h}, ["2330.TW"]), {})

# ── D. 反消音（設計辯論的定案：整批一致仍要報）──
_uniform = {t: "2026-07-31" for t in ["A", "B", "C", "D", "E"]}
_st_u = find_stale(_uniform, {"US": "2026-08-03"}, list(_uniform))
check("D1: 五檔資料日完全一致地落後 → 五檔全報，不得推論為休市",
      len(_st_u), 5)
assert "整批一致" in _block and "消音" in _block, \
    "D2: 反消音的設計理由必須留在 docstring 裡（防止未來被『優化』掉）"

# ── E. 市場歸屬 ──
_st_mk = find_stale({"2330.TW": "2026-08-03", "6488.TWO": "2026-08-03",
                     "NVDA": "2026-08-03"},
                    {"US": "2026-08-04", "TW": "2026-08-03"},
                    ["2330.TW", "6488.TWO", "NVDA"])
check("E1: .TW/.TWO 走台股基準（08-03=正常）、NVDA 走美股基準（落後）",
      sorted(_st_mk), ["NVDA"])
check("E2: 資料日缺席（掃描失敗檔）→ 不判落後也不炸（None 安全）",
      find_stale({}, {"US": "2026-08-04"}, ["NVDA"]), {})

# ── F. JSON 字串紀律 ──
_asof_line = re.search(r'_prices_asof.*?strftime\("%Y-%m-%d"\)', SRC)
assert _asof_line, "F1: asof 產出行必須 strftime 成字串（date 物件會被 json.dump 的 except pass 靜默吃掉）"
json.dumps({"prices_asof": {"SOFI": "2026-07-31"}})  # F2: 可序列化（會丟就直接炸）

# ── G. 管線六觸點 ──
for _i, _pat in enumerate([
    r'icons\.setdefault\("_prices_asof"',                       # 1 產出
    r'_icons\.pop\("_prices_asof"',                             # 2 pop
    r'st\.session_state\["_wl_prices_asof"\] = _scan_prices_asof',  # 3 session
    r'"prices_asof": _scan_prices_asof',                        # 4 寫檔
    r'_cached\.get\("prices_asof"',                             # 5 讀檔
    r'"_wl_prices_asof",',                                      # 6 清除清單
], start=1):
    if not re.search(_pat, SRC):
        fails.append(f"  ✗ G{_i}: 管線觸點缺失 {_pat} — 漏一點整條鏈靜默斷掉")

# ── 收尾 ──
if fails:
    print("❌ 測試失敗：")
    print("\n".join(fails))
    sys.exit(1)
print("✅ 全部通過")

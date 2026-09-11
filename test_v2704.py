# -*- coding: utf-8 -*-
"""[V27.04] 二次進場進四個地方：持倉總表訊號欄 / 掃描中心 / 全市場 / 乖離抄底迷你圖。

核心問題（寫程式前先證明的）：
  compute_zigzag_pivots 的迴圈是 range(n, len(df) - n)
  → 訊號根與最後一根的距離**恆 >= n**，這是迴圈範圍決定的，跟資料無關。
  若照一般訊號用「低點日距今 < 2 根」當保鮮期，結果**恆為 0** ——
  會做出一個保證永遠不亮的功能。
  所以效期改從**確認日 = 低點日 + n** 起算。

covered:
  A. 上述不可能性（用程式碼本身證明，不是靠模擬）
  B. second_entry_recent 的確認日語意與邊界
  C. 四個接點都接上了，且共用同一支偵測器（Rule 7）
  D. 新舊對照帳沒被新訊號汙染（V26.97 ① 的校準還有效）
  E. 迷你圖遮罩在完整資料上算，不是對切片重算

執行：python3 test_v2704.py   （需從 repo root 跑）
"""
import ast
import sys

import numpy as np
import pandas as pd

APP = open("app_v18.py", encoding="utf-8", newline="").read().replace("\r\n", "\n")
TREE = ast.parse(APP)

FAILS = []
def check(label, cond, detail=""):
    line = ("✅ " if cond else "❌ ") + label + (f" — {detail}" if detail else "")
    print(line)
    if not cond:
        FAILS.append(line)

def grab(name):
    for n in TREE.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name:
            return ast.get_source_segment(APP, n)
    raise AssertionError(f"找不到 {name}")

def const(name):
    for n in TREE.body:
        if (isinstance(n, ast.Assign) and len(n.targets) == 1
                and isinstance(n.targets[0], ast.Name) and n.targets[0].id == name):
            return ast.literal_eval(n.value)
    raise AssertionError(f"找不到 {name}")

N = const("_SECOND_ENTRY_ZIGZAG_N")
FRESH = const("_SECOND_ENTRY_FRESH_DAYS")
g = {"pd": pd, "np": np,
     "_SECOND_ENTRY_ZIGZAG_N": N, "_SECOND_ENTRY_FRESH_DAYS": FRESH}
for fn in ("compute_zigzag_pivots", "detect_second_entry", "second_entry_recent"):
    exec(grab(fn), g)
detect, recent = g["detect_second_entry"], g["second_entry_recent"]


def mk(lows):
    n = len(lows)
    lows = np.asarray(lows, dtype=float)
    return pd.DataFrame({"Low": lows, "High": lows + 5, "Close": lows + 3,
                         "SMA_20": lows + 1, "SMA_60": lows - 5,
                         "RSI": np.full(n, 55.0)},
                        index=pd.bdate_range("2026-01-01", periods=n))


def pattern(span, i1, peak, i2, base=100.0, d1=90.0, d2=94.0):
    """低 → 峰 → 更高的低。中間的峰是必要的：ZigZag 強制高低交替，
    兩個低點之間沒有 H pivot 會被併成一個。"""
    lows = np.full(span, base)
    lows[i1], lows[peak], lows[i2] = d1, base + 8.0, d2
    return lows


# ══════════════════════════════════════════════════════════
# A. 「低點日保鮮 2 天」在結構上不可能 —— 從程式碼證明
# ══════════════════════════════════════════════════════════
_zz = grab("compute_zigzag_pivots")
check("A1 ZigZag 迴圈上界是 len(df) - n（訊號根距最後一根恆 >= n）",
      "for i in range(n, len(df) - n):" in _zz)

# 實例：把 higher low 盡可能往後擺，看它離最後一根多遠
_best = None
for span in (40, 60, 90):
    lows = pattern(span, span // 5, span // 2, span - N - 1)
    hits = np.flatnonzero(detect(mk(lows)).values)
    if len(hits):
        _best = (span - 1) - int(hits[-1])
check(f"A2 訊號最近也只能距最後一根 {N} 根（實例驗證）", _best == N, f"實測最小距離 = {_best}")
check("A3 → 用低點日算「2 天效期」恆為 0，功能會永遠不亮",
      not (_best < FRESH), f"{_best} < {FRESH} 為 False，證明不可行")

# ══════════════════════════════════════════════════════════
# B. 改用確認日之後的語意
# ══════════════════════════════════════════════════════════
span = 60
i2 = span - N - 1                       # 最後一個可能的 pivot 位置
d_fresh = mk(pattern(span, 12, 30, i2))
r = recent(d_fresh)
check("B1 剛確認（確認日 = 最後一根）→ 算新鮮", r is not None, f"{r}")
if r:
    check("B2 confirmed_pos = 低點日 + n", r["confirmed_pos"] == r["idx_pos"] + N,
          f"{r['idx_pos']} + {N} vs {r['confirmed_pos']}")
    check("B3 bars_since_confirm = 0（今天才確認）", r["bars_since_confirm"] == 0,
          f"{r['bars_since_confirm']}")
    check("B4 回報的日期是**低點日**（那才是進場點）",
          r["date"] == d_fresh.index[r["idx_pos"]])

# 往前推：低點再早 FRESH 根 → 確認日超出效期
d_stale = mk(pattern(span + FRESH, 12, 30, i2))
check(f"B5 確認日超過 {FRESH} 天 → 不再新鮮（回 None）",
      recent(d_stale) is None, f"{recent(d_stale)}")

# 邊界：確認日剛好在效期最後一天
d_edge = mk(pattern(span + FRESH - 1, 12, 30, i2))
check(f"B6 邊界：確認後第 {FRESH - 1} 天仍算新鮮",
      recent(d_edge) is not None, f"{recent(d_edge)}")

check("B7 沒有任何訊號時回 None（不是空 dict 或例外）",
      recent(mk(np.full(60, 100.0))) is None)
check("B8 資料太短不拋例外", recent(mk([100.0] * 3)) is None)

# ══════════════════════════════════════════════════════════
# C. 四個接點
# ══════════════════════════════════════════════════════════
check("C1 掃描中心（scan_personal_signals）有加",
      'hits.append(("🔁 二次進場",' in APP)
check("C2 二次進場在逐日切片迴圈**之外**判（它是型態不是當日狀態）",
      APP.index("_se = second_entry_recent(d)")
      < APP.index("for back in range(lookback_days):"))
check("C3 持倉總表/側邊欄圖示（scan_watchlist_icons）有加",
      'parts.append("🔁")' in APP)
check("C4 方向過濾把 🔁 歸進場向（未持倉表才看得到）",
      '"🔁") + _SIG_MONEYFLOW' in APP)
check("C5 乖離抄底迷你圖有畫", 'if "_SE" in mini.columns' in APP)
check("C6 四處共用同一支 detect_second_entry（沒有第二套判定）",
      APP.count("def detect_second_entry") == 1)
check("C7 台股/美股全市場自動涵蓋（都走 scan_personal_signals，不必另外接）",
      APP.count("scan_personal_signals(") >= 1
      and "_scan_tickers, lookback_days=3, stats=_sig_stats" in APP.replace("\n", " ").replace("  ", " "))

check("C10 戰情分組用原始 _icons_map（不套方向過濾）→ 🔁 會直接出現",
      '"訊號": (_icons_map.get(_tk, "") or "").replace("|", " ").strip(),' in APP)
check("C11 有持倉/未持倉編輯表才套方向過濾（🔁 歸進場向 → 只在未持倉表）",
      "filter_signal_by_direction(_icons_map.get(_t, \"\"), _sdir)" in APP)

# 說明文字
check("C8 掃描中心說明講清楚日期會是 n 天前、且是定義使然",
      "不是資料延遲" in APP)
check("C9 圖例用常數帶出效期，沒寫死數字",
      "{_SECOND_ENTRY_FRESH_DAYS} 交易日內" in APP)

# ══════════════════════════════════════════════════════════
# D. 不汙染 V26.97 的新舊對照帳
# ══════════════════════════════════════════════════════════
_sps = grab("scan_personal_signals")
check("D1 🔁 同時加進 _hit_types 與 _legacy_types",
      '_hit_types.add("🔁 二次進場")' in _sps and '_legacy_types.add("🔁 二次進場")' in _sps)
check("D2 → hit_legacy - hit 的差額仍只反映達標閘門（校準還有效）",
      _sps.count('_hit_types.add("🔁 二次進場")') == _sps.count('_legacy_types.add("🔁 二次進場")') == 1)

# ══════════════════════════════════════════════════════════
# E. 迷你圖遮罩算在完整資料上
# ══════════════════════════════════════════════════════════
check("E1 _SE 是對完整 d 算再 reindex 到 mini（不是對切片重算）",
      'detect_second_entry(d).reindex(' in APP and '_mini["_SE"]' in APP)
check("E2 舊快取沒有 _SE 欄時不炸（用 in 判斷）",
      'if "_SE" in mini.columns' in APP)
check("E3 算不出來時退成 False，不讓整張迷你圖消失",
      '_mini["_SE"] = False' in APP)

# ══════════════════════════════════════════════════════════
# F. 隔離：次要訊號不該殺掉主要訊號
# ══════════════════════════════════════════════════════════
#   second_entry_recent 的呼叫在 per-ticker 的大 try 裡面 —— 它一拋例外就會
#   觸發 continue，整檔連同達標/吸籌/抄底一起從結果消失。
#   實際發生過：加完接線後 test_v2696/v2697 全紅，就是這個原因。
_sps_src = grab("scan_personal_signals")
_call = _sps_src.index("_se = second_entry_recent(d)")
_before = _sps_src[:_call].rstrip()
check("F1 second_entry_recent 有自己的 try（不靠外層那個）",
      _before.endswith("try:"), repr(_before[-30:]))
check("F2 失敗時退成 None 而不是讓整檔消失",
      "_se = None" in _sps_src)

print()
if FAILS:
    print(f"❌ {len(FAILS)} 項失敗")
    sys.exit(1)
print("✅ 全部通過")

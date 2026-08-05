# -*- coding: utf-8 -*-
"""V26.76 合成測試 — 驗證「為什麼」而不只是「做了什麼」。

covered:
  A. 訊號方向過濾的商業意圖：持倉不該看到進場訊號、未持倉不該看到出場訊號。
  B. 主力進出是狀態不是事件，兩個方向都必須留（漏掉就等於拿掉唯一的資金流向資訊）。
  C. 損益兩欄的插入索引：訊號欄插入後若沒 +1，損益會靜默跑到成本前面（不會拋錯）。
  D. 格式異常時 fail loud / 原樣退回，不靜默吐空字串（Rule 12）。
"""
import re
import sys

import pandas as pd

SRC = open("app_v18.py", encoding="utf-8", newline="").read().replace("\r\n", "\n")

# 從 app 原始碼把函式挖出來執行，確保測的是真的那份程式碼而不是複製品
_m = re.search(r"\n_SIG_MONEYFLOW = .*?\n    return \" \"\.join\(out\)\.strip\(\)\n",
               SRC, re.S)
assert _m, "抓不到 filter_signal_by_direction 區塊 — 補丁沒套用或結構變了"
_ns = {}
exec(_m.group(0), _ns)
f = _ns["filter_signal_by_direction"]

fails = []


def check(desc, got, want):
    if got != want:
        fails.append(f"  ✗ {desc}\n      got={got!r}\n      want={want!r}")


# ── A. 方向意圖 ────────────────────────────────────────────────
# 一檔同時有達標(出場向)和炒底(進場向)：兩張表看到的必須不一樣，
# 否則這個功能等於沒做——使用者拿到的還是同一串混合訊號。
RAW_BOTH = "🟡🟣|🟢⬆ BUY 💎2 💰"
check("持倉只留出場向", f(RAW_BOTH, "exit"), "🟡 🟢⬆ 💰")
check("未持倉只留進場向", f(RAW_BOTH, "entry"), "🟣 🟢⬆ BUY 💎2")

# 死叉是最典型的出場訊號，絕不能從持倉表消失
check("SELL 必在持倉表", f("|🔴⬇ SELL", "exit"), "🔴⬇ SELL")
check("SELL 不進未持倉表", f("|🔴⬇ SELL", "entry"), "🔴⬇")

# 過熱只對持倉有行動意義（該不該獲利了結）
check("🔥 只在持倉表", f("|🔥", "exit"), "🔥")
check("🔥 不在未持倉表", f("|🔥", "entry"), "")

# ── B. 主力進出兩邊都留 ────────────────────────────────────────
# 這是唯一的資金流向資訊。若只在單邊出現，另一邊的使用者就完全看不到
# 「主力在跑」或「主力在收」，那是比 MACD 更關鍵的判斷依據。
for d in ("exit", "entry"):
    check(f"🟢⬆ 保留於 {d}", f("|🟢⬆", d), "🟢⬆")
    check(f"🔴⬇ 保留於 {d}", f("|🔴⬇", d), "🔴⬇")

# ── C. 💎N 的 N 是計次，不能被吃掉 ──────────────────────────────
# 炒底 1 次和炒底 3 次的意義差很多，用 startswith 比對時要確認數字有留下
check("💎 計次保留", f("|💎3", "entry"), "💎3")

# ── D. 邊界 / fail loud ───────────────────────────────────────
check("空字串", f("", "exit"), "")
check("None", f(None, "entry"), "")
check("只有色點無 token", f("🟡|", "exit"), "🟡")
check("色點被過濾掉時不留空白", f("🟣|", "exit"), "")
# 缺分隔符 = 格式不符預期，整串當 token 處理而非靜默丟棄
check("無分隔符不靜默吞掉", f("BUY 💎1", "entry"), "BUY 💎1")
try:
    f("|BUY", "sideways")
    fails.append("  ✗ 未知方向應該拋 ValueError，結果沒有（Rule 12 違反）")
except ValueError:
    pass

# ── E. 欄位順序 ────────────────────────────────────────────────
# 重現 app 裡的插入序列。訊號欄插在 index 2 之後，損益兩欄若還用舊的
# insert(5)/insert(6)，會跑到「成本」前面——pandas 不會報錯，畫面靜默錯位。
df = pd.DataFrame([{"代碼": "NVDA", "名稱": "輝達", "現價": 211.94,
                    "日%": 2.6, "成本": 196.0, "股數": 50}])
df.insert(2, "訊號", ["🟢⬆"])
# 以下兩行的索引必須跟 app_v18.py 裡的一致
m_idx = re.search(r'_edit_df_held\.insert\((\d+), "損益金額"', SRC)
p_idx = re.search(r'_edit_df_held\.insert\((\d+), "損益%"', SRC)
assert m_idx and p_idx, "抓不到損益欄的 insert 呼叫"
df.insert(int(m_idx.group(1)), "損益金額", ["$797"])
df.insert(int(p_idx.group(1)), "損益%", [8.1])
check("欄位順序", list(df.columns),
      ["代碼", "名稱", "訊號", "現價", "日%", "成本", "損益金額", "損益%", "股數"])

# 反證：舊索引(5/6)確實會壞，證明這個測試抓得到迴歸
bad = pd.DataFrame([{"代碼": "X", "名稱": "", "現價": 1.0,
                     "日%": 0.0, "成本": 1.0, "股數": 1}])
bad.insert(2, "訊號", [""])
bad.insert(5, "損益金額", ["—"]); bad.insert(6, "損益%", [None])
if list(bad.columns)[5] == "成本":
    fails.append("  ✗ 反證失敗：舊索引沒有造成錯位，這個測試無法偵測迴歸（Rule 9 違反）")

if fails:
    print(f"❌ {len(fails)} 項失敗：")
    print("\n".join(fails))
    sys.exit(1)
print("✅ 全部通過")

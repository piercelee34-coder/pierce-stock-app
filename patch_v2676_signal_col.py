# -*- coding: utf-8 -*-
"""
patch_v2676_signal_col.py  —  app_v18.py  V26.75 -> V26.76

改動：有持倉 / 未持倉 兩張編輯表加「訊號」欄（插在「名稱」後、唯讀）。
      持倉表只顯示出場向訊號，未持倉表只顯示進場向訊號（主力進出兩邊都留）。

用法：
    cd C:\\Users\\ksmoo\\OneDrive\\Desktop\\stockapp
    python patch_v2676_signal_col.py

會先備份成 app_v18.py.bak_v2675。任何一段找不到就整份中止，不會寫出半套。
"""
import io
import os
import shutil
import sys

TARGET = "app_v18.py"
BACKUP = "app_v18.py.bak_v2675"

EDITS = []


def edit(name, old, new):
    EDITS.append((name, old, new))


# ── 1. 版本號（page_title + 4 條 st.title 分支）──
edit("版本號 page_title",
     'page_title="AI 實戰戰情室 V26.75"',
     'page_title="AI 實戰戰情室 V26.76"')

edit("版本號 st.title",
     '''st.title("📡 掃描中心 V26.75" if cur_t == "__SCANNER__"
         else "🎯 訊號驗證 V26.75" if cur_t == "__VERIFY__"
         else "📊 持倉戰情總表 V26.75" if cur_t == "__DASHBOARD__"
         else f"📈 {disp_main_title} 實戰戰情室 V26.75")''',
     '''st.title("📡 掃描中心 V26.76" if cur_t == "__SCANNER__"
         else "🎯 訊號驗證 V26.76" if cur_t == "__VERIFY__"
         else "📊 持倉戰情總表 V26.76" if cur_t == "__DASHBOARD__"
         else f"📈 {disp_main_title} 實戰戰情室 V26.76")''')


# ── 2. 方向過濾函式（模組層，接在 scan_watchlist_icons 之後）──
_HELPER = '''

# [V26.76] 訊號方向過濾 — 供持倉 / 未持倉編輯表分別取用。
#   scan_watchlist_icons() 產出的字串格式為 "{色點前綴}|{空白分隔的訊號}"，
#   色點前綴是連在一起的字元（🟡🟣），後段是 token（🟢⬆ BUY 💎2 ...）。
#   direction="exit"  → 持倉關心的：達標 / 死叉 / 過熱
#   direction="entry" → 未持倉關心的：炒底 / 金叉 / 吸籌
#   主力進出（🟢⬆ / 🔴⬇）是「狀態」不是「事件」，兩個方向都保留。
_SIG_MONEYFLOW = ("🟢⬆", "🔴⬇")
_SIG_PREFIX_KEEP = {"exit": ("🟡",), "entry": ("🟣",)}
_SIG_TOKEN_KEEP = {
    "exit":  ("SELL", "💰", "🔥") + _SIG_MONEYFLOW,
    "entry": ("BUY", "💎", "🤫") + _SIG_MONEYFLOW,
}


def filter_signal_by_direction(raw, direction):
    """把 scan_watchlist_icons() 的訊號字串濾成單一方向。找不到方向時 fail loud。"""
    if direction not in _SIG_TOKEN_KEEP:
        raise ValueError(f"filter_signal_by_direction: 未知方向 {direction!r}")
    if not raw:
        return ""
    _pre, _sep, _rest = str(raw).partition("|")
    if not _sep:            # 沒有分隔符 = 格式與預期不符，整串原樣退回（不靜默吞掉）
        _pre, _rest = "", str(raw)
    _keep_pre = _SIG_PREFIX_KEEP[direction]
    _keep_tok = _SIG_TOKEN_KEEP[direction]
    out = [c for c in _pre if c in _keep_pre]
    out += [t for t in _rest.split() if t.startswith(_keep_tok)]
    return " ".join(out).strip()
'''

edit("插入 filter_signal_by_direction",
     '''            icons.setdefault("_errors", {})[tk] = str(_e)[:60]
            continue
    return icons
''',
     '''            icons.setdefault("_errors", {})[tk] = str(_e)[:60]
            continue
    return icons
''' + _HELPER)


# ── 3. 欄位設定：加「訊號」（唯讀）──
edit("column_config 加訊號欄",
     '''            "名稱": st.column_config.TextColumn("名稱", disabled=True, width="small"),
            "現價": st.column_config.NumberColumn("現價", disabled=True, format="%.2f", width="small"),''',
     '''            "名稱": st.column_config.TextColumn("名稱", disabled=True, width="small"),
            "訊號": st.column_config.TextColumn("訊號", disabled=True, width="small"),  # [V26.76]
            "現價": st.column_config.NumberColumn("現價", disabled=True, format="%.2f", width="small"),''')


# ── 4. 拆表之後，各自插入方向過濾過的訊號欄 ──
edit("拆表後插入訊號欄",
     '''        else:
            _edit_df_held = _edit_df.copy(); _edit_df_unheld = _edit_df.copy()

        # [V26.68] 有持倉區加「損益金額 / 損益%」兩欄（唯讀，插在 訊號 與 成本 之間）''',
     '''        else:
            _edit_df_held = _edit_df.copy(); _edit_df_unheld = _edit_df.copy()

        # [V26.76] 訊號欄插在「名稱」後（index 2），持倉取出場向、未持倉取進場向。
        #          注意：這一插會把後面所有欄位索引 +1，下面兩個 insert() 已同步調整。
        for _sdf, _sdir in ((_edit_df_held, "exit"), (_edit_df_unheld, "entry")):
            if "代碼" in _sdf.columns and "訊號" not in _sdf.columns:
                _sdf.insert(2, "訊號",
                            [filter_signal_by_direction(_icons_map.get(_t, ""), _sdir)
                             for _t in _sdf["代碼"]])

        # [V26.68] 有持倉區加「損益金額 / 損益%」兩欄（唯讀，插在 成本 與 股數 之間）''')


# ── 5. 損益兩欄的插入索引 +1（訊號欄插入後）──
edit("損益欄索引 +1",
     '''            _edit_df_held.insert(5, "損益金額", _pl_amt_col)  # [V26.74] +1（日%欄插入後）
            _edit_df_held.insert(6, "損益%", _pl_pct_col)''',
     '''            _edit_df_held.insert(6, "損益金額", _pl_amt_col)  # [V26.76] +1（訊號欄插入後）
            _edit_df_held.insert(7, "損益%", _pl_pct_col)''')


# ── 6. 圖示說明改成分方向 ──
edit("訊號說明分方向",
     '''        st.caption("訊號圖示：🟡達標　🟣炒底　🟢⬆吸籌　🔴⬇出貨　BUY／SELL（MACD 金／死叉）　🤫吸籌　💎N＝炒底N次　💰達標　🔥過熱（保鮮3交易日）")''',
     '''        st.caption("訊號圖示：🟡達標　🟣炒底　🟢⬆吸籌　🔴⬇出貨　BUY／SELL（MACD 金／死叉）　🤫吸籌　💎N＝炒底N次　💰達標　🔥過熱（保鮮3交易日）")
        # [V26.76] 兩張表的訊號欄已分方向過濾，說明清楚免得誤讀「沒訊號」
        st.caption("⚠️ 訊號欄已分方向：**有持倉只顯示出場向**（🟡 💰 SELL 🔥）、"
                   "**未持倉只顯示進場向**（🟣 💎 BUY 🤫），主力進出（🟢⬆／🔴⬇）兩邊都顯示。"
                   "持倉表看不到 BUY 不代表沒有，要看完整訊號請開個股頁。")''')


def main():
    if not os.path.exists(TARGET):
        sys.exit(f"[中止] 找不到 {TARGET}（請在 stockapp 目錄下執行）")

    with io.open(TARGET, "r", encoding="utf-8", newline="") as f:
        src = f.read()

    if "V26.76" in src:
        sys.exit("[中止] 檔案裡已經有 V26.76，看起來已套用過。")

    # CRLF 正規化後比對，套用完再還原
    had_crlf = "\r\n" in src
    work = src.replace("\r\n", "\n")

    # 先全部驗證，任何一段找不到 / 不唯一就中止（不寫出半套）
    problems = []
    for name, old, new in EDITS:
        o = old.replace("\r\n", "\n")
        c = work.count(o)
        if c != 1:
            problems.append(f"  - [{name}] 預期出現 1 次，實際 {c} 次")
    if problems:
        sys.exit("[中止] 以下錨點對不上，檔案版本可能不是 V26.75：\n" + "\n".join(problems))

    for name, old, new in EDITS:
        work = work.replace(old.replace("\r\n", "\n"), new.replace("\r\n", "\n"), 1)

    shutil.copyfile(TARGET, BACKUP)
    out = work.replace("\n", "\r\n") if had_crlf else work
    with io.open(TARGET, "w", encoding="utf-8", newline="") as f:
        f.write(out)

    print(f"[OK] 已套用 {len(EDITS)} 段修改，備份在 {BACKUP}")
    for name, _, _ in EDITS:
        print(f"     - {name}")


if __name__ == "__main__":
    main()

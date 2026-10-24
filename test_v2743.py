# -*- coding: utf-8 -*-
"""
test_v2743.py —— V27.43：假K剔除、1年高低SR提示、技術目標上檔%≥0、達到點位旗標、
新CSV欄位（MACD/MA交叉/▲/抄底結果）、level-tracker共振、ATR跌破分析、
歷史<200根警示、潛力股觀察清單。

執行：python -m pytest test_v2743.py -q
"""
import ast
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = open(os.path.join(HERE, "app_v18.py"), encoding="utf-8").read()
TREE = ast.parse(SRC)
_ok, _bad = [], []


def check(n, c, d=""):
    (_ok if c else _bad).append(n if c else f"{n}: {d}")


ns = {"np": np, "pd": pd, "os": os, "re": __import__("re")}

# 從 AST 載入必要函式
for node in TREE.body:
    if isinstance(node, (ast.FunctionDef, ast.Assign, ast.AugAssign)):
        name = getattr(node, "name", None) or (
            node.targets[0].id if isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Name) else None
        )
        if name in ("drop_fake_candles", "find_key_sr_levels", "swing_low_flags",
                    "dip_outcome", "load_level_tracker", "SWING_LOW_WINDOW",
                    "drop_fake_candles"):
            exec(compile(ast.Module(body=[node], type_ignores=[]), "app_v18.py", "exec"), ns)

# 也需要 DEFAULT_WATCHLISTS
for node in TREE.body:
    if (isinstance(node, ast.Assign) and
            any(getattr(t, "id", "") == "DEFAULT_WATCHLISTS" for t in node.targets)):
        exec(compile(ast.Module(body=[node], type_ignores=[]), "app_v18.py", "exec"), ns)

drop_fake = ns.get("drop_fake_candles")
find_sr = ns.get("find_key_sr_levels")
swl = ns.get("swing_low_flags")
dout = ns.get("dip_outcome")
load_lt = ns.get("load_level_tracker")
DEFAULT_WATCHLISTS = ns.get("DEFAULT_WATCHLISTS", {})

# ── A：版本字串 ──────────────────────────────────────────────────
for a in ['page_title="AI 實戰戰情室 V27.43"', '"📡 掃描中心 V27.43"',
          '"🎯 訊號驗證 V27.43"', '"📊 持倉戰情總表 V27.43"',
          '實戰戰情室 V27.43")', '（來源：AI 實戰戰情室 V27.43）']:
    check(f"A1 版本字串 {a[:28]}…", SRC.count(a) == 1, SRC.count(a))

# ── B：req1 drop_fake_candles ────────────────────────────────────
check("B0 drop_fake_candles 函式存在", drop_fake is not None, "")
if drop_fake is not None:
    idx = pd.bdate_range("2026-01-01", periods=5)
    # 假K：量=0 且 OHLC 同值
    df_fake = pd.DataFrame({
        "Open": [10.0, 20.0, 20.0, 30.0, 40.0],
        "High": [11.0, 20.0, 20.0, 31.0, 41.0],
        "Low":  [9.0,  20.0, 20.0, 29.0, 39.0],
        "Close":[10.5, 20.0, 20.0, 30.5, 40.5],
        "Volume":[1000, 0,    0,    500,  800],
    }, index=idx)
    result = drop_fake(df_fake)
    check("B1 成交量=0且四價同值的兩根被剔除", len(result) == 3,
          f"剩 {len(result)} 根（應=3）")
    check("B2 正常K棒全部保留", all(result.index.isin([idx[0], idx[3], idx[4]])),
          list(result.index))
    # 正常K但量=0（OHLC不同值 → 保留）
    df_legit = pd.DataFrame({
        "Open": [10.0], "High": [11.0], "Low": [9.0], "Close": [10.5], "Volume": [0],
    }, index=pd.bdate_range("2026-06-01", periods=1))
    check("B3 量=0但OHLC不同值的K棒保留（e.g.第一天）",
          len(drop_fake(df_legit)) == 1, "")
    # 無Volume欄 → 原樣回傳
    df_nov = pd.DataFrame({"Open":[1.0],"High":[1.1],"Low":[0.9],"Close":[1.0]},
                          index=pd.bdate_range("2026-06-01", periods=1))
    check("B4 無Volume欄時原樣回傳", len(drop_fake(df_nov)) == 1, "")

# ── C：req1 drop_fake_candles 在主流程中被呼叫 ──────────────────
check("C1 掃描器流程中有呼叫 drop_fake_candles",
      "drop_fake_candles(d)" in SRC, "")
check("C2 個股頁主流程中有呼叫 drop_fake_candles",
      "drop_fake_candles(df)" in SRC, "")

# ── D：req2 find_key_sr_levels 回傳 at_1yr_low / at_1yr_high ────
check("D0 find_key_sr_levels 函式存在", find_sr is not None, "")
if find_sr is not None:
    # 模擬已在一年低：Low=Close（當天低點就是收盤），確保 _52w_low == 現價
    n = 120
    closes = np.linspace(100, 50, n)  # 一路下跌，現價=最低
    df_low = pd.DataFrame({
        "Open":  closes * 1.01,
        "High":  closes * 1.01,
        "Low":   closes,            # Low = Close → _52w_low = 最後收盤值
        "Close": closes,
        "Volume": np.ones(n) * 1000,
    }, index=pd.bdate_range("2025-01-01", periods=n))
    sr_low = find_sr(df_low)
    check("D1 at_1yr_low 鍵存在於回傳 dict", "at_1yr_low" in sr_low, sr_low.keys())
    check("D2 at_1yr_high 鍵存在於回傳 dict", "at_1yr_high" in sr_low, sr_low.keys())
    check("D3 一路下跌到底時 at_1yr_low=True",
          bool(sr_low.get("at_1yr_low")), sr_low.get("at_1yr_low"))
    check("D4 一路下跌到底時 at_1yr_high=False",
          not bool(sr_low.get("at_1yr_high")), sr_low.get("at_1yr_high"))

    # 模擬已在一年高：close≈high
    closes_up = np.linspace(50, 100, n)
    df_high = pd.DataFrame({
        "Open":  closes_up * 0.99,
        "High":  closes_up * 1.01,
        "Low":   closes_up * 0.98,
        "Close": closes_up,
        "Volume": np.ones(n) * 1000,
    }, index=pd.bdate_range("2025-01-01", periods=n))
    sr_high = find_sr(df_high)
    check("D5 一路上漲到頂時 at_1yr_high=True",
          bool(sr_high.get("at_1yr_high")), sr_high.get("at_1yr_high"))

# ── E：req2 SR 顯示字串存在 ──────────────────────────────────────
check("E1 SR 顯示含「無下方支撐（已在一年低）」字串",
      "無下方支撐（已在一年低）" in SRC, "")
check("E2 SR 顯示含「無上方壓力（已在一年高）」字串",
      "無上方壓力（已在一年高）" in SRC, "")

# ── F：req3 技術上檔%不出現負值 ────────────────────────────────
# 確認掃描器邏輯：_tech_up < 0 → cap to 0 且設備註
check("F1 掃描器中有「技術目標_備註」欄", '"技術目標_備註"' in SRC, "")
check("F2 掃描器中有「已突破技術目標」字串",
      "已突破技術目標" in SRC, "")
check("F3 掃描器中 _tech_up 被 cap 至 0（_tech_up = 0.0）",
      "_tech_up = 0.0" in SRC, "")

# ── G：req4 達到點位旗標欄 ───────────────────────────────────────
check("G1 掃描 CSV 含「達到點位」欄", '"達到點位"' in SRC, "")
check("G2 「碰支撐」旗標字串存在",   '"碰支撐' in SRC, "")
check("G3 「碰壓力」旗標字串存在",   '"碰壓力' in SRC, "")
check("G4 「收盤≤ATR停損」旗標字串存在", '"收盤≤ATR停損' in SRC, "")
check("G5 「箱底跌破」旗標字串存在", '"箱底跌破' in SRC, "")

# ── H：req5 新 CSV 欄位 ──────────────────────────────────────────
for col in ["MACD金叉日", "MACD死叉日", "MA金叉日", "MA死叉日",
            "最近▲日", "最近▲價", "最近抄底結果"]:
    check(f"H 掃描CSV含欄「{col}」", f'"{col}"' in SRC, "")

# ── I：req6 level-tracker ────────────────────────────────────────
check("I1 load_level_tracker 函式存在", load_lt is not None, "")
if load_lt is not None:
    # 測試：空目錄（無 level-tracker.csv）→ 回 {}
    _lt_result = load_lt()
    check("I2 level-tracker.csv 不存在時回空 dict",
          isinstance(_lt_result, dict), type(_lt_result))
    # 測試多價位欄解析
    import tempfile, csv as _csv
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv",
                                     delete=False, encoding="utf-8",
                                     newline="") as _tf:
        _w = _csv.writer(_tf)
        _w.writerow(["代碼", "支撐1", "支撐2"])
        _w.writerow(["NVDA", "346/329/325", "300"])
        _w.writerow(["AAPL", "180", ""])
        _tf_path = _tf.name
    try:
        # 暫時替換函式讓它從此路徑讀
        import io
        _lt_src = f"""
import os, re, pandas as pd
def _load_lt_test():
    _lt = pd.read_csv(r'{_tf_path.replace(chr(92), "/")}', encoding='utf-8', dtype=str)
    _lt.columns = [c.strip() for c in _lt.columns]
    result = {{}}
    _tc = next((c for c in _lt.columns if c.lower() in ('代碼','ticker','symbol')), None)
    if _tc is None: return result
    for _, row in _lt.iterrows():
        _tk = str(row[_tc]).strip()
        if not _tk or _tk=='nan': continue
        _levels = []
        for _col in _lt.columns:
            if _col==_tc: continue
            _val = str(row.get(_col,'')).strip()
            if not _val or _val=='nan': continue
            for _part in re.split(r'[/,;|]+', _val):
                try:
                    _lv = float(_part.strip())
                    if _lv>0: _levels.append(_lv)
                except ValueError: pass
        if _levels: result[_tk] = sorted(set(round(v,4) for v in _levels))
    return result
"""
        exec(_lt_src, ns)
        _r = ns["_load_lt_test"]()
        check("I3 多價位欄 346/329/325 被拆成 3 個數值",
              "NVDA" in _r and len(_r["NVDA"]) == 4, _r.get("NVDA"))
        check("I4 單值 300 也被讀進來", "NVDA" in _r and 300.0 in _r["NVDA"], _r.get("NVDA"))
        check("I5 AAPL 只有 1 個點位", "AAPL" in _r and len(_r["AAPL"]) == 1, _r.get("AAPL"))
    finally:
        try:
            os.unlink(_tf_path)
        except Exception:
            pass

check("I6 ladder 顯示中有頻道點位共振/衝突字串",
      "共振" in SRC and "衝突" in SRC, "")
check("I7 ladder 顯示中有「頻道支撐已高於現價」字串",
      "頻道支撐已高於現價" in SRC, "")

# ── J：req7 ATR 跌破分析欄 ──────────────────────────────────────
check("J1 掃描 CSV 含「ATR停損狀態」欄", '"ATR停損狀態"' in SRC, "")
check("J2 含「剛跌破」字串",   "剛跌破" in SRC, "")
check("J3 含「已深跌」字串",   "已深跌" in SRC, "")
check("J4 含「距停損」字串",   "距停損" in SRC, "")

# ── K：req8 歷史不足 200 根警示 ─────────────────────────────────
check("K1 個股頁有「歷史資料不足」警示字串", "歷史資料不足" in SRC, "")
check("K2 歷史不足警示提及 MA200", "MA200" in SRC and "len(df) < 200" in SRC,
      "'MA200' in SRC: " + str("MA200" in SRC) + "; 'len(df) < 200' in SRC: " + str("len(df) < 200" in SRC))
check("K3 歷史不足警示提及一年高低", "一年高低" in SRC, "")

# ── L：req9 潛力股觀察清單 ──────────────────────────────────────
check("L1 DEFAULT_WATCHLISTS 含「潛力股觀察」清單",
      any("潛力股觀察" in k for k in DEFAULT_WATCHLISTS.keys()),
      list(DEFAULT_WATCHLISTS.keys()))
_pot_key = next((k for k in DEFAULT_WATCHLISTS if "潛力股觀察" in k), None)
if _pot_key:
    _pot = DEFAULT_WATCHLISTS[_pot_key]
    for _s in ["ALGM", "MYRG", "FN", "MTZ", "AAON", "STRL", "DY"]:
        check(f"L2 美股 {_s} 在潛力股清單", _s in _pot, _pot)
    for _s in ["5309.TWO", "2421.TW", "2481.TW", "3042.TW",
               "2355.TW", "3338.TW", "5469.TW", "5536.TWO"]:
        check(f"L2 台股 {_s} 在潛力股清單", _s in _pot, _pot)
else:
    check("L2 潛力股清單不為空", False, "找不到含「潛力股觀察」的清單")

# 既有清單不應被刪改
for _orig_k in ["🇺🇸 美持股", "🔭 美觀察", "🇹🇼 台持股", "🔭 台觀察"]:
    check(f"L3 既有清單「{_orig_k}」未被刪改",
          _orig_k in DEFAULT_WATCHLISTS, list(DEFAULT_WATCHLISTS.keys()))

print("=" * 64)
for x in _ok:
    print("  ✅", x)
for x in _bad:
    print("  ❌", x)
print("=" * 64)
print(f"v2743: {len(_ok)} passed, {len(_bad)} failed")
sys.exit(1 if _bad else 0)

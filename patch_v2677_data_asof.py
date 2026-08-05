# -*- coding: utf-8 -*-
"""V26.76 → V26.77 補丁：資料日（asof）管線 + 交易日曆落後檢查。

用法：把本檔放到 stockapp 目錄，執行  python patch_v2677_data_asof.py
行為：先跑全部前置檢查（任何一條不過就整個不動），才備份、套用、驗證。

⚠️ 部署前置：requirements.txt 需新增一行
    pandas-market-calendars
（缺了不會炸 app —— 總表會顯示「交易日曆載入失敗」的黃色警告並指出這件事。）
"""
import ast
import shutil
import sys

APP = "app_v18.py"
BAK = "app_v18.py.bak_v2676"

raw = open(APP, encoding="utf-8", newline="").read()
src = raw.replace("\r\n", "\n")

# ── 補丁定義：(唯一錨點, 替換後) ──────────────────────────
NEW_FUNCS = '''# ──────────────────────────────────────────────────────
# [V26.77] 資料日基準 — 「此刻資料應已就緒的最近交易日」
#   用 pandas-market-calendars 的 NYSE / XTAI 日曆（含國定假日，套件維護，非手動表）。
#   就緒界線沿用 get_cache_anchor 既有錨點：台股 14:00、美股隔日 08:00。
#   已知不涵蓋：台股臨時颱風假 → 該日會誤標紅（設計接受：誤報成本=多看一眼）。
# ──────────────────────────────────────────────────────
def _pick_expected_trade_date(valid_dates, now_tw, market):
    """從交易日清單挑「now_tw 時刻資料應已就緒」的最近交易日，回傳 "YYYY-MM-DD" 或 None。

    valid_dates: list[str "YYYY-MM-DD"]（升冪）
    now_tw:      naive datetime（台灣時間）
    market:      "US" | "TW"
    規則：TW 交易日 D 的日 K 在 D 當天 14:00 後應就緒；
          US 交易日 D 收盤 ≈ 台灣 D+1 凌晨 → D+1 的 08:00 後應就緒。
    """
    from datetime import datetime as _dt
    today = now_tw.strftime("%Y-%m-%d")
    picked = None
    for ds in valid_dates:  # 升冪 → 最後一個就緒者即答案
        if market == "TW":
            ready = ds < today or (ds == today and now_tw.hour >= 14)
        else:
            d_next = (_dt.strptime(ds, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
            ready = d_next < today or (d_next == today and now_tw.hour >= 8)
        if ready:
            picked = ds
    return picked


def _find_stale_asof(asof_map, expected_map, tickers):
    """[V26.77] 回傳 {ticker: (資料日, 預期日)} — 資料日早於該市場預期交易日者。

    刻意不做「整批一致 → 視為休市不報」的推論：8/4 事故正是整批一致慢一天，
    該推論會把真 bug 消音（漏報成本=反向操作 >> 誤報成本=多看一眼）。
    市場歸屬沿用既有慣例：".TW" in ticker → 台股（含 .TWO），其餘美股。
    """
    stale = {}
    for tk in tickers:
        a = asof_map.get(tk)
        exp = expected_map.get("TW" if ".TW" in tk else "US")
        if a and exp and a < exp:  # "YYYY-MM-DD" 字典序 = 時間序
            stale[tk] = (a, exp)
    return stale


@st.cache_data(ttl=21600, show_spinner=False)
def get_expected_trade_dates(anchor):
    """[V26.77] 回傳 {"US": "YYYY-MM-DD", "TW": "YYYY-MM-DD"}。
    anchor 進 cache key — 錨點換了才重算。失敗回 {"_error": 原因}（Rule 12：
    顯示層負責喊出來，這裡不靜默也不讓整頁掛掉）。"""
    try:
        from datetime import datetime as _dt, timezone as _tz
        import pandas_market_calendars as mcal
        now_tw = _dt.now(_tz.utc).replace(tzinfo=None) + timedelta(hours=8)
        out = {}
        for mk, cal_name in (("US", "NYSE"), ("TW", "XTAI")):
            cal = mcal.get_calendar(cal_name)
            days = cal.valid_days(start_date=(now_tw - timedelta(days=14)).date(),
                                  end_date=now_tw.date())
            out[mk] = _pick_expected_trade_date(
                [d.strftime("%Y-%m-%d") for d in days], now_tw, mk)
        return out
    except Exception as _e:
        return {"_error": f"{type(_e).__name__}: {_e}"}


FINMIND_TOKEN = _get_secret("FINMIND_TOKEN", "")'''

BANNER = '''    else:
        # [V26.77] 資料日檢查 — 資料自帶日期 vs 交易日曆基準（8/4 整批慢一天事故的防線）
        _asof_map = st.session_state.get("_wl_prices_asof", {})
        if not _asof_map:
            st.warning("⚠️ 本批資料沒有「資料日」（V26.77 前的舊快取）——無法驗證新舊。"
                       "請按左側「🔄 清除今日快取」後重掃。")
        else:
            _exp_dates = get_expected_trade_dates(get_cache_anchor())
            if "_error" in _exp_dates:
                st.warning(f"⚠️ 交易日曆載入失敗，本次無法驗證資料日：{_exp_dates['_error']}"
                           "（若是 ModuleNotFoundError，requirements.txt 需加 pandas-market-calendars）")
            else:
                _stale = _find_stale_asof(_asof_map, _exp_dates, _all_tk)
                if _stale:
                    _ex_t, (_ex_a, _ex_e) = sorted(_stale.items())[0]
                    st.error(f"🔴 資料落後：{len(_stale)} 檔資料日早於最近交易日"
                             f"（例：{_ex_t} 資料日 {_ex_a}，預期 {_ex_e}）。"
                             f"落後時「現價」其實是前收、漲跌方向可能全反——先清快取重掃；"
                             f"重掃仍落後代表上游（Yahoo）延遲，勿依本頁數據下單。"
                             f"※ 台股臨時休市（颱風假）會誤報，屬預期行為。")
                else:
                    st.caption(f"✅ 資料日檢查通過（交易日基準：美股 {_exp_dates.get('US','—')}"
                               f"｜台股 {_exp_dates.get('TW','—')}）")

        # [V26.71] 價格資料診斷 — 抓「現價寫錯」的現行犯
        with st.expander("🔧 價格資料診斷"):
            st.caption(f"資料來源：{st.session_state.get('_wl_src', '未知（可能是舊 session）')}")
            _dbg = [{"代碼": _t,
                     "現價": _prices.get(_t),
                     "資料日": _asof_map.get(_t),  # [V26.77]
                     "昨日": st.session_state.get("_wl_prices_yd", {}).get(_t),
                     "5日前": st.session_state.get("_wl_prices_wk", {}).get(_t)}
                    for _t in _all_tk]'''

PATCHES = [
    # 1) 新函式群（插在 FINMIND_TOKEN 前）
    ('FINMIND_TOKEN = _get_secret("FINMIND_TOKEN", "")', NEW_FUNCS),
    # 2) 掃描時記錄 asof（必須字串）
    ('                icons.setdefault("_prices", {})[tk] = round(float(d["Close"].iloc[-1]), 2)\n'
     '                icons.setdefault("_prices_wk", {})[tk] = round(float(d["Close"].iloc[-6]), 2)  # [V26.63] 5日前收盤',
     '                icons.setdefault("_prices", {})[tk] = round(float(d["Close"].iloc[-1]), 2)\n'
     '                # [V26.77] 資料日：最後一根 K 棒自帶的日期。必須 strftime 成字串 —\n'
     '                #          date 物件會讓下游 json.dump 丟 TypeError 且被 except pass 吃掉（快取永遠寫不進）。\n'
     '                icons.setdefault("_prices_asof", {})[tk] = d.index[-1].strftime("%Y-%m-%d")\n'
     '                icons.setdefault("_prices_wk", {})[tk] = round(float(d["Close"].iloc[-6]), 2)  # [V26.63] 5日前收盤'),
    # 3) 快取檔讀入
    ('                    st.session_state["_wl_prices_yd"] = _cached.get("prices_yd", {})  # [V26.65]',
     '                    st.session_state["_wl_prices_yd"] = _cached.get("prices_yd", {})  # [V26.65]\n'
     '                    st.session_state["_wl_prices_asof"] = _cached.get("prices_asof", {})  # [V26.77]'),
    # 4) 清除快取清單
    ('            for _kc in ("_wl_icons", "_wl_prices", "_wl_prices_wk", "_wl_prices_yd",\n'
     '                        "_wl_prev_high", "_wl_iron", "_wl_scores", "_wl_src"):',
     '            for _kc in ("_wl_icons", "_wl_prices", "_wl_prices_wk", "_wl_prices_yd",\n'
     '                        "_wl_prices_asof",  # [V26.77]\n'
     '                        "_wl_prev_high", "_wl_iron", "_wl_scores", "_wl_src"):'),
    # 5) 掃描結果 pop
    ('        _scan_prices = _icons.pop("_prices", {}) if isinstance(_icons, dict) else {}\n'
     '        _scan_prices_wk = _icons.pop("_prices_wk", {}) if isinstance(_icons, dict) else {}',
     '        _scan_prices = _icons.pop("_prices", {}) if isinstance(_icons, dict) else {}\n'
     '        _scan_prices_asof = _icons.pop("_prices_asof", {}) if isinstance(_icons, dict) else {}  # [V26.77]\n'
     '        _scan_prices_wk = _icons.pop("_prices_wk", {}) if isinstance(_icons, dict) else {}'),
    # 6) session_state 寫入
    ('        st.session_state["_wl_prices_yd"] = _scan_prices_yd  # [V26.65] 昨日價',
     '        st.session_state["_wl_prices_yd"] = _scan_prices_yd  # [V26.65] 昨日價\n'
     '        st.session_state["_wl_prices_asof"] = _scan_prices_asof  # [V26.77] 資料日'),
    # 7) 快取檔寫出
    ('                json.dump({"icons": _icons, "prices": _scan_prices, "prices_wk": _scan_prices_wk,\n'
     '                           "prices_yd": _scan_prices_yd, "prev_high": _scan_prev_high,\n'
     '                           "iron": _scan_iron, "scores": _scan_scores}, _fh)',
     '                json.dump({"icons": _icons, "prices": _scan_prices, "prices_wk": _scan_prices_wk,\n'
     '                           "prices_yd": _scan_prices_yd, "prices_asof": _scan_prices_asof,  # [V26.77]\n'
     '                           "prev_high": _scan_prev_high,\n'
     '                           "iron": _scan_iron, "scores": _scan_scores}, _fh)'),
    # 8) 總表橫幅 + 診斷欄
    ('    else:\n'
     '        # [V26.71] 價格資料診斷 — 抓「現價寫錯」的現行犯\n'
     '        with st.expander("🔧 價格資料診斷"):\n'
     '            st.caption(f"資料來源：{st.session_state.get(\'_wl_src\', \'未知（可能是舊 session）\')}")\n'
     '            _dbg = [{"代碼": _t,\n'
     '                     "現價": _prices.get(_t),\n'
     '                     "昨日": st.session_state.get("_wl_prices_yd", {}).get(_t),\n'
     '                     "5日前": st.session_state.get("_wl_prices_wk", {}).get(_t)}\n'
     '                    for _t in _all_tk]',
     BANNER),
    # 9) 版本 bump（只動兩個標題字面，不碰 [V26.76] 歷史註解）
    ('page_title="AI 實戰戰情室 V26.76"', 'page_title="AI 實戰戰情室 V26.77"'),
    ('st.title("📡 掃描中心 V26.76" if cur_t == "__SCANNER__"\n'
     '         else "🎯 訊號驗證 V26.76" if cur_t == "__VERIFY__"\n'
     '         else "📊 持倉戰情總表 V26.76" if cur_t == "__DASHBOARD__"\n'
     '         else f"📈 {disp_main_title} 實戰戰情室 V26.76")',
     'st.title("📡 掃描中心 V26.77" if cur_t == "__SCANNER__"\n'
     '         else "🎯 訊號驗證 V26.77" if cur_t == "__VERIFY__"\n'
     '         else "📊 持倉戰情總表 V26.77" if cur_t == "__DASHBOARD__"\n'
     '         else f"📈 {disp_main_title} 實戰戰情室 V26.77")'),
]

# ── 前置檢查：全部錨點必須恰好出現一次，否則整包不動 ──
errs = []
if "V26.76" not in src:
    errs.append("找不到 V26.76 — 檔案版本不對（本補丁只適用 V26.76）")
if "_prices_asof" in src:
    errs.append("_prices_asof 已存在 — 補丁疑似已套用過")
for i, (old, _n) in enumerate(PATCHES, 1):
    c = src.count(old)
    if c != 1:
        errs.append(f"補丁 #{i} 錨點出現 {c} 次（需恰為 1）：{old[:60]!r}...")
if errs:
    print("❌ 前置檢查失敗，檔案未動：")
    for e in errs:
        print("  -", e)
    sys.exit(1)

# ── 備份 → 套用 → 寫回（CRLF）──
shutil.copy2(APP, BAK)
for old, new in PATCHES:
    src = src.replace(old, new, 1)
out = src.replace("\r\n", "\n").replace("\n", "\r\n")
with open(APP, "w", encoding="utf-8", newline="") as f:
    f.write(out)

# ── 驗證：AST 編譯 + 版本 + 觸點盤點 ──
chk = open(APP, encoding="utf-8", newline="").read()
ast.parse(chk)
assert chk.count("V26.77") >= 5 and 'page_title="AI 實戰戰情室 V26.77"' in chk
for pat in ('icons.setdefault("_prices_asof"', '_icons.pop("_prices_asof"',
            '"prices_asof": _scan_prices_asof', '_cached.get("prices_asof"',
            '"_wl_prices_asof",', '_find_stale_asof(_asof_map'):
    assert pat in chk, f"觸點缺失: {pat}"
print(f"✅ 補丁完成：V26.76 → V26.77（備份 {BAK}）")
print("   接著跑： python test_v2677.py")
print("   ⚠️ 別忘了 requirements.txt 加一行：pandas-market-calendars")

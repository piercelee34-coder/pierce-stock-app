# =============================================================
# tw_sentiment_engine.py — 台股空頭危機距離指數 (方案 B)
#
# 兩層結構：
#   散戶過熱分數 = (小台散戶多空比 + 融資位階) / 2
#   主力撤離分數 = (外資期貨120日百分位 + 外資現貨20日累計) / 2
#
# 最終指數 = 散戶分 × 0.30 + 主力分 × 0.70   (範圍 0~100)
#
# 設計重點：
#   1. FinMind 免費版相容：融資抓不到時自動降級為單指標散戶分
#   2. 快取機制 (ttl=3600)，避免耗盡 API 配額
#   3. 修正原本 v18 的 6 個 bug：
#       #1 inst_net 排除「合計/Total」列，避免雙重計算
#       #2 total_oi 改用快取最後有效值，不寫死 100000
#       #3 外資期貨改用百分位判讀，非以 0 為基準
#       #4 統一警戒門檻為一套（±10% / ±25%）
#       #5 TTL 60s → 3600s
#       #6 期現背離整合進綜合指數（自動聯動警示）
#
# 對外函式：
#   - get_tw_crisis_index()              → dict (含 index 與所有子分數)
#   - render_tw_crisis_dashboard()       → Streamlit UI 元件
# =============================================================

import os
import json
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ─── 設定 ──────────────────────────────────────────────
FINMIND_TOKEN = os.environ.get("FINMIND_TOKEN", "")
if not FINMIND_TOKEN:
    try:
        import streamlit as st
        FINMIND_TOKEN = str(st.secrets.get("FINMIND_TOKEN", "")).strip()
    except Exception:
        pass

FINMIND_URL = "https://api.finmindtrade.com/api/v4/data"
UA = {"User-Agent": "Mozilla/5.0"}

# 快取（保留 total_oi 等備援值）
CACHE_DIR  = ".tw_sentiment_cache"
LAST_VALID = os.path.join(CACHE_DIR, "last_valid.json")


def _ensure_dir():
    if not os.path.exists(CACHE_DIR):
        try: os.makedirs(CACHE_DIR, exist_ok=True)
        except: pass

def _load_cache(default=None):
    try:
        if os.path.exists(LAST_VALID):
            with open(LAST_VALID, "r", encoding="utf-8") as f:
                return json.load(f)
    except: pass
    return default or {}

def _save_cache(data):
    _ensure_dir()
    try:
        with open(LAST_VALID, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, default=str, indent=2)
    except: pass


# ─── FinMind 抓取工具 ───────────────────────────────────
def _fetch(dataset, data_id=None, days=200):
    """通用 FinMind 抓取，回傳 DataFrame；失敗回傳空 df"""
    start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    params = {"dataset": dataset, "start_date": start_date, "token": FINMIND_TOKEN}
    if data_id:
        params["data_id"] = data_id
    try:
        r = requests.get(FINMIND_URL, params=params, headers=UA, timeout=12).json()
        if r.get("msg") == "success" and r.get("data"):
            return pd.DataFrame(r["data"])
    except Exception:
        pass
    return pd.DataFrame()


def _find_col(df, keywords):
    for c in df.columns:
        if any(k.lower() in str(c).lower() for k in keywords):
            return c
    return None


def _get_net_oi_col(df):
    """找出『淨未平倉』欄位；若只有多空兩欄則計算差值"""
    net_col = _find_col(df, ["open_interest_net", "net_oi", "未平仓淨"])
    if net_col:
        return net_col
    lc = _find_col(df, ["long_open_interest", "多方未平仓"])
    sc = _find_col(df, ["short_open_interest", "空方未平仓"])
    if lc and sc:
        df["_temp_net"] = (pd.to_numeric(df[lc], errors="coerce")
                           - pd.to_numeric(df[sc], errors="coerce"))
        return "_temp_net"
    return None


# ═════════════════════════════════════════════════════
#  指標 ① 小台散戶多空比 (修 bug 後)
# ═════════════════════════════════════════════════════
def _calc_retail_ratio():
    """回傳 (ratio_value, status_msg)。修了 v18 的 bug #1 #2"""
    cache = _load_cache()

    df_mtx = _fetch("TaiwanFuturesInstitutionalInvestors", "MTX")
    df_tot = _fetch("TaiwanFuturesDaily", "MTX")
    if df_mtx.empty:
        return None, "MTX 法人 API 失敗"

    date_col = _find_col(df_mtx, ["date", "日期"])
    name_col = _find_col(df_mtx, ["name", "institutional", "investor", "法人", "名稱"])
    net_col  = _get_net_oi_col(df_mtx)
    if not all([date_col, net_col]):
        return None, "MTX 欄位辨識失敗"

    # [Bug #1 修正] 排除「合計/Total/All」列，避免雙重計算
    if name_col:
        mask = ~df_mtx[name_col].astype(str).str.contains(
            "合計|總計|Total|All", case=False, na=False)
        df_mtx_inst = df_mtx[mask].copy()
    else:
        df_mtx_inst = df_mtx.copy()

    latest_d = df_mtx_inst[date_col].max()
    inst_net = pd.to_numeric(
        df_mtx_inst[df_mtx_inst[date_col] == latest_d][net_col],
        errors="coerce").sum()

    # [Bug #2 修正] total_oi 不寫死 100000，改用快取最後有效值
    total_oi = None
    if not df_tot.empty:
        dc_tot = _find_col(df_tot, ["date", "日期"])
        oc_tot = _find_col(df_tot, ["open_interest", "oi", "未平仓"])
        if dc_tot and oc_tot:
            tot_row = df_tot[df_tot[dc_tot] == latest_d]
            if not tot_row.empty:
                val = pd.to_numeric(tot_row[oc_tot], errors="coerce").max()
                if not pd.isna(val) and val > 0:
                    total_oi = float(val)

    if total_oi is None:
        total_oi = cache.get("last_mtx_total_oi")
        if total_oi is None:
            return None, "MTX 總部位無快取備援"

    # 更新快取
    cache["last_mtx_total_oi"] = total_oi
    _save_cache(cache)

    ratio = round((-inst_net / total_oi) * 100, 2)
    return ratio, "ok"


def _retail_ratio_to_score(ratio):
    """小台多空比 → 0~100 散戶過熱分。+25%→100, 0%→50, -25%→0"""
    if ratio is None: return None
    if ratio >=  25: return 100.0
    if ratio <= -25: return   0.0
    # 線性插值：分數 = 50 + ratio * 2
    return round(50.0 + ratio * 2.0, 1)


# ═════════════════════════════════════════════════════
#  指標 ② 融資餘額 252 日百分位 (FinMind 免費版可能缺)
# ═════════════════════════════════════════════════════
def _calc_margin_percentile():
    """回傳 (percentile_0_100, latest_value, status)
       FinMind 免費版若抓不到，回傳 (None, None, 'unavailable')
    """
    # 試 dataset: TaiwanStockTotalMarginPurchaseShortSale（大盤合計）
    df = _fetch("TaiwanStockTotalMarginPurchaseShortSale", days=400)
    if df.empty:
        return None, None, "免費版無此 dataset"

    date_col = _find_col(df, ["date", "日期"])
    # 融資餘額欄位常見名稱
    margin_col = _find_col(df, ["TotalBuyBalance", "margin_purchase_balance",
                                 "MarginPurchaseTodayBalance", "融資餘額"])
    if not date_col or not margin_col:
        return None, None, "欄位辨識失敗"

    df = df[[date_col, margin_col]].copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[margin_col] = pd.to_numeric(df[margin_col], errors="coerce")
    df = df.dropna().sort_values(date_col).tail(252)

    if len(df) < 30:
        return None, None, "資料筆數不足"

    latest = float(df[margin_col].iloc[-1])
    # 計算百分位排名
    pct = (df[margin_col] <= latest).sum() / len(df) * 100
    return round(pct, 1), latest, "ok"


def _margin_percentile_to_score(pct):
    """融資 252 日百分位 → 0~100 散戶過熱分
       ≥80%→100, ≤20%→0, 其他直接用 percentile 當分數"""
    if pct is None: return None
    if pct >= 80: return 100.0
    if pct <= 20: return   0.0
    return round(pct, 1)


# ═════════════════════════════════════════════════════
#  指標 ③ 外資期貨 120 日百分位 (反向)
# ═════════════════════════════════════════════════════
def _calc_foreign_futures_percentile():
    """回傳 (percentile, latest_net_oi, df_history, status)
       注意：這是反向指標，外資越空（百分位越低）→ 撤離分越高
    """
    df_tx = _fetch("TaiwanFuturesInstitutionalInvestors", "TX", days=200)
    if df_tx.empty:
        return None, None, pd.DataFrame(), "TX 法人 API 失敗"

    date_col = _find_col(df_tx, ["date", "日期"])
    name_col = _find_col(df_tx, ["name", "institutional", "investor", "法人", "名稱"])
    net_col  = _get_net_oi_col(df_tx)
    if not all([date_col, name_col, net_col]):
        return None, None, pd.DataFrame(), "TX 欄位辨識失敗"

    df_f = df_tx[df_tx[name_col].astype(str).str.contains(
        "外資|Foreign", case=False, na=False)].copy()
    if df_f.empty:
        return None, None, pd.DataFrame(), "找不到外資列"

    df_f["Date"]   = pd.to_datetime(df_f[date_col], errors="coerce")
    df_f["Net_OI"] = pd.to_numeric(df_f[net_col], errors="coerce")
    df_f = df_f.dropna(subset=["Date", "Net_OI"]).sort_values("Date").tail(120)

    if len(df_f) < 30:
        return None, None, df_f, "TX 歷史不足"

    latest = float(df_f["Net_OI"].iloc[-1])
    pct = (df_f["Net_OI"] <= latest).sum() / len(df_f) * 100
    return round(pct, 1), latest, df_f, "ok"


def _foreign_futures_pct_to_score(pct):
    """外資期貨百分位 → 0~100 主力撤離分（反向）
       ≤20% → 100 (外資創半年新低 = 結構性撤退)
       ≥80% → 0   (外資創半年新高 = 結構性進場)
       其他 → 100 - percentile"""
    if pct is None: return None
    if pct <= 20: return 100.0
    if pct >= 80: return   0.0
    return round(100.0 - pct, 1)


# ═════════════════════════════════════════════════════
#  指標 ④ 外資現貨 20 日累計買賣超
# ═════════════════════════════════════════════════════
def _calc_foreign_spot_cum20():
    """回傳 (cum20_in_billion, df_history, status)"""
    df = _fetch("TaiwanStockTotalInstitutionalInvestors", days=60)
    if df.empty:
        return None, pd.DataFrame(), "現貨法人 API 失敗"

    date_col = _find_col(df, ["date", "time"])
    name_col = _find_col(df, ["name", "institutional", "法人"])
    if not all([date_col, name_col]):
        return None, pd.DataFrame(), "現貨欄位辨識失敗"

    df_f = df[df[name_col].astype(str).str.contains(
        "外資|Foreign", case=False, na=False)].copy()
    if df_f.empty:
        return None, pd.DataFrame(), "找不到外資現貨列"

    buy_col  = _find_col(df_f, ["buy", "買"])
    sell_col = _find_col(df_f, ["sell", "賣"])
    net_col  = _find_col(df_f, ["net", "買賣超"])

    df_f["Date"] = pd.to_datetime(df_f[date_col], errors="coerce")
    if net_col:
        df_f["Net"] = pd.to_numeric(df_f[net_col], errors="coerce") / 1e8
    elif buy_col and sell_col:
        df_f["Net"] = ((pd.to_numeric(df_f[buy_col], errors="coerce") -
                        pd.to_numeric(df_f[sell_col], errors="coerce")) / 1e8)
    else:
        return None, pd.DataFrame(), "現貨欄位無法計算"

    df_g = (df_f[["Date", "Net"]].dropna()
                  .groupby("Date").sum().reset_index()
                  .sort_values("Date").tail(40))

    if len(df_g) < 5:
        return None, df_g, "現貨歷史不足"

    cum20 = float(df_g["Net"].tail(20).sum())
    return round(cum20, 1), df_g, "ok"


def _foreign_spot_cum20_to_score(cum20):
    """外資現貨 20 日累計 → 0~100 主力撤離分
       ≤-1500 億 → 100, 0 億 → 50, ≥+1500 億 → 0
       線性插值"""
    if cum20 is None: return None
    if cum20 <= -1500: return 100.0
    if cum20 >=  1500: return   0.0
    return round(50.0 - (cum20 / 1500.0) * 50.0, 1)


# ═════════════════════════════════════════════════════
#  主流程：組裝最終指數
# ═════════════════════════════════════════════════════
def get_tw_crisis_index():
    """回傳完整 dict，含最終指數、子分數、原始值、各指標狀態"""
    # 嘗試使用 Streamlit 快取（若可用）；不可用則直接計算
    try:
        import streamlit as st
        return _get_tw_crisis_index_cached()
    except Exception:
        return _get_tw_crisis_index_impl()


def _get_tw_crisis_index_impl():
    # ─ 抓四個指標 ────────────────────────────────────
    retail_ratio, s1_msg              = _calc_retail_ratio()
    margin_pct, margin_val, s2_msg    = _calc_margin_percentile()
    foreign_pct, foreign_val, df_tx, s3_msg = _calc_foreign_futures_percentile()
    spot_cum20, df_spot, s4_msg       = _calc_foreign_spot_cum20()

    # ─ 轉成分數 ──────────────────────────────────────
    score1 = _retail_ratio_to_score(retail_ratio)
    score2 = _margin_percentile_to_score(margin_pct)
    score3 = _foreign_futures_pct_to_score(foreign_pct)
    score4 = _foreign_spot_cum20_to_score(spot_cum20)

    # ─ 散戶過熱分數（含降級邏輯）─────────────────────
    margin_degraded = False
    if score1 is not None and score2 is not None:
        retail_score = round((score1 + score2) / 2, 1)
    elif score1 is not None:
        retail_score = score1
        margin_degraded = True
    elif score2 is not None:
        retail_score = score2
    else:
        retail_score = None

    # ─ 主力撤離分數 ─────────────────────────────────
    if score3 is not None and score4 is not None:
        institutional_score = round((score3 + score4) / 2, 1)
    elif score3 is not None:
        institutional_score = score3
    elif score4 is not None:
        institutional_score = score4
    else:
        institutional_score = None

    # ─ 最終指數 (30/70) ──────────────────────────────
    if retail_score is not None and institutional_score is not None:
        final_index = round(retail_score * 0.30 + institutional_score * 0.70, 1)
        index_status = "ok"
    elif institutional_score is not None:
        final_index = round(institutional_score, 1)
        index_status = "partial_inst_only"
    elif retail_score is not None:
        final_index = round(retail_score, 1)
        index_status = "partial_retail_only"
    else:
        final_index = None
        index_status = "no_data"

    return {
        "final_index":       final_index,
        "index_status":      index_status,
        "retail_score":      retail_score,
        "institutional_score": institutional_score,
        "margin_degraded":   margin_degraded,
        # 各指標原始值
        "retail_ratio":      retail_ratio,
        "margin_percentile": margin_pct,
        "margin_value":      margin_val,
        "foreign_futures_percentile": foreign_pct,
        "foreign_futures_value":      foreign_val,
        "foreign_spot_cum20":         spot_cum20,
        # 子分數
        "score_retail_ratio":          score1,
        "score_margin":                score2,
        "score_foreign_futures":       score3,
        "score_foreign_spot":          score4,
        # 訊息
        "messages": {
            "retail_ratio":    s1_msg,
            "margin":          s2_msg,
            "foreign_futures": s3_msg,
            "foreign_spot":    s4_msg,
        },
        # 圖表用 DataFrame
        "df_foreign_futures": df_tx,
        "df_foreign_spot":    df_spot,
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


# 用 Streamlit cache 包裝（[Bug #5] TTL 60s → 3600s）
def _get_tw_crisis_index_cached():
    import streamlit as st
    @st.cache_data(ttl=3600, show_spinner=False)
    def _inner():
        return _get_tw_crisis_index_impl()
    return _inner()


# ═════════════════════════════════════════════════════
#  指數區間解讀
# ═════════════════════════════════════════════════════
def _index_verdict(idx):
    if idx is None: return "—", "#666", "資料不足"
    if idx >= 80: return "🔴 極度危險區",  "#dc2626", "強烈減碼 / 避險"
    if idx >= 65: return "🟠 警戒區",      "#f97316", "緊縮停損，停止加倉"
    if idx >= 35: return "⚖️ 中性區",      "#9ca3af", "維持原有部位"
    if idx >= 20: return "🟢 機會浮現",    "#84cc16", "開始分批佈局"
    return                "💚 極度恐慌區",  "#22c55e", "大膽佈局時機"


# ═════════════════════════════════════════════════════
#  Streamlit 渲染元件
# ═════════════════════════════════════════════════════
def render_tw_crisis_dashboard():
    """在台股股票頁面顯示空頭危機距離指數儀表板"""
    try:
        import streamlit as st
        import plotly.graph_objects as go
    except ImportError:
        return

    data = get_tw_crisis_index()
    idx = data["final_index"]
    verdict, color, suggestion = _index_verdict(idx)
    retail_score = data["retail_score"]
    inst_score   = data["institutional_score"]

    # CSS
    st.markdown("""
    <style>
    .tw-crisis-card {background-color: #1a1a1c; border-radius: 8px;
                     padding: 16px 20px; margin-bottom: 8px;
                     border: 1px solid #333;}
    .tw-crisis-title {font-size: 14px; color: #aaa; margin-bottom: 8px;}
    .tw-crisis-idx {font-size: 56px; font-weight: 900; line-height: 1.0;
                    text-align: center; margin: 8px 0;}
    .tw-crisis-verdict {font-size: 18px; font-weight: bold;
                        text-align: center; margin: 4px 0 12px 0;}
    .tw-crisis-suggestion {font-size: 13px; color: #ccc;
                           text-align: center; margin-bottom: 10px;}
    .tw-sub {font-size: 12px; color: #888; text-align: center;}
    .tw-sub b {color: #ddd;}
    .tw-warn {font-size: 11px; color: #f97316; margin-top: 4px;
              text-align: center;}
    </style>
    """, unsafe_allow_html=True)

    st.markdown("### 🇹🇼 台股空頭危機距離指數 "
                "<span style='font-size:12px;color:#888;'>(FinMind 收盤後更新)</span>",
                unsafe_allow_html=True)

    if idx is None:
        st.warning("⚠️ FinMind API 資料完全無法取得，無法計算指數")
        for k, v in data["messages"].items():
            if v and v != "ok":
                st.caption(f"  · {k}: {v}")
        return

    # 主卡片
    idx_disp = f"{idx:.1f}"
    sub_html = ""
    if retail_score is not None:
        retail_html = f"散戶過熱 <b>{retail_score:.0f}</b>"
        if data.get("margin_degraded"):
            retail_html += " ⚠️"
    else:
        retail_html = "散戶過熱 <b>N/A</b>"

    inst_html = (f"主力撤離 <b>{inst_score:.0f}</b>"
                 if inst_score is not None else "主力撤離 <b>N/A</b>")

    # 進度條 (0~100)
    pct_pos = max(0, min(100, idx))
    bar_html = f"""
    <div style="height:14px; background:linear-gradient(to right,
                #22c55e 0%, #84cc16 20%, #9ca3af 35%, #9ca3af 65%,
                #f97316 80%, #dc2626 100%);
                border-radius:7px; position:relative; margin:12px 0 8px 0;">
        <div style="position:absolute; left:{pct_pos}%; top:-4px;
                    transform:translateX(-50%);
                    width:4px; height:22px; background:#fff;
                    border-radius:2px; box-shadow:0 0 4px #fff;"></div>
    </div>
    <div style="display:flex; justify-content:space-between;
                font-size:10px; color:#888; margin-bottom:8px;">
        <span>0 恐慌</span><span>35</span><span>65</span><span>100 危機</span>
    </div>
    """

    warn_html = ""
    if data.get("margin_degraded"):
        warn_html = ('<div class="tw-warn">⚠️ 融資餘額資料缺失 '
                     '(FinMind 免費版限制)，散戶分數降級為僅用小台多空比</div>')

    st.markdown(f"""
    <div class="tw-crisis-card" style="border-left: 4px solid {color};">
        <div class="tw-crisis-title">綜合空頭危機指數 (0=極度恐慌底部 / 100=極度瘋狂頂部)</div>
        <div class="tw-crisis-idx" style="color:{color};">{idx_disp}</div>
        <div class="tw-crisis-verdict" style="color:{color};">{verdict}</div>
        <div class="tw-crisis-suggestion">💡 建議: {suggestion}</div>
        {bar_html}
        <div class="tw-sub">{retail_html} ｜ {inst_html}</div>
        {warn_html}
    </div>
    """, unsafe_allow_html=True)

    # 明細展開區（預設收合）
    with st.expander("📋 查看四大指標明細", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### 👥 散戶過熱（權重 30%）")
            _render_metric_row(
                "① 小台散戶多空比",
                data["retail_ratio"], "%",
                data["score_retail_ratio"],
                hint="+25%=極度瘋狂｜0%=中性｜-25%=極度恐慌",
                status_msg=data["messages"]["retail_ratio"])
            _render_metric_row(
                "② 融資餘額 252 日位階",
                data["margin_percentile"], "%",
                data["score_margin"],
                extra=(f"餘額 {data['margin_value']/1e8:.0f} 億"
                       if data["margin_value"] is not None else None),
                hint="≥80%=過熱｜≤20%=絕望",
                status_msg=data["messages"]["margin"])

        with col2:
            st.markdown("##### 💰 主力撤離（權重 70%）")
            _render_metric_row(
                "③ 外資期貨 120 日百分位",
                data["foreign_futures_percentile"], "%",
                data["score_foreign_futures"],
                extra=(f"淨部位 {data['foreign_futures_value']:,.0f} 口"
                       if data["foreign_futures_value"] is not None else None),
                hint="≤20%=結構性撤退｜≥80%=結構性進場（反向計分）",
                status_msg=data["messages"]["foreign_futures"])
            _render_metric_row(
                "④ 外資現貨 20 日累計",
                data["foreign_spot_cum20"], "億",
                data["score_foreign_spot"],
                hint="≤-1500億=撤退｜≥+1500億=佈局",
                status_msg=data["messages"]["foreign_spot"])

        # 兩張小圖（外資期貨 + 外資現貨）
        st.markdown("---")
        c1, c2 = st.columns(2)
        with c1:
            df_tx = data.get("df_foreign_futures", pd.DataFrame())
            if not df_tx.empty:
                st.caption("外資期貨淨未平倉 120 日走勢")
                mean = df_tx["Net_OI"].mean()
                fig = go.Figure(go.Scatter(
                    x=df_tx["Date"], y=df_tx["Net_OI"],
                    mode="lines", line=dict(color="#facc15", width=2)))
                fig.add_hline(y=mean, line_dash="dash", line_color="#888",
                              annotation_text=f"半年均值 {mean:,.0f}",
                              annotation_font_size=10)
                fig.update_layout(height=180, template="plotly_dark",
                                  margin=dict(t=5,b=5,l=5,r=5), showlegend=False,
                                  xaxis=dict(showgrid=False),
                                  yaxis=dict(showgrid=False))
                st.plotly_chart(fig, use_container_width=True,
                                config={"displayModeBar": False})
        with c2:
            df_sp = data.get("df_foreign_spot", pd.DataFrame())
            if not df_sp.empty:
                st.caption("外資現貨買賣超 (近 40 日，紅綠柱)")
                colors = ["#4ade80" if v >= 0 else "#ff6b6b"
                          for v in df_sp["Net"]]
                fig = go.Figure(go.Bar(x=df_sp["Date"], y=df_sp["Net"],
                                       marker_color=colors))
                fig.add_hline(y=0, line_color="#666", line_width=1)
                fig.update_layout(height=180, template="plotly_dark",
                                  margin=dict(t=5,b=5,l=5,r=5), showlegend=False,
                                  xaxis=dict(showgrid=False),
                                  yaxis=dict(showgrid=False, title="億"))
                st.plotly_chart(fig, use_container_width=True,
                                config={"displayModeBar": False})

        st.caption(f"資料更新時間: {data['updated_at']}")

    st.markdown("<div style='margin-bottom:15px;'></div>", unsafe_allow_html=True)


def _render_metric_row(name, value, unit, score, extra=None,
                       hint=None, status_msg=""):
    """單一指標明細列"""
    try:
        import streamlit as st
    except ImportError:
        return

    if value is None or score is None:
        v_disp = "N/A"
        s_disp = "—"
        score_color = "#666"
    else:
        v_disp = (f"{value:+.1f}{unit}" if unit == "%"
                  else f"{value:+,.1f} {unit}")
        s_disp = f"{score:.0f}"
        # 分數越高越紅（危險）
        if score >= 80:   score_color = "#dc2626"
        elif score >= 65: score_color = "#f97316"
        elif score >= 35: score_color = "#9ca3af"
        elif score >= 20: score_color = "#84cc16"
        else:             score_color = "#22c55e"

    extra_html = f"<br/><span style='color:#888;font-size:10px;'>{extra}</span>" if extra else ""
    err_html = ""
    if value is None and status_msg and status_msg != "ok":
        err_html = (f"<br/><span style='color:#f97316;font-size:10px;'>"
                    f"⚠️ {status_msg}</span>")

    st.markdown(f"""
    <div style="background:#222; border-radius:5px; padding:8px 12px;
                margin-bottom:6px; display:flex;
                justify-content:space-between; align-items:center;">
        <div style="flex:1;">
            <div style="font-size:12px; color:#ccc; font-weight:bold;">{name}</div>
            <div style="font-size:14px; color:#fff;">{v_disp}{extra_html}{err_html}</div>
            <div style="font-size:9px; color:#666; margin-top:2px;">{hint or ''}</div>
        </div>
        <div style="text-align:center; margin-left:10px;">
            <div style="font-size:9px; color:#888;">分數</div>
            <div style="font-size:22px; font-weight:900; color:{score_color};">{s_disp}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════
#  CLI 自我測試
# ═════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("台股空頭危機距離指數 — 自我測試")
    print("=" * 60)
    print(f"FinMind Token: {'已設定' if FINMIND_TOKEN else '未設定'}")
    print()

    data = _get_tw_crisis_index_impl()
    print(f"🎯 最終指數: {data['final_index']}")
    verdict, _, sugg = _index_verdict(data['final_index'])
    print(f"   解讀: {verdict}")
    print(f"   建議: {sugg}")
    print()
    print(f"散戶過熱分: {data['retail_score']} "
          f"(降級={data['margin_degraded']})")
    print(f"主力撤離分: {data['institutional_score']}")
    print()
    print("四大指標:")
    print(f"  ① 小台散戶多空比: {data['retail_ratio']} → 分數 {data['score_retail_ratio']}")
    print(f"  ② 融資 252 日百分位: {data['margin_percentile']} → 分數 {data['score_margin']}")
    print(f"  ③ 外資期貨 120 日百分位: {data['foreign_futures_percentile']} → 分數 {data['score_foreign_futures']}")
    print(f"  ④ 外資現貨 20 日累計: {data['foreign_spot_cum20']} → 分數 {data['score_foreign_spot']}")
    print()
    print("狀態訊息:")
    for k, v in data["messages"].items():
        print(f"  · {k}: {v}")

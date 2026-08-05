# TODO — 資料日期顯示（app_v18.py）

**記錄日期**：2026-08-04
**專案**：`C:\Users\ksmoo\OneDrive\Desktop\stockapp\`（**不是** sec_pipeline）
**當前版本**：V26.75

---

## 觸發事件

台北 2026-08-04 12:16，dashboard 顯示 SOFI 現價 `16.31`，券商 App 顯示 `18.03`。

查證後確認**不是個股資料錯誤，是整份清單慢一個交易日**：

| | dashboard | 券商 |
|---|---|---|
| SOFI | 16.31（7/31 收） | 18.03（8/3 收） |
| ^IXIC | 25,373.85 | 25,913.90（+2.13%） |

25,913.90 ÷ 25,373.85 = 1.0213，正好等於券商顯示的漲幅 → dashboard 的「現價」對每一檔都等於券商的「前收」。

**清除今日快取無效**，重掃仍是同一份資料。

## 根因（已定位到程式碼，未定位到上游）

`scan_watchlist_icons()`：

```python
d = yf.download(tk, period="1y", interval="1d", auto_adjust=False, progress=False)
icons["_prices"][tk]    = round(float(d["Close"].iloc[-1]), 2)   # 現價
icons["_prices_yd"][tk] = round(float(d["Close"].iloc[-2]), 2)   # 昨日
```

現價就是 `iloc[-1]`，**沒有任何偏移**。所以只有一個解釋：`yf.download` 回來的 DataFrame 缺 8/3 那根，最後一根是 7/31。

**尚未執行的確認步驟**（本機跑，非 Streamlit Cloud）：

```python
import yfinance as yf
d = yf.download("SOFI", period="1mo", interval="1d", auto_adjust=False, progress=False)
print(d[["Close"]].tail(5))
```

- 有 8/3 → 問題在 Streamlit Cloud 端（機房 IP / yfinance 版本 / 容器時鐘）
- 無 8/3 → Yahoo 對 `period=` 的最新日 K 有延遲，兩邊都抓不到，非本專案 bug

當下時間是 ET 8/4 00:16，離美股 8/3 收盤僅 8 小時，踩到 Yahoo 日 K 延遲的機率不低。

---

## 要做的改動（不論上游查證結果如何都該做）

**問題不在「漏一根」，在「漏了你不會知道」。** 畫面寫「現價 $16.31」但沒有任何地方說這是哪一天的價格。當時的戰術建議是「恐慌殺盤中，嚴禁接刀」——而 8/3 那斯達克實際 +2.13%、SOFI +10.55%。**不只是慢，方向是反的。**

V26.71 的 `_wl_src` 只說「來自快取檔」，不說快取裡的價格是哪一天。

### 規格

1. `scan_watchlist_icons()` 新增一個**平行 dict**：

   ```python
   icons.setdefault("_prices_asof", {})[tk] = d.index[-1].strftime("%Y-%m-%d")
   ```

   **必須是字串，不能是 `date` 物件。** 掃描結果會走 `json.dump` 寫進 `_wl_icon_file`，`date` 物件會丟 `TypeError: Object of type date is not JSON serializable`——而那段 `json.dump` 包在 `except Exception: pass` 裡（既有的 Rule 12 違反），會**靜默失敗**，結果是快取永遠寫不進去且毫無錯誤訊息。這個坑會吃掉一整個 session。

2. **用平行 dict，不要改 `_prices` 的值型別。** 既有慣例就是這樣（`_prices_wk` / `_prices_yd` / `_prev_high` / `_iron` 全是平行 dict）。改成 `{ticker: {price, asof}}` 會讓舊快取檔的 `_cached.get("prices", {})` 讀出錯型別。（Rule 11）

3. 一路照既有 pattern 接下去：`_scan_prices_asof` → `st.session_state["_wl_prices_asof"]` → `json.dump` 的 key → 讀快取時 `_cached.get("prices_asof", {})`。

4. 畫面上跟現價並排顯示。**日期不等於最近交易日時標紅**——這才是這個功能的重點，不然只是多一欄沒人看的字。

### 一併考慮（不要順手做，先討論）

那段 `except Exception: pass` 本身違反 Rule 12。改它會擴大 diff 面（Rule 3），但它正是讓這類問題變靜默的機制。**列為獨立議題，不要跟資料日期綁在一起做。**

---

## 與既有已知問題的區別

這跟 0050.TW / 2330.TW 的價格異常是**不同的 bug**：

- 那個：數值離譜（wildly incorrect）
- 這個：數值正確但整批落後一個交易日

不要混在一起查。

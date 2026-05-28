"""
驗證 compute_zigzag_pivots 業務意圖 (Rule 9)：
  能正確找出價格的轉折高低點，且結果交替（H L H L ...）。
測試直接 exec 真實函式原始碼，邏輯被改壞會 fail。
"""
import re
import numpy as np
import pandas as pd

SRC = open("app_v18.py", encoding="utf-8").read()
m = re.search(r"def compute_zigzag_pivots.*?(?=\ndef )", SRC, re.S)
assert m, "找不到 compute_zigzag_pivots"
ns = {"pd": pd, "np": np}
exec(m.group(0), ns)
fn = ns["compute_zigzag_pivots"]


def _mk_df(highs, lows):
    return pd.DataFrame({"High": highs, "Low": lows})


# Test 1：明顯的 V 形（先跌後漲）→ 至少抓到中間那個低點
def test_v_shape():
    n = 5
    # 11 根 K 棒：左 5 根遞減、中間 1 根最低、右 5 根遞增
    highs = [20, 18, 16, 14, 12, 10, 12, 14, 16, 18, 20]
    lows = [h - 1 for h in highs]
    df = _mk_df(highs, lows)
    pivots = fn(df, n=n)
    # 中間 (idx=5) 應該是低點
    assert any(p[0] == 5 and p[2] == 'L' for p in pivots), f"V 底未抓到: {pivots}"
    print("PASS test_v_shape")


# Test 2：高低交替不變式
def test_alternating():
    rng = np.random.default_rng(42)
    # 80 根隨機走勢
    prices = 100 + np.cumsum(rng.normal(0, 1.5, 80))
    highs = prices + 0.5
    lows = prices - 0.5
    df = _mk_df(highs, lows)
    pivots = fn(df, n=5)
    kinds = [p[2] for p in pivots]
    for i in range(1, len(kinds)):
        assert kinds[i] != kinds[i - 1], f"連續同類: {kinds}"
    print(f"PASS test_alternating ({len(pivots)} pivots)")


# Test 3：資料太短應回空 list 不能爆
def test_short_data():
    df = _mk_df([1, 2, 3], [0, 1, 2])
    assert fn(df, n=5) == []
    assert fn(None, n=5) == []
    print("PASS test_short_data")


# Test 4：單調上升不該有高點（除了不存在的右側 fractal）
def test_monotonic_up():
    df = _mk_df(list(range(1, 30)), list(range(0, 29)))
    pivots = fn(df, n=5)
    # 嚴格單調上升的中段不會有 fractal high（右邊永遠更高）
    assert pivots == [], f"單調上升不該有 pivot: {pivots}"
    print("PASS test_monotonic_up")


if __name__ == "__main__":
    test_v_shape()
    test_alternating()
    test_short_data()
    test_monotonic_up()
    print("\nALL 4 TESTS PASSED")

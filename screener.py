import yfinance as yf
import pandas as pd
import time

# --- 設定篩選參數 ---
CRITERIA = {
    'min_drop_pct': 0.0,       # 改成 0.0 (只要有跌或甚至沒跌都算)
    'max_peg': 10.0,           # 改成 10.0 (原本是 1.5，放寬讓貴的股票也能進來)
    'min_rev_growth': 0.0,     # 改成 0.0 (只要營收沒衰退就好)
    'max_debt_equity': 1000    # 改大一點
}

# --- 設定觀察名單 ---
# 這裡先放一些常見的成長股/科技股做測試
# 您之後可以把這裡改成讀取 CSV 或您的 watchlist.json
tickers = [
    'NVDA', 'AMD', 'TSLA', 'PLTR', 'CRWD', 'SNOW', 'DDOG', 'SE', 
    'SHOP', 'NET', 'U', 'RBLX', 'ZS', 'ENPH', 'SQ', 'COIN', 'MDB', 'TEAM'
]

def analyze_stock(ticker):
    """分析單一股票是否符合條件"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # 取得必要數據 (使用 .get 以防數據缺失報錯)
        current_price = info.get('currentPrice')
        high_52 = info.get('fiftyTwoWeekHigh')
        peg = info.get('pegRatio')
        rev_growth = info.get('revenueGrowth')
        debt_equity = info.get('debtToEquity')

        # 檢查數據是否完整
        if None in [current_price, high_52, peg, rev_growth]:
            return None

        # 計算下跌幅度
        drop_from_high = (high_52 - current_price) / high_52

        # --- 核心篩選判斷 ---
        if (drop_from_high >= CRITERIA['min_drop_pct'] and 
            peg <= CRITERIA['max_peg'] and 
            rev_growth >= CRITERIA['min_rev_growth']):
            
            # 債務檢查 (有些公司無債務數據，若有則需符合條件)
            if debt_equity is not None and debt_equity > CRITERIA['max_debt_equity']:
                return None

            return {
                '股票代碼': ticker,
                '現價': current_price,
                '距高點跌幅': f"{drop_from_high:.1%}",
                'PEG': peg,
                '營收成長': f"{rev_growth:.1%}",
                '債務權益比': debt_equity
            }
            
    except Exception as e:
        print(f"跳過 {ticker}: 無法取得數據")
        return None
    
    return None

# --- 主程式執行區 ---
if __name__ == "__main__":
    print(f"🚀 開始掃描 {len(tickers)} 檔股票... (請稍候，網路請求需時間)")
    print("-" * 50)
    
    results = []
    
    for ticker in tickers:
        print(f"正在分析: {ticker}...", end="\r") # end="\r" 讓文字在同一行更新
        data = analyze_stock(ticker)
        if data:
            results.append(data)
        time.sleep(0.5) # 稍微暫停避免被 Yahoo 擋 IP

    print("\n" + "=" * 50)
    
    if results:
        df = pd.DataFrame(results)
        # 依照 PEG 由小到大排序 (最便宜的在上面)
        df = df.sort_values(by='PEG', ascending=True)
        
        print(f"🎉 找到 {len(df)} 檔符合條件的潛力股：\n")
        # 格式化輸出表格
        print(df.to_string(index=False))
        
        # 提示：也可以存成 CSV
        # df.to_csv("undervalued_gems.csv", index=False)
        # print("\n結果已儲存為 undervalued_gems.csv")
    else:
        print("沒有股票符合當前的篩選條件。試著放寬標準看看？")
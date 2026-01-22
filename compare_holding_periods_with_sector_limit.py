import pandas as pd
import numpy as np
import os
from datetime import datetime

# --- Config ---
INITIAL_CAPITAL = 1_000_000.0
COMMISSION_RATE = 0.0001
SLIPPAGE = 0.001
START_DATE = "2024-10-09"
END_DATE = datetime.now().strftime("%Y-%m-%d")
CACHE_DIR = "data_cache"
SCORES = {1: 100, 3: 70, 5: 50, 10: 30, 20: 20}
TOP_N = 10
SECTOR_LIMIT = 1  # 每行业限1只

def get_theme_normalized(name):
    """More robust theme extraction for meaningful grouping"""
    if not name or pd.isna(name): return "Unknown"
    name = name.lower()
    # Priority keywords for grouping
    keywords = ["芯片", "半导体", "人工智能", "ai", "红利", "银行", "机器人", "光伏", "白酒", "医药", "医疗", "军工", "新能源", "券商", "证券", "黄金", "纳斯达克", "标普", "信创", "软件", "房地产", "中药"]
    for k in keywords:
        if k in name: return k
    # Fallback
    theme = name.replace("etf", "").replace("基金", "").replace("增强", "").replace("指数", "")
    for word in ["中证", "沪深", "上证", "深证", "科创", "创业板", "港股通", "300", "500", "1000", "50", "100"]:
        theme = theme.replace(word, "")
    return theme.strip() if theme.strip() else "宽基"

def load_data():
    """Load ETF data"""
    # Load List for Names
    list_files = [f for f in os.listdir(CACHE_DIR) if f.startswith("etf_list_")]
    name_map = {}
    if list_files:
        l_df = pd.read_csv(os.path.join(CACHE_DIR, sorted(list_files)[-1]))
        name_map = dict(zip(l_df['etf_code'], l_df['etf_name']))

    # Load Prices
    price_dict = {}
    files = [f for f in os.listdir(CACHE_DIR) if f.endswith(".csv") and "etf_list" not in f]
    for f in files:
        code = f.replace(".csv", "")
        if not (code.startswith('sh') or code.startswith('sz')): continue
        try:
            df = pd.read_csv(os.path.join(CACHE_DIR, f))
            df['日期'] = pd.to_datetime(df['日期'])
            df.set_index('日期', inplace=True)
            price_dict[code] = df
        except: pass

    closes = pd.DataFrame({k: v['收盘'] for k, v in price_dict.items()}).sort_index()[START_DATE:END_DATE]
    opens = pd.DataFrame({k: v.get('开盘', v['收盘']) for k, v in price_dict.items()}).sort_index()[START_DATE:END_DATE]

    return closes, opens, name_map

def backtest_with_period_and_sector_limit(prices, opens, roll_rets, name_map, holding_period):
    """
    Run backtest with specific holding period AND sector limit.
    """
    dates = prices.index
    cash = INITIAL_CAPITAL
    holdings = {} # code -> qty
    history = []

    for i in range(len(dates) - 1):
        today = dates[i]
        next_day = dates[i+1]

        # Calc Equity
        curr_val = cash + sum(q * prices.loc[today, c] for c, q in holdings.items() if c in prices.columns and not pd.isna(prices.loc[today, c]))
        history.append(curr_val)

        if i % holding_period == 0:
            # Rank
            daily_scores = pd.Series(0, index=prices.columns)
            valid_mask = prices.loc[today].notna()
            for d, weight in SCORES.items():
                r_d = roll_rets[d].loc[today]
                valid_r = r_d[valid_mask & (r_d > -100)]
                if not valid_r.empty:
                    thresh = max(10, int(len(valid_r) * 0.1))
                    top_codes = valid_r.nlargest(thresh).index
                    daily_scores.loc[top_codes] += weight

            # Selection with Sector Limit
            sorted_candidates = daily_scores.sort_values(ascending=False).index
            target_holdings = []
            theme_counts = {}

            for code in sorted_candidates:
                if len(target_holdings) >= TOP_N: break

                theme = get_theme_normalized(name_map.get(code, ""))
                count = theme_counts.get(theme, 0)

                if count < SECTOR_LIMIT:
                    target_holdings.append(code)
                    theme_counts[theme] = count + 1

            # --- Trade on Next Day Open ---
            exec_prices = opens.loc[next_day]
            # Sells
            for c in list(holdings.keys()):
                if c not in target_holdings:
                    qty = holdings[c]
                    price = exec_prices.get(c, 0)
                    if not pd.isna(price) and price > 0:
                        cash += qty * price * (1 - COMMISSION_RATE - SLIPPAGE)
                        del holdings[c]

            # Buys (Refined logic)
            if target_holdings:
                target_per_pos = (cash + sum(q * exec_prices.get(c, 0) for c, q in holdings.items())) / TOP_N
                for c in target_holdings:
                    price = exec_prices.get(c, 0)
                    if pd.isna(price) or price <= 0: continue
                    curr_qty = holdings.get(c, 0)
                    curr_val = curr_qty * price
                    if curr_val < target_per_pos * 0.9:
                        shortfall = target_per_pos - curr_val
                        shares = int(shortfall / (price * (1 + COMMISSION_RATE + SLIPPAGE)))
                        shares = (shares // 100) * 100
                        if shares > 0:
                            cost = shares * price * (1 + COMMISSION_RATE + SLIPPAGE)
                            if cash >= cost:
                                holdings[c] = curr_qty + shares
                                cash -= cost

    # Metrics
    h_series = pd.Series(history)
    total_ret = (h_series.iloc[-1] - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100

    # Calculate Max Drawdown
    drawdown = (h_series / h_series.cummax() - 1).min() * 100

    return total_ret, drawdown

def main():
    print("Loading data for corrected holding period comparison (with sector limit)...")
    closes, opens, name_map = load_data()

    print("Pre-calculating rolling returns...")
    roll_rets = {}
    for d in SCORES.keys():
        roll_rets[d] = closes.pct_change(periods=d).fillna(-999)

    print(f"\n=== Corrected Holding Periods with Sector Limit (每行业限{SECTOR_LIMIT}只) ===")
    print(f"Date Range: {START_DATE} to {END_DATE}")
    print(f"{'Period':<10} | {'Total Return':<15} | {'Max Drawdown':<15} | {'Score'}")
    print("-" * 60)

    results = []

    # Test T1 to T20
    for p in range(1, 21):
        ret, dd = backtest_with_period_and_sector_limit(closes, opens, roll_rets, name_map, p)

        # Simple score: Return / |DD| (Calmar-like)
        score = ret / abs(dd) if dd != 0 else 0

        print(".2f")
        results.append({"Period": f"T{p}", "Return": ret, "MaxDD": dd, "Score": score})

    # Best Period
    best = max(results, key=lambda x: x['Score'])
    print(f"\nBest Period by Score: {best['Period']} (Score: {best['Score']:.2f})")

    # Save to CSV
    df = pd.DataFrame(results)
    df.to_csv("holding_period_comparison_corrected.csv", index=False)
    print(f"\nResults saved to holding_period_comparison_corrected.csv")

if __name__ == "__main__":
    main()
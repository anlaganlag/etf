import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

# --- Config ---
INITIAL_CAPITAL = 1_000_000.0
COMMISSION_RATE = 0.0001
SLIPPAGE = 0.001
REBALANCE_DAYS = 10
START_DATE = "2024-10-09"
END_DATE = datetime.now().strftime("%Y-%m-%d")
CACHE_DIR = "data_cache"
SCORES = {1: 100, 3: 70, 5: 50, 10: 30, 20: 20}
TOP_N = 10

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

def run_backtest_with_limit(prices, opens, roll_rets, name_map, sector_limit):
    dates = prices.index
    cash = INITIAL_CAPITAL
    holdings = {} # code -> qty
    history = []
    
    for i in range(len(dates) - 1):
        today = dates[i]
        next_day = dates[i+1]
        
        # Calc Equity
        curr_val = cash + sum(q * prices.loc[today, c] for c, q in holdings.items())
        history.append(curr_val)
        
        if i % REBALANCE_DAYS == 0:
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
                
                if count < sector_limit:
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
            
            # Buys (Refined logic to avoid over-trading)
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
                        cash -= shares * price * (1 + COMMISSION_RATE + SLIPPAGE)
                        holdings[c] = holdings.get(c, 0) + shares

    final_val = history[-1]
    ret = (final_val - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    
    # Calculate Max Drawdown
    h_series = pd.Series(history)
    drawdown = (h_series / h_series.cummax() - 1).min() * 100
    
    return ret, drawdown

def main():
    print("Loading data for sector limit comparison...")
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
    
    print("Pre-calculating rolling returns...")
    roll_rets = {}
    for d in SCORES.keys():
        roll_rets[d] = closes.pct_change(periods=d).fillna(-999)

    scenarios = [
        {"name": "每行业限1只 (极致分散)", "limit": 1},
        {"name": "每行业限2只 (适度分散)", "limit": 2},
        {"name": "每行业限3只 (适度集中)", "limit": 3},
        {"name": "不限制 (原策略)", "limit": 999}
    ]

    print(f"\n=== Sector Limit Comparison (Since {START_DATE}) ===")
    print(f"{'Scenario':<25} | {'Return':<12} | {'Max DD':<12} | {'Score'}")
    print("-" * 65)

    for s in scenarios:
        ret, dd = run_backtest_with_limit(closes, opens, roll_rets, name_map, s['limit'])
        score = ret / abs(dd) if dd != 0 else 0
        print(f"{s['name']:<25} | {ret:>10.2f}% | {dd:>10.2f}% | {score:.2f}")

if __name__ == "__main__":
    main()

from gm.api import *
import pandas as pd
import os
import json
from datetime import datetime
from dotenv import load_dotenv
from config import config
from src.data_fetcher import DataFetcher
from gm_strategy_rolling import RollingPortfolioManager, Tranche, get_ranking, MIN_SCORE, TOP_N, REBALANCE_PERIOD_T, FRUIT_THEME_BOOST

load_dotenv()

def print_signal_report():
    print("\n" + "="*60)
    print(f"🚀 ROLLING STRATEGY MANUAL SIGNAL GENERATOR")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

    # 1. Initialize Context Mock
    class MockContext:
        def __init__(self):
            self.mode = MODE_LIVE
            self.whitelist = set()
            self.theme_map = {}
            self.prices_df = pd.DataFrame()
            self.now = datetime.now()
            self.days_count = 0
            
    context = MockContext()
    
    # 2. Load Whitelist & Theme
    excel_path = os.path.join(config.BASE_DIR, "ETF合并筛选结果.xlsx")
    df_excel = pd.read_excel(excel_path)
    df_excel.columns = df_excel.columns.str.strip()
    rename_map = {'symbol': 'etf_code', 'sec_name': 'etf_name', 'name_cleaned': 'theme'}
    df_excel = df_excel.rename(columns=rename_map)
    context.whitelist = set(df_excel['etf_code'])
    context.theme_map = df_excel.set_index('etf_code')['theme'].to_dict()

    # 3. Load State & Calculate Rebalance Index
    context.rpm = RollingPortfolioManager(context)
    if not context.rpm.load_state():
        print("❌ Error: No rolling_state.json found. Please run backtest or initialize once first.")
        return

    # To determine which tranche is due today, we need to know how many days have passed since start.
    # We can store 'total_days_observed' in state, or look at the last update date.
    # For now, let's ask the user or look at state.
    # Better yet: Let's calculate based on the number of entries in a history log or similar.
    # Since we don't have a reliable days_count in state yet, let's look at the tranches.
    
    # Simple logic: If we want to know what to do *today*, we need to know the 'day index'.
    # I'll update the strategy to save 'days_count' in state to make this robust.
    # For now, I'll try to find it from the state file if I add it.
    
    with open(context.rpm.state_path, 'r') as f:
        state_data = json.load(f)
        context.days_count = state_data.get("days_count", 0) + 1 # Assume we are checking for the NEXT day

    rebalance_idx = (context.days_count - 1) % context.rpm.params["T"]
    active_tranche = context.rpm.tranches[rebalance_idx]
    
    print(f"📅 Today is Day {context.days_count}. Rebalancing Tranche ID: {active_tranche.id}")

    # 4. Fetch Real-time Data
    token = os.getenv('MY_QUANT_TGM_TOKEN')
    set_token(token)
    
    symbols = list(context.whitelist) + ['SHSE.000001', 'SZSE.399006']
    print(f"📡 Fetching real-time quotes for {len(symbols)} symbols...")
    snapshots = current(symbols=symbols)
    price_map = {s.symbol: s.price for s in snapshots}
    
    # Fetch History for Ranking
    fetcher = DataFetcher(cache_dir=config.DATA_CACHE_DIR)
    # Just build the matrix from cache (assuming user ran the strategy once to refresh cache)
    price_data = {}
    for code in context.whitelist:
        cache_file = os.path.join(config.DATA_CACHE_DIR, f"{code.replace('.','_')}.csv")
        if os.path.exists(cache_file):
            df = pd.read_csv(cache_file)
            df['日期'] = pd.to_datetime(df['日期'])
            price_data[code] = df.set_index('日期')['收盘']
    
    # Append today's prices
    context.prices_df = pd.DataFrame(price_data).sort_index().ffill()
    today_row = pd.Series(price_map, name=pd.to_datetime(datetime.now().date()))
    combined_prices_df = pd.concat([context.prices_df, pd.DataFrame([today_row])]).sort_index().ffill()

    # 5. Get Ranking
    ranking_df, total_scores = get_ranking(context, datetime.now(), prices_df_override=combined_prices_df)
    
    # 6. Risk Management Check
    print("\n--- 🛡️ Risk Management (All Tranches) ---")
    any_sl_tp = False
    for t in context.rpm.tranches:
        to_sell, is_tp = t.check_guard(price_map)
        for sym in to_sell:
            reason = "TP (Take Profit)" if is_tp else "SL (Stop Loss)"
            print(f"⚠️ Tranche {t.id} -> SELL: {sym} ({context.theme_map.get(sym)}) - Reason: {reason}")
            any_sl_tp = True
    if not any_sl_tp:
        print("✅ No SL/TP signals triggered today.")

    # 7. Rebalance Signal
    print(f"\n--- 🔄 Rebalance Signal (Tranche {active_tranche.id}) ---")
    # What to Sell
    if active_tranche.holdings:
        print("Outgoing Holdings (SELL EVERYTHING in this tranche):")
        for sym, shares in active_tranche.holdings.items():
            print(f"  ❌ SELL: {sym} ({context.theme_map.get(sym)}) - Amount: {shares}")
    else:
        print("  (Tranche is currently empty/cash)")

    # What to Buy
    # Mock context for exposure check
    context.prices_df = combined_prices_df
    exposure = context.rpm.get_market_exposure(context, total_scores)
    
    if exposure == 0:
        print("🚫 Market Filter: Market is weak (MA20). STAY IN CASH.")
    else:
        if ranking_df is not None:
            targets = []
            seen = set()
            for code, row in ranking_df.iterrows():
                if row['theme'] not in seen:
                    targets.append(code)
                    seen.add(row['theme'])
                if len(targets) >= context.rpm.params["top_n"]: break
            
            print(f"Incoming Holdings (BUY with {exposure*100}% of Tranche Cash):")
            if targets:
                # Estimate budget
                # Total Value of tranche is cash + market_val
                t_val = active_tranche.total_value 
                invest_total = t_val * exposure
                per_target = invest_total / len(targets)
                for sym in targets:
                    price = price_map.get(sym, 0)
                    est_shares = int(per_target / price / 100) * 100 if price > 0 else 0
                    print(f"  💎 BUY: {sym} ({context.theme_map.get(sym)}) - Est. Price: {price:.3f} - Target Shares: {est_shares}")
            else:
                print("  (No symbols met MIN_SCORE criteria)")
        else:
            print("  (No ranking data available)")

    print("\n" + "="*60)
    print("📝 Note: After performing these trades manually, the state file will NOT update automatically.")
    print("      To sync state, you should ideally run the strategy in 'live' mode once per day.")
    print("="*60 + "\n")

if __name__ == "__main__":
    print_signal_report()

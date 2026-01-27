
import os
import pandas as pd
import numpy as np
from datetime import datetime
from src.data_fetcher import DataFetcher
from config import config

def run_t_tuning():
    print("=== Tuning T for Ultimate Theme Boost Logic (T=1 to T=14) ===")
    config.ensure_dirs()
    fetcher = DataFetcher(cache_dir=config.DATA_CACHE_DIR)
    
    # 1. Load Whitelist & Theme Map
    excel_path = os.path.join(config.BASE_DIR, "ETF合并筛选结果.xlsx")
    df_excel = pd.read_excel(excel_path)
    df_excel.columns = df_excel.columns.str.strip()
    rename_map = {'symbol': 'etf_code', 'sec_name': 'etf_name', 'name_cleaned': 'theme'}
    df_excel = df_excel.rename(columns=rename_map)
    if 'theme' not in df_excel.columns:
        df_excel['theme'] = df_excel['etf_name']
    
    whitelist = set(df_excel['etf_code'])
    theme_map = df_excel.set_index('etf_code')['theme'].to_dict()
    
    # 2. Build Global Price Matrix
    print("Loading price data...")
    price_data = {}
    files = [f for f in os.listdir(config.DATA_CACHE_DIR) if f.endswith('.csv') and (f.startswith('sh') or f.startswith('sz'))]
    for f in files:
        code = f.replace('_', '.').replace('.csv', '')
        if '.' not in code:
            if code.startswith('sh'): code = 'SHSE.' + code[2:]
            elif code.startswith('sz'): code = 'SZSE.' + code[2:]
        try:
            df = pd.read_csv(os.path.join(config.DATA_CACHE_DIR, f), usecols=['日期', '收盘'])
            df['日期'] = pd.to_datetime(df['日期']).dt.tz_localize(None)
            if not df.empty:
                price_data[code] = df.set_index('日期')['收盘']
        except: pass
    
    prices_df = pd.DataFrame(price_data).sort_index().ffill()
    
    # 3. Pre-calculate Rankings
    print("Pre-calculating rankings (this may take a minute)...")
    start_sim = pd.to_datetime("2024-09-01")
    end_sim = pd.to_datetime("2026-01-27")
    sim_dates = prices_df.index[(prices_df.index >= start_sim) & (prices_df.index <= end_sim)]
    
    # Pre-calculate scores for efficiency
    periods_rule = {1: 100, 3: 70, 5: 50, 10: 30, 20: 20}
    threshold = 15
    
    daily_rankings = {} # date -> ranking_df
    daily_strong_market = {} # date -> count of etfs with score >= 150
    
    # Daily returns (for sim)
    daily_rets = prices_df.pct_change(1).shift(-1) # return of the NEXT day
    
    for i, current_dt in enumerate(sim_dates):
        idx = prices_df.index.get_loc(current_dt)
        if idx < 250: continue
        
        # history_prices up to current_dt
        # slice for speed: just need last 21 rows for scores + 1 for base
        # But for ranking we need all columns.
        
        sub_prices = prices_df.iloc[idx-20:idx+1]
        
        base_scores = pd.Series(0.0, index=prices_df.columns)
        for p, pts in periods_rule.items():
            # (current / p-days-ago) - 1
            rets = (sub_prices.iloc[-1] / sub_prices.iloc[-(p+1)]) - 1
            ranks = rets.rank(ascending=False, method='min')
            base_scores += (ranks <= threshold) * pts
        
        valid_base = base_scores[base_scores.index.isin(whitelist)]
        strong_market_count = (valid_base >= 150).sum()
        daily_strong_market[current_dt] = strong_market_count
        
        # Theme Boost
        strong_etfs = valid_base[valid_base >= 150]
        theme_counts = {}
        for code in strong_etfs.index:
            t = theme_map.get(code, 'Unknown')
            theme_counts[t] = theme_counts.get(t, 0) + 1
        
        strong_themes = {t for t, count in theme_counts.items() if count >= 3}
        
        final_scores = valid_base.copy()
        for code in final_scores.index:
            if theme_map.get(code, 'Unknown') in strong_themes:
                final_scores[code] += 50
        
        valid_final = final_scores[final_scores >= 150] # context.min_score
        
        if valid_final.empty:
            daily_rankings[current_dt] = None
        else:
            r20 = (sub_prices.iloc[-1] / sub_prices.iloc[0]) - 1 # Roughly r20
            df = pd.DataFrame({
                'score': valid_final.values,
                'r20': r20[valid_final.index].values,
                'theme': [theme_map.get(c, 'Unknown') for c in valid_final.index]
            }, index=valid_final.index)
            daily_rankings[current_dt] = df.sort_values(by=['score', 'r20'], ascending=False)

    # 4. Simulation Loop
    top_n = 3
    results = []
    
    for T in range(1, 15):
        print(f"Simulating T={T}...")
        cash = 1000000.0
        # Simplify pos structure: {symbol: {'units': X, 'entry_price': Y, 'high_price': Z, 'theme': T}}
        positions = {}
        history = []
        
        days_count = 0
        for current_dt in sim_dates:
            days_count += 1
            if current_dt not in daily_rankings and current_dt not in daily_strong_market:
                history.append(cash + sum([pos['units'] * prices_df.loc[current_dt, sym] for sym, pos in positions.items()]))
                continue
            
            # --- 1. Exit Logic (Stop Loss / Trailing Stop) ---
            symbols_to_del = []
            for sym, pos in positions.items():
                curr_price = prices_df.loc[current_dt, sym]
                pos['high_price'] = max(pos['high_price'], curr_price)
                
                # Stop Loss 20%
                if curr_price < pos['entry_price'] * 0.80:
                    cash += pos['units'] * curr_price
                    symbols_to_del.append(sym)
                    # print(f"[{current_dt}] Stop Loss: {sym}")
                # Trailing Stop
                elif pos['high_price'] > pos['entry_price'] * 1.10:
                    if curr_price < pos['high_price'] * 0.95:
                        cash += pos['units'] * curr_price
                        symbols_to_del.append(sym)
                        # print(f"[{current_dt}] Trailing Stop: {sym}")
            
            for sym in symbols_to_del:
                del positions[sym]
                
            # --- 2. Rebalance / Filling Logic ---
            is_rebalance_day = (days_count - 1) % T == 0
            ranking_df = daily_rankings.get(current_dt)
            strong_market_count = daily_strong_market.get(current_dt, 0)
            
            if is_rebalance_day:
                # Sell everything
                for sym, pos in positions.items():
                    cash += pos['units'] * prices_df.loc[current_dt, sym]
                positions = {}
                
                # Target Selection
                if ranking_df is not None:
                    market_exposure = 1.0 if strong_market_count >= 5 else 0.3
                    seen_themes = set()
                    selected = []
                    for sym, row in ranking_df.iterrows():
                        if row['theme'] not in seen_themes:
                            selected.append(sym)
                            seen_themes.add(row['theme'])
                        if len(selected) >= top_n:
                            break
                    
                    if selected:
                        alloc_per_etf = (cash * 0.98 * market_exposure) / top_n
                        for sym in selected:
                            price = prices_df.loc[current_dt, sym]
                            units = alloc_per_etf / price
                            positions[sym] = {
                                'units': units,
                                'entry_price': price,
                                'high_price': price,
                                'theme': theme_map.get(sym, 'Unknown')
                            }
                        cash -= (len(selected) * alloc_per_etf)
            else:
                # Fill logic
                if len(positions) < top_n and strong_market_count >= 5:
                    if ranking_df is not None:
                        held_themes = {pos['theme'] for pos in positions.values()}
                        empty_slots = top_n - len(positions)
                        
                        alloc_per_etf = (1000000.0 * 0.98) / top_n # Use initial cash or current port value? GM uses target percent.
                        # Let's use 1/top_n of current portfolio value roughly
                        port_value = cash + sum([pos['units'] * prices_df.loc[current_dt, sym] for sym, pos in positions.items()])
                        weight = 0.98 / top_n
                        
                        for sym, row in ranking_df.iterrows():
                            if sym not in positions and row['theme'] not in held_themes:
                                price = prices_df.loc[current_dt, sym]
                                # How much to buy? 
                                # In GM: order_target_percent(percent=weight)
                                # So it buys weight * current_total_equity
                                target_val = port_value * weight
                                if cash >= target_val:
                                    positions[sym] = {
                                        'units': target_val / price,
                                        'entry_price': price,
                                        'high_price': price,
                                        'theme': row['theme']
                                    }
                                    cash -= target_val
                                    held_themes.add(row['theme'])
                                    if len(positions) >= top_n:
                                        break

            # End of day valuation
            total_val = cash + sum([pos['units'] * prices_df.loc[current_dt, sym] for sym, pos in positions.items()])
            history.append(total_val)
            
        final_ret = (history[-1] / 1000000.0 - 1) * 100
        results.append({'T': T, 'Return': final_ret})
        print(f"  T={T}: {final_ret:.2f}%")

    df_res = pd.DataFrame(results)
    print("\nFinal Results:")
    print(df_res)
    
    # Save results
    df_res.to_csv("output/data/tuning_t_results.csv", index=False)
    
if __name__ == "__main__":
    run_t_tuning()

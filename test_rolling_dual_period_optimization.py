# coding=utf-8
"""
Rolling策略双周期全维度优化测试 (极速版 V4 - 全市场对齐)
1. 全周期 (2021-12-03 起)
2. 牛市起点 (2024-09-01 起)
对齐要点:
- 加载缓存中全量ETF (约800+), 确保Top15含金量
- 统一使用 SHSE_000001 (上证指数) SMA20 择时
- 交易品种限于 掘金白名单 (whitelist)
- 网格搜索: T, N, Score
"""
from __future__ import print_function, absolute_import
import pandas as pd
import numpy as np
import os
import itertools
import re
from datetime import datetime
from dotenv import load_dotenv
from config import config

load_dotenv()

# --- Rolling Backtest Engine ---
class Tranche:
    def __init__(self, t_id, initial_cash=0):
        self.id = t_id
        self.cash = initial_cash
        self.holdings = {}
        self.pos_records = {} # {symbol: {'entry_price': x, 'high_price': y}}
        self.total_value = initial_cash
        self.rest_days = 0

    def update_value(self, price_map):
        val = self.cash
        for sym, shares in list(self.holdings.items()):
            price = price_map.get(sym, 0)
            if price > 0:
                val += shares * price
                if sym in self.pos_records:
                    self.pos_records[sym]['high_price'] = max(
                        self.pos_records[sym]['high_price'], price
                    )
        self.total_value = val

    def check_guard(self, price_map, stop_loss=0.15, trailing_trigger=0.08, trailing_drop=0.03):
        to_sell = []
        is_tp = False
        for sym, rec in self.pos_records.items():
            if sym not in self.holdings: continue
            curr_price = price_map.get(sym, 0)
            if curr_price <= 0: continue
            entry = rec['entry_price']; high = rec['high_price']
            
            if curr_price < entry * (1 - stop_loss):
                to_sell.append(sym); continue
            if high > entry * (1 + trailing_trigger):
                if curr_price < high * (1 - trailing_drop):
                    to_sell.append(sym); is_tp = True
        return to_sell, is_tp

    def sell(self, symbol, price):
        if symbol in self.holdings:
            shares = self.holdings[symbol]
            self.cash += shares * price
            del self.holdings[symbol]
            if symbol in self.pos_records: del self.pos_records[symbol]

    def buy(self, symbol, cash_allocated, price):
        if price <= 0: return
        shares = int(cash_allocated / price / 100) * 100
        cost = shares * price
        if shares > 0 and self.cash >= cost:
            self.cash -= cost
            self.holdings[symbol] = self.holdings.get(symbol, 0) + shares
            self.pos_records[symbol] = {'entry_price': price, 'high_price': price}

def precalculate_rankings(prices_df, whitelist, theme_map):
    print(f"Pre-calculating rankings across {prices_df.shape[1]} symbols...")
    rank_cache = {}
    base_score_cache = {}
    periods_rule = {1: 100, 3: 70, 5: 50, 10: 30, 20: 20}
    threshold = 15
    dates = prices_df.index
    
    for i, current_dt in enumerate(dates):
        if i % 200 == 0: print(f"  Progress: {i}/{len(dates)}")
        history_prices = prices_df.iloc[max(0, i-252):i+1] # Need ~1 year for some filters, 50 days min
        if len(history_prices) < 21: continue # Basic minimum for ranking
        
        # 1. Base Score (Ranking across ALL loaded ETFs)
        base_scores = pd.Series(0.0, index=history_prices.columns)
        for p, pts in periods_rule.items():
            if len(history_prices) < p + 1: continue
            try:
                rets = (history_prices.iloc[-1] / history_prices.iloc[-(p+1)]) - 1
                ranks = rets.rank(ascending=False, method='min')
                base_scores += (ranks <= threshold) * pts
            except: pass

        # 2. Theme Boost (Only for whitelist items)
        valid_whitelist = base_scores[base_scores.index.isin(whitelist)]
        strong_etfs = valid_whitelist[valid_whitelist >= 150]
        theme_counts = {}
        for code in strong_etfs.index:
            t = theme_map.get(code, 'Unknown'); theme_counts[t] = theme_counts.get(t, 0) + 1
        strong_themes = {t for t, count in theme_counts.items() if count >= 3}
        
        final_scores = valid_whitelist.copy()
        for code in final_scores.index:
            if theme_map.get(code, 'Unknown') in strong_themes: 
                final_scores[code] += 50
        
        r20 = (history_prices.iloc[-1] / history_prices.iloc[-21] - 1) if len(history_prices) >= 21 else pd.Series(0.0, index=history_prices.columns)
        
        rank_cache[current_dt] = pd.DataFrame({
            'score': final_scores,
            'r20': r20[final_scores.index],
            'theme': [theme_map.get(c, 'Unknown') for c in final_scores.index]
        })
        base_score_cache[current_dt] = base_scores # For exposure calculation
        
    return rank_cache, base_score_cache

def run_single_backtest(params, prices_df, rank_cache, base_score_cache, start_date, end_date):
    T = params['T']; top_n = params['top_n']; min_score = params['min_score']
    actual_bt_df = prices_df[(prices_df.index >= start_date) & (prices_df.index <= end_date)]
    if actual_bt_df.empty: return None

    initial_cash = 1000000
    tranches = [Tranche(i, initial_cash / T) for i in range(T)]
    days_count = 0; equity_curve = []
    
    mkt_idx = 'SHSE.000001'

    for current_dt in actual_bt_df.index:
        days_count += 1
        price_map = prices_df.loc[current_dt].to_dict()
        
        # A. Update and Guard
        for tranche in tranches:
            tranche.update_value(price_map)
            if tranche.rest_days > 0: tranche.rest_days -= 1
            to_sell_list, is_tp = tranche.check_guard(price_map)
            if to_sell_list:
                if is_tp: tranche.rest_days = 1
                for sym in to_sell_list: tranche.sell(sym, price_map.get(sym, 0))

        # B. Rebalance
        rebalance_idx = (days_count - 1) % T
        active_tranche = tranches[rebalance_idx]
        for sym in list(active_tranche.holdings.keys()):
            price = price_map.get(sym, 0); active_tranche.sell(sym, price) if price > 0 else None

        # C. Market Exposure (SHSE.000001 MA20)
        exposure = 0.3 # Default
        if current_dt in prices_df.index and mkt_idx in prices_df.columns:
            mkt_prices = prices_df[prices_df.index <= current_dt][mkt_idx]
            if len(mkt_prices) >= 20:
                ma20 = mkt_prices.rolling(20).mean().iloc[-1]
                if mkt_prices.iloc[-1] >= ma20:
                    base_scores = base_score_cache.get(current_dt)
                    strong = (base_scores >= 150).sum() if base_scores is not None else 0
                    exposure = 1.0 if strong >= 5 else 0.3
                else: 
                    exposure = 0.0 # Market weak
            else:
                exposure = 0.3 # Not enough data for MA20

        # D. Buy
        ranking_all = rank_cache.get(current_dt)
        if ranking_all is not None and active_tranche.rest_days == 0 and exposure > 0:
            valid_ranking = ranking_all[ranking_all['score'] >= min_score].sort_values(by=['score', 'r20'], ascending=False)
            if not valid_ranking.empty:
                targets = []; seen = set()
                for code, row in valid_ranking.iterrows():
                    if row['theme'] not in seen:
                        targets.append(code); seen.add(row['theme'])
                    if len(targets) >= top_n: break
                if targets:
                    per_amt = (active_tranche.cash * exposure) / len(targets)
                    for sym in targets:
                        price = price_map.get(sym, 0); active_tranche.buy(sym, per_amt, price) if price > 0 else None

        active_tranche.update_value(price_map)
        equity_curve.append(sum(t.total_value for t in tranches))

    res_df = pd.DataFrame({'val': equity_curve})
    final_ret = (res_df['val'].iloc[-1] / initial_cash - 1) * 100
    max_dd = ((res_df['val'] / res_df['val'].cummax() - 1).min()) * 100
    return {'Ret': final_ret, 'MaxDD': max_dd}

def main():
    # 1. Load Whitelist
    print("Loading whitelist and theme map...")
    excel_path = os.path.join(config.BASE_DIR, "ETF合并筛选结果.xlsx")
    df_excel = pd.read_excel(excel_path)
    df_excel.columns = df_excel.columns.str.strip()
    whitelist = set(df_excel.rename(columns={'symbol': 'etf_code'})['etf_code'])
    theme_map = df_excel.rename(columns={'symbol': 'etf_code', 'sec_name': 'etf_name', 'name_cleaned': 'theme'}).set_index('etf_code')['theme'].to_dict()

    # 2. Load ALL Relevant Data
    print("Loading all ETFs from cache for comprehensive ranking...")
    price_data = {}
    files = [f for f in os.listdir(config.DATA_CACHE_DIR) if f.endswith('.csv')]
    
    # Filter: (SHSE|SZSE)_[0-9]{6}.csv
    pattern = re.compile(r'^(SHSE|SZSE|sh|sz)_?(\d{6})\.csv$', re.I)
    target_files = []
    for f in files:
        m = pattern.match(f)
        if m: target_files.append(f)
        elif '000001' in f: target_files.append(f) # Benchmark

    print(f"Loading {len(target_files)} CSV files...")
    for f in target_files:
        path = os.path.join(config.DATA_CACHE_DIR, f)
        # Standardize symbol name
        m = pattern.match(f)
        if m:
            prefix = m.group(1).upper()
            if prefix == 'SH': prefix = 'SHSE'
            elif prefix == 'SZ': prefix = 'SZSE'
            code = f"{prefix}.{m.group(2)}"
        elif '000001' in f:
            code = 'SHSE.000001'
        else: continue
            
        try:
            df = pd.read_csv(path, usecols=['日期', '收盘'])
            df['日期'] = pd.to_datetime(df['日期']).dt.tz_localize(None)
            price_data[code] = df.set_index('日期')['收盘']
        except: pass
    
    prices_df = pd.DataFrame(price_data).sort_index().ffill()
    print(f"Price matrix built: {prices_df.shape}")

    # 3. Precompute Rankings (Ranking needs full market)
    rank_cache, base_score_cache = precalculate_rankings(prices_df, whitelist, theme_map)

    # 3.5 Optimize backtest matrix
    # Once rankings are precalculated, we only need prices for whitelist ETFs and benchmark
    needed_cols = list(whitelist) + ['SHSE.000001']
    backtest_prices_df = prices_df[[c for c in needed_cols if c in prices_df.columns]].copy()
    print(f"Optimized backtest price matrix: {backtest_prices_df.shape}")

    # 4. Grid Search
    T_vals = [4, 6, 8, 10, 14]
    N_vals = [3, 5]
    S_vals = [50, 100, 150]
    periods = [
        {'n': 'Full', 's': '2021-12-03', 'e': '2026-01-27'}, 
        {'n': 'Bull', 's': '2024-09-01', 'e': '2026-01-27'}
    ]
    
    results = []
    combos = list(itertools.product(T_vals, N_vals, S_vals))
    print(f"Starting Grid Search (Total combos: {len(combos)})...")
    for i, (T, N, S) in enumerate(combos):
        print(f"Combo {i+1}/{len(combos)}: T={T}, N={N}, Score={S}")
        row = {'T': T, 'N': N, 'S': S}
        for p in periods:
            res = run_single_backtest({'T': T, 'top_n': N, 'min_score': S}, backtest_prices_df, rank_cache, base_score_cache, p['s'], p['e'])
            if res:
                row[f"{p['n']}_Ret"] = res['Ret']
                row[f"{p['n']}_MaxDD"] = res['MaxDD']
        results.append(row)

    res_df = pd.DataFrame(results)
    output_path = os.path.join(config.BASE_DIR, 'output', 'data', 'rolling_final_grid_search.csv')
    res_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    print("\n--- RESULTS SORTED BY BULL MARKET RETURN ---")
    print(res_df.sort_values('Bull_Ret', ascending=False).head(10)[['T', 'N', 'S', 'Bull_Ret', 'Bull_MaxDD', 'Full_Ret', 'Full_MaxDD']])
    
    print("\n--- RESULTS SORTED BY FULL CYCLE RETURN ---")
    print(res_df.sort_values('Full_Ret', ascending=False).head(10)[['T', 'N', 'S', 'Bull_Ret', 'Bull_MaxDD', 'Full_Ret', 'Full_MaxDD']])

if __name__ == '__main__':
    main()

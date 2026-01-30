# coding=utf-8
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime
from config import config

# --- 策略参数 ---
STOP_LOSS = 0.05
TRAILING_TRIGGER = 0.06
TRAILING_DROP = 0.02
MIN_SCORE = 20
MAX_PER_THEME = 1
SCORING_METHOD = 'SMOOTH'
DYNAMIC_POSITION = True

class FixedSimulator:
    def __init__(self):
        self.load_data()
        self.precompute_rankings()

    def load_data(self):
        excel_path = os.path.join(config.BASE_DIR, "ETF合并筛选结果.xlsx")
        df_excel = pd.read_excel(excel_path)
        df_excel.columns = df_excel.columns.str.strip()
        target_col = 'symbol' if 'symbol' in df_excel.columns else df_excel.columns[1]
        self.whitelist = set(df_excel[target_col].astype(str).str.strip())
        df_excel['etf_code'] = df_excel[target_col].astype(str).str.strip()
        t_col = 'theme' if 'theme' in df_excel.columns else ('name_cleaned' if 'name_cleaned' in df_excel.columns else 'sec_name')
        self.theme_map = df_excel.set_index('etf_code')[t_col].to_dict()

        price_data = {}
        files = [f for f in os.listdir(config.DATA_CACHE_DIR) if f.endswith('.csv') and (f.startswith('sh') or f.startswith('sz'))]
        for f in files:
            prefix = 'SHSE.' if f.startswith('sh') else 'SZSE.'
            code = prefix + f[2:-4]
            if code in self.whitelist:
                try:
                    df = pd.read_csv(os.path.join(config.DATA_CACHE_DIR, f), usecols=['日期', '收盘', 'open'])
                    df = df.rename(columns={'open': '开盘'})
                    df['日期'] = pd.to_datetime(df['日期']).dt.tz_localize(None)
                    if not df.empty:
                        price_data[code] = df.set_index('日期')
                except: pass
        
        self.raw_data = price_data
        all_closes = {c: df['收盘'] for c, df in price_data.items()}
        self.prices_df = pd.DataFrame(all_closes).sort_index().ffill()
        print(f"Loaded {self.prices_df.shape[1]} symbols.")

    def precompute_rankings(self):
        print("Precomputing rankings for all dates...")
        self.rankings = {}
        self.market_positions = {}
        dates = self.prices_df.index
        
        # Precompute market regime
        for i, dt in enumerate(dates):
            if i < 60:
                self.market_positions[dt] = 1.0
                continue
            recent = self.prices_df.iloc[i-59:i+1]
            ma20 = recent.tail(20).mean()
            ma60 = recent.mean()
            current = recent.iloc[-1]
            strength = ((current > ma20).sum() / len(current) + (current > ma60).sum() / len(current)) / 2
            if strength > 0.6: self.market_positions[dt] = 1.0
            elif strength > 0.4: self.market_positions[dt] = 0.8
            else: self.market_positions[dt] = 0.6

        # Precompute Ranking DataFrames
        periods_rule = {1: 100, 3: 70, 5: 50, 10: 30, 20: 20}
        
        # Calculate returns for all periods at once for all dates
        rets_all = {}
        for p in periods_rule.keys():
            # Shift by p days and calculate return
            rets_all[p] = (self.prices_df / self.prices_df.shift(p)) - 1
            
        for i, dt in enumerate(dates):
            if i < 251: continue
            
            base_scores = pd.Series(0.0, index=self.prices_df.columns)
            valid_rets = {}
            
            for p, pts in periods_rule.items():
                rets = rets_all[p].iloc[i]
                ranks = rets.rank(ascending=False, method='min')
                if SCORING_METHOD == 'SMOOTH':
                    decay = ((30 - ranks) / 30).clip(lower=0)
                    base_scores += decay * pts
                else:
                    base_scores += (ranks <= 15) * pts
                valid_rets[f'r{p}'] = rets
            
            valid_scores = base_scores[base_scores.index.isin(self.whitelist) & (base_scores >= MIN_SCORE)]
            if valid_scores.empty: continue
            
            data_to_df = {'score': valid_scores, 'theme': [self.theme_map.get(c, 'Unknown') for c in valid_scores.index], 'etf_code': valid_scores.index}
            for p in periods_rule.keys(): data_to_df[f'r{p}'] = valid_rets[f'r{p}'][valid_scores.index]
            
            df = pd.DataFrame(data_to_df)
            sort_cols = ['score', 'r1', 'r3', 'r5', 'r10', 'r20', 'etf_code']
            asc_order = [False, False, False, False, False, False, True]
            self.rankings[dt] = df.sort_values(by=sort_cols, ascending=asc_order)
        print("Precomputation finished.")

    def simulate(self, T, top_n, stop_loss, trailing_trigger, trailing_drop, start_date, end_date):
        dates = self.prices_df.index[(self.prices_df.index >= pd.Timestamp(start_date)) & (self.prices_df.index <= pd.Timestamp(end_date))]
        if len(dates) < T: return None

        tranches_cash = [1000000.0 / T] * T
        tranches_holdings = [{} for _ in range(T)]
        nav_history = []
        
        for i, dt in enumerate(dates):
            price_map = self.prices_df.loc[dt].to_dict()
            
            for t_idx in range(T):
                holdings = tranches_holdings[t_idx]
                guard_triggered = False
                to_sell = []
                for sym, info in list(holdings.items()):
                    curr_p = price_map.get(sym, 0)
                    if curr_p <= 0: continue
                    info['high'] = max(info['high'], curr_p)
                    # Use provided stop_loss and trailing parameters
                    if (curr_p < info['entry'] * (1 - stop_loss)) or \
                       (info['high'] > info['entry'] * (1 + trailing_trigger) and curr_p < info['high'] * (1 - trailing_drop)):
                        to_sell.append(sym)
                
                if to_sell:
                    guard_triggered = True
                    for sym in to_sell:
                        tranches_cash[t_idx] += holdings[sym]['shares'] * price_map[sym]
                        del holdings[sym]
                
                if i % T == t_idx:
                    # ========= T日尾盘成交模式 (更加激进的配置) =========
                    for sym, info in list(holdings.items()):
                        exit_p = price_map.get(sym, 0)
                        if exit_p > 0: tranches_cash[t_idx] += info['shares'] * exit_p
                    holdings.clear()
                    
                    if not guard_triggered:
                        ranking_df = self.rankings.get(dt)
                        if ranking_df is not None:
                            targets = []
                            theme_count = {}
                            # 优化: 允许每个主题持仓 2 只，捕捉主线爆发
                            for code, row in ranking_df.iterrows():
                                theme = row['theme']
                                if theme_count.get(theme, 0) < 2: 
                                    targets.append(code)
                                    theme_count[theme] = theme_count.get(theme, 0) + 1
                                if len(targets) >= top_n: break
                            
                            if targets:
                                strength = self.market_positions.get(dt, 0.6)
                                # 优化: 更加激进的择时，强势期满仓，中性期不大幅减仓
                                if strength > 0.6: exposure = 1.0 # 强势
                                elif strength > 0.4: exposure = 0.9 # 中性
                                else: exposure = 0.5 # 弱势
                                
                                budget = tranches_cash[t_idx] * exposure
                                per_amt = budget / len(targets)
                                for sym in targets:
                                    entry_p = price_map.get(sym, 0)
                                    if entry_p > 0:
                                        shares = int(per_amt / entry_p / 100) * 100
                                        if shares > 0:
                                            tranches_cash[t_idx] -= shares * entry_p
                                            holdings[sym] = {'shares': shares, 'entry': entry_p, 'high': entry_p}

            total_val = sum(tranches_cash) + sum(sum(info['shares'] * price_map.get(sym, 0) for sym, info in h.items()) for h in tranches_holdings)
            nav_history.append(total_val)

        nav = pd.Series(nav_history)
        ret = (nav.iloc[-1] / nav.iloc[0] - 1) * 100
        dd = ((nav - nav.cummax()) / nav.cummax()).min() * 100
        daily_ret = nav.pct_change().dropna()
        sharpe = np.sqrt(252) * daily_ret.mean() / daily_ret.std() if daily_ret.std() > 0 else 0
        return {'return': ret, 'max_dd': dd, 'sharpe': sharpe}

    def grid_search(self):
        print("\n=== Grid Search: TOP_N × REBALANCE_PERIOD_T ===")
        print("Test Period: 2024-09-01 to 2026-01-23 (Bull Market)")
        print("Fixed Params: SL=0.03, TG=0.08, DROP=0.03, MAX_PER_THEME=2\n")
        
        n_range = [5, 6, 7, 8, 9, 10]
        t_range = [10, 11, 12, 13, 14, 15]
        
        results = []
        for n in n_range:
            for t in t_range:
                res = self.simulate(T=t, top_n=n, stop_loss=0.03, trailing_trigger=0.08, trailing_drop=0.03, 
                                   start_date='2024-09-01', end_date='2026-01-23')
                if res:
                    # Composite score = Sharpe * (1 - |MaxDD|/100)
                    composite = res['sharpe'] * (1 - abs(res['max_dd'])/100)
                    print(f"N={n}, T={t} | Ret={res['return']:6.2f}% DD={res['max_dd']:6.2f}% Sharpe={res['sharpe']:.2f} Score={composite:.2f}")
                    results.append({'N': n, 'T': t, 'Return': res['return'], 'MaxDD': res['max_dd'], 'Sharpe': res['sharpe'], 'Composite': composite})
        return pd.DataFrame(results)

if __name__ == '__main__':
    sim = FixedSimulator()
    df = sim.grid_search()
    df.to_csv('grid_search_results.csv', index=False)
    print("\nSaved result to: grid_search_results.csv")
    
    # Find best combination
    if not df.empty:
        best = df.loc[df['Composite'].idxmax()]
        print(f"\n=== BEST COMBINATION ===")
        print(f"N={int(best['N'])}, T={int(best['T'])}")
        print(f"Return: {best['Return']:.2f}%")
        print(f"Max DD: {best['MaxDD']:.2f}%")
        print(f"Sharpe: {best['Sharpe']:.2f}")
        print(f"Composite Score: {best['Composite']:.2f}")
    
    # Print matrices
    print("\n=== RETURN MATRIX (%) ===")
    pivot_ret = df.pivot(index='N', columns='T', values='Return')
    print(pivot_ret.round(2))
    
    print("\n=== SHARPE MATRIX ===")
    pivot_shp = df.pivot(index='N', columns='T', values='Sharpe')
    print(pivot_shp.round(2))
    
    print("\n=== COMPOSITE SCORE MATRIX ===")
    pivot_comp = df.pivot(index='N', columns='T', values='Composite')
    print(pivot_comp.round(2))


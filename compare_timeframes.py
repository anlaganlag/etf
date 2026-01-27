# coding=utf-8
from __future__ import print_function, absolute_import
from gm.api import *
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime
from dotenv import load_dotenv

from config import config

load_dotenv()

# --- Configuration ---
# Hardcoded to User Request V6.0 Defaults
TOP_N = 3
MIN_SCORE = 150
FRUIT_THEME_BOOST = True
STOP_LOSS = 0.20
TRAILING_TRIGGER = 0.10
TRAILING_DROP = 0.05
FIXED_T = 14
END_DATE_STR = '2026-01-23 16:00:00'

class BacktestEngine:
    def __init__(self, t_period, mode='periodic'):
        self.T = t_period
        self.mode = mode # 'periodic' or 'rolling'
        self.prices_df = None
        self.whitelist = set()
        self.theme_map = {}
        self.cash = 1000000.0
        self.history_nav = []
        self.dates = []
        
        # Internal State
        self.days_count = 0
        self.tranches = [] # For rolling
        self.periodic_holdings = {} # {sym: shares}
        self.periodic_pos_records = {} # {sym: {entry, high}}
        
        self.load_data()
        self.init_portfolio()

    def load_data(self):
        # Load Whitelist
        excel_path = os.path.join(config.BASE_DIR, "ETF合并筛选结果.xlsx")
        df_excel = pd.read_excel(excel_path)
        df_excel.columns = df_excel.columns.str.strip()
        rename_map = {'symbol': 'etf_code', 'sec_name': 'etf_name', 'name_cleaned': 'theme'}
        df_excel = df_excel.rename(columns=rename_map)
        if 'theme' not in df_excel.columns: df_excel['theme'] = df_excel['etf_name']
        self.whitelist = set(df_excel['etf_code'])
        self.theme_map = df_excel.set_index('etf_code')['theme'].to_dict()

        # Load Prices (Optimization: Load once globally if possible, but class isolation is safer)
        if not hasattr(BacktestEngine, 'global_prices'):
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
                    price_data[code] = df.set_index('日期')['收盘']
                except: pass
            BacktestEngine.global_prices = pd.DataFrame(price_data).sort_index().ffill()
        self.prices_df = BacktestEngine.global_prices

    def init_portfolio(self):
        if self.mode == 'rolling':
            share = self.cash / self.T
            for i in range(self.T):
                # Tranche structure: {cash, holdings: {sym: shares}, records: {sym: {entry, high}}}
                self.tranches.append({
                    'id': i,
                    'cash': share,
                    'holdings': {},
                    'records': {},
                    'value': share
                })
        else:
            self.periodic_holdings = {}
            self.periodic_pos_records = {}

    def get_ranking(self, current_dt):
        history_prices = self.prices_df[self.prices_df.index <= current_dt]
        if len(history_prices) < 251: return None, None

        periods_rule = {1: 100, 3: 70, 5: 50, 10: 30, 20: 20}
        threshold = 15
        base_scores = pd.Series(0.0, index=history_prices.columns)
        for p, pts in periods_rule.items():
            rets = (history_prices.iloc[-1] / history_prices.iloc[-(p+1)]) - 1
            ranks = rets.rank(ascending=False, method='min')
            base_scores += (ranks <= threshold) * pts
        
        valid_base = base_scores[base_scores.index.isin(self.whitelist)]
        
        if FRUIT_THEME_BOOST:
            strong_etfs = valid_base[valid_base >= 150]
            theme_counts = {}
            for code in strong_etfs.index:
                t = self.theme_map.get(code, 'Unknown')
                theme_counts[t] = theme_counts.get(t, 0) + 1
            strong_themes = {t for t, count in theme_counts.items() if count >= 3}
            final_scores = valid_base.copy()
            for code in final_scores.index:
                if self.theme_map.get(code, 'Unknown') in strong_themes:
                    final_scores[code] += 50
        else:
            final_scores = valid_base

        valid_final = final_scores[final_scores >= MIN_SCORE]
        return valid_final, base_scores

    def run_daily_logic(self, current_dt):
        # Slice prices
        if current_dt not in self.prices_df.index: return
        today_prices = self.prices_df.loc[current_dt]
        price_map = today_prices.to_dict()

        ranking_df, total_scores = None, None
        
        # Ranking Logic
        need_ranking = False
        if self.mode == 'rolling': 
            need_ranking = True
        else:
            if (self.days_count - 1) % self.T == 0: need_ranking = True
            
        if need_ranking:
            valid_scores, base_scores = self.get_ranking(current_dt)
            if valid_scores is not None and not valid_scores.empty:
                df = pd.DataFrame({'score': valid_scores, 'theme': [self.theme_map.get(c, 'Unknown') for c in valid_scores.index]})
                ranking_df = df.sort_values(by='score', ascending=False)
                total_scores = base_scores

        # --- EXECUTION ---
        if self.mode == 'periodic':
            self.run_periodic(current_dt, price_map, ranking_df, total_scores)
        else:
            self.run_rolling(current_dt, price_map, ranking_df, total_scores)

        self.days_count += 1
        
        # Record NAV
        total_val = self.get_total_value(price_map)
        self.history_nav.append(total_val)
        self.dates.append(current_dt)

    def run_periodic(self, current_dt, price_map, ranking_df, total_scores):
        # 1. Guard
        to_sell = []
        for sym in list(self.periodic_holdings.keys()):
            price = price_map.get(sym, 0)
            if price <= 0: continue
            
            if sym not in self.periodic_pos_records: self.periodic_pos_records[sym] = {'entry': price, 'high': price}
            rec = self.periodic_pos_records[sym]
            rec['high'] = max(rec['high'], price)
            
            if price < rec['entry'] * (1 - STOP_LOSS) or \
               (rec['high'] > rec['entry'] * (1 + TRAILING_TRIGGER) and price < rec['high'] * (1 - TRAILING_DROP)):
               to_sell.append(sym)
        
        for sym in to_sell:
            shares = self.periodic_holdings[sym]
            self.cash += shares * price_map[sym]
            del self.periodic_holdings[sym]
            if sym in self.periodic_pos_records: del self.periodic_pos_records[sym]

        # 2. Rebalance
        if (self.days_count - 1) % self.T == 0:
            for sym, shares in list(self.periodic_holdings.items()):
                self.cash += shares * price_map.get(sym, 0)
            self.periodic_holdings = {}
            self.periodic_pos_records = {} # Reset
            
            if ranking_df is not None:
                targets = []
                seen = set()
                for sym, row in ranking_df.iterrows():
                    if row['theme'] not in seen:
                        targets.append(sym)
                        seen.add(row['theme'])
                    if len(targets) >= TOP_N: break
                
                strong_cnt = (total_scores >= 150).sum() if total_scores is not None else 0
                exposure = 1.0 if strong_cnt >= 5 else 0.3
                
                invest_amt = self.cash * exposure
                if targets:
                    per_amt = invest_amt / len(targets)
                    for sym in targets:
                        price = price_map.get(sym, 0)
                        if price > 0:
                            shares = int(per_amt / price / 100) * 100
                            if shares > 0 and self.cash >= shares * price:
                                self.cash -= shares * price
                                self.periodic_holdings[sym] = shares
                                self.periodic_pos_records[sym] = {'entry': price, 'high': price}

    def run_rolling(self, current_dt, price_map, ranking_df, total_scores):
        # 1. Update & Guard All Tranches
        for t in self.tranches:
            current_val = t['cash']
            to_sell = []
            for sym, shares in t['holdings'].items():
                price = price_map.get(sym, 0)
                if price <= 0: continue
                current_val += shares * price
                rec = t['records'][sym]
                rec['high'] = max(rec['high'], price)
                if price < rec['entry']*(1-STOP_LOSS) or \
                   (rec['high'] > rec['entry']*(1+TRAILING_TRIGGER) and price < rec['high']*(1-TRAILING_DROP)):
                   to_sell.append(sym)
            for sym in to_sell:
                rev = t['holdings'][sym] * price_map[sym]
                t['cash'] += rev
                del t['holdings'][sym]
                del t['records'][sym]
            t['value'] = current_val

        # 2. Rebalance Active Tranche
        idx = (self.days_count - 1) % self.T
        active_t = self.tranches[idx]
        
        for sym, shares in list(active_t['holdings'].items()):
            active_t['cash'] += shares * price_map.get(sym, 0)
        active_t['holdings'] = {}
        active_t['records'] = {}
        
        if ranking_df is not None:
             targets = []
             seen = set()
             for sym, row in ranking_df.iterrows():
                 if row['theme'] not in seen:
                     targets.append(sym)
                     seen.add(row['theme'])
                 if len(targets) >= TOP_N: break
             
             strong_cnt = (total_scores >= 150).sum() if total_scores is not None else 0
             exposure = 1.0 if strong_cnt >= 5 else 0.3
             
             invest_amt = active_t['cash'] * exposure
             if targets:
                 per_amt = invest_amt / len(targets)
                 for sym in targets:
                     price = price_map.get(sym, 0)
                     if price > 0:
                         shares = int(per_amt / price / 100) * 100
                         if shares > 0 and active_t['cash'] >= shares*price:
                             active_t['cash'] -= shares*price
                             active_t['holdings'][sym] = shares
                             active_t['records'][sym] = {'entry': price, 'high': price}

    def get_total_value(self, price_map):
        if self.mode == 'periodic':
            val = self.cash
            for sym, shares in self.periodic_holdings.items():
                val += shares * price_map.get(sym, 0)
            return val
        else:
            total = 0
            for t in self.tranches:
                t_val = t['cash']
                for sym, shares in t['holdings'].items():
                    t_val += shares * price_map.get(sym, 0)
                total += t_val
            return total

def run_timeframe_analysis():
    if not hasattr(BacktestEngine, 'global_prices'):
        BacktestEngine(14)
    
    full_dates = BacktestEngine.global_prices.index
    end_dt = pd.Timestamp(END_DATE_STR)
    
    # Define Timeframes
    timeframes = [1, 2, 3, 4, 5, 6] # Months
    
    print("\n" + "="*50)
    print(f"TIMEFRAME ANALYSIS (End Date: {END_DATE_STR[:10]}, T={FIXED_T}, Top {TOP_N})")
    print("="*50)
    print(f"{'Period':<10} | {'Start Date':<12} | {'Periodic Ret':<15} | {'Rolling Ret':<15} | {'Diff':<10}")
    print("-" * 70)
    
    for m in timeframes:
        start_dt = end_dt - pd.DateOffset(months=m)
        
        # Filter Dates
        mask = (full_dates >= start_dt) & (full_dates <= end_dt)
        sim_dates = full_dates[mask]
        
        if len(sim_dates) == 0: continue
        
        # Run Engines
        eng_p = BacktestEngine(FIXED_T, mode='periodic')
        eng_r = BacktestEngine(FIXED_T, mode='rolling')
        
        for d in sim_dates:
            eng_p.run_daily_logic(d)
            eng_r.run_daily_logic(d)
            
        ret_p = (eng_p.history_nav[-1]/eng_p.history_nav[0] - 1) * 100
        ret_r = (eng_r.history_nav[-1]/eng_r.history_nav[0] - 1) * 100
        
        print(f"{m} Month(s) | {start_dt.strftime('%Y-%m-%d')} | {ret_p:>13.2f}% | {ret_r:>13.2f}% | {ret_r-ret_p:>8.2f}%")

if __name__ == '__main__':
    run_timeframe_analysis()

# coding=utf-8
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from dotenv import load_dotenv
from config import config

load_dotenv()

# --- Configuration (V6.0 Theme Boost Aggressive) ---
TOP_N = 3
MIN_SCORE = 150
FRUIT_THEME_BOOST = True
STOP_LOSS = 0.20
TRAILING_TRIGGER = 0.10
TRAILING_DROP = 0.05
T_PERIOD = 14
END_DATE_STR = '2026-01-23'

class StrategyEngine:
    def __init__(self, t_period, mode='periodic'):
        self.T = t_period
        self.mode = mode  # 'periodic' or 'rolling'
        self.prices_df = None
        self.whitelist = set()
        self.theme_map = {}
        self.cash = 1000000.0
        self.history_nav = []
        self.dates = []
        
        # Internal State
        self.days_count = 0
        self.tranches = []  # For rolling
        self.periodic_holdings = {}
        self.periodic_pos_records = {}
        
        self.load_data()
        self.init_portfolio()

    def load_data(self):
        excel_path = os.path.join(config.BASE_DIR, "ETF合并筛选结果.xlsx")
        df_excel = pd.read_excel(excel_path)
        df_excel.columns = df_excel.columns.str.strip()
        rename_map = {'symbol': 'etf_code', 'sec_name': 'etf_name', 'name_cleaned': 'theme'}
        df_excel = df_excel.rename(columns=rename_map)
        if 'theme' not in df_excel.columns: df_excel['theme'] = df_excel['etf_name']
        self.whitelist = set(df_excel['etf_code'])
        self.theme_map = df_excel.set_index('etf_code')['theme'].to_dict()

        if not hasattr(StrategyEngine, 'global_prices'):
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
            StrategyEngine.global_prices = pd.DataFrame(price_data).sort_index().ffill()
        self.prices_df = StrategyEngine.global_prices

    def init_portfolio(self):
        if self.mode == 'rolling':
            share = self.cash / self.T
            for i in range(self.T):
                self.tranches.append({'id': i, 'cash': share, 'holdings': {}, 'records': {}, 'value': share})
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

    def run_daily(self, current_dt):
        if current_dt not in self.prices_df.index: return
        today_prices = self.prices_df.loc[current_dt]
        price_map = today_prices.to_dict()
        ranking_df, total_scores = None, None
        need_ranking = (self.mode == 'rolling') or ((self.days_count) % self.T == 0)
            
        if need_ranking:
            valid_scores, base_scores = self.get_ranking(current_dt)
            if valid_scores is not None and not valid_scores.empty:
                df = pd.DataFrame({'score': valid_scores, 'theme': [self.theme_map.get(c, 'Unknown') for c in valid_scores.index]})
                ranking_df = df.sort_values(by='score', ascending=False)
                total_scores = base_scores

        if self.mode == 'periodic':
            self.run_periodic(current_dt, price_map, ranking_df, total_scores)
        else:
            self.run_rolling(current_dt, price_map, ranking_df, total_scores)
        self.days_count += 1
        self.history_nav.append(self.get_total_value(price_map))
        self.dates.append(current_dt)

    def run_periodic(self, current_dt, price_map, ranking_df, total_scores):
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
            self.cash += self.periodic_holdings[sym] * price_map[sym]
            del self.periodic_holdings[sym]
            del self.periodic_pos_records[sym]

        if (self.days_count) % self.T == 0:
            for sym, shares in list(self.periodic_holdings.items()):
                self.cash += shares * price_map.get(sym, 0)
            self.periodic_holdings, self.periodic_pos_records = {}, {}
            if ranking_df is not None:
                targets, seen = [], set()
                for sym, row in ranking_df.iterrows():
                    if row['theme'] not in seen:
                        targets.append(sym); seen.add(row['theme'])
                    if len(targets) >= TOP_N: break
                exposure = 1.0 if (total_scores >= 150).sum() >= 5 else 0.3
                invest_amt = self.cash * exposure
                if targets:
                    per_amt = invest_amt / len(targets)
                    for sym in targets:
                        p = price_map.get(sym, 0)
                        if p > 0:
                            sh = int(per_amt / p / 100) * 100
                            if sh > 0 and self.cash >= sh*p:
                                self.cash -= sh*p; self.periodic_holdings[sym] = sh; self.periodic_pos_records[sym] = {'entry': p, 'high': p}

    def run_rolling(self, current_dt, price_map, ranking_df, total_scores):
        for t in self.tranches:
            to_sell = []
            for sym, shares in t['holdings'].items():
                p = price_map.get(sym, 0)
                if p <= 0: continue
                rec = t['records'][sym]
                rec['high'] = max(rec['high'], p)
                if p < rec['entry']*(1-STOP_LOSS) or (rec['high'] > rec['entry']*(1+TRAILING_TRIGGER) and p < rec['high']*(1-TRAILING_DROP)):
                   to_sell.append(sym)
            for sym in to_sell:
                t['cash'] += t['holdings'][sym] * price_map[sym]
                del t['holdings'][sym]; del t['records'][sym]

        idx = (self.days_count) % self.T
        active_t = self.tranches[idx]
        for sym, shares in list(active_t['holdings'].items()):
            active_t['cash'] += shares * price_map.get(sym, 0)
        active_t['holdings'], active_t['records'] = {}, {}
        
        if ranking_df is not None:
             targets, seen = [], set()
             for sym, row in ranking_df.iterrows():
                 if row['theme'] not in seen:
                     targets.append(sym); seen.add(row['theme'])
                 if len(targets) >= TOP_N: break
             exposure = 1.0 if (total_scores >= 150).sum() >= 5 else 0.3
             invest_amt = active_t['cash'] * exposure
             if targets:
                 per_amt = invest_amt / len(targets)
                 for sym in targets:
                     p = price_map.get(sym, 0)
                     if p > 0:
                         sh = int(per_amt / p / 100) * 100
                         if sh > 0 and active_t['cash'] >= sh*p:
                             active_t['cash'] -= sh*p; active_t['holdings'][sym] = sh; active_t['records'][sym] = {'entry': p, 'high': p}

    def get_total_value(self, price_map):
        if self.mode == 'periodic':
            return self.cash + sum(s * price_map.get(sym, 0) for sym, s in self.periodic_holdings.items())
        return sum(t['cash'] + sum(s * price_map.get(sym, 0) for sym, s in t['holdings'].items()) for t in self.tranches)

def run_multi_start_test():
    if not hasattr(StrategyEngine, 'global_prices'): StrategyEngine(14)
    df_p = StrategyEngine.global_prices
    
    start_base = pd.Timestamp('2024-09-01')
    end_limit = pd.Timestamp(END_DATE_STR)
    
    start_dates = []
    curr = start_base
    while curr < end_limit - timedelta(days=60):
        start_dates.append(curr)
        curr += timedelta(days=30)
    
    summary = []
    for start_dt in start_dates:
        mask = (df_p.index >= start_dt) & (df_p.index <= end_limit)
        sim_dates = df_p.index[mask]
        if len(sim_dates) < 20: continue
        
        eng_p = StrategyEngine(T_PERIOD, mode='periodic')
        eng_r = StrategyEngine(T_PERIOD, mode='rolling')
        
        for d in sim_dates:
            eng_p.run_daily(d)
            eng_r.run_daily(d)
        
        ret_p = (eng_p.history_nav[-1]/eng_p.history_nav[0] - 1)*100
        ret_r = (eng_r.history_nav[-1]/eng_r.history_nav[0] - 1)*100
        
        # Benchmark
        if 'SZSE.399006' not in df_p.columns:
            # Try to load it specifically if missing
            try:
                df_bm = pd.read_csv(os.path.join(config.DATA_CACHE_DIR, 'sz_399006.csv'), usecols=['日期', '收盘'])
                df_bm['日期'] = pd.to_datetime(df_bm['日期']).dt.tz_localize(None)
                bm_series = df_bm.set_index('日期')['收盘']
                bm = bm_series.reindex(sim_dates).ffill()
            except:
                bm = pd.Series(1.0, index=sim_dates) # Fallback
        else:
            bm = df_p['SZSE.399006'].reindex(sim_dates).ffill()
        
        ret_bm = (bm.iloc[-1]/bm.iloc[0] - 1)*100
        
        summary.append({
            'Start Date': start_dt.strftime('%Y-%m-%d'),
            'Periodic': ret_p,
            'Rolling': ret_r,
            'Benchmark': ret_bm
        })
        print(f"Processed {start_dt.strftime('%Y-%m-%d')}: P={ret_p:.1f}%, R={ret_r:.1f}%, BM={ret_bm:.1f}%")

    df_res = pd.DataFrame(summary)
    print("\n" + "="*60)
    print("Multi-Start Backtest Summary (End Date: 2026-01-23)")
    print("="*60)
    print(df_res.to_string(index=False))
    
    # Plotting summary
    plt.figure(figsize=(12, 6))
    plt.plot(df_res['Start Date'], df_res['Periodic'], 'o-', label='Periodic (Standard v6)')
    plt.plot(df_res['Start Date'], df_res['Rolling'], 's-', label='Rolling (Stateful)')
    plt.plot(df_res['Start Date'], df_res['Benchmark'], 'x--', color='gray', label='Benchmark (Chinext)')
    plt.xticks(rotation=45)
    plt.ylabel('Total Return (%)')
    plt.title('Performance Comparison by Entry Date')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('multi_start_comparison.png')
    print("\nSummary plot saved to multi_start_comparison.png")

if __name__ == '__main__':
    run_multi_start_test()

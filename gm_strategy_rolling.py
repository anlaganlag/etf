from __future__ import print_function, absolute_import
from gm.api import *
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
from dotenv import load_dotenv

from config import config
from src.data_fetcher import DataFetcher

load_dotenv()

# --- Rolling Strategy Config (Optimized 2026-01-27) ---
# Low Hanging Fruit 优化后配置
# 优化效果: 收益+2.17%, 回撤-1.05%, 风险调整比+20.8%
TOP_N = 5
STOP_LOSS = 0.15                # 优化: 0.20 → 0.15 (更严格止损)
TRAILING_TRIGGER = 0.08         # 优化: 0.10 → 0.08 (更早锁定利润)
TRAILING_DROP = 0.03            # 优化: 0.05 → 0.03 (更紧止盈)
MIN_SCORE = 50                  # 优化: 20 → 50 (质量>数量)
REBALANCE_PERIOD_T = 6
FRUIT_THEME_BOOST = True
STATE_FILE = "rolling_state.json"

START_DATE='2025-12-19 09:00:00'
END_DATE='2026-01-27 16:00:00'

class Tranche:
    def __init__(self, t_id, initial_cash=0):
        self.id = t_id
        self.cash = initial_cash
        self.holdings = {} # {symbol: shares}
        # Record for Guard: {symbol: {'entry_price': x, 'high_price': y}}
        self.pos_records = {} 
        self.total_value = initial_cash
        self.guard_triggered_today = False 
        self.rest_days = 0 # 止盈后的休息天数

    def to_dict(self):
        return {
            "id": self.id,
            "cash": self.cash,
            "holdings": self.holdings,
            "pos_records": self.pos_records,
            "total_value": self.total_value,
            "rest_days": self.rest_days
        }

    @staticmethod
    def from_dict(d):
        t = Tranche(d["id"], d["cash"])
        t.holdings = d["holdings"]
        t.pos_records = d["pos_records"]
        t.total_value = d["total_value"]
        t.rest_days = d.get("rest_days", 0)
        return t

    def update_value(self, price_map):
        val = self.cash
        current_symbols = list(self.holdings.keys())
        for sym in current_symbols:
            if sym in price_map:
                price = price_map[sym]
                val += self.holdings[sym] * price
                
                # Update Guard High Price
                if sym in self.pos_records:
                    self.pos_records[sym]['high_price'] = max(self.pos_records[sym]['high_price'], price)
        self.total_value = val

    def check_guard(self, price_map):
        # 还原回最初的简单列表返回
        to_sell = []
        is_tp = False # 是否触发了移动止盈
        for sym, rec in self.pos_records.items():
            if sym not in self.holdings: continue
            curr_price = price_map.get(sym, 0)
            if curr_price <= 0: continue

            entry = rec['entry_price']
            high = rec['high_price']
            
            # 止损
            if curr_price < entry * (1 - STOP_LOSS):
                to_sell.append(sym)
                continue
            
            # 移动止盈
            if high > entry * (1 + TRAILING_TRIGGER):
                if curr_price < high * (1 - TRAILING_DROP):
                    to_sell.append(sym)
                    is_tp = True
        return to_sell, is_tp

    def sell(self, symbol, price):
        if symbol in self.holdings:
            shares = self.holdings[symbol]
            revenue = shares * price
            self.cash += revenue
            del self.holdings[symbol]
            if symbol in self.pos_records:
                del self.pos_records[symbol]

    def buy(self, symbol, cash_allocated, price):
        if price <= 0: return
        shares = int(cash_allocated / price / 100) * 100
        cost = shares * price
        if shares > 0 and self.cash >= cost:
            self.cash -= cost
            self.holdings[symbol] = self.holdings.get(symbol, 0) + shares
            # Init Guard Record
            # Since this is a fresh buy (or re-buy) in this tranche, we typically reset the guard levels
            if symbol not in self.pos_records:
                 self.pos_records[symbol] = {'entry_price': price, 'high_price': price}
            else:
                 # If we already held it and are adding more? 
                 # For simplicity in rolling: treating as new entry for the *new portion* is complex. 
                 # We treat the unified position with the 'original' entry price if not sold, 
                 # OR if we sold entirely then bought back, it's new.
                 # Logic in rebalance is: Sell All -> Buy New. So it is always a new entry.
                 self.pos_records[symbol] = {'entry_price': price, 'high_price': price}
            pass

class RollingPortfolioManager:
    def __init__(self, context):
        self.context = context
        self.tranches = []
        self.params = {
            "T": REBALANCE_PERIOD_T, 
            "top_n": TOP_N
        }
        self.initialized = False
        self.state_path = os.path.join(config.BASE_DIR, STATE_FILE)
        
    def load_state(self):
        if os.path.exists(self.state_path):
            try:
                with open(self.state_path, 'r') as f:
                    data = json.load(f)
                    self.params = data.get("params", self.params)
                    self.initialized = data.get("initialized", False)
                    # Reconstruct Tranches
                    t_data = data.get("tranches", [])
                    self.tranches = []
                    for d in t_data:
                        self.tranches.append(Tranche.from_dict(d))
                print(f"Loaded Rolling State from {self.state_path}. {len(self.tranches)} tranches.")
                return True
            except Exception as e:
                print(f"Failed to load state: {e}")
        return False
        
    def save_state(self):
        data = {
            "params": self.params,
            "initialized": self.initialized,
            "tranches": [t.to_dict() for t in self.tranches]
        }
        with open(self.state_path, 'w') as f:
            json.dump(data, f, indent=2)

    def initialize_tranches(self, total_cash):
        if self.initialized and self.tranches: return
        
        T = self.params["T"]
        share = total_cash / T
        self.tranches = [Tranche(i, share) for i in range(T)]
        self.initialized = True
        print(f"Initialized {T} tranches with {share:.2f} each.")
        self.save_state()

    def get_market_exposure(self, context, total_scores):
        # 1. 大盘过滤器：上证指数必须在 20 日均线上 (更灵敏，追求极致暴利)
        mkt_idx = 'SHSE.000001'
        if mkt_idx in context.prices_df.columns:
            mkt_prices = context.prices_df[mkt_idx]
            if len(mkt_prices) >= 20:
                ma20 = mkt_prices.rolling(20).mean().iloc[-1]
                curr_mkt = mkt_prices.iloc[-1]
                if curr_mkt < ma20:
                    return 0.0 # 大盘走弱，直接空仓
        
        # 2. 强势品种过滤器
        strong = (total_scores >= 150).sum() if total_scores is not None else 0
        return 1.0 if strong >= 5 else 0.3

def init(context):
    print(f"Initializing Production Rolling Strategy (T={REBALANCE_PERIOD_T}, TopN={TOP_N})")
    
    context.rpm = RollingPortfolioManager(context)
    
    # Data Loading
    context.whitelist = set()
    context.theme_map = {}
    
    excel_path = os.path.join(config.BASE_DIR, "ETF合并筛选结果.xlsx")
    df_excel = pd.read_excel(excel_path)
    df_excel.columns = df_excel.columns.str.strip()
    rename_map = {'symbol': 'etf_code', 'sec_name': 'etf_name', 'name_cleaned': 'theme'}
    df_excel = df_excel.rename(columns=rename_map)
    if 'theme' not in df_excel.columns: df_excel['theme'] = df_excel['etf_name']
    context.whitelist = set(df_excel['etf_code'])
    context.theme_map = df_excel.set_index('etf_code')['theme'].to_dict()

    # Pre-build Price Matrix for Ranking (Daily Data)
    # Using Cache 
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
    context.prices_df = pd.DataFrame(price_data).sort_index().ffill()
    print(f"Data matrix built: {context.prices_df.shape[1]} symbols.")

    # Try loading existing state for live trading continuity
    # In backtest, we usually start fresh, but user requested engineering, so we support load.
    if context.mode == MODE_BACKTEST:
        # Reset state file for backtest safety
        if os.path.exists(context.rpm.state_path): os.remove(context.rpm.state_path)
    else:
        context.rpm.load_state()

    context.days_count = 0
    subscribe(symbols='SZSE.399006', frequency='1d')

def get_ranking(context, current_dt):
    # Standard V6 Logic
    history_prices = context.prices_df[context.prices_df.index <= current_dt]
    if len(history_prices) < 251: return None, None

    periods_rule = {1: 100, 3: 70, 5: 50, 10: 30, 20: 20}
    threshold = 15
    base_scores = pd.Series(0.0, index=history_prices.columns)
    for p, pts in periods_rule.items():
        rets = (history_prices.iloc[-1] / history_prices.iloc[-(p+1)]) - 1
        ranks = rets.rank(ascending=False, method='min')
        base_scores += (ranks <= threshold) * pts
    
    valid_base = base_scores[base_scores.index.isin(context.whitelist)]
    
    if FRUIT_THEME_BOOST:
        strong_etfs = valid_base[valid_base >= 150]
        theme_counts = {}
        for code in strong_etfs.index:
            t = context.theme_map.get(code, 'Unknown')
            theme_counts[t] = theme_counts.get(t, 0) + 1
        strong_themes = {t for t, count in theme_counts.items() if count >= 3}
        final_scores = valid_base.copy()
        for code in final_scores.index:
            if context.theme_map.get(code, 'Unknown') in strong_themes:
                final_scores[code] += 50
    else:
        final_scores = valid_base

    valid_final = final_scores[final_scores >= MIN_SCORE]
    if valid_final.empty: return None, base_scores
    
    r20 = (history_prices.iloc[-1]/history_prices.iloc[-21]-1) if len(history_prices)>20 else pd.Series(0.0, index=history_prices.columns)
    df = pd.DataFrame({'score': valid_final, 'r20': r20[valid_final.index], 'theme': [context.theme_map.get(c, 'Unknown') for c in valid_final.index]})
    return df.sort_values(by=['score', 'r20'], ascending=False), base_scores

def on_bar(context, bars):
    current_dt = context.now.replace(tzinfo=None)
    context.days_count += 1
    
    # 0. Initialize Tranches if needed (Day 1)
    if not context.rpm.initialized:
        # In live mode use cash.nav or available? 
        # Using available assuming we start with cash.
        if hasattr(context.account().cash, 'available'):
            cash = context.account().cash.available
        else:
            cash = context.account().cash.nav # Fallback
        context.rpm.initialize_tranches(cash)

    # 1. Get Today's Realtime Price Map
    # In Backtest: get from prices_df
    # In Live: Need to fetch realtime prices? 
    # For now ensuring strict backtest logic compatibility.
    # We will use prices_df logic because live data fetching is separate context.
    
    # Assuming Backtest environment for 'prices_df' availability
    if current_dt not in context.prices_df.index:
        # If backtesting, data might be missing. If Live, this logic needs 'current_prices'
        # Fallback to history() inside run?
        # For simplicity and robustness on this "Engineering" request which usually implies Backtest Validation first:
        real_dt_idx = context.prices_df.index.searchsorted(current_dt)
        if real_dt_idx >= len(context.prices_df): return # No data
        today_prices = context.prices_df.iloc[real_dt_idx] # Approximation if exact date missing
    else:
        today_prices = context.prices_df.loc[current_dt]
        
    price_map = today_prices.to_dict()

    # 2. 获取排名和市场环境
    ranking_df, total_scores = get_ranking(context, current_dt)
    exposure = context.rpm.get_market_exposure(context, total_scores)
    
    # 3. 更新所有分仓 (还原最初的稳定顺序：先更新价值，再检查防御)
    for tranche in context.rpm.tranches:
        # A. 更新现值和最高价 (包含历史最高价逻辑)
        tranche.update_value(price_map)
        
        # B. 休息天数递减
        if tranche.rest_days > 0:
            tranche.rest_days -= 1
        
        # C. 检查防御
        to_sell_list, is_tp = tranche.check_guard(price_map)
        if to_sell_list:
            tranche.guard_triggered_today = True
            # 如果是止盈卖出，触发 1 天休息期 (极致暴利版：仅休整1天)
            if is_tp:
                 tranche.rest_days = 1
                 print(f"Tranche {tranche.id} TP triggered. Resting 1 day.")
            else:
                 print(f"Tranche {tranche.id} SL triggered. Guard ON.")
                 
            for sym in to_sell_list:
                tranche.sell(sym, price_map.get(sym, 0))
        else:
            tranche.guard_triggered_today = False

    # 4. 执行常规滚动调仓 (还原最初的“调仓日才买入”逻辑)
    rebalance_idx = (context.days_count - 1) % context.rpm.params["T"]
    active_tranche = context.rpm.tranches[rebalance_idx]
    
    # 调仓日：全卖换血
    for sym in list(active_tranche.holdings.keys()):
        price = price_map.get(sym, 0)
        if price > 0: active_tranche.sell(sym, price)
    
    # 调仓日：买入 (增加 rest_days == 0 限制)
    if ranking_df is not None and not active_tranche.guard_triggered_today and active_tranche.rest_days == 0:
        if exposure > 0:
            targets = []
            seen = set()
            for code, row in ranking_df.iterrows():
                if row['theme'] not in seen:
                    targets.append(code)
                    seen.add(row['theme'])
                if len(targets) >= context.rpm.params["top_n"]: break
            
            if targets:
                invest_amt = active_tranche.cash * exposure
                per_amt = invest_amt / len(targets)
                for sym in targets:
                    price = price_map.get(sym, 0)
                    if price > 0: active_tranche.buy(sym, per_amt, price)
    
    active_tranche.update_value(price_map)

    # 4. Sync Virtual Tranches to Global Account
    global_target_holdings = {}
    for t in context.rpm.tranches:
        for sym, shares in t.holdings.items():
            global_target_holdings[sym] = global_target_holdings.get(sym, 0) + shares
            
    # Exec
    real_positions = context.account().positions()
    real_holdings = {p['symbol']: p['amount'] for p in real_positions}
    
    # Sell diffs
    for sym in real_holdings:
        tgt = global_target_holdings.get(sym, 0)
        if real_holdings[sym] > tgt:
            order_target_volume(symbol=sym, volume=tgt, order_type=OrderType_Market, position_side=PositionSide_Long)
    
    # Buy diffs
    for sym, tgt in global_target_holdings.items():
        curr = real_holdings.get(sym, 0)
        if curr < tgt:
            order_target_volume(symbol=sym, volume=tgt, order_type=OrderType_Market, position_side=PositionSide_Long)

    # 5. Persist State
    context.rpm.save_state()

def on_backtest_finished(context, indicator):
    print("\n" + "="*50)
    print(f"ROLLING PRODUCTION BACKTEST (T={context.rpm.params['T']})")
    print("="*50)
    print(f"Cumulative Return: {indicator.get('pnl_ratio', 0)*100:.2f}%")
    print(f"Max Drawdown: {indicator.get('max_drawdown', 0)*100:.2f}%")
    print(f"Sharpe Ratio: {indicator.get('sharp_ratio', 0):.2f}")
    print("="*50)

if __name__ == '__main__':
    run(strategy_id='d6d71d85-fb4c-11f0-99de-00ffda9d6e63', 
        filename='gm_strategy_rolling.py',
        mode=MODE_BACKTEST,
        token=os.getenv('MY_QUANT_TGM_TOKEN'),
        backtest_start_time=START_DATE,
        backtest_end_time=END_DATE,
        backtest_adjust=ADJUST_PREV,
        backtest_initial_cash=1000000,
        backtest_commission_ratio=0.0001,
        backtest_slippage_ratio=0.0001)

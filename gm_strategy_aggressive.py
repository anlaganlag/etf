from __future__ import print_function, absolute_import
from gm.api import *
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
from dotenv import load_dotenv
from config import config

load_dotenv()

# --- Aggressive Config: T=8 + Extreme Trend ---
TOP_N = 5
STOP_LOSS = 0.18  # 放宽止损
TRAILING_TRIGGER = 0.08
TRAILING_DROP = 0.05
MIN_SCORE = 20
REBALANCE_PERIOD_T = 8  # 更快换手
STATE_FILE = "rolling_state_aggressive.json"

START_DATE='2021-12-03 09:00:00'
END_DATE='2026-01-23 16:00:00'

class Tranche:
    def __init__(self, t_id, initial_cash=0):
        self.id = t_id
        self.cash = initial_cash
        self.holdings = {}
        self.pos_records = {}
        self.total_value = initial_cash
        self.guard_triggered_today = False

    def to_dict(self):
        return self.__dict__

    @staticmethod
    def from_dict(d):
        t = Tranche(d["id"], d["cash"])
        t.holdings = d["holdings"]
        t.pos_records = d["pos_records"]
        t.total_value = d["total_value"]
        return t

    def update_value(self, price_map):
        val = self.cash
        current_symbols = list(self.holdings.keys())
        for sym in current_symbols:
            if sym in price_map:
                price = price_map[sym]
                val += self.holdings[sym] * price
                if sym in self.pos_records:
                    self.pos_records[sym]['high_price'] = max(self.pos_records[sym]['high_price'], price)
        self.total_value = val

    def check_guard(self, price_map):
        to_sell = []
        is_tp = False
        for sym, rec in self.pos_records.items():
            if sym not in self.holdings: continue
            curr_price = price_map.get(sym, 0)
            if curr_price <= 0: continue

            entry, high = rec['entry_price'], rec['high_price']

            if (curr_price < entry * (1 - STOP_LOSS)) or \
               (high > entry * (1 + TRAILING_TRIGGER) and curr_price < high * (1 - TRAILING_DROP)):
                to_sell.append(sym)
                if curr_price >= entry: is_tp = True

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

class RollingPortfolioManager:
    def __init__(self):
        self.tranches = []
        self.params = {"T": REBALANCE_PERIOD_T, "top_n": TOP_N}
        self.initialized = False
        self.state_path = os.path.join(config.BASE_DIR, STATE_FILE)

    def load_state(self):
        if os.path.exists(self.state_path):
            try:
                with open(self.state_path, 'r') as f:
                    data = json.load(f)
                    self.params = data.get("params", self.params)
                    self.initialized = data.get("initialized", False)
                    self.tranches = [Tranche.from_dict(d) for d in data.get("tranches", [])]
                return True
            except: pass
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
        share = total_cash / self.params["T"]
        self.tranches = [Tranche(i, share) for i in range(self.params["T"])]
        self.initialized = True
        self.save_state()

def init(context):
    context.rpm = RollingPortfolioManager()

    excel_path = os.path.join(config.BASE_DIR, "ETF合并筛选结果.xlsx")
    df_excel = pd.read_excel(excel_path)
    df_excel.columns = df_excel.columns.str.strip()
    rename_map = {'symbol': 'etf_code', 'sec_name': 'etf_name', 'name_cleaned': 'theme'}
    df_excel = df_excel.rename(columns=rename_map)
    if 'theme' not in df_excel.columns: df_excel['theme'] = df_excel['etf_name']
    context.whitelist = set(df_excel['etf_code'])
    context.theme_map = df_excel.set_index('etf_code')['theme'].to_dict()

    price_data = {}
    files = [f for f in os.listdir(config.DATA_CACHE_DIR) if f.endswith('.csv') and (f.startswith('sh') or f.startswith('sz'))]
    for f in files:
        code = f.replace('_', '.').replace('.csv', '')
        if '.' not in code:
            code = ('SHSE.' if code.startswith('sh') else 'SZSE.') + code[2:]
        try:
            df = pd.read_csv(os.path.join(config.DATA_CACHE_DIR, f), usecols=['日期', '收盘'])
            df['日期'] = pd.to_datetime(df['日期']).dt.tz_localize(None)
            price_data[code] = df.set_index('日期')['收盘']
        except: pass
    context.prices_df = pd.DataFrame(price_data).sort_index().ffill()

    if context.mode == MODE_BACKTEST and os.path.exists(context.rpm.state_path):
        os.remove(context.rpm.state_path)
    else:
        context.rpm.load_state()

    context.days_count = 0
    subscribe(symbols='SZSE.399006', frequency='1d')

def get_ranking(context, current_dt):
    history = context.prices_df[context.prices_df.index <= current_dt]
    if len(history) < 251: return None, None

    base_scores = pd.Series(0.0, index=history.columns)
    periods_rule = {1: 10, 3: 20, 5: 30, 10: 60, 20: 150}  # 极端趋势权重

    rets_dict = {}
    last_row = history.iloc[-1]

    for p, pts in periods_rule.items():
        rets = (last_row / history.iloc[-(p+1)]) - 1
        rets_dict[f'r{p}'] = rets
        ranks = rets.rank(ascending=False, method='min')
        base_scores += (ranks <= 15) * pts

    valid_scores = base_scores[base_scores.index.isin(context.whitelist)]
    valid_scores = valid_scores[valid_scores >= MIN_SCORE]

    if valid_scores.empty: return None, base_scores

    data_to_df = {
        'score': valid_scores,
        'theme': [context.theme_map.get(c, 'Unknown') for c in valid_scores.index],
        'etf_code': valid_scores.index
    }

    for p in periods_rule.keys():
        data_to_df[f'r{p}'] = rets_dict[f'r{p}'][valid_scores.index]

    df = pd.DataFrame(data_to_df)
    sort_cols = ['score', 'r1', 'r3', 'r5', 'r10', 'r20', 'etf_code']
    asc_order = [False, False, False, False, False, False, True]

    return df.sort_values(by=sort_cols, ascending=asc_order), base_scores

def on_bar(context, bars):
    current_dt = context.now.replace(tzinfo=None)
    context.days_count += 1

    if not context.rpm.initialized:
        cash = context.account().cash.available if hasattr(context.account().cash, 'available') else context.account().cash.nav
        context.rpm.initialize_tranches(cash)

    if current_dt not in context.prices_df.index:
        idx = context.prices_df.index.searchsorted(current_dt)
        if idx >= len(context.prices_df): return
        today_prices = context.prices_df.iloc[idx]
    else:
        today_prices = context.prices_df.loc[current_dt]
    price_map = today_prices.to_dict()

    ranking_df, _ = get_ranking(context, current_dt)

    for t in context.rpm.tranches:
        t.update_value(price_map)
        to_sell, _ = t.check_guard(price_map)
        if to_sell:
            t.guard_triggered_today = True
            for sym in to_sell: t.sell(sym, price_map.get(sym, 0))
        else:
            t.guard_triggered_today = False

    active_tranche = context.rpm.tranches[(context.days_count - 1) % REBALANCE_PERIOD_T]

    for sym in list(active_tranche.holdings.keys()):
        price = price_map.get(sym, 0)
        if price > 0: active_tranche.sell(sym, price)

    if ranking_df is not None and not active_tranche.guard_triggered_today:
        targets = ranking_df.head(TOP_N).index.tolist()

        if targets:
            per_amt = active_tranche.cash / len(targets)
            for sym in targets:
                active_tranche.buy(sym, per_amt, price_map.get(sym, 0))

    active_tranche.update_value(price_map)

    global_tgt = {}
    for t in context.rpm.tranches:
        for sym, shares in t.holdings.items():
            global_tgt[sym] = global_tgt.get(sym, 0) + shares

    real = {p['symbol']: p['amount'] for p in context.account().positions()}

    for sym in real:
        if real[sym] > global_tgt.get(sym, 0):
            order_target_volume(symbol=sym, volume=global_tgt.get(sym, 0), order_type=OrderType_Market, position_side=PositionSide_Long)

    for sym, tgt in global_tgt.items():
        if real.get(sym, 0) < tgt:
            order_target_volume(symbol=sym, volume=tgt, order_type=OrderType_Market, position_side=PositionSide_Long)

    context.rpm.save_state()

def on_backtest_finished(context, indicator):
    print(f"\n=== AGGRESSIVE (T={REBALANCE_PERIOD_T}) RESULTS ===")
    print(f"Return: {indicator.get('pnl_ratio', 0)*100:.2f}%")
    print(f"Max DD: {indicator.get('max_drawdown', 0)*100:.2f}%")
    print(f"Sharpe: {indicator.get('sharp_ratio', 0):.2f}\n")

if __name__ == '__main__':
    run(strategy_id='aggressive_rolling', filename='gm_strategy_aggressive.py', mode=MODE_BACKTEST,
        token=os.getenv('MY_QUANT_TGM_TOKEN'), backtest_start_time=START_DATE, backtest_end_time=END_DATE,
        backtest_adjust=ADJUST_PREV, backtest_initial_cash=1000000)

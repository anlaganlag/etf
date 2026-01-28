from __future__ import print_function, absolute_import
from gm.api import *
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
from dotenv import load_dotenv

from config import config
from src.utils import load_price_matrix, load_etf_config
from src.theme_booster import ThemeBooster

load_dotenv()

# --- 核心配置 ---
TOP_N = 3
STOP_LOSS = 0.15                
TRAILING_TRIGGER = 0.08         
TRAILING_DROP = 0.03            
MIN_SCORE = 50                  
REBALANCE_PERIOD_T = 6
CONCEPT_THEME_BOOST = True  # 开启大模型/实时热点增强
CONCEPT_BOOST_POINTS = 40
STATE_FILE = "rolling_state.json"

START='2021-12-01 09:00:00'
END='2026-01-27 16:00:00'

# --- 核心类: 分仓管理 (保留以维持状态稳健性) ---
class Tranche:
    def __init__(self, t_id, initial_cash=0):
        self.id = t_id
        self.cash = initial_cash
        self.holdings = {} # {symbol: shares}
        self.pos_records = {} # {symbol: {'entry': x, 'high': y}} 用于风控
        self.rest_days = 0 

    def to_dict(self):
        return self.__dict__

    @staticmethod
    def from_dict(d):
        t = Tranche(d["id"], d["cash"])
        t.holdings = d["holdings"]
        t.pos_records = d["pos_records"]
        t.rest_days = d.get("rest_days", 0)
        return t

    def check_guard_and_sell(self, price_map):
        """检查止损/止盈并执行卖出"""
        to_sell = []
        is_tp = False
        
        for sym, rec in list(self.pos_records.items()):
            price = price_map.get(sym, 0)
            if price <= 0 or sym not in self.holdings: continue
            
            # 更新最高价
            rec['high'] = max(rec['high'], price)
            
            # 止损判断
            if price < rec['entry'] * (1 - STOP_LOSS):
                to_sell.append(sym)
                print(f"Tranche {self.id}: SL {sym} @ {price:.3f}")
                continue
            
            # 移动止盈判断
            if rec['high'] > rec['entry'] * (1 + TRAILING_TRIGGER):
                if price < rec['high'] * (1 - TRAILING_DROP):
                    to_sell.append(sym)
                    is_tp = True
                    print(f"Tranche {self.id}: TP {sym} @ {price:.3f}")

        # 执行卖出
        if to_sell:
            self.rest_days = 1 if is_tp else 0 # 止盈休息1天，止损不休
            for sym in to_sell:
                self.sell(sym, price_map.get(sym, 0))
        elif self.rest_days > 0:
            self.rest_days -= 1

    def sell(self, symbol, price):
        if symbol in self.holdings:
            self.cash += self.holdings[symbol] * price
            del self.holdings[symbol]
            if symbol in self.pos_records: del self.pos_records[symbol]

    def buy(self, symbol, amount, price):
        if price <= 0: return
        shares = int(amount / price / 100) * 100
        if shares > 0 and self.cash >= shares * price:
            self.cash -= shares * price
            self.holdings[symbol] = self.holdings.get(symbol, 0) + shares
            # 简化：每次买入都视为新开仓或更新成本
            self.pos_records[symbol] = {'entry': price, 'high': price}

    @property
    def value(self, price_map=None):
        val = self.cash
        if price_map:
            for sym, shares in self.holdings.items():
                val += shares * price_map.get(sym, 0)
        return val

# --- 策略函数 ---

def init(context):
    print(f"init Rolling Strategy (T={REBALANCE_PERIOD_T}, TopN={TOP_N})")
    
    # 1. 加载配置和数据 (使用Utils简化)
    df_config = load_etf_config()
    context.whitelist = set(df_config['etf_code'])
    context.theme_map = df_config.set_index('etf_code')['theme'].to_dict()
    context.prices_df = load_price_matrix(context.whitelist)
    
    # 2. 初始化 ThemeBooster
    context.theme_booster = None
    if CONCEPT_THEME_BOOST:
        try:
            context.theme_booster = ThemeBooster(list(set(context.theme_map.values())), top_n_concepts=15, boost_points=CONCEPT_BOOST_POINTS)
        except: pass

    # 3. 状态管理
    context.tranches = []
    context.initialized = False
    context.days_count = 0
    
    # 实盘加载状态
    state_path = os.path.join(config.BASE_DIR, STATE_FILE)
    if context.mode != MODE_BACKTEST and os.path.exists(state_path):
        try:
            with open(state_path, 'r') as f:
                data = json.load(f)
                context.tranches = [Tranche.from_dict(d) for d in data.get("tranches", [])]
                context.initialized = True
                print(f"Loaded {len(context.tranches)} tranches.")
        except: pass
        
    subscribe(symbols='SZSE.399006', frequency='1d')

def save_state(context):
    state_path = os.path.join(config.BASE_DIR, STATE_FILE)
    data = {"tranches": [t.to_dict() for t in context.tranches]}
    with open(state_path, 'w') as f:
        json.dump(data, f)

def get_ranking(context, current_dt):
    # 获取历史数据切片
    hist = context.prices_df[context.prices_df.index <= current_dt]
    if len(hist) < 251: return None
    
    # 1. 计算动量得分 (Vectorized)
    curr = hist.iloc[-1]
    scores = pd.Series(0.0, index=curr.index)
    
    for p, weight in {1: 100, 3: 70, 5: 50, 10: 30, 20: 20}.items():
        ret = (curr / hist.iloc[-(p+1)]) - 1
        scores += (ret.rank(ascending=False) <= 15) * weight
        
    scores = scores[scores.index.isin(context.whitelist)]
    
    # 2. 主题增强 (ThemeBooster)
    if CONCEPT_THEME_BOOST and context.theme_booster:
        # 简单缓存机制：每天只刷一次
        if not hasattr(context, '_last_boost_dt') or context._last_boost_dt != current_dt.date():
            # 仅在回测的第一天或者实盘模式下才打印，避免刷屏
            should_print = (context.mode != MODE_BACKTEST) or (not hasattr(context, '_has_printed_themes'))
            
            context._hot_themes = context.theme_booster.get_hot_themes(verbose=should_print)
            context._last_boost_dt = current_dt.date()
            context._has_printed_themes = True
            
        for code in scores.index:
            if context.theme_map.get(code) in context._hot_themes:
                 scores[code] += CONCEPT_BOOST_POINTS

    # 3. 过滤
    df = scores[scores >= MIN_SCORE].sort_values(ascending=False)
    
    # 4. 辅助指标 (R20) 用于同分排序
    r20 = (curr / hist.iloc[-21]) - 1
    return pd.DataFrame({'score': df, 'r20': r20[df.index]}).sort_values(['score', 'r20'], ascending=False)

def on_bar(context, bars):
    current_dt = context.now.replace(tzinfo=None)
    context.days_count += 1
    
    # 0. 初始化分仓 (如果是第一天)
    if not context.initialized:
        cash = context.account().cash.available
        share = cash / REBALANCE_PERIOD_T
        context.tranches = [Tranche(i, share) for i in range(REBALANCE_PERIOD_T)]
        context.initialized = True
        
    prices = context.prices_df.loc[current_dt] if current_dt in context.prices_df.index else context.prices_df.iloc[-1]
    price_map = prices.to_dict()
    
    # 1. 市场风控 (MA20)
    mkt = context.prices_df.get('SHSE.000001')
    market_ok = True
    if mkt is not None and len(mkt[:current_dt]) > 20:
        market_ok = mkt[:current_dt].iloc[-1] > mkt[:current_dt].rolling(20).mean().iloc[-1]
    
    # 2. 处理每个分仓
    ranking_df = get_ranking(context, current_dt)
    rebalance_idx = (context.days_count - 1) % REBALANCE_PERIOD_T
    
    global_target = {} # {sym: shares}
    
    for tranche in context.tranches:
        # A. 风控检查 (无论是否调仓日都要检查)
        tranche.check_guard_and_sell(price_map)
        
        # B. 调仓逻辑 (仅在对应的调仓日)
        if tranche.id == rebalance_idx:
            # 卖出旧持仓
            for sym in list(tranche.holdings.keys()):
                tranche.sell(sym, price_map.get(sym, 0))
            
            # 买入新持仓 (如果市场好 & 不在休息期)
            if market_ok and tranche.rest_days == 0 and ranking_df is not None and not ranking_df.empty:
                targets = []
                seen_themes = set()
                # 筛选TopN (同一个主题只买一个)
                for code in ranking_df.index:
                    theme = context.theme_map.get(code)
                    if theme not in seen_themes:
                        targets.append(code)
                        seen_themes.add(theme)
                    if len(targets) >= TOP_N: break
                
                if targets:
                    amt_per_etf = tranche.cash / len(targets)
                    for sym in targets:
                        tranche.buy(sym, amt_per_etf, price_map.get(sym, 0))

        # C. 汇总目标持仓 (用于Sync)
        for sym, shares in tranche.holdings.items():
            global_target[sym] = global_target.get(sym, 0) + shares

    # 3. 执行交易 Sync (这是最稳健的交易方式)
    real_pos = {p['symbol']: p['amount'] for p in context.account().positions()}
    
    # 先卖
    for sym, curr_vol in real_pos.items():
        tgt_vol = global_target.get(sym, 0)
        if curr_vol > tgt_vol:
            order_target_volume(symbol=sym, volume=tgt_vol, order_type=OrderType_Market, position_side=PositionSide_Long)
    
    # 后买
    for sym, tgt_vol in global_target.items():
        if real_pos.get(sym, 0) < tgt_vol:
            order_target_volume(symbol=sym, volume=tgt_vol, order_type=OrderType_Market, position_side=PositionSide_Long)
            
    save_state(context)

def on_backtest_finished(context, indicator):
    print(f"Backtest Finished. Return: {indicator['pnl_ratio']*100:.2f}%, Sharpe: {indicator['sharp_ratio']:.2f}")

if __name__ == '__main__':
    run(strategy_id='d6d71d85-fb4c-11f0-99de-00ffda9d6e63', 
        filename='gm_strategy_rolling.py',
        mode=MODE_BACKTEST,
        token=os.getenv('MY_QUANT_TGM_TOKEN'),
        backtest_start_time=START,
        backtest_end_time=END,
        backtest_adjust=ADJUST_PREV,
        backtest_initial_cash=1000000,
        backtest_commission_ratio=0.0001,
        backtest_slippage_ratio=0.0001)

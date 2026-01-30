# coding=utf-8
from __future__ import print_function, absolute_import
from gm.api import *
import pandas as pd
import numpy as np
import os
import json
import time
from datetime import datetime
from dotenv import load_dotenv
from config import config

load_dotenv()

# ================= 策略配置 (保持与回测一致) =================
TOP_N = 8
REBALANCE_PERIOD_T = 12
STOP_LOSS = 0.03
TRAILING_TRIGGER = 0.08
TRAILING_DROP = 0.03
MIN_SCORE = 20
MAX_PER_THEME = 2
DYNAMIC_POSITION = True
SCORING_METHOD = 'SMOOTH'
STATE_FILE = "rolling_state_simple.json"
# ==========================================================

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
        for sym, shares in self.holdings.items():
            if sym in price_map:
                price = price_map[sym]
                val += shares * price
                if sym in self.pos_records:
                    self.pos_records[sym]['high_price'] = max(self.pos_records[sym].get('high_price', price), price)
        self.total_value = val

    def check_guard(self, price_map):
        to_sell = []
        for sym, rec in self.pos_records.items():
            if sym not in self.holdings: continue
            curr_price = price_map.get(sym, 0)
            if curr_price <= 0: continue
            entry, high = rec['entry_price'], rec.get('high_price', curr_price)
            if (curr_price < entry * (1 - STOP_LOSS)) or \
               (high > entry * (1 + TRAILING_TRIGGER) and curr_price < high * (1 - TRAILING_DROP)):
                to_sell.append(sym)
        return to_sell

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
        self.days_count = 0
        self.last_run_date = ""
        self.initialized = False
        self.state_path = os.path.join(config.BASE_DIR, STATE_FILE)
        
    def load_state(self):
        if os.path.exists(self.state_path):
            with open(self.state_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.days_count = data.get("days_count", 0)
                self.last_run_date = data.get("last_run_date", "")
                self.initialized = data.get("initialized", False)
                self.tranches = [Tranche.from_dict(d) for d in data.get("tranches", [])]
            return True
        return False
        
    def save_state(self):
        data = {
            "days_count": self.days_count,
            "last_run_date": self.last_run_date,
            "initialized": self.initialized,
            "tranches": [t.to_dict() for t in self.tranches]
        }
        with open(self.state_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

def get_market_regime(context, current_dt):
    """判断市场环境：返回仓位系数 0.5-1.0
    Turbo版：仅使用微观ETF市场广度，不使用宏观年线（避免牛市踏空）
    """
    history = context.prices_df
    if len(history) < 60: return 1.0
    
    # === 微观强度: ETF市场广度 ===
    recent = history.tail(60)
    ma20 = recent.tail(20).mean()
    ma60 = recent.mean()
    current = recent.iloc[-1]
    above_ma20 = (current > ma20).sum() / len(current)
    above_ma60 = (current > ma60).sum() / len(current)
    strength = (above_ma20 + above_ma60) / 2

    # Turbo 激进版：强势满仓，中性90%，弱势50%
    if strength > 0.6: return 1.0
    elif strength > 0.4: return 0.9
    else: return 0.5

def get_ranking(context, current_dt):
    # 实盘模式下，prices_df 已经包含今天的 14:55 价格
    history = context.prices_df
    if len(history) < 251: return None
    base_scores = pd.Series(0.0, index=history.columns)
    periods_rule = {1: 100, 3: 70, 5: 50, 10: 30, 20: 20}
    rets_dict = {}
    last_row = history.iloc[-1]
    for p, pts in periods_rule.items():
        rets = (last_row / history.iloc[-(p+1)]) - 1
        rets_dict[f'r{p}'] = rets
        ranks = rets.rank(ascending=False, method='min')
        if SCORING_METHOD == 'SMOOTH':
            decay = ((30 - ranks) / 30).clip(lower=0)
            base_scores += decay * pts
        else:
            base_scores += (ranks <= 15) * pts
    valid_scores = base_scores[base_scores.index.isin(context.whitelist) & (base_scores >= MIN_SCORE)]
    if valid_scores.empty: return None
    data_to_df = {'score': valid_scores, 'theme': [context.theme_map.get(c, 'Unknown') for c in valid_scores.index], 'etf_code': valid_scores.index}
    for p in periods_rule.keys(): data_to_df[f'r{p}'] = rets_dict[f'r{p}'][valid_scores.index]
    df = pd.DataFrame(data_to_df)
    sort_cols = ['score', 'r1', 'r3', 'r5', 'r10', 'r20', 'etf_code']
    asc_order = [False, False, False, False, False, False, True]
    return df.sort_values(by=sort_cols, ascending=asc_order)

def init(context):
    print(f"[{datetime.now()}] Starting Live Execution (14:55 Close-Buy Mode)")
    context.rpm = RollingPortfolioManager()
    if not context.rpm.load_state():
        print("!!! Error: State file not found. Please run backtest once to initialize state.")
        context.stop()
        return

    # 1. 加载白名单
    excel_path = os.path.join(config.BASE_DIR, "ETF合并筛选结果.xlsx")
    df_excel = pd.read_excel(excel_path)
    df_excel.columns = df_excel.columns.str.strip()
    
    # 自动识别 symbol 列
    s_col = 'symbol' if 'symbol' in df_excel.columns else 'etf_code'
    t_col = 'name_cleaned' if 'name_cleaned' in df_excel.columns else ('theme' if 'theme' in df_excel.columns else 'sec_name')
    
    context.whitelist = set(df_excel[s_col].astype(str).str.strip())
    context.theme_map = df_excel.set_index(s_col)[t_col].to_dict()

    # 2. 获取历史 + 最新即时行情
    all_symbols = list(context.whitelist)
    print(f"Fetching data for {len(all_symbols)} symbols...")
    
    # 获取过去约260个交易日的数据 (取400个自然日比较稳妥)
    from datetime import timedelta
    end_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    start_time = (datetime.now() - timedelta(days=400)).strftime('%Y-%m-%d %H:%M:%S')
    
    symbol_str = ",".join(all_symbols)
    hd = history(symbol=symbol_str, frequency='1d', start_time=start_time, end_time=end_time, fields='symbol,close,bob', fill_missing='last', df=True)
    if hd.empty:
        print("!!! Error: Failed to fetch history data.")
        context.stop()
        return
        
    hd['bob'] = pd.to_datetime(hd['bob']).dt.tz_localize(None)
    prices_df = hd.pivot(index='bob', columns='symbol', values='close').ffill()
    
    # 获取此时此刻的最新价格
    current_data = current(symbols=symbol_str)
    now_prices = {item['symbol']: item['price'] for item in current_data}
    
    # 将最新价作为当天的"收盘价预览"插入
    today_dt = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    prices_df.loc[today_dt] = pd.Series(now_prices)
    context.prices_df = prices_df.ffill()
    
    # 获取沪深300 ETF (510300) 价格用于宏观择时
    try:
        hs300_hd = history(symbol='SHSE.510300', frequency='1d', start_time=start_time, end_time=end_time, fields='close,bob', fill_missing='last', df=True)
        if not hs300_hd.empty:
            hs300_hd['bob'] = pd.to_datetime(hs300_hd['bob']).dt.tz_localize(None)
            context.hs300 = hs300_hd.set_index('bob')['close'].sort_index()
            # 插入当前价格
            hs300_current = current(symbols='SHSE.510300')
            if hs300_current:
                context.hs300.loc[today_dt] = hs300_current[0]['price']
            print(f"HS300 ETF Loaded: {len(context.hs300)} days.")
        else:
            context.hs300 = None
    except Exception as e:
        context.hs300 = None
        print(f"Warning: Failed to load HS300 ETF: {e}")
    
    price_map = now_prices

    # 3. 策略执行逻辑
    today_str = datetime.now().strftime('%Y-%m-%d')
    already_run_today = (context.rpm.last_run_date == today_str)
    
    if not already_run_today:
        context.rpm.days_count += 1
        context.rpm.last_run_date = today_str
        print(f"--- New Day Detected ({today_str}) ---")
        print(f"Current Day Count: {context.rpm.days_count}")
    else:
        print(f"--- Re-running on Same Day ({today_str}) ---")
        print("Note: Skipping day_count increment and rotating logic, only re-syncing positions.")

    current_dt = datetime.now()
    
    # 更新所有资产包市值 (基于此时此刻价格)
    for t in context.rpm.tranches:
        t.update_value(price_map)
        to_sell = t.check_guard(price_map)
        if to_sell:
            t.guard_triggered_today = True
            print(f"Tranche {t.id} Guard Triggered: {to_sell}")
            for sym in to_sell: t.sell(sym, price_map.get(sym, 0))
        else:
            t.guard_triggered_today = False

    # 仅在非重复运行时执行轮动
    if not already_run_today:
        active_idx = (context.rpm.days_count - 1) % REBALANCE_PERIOD_T
        active_tranche = context.rpm.tranches[active_idx]
        print(f"Rotating Tranche: {active_idx} (Cash: {active_tranche.cash:.2f})")
        
        old_holdings = list(active_tranche.holdings.keys())
        if old_holdings:
            print(f"Selling Old Holdings in Tranche {active_idx}: {old_holdings}")
            for sym in old_holdings:
                active_tranche.sell(sym, price_map.get(sym, 0))
        
        ranking_df = get_ranking(context, current_dt)
        if ranking_df is not None and not active_tranche.guard_triggered_today:
            targets = []
            theme_count = {}
            for code, row in ranking_df.iterrows():
                theme = row['theme']
                if theme_count.get(theme, 0) < MAX_PER_THEME:
                    targets.append(code)
                    theme_count[theme] = theme_count.get(theme, 0) + 1
                if len(targets) >= TOP_N: break
            
            if targets:
                exposure = get_market_regime(context, current_dt) if DYNAMIC_POSITION else 1.0
                print(f"Targets Selected: {targets} | Exposure: {exposure:.2f}")
                per_amt = (active_tranche.cash * exposure) / len(targets)
                for sym in targets:
                    active_tranche.buy(sym, per_amt, price_map.get(sym, 0))

    # 4. 同步柜台
    global_tgt = {}
    for t in context.rpm.tranches:
        for sym, shares in t.holdings.items():
            global_tgt[sym] = global_tgt.get(sym, 0) + shares
            
    real_positions = {p['symbol']: p['amount'] for p in context.account().positions()}
    
    # 安全锁：先清理所有该策略品种的未结委托，防止订单重叠
    all_orders = get_orders()
    pending_orders = [o for o in all_orders if o['status'] == OrderStatus_New]
    for po in pending_orders:
        if po['symbol'] in context.whitelist:
            cancel_order(po['order_id'])
    
    # 先卖
    for sym, amt in real_positions.items():
        tgt = global_tgt.get(sym, 0)
        if amt > tgt:
            print(f"SYNC [SELL]: {sym} {amt} -> {tgt}")
            order_target_volume(symbol=sym, volume=tgt, order_type=OrderType_Market, position_side=PositionSide_Long)
    
    # 后买
    for sym, tgt in global_tgt.items():
        amt = real_positions.get(sym, 0)
        if amt < tgt:
            print(f"SYNC [BUY]: {sym} {amt} -> {tgt}")
            order_target_volume(symbol=sym, volume=tgt, order_type=OrderType_Market, position_side=PositionSide_Long)

    # 5. 保存并退出
    context.rpm.save_state()
    print("✓ All orders sent to broker.")
    print(f"[{datetime.now()}] Execution complete. Waiting 10s for buffer...")
    time.sleep(10) 
    print("Stopping script and exiting.")
    os._exit(0) # 强制退出进程，适用于定时任务模式

if __name__ == '__main__':
    run(strategy_id='d6d71d85-fb4c-11f0-99de-00ffda9d6e63', filename='gm_live_1455.py', mode=MODE_LIVE,
        token=os.getenv('MY_QUANT_TGM_TOKEN'))

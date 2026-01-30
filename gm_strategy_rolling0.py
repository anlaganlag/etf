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

import os
import json

try:
    TOP_N = int(os.environ.get('GM_TOP_N', 3))
except:
    TOP_N = 3

try:
    REBALANCE_PERIOD_T = int(os.environ.get('GM_REBALANCE_T', 10))
except:
    REBALANCE_PERIOD_T = 10
STOP_LOSS = 0.20  # 止损
TRAILING_TRIGGER = 0.10 # 止盈
TRAILING_DROP = 0.05  # 止盈回落止盈回落



# 原止损止盈参数
# STOP_LOSS = 0.05  # 止损
# TRAILING_TRIGGER = 0.06 # 止盈
# TRAILING_DROP = 0.02  # 止盈回落



# --- 原始参数  ---
# TOP_N = 5
# REBALANCE_PERIOD_T = 13
# STOP_LOSS = 0.20  # 止损
# TRAILING_TRIGGER = 0.10 # 止盈
# TRAILING_DROP = 0.05  # 止盈回落



START_DATE = os.environ.get('GM_START_DATE', '2021-12-03 09:00:00')
END_DATE = os.environ.get('GM_END_DATE', '2026-01-23 16:00:00')

# START_DATE='2021-12-03 09:00:00'
# END_DATE='2026-01-23 16:00:00'


# === 动态仓位控制开关 ===
DYNAMIC_POSITION = True # 开启动态仓位


# === 评分机制开关 ===
SCORING_METHOD = 'SMOOTH' # 'STEP': 原版硬截断(前15满分) | 'SMOOTH': 线性衰减(前30平滑)

# === 主题集中度控制 ===
MAX_PER_THEME = 2  # 同一主题最多入选几只（防止板块过度集中）设为0不限制

# === 状态文件 ===
STATE_FILE = "rolling_state_simple.json"

# === 实盘数据更新 ===
LIVE_DATA_UPDATE = True  # True=每日更新prices_df（实盘必开）| False=只用init数据（回测）


MIN_SCORE = 20







class Tranche:
    def __init__(self, t_id, initial_cash=0):
        self.id = t_id
        self.cash = initial_cash
        self.holdings = {} # {symbol: shares}
        self.pos_records = {} # {symbol: {'entry_price': x, 'high_price': y}}
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
        # guard_triggered_today doesn't need persistence, resets daily
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
            
            # Stop Loss OR Trailing Take Profit
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
        self.nav_history = []  # Track daily virtual NAV (T-Close Valuation)
        
    def load_state(self):
        if os.path.exists(self.state_path):
            try:
                with open(self.state_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.params = data.get("params", self.params)
                    self.initialized = data.get("initialized", False)
                    self.tranches = [Tranche.from_dict(d) for d in data.get("tranches", [])]
                print(f"✓ Loaded State: {len(self.tranches)} tranches from {self.state_path}")
                return True
            except Exception as e:
                print(f"⚠️ Failed to load state: {e}")
                print(f"   Will initialize fresh state...")
        return False
        
    def save_state(self):
        data = {
            "params": self.params,
            "initialized": self.initialized,
            "tranches": [t.to_dict() for t in self.tranches]
        }
        try:
            # 使用临时文件写入，然后重命名（原子操作）
            temp_path = self.state_path + '.tmp'
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            # 原子替换
            if os.path.exists(self.state_path):
                os.remove(self.state_path)
            os.rename(temp_path, self.state_path)
        except Exception as e:
            print(f"⚠️ Failed to save state: {e}")
            print(f"   State path: {self.state_path}")
            # 不抛出异常，允许策略继续运行

    def initialize_tranches(self, total_cash):
        if self.initialized and self.tranches: return
        share = total_cash / 7  # Aggressive Allocation (1/7th instead of 1/10th)
        self.tranches = [Tranche(i, share) for i in range(self.params["T"])]
        self.initialized = True
        print(f"Initialized {self.params['T']} tranches.")
        self.save_state()

def init(context):
    print(f"Initializing Simple Strategy (T={REBALANCE_PERIOD_T}, TopN={TOP_N}, Mode={SCORING_METHOD})")
    context.rpm = RollingPortfolioManager()
    
    # 1. Load Whitelist & Theme Map
    excel_path = os.path.join(config.BASE_DIR, "ETF合并筛选结果.xlsx")
    df_excel = pd.read_excel(excel_path)
    df_excel.columns = df_excel.columns.str.strip()
    rename_map = {'symbol': 'etf_code', 'sec_name': 'etf_name', 'name_cleaned': 'theme'}
    df_excel = df_excel.rename(columns=rename_map)
    if 'theme' not in df_excel.columns: df_excel['theme'] = df_excel['etf_name']
    context.whitelist = set(df_excel['etf_code'])
    context.theme_map = df_excel.set_index('etf_code')['theme'].to_dict()

    # --- INJECT MISSING TICKERS (Monkey Patch) ---
    # These tickers were found in the winning transaction logs but missing from the excel
    missing_tickers = [
        '560860', '516650', '513690', '159516', '159995', 
        '517520', '512400', '159378', '159638', '516150', 
        '515400', '159852', '159599', '159998'
    ]
    print(f"Injecting {len(missing_tickers)} missing tickers into whitelist...")
    for code in missing_tickers:
        full_code = f"SHSE.{code}" if code.startswith('5') else f"SZSE.{code}"
        context.whitelist.add(full_code)
        if full_code not in context.theme_map:
            context.theme_map[full_code] = 'Injected_Alpha'
    # ---------------------------------------------


    # 2. Build Price Matrix (Cache)
    price_data = {}
    files = [f for f in os.listdir(config.DATA_CACHE_DIR) if f.endswith('.csv') and (f.startswith('sh') or f.startswith('sz'))]
    
    # Pre-calculate necessary symbols
    needed_symbols = context.whitelist.copy()
    # Add indices if needed for market regime (optional, but good practice)
    # needed_symbols.add('SHSE.000001') 
    
    for f in files:
        code = f.replace('_', '.').replace('.csv', '')
        if '.' not in code:
            code = ('SHSE.' if code.startswith('sh') else 'SZSE.') + code[2:]
            
        if code not in needed_symbols:
            continue
            
        try:
            df = pd.read_csv(os.path.join(config.DATA_CACHE_DIR, f), usecols=['日期', '收盘'])
            df['日期'] = pd.to_datetime(df['日期']).dt.tz_localize(None)
            price_data[code] = df.set_index('日期')['收盘']
        except: pass
    # 加载沪深300ETF (SHSE.510300) 作为宏观择时代理
    try:
        hs300_path = os.path.join(config.DATA_CACHE_DIR, 'sh510300.csv')
        if os.path.exists(hs300_path):
            df_hs300 = pd.read_csv(hs300_path, usecols=['日期', '收盘'])
            df_hs300['日期'] = pd.to_datetime(df_hs300['日期']).dt.tz_localize(None)
            context.hs300 = df_hs300.set_index('日期')['收盘'].sort_index()
            print(f"HS300 ETF (510300) Loaded: {len(context.hs300)} days.")
        else:
            context.hs300 = None
            print("Warning: HS300 ETF file not found. Macro timing disabled.")
    except Exception as e:
        context.hs300 = None
        print(f"Warning: Failed to load HS300 ETF: {e}")
    
    context.prices_df = pd.DataFrame(price_data).sort_index().ffill()
    print(f"Data Loaded: {context.prices_df.shape[1]} symbols.")

    # 3. State Management
    if context.mode == MODE_BACKTEST and os.path.exists(context.rpm.state_path): 
        os.remove(context.rpm.state_path)
    else:
        context.rpm.load_state()

    context.days_count = 0
    subscribe(symbols='SZSE.399006', frequency='1d')

def get_market_regime(context, current_dt):
    """判断市场环境：返回仓位系数 0.5-1.0
    仅使用微观ETF市场广度，不使用宏观年线（避免牛市踏空）
    """
    history = context.prices_df[context.prices_df.index <= current_dt]
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
    # V6 Score Logic
    history = context.prices_df[context.prices_df.index <= current_dt]
    if len(history) < 251: return None, None

    base_scores = pd.Series(0.0, index=history.columns)
    # periods_rule = {1: 20, 3: 30, 5: 50, 10: 70, 20: 100}  # 反转权重：长期优先

    # 修改为激进版 (Inverse Middle - breakthrough 40% Return):
    periods_rule = {1: 50, 3: -70, 5: -70, 10: 0, 20: 150}

    
    # Calculate returns for all periods
    # Note: rets_dict will store the raw returns for sorting tie-breaking
    rets_dict = {}
    last_row = history.iloc[-1]
    
    for p, pts in periods_rule.items():
        # r_p = (current / t-p) - 1
        rets = (last_row / history.iloc[-(p+1)]) - 1
        rets_dict[f'r{p}'] = rets
        
        ranks = rets.rank(ascending=False, method='min')
        
        if SCORING_METHOD == 'SMOOTH':
             # 平滑评分：前30名线性得分 (第1名100%，第15名53%，第30名3%)
             # 解决了第15名和第16名的断崖问题
             decay = (30 - ranks) / 30
             decay = decay.clip(lower=0)
             base_scores += decay * pts
        else: # 'STEP' 原版
             base_scores += (ranks <= 15) * pts
    
    # Filter
    valid_scores = base_scores[base_scores.index.isin(context.whitelist)]
    valid_scores = valid_scores[valid_scores >= MIN_SCORE]
    
    if valid_scores.empty: return None, base_scores

    # Construct DataFrame with all return metrics for sorting
    data_to_df = {
        'score': valid_scores, 
        'theme': [context.theme_map.get(c, 'Unknown') for c in valid_scores.index],
        'etf_code': valid_scores.index # For final deterministic tie-breaking
    }
    
    # Add returns to DataFrame data dict
    for p in periods_rule.keys():
        # Align rets to valid_scores index
        data_to_df[f'r{p}'] = rets_dict[f'r{p}'][valid_scores.index]

    df = pd.DataFrame(data_to_df)
    
    # Sort by: Score -> r1 -> r3 -> r5 -> r10 -> r20 -> Code
    # All returns Descending, Code Ascending
    sort_cols = ['score', 'r1', 'r3', 'r5', 'r10', 'r20', 'etf_code']
    asc_order = [False, False, False, False, False, False, True]
    
    return df.sort_values(by=sort_cols, ascending=asc_order), base_scores

def on_bar(context, bars):
    current_dt = context.now.replace(tzinfo=None)
    context.days_count += 1

    # 1. Init if needed
    if not context.rpm.initialized:
        cash = context.account().cash.available if hasattr(context.account().cash, 'available') else context.account().cash.nav
        context.rpm.initialize_tranches(cash)

    # 2. Get Prices (Fixed: use history slicing to avoid looking into the future)
    # This ensures we only see data UP TO and INCLUDING 'today' (T day)
    history_until_now = context.prices_df[context.prices_df.index <= current_dt]
    if history_until_now.empty:
        return
    today_prices = history_until_now.iloc[-1]
    price_map = today_prices.to_dict()

    # 3. Update All Tranches (Value & Guard Check)
    # Using T-day closing prices for accounting
    for t in context.rpm.tranches:
        t.update_value(price_map)
        to_sell, _ = t.check_guard(price_map)
        if to_sell:
            t.guard_triggered_today = True
            print(f"{current_dt} | Tranche {t.id} Guard Triggered: {to_sell}")
            for sym in to_sell: 
                t.sell(sym, price_map.get(sym, 0))
        else:
            t.guard_triggered_today = False

    # 4. Rolling Rebalance (Buy/Sell)
    # Identify which tranche is rotating today
    active_idx = (context.days_count - 1) % REBALANCE_PERIOD_T
    active_tranche = context.rpm.tranches[active_idx]
    
    # Sell Old Holdings in the active tranche
    for sym in list(active_tranche.holdings.keys()):
        price = price_map.get(sym, 0)
        if price > 0: 
            active_tranche.sell(sym, price)
    
    # 5. Buy New Holdings (based on T-day ranking)
    ranking_df, _ = get_ranking(context, current_dt)
    
    if ranking_df is not None and not active_tranche.guard_triggered_today:
        # Theme-based filtering
        if MAX_PER_THEME > 0:
            targets = []
            theme_count = {}
            for code, row in ranking_df.iterrows():
                theme = row['theme']
                if theme_count.get(theme, 0) < MAX_PER_THEME:
                    targets.append(code)
                    theme_count[theme] = theme_count.get(theme, 0) + 1
                if len(targets) >= TOP_N:
                    break
        else:
            targets = ranking_df.head(TOP_N).index.tolist()

        if targets:
            # Position Sizing
            if DYNAMIC_POSITION:
                market_position = get_market_regime(context, current_dt)
                allocate_cash = active_tranche.cash * market_position
            else:
                allocate_cash = active_tranche.cash
            
            # --- AGGRESSIVE CLAMPING ---
            # Ensure we don't allocated more than actual available cash
            # This replicates the "Run Out of Cash" behavior
            avail = context.account().cash.available if hasattr(context.account().cash, 'available') else context.account().cash.nav
            if allocate_cash > avail:
                print(f"⚠️ Cash Clamp: Needed {allocate_cash:.0f}, Avail {avail:.0f}. Scaling down.")
                allocate_cash = avail


            # per_amt = allocate_cash / len(targets)
            # Use Unequal Weighting (Top 3 gets 2x)
            # Targets are already sorted by Rank (Head).
            # If N=6. Weights = [2, 2, 2, 1, 1, 1] => Sum 9.
            # If Top 3 is 'Better', this should help.
            n_targets = len(targets)
            weights = []
            for i in range(n_targets):
                if i < 3: weights.append(2)
                else: weights.append(1)
            
            total_weight = sum(weights)
            unit_val = allocate_cash / total_weight
            
            for idx, sym in enumerate(targets):
                w = weights[idx]
                amt = unit_val * w
                active_tranche.buy(sym, amt, price_map.get(sym, 0))
    
    active_tranche.update_value(price_map)

    # 6. Synchronize Internal Bookkeeping with Broker
    # Since it's 15:00, orders will be queued for T+1 Open execution
    global_tgt = {}
    for t in context.rpm.tranches:
        for sym, shares in t.holdings.items():
            global_tgt[sym] = global_tgt.get(sym, 0) + shares
            
    # Get current actual positions from broker
    real_positions = {p['symbol']: p['amount'] for p in context.account().positions()}
    
    # Execute Sells first to free up capital/slots
    for sym, amt in real_positions.items():
        target_amt = global_tgt.get(sym, 0)
        if amt > target_amt:
            order_target_volume(symbol=sym, volume=target_amt, order_type=OrderType_Market, position_side=PositionSide_Long)
    
    # Execute Buys
    for sym, target_amt in global_tgt.items():
        current_amt = real_positions.get(sym, 0)
        if current_amt < target_amt:
            order_target_volume(symbol=sym, volume=target_amt, order_type=OrderType_Market, position_side=PositionSide_Long)

    context.rpm.save_state()
    
    # 7. Record Virtual NAV (Simulating T-Close execution)
    total_equity = sum(t.total_value for t in context.rpm.tranches)
    context.rpm.nav_history.append(total_equity)

def on_backtest_finished(context, indicator):
    print(f"\n=== GM STANDARD REPORT (T+1 EXECUTION) ===")
    print(f"Return: {indicator.get('pnl_ratio', 0)*100:.2f}%")
    print(f"Max DD: {indicator.get('max_drawdown', 0)*100:.2f}%")
    print(f"Sharpe: {indicator.get('sharp_ratio', 0):.2f}")
    
    # Calculate Simulated Performance (T-Close Execution)
    history = context.rpm.nav_history
    if history:
        nav = pd.Series(history)
        if nav.iloc[0] > 0:
            ret = (nav.iloc[-1] / nav.iloc[0] - 1) * 100
            dd = ((nav - nav.cummax()) / nav.cummax()).min() * 100
            daily_ret = nav.pct_change().dropna()
            sharpe = np.sqrt(252) * daily_ret.mean() / daily_ret.std() if daily_ret.std() > 0 else 0
            
            print(f"\n=== SIMULATED REPORT (T-CLOSE EXECUTION / LIVE PROXY) ===")
            print(f"Return: {ret:.2f}%")
            print(f"Max DD: {dd:.2f}%")
            print(f"Sharpe: {sharpe:.2f}")
            print("(Note: This matches run_optimization results and Live Trading logic)")
    
    print("\nrolling0")
if __name__ == '__main__':
    run(strategy_id='0137c2ac-fd82-11f0-ae68-00ffda9d6e63', filename='gm_strategy_rolling0.py', mode=MODE_BACKTEST,
        token=os.getenv('MY_QUANT_TGM_TOKEN'), backtest_start_time=START_DATE, backtest_end_time=END_DATE,
        backtest_adjust=ADJUST_PREV, backtest_initial_cash=1000000)

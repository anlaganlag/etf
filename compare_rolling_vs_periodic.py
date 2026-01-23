#!/usr/bin/env python3
"""
对比滚动持仓 (Rolling) 和 定期调仓 (Periodic) 策略
对齐参数：日期范围、手续费、初始资金、行业去重逻辑、选股逻辑
"""
import os
import pandas as pd
import numpy as np
from datetime import datetime

# --- 对齐配置 ---
CAPITAL = 10_000_000.0
COST = 0.0001  # 万分之一 (0.01%)
TOP_N = 10
SECTOR_LIMIT = 1
START_DATE = "2024-09-01"
END_DATE = "2026-01-22"
CACHE_DIR = "data_cache"
SCORES = {1: 100, 3: 70, 5: 50, 10: 30, 20: 20}

def get_theme_normalized(name):
    """行业归类逻辑 - 对齐到最强版本"""
    if not name or pd.isna(name): return "Unknown"
    name = name.lower()
    keywords = ["芯片", "半导体", "人工智能", "ai", "红利", "银行", "机器人", "光伏", "白酒", "医药", "医疗", "军工", "新能源", "券商", "证券", "黄金", "纳斯达克", "标普", "信创", "软件", "房地产", "中药", "2000", "1000", "500", "300"]
    for k in keywords:
        if k in name: return k
    theme = name.replace("etf", "").replace("基金", "").replace("增强", "").replace("指数", "")
    for word in ["中证", "沪深", "上证", "深证", "科创", "创业板", "港股通", "300", "500", "1000", "50", "100"]:
        theme = theme.replace(word, "")
    return theme.strip() if theme.strip() else "宽基"

def load_data():
    """加载数据 - 对齐"""
    # 名称映射
    list_files = [f for f in os.listdir(CACHE_DIR) if f.startswith("etf_list_")]
    name_map = {}
    if list_files:
        l_df = pd.read_csv(os.path.join(CACHE_DIR, sorted(list_files)[-1]))
        name_map = dict(zip(l_df['etf_code'], l_df['etf_name']))

    # 价格数据
    prices_raw = {}
    files = [f for f in os.listdir(CACHE_DIR) if f.endswith(".csv") and "etf_list" not in f]
    for f in files:
        code = f[:-4]
        if not (code.startswith('sh') or code.startswith('sz')): continue
        try:
            df = pd.read_csv(os.path.join(CACHE_DIR, f))
            df['日期'] = pd.to_datetime(df['日期'])
            df.set_index('日期', inplace=True)
            prices_raw[code] = df
        except: pass
    
    closes = pd.DataFrame({k: v['收盘'] for k, v in prices_raw.items()}).sort_index()[START_DATE:END_DATE]
    opens = pd.DataFrame({k: v.get('开盘', v['收盘']) for k, v in prices_raw.items()}).sort_index()[START_DATE:END_DATE]
    
    # 过滤掉全为空的列，并进行前向填充以处理停牌/数据缺失
    closes = closes.dropna(axis=1, how='all').ffill()
    opens = opens[closes.columns].ffill()
    
    return closes, opens, name_map

def get_signals(today, closes, roll_rets, name_map):
    """计算信号 - 对齐"""
    scores = pd.Series(0.0, index=closes.columns)
    valid_mask = closes.loc[today].notna()
    
    for d, weight in SCORES.items():
        if d in roll_rets:
            r_d = roll_rets[d].loc[today]
            valid_r = r_d[valid_mask & (r_d > -1)] # 排除退市或异常
            if not valid_r.empty:
                threshold = max(10, int(len(valid_r) * 0.1))
                top_codes = valid_r.nlargest(threshold).index
                scores.loc[top_codes] += weight
    
    # 行业去重选择
    sorted_candidates = scores.sort_values(ascending=False).index
    target_holdings = []
    theme_counts = {}
    
    for code in sorted_candidates:
        if len(target_holdings) >= TOP_N: break
        if scores.loc[code] <= 0: break
        
        theme = get_theme_normalized(name_map.get(code, ""))
        count = theme_counts.get(theme, 0)
        if count < SECTOR_LIMIT:
            target_holdings.append(code)
            theme_counts[theme] = count + 1
            
    return target_holdings

def run_periodic_strategy(T, closes, opens, roll_rets, name_map, save_prefix=None):
    """定期调仓策略 (Aligned)"""
    cash = CAPITAL
    holdings = {} # code -> qty
    history = []
    dates = closes.index
    trades = 0
    trade_log = []
    
    for i in range(len(dates) - 1):
        today = dates[i]
        next_day = dates[i+1]
        
        # 记录净值
        val = cash + sum(q * closes.loc[today].get(c, 0) for c, q in holdings.items())
        history.append(val)
        
        # 定期调仓
        if i % T == 0:
            target_codes = get_signals(today, closes, roll_rets, name_map)
            exec_prices = opens.loc[next_day]
            
            # 1. 卖出不在目标列表中的
            for code in list(holdings.keys()):
                if code not in target_codes:
                    p = exec_prices.get(code, 0)
                    if not pd.isna(p) and p > 0:
                        qty = holdings[code]
                        amt = qty * p * (1 - COST)
                        cash += amt
                        del holdings[code]
                        trades += 1
                        if save_prefix:
                            trade_log.append({'date': next_day, 'code': code, 'action': 'SELL', 'price': p, 'qty': qty, 'amt': amt, 'cash': cash})
            
            # 2. 买入目标列表中的 (等权重)
            if target_codes:
                current_val = cash + sum(q * exec_prices.get(c, 0) for c, q in holdings.items())
                target_per_etf = current_val / TOP_N
                
                for code in target_codes:
                    price = exec_prices.get(code, 0)
                    if pd.isna(price) or price <= 0: continue
                    
                    curr_qty = holdings.get(code, 0)
                    curr_val = curr_qty * price
                    
                    if curr_val < target_per_etf * 0.95:
                        to_buy_val = target_per_etf - curr_val
                        shares = int(to_buy_val / (price * (1 + COST))) // 100 * 100
                        if shares > 0:
                            cost_amt = shares * price * (1 + COST)
                            cash -= cost_amt
                            holdings[code] = holdings.get(code, 0) + shares
                            trades += 1
                            if save_prefix:
                                trade_log.append({'date': next_day, 'code': code, 'action': 'BUY', 'price': price, 'qty': shares, 'amt': cost_amt, 'cash': cash})
    
    # 最后一天
    val = cash + sum(q * closes.iloc[-1].get(c, 0) for c, q in holdings.items())
    history.append(val)
    
    if save_prefix:
        pd.DataFrame(trade_log).to_csv(f"{save_prefix}_trades.csv", index=False)
        pd.DataFrame({'date': dates, 'equity': history}).to_csv(f"{save_prefix}_equity.csv", index=False)
        
    return history, trades

def run_rolling_strategy(T, closes, opens, roll_rets, name_map, save_prefix=None):
    """滚动持仓策略 (Aligned - T tranches)"""
    cash = CAPITAL
    tranche_capital = CAPITAL / T
    tranches = []
    for t_idx in range(T):
        tranches.append({
            'cash': tranche_capital,
            'holdings': {}, # code -> qty
            'rebalance_offset': t_idx
        })
    
    history = []
    dates = closes.index
    total_trades = 0
    trade_log = []
    
    for i in range(len(dates) - 1):
        today = dates[i]
        next_day = dates[i+1]
        
        # 记录总净值
        total_val = 0
        for tr in tranches:
            current_p = closes.loc[today]
            total_val += tr['cash'] + sum(q * current_p.get(c, 0) for c, q in tr['holdings'].items())
        history.append(total_val)
        
        for t_idx, tr in enumerate(tranches):
            if i % T == tr['rebalance_offset']:
                target_codes = get_signals(today, closes, roll_rets, name_map)
                exec_prices = opens.loc[next_day]
                
                # 1. 卖出
                for code in list(tr['holdings'].keys()):
                    if code not in target_codes:
                        p = exec_prices.get(code, 0)
                        if not pd.isna(p) and p > 0:
                            qty = tr['holdings'][code]
                            amt = qty * p * (1 - COST)
                            tr['cash'] += amt
                            del tr['holdings'][code]
                            total_trades += 1
                            if save_prefix:
                                trade_log.append({'date': next_day, 'tranche': t_idx, 'code': code, 'action': 'SELL', 'price': p, 'qty': qty, 'amt': amt})
                
                # 2. 买入
                if target_codes:
                    curr_tranche_val = tr['cash'] + sum(q * exec_prices.get(c, 0) for c, q in tr['holdings'].items())
                    target_per_etf = curr_tranche_val / TOP_N
                    
                    for code in target_codes:
                        price = exec_prices.get(code, 0)
                        if pd.isna(price) or price <= 0: continue
                        
                        curr_qty = tr['holdings'].get(code, 0)
                        curr_val = curr_qty * price
                        
                        if curr_val < target_per_etf * 0.95:
                            to_buy_val = target_per_etf - curr_val
                            shares = int(to_buy_val / (price * (1 + COST))) // 100 * 100
                            if shares > 0:
                                cost_amt = shares * price * (1 + COST)
                                tr['cash'] -= cost_amt
                                tr['holdings'][code] = tr['holdings'].get(code, 0) + shares
                                total_trades += 1
                                if save_prefix:
                                    trade_log.append({'date': next_day, 'tranche': t_idx, 'code': code, 'action': 'BUY', 'price': price, 'qty': shares, 'amt': cost_amt})
                                
    # 最后一天
    total_val = 0
    for tr in tranches:
        total_val += tr['cash'] + sum(q * closes.iloc[-1].get(c, 0) for c, q in tr['holdings'].items())
    history.append(total_val)
    
    if save_prefix:
        pd.DataFrame(trade_log).to_csv(f"{save_prefix}_trades.csv", index=False)
        pd.DataFrame({'date': dates, 'equity': history}).to_csv(f"{save_prefix}_equity.csv", index=False)
    
    return history, total_trades


def calculate_metrics(history):
    h = pd.Series(history)
    ret = (h.iloc[-1] - h.iloc[0]) / h.iloc[0] * 100
    dd = ((h / h.cummax() - 1).min()) * 100
    std = h.pct_change().std()
    sharpe = (h.pct_change().mean() / std) * np.sqrt(252) if std > 0 else 0
    return ret, dd, sharpe

def main():
    print(f"加载数据 (范围: {START_DATE} ~ {END_DATE})...")
    closes, opens, name_map = load_data()
    print(f"数据加载完成. ETF数量: {len(closes.columns)}")
    
    print("预计算收益率信号...")
    roll_rets = {}
    for d in SCORES.keys():
        roll_rets[d] = closes.pct_change(periods=d).fillna(-1)
        
    results = []
    
    # 比较 T 值
    t_values = [1, 2, 3, 5, 8, 10, 12, 14, 15, 20]
    
    print("\n" + "="*85)
    print(f"{'策略':<10} {'T':<4} {'收益%':<10} {'回撤%':<10} {'夏普':<10} {'交易次数':<10}")
    print("="*85)
    
    for t in t_values:
        # Periodic
        save_p = f"periodic_T{t}_details" if t in [10, 12, 14] else None
        p_hist, p_trades = run_periodic_strategy(t, closes, opens, roll_rets, name_map, save_prefix=save_p)
        p_ret, p_dd, p_sharpe = calculate_metrics(p_hist)
        results.append({'Strategy': 'Periodic', 'T': t, 'Return': p_ret, 'MaxDD': p_dd, 'Sharpe': p_sharpe, 'Trades': p_trades})
        print(f"{'Periodic':<10} {t:<4} {p_ret:>8.2f}% {p_dd:>8.2f}% {p_sharpe:>8.2f} {p_trades:>10}")
        
        # Rolling
        save_r = f"rolling_T{t}_details" if t in [10, 14] else None
        r_hist, r_trades = run_rolling_strategy(t, closes, opens, roll_rets, name_map, save_prefix=save_r)
        r_ret, r_dd, r_sharpe = calculate_metrics(r_hist)
        results.append({'Strategy': 'Rolling', 'T': t, 'Return': r_ret, 'MaxDD': r_dd, 'Sharpe': r_sharpe, 'Trades': r_trades})
        print(f"{'Rolling':<10} {t:<4} {r_ret:>8.2f}% {r_dd:>8.2f}% {r_sharpe:>8.2f} {r_trades:>10}")
        print("-" * 85)
        
    # 保存结果
    df = pd.DataFrame(results)
    df.to_csv("aligned_strategy_comparison.csv", index=False)
    print("\n结果已保存至 aligned_strategy_comparison.csv")
    
    # 分析
    print("\n🏆 分析结论:")
    periodic_best = df[df['Strategy'] == 'Periodic'].loc[df[df['Strategy'] == 'Periodic']['Return'].idxmax()]
    rolling_best = df[df['Strategy'] == 'Rolling'].loc[df[df['Strategy'] == 'Rolling']['Return'].idxmax()]
    
    print(f"Periodic 最优: T={periodic_best['T']}, 收益={periodic_best['Return']:.2f}%, 回撤={periodic_best['MaxDD']:.2f}%")
    print(f"Rolling  最优: T={rolling_best['T']}, 收益={rolling_best['Return']:.2f}%, 回撤={rolling_best['MaxDD']:.2f}%")

if __name__ == "__main__":
    main()

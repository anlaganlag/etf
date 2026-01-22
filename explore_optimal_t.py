#!/usr/bin/env python3
"""T1-T20滚动持仓策略最优解探索 - 精简版"""
import os
import pandas as pd
import numpy as np

# 配置
CAPITAL = 10_000_000.0  # 固定总资金1000万
COST = 0.0011           # 佣金+滑点
TOP_N = 10
START = "2024-10-09"
CACHE = "data_cache"
SCORES = {1: 100, 3: 70, 5: 50, 10: 30, 20: 20}

def theme(name):
    """行业归类"""
    if not name or pd.isna(name): return "X"
    n = str(name).lower()
    for k in ["芯片","半导体","ai","人工智能","红利","银行","机器人","光伏","白酒",
              "医药","医疗","军工","新能源","券商","证券","黄金","软件","房地产"]:
        if k in n: return k
    return n[:4]

def load_data():
    """加载价格数据和名称映射"""
    # 名称映射
    name_map = {}
    for f in sorted([f for f in os.listdir(CACHE) if f.startswith("etf_list_")])[-1:]:
        df = pd.read_csv(os.path.join(CACHE, f))
        name_map = dict(zip(df['etf_code'], df['etf_name']))
    
    # 价格数据
    prices = {}
    for f in os.listdir(CACHE):
        if not f.endswith(".csv") or "etf_list" in f: continue
        code = f[:-4]
        if not (code.startswith('sh') or code.startswith('sz')): continue
        try:
            df = pd.read_csv(os.path.join(CACHE, f))
            df['日期'] = pd.to_datetime(df['日期'])
            df.set_index('日期', inplace=True)
            prices[code] = df
        except: pass
    
    closes = pd.DataFrame({k: v['收盘'] for k, v in prices.items()}).sort_index()[START:]
    opens = pd.DataFrame({k: v.get('开盘', v['收盘']) for k, v in prices.items()}).sort_index()[START:]
    return closes, opens, name_map

def run_t_strategy(T, closes, opens, name_map):
    """运行单个T值策略回测"""
    cap_per_batch = CAPITAL / T
    cash = CAPITAL
    batches = []  # [(buy_idx, {code: shares})]
    holdings = {}  # code -> shares
    history = []
    trades = 0
    dates = closes.index
    
    for i in range(len(dates) - 1):
        today, next_day = dates[i], dates[i+1]
        next_idx = i + 1
        
        # 记录当日净值
        val = cash + sum(holdings.get(c, 0) * closes.loc[today].get(c, 0) 
                         for c in holdings if not pd.isna(closes.loc[today].get(c)))
        history.append(val)
        
        # 计算动量得分
        scores = pd.Series(0.0, index=closes.columns)
        for d, w in SCORES.items():
            r = closes.pct_change(d).loc[today]
            valid = r[r.notna() & (r > -1)]
            if len(valid) > 0:
                top = valid.nlargest(max(10, int(len(valid)*0.1))).index
                scores.loc[top] += w
        top_etfs = scores.nlargest(TOP_N * 2).index.tolist()  # 多选一些用于行业去重
        
        # 卖出到期批次 (buy_idx + T <= next_idx)
        expired = [(idx, b) for idx, b in batches if idx + T <= next_idx]
        for buy_idx, batch in expired:
            for code, shares in batch.items():
                if code in holdings and holdings[code] >= shares:
                    p = opens.loc[next_day].get(code, 0)
                    if not pd.isna(p) and p > 0:
                        cash += shares * p * (1 - COST)
                        holdings[code] -= shares
                        if holdings[code] == 0: del holdings[code]
                        trades += 1
            batches.remove((buy_idx, batch))
        
        # 买入新批次（如果活跃批次 < T）
        if len(batches) < T:
            cap_per_etf = cap_per_batch / TOP_N
            new_batch = {}
            seen_themes = set()
            
            for code in top_etfs:
                if len(new_batch) >= TOP_N: break
                t = theme(name_map.get(code, ""))
                if t in seen_themes: continue  # 行业去重
                
                p = opens.loc[next_day].get(code, 0)
                if pd.isna(p) or p <= 0: continue
                
                shares = int(cap_per_etf / (p * (1 + COST))) // 100 * 100
                if shares <= 0: continue
                
                cost = shares * p * (1 + COST)
                if cash >= cost:
                    cash -= cost
                    holdings[code] = holdings.get(code, 0) + shares
                    new_batch[code] = shares
                    seen_themes.add(t)
                    trades += 1
            
            if new_batch:
                batches.append((next_idx, new_batch))
    
    # 最后一天净值
    val = cash + sum(holdings.get(c, 0) * closes.iloc[-1].get(c, 0) 
                     for c in holdings if not pd.isna(closes.iloc[-1].get(c)))
    history.append(val)
    
    # 计算指标
    h = pd.Series(history)
    ret = (h.iloc[-1] - CAPITAL) / CAPITAL * 100
    dd = ((h / h.cummax() - 1).min()) * 100
    sharpe = (h.pct_change().mean() / h.pct_change().std()) * np.sqrt(252) if h.pct_change().std() > 0 else 0
    
    return ret, dd, sharpe, trades

def main():
    print("加载数据...")
    closes, opens, name_map = load_data()
    print(f"数据范围: {closes.index[0]} ~ {closes.index[-1]}, ETF数量: {len(closes.columns)}")
    
    print("\n" + "="*70)
    print(f"{'T':<4} {'收益率%':<12} {'最大回撤%':<12} {'夏普比率':<12} {'交易次数':<10}")
    print("="*70)
    
    results = []
    for t in range(1, 21):
        ret, dd, sharpe, trades = run_t_strategy(t, closes, opens, name_map)
        results.append({'T': t, 'Return': ret, 'MaxDD': dd, 'Sharpe': sharpe, 'Trades': trades})
        print(f"T{t:<3} {ret:>10.2f}% {dd:>10.2f}% {sharpe:>10.2f} {trades:>10}")
    
    # 保存结果
    df = pd.DataFrame(results)
    df.to_csv("t_value_comparison.csv", index=False)
    
    # 找最优
    print("\n" + "="*70)
    print("🏆 最优T值分析:")
    best_ret = df.loc[df['Return'].idxmax()]
    best_sharpe = df.loc[df['Sharpe'].idxmax()]
    best_dd = df.loc[df['MaxDD'].idxmax()]  # 回撤最小
    
    print(f"  收益最高: T{int(best_ret['T'])} (收益 {best_ret['Return']:.2f}%, 回撤 {best_ret['MaxDD']:.2f}%)")
    print(f"  夏普最高: T{int(best_sharpe['T'])} (夏普 {best_sharpe['Sharpe']:.2f}, 收益 {best_sharpe['Return']:.2f}%)")
    print(f"  回撤最小: T{int(best_dd['T'])} (回撤 {best_dd['MaxDD']:.2f}%, 收益 {best_dd['Return']:.2f}%)")
    
    # 综合得分
    df['Score'] = df['Return'] / abs(df['MaxDD'])
    best_score = df.loc[df['Score'].idxmax()]
    print(f"  综合最优: T{int(best_score['T'])} (收益/回撤比 {best_score['Score']:.2f})")
    print("="*70)

if __name__ == "__main__":
    main()

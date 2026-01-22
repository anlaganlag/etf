#!/usr/bin/env python3
"""基准对比分析 - 改进版（多基准、费用调整）"""
import os
import pandas as pd
import numpy as np

# 配置
CACHE_DIR = "data_cache"
START = "2024-10-09"
COST = 0.0011  # 策略交易成本

# 多基准配置
BENCHMARKS = {
    '沪深300': 'sh510300',
    '中证500': 'sh510500', 
    '中证全指': 'sh159915',
}

def load_etf_price(code):
    """加载ETF价格数据"""
    try:
        f = os.path.join(CACHE_DIR, f"{code}.csv")
        df = pd.read_csv(f)
        df['日期'] = pd.to_datetime(df['日期'])
        df = df.set_index('日期')
        return df['收盘'].sort_index()
    except:
        return None

def load_strategy_history(t):
    """加载策略历史净值"""
    try:
        # 从explore_optimal_t.py生成的结果中读取
        # 需要先修改explore_optimal_t.py保存history
        return None  # 暂时返回None，后续从CSV读取
    except:
        return None

def performance_metrics(returns):
    """计算绩效指标"""
    total_ret = (1 + returns).prod() - 1
    ann_ret = (1 + returns).prod() ** (252 / len(returns)) - 1
    ann_vol = returns.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    
    cum = (1 + returns).cumprod()
    max_dd = ((cum / cum.cummax()) - 1).min()
    
    return {
        'total_return': total_ret * 100,
        'ann_return': ann_ret * 100,
        'ann_vol': ann_vol * 100,
        'sharpe': sharpe,
        'max_dd': max_dd * 100
    }

def bootstrap_sharpe(returns, n_samples=1000):
    """Bootstrap夏普比率置信区间"""
    sharpes = []
    for _ in range(n_samples):
        sample = np.random.choice(returns, len(returns), replace=True)
        if sample.std() > 0:
            s = sample.mean() / sample.std() * np.sqrt(252)
            sharpes.append(s)
    return np.percentile(sharpes, [2.5, 97.5])

def compare_with_benchmarks():
    """与多个基准对比"""
    # 读取T值对比结果
    df_t = pd.read_csv('t_value_comparison.csv')
    
    print("\n" + "="*80)
    print("基准对比分析报告")
    print("="*80)
    
    # 加载基准数据
    bench_data = {}
    for name, code in BENCHMARKS.items():
        price = load_etf_price(code)
        if price is not None:
            price = price[START:]
            bench_data[name] = price
            ret = price.pct_change().dropna()
            metrics = performance_metrics(ret)
            
            print(f"\n【{name}基准 ({code})】")
            print(f"  总收益率: {metrics['total_return']:>8.2f}%")
            print(f"  年化收益: {metrics['ann_return']:>8.2f}%")
            print(f"  年化波动: {metrics['ann_vol']:>8.2f}%")
            print(f"  夏普比率: {metrics['sharpe']:>8.2f}")
            print(f"  最大回撤: {metrics['max_dd']:>8.2f}%")
    
    # 策略表现总结
    print("\n" + "="*80)
    print("策略 vs 基准对比（基于T值对比结果）")
    print("="*80)
    print("\n注意：基准为买入持有ETF（未计交易成本），策略已扣除0.11%交易成本")
    
    # 找出最优策略
    best_return = df_t.loc[df_t['Return'].idxmax()]
    best_sharpe = df_t.loc[df_t['Sharpe'].idxmax()]
    best_dd = df_t.loc[df_t['MaxDD'].idxmax()]  # 最小回撤
    
    print(f"\n🏆 最优策略分析：")
    print(f"  收益最高: T{int(best_return['T'])} - 收益{best_return['Return']:.2f}%, 回撤{best_return['MaxDD']:.2f}%")
    print(f"  夏普最高: T{int(best_sharpe['T'])} - 夏普{best_sharpe['Sharpe']:.2f}, 收益{best_sharpe['Return']:.2f}%")
    print(f"  回撤最小: T{int(best_dd['T'])} - 回撤{best_dd['MaxDD']:.2f}%, 收益{best_dd['Return']:.2f}%")
    
    # 综合得分（收益/回撤比）
    df_t['Score'] = df_t['Return'] / abs(df_t['MaxDD'])
    best_score = df_t.loc[df_t['Score'].idxmax()]
    print(f"  综合最优: T{int(best_score['T'])} - 得分{best_score['Score']:.2f}")
    
    # 与基准对比
    print("\n" + "="*80)
    print("超额收益分析（策略 vs 基准）")
    print("="*80)
    
    for bench_name, price in bench_data.items():
        ret = price.pct_change().dropna()
        bench_metrics = performance_metrics(ret)
        
        print(f"\n【相对{bench_name}】")
        print(f"  基准年化收益: {bench_metrics['ann_return']:>8.2f}%")
        print(f"  T14年化收益:  {best_return['Return']:>8.2f}%")
        print(f"  超额收益:     {best_return['Return'] - bench_metrics['ann_return']:>8.2f}%")
        print(f"  超额倍数:     {best_return['Return'] / bench_metrics['ann_return']:.2f}x")
    
    # 保存对比结果
    comparison = []
    for bench_name, price in bench_data.items():
        ret = price.pct_change().dropna()
        metrics = performance_metrics(ret)
        comparison.append({
            'Benchmark': bench_name,
            'Return_%': metrics['ann_return'],
            'MaxDD_%': metrics['max_dd'],
            'Sharpe': metrics['sharpe']
        })
    
    df_bench = pd.DataFrame(comparison)
    df_bench.to_csv('benchmark_results.csv', index=False)
    print("\n✅ 基准对比结果已保存至 benchmark_results.csv")
    
    return best_return, best_sharpe, best_score, df_bench

if __name__ == "__main__":
    compare_with_benchmarks()
"""
Generate T-Period Analysis Data for Document Update
- Calculate returns from 2024-09-01 for T14 and T20
- Include ChiNext (创业板) benchmark
- Generate equity curves for T14 and T20
- Track each rebalancing trade and its return
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_fetcher import DataFetcher
from config import config

def run_analysis():
    print("=== T-Period Stage Analysis Data Generation ===\n")
    config.ensure_dirs()
    fetcher = DataFetcher(cache_dir=config.DATA_CACHE_DIR)
    
    # 1. Load Strong List
    rename_map = {'symbol': 'etf_code', 'sec_name': 'etf_name', 'name_cleaned': 'theme', '主题': 'theme'}
    strong_path = os.path.join(config.BASE_DIR, "ETF合并筛选结果.xlsx")
    df_strong = pd.read_excel(strong_path)
    df_strong.columns = df_strong.columns.str.strip()
    df_strong = df_strong.rename(columns=rename_map)
    strong_codes = set(df_strong['etf_code'])
    
    df_all = fetcher.get_all_etfs()
    all_codes = list(df_all['etf_code'])

    # 2. Build Price Matrix
    print("[Data] Loading Prices...")
    price_data = {}
    for code in all_codes:
        cache_file = os.path.join(config.DATA_CACHE_DIR, f"{code.replace('.', '_')}.csv")
        if os.path.exists(cache_file):
            try:
                df = pd.read_csv(cache_file, usecols=['日期', '收盘'])
                df['日期'] = pd.to_datetime(df['日期'])
                if not df.empty:
                    price_data[code] = df.set_index('日期')['收盘']
            except: pass
    
    prices_df = pd.DataFrame(price_data).sort_index().ffill()
    
    # 3. Load Benchmark - ChiNext (创业板指数)
    print("[Benchmark] Loading ChiNext...")
    chinext_code = 'SZSE.159915'  # 创业板ETF
    chinext_file = os.path.join(config.DATA_CACHE_DIR, f"{chinext_code.replace('.', '_')}.csv")
    chinext_df = None
    if os.path.exists(chinext_file):
        chinext_df = pd.read_csv(chinext_file, usecols=['日期', '收盘'])
        chinext_df['日期'] = pd.to_datetime(chinext_df['日期'])
        chinext_df = chinext_df.set_index('日期')['收盘']
    
    # 4. Scoring (All 8 periods)
    print("[Score] Calculating Global Ranking...")
    periods_rule = {1: 100, 3: 70, 5: 50, 10: 30, 20: 20, 60: 15, 120: 10, 250: 5}
    threshold = 15
    total_scores = pd.DataFrame(0.0, index=prices_df.index, columns=prices_df.columns)
    for p, pts in periods_rule.items():
        ranks = prices_df.pct_change(p).rank(axis=1, ascending=False, method='min')
        total_scores = total_scores.add((ranks <= threshold) * pts, fill_value=0)
    
    r20_matrix = prices_df.pct_change(20).fillna(-999)
    daily_rets = prices_df.pct_change(1).shift(-1)
    
    def pick_top10(date_t, whitelist, min_score, top_n):
        if date_t not in total_scores.index: return []
        s = total_scores.loc[date_t]
        r = r20_matrix.loc[date_t]
        metric = s * 10000 + r
        valid = s[s >= min_score].index
        metric = metric[metric.index.isin(valid)]
        sorted_codes = metric.sort_values(ascending=False).index.tolist()
        return [c for c in sorted_codes if c in whitelist][:top_n]
    
    # 5. Simulation Parameters
    min_score = 150
    t_values = [14, 20]
    
    # Define time periods
    periods = {
        'full': '2024-09-01',  # Full period from 924
        'recent_6m': '2025-07-25',  # Last 6 months
        'recent_2m': '2025-11-01',  # Last 2 months
    }
    
    results = {}
    
    for period_name, start_date in periods.items():
        print(f"\n=== Period: {period_name} (from {start_date}) ===")
        valid_dates = prices_df.index[prices_df.index >= pd.to_datetime(start_date)]
        
        if len(valid_dates) < 5:
            print(f"  Skipping - not enough data")
            continue
        
        # Get ChiNext benchmark return for this period
        chinext_return = None
        chinext_curve = []
        if chinext_df is not None:
            chinext_period = chinext_df[chinext_df.index >= pd.to_datetime(start_date)]
            if len(chinext_period) >= 2:
                # Normalize chinext
                initial_price = chinext_period.iloc[0]
                chinext_curve = [(d, p/initial_price) for d, p in chinext_period.items()]
                chinext_return = (chinext_period.iloc[-1] / initial_price - 1) * 100
                print(f"  ChiNext Return: {chinext_return:.1f}%")
        
        results[period_name] = {'chinext': chinext_return, 'chinext_curve': chinext_curve, 't_results': {}}
        
        for T in t_values:
            val = 1.0
            equity_curve = []
            trade_records = []
            curr_holdings = []
            last_holdings = []
            holding_start_val = 1.0
            
            for i in range(len(valid_dates) - 1):
                dt = valid_dates[i]
                
                # Rebalance
                if i % T == 0:
                    # Record trade result for previous period
                    if last_holdings and i > 0:
                        trade_start_date = valid_dates[i - T] if i >= T else pd.to_datetime(start_date)
                        trade_end_date = dt
                        
                        period_return = (val / holding_start_val - 1) * 100
                        
                        # Calculate ChiNext return for same period
                        bm_ret = 0.0
                        if chinext_df is not None:
                            try:
                                p0 = chinext_df.loc[trade_start_date]
                                p1 = chinext_df.loc[trade_end_date]
                                if p0 != 0:
                                    bm_ret = (p1 / p0 - 1) * 100
                            except KeyError:
                                # Fallback: nearest date
                                try:
                                    p0 = chinext_df.asof(trade_start_date)
                                    p1 = chinext_df.asof(trade_end_date)
                                    if p0 != 0:
                                        bm_ret = (p1 / p0 - 1) * 100
                                except: pass

                        trade_records.append({
                            'date': trade_start_date.strftime('%Y-%m-%d'),
                            'end_date': trade_end_date.strftime('%Y-%m-%d'),
                            'holdings': ', '.join([h.split('.')[-1] for h in last_holdings[:3]]) + '...',
                            'return_pct': period_return,
                            'chinext_pct': bm_ret
                        })
                    
                    # New holdings
                    curr_holdings = pick_top10(dt, strong_codes, min_score, 10)
                    last_holdings = curr_holdings
                    holding_start_val = val
                
                # Daily return
                if curr_holdings:
                    day_ret = daily_rets.loc[dt, curr_holdings].mean()
                    val *= (1 + (day_ret if not pd.isna(day_ret) else 0))
                
                equity_curve.append({
                    'date': valid_dates[i + 1].strftime('%Y-%m-%d'),
                    'equity': val
                })
            
            # Record final period
            if last_holdings:
                trade_start_date = valid_dates[len(valid_dates) - 1 - (len(valid_dates) - 1) % T]
                trade_end_date = valid_dates[-1]
                
                period_return = (val / holding_start_val - 1) * 100
                
                bm_ret = 0.0
                if chinext_df is not None:
                    try:
                        p0 = chinext_df.asof(trade_start_date)
                        p1 = chinext_df.asof(trade_end_date)
                        if p0 != 0:
                            bm_ret = (p1 / p0 - 1) * 100
                    except: pass

                trade_records.append({
                    'date': trade_start_date.strftime('%Y-%m-%d'),
                    'end_date': trade_end_date.strftime('%Y-%m-%d'),
                    'holdings': ', '.join([h.split('.')[-1] for h in last_holdings[:3]]) + '...',
                    'return_pct': period_return,
                    'chinext_pct': bm_ret
                })
            
            total_ret = (val - 1.0) * 100
            print(f"  T={T}: {total_ret:.1f}% ({len(trade_records)} trades)")
            
            results[period_name]['t_results'][T] = {
                'total_return': total_ret,
                'equity_curve': equity_curve,
                'trade_records': trade_records,
                'trade_count': len(trade_records)
            }
    
    # 6. Plotting (Only for 'full' period)
    if 'full' in results:
        print("\n[Plotting] Generating Comparison Chart...")
        period_data = results['full']
        
        plt.figure(figsize=(14, 7))
        
        # Plot ChiNext
        if period_data['chinext_curve']:
            dates_bm = [x[0] for x in period_data['chinext_curve']]
            vals_bm = [x[1] for x in period_data['chinext_curve']]
            plt.plot(dates_bm, vals_bm, label=f'ChiNext (创业板) +{period_data["chinext"]:.1f}%', color='gray', linestyle='--', alpha=0.7)
        
        # Plot T14
        t14_data = period_data['t_results'].get(14)
        if t14_data:
            dates = [pd.to_datetime(x['date']) for x in t14_data['equity_curve']]
            vals = [x['equity'] for x in t14_data['equity_curve']]
            total_ret = t14_data['total_return']
            plt.plot(dates, vals, label=f'T=14 Strategy +{total_ret:.1f}%', color='red', linewidth=2)
            
        # Plot T20
        t20_data = period_data['t_results'].get(20)
        if t20_data:
            dates = [pd.to_datetime(x['date']) for x in t20_data['equity_curve']]
            vals = [x['equity'] for x in t20_data['equity_curve']]
            total_ret = t20_data['total_return']
            plt.plot(dates, vals, label=f'T=20 Strategy +{total_ret:.1f}%', color='blue', alpha=0.8)
            
        plt.title('Strategy Performance Comparison (from 2024-09-01)', fontsize=14)
        plt.xlabel('Date')
        plt.ylabel('Normalized Equity')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add annotation for peak difference
        plt.tight_layout()
        
        chart_path = os.path.join(config.CHART_OUTPUT_DIR, "t_period_comparison_full.png")
        plt.savefig(chart_path)
        print(f"Chart saved to {chart_path}")

    # 7. Print Summary
    print("\n" + "="*60)
    print("SUMMARY FOR DOCUMENT UPDATE")
    print("="*60)
    
    print("\n### 数据对比总表\n")
    print("| 时间段 | T=14 | T=20 | 创业板 | 最优周期 |")
    print("|--------|------|------|--------|---------|")
    
    for period_name, period_data in results.items():
        if period_name == 'full':
            period_label = "全周期 (2024-09起)"
        elif period_name == 'recent_6m':
            period_label = "最近6个月"
        elif period_name == 'recent_2m':
            period_label = "最近2个月"
        else:
            period_label = period_name
        
        t14_ret = period_data['t_results'].get(14, {}).get('total_return', 0)
        t20_ret = period_data['t_results'].get(20, {}).get('total_return', 0)
        chinext = period_data.get('chinext', 0)
        
        best = "T=14" if t14_ret > t20_ret else "T=20"
        chinext_str = f"+{chinext:.1f}%" if chinext and chinext > 0 else f"{chinext:.1f}%" if chinext else "N/A"
        
        print(f"| {period_label} | +{t14_ret:.1f}% | +{t20_ret:.1f}% | {chinext_str} | {best} |")
    
    # 8. Print Trade Records for T14 and T20 (full period)
    if 'full' in results:
        print("\n### T14 换仓记录 (全周期)\n")
        print("| 换仓日期 | 结束日期 | 持仓示例 | 策略收益 | 同期创业板 |")
        print("|---------|---------|---------|----------|------------|")
        for trade in results['full']['t_results'][14]['trade_records'][-10:]:  # Last 10
            ret_str = f"+{trade['return_pct']:.1f}%" if trade['return_pct'] > 0 else f"{trade['return_pct']:.1f}%"
            bm_str = f"+{trade['chinext_pct']:.1f}%" if trade['chinext_pct'] > 0 else f"{trade['chinext_pct']:.1f}%"
            
            # Winner indicator
            winner = "✅" if trade['return_pct'] > trade['chinext_pct'] else "❌"
            
            print(f"| {trade['date']} | {trade['end_date']} | {trade['holdings']} | {ret_str} | {bm_str} {winner} |")
        
        print("\n### T20 换仓记录 (全周期)\n")
        print("| 换仓日期 | 结束日期 | 持仓示例 | 策略收益 | 同期创业板 |")
        print("|---------|---------|---------|----------|------------|")
        for trade in results['full']['t_results'][20]['trade_records'][-10:]:  # Last 10
            ret_str = f"+{trade['return_pct']:.1f}%" if trade['return_pct'] > 0 else f"{trade['return_pct']:.1f}%"
            bm_str = f"+{trade['chinext_pct']:.1f}%" if trade['chinext_pct'] > 0 else f"{trade['chinext_pct']:.1f}%"
            winner = "✅" if trade['return_pct'] > trade['chinext_pct'] else "❌"
            print(f"| {trade['date']} | {trade['end_date']} | {trade['holdings']} | {ret_str} | {bm_str} {winner} |")
    
    # 9. Save CSVs
    if 'full' in results:
        for T in [14, 20]:
            # Save trade records
            trades_df = pd.DataFrame(results['full']['t_results'][T]['trade_records'])
            output_path = os.path.join(config.DATA_OUTPUT_DIR, f"t{T}_trade_records.csv")
            trades_df.to_csv(output_path, index=False)
            print(f"Saved T{T} trade records to {output_path}")
    
    return results

if __name__ == "__main__":
    run_analysis()

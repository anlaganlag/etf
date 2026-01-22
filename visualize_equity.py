import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import os

def plot_equity_curve():
    # Load account history (simulation results)
    history_file = 'account_history.csv'
    if not os.path.exists(history_file):
        print(f"Error: {history_file} not found.")
        return
    
    df = pd.read_csv(history_file)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    
    # Calculate cumulative return
    initial_value = df['value'].iloc[0]
    df['cum_return'] = (df['value'] / initial_value - 1) * 100
    
    # Load benchmark if available (from backtest_results.csv)
    benchmark_file = 'backtest_results.csv'
    has_benchmark = False
    if os.path.exists(benchmark_file):
        bench_df = pd.read_csv(benchmark_file)
        # Use '日期' instead of 'date' for backtest_results.csv
        date_col = '日期' if '日期' in bench_df.columns else 'date'
        bench_df[date_col] = pd.to_datetime(bench_df[date_col])
        bench_df.set_index(date_col, inplace=True)
        # Strategy in backtest_results is theoretical, but we can plot Benchmark curve
        if 'Benchmark' in bench_df.columns:
            bench_cum_return = (bench_df['Benchmark'] / bench_df['Benchmark'].iloc[0] - 1) * 100
            has_benchmark = True

    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Plot Strategy
    ax.plot(df.index, df['cum_return'], label='Simulation Strategy (T10)', color='#1f77b4', linewidth=2.5, alpha=0.9)
    ax.fill_between(df.index, df['cum_return'], color='#1f77b4', alpha=0.1)
    
    # Plot Benchmark
    if has_benchmark:
        # Reindex benchmark to match strategy dates if necessary
        common_bench = bench_cum_return.reindex(df.index, method='ffill')
        ax.plot(common_bench.index, common_bench, label='Benchmark (HS300)', color='#ff7f0e', linewidth=2, linestyle='--', alpha=0.8)

    # Formatting
    ax.set_title('ETF Rotation Strategy: Cumulative Return (%)', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Return (%)', fontsize=12)
    
    # Add horizontal line at 0
    ax.axhline(0, color='black', linewidth=0.8, linestyle='-', alpha=0.3)
    
    # Format dates on X-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    plt.xticks(rotation=45)
    
    # Add latest value text
    final_ret = df['cum_return'].iloc[-1]
    ax.annotate(f'Final: {final_ret:.2f}%', 
                xy=(df.index[-1], final_ret), 
                xytext=(10, 0), 
                textcoords='offset points', 
                color='#1f77b4', 
                fontweight='bold',
                fontsize=12)

    ax.legend(fontsize=12, loc='upper left')
    plt.tight_layout()
    
    # Save the plot
    output_file = 'equity_curve.png'
    plt.savefig(output_file, dpi=150)
    print(f"Equity curve visualization saved to {output_file}")

if __name__ == "__main__":
    plot_equity_curve()

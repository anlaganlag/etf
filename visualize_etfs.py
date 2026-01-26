import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import numpy as np

def visualize():
    print("Creating visualizations for top_10_etfs.csv...")
    
    csv_path = 'output/data/top_10_etfs.csv'
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)
    
    # Ensure output directories exist
    os.makedirs('output/charts', exist_ok=True)
    os.makedirs('output/reports', exist_ok=True)

    # --- 1. Static Visualization (Matplotlib) ---
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
    
    # Sort for bar chart
    df_sorted = df.sort_values('total_score', ascending=True)
    colors = plt.cm.viridis(np.linspace(0, 1, len(df_sorted)))
    
    ax1.barh(df_sorted['etf_name'], df_sorted['total_score'], color=colors)
    ax1.set_title('Top 10 ETFs by Total Score', fontsize=15, pad=15)
    ax1.set_xlabel('Total Score')
    ax1.grid(axis='x', linestyle='--', alpha=0.7)

    # Momentum comparison
    periods = ['r1', 'r5', 'r10', 'r20']
    avail_periods = [p for p in periods if p in df.columns]
    
    if avail_periods:
        width = 0.2
        multiplier = 0
        x = np.arange(len(df['etf_name']))
        
        for p in avail_periods:
            offset = width * multiplier
            ax2.bar(x + offset, df[p], width, label=p)
            multiplier += 1
            
        ax2.set_title('Short to Medium Term Momentum Comparison (%)', fontsize=15, pad=15)
        ax2.set_xticks(x + width * (len(avail_periods)-1) / 2)
        ax2.set_xticklabels(df['etf_name'], rotation=45, ha='right')
        ax2.legend()
        ax2.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    static_img_path = 'output/charts/etf_analysis_summary.png'
    plt.savefig(static_img_path, dpi=300)
    print(f"Saved static summary to {static_img_path}")

    # --- 2. Interactive Dashboard (Plotly) ---
    fig_plotly = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Total Score Ranking", "Momentum Heatmap", "Short-term Momentum (1D & 5D)", "Long-term Momentum (60D+)"),
        vertical_spacing=0.15,
        horizontal_spacing=0.1,
        specs=[[{"type": "bar"}, {"type": "heatmap"}],
               [{"type": "bar"}, {"type": "bar"}]]
    )

    # 1. Total Score
    fig_plotly.add_trace(
        go.Bar(x=df['etf_name'], y=df['total_score'], name="Total Score", marker=dict(color=df['total_score'], colorscale='Plasma')),
        row=1, col=1
    )

    # 2. Heatmap
    all_periods = ['r1', 'r3', 'r5', 'r10', 'r20', 'r60', 'r120', 'r250']
    heatmap_periods = [p for p in all_periods if p in df.columns]
    z_data = df[heatmap_periods].fillna(0).values.T
    
    fig_plotly.add_trace(
        go.Heatmap(
            z=z_data,
            x=df['etf_name'],
            y=heatmap_periods,
            colorscale='RdBu',
            reversescale=True,
            zmid=0,
            name="Momentum Heatmap"
        ),
        row=1, col=2
    )

    # 3. Short term
    for p in ['r1', 'r5']:
        if p in df.columns:
            fig_plotly.add_trace(
                go.Bar(x=df['etf_name'], y=df[p], name=p),
                row=2, col=1
            )

    # 4. Long term
    for p in ['r60', 'r120', 'r250']:
        if p in df.columns:
            fig_plotly.add_trace(
                go.Bar(x=df['etf_name'], y=df[p], name=p),
                row=2, col=2
            )

    fig_plotly.update_layout(
        title_text="A-Share ETF Selection Strategy Analysis",
        template="plotly_dark",
        height=900,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    html_path = 'output/reports/etf_dashboard.html'
    fig_plotly.write_html(html_path)
    print(f"Saved interactive dashboard to {html_path}")

if __name__ == "__main__":
    visualize()

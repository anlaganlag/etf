import pandas as pd
import os

holdings_path = r'd:\antigravity\127\etf\216策略\持仓数据_20211202_20260123.csv'

if not os.path.exists(holdings_path):
    print(f"Error: File not found {holdings_path}")
    exit()

try:
    # Try reading with GBK or GB18030
    df = pd.read_csv(holdings_path, encoding='gb18030', encoding_errors='replace')
    
    # Identify date column (usually first column or named 'date'/'日期')
    date_col = None
    for col in df.columns:
        if 'date' in str(col).lower() or '日期' in str(col):
            date_col = col
            break
            
    if not date_col:
        # Fallback: assume first column
        date_col = df.columns[0]
        
    print(f"Using date column: {date_col}")
    
    # Group by date and count
    daily_counts = df.groupby(date_col).size()
    
    print("\n=== Holdings Analysis ===")
    print(f"Average Daily Holdings: {daily_counts.mean():.2f}")
    print(f"Median Daily Holdings:  {daily_counts.median():.2f}")
    print(f"Max Daily Holdings:     {daily_counts.max()}")
    print(f"Min Daily Holdings:     {daily_counts.min()}")
    
    print("\nSample (Last 5 days):")
    print(daily_counts.tail())

except Exception as e:
    print(f"Error: {e}")

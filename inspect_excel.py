
import pandas as pd
import os

file_path = 'ETF合并筛选结果.xlsx'
if os.path.exists(file_path):
    try:
        df = pd.read_excel(file_path)
        print("Columns:", df.columns.tolist())
        print("First 5 rows:")
        print(df.head())
        print("\nData Types:")
        print(df.dtypes)
    except Exception as e:
        print(f"Error reading excel: {e}")
else:
    print(f"File not found: {file_path}")

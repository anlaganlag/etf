
import pandas as pd
import os
from config import config

excel_path = os.path.join(config.BASE_DIR, "ETF合并筛选结果.xlsx")
df = pd.read_excel(excel_path)
print("Columns:", df.columns.tolist())
print("First 5 symbols:", df['symbol'].head().tolist())

import pandas as pd
import os

def inspect():
    path = "ETF合并筛选结果.xlsx"
    if os.path.exists(path):
        df = pd.read_excel(path)
        print("Columns:", df.columns.tolist())
        print("\nHead:")
        print(df.head(20))
        print("\nTheme distribution:")
        if '主题' in df.columns:
            print(df['主题'].value_counts())
        else:
            # check if renamed
            print("Column '主题' not found.")
        
        print("\nName counts (checking for similar names):")
        print(df['sec_name'].value_counts()[:20])

if __name__ == "__main__":
    inspect()

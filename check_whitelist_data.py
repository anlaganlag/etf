import os
import pandas as pd

excel_path = r'd:\antigravity\old_etf\etf\ETF合并筛选结果.xlsx'
df_excel = pd.read_excel(excel_path)
whitelist = df_excel['symbol'].tolist()

cache_dir = r'd:\antigravity\old_etf\etf\data_cache'

coverage = []
for code in whitelist:
    f = code.replace('.', '_') + '.csv'
    path = os.path.join(cache_dir, f)
    if os.path.exists(path):
        try:
            df = pd.read_csv(path)
            df['日期'] = pd.to_datetime(df['日期']).dt.tz_localize(None)
            coverage.append({
                'symbol': code,
                'start': df['日期'].min(),
                'end': df['日期'].max(),
                'len': len(df)
            })
        except:
            pass
    else:
        coverage.append({'symbol': code, 'start': None, 'end': None, 'len': 0})

cov_df = pd.DataFrame(coverage)
print(f"Whitelist size: {len(whitelist)}")
print(f"Files found: {len(cov_df[cov_df['len'] > 0])}")
print(f"Files starting pre-2022: {len(cov_df[cov_df['start'] <= '2022-01-01'])}")
print(f"Files ending in 2026: {len(cov_df[cov_df['end'] >= '2026-01-01'])}")

print("\n--- Whitelist Status ---")
print(cov_df.sort_values('len').head(20))

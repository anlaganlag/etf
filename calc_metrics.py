
import pandas as pd
import numpy as np

def calc_max_dd(series):
    vals = series.values
    if len(vals) == 0: return 0.0
    # Prepend 1.0 if not present, but here values are already indices
    peak = np.maximum.accumulate(vals)
    drawdowns = (vals - peak) / peak
    return np.min(drawdowns) * 100

def calc_return(series):
    if len(series) == 0: return 0.0
    return (series.iloc[-1] - 1.0) * 100

df = pd.read_csv('output/data/backtest_result.csv')

metrics = {}
for col in ['Value', 'CSI300', 'ChiNext']:
    if col in df.columns:
        ret = calc_return(df[col])
        mdd = calc_max_dd(df[col])
        metrics[col] = (ret, mdd)

print("Metrics:")
for name, (ret, mdd) in metrics.items():
    print(f"{name}: Return={ret:.2f}%, MaxDD={mdd:.2f}%")

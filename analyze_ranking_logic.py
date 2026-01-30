
import pandas as pd
import numpy as np
import os
from config import config

# --- Target Data (From 216 Strategy CSV) ---
# Date: 2021-12-03
TARGET_BUYS_1203 = [
    'SHSE.515220', 'SZSE.159930', 'SHSE.512710', 
    'SHSE.515210', 'SZSE.159745', 'SHSE.512040'
]

# Date: 2021-12-06
TARGET_BUYS_1206 = [
    'SHSE.515220', 'SHSE.512710', 'SZSE.159745',
    'SHSE.512040', 'SHSE.512200', 'SHSE.512690'
]

def load_data():
    price_data = {}
    files = [f for f in os.listdir(config.DATA_CACHE_DIR) if f.endswith('.csv') and (f.startswith('sh') or f.startswith('sz'))]
    
    # Load Whitelist from Excel to ensure we filter correctly
    excel_path = os.path.join(config.BASE_DIR, "ETF合并筛选结果.xlsx")
    df_excel = pd.read_excel(excel_path)
    whitelist = set(df_excel['symbol'])
    
    print(f"Loading data for {len(whitelist)} symbols...")
    
    for f in files:
        code = f.replace('_', '.').replace('.csv', '')
        if '.' not in code:
            code = ('SHSE.' if code.startswith('sh') else 'SZSE.') + code[2:]
            
        if code not in whitelist:
            continue
            
        try:
            df = pd.read_csv(os.path.join(config.DATA_CACHE_DIR, f), usecols=['日期', '收盘'])
            df['日期'] = pd.to_datetime(df['日期']).dt.tz_localize(None)
            price_data[code] = df.set_index('日期')['收盘']
        except: pass
        
    prices_df = pd.DataFrame(price_data).sort_index().ffill()
    return prices_df

def test_weights(prices_df, target_date, targets):
    current_dt = pd.to_datetime(target_date)
    history = prices_df[prices_df.index <= current_dt]
    
    if len(history) < 30: return
    
    last_row = history.iloc[-1]
    
    # Define weight combinations to test
    weight_configs = [
        {'name': 'Original', 'weights': {1: 20, 3: 30, 5: 50, 10: 70, 20: 100}},
        {'name': 'ShortTerm', 'weights': {1: 100, 3: 70, 5: 50, 10: 30, 20: 20}},
        {'name': 'Balanced', 'weights': {1: 50, 3: 50, 5: 50, 10: 50, 20: 50}},
        {'name': 'R20_Only', 'weights': {1: 0, 3: 0, 5: 0, 10: 0, 20: 100}},
        {'name': 'R1_Only', 'weights': {1: 100, 3: 0, 5: 0, 10: 0, 20: 0}},
        {'name': 'R3_Only', 'weights': {1: 0, 3: 100, 5: 0, 10: 0, 20: 0}},
        {'name': 'R5_Only', 'weights': {1: 0, 3: 0, 5: 100, 10: 0, 20: 0}},
        {'name': 'R10_Only', 'weights': {1: 0, 3: 0, 5: 0, 10: 100, 20: 0}},
        {'name': 'Reverse_Orig', 'weights': {1: 100, 3: 70, 5: 50, 10: 30, 20: 20}}, # Same as ShortTerm
        {'name': 'High_Short_Med_Long', 'weights': {1: 80, 3: 60, 5: 40, 10: 20, 20: 10}}, 
    ]
    
    # SCORING_METHOD = 'SMOOTH' # Assuming Smooth as per user context, or test both?
    # Let's test standard rank sum first (STEP like but simple sum of weighted ranks)
    # The 'SMOOTH' logic in script: (30-rank)/30 * pts.
    
    print(f"\n--- Testing Date: {target_date} ---")
    print(f"Targets({len(targets)}): {targets}")
    
    results = []
    
    for config in weight_configs:
        weights = config['weights']
        scores = pd.Series(0.0, index=history.columns)
        
        for p, w in weights.items():
            if len(history) <= p + 1: continue
            ret = (last_row / history.iloc[-(p+1)]) - 1
            # Simple weighted return ranking
            # Or replicate exact script logic
            
            # Script Logic (Approximation):
            ranks = ret.rank(ascending=False, method='min')
            # Assume SMOOTH logic:
            decay = (30 - ranks) / 30
            decay = decay.clip(lower=0)
            scores += decay * w
            
        top_n = scores.sort_values(ascending=False).head(20).index.tolist()
        
        # Calculate intersection
        matches = [t for t in targets if t in top_n]
        match_count = len(matches)
        
        # Find exact ranks of targets
        target_ranks = {}
        for t in targets:
            if t in scores.index:
                r = scores.rank(ascending=False)[t]
                target_ranks[t] = int(r)
            else:
                target_ranks[t] = -1
                
        results.append({
            'name': config['name'],
            'match_count_top20': match_count,
            'avg_target_rank': np.mean(list(target_ranks.values())),
            'target_ranks': target_ranks
        })

    # Sort by best correlation (lowest avg rank)
    results.sort(key=lambda x: x['avg_target_rank'])
    
    for r in results:
        print(f"Config: {r['name']:<15} | Avg Rank: {r['avg_target_rank']:.1f} | Matches(Top20): {r['match_count_top20']}")
        print(f"  Ranks: {r['target_ranks']}")

if __name__ == '__main__':
    df = load_data()
    test_weights(df, '2021-12-03', TARGET_BUYS_1203)
    test_weights(df, '2021-12-06', TARGET_BUYS_1206)

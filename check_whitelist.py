import pandas as pd
import os

whitelist_path = r'd:\antigravity\127\etf\ETF合并筛选结果.xlsx'

transactions_path = r'd:\antigravity\127\etf\216策略\交易数据_20211202_20260123.csv'

if not os.path.exists(transactions_path):
    print(f"Error: Transaction file not found at {transactions_path}")
    target_tickers = []
else:
    try:
        trans_df = pd.read_csv(transactions_path, encoding='gb18030', encoding_errors='replace') # Try GB18030 which is superset of GBK
        # Assuming symbol column is the 4th column (index 3) based on previous view_file
        # 2021-12-03, ..., ..., SHSE.515220, ...
        # Let's try to find a column with 'SHSE' or 'SZSE'
        
        target_tickers = set()
        for col in trans_df.columns:
            if trans_df[col].dtype == object:
                sample = trans_df[col].iloc[0] if len(trans_df) > 0 else ''
                if isinstance(sample, str) and ('SHSE.' in sample or 'SZSE.' in sample):
                    print(f"Found symbol column in transactions: {col}")
                    symbols = trans_df[col].unique()
                    # Extract 6 digits: SHSE.123456 -> 123456
                    for s in symbols:
                        if isinstance(s, str) and len(s) >= 6:
                            code = s.split('.')[-1]
                            target_tickers.add(code)
                    break
        
        target_tickers = sorted(list(target_tickers))
        print(f"Extracted {len(target_tickers)} unique tickers from transaction log.")
        
    except Exception as e:
        print(f"Error reading transactions: {e}")
        target_tickers = []

if not os.path.exists(whitelist_path):
    print(f"Error: File not found at {whitelist_path}")
else:
    try:
        df = pd.read_excel(whitelist_path)
        print(f"Loaded whitelist with {len(df)} rows.")
        
        # Assuming the code column is named 'code' or similar. Let's find it.
        code_col = 'sec_id' if 'sec_id' in df.columns else None
        
        if code_col:
            print(f"Found code column: {code_col}")
            whitelist_codes = df[code_col].astype(str).tolist()
            # Clean up codes (ensure 6 digits just in case)
            whitelist_codes = [c.split('.')[0].zfill(6) for c in whitelist_codes]
            
            print("\n--- Checking Target Tickers ---")
            found_count = 0
            for ticker in target_tickers:
                if ticker in whitelist_codes:
                    print(f"[OK] {ticker} is in whitelist.")
                    found_count += 1
                else:
                    print(f"[MISSING] {ticker} is NOT in whitelist.")
            
            print(f"\nSummary: Found {found_count}/{len(target_tickers)} targets.")
        else:
            print("Error: Could not identify 'Code' column in Excel.")
            print("Columns:", df.columns.tolist())
            
    except Exception as e:
        print(f"Error reading Excel: {e}")

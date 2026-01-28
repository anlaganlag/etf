import os
import pandas as pd
from config import config

def load_price_matrix(whitelist, cache_dir=config.DATA_CACHE_DIR):
    """
    高效加载价格矩阵，仅加载白名单和重要指数
    """
    print("Building price matrix from cache...", flush=True)
    price_data = {}
    files = [f for f in os.listdir(cache_dir) if f.endswith('.csv')]
    
    whitelist_codes = {c.lower().replace('.', '').replace('shse', '').replace('szse', '') for c in whitelist}
    important_indices = {'shse_000001.csv', 'szse_399006.csv', 'sh000001.csv', 'sz399006.csv'}
    
    for f in files:
        f_lower = f.lower()
        is_target = f_lower in important_indices
        
        if not is_target:
            # 简单的正则匹配提取代码
            import re
            m = re.search(r'(\d{6})', f_lower)
            if m and m.group(1) in whitelist_codes:
                is_target = True
                
        if is_target:
            code = f.replace('_', '.').replace('.csv', '')
            if '.' not in code:
                if code.startswith('sh'): code = 'SHSE.' + code[2:]
                elif code.startswith('sz'): code = 'SZSE.' + code[2:]
            try:
                df = pd.read_csv(os.path.join(cache_dir, f), usecols=['日期', '收盘'])
                # 极速解析日期
                df['日期'] = pd.to_datetime(df['日期']).dt.tz_localize(None)
                price_data[code] = df.set_index('日期')['收盘']
            except: pass

    # 对齐数据
    print("Aligning data into matrix...", flush=True)
    return pd.DataFrame(price_data).sort_index().ffill()

def load_etf_config():
    """读取Excel配置"""
    excel_path = os.path.join(config.BASE_DIR, "ETF合并筛选结果.xlsx")
    df = pd.read_excel(excel_path)
    df.columns = df.columns.str.strip()
    rename_map = {'symbol': 'etf_code', 'sec_name': 'etf_name', 'name_cleaned': 'theme'}
    df = df.rename(columns=rename_map)
    if 'theme' not in df.columns: df['theme'] = df['etf_name']
    return df

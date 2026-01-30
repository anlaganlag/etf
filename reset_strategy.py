from gm.api import *
import json
import os
from dotenv import load_dotenv
from config import config

# 加载环境变量
load_dotenv()

TOKEN = os.getenv('MY_QUANT_TGM_TOKEN')
STRATEGY_ID = 'd6d71d85-fb4c-11f0-99de-00ffda9d6e63'
STATE_FILE = "rolling_state_simple.json"

def init(context):
    print("Connecting to account...")
    # 获取账户信息
    cash = get_cash()
    
    # 在实盘/仿真中获取总资产 (NAV)
    equity = cash.get('nav', cash.get('available'))
    
    if equity is None or equity <= 0:
        print("Error: Could not retrieve account equity. Please check your token and strategy status.")
        os._exit(1)

    print(f"Total Account Equity (NAV): {equity:.2f}")
    
    tranche_count = 12
    initial_cash = equity / tranche_count
    
    tranches = []
    for i in range(tranche_count):
        tranches.append({
            "id": i,
            "cash": initial_cash,
            "holdings": {},
            "pos_records": {},
            "total_value": initial_cash,
            "guard_triggered_today": False
        })
        
    state = {
        "days_count": 0,
        "last_run_date": "",
        "initialized": True,
        "tranches": tranches
    }
    
    # 保存到配置文件目录下
    state_path = os.path.join(config.BASE_DIR, STATE_FILE)
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
        
    print(f"Successfully RESET strategy state at: {state_path}")
    print(f"Each of the {tranche_count} tranches initialized with {initial_cash:.2f} cash.")
    print("-" * 50)
    print("CRITICAL: Running the main strategy now will attempt to SYNC your broker account.")
    print("If you have existing positions, they will be SOLD to match the empty state.")
    print("-" * 50)
    print("Ready to start from Day 1 tomorrow (or today if you run the live script now).")
    os._exit(0)

if __name__ == '__main__':
    if not TOKEN:
        print("Error: MY_QUANT_TGM_TOKEN not found in .env file.")
    else:
        run(strategy_id=STRATEGY_ID, filename='reset_strategy.py', mode=MODE_LIVE, token=TOKEN)

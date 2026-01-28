"""
Integration Backtest Verification
运行短期回测，验证系统整体链路
"""
import os
import sys
from gm.api import run, MODE_BACKTEST, ADJUST_PREV
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 修改策略文件中的时间配置（通过替换变量）不需要，我们可以直接传给 run 函数
# 但是 gm_strategy_rolling.py 里的 init 会用到全局配置吗？
# 查看代码，init主要用 REBALANCE_PERIOD_T 等，而时间只在 __main__ 块里使用。
# 非常完美。

def run_verification():
    print("Starting Verification Backtest...")
    
    # 设定最近一个月的时间窗口
    START_TIME = '2024-12-01 09:00:00'
    END_TIME = '2025-01-27 16:00:00'
    
    # 获取 Token
    token = os.getenv('MY_QUANT_TGM_TOKEN')
    if not token:
        print("Error: Token not found in .env")
        return

    print(f"Token: {token[:6]}...")
    print(f"Time Range: {START_TIME} -> {END_TIME}")
    
    try:
        run(strategy_id='verification-test-001', 
            filename='gm_strategy_rolling.py',
            mode=MODE_BACKTEST,
            token=token,
            backtest_start_time=START_TIME,
            backtest_end_time=END_TIME,
            backtest_adjust=ADJUST_PREV,
            backtest_initial_cash=1000000,
            backtest_commission_ratio=0.0001,
            backtest_slippage_ratio=0.0001)
            
        print("\n✅ Backtest completed successfully without crash.")
        
    except Exception as e:
        print(f"\n❌ Backtest Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_verification()

# coding=utf-8
from __future__ import print_function, absolute_import
from gm.api import *
import pandas as pd
import os
from dotenv import load_dotenv

# 导入项目现有的逻辑
from src.etf_ranker import EtfRanker
from src.data_fetcher import DataFetcher
from config import config

# 加载环境变量中的Token
load_dotenv()

def init(context):
    # 1. 初始化数据抓取和排名器
    context.fetcher = DataFetcher(cache_dir=config.DATA_CACHE_DIR)
    context.ranker = EtfRanker(context.fetcher)
    
    # 2. 加载备选池并过滤跨境ETF
    excel_path = os.path.join(config.BASE_DIR, "ETF合并筛选结果.xlsx")
    df_excel = pd.read_excel(excel_path)
    candidates = df_excel.rename(columns={
        'symbol': 'etf_code',
        'sec_name': 'etf_name',
        '主题': 'theme'
    })
    
    # 定义跨境/海外ETF关键词进行排除
    exclude_kws = ['纳斯达克', '标普', '恒生', '日经', '德国', '港股', '海外', '国外', '纳指', '亚洲', '美国', '道琼斯']
    context.candidates = candidates[~candidates['etf_name'].str.contains('|'.join(exclude_kws), na=False)].copy()
    print(f"初始化完成。备选池已排除跨境ETF，剩余: {len(context.candidates)} 只。")
    
    # 3. 订阅基准数据作为时间驱动 (日线)
    subscribe(symbols='SHSE.000300', frequency='1d')
    
    # 4. 极致独狼策略参数配置 (冲刺 150%+)
    context.top_n = 1              # 独狼模式：只持仓全市场最强 1 只
    context.buffer_n = 2           # 缓冲区
    context.min_score = 150        # 严格门槛
    context.rebalance_period = 14  # 保持 14 天稳定性
    context.days_count = 0 
    context.target_percent = 0.95  # 全仓单押 95%

def on_bar(context, bars):
    context.days_count += 1
    current_date = context.now.strftime('%Y-%m-%d')
    
    # 周频检查
    if context.days_count % context.rebalance_period == 0:
        print(f"\n--- [{current_date}] 第 {context.days_count} 交易日，极致独狼巡检 ---")

        # 1. 获取全市场排名
        rank_df_full = context.ranker.select_top_etfs(
            context.candidates, 
            top_n=10, 
            reference_date=current_date
        )
        
        # 2. 筛选
        rank_df = rank_df_full[rank_df_full['total_score'] >= context.min_score]
        
        if rank_df.empty:
            print(f"  [避险] 评分太低，全仓退出。")
            top_new_candidates = []
            buffer_list = []
        else:
            all_list = rank_df['etf_code'].tolist()
            top_new_candidates = all_list[:context.top_n]
            buffer_list = all_list[:context.buffer_n] 
        
        # 3. 获取持仓
        positions = context.account().positions()
        current_holdings = [p['symbol'] for p in positions if p['amount'] > 0]
        
        final_targets = []
        # A. 检查现有持仓：只要在前2名就保留
        for symbol in current_holdings:
            if symbol in buffer_list:
                final_targets.append(symbol)
                print(f"  [守候] {symbol} 依然是最强王者之一，保留持仓")
        
        # B. 补足名额
        for symbol in top_new_candidates:
            if len(final_targets) >= context.top_n:
                break
            if symbol not in final_targets:
                final_targets.append(symbol)
                print(f"  [登顶] 换入全市场年度第一标的: {symbol}")

        # 4. 执行
        for symbol in current_holdings:
            if symbol not in final_targets:
                order_target_percent(symbol=symbol, percent=0, order_type=OrderType_Market, position_side=PositionSide_Long)
                print(f"  >>> 卖出: {symbol}")

        if final_targets:
            for symbol in final_targets:
                order_target_percent(symbol=symbol, percent=context.target_percent, order_type=OrderType_Market, position_side=PositionSide_Long)
                if symbol not in current_holdings:
                    print(f"  <<< 买入(全仓): {symbol}, 目标比例 {context.target_percent*100}%")

def on_backtest_finished(context, indicator):
    print("\n" + "="*50)
    print("极致集中策略 - 回测结果报告")
    print("="*50)
    try:
        # MyQuant standard keys
        pnl = indicator.get('pnl_ratio', 0) * 100
        ann_pnl = indicator.get('pnl_ratio_annual', 0) * 100
        mdd = indicator.get('max_drawdown', 0) * 100
        sharp = indicator.get('sharp_ratio', 0)
        
        print(f"累计收益率: {pnl:.2f}%")
        print(f"年化收益率: {ann_pnl:.2f}%")
        print(f"最大回撤: {mdd:.2f}%")
        print(f"夏普比率: {sharp:.2f}")
    except Exception as e:
        print(f"报告打印异常: {e}")
    print("="*50)

if __name__ == '__main__':
    TGM_TOKEN = os.getenv('MY_QUANT_TGM_TOKEN', '你的TOKEN')
    
    run(strategy_id='fe474f75-fa8e-11f0-b097-00ffda9d6e63', 
        filename='gm_backtest.py',
        mode=MODE_BACKTEST,
        token=TGM_TOKEN,
        backtest_start_time='2024-09-01 09:00:00',
        backtest_end_time='2026-01-20 16:00:00',
        backtest_adjust=ADJUST_PREV,
        backtest_initial_cash=1000000,
        backtest_commission_ratio=0.0001,
        backtest_slippage_ratio=0.0000,
        backtest_match_mode=1)

import pandas as pd


def verify_returns():
    print("开始验证收益计算...")

    # 1. 读取持仓明细 (计算每日持仓市值)
    # ---------------------------------------------------------
    file_holdings = 'daily_holdings_T14_rolling.csv'
    df_holdings = pd.read_csv(file_holdings)
    print(f"正在读取持仓文件: {file_holdings}")
    
    # 过滤掉名为 "CASH" 的行
    mask_stock = df_holdings['code'] != 'CASH'
    
    # 按日期汇总持仓市值
    daily_mv = df_holdings[mask_stock].groupby('date')['value'].sum()
    
    # 2. 读取交易记录 (获取每日剩余现金)
    # ---------------------------------------------------------
    file_trade = 'trade_log_T14_rolling.csv'
    df_trade = pd.read_csv(file_trade)
    print(f"正在读取交易文件: {file_trade}")
    
    # 既然一天可能多笔交易，我们只取每一天最后的一笔交易的 remaining_cash
    daily_cash = df_trade.drop_duplicates(subset=['date'], keep='last').set_index('date')['remaining_cash']
    
    # 3. 合并数据 (对齐日期)
    # ---------------------------------------------------------
    # 获取完整的日期范围 (取两个文件的日期并集)
    all_dates = sorted(list(set(daily_mv.index) | set(daily_cash.index)))
    df_result = pd.DataFrame(index=all_dates)
    
    # 填充市值 (如果没有持仓，市值为0)
    df_result['market_value'] = daily_mv
    df_result['market_value'] = df_result['market_value'].fillna(0)
    
    # 填充现金 (如果没有交易，现金维持前一日的水平)
    # 注意：daily_cash 只有交易日才有，非交易日现金不变，所以要 reindex 然后 ffill
    df_result['cash'] = daily_cash
    # 初始现金 1000万 (如果第一天没有交易记录，设为初始值)
    if pd.isna(df_result.iloc[0]['cash']):
        df_result.iloc[0, df_result.columns.get_loc('cash')] = 10_000_000.0  # 假设初始
    
    df_result['cash'] = df_result['cash'].ffill()
    
    # 4. 计算总资产
    # ---------------------------------------------------------
    df_result['total_asset'] = df_result['market_value'] + df_result['cash']
    
    # 初始资金
    initial_captial = 10_000_000.0
    
    # 5. 输出结果
    # ---------------------------------------------------------
    # 排除最后一天（可能是数据截断导致的异常）
    df_result = df_result.iloc[:-1]
    
    print("-" * 50)
    print(f"验证时间范围 (排除最后一天): {df_result.index[0]} 至 {df_result.index[-1]}")
    print("-" * 50)
    
    print("关键节点总资产:")
    print(df_result['total_asset'].head(5))
    print("...")
    print(df_result['total_asset'].tail(5))
    
    final_asset = df_result['total_asset'].iloc[-1]
    total_return = (final_asset - initial_captial) / initial_captial * 100
    
    print("-" * 50)
    print(f"初始资金: {initial_captial:,.2f}")
    print(f"最终资产: {final_asset:,.2f}")
    print(f"计算出的累计收益率: {total_return:.2f}%")
    print("-" * 50)
    
    # 检查是否有异常波动 (例如现金没接上导致资产腰斩)
    # 计算日收益率
    df_result['pct_change'] = df_result['total_asset'].pct_change()
    max_drop = df_result['pct_change'].min()
    if max_drop < -0.1:
        print(f"警告：检测到单日最大跌幅 {max_drop*100:.2f}%，可能数据合并有误（如现金未对齐）。")
    else:
        print("数据连续性检查通过，未发现异常断层。")

if __name__ == "__main__":
    verify_returns()

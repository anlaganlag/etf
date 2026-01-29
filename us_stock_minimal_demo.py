"""
美股动量轮动策略 - 最小可行性验证Demo
目标：验证yfinance数据质量 + 基础动量策略是否可行
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ============ 配置参数 ============
TOP_N = 5  # 持仓数量
REBALANCE_DAYS = 13  # 轮动周期（对应原策略的T值）
INITIAL_CASH = 100000  # 初始资金10万美元
START_DATE = '2021-01-01'
END_DATE = '2024-12-31'

# ============ Step 1: 获取股票池 ============
def get_stock_universe():
    """
    获取股票池：纳斯达克100成分股
    （比抓取成交量前100更稳定可靠）
    """
    # 方案1：手动定义纳指核心成分股（快速验证）
    # 包含科技、消费、医疗等主流板块
    stocks = [
        # 科技巨头
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA',
        # 芯片
        'AVGO', 'AMD', 'INTC', 'QCOM', 'MU', 'AMAT',
        # 软件/云
        'CRM', 'ADBE', 'ORCL', 'CSCO', 'INTU', 'NOW',
        # 消费/电商
        'COST', 'NFLX', 'CMCSA', 'PEP', 'TMUS',
        # 支付/金融科技
        'PYPL', 'ADP', 'ISRG',
        # 生物医药
        'AMGN', 'GILD', 'REGN', 'VRTX', 'MRNA',
        # 其他
        'HON', 'SBUX', 'LRCX', 'ADI', 'KLAC', 'SNPS', 'CDNS',
        'MRVL', 'ASML', 'PANW', 'ABNB', 'COIN', 'DDOG', 'ZS'
    ]

    # 方案2：从网页抓取（备选，需要额外依赖）
    # try:
    #     url = 'https://en.wikipedia.org/wiki/Nasdaq-100'
    #     tables = pd.read_html(url)
    #     stocks = tables[4]['Ticker'].tolist()  # 根据实际表格位置调整
    # except:
    #     pass

    print(f"✓ 股票池构建完成: {len(stocks)}只股票")
    return stocks


# ============ Step 2: 下载历史数据 ============
def download_price_data(symbols, start, end):
    """
    下载价格数据并缓存
    返回：DataFrame (日期 x 股票代码)
    """
    print(f"\n开始下载数据 ({start} 至 {end})...")
    print(f"预计耗时: {len(symbols) * 2 // 60} 分钟")

    all_data = {}
    failed = []

    for i, symbol in enumerate(symbols, 1):
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start, end=end, interval='1d')

            if len(df) < 100:  # 数据太少，跳过
                failed.append(symbol)
                continue

            all_data[symbol] = df['Close']

            if i % 10 == 0:
                print(f"  进度: {i}/{len(symbols)} ({i/len(symbols)*100:.1f}%)")

        except Exception as e:
            print(f"  ⚠️  {symbol} 下载失败: {e}")
            failed.append(symbol)

    if failed:
        print(f"\n⚠️  失败股票 ({len(failed)}): {failed[:5]}...")

    # 合并为DataFrame
    df = pd.DataFrame(all_data)
    df = df.fillna(method='ffill').dropna(how='all')  # 前向填充

    print(f"✓ 数据下载完成: {df.shape[1]}只股票, {df.shape[0]}个交易日")
    print(f"  日期范围: {df.index[0].date()} 至 {df.index[-1].date()}")

    return df


# ============ Step 3: 动量评分（简化版）============
def calculate_momentum_score(prices_df, as_of_date):
    """
    多周期动量评分（简化版，参考原策略）
    """
    history = prices_df[prices_df.index <= as_of_date]

    if len(history) < 251:  # 至少需要1年数据
        return None

    # 多周期收益率（类似原策略的periods_rule）
    periods = {
        '1d': 1,
        '1w': 5,
        '1m': 20,
        '3m': 60,
        '6m': 120
    }

    scores = pd.Series(0.0, index=history.columns)

    for name, days in periods.items():
        if len(history) <= days:
            continue

        # 计算收益率
        returns = (history.iloc[-1] / history.iloc[-(days+1)]) - 1

        # 排名打分（前20%满分，逐级递减）
        ranks = returns.rank(ascending=False, pct=True)

        # 权重：长期优先（与原策略相同）
        weight = days / 10
        scores += (ranks > 0.8) * weight  # 只给前20%打分

    return scores.sort_values(ascending=False)


# ============ Step 4: 简化回测引擎 ============
class SimpleBacktest:
    def __init__(self, prices_df, top_n=5, rebalance_days=13):
        self.prices = prices_df
        self.top_n = top_n
        self.rebalance_days = rebalance_days
        self.cash = INITIAL_CASH
        self.holdings = {}  # {symbol: shares}
        self.portfolio_values = []
        self.trade_dates = []

    def run(self):
        """执行回测"""
        trading_days = self.prices.index

        for i, date in enumerate(trading_days):
            # 每N天轮动一次
            if i % self.rebalance_days == 0:
                self.rebalance(date)

            # 计算当日组合价值
            daily_value = self.cash
            for symbol, shares in self.holdings.items():
                if symbol in self.prices.columns:
                    daily_value += shares * self.prices.loc[date, symbol]

            self.portfolio_values.append(daily_value)
            self.trade_dates.append(date)

        return self.get_performance()

    def rebalance(self, date):
        """轮动调仓"""
        # 1. 获取当日排名
        scores = calculate_momentum_score(self.prices, date)
        if scores is None:
            return

        # 2. 清仓
        for symbol in list(self.holdings.keys()):
            shares = self.holdings[symbol]
            price = self.prices.loc[date, symbol]
            self.cash += shares * price
        self.holdings = {}

        # 3. 买入TopN
        top_stocks = scores.head(self.top_n).index.tolist()
        allocation_per_stock = self.cash / self.top_n

        for symbol in top_stocks:
            if symbol not in self.prices.columns:
                continue
            price = self.prices.loc[date, symbol]
            shares = int(allocation_per_stock / price)

            if shares > 0:
                self.holdings[symbol] = shares
                self.cash -= shares * price

    def get_performance(self):
        """计算绩效指标"""
        values = pd.Series(self.portfolio_values, index=self.trade_dates)

        # 收益率
        total_return = (values.iloc[-1] / values.iloc[0] - 1) * 100

        # 年化收益
        years = (values.index[-1] - values.index[0]).days / 365.25
        annual_return = ((values.iloc[-1] / values.iloc[0]) ** (1/years) - 1) * 100

        # 最大回撤
        cummax = values.cummax()
        drawdown = (values - cummax) / cummax * 100
        max_dd = drawdown.min()

        # 夏普比率（简化版）
        daily_returns = values.pct_change().dropna()
        sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if daily_returns.std() > 0 else 0

        # 胜率
        win_rate = (daily_returns > 0).sum() / len(daily_returns) * 100

        return {
            '总收益率': f"{total_return:.2f}%",
            '年化收益': f"{annual_return:.2f}%",
            '最大回撤': f"{max_dd:.2f}%",
            '夏普比率': f"{sharpe:.2f}",
            '日胜率': f"{win_rate:.1f}%",
            '交易天数': len(values),
            '最终资金': f"${values.iloc[-1]:,.0f}",
            '起始资金': f"${values.iloc[0]:,.0f}"
        }


# ============ Step 5: 基准对比 ============
def get_benchmark_performance(start, end):
    """获取标普500和纳指100基准表现"""
    benchmarks = {
        'S&P 500': '^GSPC',
        'Nasdaq 100': '^NDX',
        'QQQ ETF': 'QQQ'  # 纳指ETF作为备选
    }

    results = {}
    for name, ticker in benchmarks.items():
        try:
            data = yf.Ticker(ticker).history(start=start, end=end, interval='1d')

            if len(data) < 10:
                results[name] = {'收益率': 'N/A', '最大回撤': 'N/A'}
                continue

            total_return = (data['Close'].iloc[-1] / data['Close'].iloc[0] - 1) * 100

            # 最大回撤
            cummax = data['Close'].cummax()
            drawdown = ((data['Close'] - cummax) / cummax * 100).min()

            # 年化收益
            years = (data.index[-1] - data.index[0]).days / 365.25
            annual_return = ((data['Close'].iloc[-1] / data['Close'].iloc[0]) ** (1/years) - 1) * 100

            results[name] = {
                '总收益': f"{total_return:.2f}%",
                '年化收益': f"{annual_return:.2f}%",
                '最大回撤': f"{drawdown:.2f}%"
            }
        except Exception as e:
            results[name] = {'收益率': f'Error: {str(e)[:30]}', '最大回撤': 'N/A'}

    return results


# ============ 主函数 ============
def main():
    print("=" * 60)
    print("美股动量轮动策略 - 最小Demo验证")
    print("=" * 60)

    # Step 1: 构建股票池
    symbols = get_stock_universe()

    # Step 2: 下载数据
    prices_df = download_price_data(symbols, START_DATE, END_DATE)

    if prices_df.empty:
        print("❌ 数据下载失败，程序退出")
        return

    # Step 3: 执行回测
    print(f"\n{'='*60}")
    print("开始回测...")
    print(f"参数: TopN={TOP_N}, 轮动周期={REBALANCE_DAYS}天")

    backtest = SimpleBacktest(prices_df, TOP_N, REBALANCE_DAYS)
    performance = backtest.run()

    # Step 4: 输出结果
    print(f"\n{'='*60}")
    print("策略表现:")
    print(f"{'='*60}")
    for key, value in performance.items():
        print(f"  {key:12s}: {value}")

    # Step 5: 对比基准
    print(f"\n{'='*60}")
    print("基准对比:")
    print(f"{'='*60}")
    benchmarks = get_benchmark_performance(START_DATE, END_DATE)
    for name, stats in benchmarks.items():
        print(f"  {name}:")
        for k, v in stats.items():
            print(f"    {k}: {v}")

    print(f"\n{'='*60}")
    print("✓ Demo执行完成")
    print(f"{'='*60}")

    # 数据质量检查
    print(f"\n数据质量报告:")
    print(f"  缺失值比例: {prices_df.isnull().sum().sum() / (prices_df.shape[0] * prices_df.shape[1]) * 100:.2f}%")
    print(f"  数据完整度: {(1 - prices_df.isnull().any(axis=1).sum() / len(prices_df)) * 100:.1f}%")

    # 保存价格数据到CSV（可选）
    cache_file = 'us_stock_prices_cache.csv'
    prices_df.to_csv(cache_file)
    print(f"\n✓ 价格数据已缓存至: {cache_file}")


if __name__ == '__main__':
    main()

# coding=utf-8
"""
参数优化测试：T值（重平衡周期）× top_n（持股数量）网格搜索
目标：找到最优参数组合，最大化收益/风险比
"""
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from config import config

class ParamOptimizer:
    def __init__(self):
        self.load_data()

    def load_data(self):
        """加载价格数据和白名单"""
        excel_path = os.path.join(config.BASE_DIR, "ETF合并筛选结果.xlsx")
        df_excel = pd.read_excel(excel_path)
        df_excel.columns = df_excel.columns.str.strip()
        rename_map = {'symbol': 'etf_code', 'sec_name': 'etf_name', 'name_cleaned': 'theme'}
        df_excel = df_excel.rename(columns=rename_map)
        if 'theme' not in df_excel.columns:
            df_excel['theme'] = df_excel['etf_name']
        self.whitelist = set(df_excel['etf_code'])
        self.theme_map = df_excel.set_index('etf_code')['theme'].to_dict()

        # 加载价格数据
        price_data = {}
        files = [f for f in os.listdir(config.DATA_CACHE_DIR)
                 if f.endswith('.csv') and (f.startswith('sh') or f.startswith('sz'))]
        for f in files:
            code = f.replace('_', '.').replace('.csv', '')
            if '.' not in code:
                if code.startswith('sh'): code = 'SHSE.' + code[2:]
                elif code.startswith('sz'): code = 'SZSE.' + code[2:]
            try:
                df = pd.read_csv(os.path.join(config.DATA_CACHE_DIR, f), usecols=['日期', '收盘'])
                df['日期'] = pd.to_datetime(df['日期']).dt.tz_localize(None)
                if not df.empty:
                    price_data[code] = df.set_index('日期')['收盘']
            except:
                pass
        self.prices_df = pd.DataFrame(price_data).sort_index().ffill()
        print(f"[数据加载] {self.prices_df.shape[1]} 个ETF, {self.prices_df.shape[0]} 个交易日")

    def get_ranking(self, current_dt, use_theme_boost=True):
        """5P评分+主题加成"""
        history_prices = self.prices_df[self.prices_df.index <= current_dt]
        if len(history_prices) < 251:
            return None, None

        # 5P基础评分
        periods_rule = {1: 100, 3: 70, 5: 50, 10: 30, 20: 20}
        threshold = 15
        base_scores = pd.Series(0.0, index=history_prices.columns)
        for p, pts in periods_rule.items():
            rets = (history_prices.iloc[-1] / history_prices.iloc[-(p+1)]) - 1
            ranks = rets.rank(ascending=False, method='min')
            base_scores += (ranks <= threshold) * pts

        valid_base = base_scores[base_scores.index.isin(self.whitelist)]

        # 主题加成
        if use_theme_boost:
            strong_etfs = valid_base[valid_base >= 150]
            theme_counts = {}
            for code in strong_etfs.index:
                t = self.theme_map.get(code, 'Unknown')
                theme_counts[t] = theme_counts.get(t, 0) + 1
            strong_themes = {t for t, count in theme_counts.items() if count >= 3}
            final_scores = valid_base.copy()
            for code in final_scores.index:
                if self.theme_map.get(code, 'Unknown') in strong_themes:
                    final_scores[code] += 50
        else:
            final_scores = valid_base

        return final_scores, base_scores

    def simulate_strategy(self, T, top_n, start_date, end_date,
                          stop_loss=0.20, trailing_trigger=0.10, trailing_drop=0.05):
        """
        回测单个参数组合

        Args:
            T: 重平衡周期（天）
            top_n: 持股数量
            start_date: 回测起始日期
            end_date: 回测结束日期

        Returns:
            dict: 包含收益率、夏普、最大回撤等指标
        """
        dates = self.prices_df.index[(self.prices_df.index >= start_date) &
                                      (self.prices_df.index <= end_date)]
        if len(dates) < 20:
            return None

        # 初始化
        cash = 1000000.0
        holdings = {}  # {symbol: shares}
        pos_records = {}  # {symbol: {'entry': price, 'high': price}}
        nav_history = []
        days_count = 0

        for current_dt in dates:
            today_prices = self.prices_df.loc[current_dt]

            # 1. 风控检查
            to_sell = []
            for sym, shares in list(holdings.items()):
                price = today_prices.get(sym, 0)
                if price <= 0:
                    continue
                if sym not in pos_records:
                    pos_records[sym] = {'entry': price, 'high': price}
                rec = pos_records[sym]
                rec['high'] = max(rec['high'], price)

                # 止损/止盈判断
                if price < rec['entry'] * (1 - stop_loss):
                    to_sell.append(sym)
                elif rec['high'] > rec['entry'] * (1 + trailing_trigger):
                    if price < rec['high'] * (1 - trailing_drop):
                        to_sell.append(sym)

            for sym in to_sell:
                cash += holdings[sym] * today_prices.get(sym, 0)
                del holdings[sym]
                if sym in pos_records:
                    del pos_records[sym]

            # 2. 定期重平衡
            if days_count % T == 0:
                # 清空所有持仓
                for sym, shares in list(holdings.items()):
                    cash += shares * today_prices.get(sym, 0)
                holdings = {}
                pos_records = {}

                # 重新选股
                ranking_scores, base_scores = self.get_ranking(current_dt)
                if ranking_scores is not None:
                    valid_scores = ranking_scores[ranking_scores >= 150]
                    if not valid_scores.empty:
                        df_rank = pd.DataFrame({
                            'score': valid_scores,
                            'theme': [self.theme_map.get(c, 'Unknown') for c in valid_scores.index]
                        })
                        df_rank = df_rank.sort_values('score', ascending=False)

                        # 主题去重选股
                        selected = []
                        seen_themes = set()
                        for sym, row in df_rank.iterrows():
                            if row['theme'] not in seen_themes:
                                selected.append(sym)
                                seen_themes.add(row['theme'])
                            if len(selected) >= top_n:
                                break

                        # 市场环境判断
                        strong_count = (base_scores >= 150).sum()
                        exposure = 1.0 if strong_count >= 5 else 0.3

                        # 建仓
                        if selected:
                            invest_amt = cash * 0.98 * exposure
                            per_amt = invest_amt / len(selected)
                            for sym in selected:
                                price = today_prices.get(sym, 0)
                                if price > 0:
                                    shares = int(per_amt / price / 100) * 100
                                    if shares > 0 and cash >= shares * price:
                                        cash -= shares * price
                                        holdings[sym] = shares
                                        pos_records[sym] = {'entry': price, 'high': price}

            # 计算当日净值
            total_value = cash + sum(shares * today_prices.get(sym, 0)
                                     for sym, shares in holdings.items())
            nav_history.append(total_value)
            days_count += 1

        # 计算指标
        nav_series = pd.Series(nav_history)
        total_return = (nav_series.iloc[-1] / nav_series.iloc[0] - 1) * 100

        # 最大回撤
        cummax = nav_series.cummax()
        drawdown = (nav_series - cummax) / cummax * 100
        max_drawdown = drawdown.min()

        # 年化收益和夏普
        daily_returns = nav_series.pct_change().dropna()
        annual_return = (1 + total_return/100) ** (252/len(nav_series)) - 1
        sharpe = np.sqrt(252) * daily_returns.mean() / daily_returns.std() if daily_returns.std() > 0 else 0

        return {
            'total_return': total_return,
            'max_drawdown': max_drawdown,
            'sharpe': sharpe,
            'annual_return': annual_return * 100,
            'final_nav': nav_series.iloc[-1],
            'nav_history': nav_history
        }

    def grid_search(self, start_date='2024-09-01', end_date='2026-01-23'):
        """
        网格搜索最优参数

        测试范围：
        - T: [7, 10, 14, 20, 30]
        - top_n: [2, 3, 5, 10, 15]
        """
        T_values = [7, 10, 14, 20, 30]
        top_n_values = [2, 3, 5, 10, 15]

        results = []
        total_tests = len(T_values) * len(top_n_values)
        current = 0

        print(f"\n{'='*60}")
        print(f"参数网格搜索：{total_tests}个组合")
        print(f"起始日期：{start_date}")
        print(f"结束日期：{end_date}")
        print(f"{'='*60}\n")

        for T in T_values:
            for top_n in top_n_values:
                current += 1
                print(f"[{current}/{total_tests}] 测试 T={T}, top_n={top_n}...", end=' ')

                result = self.simulate_strategy(
                    T=T,
                    top_n=top_n,
                    start_date=pd.Timestamp(start_date),
                    end_date=pd.Timestamp(end_date)
                )

                if result:
                    results.append({
                        'T': T,
                        'top_n': top_n,
                        'return': result['total_return'],
                        'max_dd': result['max_drawdown'],
                        'sharpe': result['sharpe'],
                        'annual_return': result['annual_return']
                    })
                    print(f"✓ 收益={result['total_return']:.1f}%, 回撤={result['max_drawdown']:.1f}%, 夏普={result['sharpe']:.2f}")
                else:
                    print("✗ 数据不足")

        return pd.DataFrame(results)

    def analyze_results(self, df_results):
        """分析并可视化结果"""
        print(f"\n{'='*60}")
        print("参数优化结果分析")
        print(f"{'='*60}\n")

        # 1. 按收益率排序
        print("【Top 5 最高收益率组合】")
        top5_return = df_results.nlargest(5, 'return')
        print(top5_return[['T', 'top_n', 'return', 'max_dd', 'sharpe']].to_string(index=False))

        # 2. 按夏普比率排序
        print("\n【Top 5 最高夏普比率组合】")
        top5_sharpe = df_results.nlargest(5, 'sharpe')
        print(top5_sharpe[['T', 'top_n', 'return', 'max_dd', 'sharpe']].to_string(index=False))

        # 3. 按回撤控制排序（最小回撤）
        print("\n【Top 5 最小回撤组合】")
        top5_dd = df_results.nsmallest(5, 'max_dd')
        print(top5_dd[['T', 'top_n', 'return', 'max_dd', 'sharpe']].to_string(index=False))

        # 4. 综合评分（收益/回撤比）
        df_results['score'] = df_results['return'] / abs(df_results['max_dd'])
        print("\n【Top 5 综合评分（收益/回撤比）】")
        top5_score = df_results.nlargest(5, 'score')
        print(top5_score[['T', 'top_n', 'return', 'max_dd', 'sharpe', 'score']].to_string(index=False))

        # 5. 热力图
        self.plot_heatmap(df_results)

        return df_results

    def plot_heatmap(self, df_results):
        """绘制参数热力图"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # 收益率热力图
        pivot_return = df_results.pivot(index='top_n', columns='T', values='return')
        im1 = axes[0].imshow(pivot_return.values, cmap='RdYlGn', aspect='auto')
        axes[0].set_xticks(range(len(pivot_return.columns)))
        axes[0].set_yticks(range(len(pivot_return.index)))
        axes[0].set_xticklabels(pivot_return.columns)
        axes[0].set_yticklabels(pivot_return.index)
        axes[0].set_xlabel('T (重平衡周期)')
        axes[0].set_ylabel('top_n (持股数量)')
        axes[0].set_title('总收益率 (%)')
        plt.colorbar(im1, ax=axes[0])

        # 在每个格子上标注数值
        for i in range(len(pivot_return.index)):
            for j in range(len(pivot_return.columns)):
                text = axes[0].text(j, i, f"{pivot_return.values[i, j]:.0f}",
                                   ha="center", va="center", color="black", fontsize=8)

        # 最大回撤热力图
        pivot_dd = df_results.pivot(index='top_n', columns='T', values='max_dd')
        im2 = axes[1].imshow(pivot_dd.values, cmap='RdYlGn_r', aspect='auto')
        axes[1].set_xticks(range(len(pivot_dd.columns)))
        axes[1].set_yticks(range(len(pivot_dd.index)))
        axes[1].set_xticklabels(pivot_dd.columns)
        axes[1].set_yticklabels(pivot_dd.index)
        axes[1].set_xlabel('T (重平衡周期)')
        axes[1].set_ylabel('top_n (持股数量)')
        axes[1].set_title('最大回撤 (%)')
        plt.colorbar(im2, ax=axes[1])

        for i in range(len(pivot_dd.index)):
            for j in range(len(pivot_dd.columns)):
                text = axes[1].text(j, i, f"{pivot_dd.values[i, j]:.0f}",
                                   ha="center", va="center", color="black", fontsize=8)

        # 夏普比率热力图
        pivot_sharpe = df_results.pivot(index='top_n', columns='T', values='sharpe')
        im3 = axes[2].imshow(pivot_sharpe.values, cmap='viridis', aspect='auto')
        axes[2].set_xticks(range(len(pivot_sharpe.columns)))
        axes[2].set_yticks(range(len(pivot_sharpe.index)))
        axes[2].set_xticklabels(pivot_sharpe.columns)
        axes[2].set_yticklabels(pivot_sharpe.index)
        axes[2].set_xlabel('T (重平衡周期)')
        axes[2].set_ylabel('top_n (持股数量)')
        axes[2].set_title('夏普比率')
        plt.colorbar(im3, ax=axes[2])

        for i in range(len(pivot_sharpe.index)):
            for j in range(len(pivot_sharpe.columns)):
                text = axes[2].text(j, i, f"{pivot_sharpe.values[i, j]:.2f}",
                                   ha="center", va="center", color="white", fontsize=8)

        plt.tight_layout()
        plt.savefig('parameter_optimization_heatmap.png', dpi=150)
        print("\n热力图已保存至 parameter_optimization_heatmap.png")

if __name__ == '__main__':
    # 运行优化
    optimizer = ParamOptimizer()
    df_results = optimizer.grid_search(start_date='2024-09-01', end_date='2026-01-23')

    # 分析结果
    optimizer.analyze_results(df_results)

    # 保存完整结果
    df_results.to_csv('parameter_optimization_results.csv', index=False)
    print("\n完整结果已保存至 parameter_optimization_results.csv")

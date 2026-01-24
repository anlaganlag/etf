"""
获取过去2年的ETF历史数据
"""
from src.data_fetcher import DataFetcher
from datetime import datetime, timedelta
import os

def fetch_2year_data():
    """获取所有ETF过去2年的历史数据"""
    print("=" * 60)
    print("开始获取ETF过去2年历史数据")
    print("=" * 60)

    # 初始化DataFetcher
    fetcher = DataFetcher(cache_dir="data_cache")

    # 设置时间范围：过去2年
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=730)).strftime("%Y-%m-%d")

    print(f"\n时间范围: {start_date} 至 {end_date}")
    print(f"共计: 730天 (约2年)\n")

    # 获取ETF列表
    print("步骤1: 获取ETF列表...")
    etf_list = fetcher.get_all_etfs()

    if etf_list.empty:
        print("❌ 无法获取ETF列表")
        return

    total_etfs = len(etf_list)
    print(f"✓ 共找到 {total_etfs} 个ETF\n")

    # 获取历史数据
    print("步骤2: 批量获取历史数据...")
    print("-" * 60)

    success_count = 0
    fail_count = 0

    for idx, row in etf_list.iterrows():
        code = row['etf_code']
        name = row['etf_name']

        # 显示进度
        if (idx + 1) % 50 == 0:
            print(f"进度: {idx + 1}/{total_etfs} ({(idx+1)/total_etfs*100:.1f}%) - "
                  f"成功: {success_count}, 失败: {fail_count}")

        # 获取历史数据
        try:
            df = fetcher.get_etf_daily_history(code, start_date, end_date)

            if df is not None and not df.empty and len(df) >= 10:
                success_count += 1
                # 每10个显示一次详细信息
                if success_count % 10 == 0:
                    print(f"  ✓ {code} ({name}): {len(df)} 条记录")
            else:
                fail_count += 1

        except Exception as e:
            fail_count += 1
            if fail_count <= 5:  # 只显示前5个错误
                print(f"  ✗ {code} ({name}): {str(e)[:50]}")

    # 最终统计
    print("\n" + "=" * 60)
    print("数据获取完成!")
    print("=" * 60)
    print(f"总ETF数量: {total_etfs}")
    print(f"成功获取: {success_count} ({success_count/total_etfs*100:.1f}%)")
    print(f"获取失败: {fail_count} ({fail_count/total_etfs*100:.1f}%)")

    # 检查缓存目录
    cache_files = [f for f in os.listdir("data_cache") if f.endswith('.csv') and f != f"etf_list_{datetime.now().strftime('%Y%m%d')}.csv"]
    print(f"\n缓存文件数: {len(cache_files)}")

    # 计算缓存总大小
    total_size = sum(os.path.getsize(os.path.join("data_cache", f)) for f in cache_files)
    print(f"缓存总大小: {total_size / 1024 / 1024:.2f} MB")

    print("\n✅ 所有数据已缓存到 data_cache/ 目录")

if __name__ == "__main__":
    fetch_2year_data()

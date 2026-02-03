"""
测试qstock获取概念板块数据的各种方法
"""
import warnings
warnings.filterwarnings('ignore')

import qstock as qs
import pandas as pd

print("=" * 60)
print("QSTOCK 概念板块数据获取测试")
print("=" * 60)

# 方法1: ths_index_name 获取概念板块名称列表
print("\n[1] ths_index_name('概念') - 获取概念板块名称列表")
try:
    names = qs.ths_index_name('概念')
    print(f"    ✓ 成功! 获取到 {len(names)} 个概念板块")
    print(f"    示例: {names[:10]}")
    METHOD_1_OK = True
except Exception as e:
    print(f"    ✗ 失败: {e}")
    METHOD_1_OK = False
    names = []

# 方法2: realtime_data('概念板块') 获取实时行情
print("\n[2] realtime_data('概念板块') - 获取实时行情")
try:
    df = qs.realtime_data('概念板块')
    if df is not None and not df.empty:
        print(f"    ✓ 成功! 获取到 {len(df)} 行")
        print(f"    列名: {df.columns.tolist()}")
        METHOD_2_OK = True
    else:
        print("    ✗ 返回空数据")
        METHOD_2_OK = False
except Exception as e:
    print(f"    ✗ 失败: {e}")
    METHOD_2_OK = False

# 方法3: ths_money('概念', n=1) 获取资金流数据
print("\n[3] ths_money('概念', n=1) - 获取概念资金流")
try:
    df = qs.ths_money('概念', n=1)
    if df is not None and not df.empty:
        print(f"    ✓ 成功! 获取到 {len(df)} 行")
        print(f"    列名: {df.columns.tolist()[:8]}")
        METHOD_3_OK = True
    else:
        print("    ✗ 返回空数据")
        METHOD_3_OK = False
except Exception as e:
    print(f"    ✗ 失败: {e}")
    METHOD_3_OK = False

# 方法4: north_money('概念', n) 北向资金增持概念
print("\n[4] north_money('概念', 5) - 北向资金增持概念")
try:
    df = qs.north_money('概念', 5)
    if df is not None and not df.empty:
        print(f"    ✓ 成功! 获取到 {len(df)} 行")
        print(f"    列名: {df.columns.tolist()[:8]}")
        print(df.head(5).to_string())
        METHOD_4_OK = True
    else:
        print("    ✗ 返回空数据")
        METHOD_4_OK = False
except Exception as e:
    print(f"    ✗ 失败: {e}")
    METHOD_4_OK = False

# 方法5: wencai 问财选股
print("\n[5] wencai - 问财查询热门概念")
try:
    # 查询涨幅榜的股票，提取它们所属的概念
    df = qs.wencai('涨幅前100')
    if df is not None and not df.empty:
        print(f"    ✓ 成功! 获取到 {len(df)} 行")
        print(f"    列名: {df.columns.tolist()[:6]}")
        METHOD_5_OK = True
    else:
        print("    ✗ 返回空数据")
        METHOD_5_OK = False
except Exception as e:
    print(f"    ✗ 失败: {e}")
    METHOD_5_OK = False

# 方法6: ths_index_data 获取单个概念的历史行情
print("\n[6] ths_index_data - 获取单个概念历史行情")
try:
    if names:
        test_name = names[0]  # 取第一个概念测试
        df = qs.ths_index_data(test_name)
        if df is not None and not df.empty:
            print(f"    ✓ 成功! '{test_name}' 获取到 {len(df)} 天数据")
            print(f"    列名: {df.columns.tolist()}")
            print(f"    最新: {df.tail(1).to_string()}")
            METHOD_6_OK = True
        else:
            print("    ✗ 返回空数据")
            METHOD_6_OK = False
except Exception as e:
    print(f"    ✗ 失败: {e}")
    METHOD_6_OK = False

# 总结
print("\n" + "=" * 60)
print("测试总结:")
print("=" * 60)
results = {
    "ths_index_name": METHOD_1_OK,
    "realtime_data": METHOD_2_OK if 'METHOD_2_OK' in dir() else False,
    "ths_money": METHOD_3_OK if 'METHOD_3_OK' in dir() else False,
    "north_money": METHOD_4_OK if 'METHOD_4_OK' in dir() else False,
    "wencai": METHOD_5_OK if 'METHOD_5_OK' in dir() else False,
    "ths_index_data": METHOD_6_OK if 'METHOD_6_OK' in dir() else False,
}

for name, ok in results.items():
    status = "✓ 可用" if ok else "✗ 不可用"
    print(f"  {name}: {status}")

# 推荐方案
print("\n" + "=" * 60)
print("推荐方案:")
print("=" * 60)
if METHOD_1_OK:
    print("1. 使用 qs.ths_index_name('概念') 获取概念板块列表")
if results.get("north_money"):
    print("2. 使用 qs.north_money('概念', 5) 获取北向资金热门概念")
if results.get("wencai"):
    print("3. 使用 qs.wencai() 配合问财查询获取热门主题")

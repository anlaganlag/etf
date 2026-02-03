"""
ThemeBooster 功能测试脚本

测试流程：
1. 从Excel加载ETF主题列表
2. 初始化ThemeBooster
3. 获取热门概念板块（akshare）
4. 使用魔塔LLM映射到ETF主题
5. 展示热门主题
"""

import pandas as pd
from config import config
from src.theme_booster import ThemeBooster
import os

def main():
    print("=" * 60)
    print("ThemeBooster 功能测试")
    print("=" * 60)
    
    # 1. 加载ETF主题
    excel_path = os.path.join(config.BASE_DIR, "ETF合并筛选结果.xlsx")
    df = pd.read_excel(excel_path)
    df.columns = df.columns.str.strip()
    
    if 'name_cleaned' in df.columns:
        etf_themes = df['name_cleaned'].unique().tolist()
    else:
        etf_themes = df['sec_name'].unique().tolist()
    
    print(f"\n[1] 加载ETF主题: {len(etf_themes)} 个唯一主题")
    print(f"    示例: {etf_themes[:10]}")
    
    # 2. 初始化ThemeBooster
    print("\n[2] 初始化ThemeBooster...")
    booster = ThemeBooster(
        etf_themes=etf_themes,
        top_n_concepts=15,
        boost_points=40
    )
    
    # 3. 获取热门概念
    print("\n[3] 获取热门概念板块（从akshare或fallback）...")
    concepts_df = booster.get_top_concepts()
    if not concepts_df.empty:
        print(f"    获取到 {len(concepts_df)} 个概念板块")
        print(concepts_df.head(10).to_string())
    
    # 4. 获取热门主题（包含LLM映射）
    print("\n[4] 映射到ETF主题（使用魔塔LLM）...")
    hot_themes = booster.get_hot_themes()
    
    print("\n" + "=" * 60)
    print("识别的热门ETF主题:")
    print("=" * 60)
    for theme in sorted(hot_themes):
        print(f"  ★ {theme}")
    
    print(f"\n共 {len(hot_themes)} 个热门主题将获得 +40 分加成")
    print("=" * 60)
    
    return hot_themes


if __name__ == "__main__":
    main()

"""
Theme Booster Module: 增强主题识别
使用同花顺/东财概念板块实时涨幅数据 + 魔塔LLM智能匹配

功能：
1. 从akshare获取当日概念板块涨幅排行
2. 使用LLM将概念板块名称映射到ETF主题
3. 为命中热门主题的ETF增加得分boost
"""

import os
import json
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Set, Optional
import hashlib

# 配置
CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data_cache", "theme_cache")
CONCEPT_CACHE_HOURS = 4  # 概念板块数据缓存时间
LLM_MAPPING_CACHE_DAYS = 7  # LLM映射结果缓存天数

# 魔塔API配置
MODELSCOPE_API_KEY = os.getenv("MODELSCOPE_API_KEY", "ms-e6dfc8b2-765c-4a45-ae9c-8bf98513ab36")
MODELSCOPE_BASE_URL = "https://api-inference.modelscope.cn/v1"


class ThemeBooster:
    """
    主题增强器：识别当前市场最热主题并给予ETF评分加成
    """
    
    def __init__(self, etf_themes: List[str], top_n_concepts: int = 20, boost_points: int = 30):
        """
        Args:
            etf_themes: ETF主题列表（来自Excel的name_cleaned列）
            top_n_concepts: 取涨幅最高的N个概念板块
            boost_points: 匹配热门主题时的加分
        """
        self.etf_themes = list(set(etf_themes))  # 去重
        self.top_n_concepts = top_n_concepts
        self.boost_points = boost_points
        
        # 确保缓存目录存在
        if not os.path.exists(CACHE_DIR):
            os.makedirs(CACHE_DIR)
        
        # 缓存的映射关系
        self._concept_to_theme_cache: Dict[str, str] = {}
        self._load_mapping_cache()
    
    def _load_mapping_cache(self):
        """加载LLM映射缓存"""
        cache_file = os.path.join(CACHE_DIR, "concept_theme_mapping.json")
        if os.path.exists(cache_file):
            try:
                # 检查缓存是否过期
                file_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
                if (datetime.now() - file_time).days < LLM_MAPPING_CACHE_DAYS:
                    with open(cache_file, 'r', encoding='utf-8') as f:
                        self._concept_to_theme_cache = json.load(f)
                    print(f"[ThemeBooster] Loaded {len(self._concept_to_theme_cache)} cached mappings")
            except Exception as e:
                print(f"[ThemeBooster] Failed to load mapping cache: {e}")
    
    def _save_mapping_cache(self):
        """保存LLM映射缓存"""
        cache_file = os.path.join(CACHE_DIR, "concept_theme_mapping.json")
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(self._concept_to_theme_cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[ThemeBooster] Failed to save mapping cache: {e}")
    
    def get_top_concepts(self, verbose: bool = True) -> pd.DataFrame:
        """
        获取同花顺概念板块涨幅排行（带缓存）
        数据源优先级: qstock > akshare > fallback
        
        Returns:
            DataFrame with columns: ['板块名称', '涨跌幅', ...]
        """
        cache_file = os.path.join(CACHE_DIR, f"concept_rank_{datetime.now().strftime('%Y%m%d')}.csv")
        
        # 检查缓存
        if os.path.exists(cache_file):
            file_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
            if (datetime.now() - file_time).total_seconds() < CONCEPT_CACHE_HOURS * 3600:
                if verbose: print(f"[ThemeBooster] Loading concept data from cache...")
                return pd.read_csv(cache_file)
        
        import time
        
        # ========== 方法1: qstock (首选，更稳定) ==========
        try:
            import qstock as qs
            import warnings
            warnings.filterwarnings('ignore')
            print(f"[ThemeBooster] Fetching concept data from qstock...")
            
            # qstock方法1: 北向资金增持概念板块（稳定，包含涨跌幅）
            try:
                df = qs.north_money('概念', 5)  # 5日北向资金增持
                if df is not None and not df.empty:
                    # 提取板块名称列
                    name_col = None
                    for col in ['板块', '名称', '概念']:
                        if col in df.columns:
                            name_col = col
                            break
                    if name_col is None:
                        name_col = df.columns[1]  # 通常第二列是名称
                    
                    # 如果有涨跌幅列，按涨幅排序
                    if '涨跌幅' in df.columns:
                        df = df.sort_values('涨跌幅', ascending=False)
                    
                    # 标准化列名
                    df = df.rename(columns={name_col: '板块名称'})
                    df = df.head(self.top_n_concepts)
                    df.to_csv(cache_file, index=False)
                    print(f"[ThemeBooster] Got {len(df)} top concepts from qstock.north_money")
                    return df
            except Exception as e:
                print(f"[ThemeBooster] qstock.north_money failed: {e}")
            
            # qstock方法2: 问财查询涨幅榜股票，提取所属概念
            try:
                df = qs.wencai('涨幅前50')
                if df is not None and not df.empty:
                    # 从涨幅榜股票中提取概念标签
                    concept_cols = [c for c in df.columns if '概念' in c or '所属' in c]
                    if concept_cols:
                        concepts = []
                        for col in concept_cols:
                            for val in df[col].dropna():
                                if isinstance(val, str):
                                    for c in val.split(';'):
                                        c = c.strip()
                                        if c and len(c) < 20:  # 过滤过长的字符串
                                            concepts.append(c)
                        
                        # 统计概念出现频率
                        from collections import Counter
                        concept_counts = Counter(concepts)
                        top_concepts = concept_counts.most_common(self.top_n_concepts)
                        
                        result_df = pd.DataFrame({
                            '板块名称': [c[0] for c in top_concepts],
                            '涨跌幅': [c[1] for c in top_concepts]  # 这里用频率代替涨幅
                        })
                        result_df.to_csv(cache_file, index=False)
                        print(f"[ThemeBooster] Got {len(result_df)} hot concepts from qstock.wencai")
                        return result_df
            except Exception as e:
                print(f"[ThemeBooster] qstock.wencai failed: {e}")
            
            # qstock方法3: 获取概念板块名称列表（作为备用）
            try:
                names = qs.ths_index_name('概念')
                if names:
                    # 只能获取名称，无法获取涨幅，返回前N个概念
                    result_df = pd.DataFrame({
                        '板块名称': names[:self.top_n_concepts],
                        '涨跌幅': [0.0] * min(len(names), self.top_n_concepts)
                    })
                    # 不缓存，因为没有涨跌幅信息
                    print(f"[ThemeBooster] Got {len(result_df)} concept names from qstock (no price data)")
                    return result_df
            except Exception as e:
                print(f"[ThemeBooster] qstock.ths_index_name failed: {e}")
                
        except ImportError:
            print("[ThemeBooster] qstock not installed. Run: pip install qstock")
        except Exception as e:
            print(f"[ThemeBooster] qstock failed: {e}")
        
        # ========== 方法2: akshare (备选) ==========
        max_retries = 2
        try:
            import akshare as ak
            print(f"[ThemeBooster] Trying akshare as fallback...")
            
            # akshare: 同花顺概念板块
            for attempt in range(max_retries):
                try:
                    df = ak.stock_board_concept_name_ths()
                    if df is not None and not df.empty:
                        if '涨跌幅' in df.columns:
                            df['涨跌幅'] = pd.to_numeric(df['涨跌幅'], errors='coerce')
                            df = df.sort_values('涨跌幅', ascending=False).head(self.top_n_concepts)
                            df.to_csv(cache_file, index=False)
                            print(f"[ThemeBooster] Got {len(df)} top concepts from akshare THS")
                            return df
                except Exception as e:
                    print(f"[ThemeBooster] akshare THS attempt {attempt+1} failed: {e}")
                    time.sleep(1)
            
            # akshare: 东方财富概念板块
            for attempt in range(max_retries):
                try:
                    df = ak.stock_board_concept_name_em()
                    if df is not None and not df.empty:
                        if '涨跌幅' in df.columns:
                            df['涨跌幅'] = pd.to_numeric(df['涨跌幅'], errors='coerce')
                            df = df.sort_values('涨跌幅', ascending=False).head(self.top_n_concepts)
                            df.to_csv(cache_file, index=False)
                            print(f"[ThemeBooster] Got {len(df)} top concepts from akshare EM")
                            return df
                except Exception as e:
                    print(f"[ThemeBooster] akshare EM attempt {attempt+1} failed: {e}")
                    time.sleep(1)
                    
        except ImportError:
            print("[ThemeBooster] akshare not installed")
        except Exception as e:
            print(f"[ThemeBooster] akshare failed: {e}")
        
        # ========== 方法3: Fallback ==========
        print("[ThemeBooster] All API methods failed, using fallback hot themes")
        return self._get_fallback_concepts()
    
    def _get_fallback_concepts(self) -> pd.DataFrame:
        """
        当所有API都失败时，返回预设的热门概念板块
        这些是近期市场持续热门的主题，作为降级方案
        """
        # 预设的当前热门概念（可根据市场情况定期手动更新）
        fallback_concepts = [
            {"板块名称": "人工智能", "涨跌幅": 3.5},
            {"板块名称": "机器人", "涨跌幅": 3.2},
            {"板块名称": "芯片", "涨跌幅": 2.8},
            {"板块名称": "算力", "涨跌幅": 2.5},
            {"板块名称": "消费电子", "涨跌幅": 2.3},
            {"板块名称": "新能源汽车", "涨跌幅": 2.0},
            {"板块名称": "光伏", "涨跌幅": 1.8},
            {"板块名称": "军工", "涨跌幅": 1.5},
            {"板块名称": "医药", "涨跌幅": 1.2},
            {"板块名称": "证券", "涨跌幅": 1.0},
        ]
        print(f"[ThemeBooster] Using {len(fallback_concepts)} fallback concepts")
        return pd.DataFrame(fallback_concepts[:self.top_n_concepts])
    
    def _call_llm_for_mapping(self, concept_names: List[str]) -> Dict[str, str]:
        """
        使用魔塔LLM批量将概念板块名称映射到ETF主题
        
        Args:
            concept_names: 概念板块名称列表
            
        Returns:
            Dict[concept_name, matched_etf_theme]
        """
        # 检查缓存命中
        unmapped = [c for c in concept_names if c not in self._concept_to_theme_cache]
        if not unmapped:
            return {c: self._concept_to_theme_cache.get(c, "") for c in concept_names}
        
        print(f"[ThemeBooster] Calling LLM to map {len(unmapped)} concepts...")
        
        try:
            from openai import OpenAI
            
            client = OpenAI(
                api_key=MODELSCOPE_API_KEY,
                base_url=MODELSCOPE_BASE_URL
            )
            
            # 构造prompt
            etf_themes_str = ", ".join(sorted(self.etf_themes))
            concepts_str = ", ".join(unmapped)
            
            prompt = f"""你是一个金融数据专家。我需要将同花顺概念板块名称映射到ETF主题名称。

ETF主题列表（只能从这里选择）：
{etf_themes_str}

需要映射的概念板块：
{concepts_str}

规则：
1. 每个概念板块只能映射到一个最相关的ETF主题
2. 如果没有相关主题，返回空字符串
3. 优先选择语义最接近的主题

请以JSON格式返回映射结果，格式如下：
{{"概念名1": "ETF主题1", "概念名2": "", ...}}

只返回JSON，不要其他解释。"""

            response = client.chat.completions.create(
                model="Qwen/Qwen2.5-72B-Instruct",  # 魔塔上的Qwen模型
                messages=[
                    {"role": "system", "content": "你是一个专业的金融数据分析师，擅长概念分类和语义匹配。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,  # 低温度，确保稳定输出
                max_tokens=2000
            )
            
            result_text = response.choices[0].message.content.strip()
            
            # 解析JSON
            # 处理可能的markdown代码块
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0]
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0]
            
            mapping = json.loads(result_text)
            
            # 验证映射结果（只保留有效的ETF主题）
            valid_themes = set(self.etf_themes)
            for concept, theme in mapping.items():
                if theme and theme in valid_themes:
                    self._concept_to_theme_cache[concept] = theme
                else:
                    self._concept_to_theme_cache[concept] = ""
            
            # 保存缓存
            self._save_mapping_cache()
            
            print(f"[ThemeBooster] LLM mapped {len([v for v in mapping.values() if v])} concepts successfully")
            
        except ImportError:
            print("[ThemeBooster] openai package not installed. Run: pip install openai")
        except Exception as e:
            print(f"[ThemeBooster] LLM mapping failed: {e}")
            # 失败时使用简单字符串匹配作为fallback
            for concept in unmapped:
                self._concept_to_theme_cache[concept] = self._simple_match(concept)
        
        return {c: self._concept_to_theme_cache.get(c, "") for c in concept_names}
    
    def _simple_match(self, concept: str) -> str:
        """
        简单字符串匹配作为LLM的备选方案
        """
        concept_lower = concept.lower()
        for theme in self.etf_themes:
            theme_lower = theme.lower()
            # 完全匹配或包含关系
            if theme_lower == concept_lower or theme_lower in concept_lower or concept_lower in theme_lower:
                return theme
        return ""
    
    def get_hot_themes(self, verbose: bool = True) -> Set[str]:
        """
        获取当前热门主题集合
        
        Returns:
            热门ETF主题名称的集合
        """
        # 1. 获取涨幅最高的概念板块
        concepts_df = self.get_top_concepts(verbose=verbose)
        if concepts_df.empty:
            if verbose: print("[ThemeBooster] No concept data available, using fallback")
            return set()
        
        # 2. 提取概念名称
        name_col = None
        for col in ['板块名称', '板块', '概念名称', '名称']:
            if col in concepts_df.columns:
                name_col = col
                break
        
        if name_col is None:
            # 尝试第一列
            name_col = concepts_df.columns[0]
        
        concept_names = concepts_df[name_col].tolist()[:self.top_n_concepts]
        
        # 3. LLM映射到ETF主题
        mapping = self._call_llm_for_mapping(concept_names)
        
        # 4. 返回映射成功的主题集合
        hot_themes = set()
        for concept, theme in mapping.items():
            if theme:
                hot_themes.add(theme)
                if verbose: print(f"  [Hot] {concept} -> {theme}")
        
        if verbose: print(f"[ThemeBooster] Identified {len(hot_themes)} hot themes: {hot_themes}")
        return hot_themes
    
    def boost_scores(self, scores: pd.Series, theme_map: Dict[str, str]) -> pd.Series:
        """
        为匹配热门主题的ETF增加得分
        
        Args:
            scores: ETF得分Series，index为ETF代码
            theme_map: ETF代码到主题的映射
            
        Returns:
            增强后的得分Series
        """
        hot_themes = self.get_hot_themes()
        if not hot_themes:
            return scores
        
        boosted = scores.copy()
        boost_count = 0
        
        for code in scores.index:
            etf_theme = theme_map.get(code, "")
            if etf_theme in hot_themes:
                boosted[code] += self.boost_points
                boost_count += 1
        
        print(f"[ThemeBooster] Boosted {boost_count} ETFs with +{self.boost_points} points")
        return boosted


# 便捷函数
def get_hot_themes_today(etf_themes: List[str], top_n: int = 20) -> Set[str]:
    """
    获取今日热门主题（便捷函数）
    
    Args:
        etf_themes: ETF主题列表
        top_n: 取涨幅最高的N个概念板块
        
    Returns:
        热门主题集合
    """
    booster = ThemeBooster(etf_themes, top_n_concepts=top_n)
    return booster.get_hot_themes()


if __name__ == "__main__":
    # 测试
    test_themes = [
        "人工智能", "机器人", "半导体芯片", "新能源", "光伏", 
        "军工", "医药卫生", "消费电子", "银行", "证券公司",
        "沪深300", "中证500", "创业板", "科技", "软件服务"
    ]
    
    print("=" * 60)
    print("ThemeBooster Test")
    print("=" * 60)
    
    booster = ThemeBooster(test_themes, top_n_concepts=15)
    hot = booster.get_hot_themes()
    
    print("\n" + "=" * 60)
    print(f"Hot Themes Today: {hot}")
    print("=" * 60)

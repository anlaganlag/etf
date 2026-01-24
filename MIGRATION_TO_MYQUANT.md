# 掘金数据源迁移完成报告

## 概述
已成功将ETF选择系统的数据源从 **AkShare + Baostock** 完全切换到 **掘金量化（MyQuant）**。

## 迁移日期
2026-01-24

## 主要修改

### 1. 依赖更新 (`requirements.txt`)
**之前**：使用 AkShare 及其大量依赖
```
akshare==1.18.19
beautifulsoup4==4.14.3
... (27个依赖包)
```

**现在**：精简为掘金SDK核心依赖
```
gm>=3.0.180
numpy<2.0
pandas<2.0
python-dateutil>=2.8.1
python-dotenv>=1.0.0
```

### 2. 数据获取类重构 (`src/data_fetcher.py`)

#### 导入变化
```python
# 之前
import akshare as ak
import baostock as bs

# 现在
from gm.api import *
from dotenv import load_dotenv
```

#### 认证机制
新增掘金Token认证：
```python
def __init__(self, ...):
    # 从环境变量加载Token
    token = os.getenv('MY_QUANT_TGM_TOKEN')
    set_token(token)
```

#### ETF列表获取
**之前**：使用AkShare Sina接口，返回格式 `sz159995`
```python
df = ak.fund_etf_category_sina(symbol="ETF基金")
```

**现在**：使用掘金API，返回格式 `SZSE.159995`
```python
df = get_instruments(
    exchanges='SZSE,SHSE',
    sec_types=[1, 2],  # 股票和基金
    fields='symbol,sec_name',
    df=True
)
# 过滤ETF代码（SHSE.51*/56*/58* 或 SZSE.15*）
```

#### 历史数据获取
**之前**：使用AkShare Sina历史接口
```python
df = ak.fund_etf_hist_sina(symbol=etf_code)
```

**现在**：使用掘金history API
```python
df = history(
    symbol=etf_code,
    frequency='1d',
    start_time=f'{start_date} 09:00:00',
    end_time=f'{end_date} 16:00:00',
    fields='eob,close',
    adjust=ADJUST_PREV,  # 前复权
    df=True
)
```

### 3. 时区处理
掘金API返回的时间戳带有时区信息（Asia/Shanghai），需要转换为不带时区的格式以匹配现有代码：
```python
# 移除时区信息
if df['日期'].dt.tz is not None:
    df['日期'] = df['日期'].dt.tz_localize(None)
```

### 4. 环境变量配置 (`.env`)
```
MY_QUANT_TGM_TOKEN=d5b1705aec86683911e5a220b8015454ef2739ec
MY_QUANT_ACCOUNT_ID=9eb08c0d-f87f-11f0-9597-00163e022aa6
```

## 数据格式对比

| 项目 | AkShare格式 | 掘金格式 |
|------|------------|---------|
| ETF代码 | `sz159995` | `SZSE.159995` |
| ETF代码 | `sh510300` | `SHSE.510300` |
| 时间字段 | 无时区 | 带时区（Asia/Shanghai） |
| ETF数量 | 1407个 | 1761个 |

## 优势对比

### 掘金数据源优势
✅ **数据质量更高**：专业量化平台，数据更准确
✅ **数据更全面**：1761个ETF vs 1407个
✅ **更新更及时**：实时数据更新
✅ **稳定性更好**：不依赖爬虫，官方API支持
✅ **依赖更少**：5个包 vs 27个包
✅ **支持更多功能**：支持实盘交易、回测等高级功能

### AkShare/Baostock劣势
❌ 依赖网页爬虫，容易失效
❌ 数据更新延迟
❌ 依赖包臃肿
❌ 稳定性差

## 测试结果

### 测试1：ETF列表获取
```
✓ Successfully fetched 1761 ETFs
Sample: SZSE.159630, 汇添富中证A100ETF
```

### 测试2：历史数据获取
```
✓ Successfully fetched 20 records for SHSE.510300
Latest close: 4.70
```

### 测试3：完整流程运行
```
✓ 成功筛选出Top 10 ETFs
✓ 相关性去重正常工作
✓ 输出文件正常生成
```

## 代码兼容性

### 无需修改的部分
- `src/etf_ranker.py` - 完全兼容，无需任何修改
- `config.py` - 无需修改
- `main.py` - 无需修改

### 缓存机制
保留了原有的文件缓存机制：
- ETF列表：`data_cache/etf_list_YYYYMMDD.csv`（12小时刷新）
- 历史数据：`data_cache/EXCHANGE_CODE.csv`（12小时刷新）

## 迁移步骤总结

1. ✅ 安装掘金SDK：`pip install gm python-dotenv`
2. ✅ 配置环境变量（.env文件）
3. ✅ 重构DataFetcher类
4. ✅ 处理数据格式差异
5. ✅ 修复时区问题
6. ✅ 清除旧缓存
7. ✅ 测试验证
8. ✅ 完整流程测试

## 使用说明

### 安装依赖
```bash
pip install -r requirements.txt
```

### 配置Token
确保 `.env` 文件包含：
```
MY_QUANT_TGM_TOKEN=your_token_here
MY_QUANT_ACCOUNT_ID=your_account_id_here
```

### 运行系统
```bash
python main.py
```

### 测试数据源
```bash
python test_myquant.py
```

## 注意事项

1. **Token安全**：`.env` 文件已加入 `.gitignore`，不会被提交到Git仓库
2. **数据限制**：单次history调用最多返回33,000条记录
3. **缓存机制**：保留了12小时缓存，减少API调用
4. **时区处理**：自动转换为不带时区的datetime

## 后续改进建议

1. 可以利用掘金的更多字段（成交量、换手率等）进一步优化选股策略
2. 可以接入掘金的实盘交易功能
3. 可以使用掘金的回测框架进行策略验证
4. 可以获取更多维度的数据（如分钟级数据）

## 结论

✅ **迁移成功！** 系统已完全切换到掘金数据源，运行稳定，数据质量更高。

import os

class Config:
    # Path Definitions
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_CACHE_DIR = os.path.join(BASE_DIR, "data_cache")
    OUTPUT_DIR = os.path.join(BASE_DIR, "output")
    DATA_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "data")
    REPORT_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "reports")
    CHART_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "charts")
    
    # Ensure directories exist
    @classmethod
    def ensure_dirs(cls):
        for path in [cls.DATA_CACHE_DIR, cls.DATA_OUTPUT_DIR, cls.REPORT_OUTPUT_DIR, cls.CHART_OUTPUT_DIR]:
            if not os.path.exists(path):
                os.makedirs(path, exist_ok=True)
    
    # Strategy Parameters
    # 板块多周期排名评分：进入前15名即得分
    SECTOR_TOP_N_THRESHOLD = 15  # 进入前15名即得分
    SECTOR_PERIOD_SCORES = {
        1: 50,    # r1: 提升短期响应
        3: 80,    # r3: 确认短线力度
        5: 150,   # r5: 核心爆发力 (关键周线)
        10: 150,  # r10: 核心波段起爆 (关键两周)
        20: 150,  # r20: 动量基准 (核心月线)
        60: 50,   # r60: 趋势支撑
        120: 20,  # r120: 极低权重
        250: 10,  # r250: 极低权重
    }
    
    # 每个行业板块最多持有的ETF数量 (极致分散建议设为 1)
    ETF_SECTOR_LIMIT = 1


config = Config()
config.ensure_dirs()

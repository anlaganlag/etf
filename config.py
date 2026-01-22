class Config:

    
    # Strategy Parameters
    # 板块多周期排名评分：进入前N名即得分
    SECTOR_TOP_N_THRESHOLD = 15  # 进入前15名即得分
    SECTOR_PERIOD_SCORES = {
        1: 100,   # r1: 1日涨幅前15名 +100分
        3: 70,    # r3: 3日涨幅前15名 +70分
        5: 50,    # r5: 5日涨幅前15名 +50分
        10: 30,   # r10: 10日涨幅前15名 +30分
        20: 20,   # r20: 20日涨幅前15名 +20分
    }
    
    # 每个行业板块最多持有的ETF数量 (极致分散建议设为 1)
    ETF_SECTOR_LIMIT = 1


config = Config()

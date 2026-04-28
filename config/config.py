import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# API配置
TUSHARE_TOKEN = os.getenv('TUSHARE_TOKEN', '')

# 数据配置
DATA_DIR = 'data'
HISTORY_DAYS = 180  # 历史数据天数
PREDICT_DAYS = 5    # 预测天数

# 模型配置
MODEL_DIR = 'src/models/saved'
FEATURES = ['open', 'high', 'low', 'close', 'volume', 'amount',
            'ma5', 'ma10', 'ma20', 'ma60',
            'macd', 'macd_signal', 'macd_hist',
            'rsi', 'kdj_k', 'kdj_d', 'kdj_j',
            'boll_upper', 'boll_mid', 'boll_lower',
            'cci', 'wr', 'bias']

# 回测配置
INITIAL_CAPITAL = 10000  # 初始资金
COMMISSION_RATE = 0.0005   # 佣金率（万分之五）
SLIPPAGE = 0.0001          # 滑点

# 预测配置
THRESHOLD_UP = 0.02   # 上涨阈值
THRESHOLD_DOWN = -0.02  # 下跌阈值
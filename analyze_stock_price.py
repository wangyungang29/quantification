#!/usr/bin/env python3
"""
分析股票价格走势，找出可能的买入和卖出点
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.data.data_fetcher import DataFetcher
from src.features.feature_extractor import FeatureExtractor

# 测试数据（用户指定的参数）
ts_code = "000564.SZ"  # 供销大集
start_date = "20250115"  # 使用过去的日期
end_date = "20250415"  # 使用过去的日期

# 获取股票历史数据
data_fetcher = DataFetcher()
df = data_fetcher.get_stock_history(ts_code, start_date, end_date)

# 计算技术指标
print(f"提取特征前的数据行数: {len(df)}")
feature_extractor = FeatureExtractor()
df = feature_extractor.extract_features(df)
print(f"提取特征后的数据行数: {len(df)}")

# 显示股票价格走势
print("股票价格走势:")
print(df[['date', 'close']])

# 计算每日收益率
df['daily_return'] = df['close'].pct_change()

# 显示每日收益率
print("\n每日收益率:")
print(df[['date', 'daily_return']])

# 计算累计收益率
df['cumulative_return'] = (1 + df['daily_return']).cumprod() - 1

# 显示累计收益率
print("\n累计收益率:")
print(df[['date', 'cumulative_return']])

# 找出上涨和下跌的天数
up_days = len(df[df['daily_return'] > 0])
down_days = len(df[df['daily_return'] < 0])
flat_days = len(df[df['daily_return'] == 0])

print(f"\n上涨天数: {up_days}")
print(f"下跌天数: {down_days}")
print(f"平盘天数: {flat_days}")
print(f"总天数: {len(df)}")

# 找出最大的单日涨幅和跌幅
max_gain = df['daily_return'].max()
max_loss = df['daily_return'].min()

print(f"\n最大单日涨幅: {max_gain:.2%}")
print(f"最大单日跌幅: {max_loss:.2%}")

# 计算平均每日收益率
avg_daily_return = df['daily_return'].mean()
print(f"\n平均每日收益率: {avg_daily_return:.2%}")

# 计算胜率
win_rate = up_days / (up_days + down_days) if (up_days + down_days) > 0 else 0
print(f"\n胜率: {win_rate:.2%}")
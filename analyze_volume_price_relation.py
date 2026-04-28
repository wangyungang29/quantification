#!/usr/bin/env python3
"""
分析成交量和价格的关系，优化交易策略
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
feature_extractor = FeatureExtractor()
df = feature_extractor.extract_features(df)

# 计算t-1的成交量和价格
if not df.empty:
    # 计算t-1的成交量
    df['volume_t-1'] = df['volume'].shift(1)
    # 计算t-1的收盘价
    df['close_t-1'] = df['close'].shift(1)
    # 计算t-1的开盘价
    df['open_t-1'] = df['open'].shift(1)
    # 计算t-1的最高价
    df['high_t-1'] = df['high'].shift(1)
    # 计算t-1的最低价
    df['low_t-1'] = df['low'].shift(1)
    
    # 计算成交量变化率
    df['volume_change'] = df['volume'].pct_change()
    # 计算价格变化率
    df['price_change'] = df['close'].pct_change()
    
    # 计算成交量与价格的相关性
    volume_price_corr = df['volume_change'].corr(df['price_change'])
    print(f"成交量变化率与价格变化率的相关性: {volume_price_corr:.2f}")
    
    # 分析成交量与价格的关系
    print("\n成交量与价格的关系分析:")
    
    # 1. 当成交量放大时，价格的变化
    volume_increase = df[df['volume_change'] > 0.1]  # 成交量增加10%以上
    if not volume_increase.empty:
        avg_price_change_volume_increase = volume_increase['price_change'].mean()
        print(f"成交量增加10%以上时，平均价格变化: {avg_price_change_volume_increase:.2%}")
    
    # 2. 当成交量缩小时，价格的变化
    volume_decrease = df[df['volume_change'] < -0.1]  # 成交量减少10%以上
    if not volume_decrease.empty:
        avg_price_change_volume_decrease = volume_decrease['price_change'].mean()
        print(f"成交量减少10%以上时，平均价格变化: {avg_price_change_volume_decrease:.2%}")
    
    # 3. 当成交量放大且价格上涨时的情况
    volume_increase_price_up = df[(df['volume_change'] > 0.1) & (df['price_change'] > 0)]
    if not volume_increase_price_up.empty:
        print(f"成交量增加10%以上且价格上涨的次数: {len(volume_increase_price_up)}")
    
    # 4. 当成交量放大且价格下跌时的情况
    volume_increase_price_down = df[(df['volume_change'] > 0.1) & (df['price_change'] < 0)]
    if not volume_increase_price_down.empty:
        print(f"成交量增加10%以上且价格下跌的次数: {len(volume_increase_price_down)}")
    
    # 5. 当成交量缩小且价格上涨时的情况
    volume_decrease_price_up = df[(df['volume_change'] < -0.1) & (df['price_change'] > 0)]
    if not volume_decrease_price_up.empty:
        print(f"成交量减少10%以上且价格上涨的次数: {len(volume_decrease_price_up)}")
    
    # 6. 当成交量缩小且价格下跌时的情况
    volume_decrease_price_down = df[(df['volume_change'] < -0.1) & (df['price_change'] < 0)]
    if not volume_decrease_price_down.empty:
        print(f"成交量减少10%以上且价格下跌的次数: {len(volume_decrease_price_down)}")
    
    # 7. 计算不同成交量区间的价格变化
    print("\n不同成交量区间的价格变化:")
    # 计算成交量的分位数
    volume_quantiles = df['volume_change'].quantile([0.25, 0.5, 0.75])
    print(f"成交量变化的25分位数: {volume_quantiles[0.25]:.2f}")
    print(f"成交量变化的50分位数: {volume_quantiles[0.5]:.2f}")
    print(f"成交量变化的75分位数: {volume_quantiles[0.75]:.2f}")
    
    # 低成交量变化区间
    low_volume_change = df[df['volume_change'] < volume_quantiles[0.25]]
    if not low_volume_change.empty:
        avg_price_change_low = low_volume_change['price_change'].mean()
        print(f"低成交量变化区间的平均价格变化: {avg_price_change_low:.2%}")
    
    # 中等成交量变化区间
    mid_volume_change = df[(df['volume_change'] >= volume_quantiles[0.25]) & (df['volume_change'] <= volume_quantiles[0.75])]
    if not mid_volume_change.empty:
        avg_price_change_mid = mid_volume_change['price_change'].mean()
        print(f"中等成交量变化区间的平均价格变化: {avg_price_change_mid:.2%}")
    
    # 高成交量变化区间
    high_volume_change = df[df['volume_change'] > volume_quantiles[0.75]]
    if not high_volume_change.empty:
        avg_price_change_high = high_volume_change['price_change'].mean()
        print(f"高成交量变化区间的平均价格变化: {avg_price_change_high:.2%}")
    
    # 8. 计算成交量与价格的滞后关系
    print("\n成交量与价格的滞后关系:")
    # 计算t-1的成交量变化与t的价格变化的相关性
    lag1_corr = df['volume_change'].shift(1).corr(df['price_change'])
    print(f"t-1的成交量变化与t的价格变化的相关性: {lag1_corr:.2f}")
    
    # 计算t-2的成交量变化与t的价格变化的相关性
    lag2_corr = df['volume_change'].shift(2).corr(df['price_change'])
    print(f"t-2的成交量变化与t的价格变化的相关性: {lag2_corr:.2f}")
    
    # 9. 分析成交量与价格的关系，为交易策略提供建议
    print("\n交易策略建议:")
    if volume_price_corr > 0:
        print("成交量与价格正相关，当成交量增加时，价格倾向于上涨")
    else:
        print("成交量与价格负相关，当成交量增加时，价格倾向于下跌")
    
    if not volume_increase.empty and avg_price_change_volume_increase > 0:
        print("当成交量增加时，价格倾向于上涨，可以考虑买入")
    else:
        print("当成交量增加时，价格倾向于下跌，可以考虑卖出")
    
    if not volume_decrease.empty and avg_price_change_volume_decrease > 0:
        print("当成交量减少时，价格倾向于上涨，可以考虑买入")
    else:
        print("当成交量减少时，价格倾向于下跌，可以考虑卖出")
else:
    print("无法获取历史数据")
#!/usr/bin/env python3
"""
测试tushare API是否正常工作
"""

import tushare as ts
from config.config import TUSHARE_TOKEN

# 初始化tushare
pro = ts.pro_api(TUSHARE_TOKEN)

# 测试获取股票列表
try:
    stock_list = pro.stock_basic(exchange='', list_status='L', fields='ts_code, symbol, name, area, industry, market, list_date')
    print(f"股票列表数量: {len(stock_list)}")
    print(stock_list.head())
except Exception as e:
    print(f"获取股票列表失败: {e}")

# 测试获取单只股票数据
ts_code = "000564.SZ"  # 供销大集
try:
    df = pro.daily(ts_code=ts_code, start_date='20250101', end_date='20250430')
    print(f"\n{ts_code} 历史数据行数: {len(df)}")
    print(df.head())
except Exception as e:
    print(f"\n获取股票 {ts_code} 历史数据失败: {e}")
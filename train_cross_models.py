#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""训练金叉和死叉预测模型"""

from src.models.model_trainer import ModelTrainer
import pandas as pd
import numpy as np
import tushare as ts
import datetime

# 初始化
trainer = ModelTrainer()

# 直接使用tushare获取数据（与API相同的方式）
TUSHARE_TOKEN = '68e431d47c0319d7bdea7cd1daf164392d22c8a9216d99476354be78'
pro = ts.pro_api(TUSHARE_TOKEN)

# 获取上证指数数据
ts_code = '000001.SH'
today = datetime.datetime.now().strftime('%Y%m%d')
start_date = (datetime.datetime.now() - datetime.timedelta(days=445)).strftime('%Y%m%d')

print(f"正在获取股票 {ts_code} 的数据，时间范围：{start_date} 到 {today}")
df = pro.daily(ts_code=ts_code, start_date=start_date, end_date=today)

if df.empty:
    print(f"无法获取股票 {ts_code} 的数据")
else:
    print(f'获取到 {len(df)} 条数据')
    
    # 按日期排序
    df = df.sort_values('trade_date').reset_index(drop=True)
    
    # 重命名列
    df.rename(columns={
        'trade_date': 'date',
        'open': 'open',
        'high': 'high',
        'low': 'low',
        'close': 'close',
        'vol': 'volume',
        'amount': 'amount'
    }, inplace=True)
    
    # 转换日期格式
    df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
    
    # 训练金叉预测模型
    print('训练金叉预测模型...')
    try:
        trainer.train_golden_cross_model(df, model_type='xgboost')
        print('金叉预测模型训练完成')
    except Exception as e:
        print(f'金叉预测模型训练失败: {e}')
    
    # 训练死叉预测模型
    print('训练死叉预测模型...')
    try:
        trainer.train_death_cross_model(df, model_type='xgboost')
        print('死叉预测模型训练完成')
    except Exception as e:
        print(f'死叉预测模型训练失败: {e}')
    
    print('模型训练完成')
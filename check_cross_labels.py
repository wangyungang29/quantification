import pandas as pd
from src.data.data_fetcher import DataFetcher
from src.models.model_trainer import ModelTrainer

# 获取数据
fetcher = DataFetcher()
df = fetcher.get_stock_history('000564.SZ', '20200101', '20260409')
print('数据量:', len(df))

# 创建金叉标签
trainer = ModelTrainer()
df = trainer.create_golden_cross_label(df)

# 检查金叉标签分布
print('金叉标签分布:')
print(df['Y_golden_cross'].value_counts())
print('金叉数量:', df['Y_golden_cross'].sum())
print('总数据量:', len(df))
print('金叉比例:', df['Y_golden_cross'].sum() / len(df))

# 检查死叉标签分布
df = trainer.create_death_cross_label(df)
print('\n死叉标签分布:')
print(df['Y_death_cross'].value_counts())
print('死叉数量:', df['Y_death_cross'].sum())
print('总数据量:', len(df))
print('死叉比例:', df['Y_death_cross'].sum() / len(df))
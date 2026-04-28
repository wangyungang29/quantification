import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 模拟股票数据
np.random.seed(42)
dates = pd.date_range('2026-01-01', '2026-04-15')
close_prices = np.random.normal(2.5, 0.1, len(dates))
df = pd.DataFrame({'date': dates, 'close': close_prices})

# 计算移动平均线
df['ma5'] = df['close'].rolling(window=5).mean()
df['ma20'] = df['close'].rolling(window=20).mean()

# 计算金叉死叉
df['golden_cross'] = 0.0
df['death_cross'] = 0.0
for i in range(1, len(df)):
    try:
        if not pd.isna(df['ma5'].iloc[i]) and not pd.isna(df['ma20'].iloc[i]) and not pd.isna(df['ma5'].iloc[i-1]) and not pd.isna(df['ma20'].iloc[i-1]):
            if df['ma5'].iloc[i] > df['ma20'].iloc[i] and df['ma5'].iloc[i-1] <= df['ma20'].iloc[i-1]:
                df.loc[df.index[i], 'golden_cross'] = 1.0
            if df['ma5'].iloc[i] < df['ma20'].iloc[i] and df['ma5'].iloc[i-1] >= df['ma20'].iloc[i-1]:
                df.loc[df.index[i], 'death_cross'] = 1.0
    except Exception as e:
        print(f"计算金叉死叉时出错: {e}")
        continue

# 打印结果
print("金叉死叉数据统计:")
print(f"金叉次数: {df['golden_cross'].sum()}")
print(f"死叉次数: {df['death_cross'].sum()}")
print("\n最近10天数据:")
print(df.tail(10)[['date', 'close', 'ma5', 'ma20', 'golden_cross', 'death_cross']])

# 可视化
plt.figure(figsize=(15, 6))
plt.plot(df['date'], df['close'], label='收盘价')
plt.plot(df['date'], df['ma5'], label='MA5')
plt.plot(df['date'], df['ma20'], label='MA20')

# 标记金叉死叉
golden_cross_dates = df[df['golden_cross'] == 1.0]['date']
golden_cross_prices = df[df['golden_cross'] == 1.0]['close']
death_cross_dates = df[df['death_cross'] == 1.0]['date']
death_cross_prices = df[df['death_cross'] == 1.0]['close']

plt.scatter(golden_cross_dates, golden_cross_prices, color='green', marker='^', s=100, label='金叉')
plt.scatter(death_cross_dates, death_cross_prices, color='red', marker='v', s=100, label='死叉')

plt.title('股价走势与金叉死叉')
plt.xlabel('日期')
plt.ylabel('价格')
plt.legend()
plt.grid(True)
plt.show()
import tushare as ts
import pandas as pd
from src.models.model_trainer import ModelTrainer

TUSHARE_TOKEN = '68e431d47c0319d7bdea7cd1daf164392d22c8a9216d99476354be78'

pro = ts.pro_api(TUSHARE_TOKEN)
df = pro.daily(ts_code='000564.SZ', start_date='20251001', end_date='20260415')
df = df.sort_values('trade_date').reset_index(drop=True)
df.rename(columns={'trade_date': 'date', 'vol': 'volume'}, inplace=True)
df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
df = df.set_index('date')

trainer = ModelTrainer()
features, _ = trainer.create_features(df)

print("2026-03-23 到 2026-03-27 的数据:")
for date in ['2026-03-23', '2026-03-24', '2026-03-25', '2026-03-26', '2026-03-27']:
    if date in features.index:
        row = features.loc[date]
        print(f"{date}: 收盘价={row['close']:.2f}, RSI={row['rsi']:.1f}, 量比={row['vol/Vol_MA5']:.2f}, 连跌={row['consecutive_down']}, 连涨={row['consecutive_up']}")

print("\n2026-03-23 的买入条件检查:")
if '2026-03-23' in features.index:
    row = features.loc['2026-03-23']
    buy_condition = row['rsi'] < 40 and row['vol/Vol_MA5'] > 1.2 and row['consecutive_down'] >= 2
    print(f"RSI < 40: {row['rsi']:.1f} < 40 = {row['rsi'] < 40}")
    print(f"量比 > 1.2: {row['vol/Vol_MA5']:.2f} > 1.2 = {row['vol/Vol_MA5'] > 1.2}")
    print(f"连跌 >= 2: {row['consecutive_down']:.0f} >= 2 = {row['consecutive_down'] >= 2}")
    print(f"买入条件: {buy_condition}")

print("\n2026-03-25 的卖出条件检查:")
if '2026-03-25' in features.index:
    row = features.loc['2026-03-25']
    print(f"持仓: 假设有持仓")
    print(f"RSI > 60: {row['rsi']:.1f} > 60 = {row['rsi'] > 60}")
    print(f"连涨 >= 2: {row['consecutive_up']:.0f} >= 2 = {row['consecutive_up'] >= 2}")
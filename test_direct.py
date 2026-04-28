import tushare as ts
import pandas as pd
from src.models.model_trainer import ModelTrainer

TUSHARE_TOKEN = '68e431d47c0319d7bdea7cd1daf164392d22c8a9216d99476354be78'

# 获取数据
pro = ts.pro_api(TUSHARE_TOKEN)
df = pro.daily(ts_code='000564.SZ', start_date='20251001', end_date='20260415')
df = df.sort_values('trade_date').reset_index(drop=True)
df.rename(columns={'trade_date': 'date', 'vol': 'volume'}, inplace=True)
df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
df = df.set_index('date')

# 计算特征
trainer = ModelTrainer()
features, _ = trainer.create_features(df)

# 回测策略（与数据分析脚本一致）
capital = 10000
shares = 0
buy_price = 0
trades = []
total_profit = 0

print("回测综合策略:")
for i in range(1, len(features)):
    row = features.iloc[i]
    date = features.index[i]
    
    # 买入条件
    buy_condition = row['rsi'] < 40 and row['vol/Vol_MA5'] > 1.2 and row['consecutive_down'] >= 2
    
    # 卖出条件
    sell_condition = False
    if shares > 0:
        if row['rsi'] > 60 or row['consecutive_up'] >= 2:
            sell_condition = True
    
    if shares == 0 and buy_condition:
        buy_price = row['close']
        shares = capital / buy_price
        capital = 0
        trades.append({'date': date, 'action': '买入', 'price': buy_price, 'shares': shares})
        print(f"{date.date()}: 买入，价格={buy_price:.2f}")
    elif shares > 0 and sell_condition:
        sell_price = row['close']
        capital = shares * sell_price
        profit = (sell_price - buy_price) / buy_price * 100
        total_profit += profit
        trades.append({'date': date, 'action': '卖出', 'price': sell_price, 'shares': shares, 'profit': profit})
        print(f"{date.date()}: 卖出，价格={sell_price:.2f}，收益={profit:.2f}%")
        shares = 0

# 最后一天卖出
if shares > 0:
    sell_price = features['close'].iloc[-1]
    capital = shares * sell_price
    profit = (sell_price - buy_price) / buy_price * 100
    total_profit += profit
    trades.append({'date': features.index[-1], 'action': '卖出', 'price': sell_price, 'shares': shares, 'profit': profit})
    print(f"{features.index[-1].date()}: 最后一天卖出，价格={sell_price:.2f}，收益={profit:.2f}%")

total_return = (capital - 10000) / 100 * 100
win_rate = len([t for t in trades if t.get('profit', 0) > 0]) / len(trades) if trades else 0

print(f"\n策略总收益率: {total_return:.2f}%")
print(f"交易次数: {len(trades) // 2}")
print(f"胜率: {win_rate:.2f}")
print(f"最终资金: {capital:.2f}")
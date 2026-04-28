import pandas as pd
import numpy as np
import tushare as ts
from datetime import datetime

TUSHARE_TOKEN = '68e431d47c0319d7bdea7cd1daf164392d22c8a9216d99476354be78'
pro = ts.pro_api(TUSHARE_TOKEN)

def calculate_indicators(df):
    """计算技术指标"""
    df['ma5'] = df['close'].rolling(window=5).mean()
    df['ma20'] = df['close'].rolling(window=20).mean()

    def calculate_rsi(series, period=14):
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
        rs = avg_gain / (avg_loss + 1e-8)
        return 100 - (100 / (1 + rs))

    df['rsi'] = calculate_rsi(df['close'])

    def calculate_macd(close_series, fast=12, slow=26, signal=9):
        ema_fast = close_series.ewm(span=fast, adjust=False).mean()
        ema_slow = close_series.ewm(span=slow, adjust=False).mean()
        dif = ema_fast - ema_slow
        dea = dif.ewm(span=signal, adjust=False).mean()
        macd_hist = (dif - dea) * 2
        return dif, dea, macd_hist

    df['macd'], df['macd_signal'], df['macd_hist'] = calculate_macd(df['close'])

    df['vol_ma5'] = df['volume'].rolling(window=5).mean()
    df['volume_ratio'] = df['volume'] / df['vol_ma5']

    df['pct_chg'] = df['close'].pct_change() * 100
    df['consecutive_up'] = 0
    df['consecutive_down'] = 0

    for i in range(1, len(df)):
        if df['pct_chg'].iloc[i] > 0:
            df.loc[df.index[i], 'consecutive_up'] = df['consecutive_up'].iloc[i-1] + 1
            df.loc[df.index[i], 'consecutive_down'] = 0
        elif df['pct_chg'].iloc[i] < 0:
            df.loc[df.index[i], 'consecutive_down'] = df['consecutive_down'].iloc[i-1] + 1
            df.loc[df.index[i], 'consecutive_up'] = 0
        else:
            df.loc[df.index[i], 'consecutive_up'] = 0
            df.loc[df.index[i], 'consecutive_down'] = 0

    return df

print("获取数据...")
df = pro.daily(ts_code='000564.SZ', start_date='20260101', end_date='20260415')
print(f"数据列: {df.columns.tolist()}")
df = df.sort_values('trade_date').reset_index(drop=True)
df.rename(columns={'trade_date': 'date', 'vol': 'volume'}, inplace=True)
df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')

print("计算指标...")
df = calculate_indicators(df)

print(f"\n数据统计:")
print(f"数据长度: {len(df)} 天")
print(f"时间范围: {df['date'].min()} 到 {df['date'].max()}")
print(f"收盘价范围: {df['close'].min():.2f} - {df['close'].max():.2f}")

print("\n技术指标分析:")
print(f"  RSI范围: {df['rsi'].min():.2f} - {df['rsi'].max():.2f}")
print(f"  量比范围: {df['volume_ratio'].min():.2f} - {df['volume_ratio'].max():.2f}")
print(f"  连续下跌范围: {df['consecutive_down'].min()} - {df['consecutive_down'].max()}")

print("\n查看满足买入条件的数据点:")
for i, row in df.iterrows():
    condition_rsi = (row['rsi'] < 40)
    condition_volume = (row['volume_ratio'] > 1.2)
    condition_down = (row['consecutive_down'] >= 2)
    conditions_met = condition_rsi + condition_volume + condition_down

    if conditions_met >= 2:
        print(f"{row['date']}: RSI={row['rsi']:.1f}, 量比={row['volume_ratio']:.2f}, 连跌={row['consecutive_down']}天, 满足条件数={conditions_met}")

print("\n回测（完全匹配数据分析脚本逻辑）:")
capital = 10000
shares = 0
buy_price = 0
total_trades = 0
won_trades = 0

for i, row in df.iterrows():
    condition_rsi = (row['rsi'] < 40)
    condition_volume = (row['volume_ratio'] > 1.2)
    condition_down = (row['consecutive_down'] >= 2)
    conditions_met = condition_rsi + condition_volume + condition_down

    current_price = row['close']

    # 卖出条件（与数据分析脚本完全一致：RSI > 60 或 连续上涨 >= 2）
    sell_reason = None
    if shares > 0:
        if row['rsi'] > 60 or row['consecutive_up'] >= 2:
            sell_reason = '卖出'

    # 买入条件（与数据分析脚本完全一致）
    if shares == 0 and conditions_met >= 2:
        buy_price = current_price
        shares = capital / buy_price
        capital = 0
        print(f"买入: {row['date']}, 价格={buy_price:.2f}, RSI={row['rsi']:.1f}, 量比={row['volume_ratio']:.2f}, 连跌={row['consecutive_down']}天")
    elif shares > 0 and sell_reason:
        capital = shares * current_price
        shares = 0
        total_trades += 1
        profit = (current_price - buy_price) / buy_price
        if profit > 0:
            won_trades += 1
        print(f"卖出: {row['date']}, 价格={current_price:.2f}, 收益={profit*100:.2f}%")

if shares > 0:
    sell_price = df['close'].iloc[-1]
    capital = shares * sell_price
    profit = (sell_price - buy_price) / buy_price
    total_trades += 1
    if profit > 0:
        won_trades += 1
    print(f"最后一天卖出: {df['date'].iloc[-1]}, 价格={sell_price:.2f}, 收益={profit*100:.2f}%")

total_return = (capital - 10000) / 10000 * 100
win_rate = won_trades / total_trades if total_trades > 0 else 0
print(f"\n总收益率: {total_return:.2f}%")
print(f"最终资金: {capital:.2f}")
print(f"总交易次数: {total_trades}")
print(f"胜率: {win_rate:.2%}")
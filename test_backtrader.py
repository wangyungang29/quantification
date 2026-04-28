import tushare as ts
import pandas as pd
import backtrader as bt
from src.backtest.backtester import PredictionStrategy

TUSHARE_TOKEN = '68e431d47c0319d7bdea7cd1daf164392d22c8a9216d99476354be78'

pro = ts.pro_api(TUSHARE_TOKEN)
df = pro.daily(ts_code='000564.SZ', start_date='20260101', end_date='20260415')
df = df.sort_values('trade_date').reset_index(drop=True)
df.rename(columns={'trade_date': 'date', 'vol': 'volume'}, inplace=True)
df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
df = df.sort_index()  # 按日期升序排列
df = df.set_index('date')

cerebro = bt.Cerebro()
cerebro.addstrategy(PredictionStrategy)

data = bt.feeds.PandasData(
    dataname=df,
    datetime=None,
    open=1,
    high=2,
    low=3,
    close=4,
    volume=8,
    openinterest=-1
)

cerebro.adddata(data)
cerebro.broker.setcash(10000)

results = cerebro.run()

with open('/Users/wangyungang/Desktop/quantification/backtest_log.txt', 'w') as f:
    import sys
    original_stdout = sys.stdout
    sys.stdout = f
    print(f"最终资金: {cerebro.broker.getvalue():.2f}")
    print(f"总收益率: {(cerebro.broker.getvalue() - 10000) / 10000 * 100:.2f}%")
    sys.stdout = original_stdout

print(f"日志已保存")
print(f"最终资金: {cerebro.broker.getvalue():.2f}")
print(f"总收益率: {(cerebro.broker.getvalue() - 10000) / 10000 * 100:.2f}%")
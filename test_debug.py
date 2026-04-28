import tushare as ts
import pandas as pd
import backtrader as bt

TUSHARE_TOKEN = '68e431d47c0319d7bdea7cd1daf164392d22c8a9216d99476354be78'

pro = ts.pro_api(TUSHARE_TOKEN)
df = pro.daily(ts_code='000564.SZ', start_date='20260101', end_date='20260415')
df = df.sort_values('trade_date').reset_index(drop=True)
df.rename(columns={'trade_date': 'date', 'vol': 'volume'}, inplace=True)
df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
df = df.sort_index()  # 按日期升序排列
df = df.set_index('date')

print("Data shape:", df.shape)
print("Data index (first 5):", df.index[:5].tolist())
print("Data index (last 5):", df.index[-5:].tolist())

class DebugStrategy(bt.Strategy):
    def __init__(self):
        pass

    def next(self):
        if len(self) < 60:
            return

        size = 60
        dates = []
        for i in range(size-1, -1, -1):
            try:
                date_obj = self.data.datetime.date(-i)
                dates.append(date_obj)
            except:
                dates.append(None)

        print(f"当前日期={self.data.datetime.date(0)}, 数据范围={dates[0]} 到 {dates[-1]}")

cerebro = bt.Cerebro()
cerebro.addstrategy(DebugStrategy)

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

cerebro.run()
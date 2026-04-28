import tushare as ts
import pandas as pd
import backtrader as bt
from src.models.model_trainer import ModelTrainer

TUSHARE_TOKEN = '68e431d47c0319d7bdea7cd1daf164392d22c8a9216d99476354be78'

pro = ts.pro_api(TUSHARE_TOKEN)
df = pro.daily(ts_code='000564.SZ', start_date='20260101', end_date='20260415')
df = df.sort_values('trade_date').reset_index(drop=True)
df.rename(columns={'trade_date': 'date', 'vol': 'volume'}, inplace=True)
df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
df = df.sort_index()
df = df.set_index('date')

class DebugStrategy(bt.Strategy):
    def __init__(self):
        self.trainer = ModelTrainer()
        self.count = 0

    def next(self):
        self.count += 1
        if len(self) < 60:
            return

        size = 60
        opens = self.data.open.get(size=size)
        highs = self.data.high.get(size=size)
        lows = self.data.low.get(size=size)
        closes = self.data.close.get(size=size)
        volumes = self.data.volume.get(size=size)

        dates = []
        for i in range(size-1, -1, -1):
            try:
                date_obj = self.data.datetime.date(-i)
                dates.append(date_obj)
            except:
                import datetime
                dates.append(datetime.datetime.now().date())

        local_df = pd.DataFrame({
            'date': dates,
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': volumes
        })
        local_df.set_index('date', inplace=True)

        features, _ = self.trainer.create_features(local_df)

        current_date = self.data.datetime.date(0)
        rsi = features['rsi'].iloc[-1]
        volume_ratio = features['vol/Vol_MA5'].iloc[-1]
        consecutive_down = features['consecutive_down'].iloc[-1]
        buy_cond = rsi < 40 and volume_ratio > 1.2 and consecutive_down >= 2

        if buy_cond:
            print(f"*** 买入信号 ***")
            print(f"当前日期={current_date}")
            print(f"  RSI={rsi:.1f}, 量比={volume_ratio:.2f}, 连跌={consecutive_down}")

        if self.count >= 10:
            return

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
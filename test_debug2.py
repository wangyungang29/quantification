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

print(f"原始数据范围: {df.index.min()} 到 {df.index.max()}, 共 {len(df)} 行")

class DebugStrategy(bt.Strategy):
    def __init__(self):
        self.trainer = ModelTrainer()

    def next(self):
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
        print(f"当前日期={current_date}")
        print(f"  特征最后3行: close={features['close'].iloc[-3:].tolist()}, rsi={features['rsi'].iloc[-3:].tolist()}, consecutive_down={features['consecutive_down'].iloc[-3:].tolist()}")
        print(f"  当前值: RSI={features['rsi'].iloc[-1]:.1f}, 量比={features['vol/Vol_MA5'].iloc[-1]:.2f}, 连跌={features['consecutive_down'].iloc[-1]}")

        if len(self) > 65:
            print(f"\n===== 2026-03-23 分析 =====")
            print(f"  2026-03-23的特征: RSI={features.loc['2026-03-23', 'rsi']:.1f}, 量比={features.loc['2026-03-23', 'vol/Vol_MA5']:.2f}, 连跌={features.loc['2026-03-23', 'consecutive_down']}")
            buy_cond = features.loc['2026-03-23', 'rsi'] < 40 and features.loc['2026-03-23', 'vol/Vol_MA5'] > 1.2 and features.loc['2026-03-23', 'consecutive_down'] >= 2
            print(f"  买入条件: RSI<40={features.loc['2026-03-23', 'rsi'] < 40}, 量比>1.2={features.loc['2026-03-23', 'vol/Vol_MA5'] > 1.2}, 连跌>=2={features.loc['2026-03-23', 'consecutive_down'] >= 2}, 全部满足={buy_cond}")

        if len(self) >= 67:
            return  # 只打印几个就退出

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
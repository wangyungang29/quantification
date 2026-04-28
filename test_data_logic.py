import tushare as ts
import pandas as pd
import backtrader as bt
from src.models.model_trainer import ModelTrainer

TUSHARE_TOKEN = '68e431d47c0319d7bdea7cd1daf164392d22c8a9216d99476354be78'

class TestStrategy(bt.Strategy):
    def __init__(self):
        self.trainer = ModelTrainer()
        self.count = 0

    def next(self):
        self.count += 1
        if self.count < 60:
            return

        data = self.datas[0]
        current_date = data.datetime.date(0)
        current_close = data.close[0]

        # 获取60天数据
        size = 60
        closes = []
        volumes = []
        dates = []

        for i in range(size):
            try:
                closes.append(data.close[-i])
                volumes.append(data.volume[-i])
                dates.append(data.datetime.date(-i))
            except:
                pass

        df = pd.DataFrame({
            'date': dates,
            'close': closes,
            'volume': volumes
        })
        df.set_index('date', inplace=True)

        # 反转数据顺序，使得最新日期在最后
        df = df.iloc[::-1]

        # 检查数据
        if current_date.strftime('%Y%m%d') == '20260323':
            print(f"当前日期: {current_date}")
            print(f"DataFrame前5行:\n{df.head()}")
            print(f"DataFrame后5行:\n{df.tail()}")

        # 计算特征
        try:
            df_features, _ = self.trainer.create_features(df)

            if current_date in df_features.index:
                row = df_features.loc[current_date]
                rsi = row['rsi']
                volume_ratio = row['vol/Vol_MA5']
                consecutive_down = row['consecutive_down']
                consecutive_up = row['consecutive_up']

                buy_condition = (rsi < 40) and (volume_ratio > 1.2) and (consecutive_down >= 2)

                if buy_condition or current_date.strftime('%Y%m%d') in ['20260323', '20260324', '20260325']:
                    print(f"{current_date}: RSI={rsi:.1f}, 量比={volume_ratio:.2f}, 连跌={consecutive_down}, 连涨={consecutive_up}, 买入={buy_condition}, 当前价={current_close:.2f}")
        except Exception as e:
            import traceback
            traceback.print_exc()

# 获取数据
pro = ts.pro_api(TUSHARE_TOKEN)
df = pro.daily(ts_code='000564.SZ', start_date='20251001', end_date='20260415')
df = df.sort_values('trade_date').reset_index(drop=True)
df.rename(columns={'trade_date': 'date', 'vol': 'volume'}, inplace=True)
df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
df = df.set_index('date')

print(f"数据范围: {df.index.min()} 到 {df.index.max()}")

cerebro = bt.Cerebro()
cerebro.addstrategy(TestStrategy)

data = bt.feeds.PandasData(
    dataname=df,
    open='open',
    high='high',
    low='low',
    close='close',
    volume='volume',
    openinterest=None
)

cerebro.adddata(data)
cerebro.broker.setcash(10000)

cerebro.run()
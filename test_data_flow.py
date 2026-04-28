import tushare as ts
import pandas as pd
import backtrader as bt
from src.models.model_trainer import ModelTrainer

TUSHARE_TOKEN = '68e431d47c0319d7bdea7cd1daf164392d22c8a9216d99476354be78'

class DebugStrategy(bt.Strategy):
    def __init__(self):
        self.trainer = ModelTrainer()

    def next(self):
        if len(self) < 60:
            return

        # 获取数据 - 从当前位置往前60天
        data = self.datas[0]
        size = 60
        opens = []
        highs = []
        lows = []
        closes = []
        volumes = []
        dates = []
        
        for i in range(size-1, -1, -1):
            try:
                opens.append(data.open[-i])
                highs.append(data.high[-i])
                lows.append(data.low[-i])
                closes.append(data.close[-i])
                volumes.append(data.volume[-i])
                dates.append(data.datetime.date(-i))
            except:
                opens.append(data.open[0])
                highs.append(data.high[0])
                lows.append(data.low[0])
                closes.append(data.close[0])
                volumes.append(data.volume[0])
                dates.append(data.datetime.date(0))

        # 创建DataFrame
        df = pd.DataFrame({
            'date': dates,
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': volumes
        })
        df.set_index('date', inplace=True)

        # 计算特征
        features, _ = self.trainer.create_features(df)

        # 检查当前日期
        current_date = data.datetime.date(0)
        print(f"当前日期: {current_date}")
        
        # 检查特征
        if current_date in features.index:
            rsi = features.loc[current_date, 'rsi']
            vol_ratio = features.loc[current_date, 'vol/Vol_MA5']
            consec_down = features.loc[current_date, 'consecutive_down']
            buy_cond = rsi < 40 and vol_ratio > 1.2 and consec_down >= 2
            print(f"  RSI={rsi:.1f}, 量比={vol_ratio:.2f}, 连跌={consec_down}, 买入={buy_cond}")
        else:
            print(f"  {current_date} 不在特征索引中")

        # 只运行几天
        if len(self) >= 65:
            return

# 获取数据
pro = ts.pro_api(TUSHARE_TOKEN)
df = pro.daily(ts_code='000564.SZ', start_date='20251001', end_date='20260415')
df = df.sort_values('trade_date').reset_index(drop=True)
df.rename(columns={'trade_date': 'date', 'vol': 'volume'}, inplace=True)
df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
df = df.set_index('date')

print(f"数据范围: {df.index.min()} 到 {df.index.max()}")
print(f"数据长度: {len(df)}天")

# 运行回测
cerebro = bt.Cerebro()
cerebro.addstrategy(DebugStrategy)

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

print("\n开始回测...")
cerebro.run()
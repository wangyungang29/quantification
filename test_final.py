import tushare as ts
import pandas as pd
import backtrader as bt
from src.models.model_trainer import ModelTrainer

TUSHARE_TOKEN = '68e431d47c0319d7bdea7cd1daf164392d22c8a9216d99476354be78'

class FinalStrategy(bt.Strategy):
    def __init__(self):
        self.trainer = ModelTrainer()
        self.trades = []

    def next(self):
        if len(self) < 60:
            return

        # 获取完整的数据范围
        data = self.datas[0]
        size = len(data)
        
        # 获取所有历史数据
        opens = []
        highs = []
        lows = []
        closes = []
        volumes = []
        dates = []
        
        for i in range(size):
            try:
                opens.append(data.open[i])
                highs.append(data.high[i])
                lows.append(data.low[i])
                closes.append(data.close[i])
                volumes.append(data.volume[i])
                dates.append(data.datetime.date(i))
            except:
                pass

        # 创建完整的DataFrame
        df = pd.DataFrame({
            'date': dates,
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': volumes
        })
        df.set_index('date', inplace=True)

        # 计算完整的特征
        features, _ = self.trainer.create_features(df)

        # 获取当前日期
        current_date = data.datetime.date(0)
        print(f"当前日期: {current_date}")
        
        # 检查当前日期是否在特征中
        if current_date in features.index:
            # 获取当前日期的特征
            row = features.loc[current_date]
            rsi = row['rsi']
            volume_ratio = row['vol/Vol_MA5']
            consecutive_down = row['consecutive_down']
            consecutive_up = row['consecutive_up']

            # 买入条件
            buy_condition = (rsi < 40) and (volume_ratio > 1.2) and (consecutive_down >= 2)

            # 卖出条件
            sell_condition = False
            if self.position.size > 0:
                if rsi > 60 or consecutive_up >= 2:
                    sell_condition = True

            # 执行交易
            current_price = data.close[0]
            cash = self.broker.getcash()
            total_capital = cash + (self.position.size * current_price if self.position.size > 0 else 0)

            print(f"  RSI={rsi:.1f}, 量比={volume_ratio:.2f}, 连跌={consecutive_down}, 连涨={consecutive_up}")
            print(f"  买入条件={buy_condition}, 卖出条件={sell_condition}")

            if self.position.size == 0 and buy_condition:
                trade_size = int((total_capital * 0.45) / current_price)
                if cash >= current_price * trade_size:
                    self.buy(size=trade_size)
                    print(f"  ✅ 执行买入: {trade_size}股, 价格={current_price:.2f}")
            elif self.position.size > 0 and sell_condition:
                self.sell(size=self.position.size)
                print(f"  ✅ 执行卖出: {self.position.size}股, 价格={current_price:.2f}")

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status in [order.Completed]:
            action = "买入" if order.isbuy() else "卖出"
            print(f"  📊 交易完成: {action} {order.executed.size}股, 价格={order.executed.price:.2f}")

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
cerebro.addstrategy(FinalStrategy)

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
cerebro.broker.setcommission(commission=0.0005)

print("\n开始回测...")
cerebro.run()
print(f"最终资金: {cerebro.broker.getvalue():.2f}")
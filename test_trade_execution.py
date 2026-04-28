import tushare as ts
import pandas as pd
import backtrader as bt
from src.models.model_trainer import ModelTrainer

TUSHARE_TOKEN = '68e431d47c0319d7bdea7cd1daf164392d22c8a9216d99476354be78'

class TestStrategy(bt.Strategy):
    def __init__(self):
        self.trainer = ModelTrainer()
        self.trades = []

    def next(self):
        # 只处理2026-03-20到2026-03-26期间的数据
        current_date = self.data.datetime.date(0)
        if current_date < pd.Timestamp('2026-03-20').date() or current_date > pd.Timestamp('2026-03-26').date():
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

        # 获取特征值
        rsi = features['rsi'].iloc[-1]
        volume_ratio = features['vol/Vol_MA5'].iloc[-1]
        consecutive_down = features['consecutive_down'].iloc[-1]

        # 计算连续上升天数
        consecutive_up = 0
        if len(features) > 0 and 'close' in features:
            close_prices = features['close']
            if len(close_prices) >= 3:
                if close_prices.iloc[-1] > close_prices.iloc[-2] and close_prices.iloc[-2] > close_prices.iloc[-3]:
                    consecutive_up = 2
                elif close_prices.iloc[-1] > close_prices.iloc[-2]:
                    consecutive_up = 1

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

        print(f"{current_date}: 价格={current_price:.2f}, RSI={rsi:.1f}, 量比={volume_ratio:.2f}, 连跌={consecutive_down}, 连涨={consecutive_up}")
        print(f"  持仓={self.position.size}, 现金={cash:.2f}, 总资产={total_capital:.2f}")
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

# 运行回测
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
cerebro.broker.setcommission(commission=0.0005)

print("开始回测...")
cerebro.run()
print(f"最终资金: {cerebro.broker.getvalue():.2f}")
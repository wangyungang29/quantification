import tushare as ts
import pandas as pd
import backtrader as bt
from src.models.model_trainer import ModelTrainer

TUSHARE_TOKEN = '68e431d47c0319d7bdea7cd1daf164392d22c8a9216d99476354be78'

class TestStrategy(bt.Strategy):
    def __init__(self):
        self.trainer = ModelTrainer()
        self.dataclose = self.datas[0].close
        self.datavolume = self.datas[0].volume

    def next(self):
        if len(self) < 60:
            return

        # 模拟backtester中的数据获取逻辑
        data = self.datas[0]
        size = 60
        
        opens = []
        highs = []
        lows = []
        closes = []
        volumes = []
        dates = []
        
        for i in range(60):
            try:
                opens.append(data.open[-i-1])
                highs.append(data.high[-i-1])
                lows.append(data.low[-i-1])
                closes.append(data.close[-i-1])
                volumes.append(data.volume[-i-1])
                dates.append(data.datetime.date(-i-1))
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
        try:
            df_features, _ = self.trainer.create_features(df)
            
            # 获取当前日期
            current_date = data.datetime.date(0)
            print(f"\n当前日期: {current_date}")
            
            if current_date in df_features.index:
                row = df_features.loc[current_date]
                rsi = row['rsi']
                volume_ratio = row['vol/Vol_MA5']
                consecutive_down = row['consecutive_down']
                consecutive_up = row['consecutive_up']
                
                print(f"  RSI: {rsi:.1f}")
                print(f"  量比: {volume_ratio:.2f}")
                print(f"  连续下跌: {consecutive_down}")
                print(f"  连续上涨: {consecutive_up}")
                
                # 买入条件
                buy_condition = (rsi < 40) and (volume_ratio > 1.2) and (consecutive_down >= 2)
                print(f"  买入条件: {buy_condition}")
                
                # 卖出条件
                sell_condition = False
                if self.position.size > 0:
                    if rsi > 60 or consecutive_up >= 2:
                        sell_condition = True
                print(f"  卖出条件: {sell_condition}")
        except Exception as e:
            print(f"特征计算错误: {e}")

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

print("\n开始测试...")
cerebro.run()
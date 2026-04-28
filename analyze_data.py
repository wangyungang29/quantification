import pandas as pd
import numpy as np
import tushare as ts
import matplotlib.pyplot as plt
import os
from datetime import datetime, timedelta

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

class DataAnalyzer:
    def __init__(self):
        # 初始化tushare
        self.TUSHARE_TOKEN = '68e431d47c0319d7bdea7cd1daf164392d22c8a9216d99476354be78'
        self.pro = ts.pro_api(self.TUSHARE_TOKEN)
        self.data_dir = 'data/analysis'
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
    
    def get_stock_data(self, ts_code, start_date, end_date):
        """获取股票数据"""
        print(f"正在获取股票 {ts_code} 的数据，时间范围：{start_date} 到 {end_date}")
        df = self.pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
        
        if df.empty:
            print(f"无法获取股票 {ts_code} 的数据")
            return None
        
        # 按日期排序
        df = df.sort_values('trade_date').reset_index(drop=True)
        
        # 重命名列
        df.rename(columns={
            'trade_date': 'date',
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'vol': 'volume',
            'amount': 'amount'
        }, inplace=True)
        
        # 转换日期格式
        df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
        
        # 计算技术指标
        df = self.calculate_indicators(df)
        
        # 保存数据
        save_path = os.path.join(self.data_dir, f"{ts_code}_{start_date}_{end_date}.csv")
        df.to_csv(save_path, index=False)
        print(f"数据已保存到: {save_path}")
        
        return df
    
    def calculate_indicators(self, df):
        """计算技术指标"""
        # 移动平均线
        df['ma5'] = df['close'].rolling(window=5).mean()
        df['ma20'] = df['close'].rolling(window=20).mean()
        df['ma60'] = df['close'].rolling(window=60).mean()
        
        # RSI
        def calculate_rsi(series, period=14):
            delta = series.diff()
            gain = delta.clip(lower=0)
            loss = -delta.clip(upper=0)
            avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
            avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
            rs = avg_gain / (avg_loss + 1e-8)
            rsi = 100 - (100 / (1 + rs))
            return rsi
        
        df['rsi'] = calculate_rsi(df['close'])
        
        # MACD
        def calculate_macd(close_series, fast=12, slow=26, signal=9):
            ema_fast = close_series.ewm(span=fast, adjust=False).mean()
            ema_slow = close_series.ewm(span=slow, adjust=False).mean()
            dif = ema_fast - ema_slow
            dea = dif.ewm(span=signal, adjust=False).mean()
            macd_hist = (dif - dea) * 2
            return dif, dea, macd_hist
        
        df['macd'], df['macd_signal'], df['macd_hist'] = calculate_macd(df['close'])
        
        # 量比
        df['vol_ma5'] = df['volume'].rolling(window=5).mean()
        df['volume_ratio'] = df['volume'] / df['vol_ma5']
        
        # 连续涨跌天数
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
    
    def analyze_strategies(self, df):
        """分析不同策略的收益"""
        strategies = [
            {'name': 'RSI超卖策略', 'buy_condition': lambda row: row['rsi'] < 30, 'sell_condition': lambda row: row['rsi'] > 70},
            {'name': 'MACD金叉策略', 'buy_condition': lambda row: row['macd'] > row['macd_signal'] and row['macd_hist'] > 0, 'sell_condition': lambda row: row['macd'] < row['macd_signal'] and row['macd_hist'] < 0},
            {'name': 'MA金叉策略', 'buy_condition': lambda row: row['ma5'] > row['ma20'], 'sell_condition': lambda row: row['ma5'] < row['ma20']},
            {'name': '量比策略', 'buy_condition': lambda row: row['volume_ratio'] > 1.5, 'sell_condition': lambda row: row['volume_ratio'] < 1.0},
            {'name': '连续下跌策略', 'buy_condition': lambda row: row['consecutive_down'] >= 2, 'sell_condition': lambda row: row['consecutive_up'] >= 2}
        ]
        
        results = []
        for strategy in strategies:
            print(f"\n测试策略: {strategy['name']}")
            profit, trades = self.backtest_strategy(df, strategy['buy_condition'], strategy['sell_condition'])
            results.append({'strategy': strategy['name'], 'total_return': profit, 'trades': trades})
            print(f"{strategy['name']} 收益率: {profit:.2f}%")
        
        # 分析最佳策略
        best_strategy = max(results, key=lambda x: x['total_return'])
        print(f"\n最佳策略: {best_strategy['strategy']}，收益率: {best_strategy['total_return']:.2f}%")
        
        return results, best_strategy
    
    def backtest_strategy(self, df, buy_condition, sell_condition):
        """回测策略"""
        capital = 10000
        shares = 0
        buy_price = 0
        trades = []
        
        for i in range(1, len(df)):
            row = df.iloc[i]
            date = row['date']
            
            # 买入条件
            if shares == 0 and buy_condition(row):
                buy_price = row['close']
                shares = capital / buy_price
                capital = 0
                trades.append({'date': date, 'action': '买入', 'price': buy_price, 'shares': shares})
                print(f"{date}: 买入，价格={buy_price:.2f}")
            # 卖出条件
            elif shares > 0 and sell_condition(row):
                sell_price = row['close']
                capital = shares * sell_price
                profit = (sell_price - buy_price) / buy_price * 100
                trades.append({'date': date, 'action': '卖出', 'price': sell_price, 'shares': shares, 'profit': profit})
                print(f"{date}: 卖出，价格={sell_price:.2f}，收益={profit:.2f}%")
                shares = 0
        
        # 最后一天卖出
        if shares > 0:
            sell_price = df['close'].iloc[-1]
            capital = shares * sell_price
            profit = (sell_price - buy_price) / buy_price * 100
            trades.append({'date': df['date'].iloc[-1], 'action': '卖出', 'price': sell_price, 'shares': shares, 'profit': profit})
            print(f"{df['date'].iloc[-1]}: 最后一天卖出，价格={sell_price:.2f}，收益={profit:.2f}%")
        
        total_return = (capital - 10000) / 100 * 100
        print(f"策略总收益率: {total_return:.2f}%")
        return total_return, trades
    
    def find_optimal_parameters(self, df):
        """寻找最优参数"""
        print("\n寻找最优参数...")
        
        # 测试不同的RSI阈值
        rsi_results = []
        for buy_rsi in range(20, 40, 5):
            for sell_rsi in range(60, 80, 5):
                profit, _ = self.backtest_strategy(df, 
                                             lambda row: row['rsi'] < buy_rsi, 
                                             lambda row: row['rsi'] > sell_rsi)
                rsi_results.append({'buy_rsi': buy_rsi, 'sell_rsi': sell_rsi, 'return': profit})
        
        best_rsi = max(rsi_results, key=lambda x: x['return'])
        print(f"最佳RSI参数: 买入RSI={best_rsi['buy_rsi']}, 卖出RSI={best_rsi['sell_rsi']}, 收益率={best_rsi['return']:.2f}%")
        
        # 测试不同的量比阈值
        volume_results = []
        for buy_volume in [1.2, 1.5, 1.8, 2.0]:
            for sell_volume in [0.8, 1.0, 1.2]:
                profit, _ = self.backtest_strategy(df, 
                                             lambda row: row['volume_ratio'] > buy_volume, 
                                             lambda row: row['volume_ratio'] < sell_volume)
                volume_results.append({'buy_volume': buy_volume, 'sell_volume': sell_volume, 'return': profit})
        
        best_volume = max(volume_results, key=lambda x: x['return'])
        print(f"最佳量比参数: 买入量比={best_volume['buy_volume']}, 卖出量比={best_volume['sell_volume']}, 收益率={best_volume['return']:.2f}%")
        
        return best_rsi, best_volume
    
    def plot_analysis(self, df, best_strategy):
        """绘制分析图表"""
        plt.figure(figsize=(15, 10))
        
        # 价格和移动平均线
        plt.subplot(4, 1, 1)
        plt.plot(df['date'], df['close'], label='收盘价')
        plt.plot(df['date'], df['ma5'], label='MA5')
        plt.plot(df['date'], df['ma20'], label='MA20')
        plt.title('价格走势和移动平均线')
        plt.legend()
        
        # RSI
        plt.subplot(4, 1, 2)
        plt.plot(df['date'], df['rsi'], label='RSI')
        plt.axhline(30, color='r', linestyle='--')
        plt.axhline(70, color='r', linestyle='--')
        plt.title('RSI指标')
        plt.legend()
        
        # MACD
        plt.subplot(4, 1, 3)
        plt.plot(df['date'], df['macd'], label='MACD')
        plt.plot(df['date'], df['macd_signal'], label='Signal')
        plt.bar(df['date'], df['macd_hist'], label='Histogram')
        plt.title('MACD指标')
        plt.legend()
        
        # 成交量和量比
        plt.subplot(4, 1, 4)
        plt.bar(df['date'], df['volume'], label='成交量')
        plt.title('成交量')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.data_dir, 'analysis_chart.png'))
        plt.show()
    
    def generate_trading_rules(self, best_rsi, best_volume):
        """生成交易规则"""
        rules = {
            '买入条件': [
                f'RSI < {best_rsi["buy_rsi"]} (超卖)',
                f'量比 > {best_volume["buy_volume"]} (成交量放大)',
                '连续下跌 >= 2天',
                'MACD金叉（MACD > Signal且Histogram > 0）',
                'MA5上穿MA20（金叉）'
            ],
            '卖出条件': [
                f'RSI > {best_rsi["sell_rsi"]} (超买)',
                f'量比 < {best_volume["sell_volume"]} (成交量萎缩)',
                '连续上涨 >= 2天',
                'MACD死叉（MACD < Signal且Histogram < 0）',
                'MA5下穿MA20（死叉）'
            ],
            '止盈止损': [
                '止盈：15%',
                '止损：8%',
                '持仓时间：最多15天'
            ],
            '仓位管理': [
                '初始仓位：总资金的45%',
                '加仓条件：价格突破2.00元且成交量持续放大',
                '加仓比例：总资金的17.5%',
                '最大仓位：总资金的70%'
            ]
        }
        
        print("\n交易规则：")
        for rule_type, rule_list in rules.items():
            print(f"\n{rule_type}：")
            for rule in rule_list:
                print(f"  - {rule}")
        
        return rules

if __name__ == "__main__":
    analyzer = DataAnalyzer()
    
    # 获取2026年1月1日至今的数据
    ts_code = '000564.SZ'  # 大商股份
    start_date = '20260101'
    end_date = datetime.now().strftime('%Y%m%d')
    
    df = analyzer.get_stock_data(ts_code, start_date, end_date)
    
    if df is not None:
        # 分析不同策略
        results, best_strategy = analyzer.analyze_strategies(df)
        
        # 寻找最优参数
        best_rsi, best_volume = analyzer.find_optimal_parameters(df)
        
        # 生成交易规则
        rules = analyzer.generate_trading_rules(best_rsi, best_volume)
        
        # 绘制分析图表
        analyzer.plot_analysis(df, best_strategy)
        
        print("\n分析完成！")
        print(f"最佳策略: {best_strategy['strategy']}")
        print(f"最佳收益率: {best_strategy['total_return']:.2f}%")
import backtrader as bt
import pandas as pd
import numpy as np
from src.data.data_fetcher import DataFetcher
from src.models.chan_parser import ChanParser
from config.config import INITIAL_CAPITAL, COMMISSION_RATE, SLIPPAGE


class ChanStrategy(bt.Strategy):
    """基于缠论的交易策略"""
    params = (
        ('stop_loss_ratio', 0.03),  # 止损比例
        ('take_profit_ratio', 0.1),  # 止盈比例
        ('max_position_ratio', 0.5),  # 最大仓位比例
    )
    
    def __init__(self):
        self.dataclose = self.datas[0].close
        self.datahigh = self.datas[0].high
        self.datalow = self.datas[0].low
        self.order = None
        self.parser = ChanParser()
        self.signals = []
        self.signal_index = 0
        self.entry_price = 0
        self.stop_loss = 0
        self.take_profit = 0
        self.trades = []  # 记录所有交易详情
        self.current_trade = None  # 当前持仓的交易信息
    
    def next(self):
        if self.order:
            return
        
        # 获取当前日期
        current_date = self.datas[0].datetime.date(0)
        current_price = self.dataclose[0]
        
        # 检查止损
        if self.position.size > 0 and current_price < self.stop_loss:
            if self.current_trade:
                self.current_trade['sell_reason'] = '止损'
            self.sell(size=self.position.size)
            print(f"止损卖出: {current_date}, 价格={current_price:.2f}, 止损位={self.stop_loss:.2f}")
            return
        
        # 检查止盈
        if self.position.size > 0 and current_price > self.take_profit:
            if self.current_trade:
                self.current_trade['sell_reason'] = '止盈'
            self.sell(size=self.position.size)
            print(f"止盈卖出: {current_date}, 价格={current_price:.2f}, 止盈位={self.take_profit:.2f}")
            return
        
        # 更新缠论信号（每N天更新一次）
        if len(self) % 5 == 0 or len(self.signals) == 0:
            self._update_chan_signals()
        
        # 获取当前日期的信号
        current_signal = self._get_signal_for_date(current_date)
        
        if current_signal:
            if current_signal['type'] in ['一买', '二买', '三买'] and self.position.size == 0:
                # 买入
                total_capital = self.broker.getvalue()
                trade_size = int((total_capital * self.p.max_position_ratio) / current_price)
                
                if trade_size > 0:
                    self.buy(size=trade_size)
                    self.entry_price = current_price
                    self.stop_loss = current_price * (1 - self.p.stop_loss_ratio)
                    self.take_profit = current_price * (1 + self.p.take_profit_ratio)
                    # 记录买入信息
                    self.current_trade = {
                        'buy_date': current_date,
                        'buy_price': current_price,
                        'buy_signal_type': current_signal['type'],
                        'stop_loss': self.stop_loss,
                        'take_profit': self.take_profit,
                        'quantity': trade_size,
                        'sell_date': None,
                        'sell_price': None,
                        'sell_reason': None,
                        'profit': None,
                        'profit_ratio': None
                    }
                    print(f"买入信号({current_signal['type']}): {current_date}, 价格={current_price:.2f}, 数量={trade_size}")
            
            elif current_signal['type'] in ['一卖', '二卖', '三卖'] and self.position.size > 0:
                # 卖出
                self.sell(size=self.position.size)
                print(f"卖出信号({current_signal['type']}): {current_date}, 价格={current_price:.2f}")
    
    def _update_chan_signals(self):
        """更新缠论信号"""
        # 获取历史数据
        data = self.datas[0]
        size = min(len(self), 200)  # 最多使用200天数据
        
        opens = []
        highs = []
        lows = []
        closes = []
        
        for i in range(size):
            try:
                opens.append(data.open[-i])
                highs.append(data.high[-i])
                lows.append(data.low[-i])
                closes.append(data.close[-i])
            except:
                opens.append(data.open[0])
                highs.append(data.high[0])
                lows.append(data.low[0])
                closes.append(data.close[0])
        
        df = pd.DataFrame({
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes
        })
        
        # 使用缠论解析器分析
        result = self.parser.analyze(df)
        self.signals = result.get('signals', [])
    
    def _get_signal_for_date(self, date):
        """获取指定日期的信号"""
        # 简化实现：返回最新的信号
        if self.signals:
            return self.signals[-1]
        return None
    
    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
        
        if order.status in [order.Completed]:
            action = "买入" if order.isbuy() else "卖出"
            current_date = self.data.datetime.date(0)
            executed_price = order.executed.price
            executed_size = order.executed.size
            
            print(f"{action}完成: 日期={current_date}, 价格={executed_price:.2f}, 数量={executed_size}")
            
            if not order.isbuy() and self.current_trade:
                # 完成卖出，更新交易记录
                self.current_trade['sell_date'] = current_date
                self.current_trade['sell_price'] = executed_price
                if self.current_trade['sell_reason'] is None:
                    self.current_trade['sell_reason'] = '信号卖出'
                
                # 计算利润
                buy_price = self.current_trade['buy_price']
                sell_price = executed_price
                quantity = self.current_trade['quantity']
                profit = (sell_price - buy_price) * quantity
                profit_ratio = (sell_price - buy_price) / buy_price * 100
                
                self.current_trade['profit'] = round(profit, 2)
                self.current_trade['profit_ratio'] = round(profit_ratio, 2)
                
                # 添加到交易列表
                self.trades.append(self.current_trade)
                self.current_trade = None
        
        self.order = None


class Backtester:
    def __init__(self):
        self.fetcher = DataFetcher()
    
    def prepare_datafeed(self, ts_code, start_date, end_date):
        """准备回测数据"""
        import datetime
        extended_start = (pd.to_datetime(start_date) - pd.Timedelta(days=90)).strftime('%Y%m%d')
        print(f"原始起始日期: {start_date}, 扩展起始日期: {extended_start}")
        df = self.fetcher.get_stock_history(ts_code, extended_start, end_date)
        
        if df.empty:
            print(f"无法获取股票 {ts_code} 的数据")
            return None
        
        if len(df) < 60:
            print(f"警告: 数据长度不足60天，当前长度: {len(df)}天")
            current_date = datetime.datetime.now().strftime('%Y%m%d')
            new_start = (datetime.datetime.now() - datetime.timedelta(days=180)).strftime('%Y%m%d')
            df = self.fetcher.get_stock_history(ts_code, new_start, current_date)
            if len(df) < 60:
                print(f"错误: 无法获取足够的历史数据")
                return None
        
        print(f"数据列: {list(df.columns)}")
        print(f"数据长度: {len(df)}天")
        
        if 'date' not in df.columns and 'trade_date' in df.columns:
            df.rename(columns={'trade_date': 'date'}, inplace=True)
        
        if 'date' in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df['date']):
                df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
        else:
            print("错误: 没有找到日期列")
            return None
        
        data = bt.feeds.PandasData(
            dataname=df,
            open='open',
            high='high',
            low='low',
            close='close',
            volume='volume',
            openinterest=-1
        )
        
        return data
    
    def run_backtest(self, ts_code, start_date, end_date, model_type='chan'):
        """运行回测"""
        print(f"开始回测: {ts_code}, 时间范围: {start_date} - {end_date}")
        
        cerebro = bt.Cerebro()
        
        # 设置初始资金
        cerebro.broker.setcash(INITIAL_CAPITAL)
        
        # 设置佣金和滑点
        cerebro.broker.setcommission(commission=COMMISSION_RATE)
        cerebro.broker.set_slippage_perc(perc=SLIPPAGE)
        
        # 获取数据
        data = self.prepare_datafeed(ts_code, start_date, end_date)
        if data is None:
            return None
        
        cerebro.adddata(data)
        
        # 添加策略
        cerebro.addstrategy(ChanStrategy)
        
        # 添加分析器
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
        
        # 运行回测
        print(f"初始资金: {INITIAL_CAPITAL:.2f}")
        results = cerebro.run()
        
        # 获取回测结果
        strat = results[0]
        final_value = cerebro.broker.getvalue()
        profit = final_value - INITIAL_CAPITAL
        profit_ratio = profit / INITIAL_CAPITAL * 100
        
        # 获取分析器结果
        sharpe_ratio = strat.analyzers.sharpe.get_analysis().get('sharperatio', 0)
        drawdown = strat.analyzers.drawdown.get_analysis()
        max_drawdown = drawdown.get('max', {}).get('drawdown', 0)
        
        returns = strat.analyzers.returns.get_analysis()
        total_return = returns.get('rtot', 0) * 100
        annual_return = returns.get('rnorm', 0) * 100
        
        trade_analysis = strat.analyzers.trades.get_analysis()
        total_trades = trade_analysis.get('total', {}).get('total', 0)
        win_trades = trade_analysis.get('won', {}).get('total', 0)
        lose_trades = trade_analysis.get('lost', {}).get('total', 0)
        
        win_rate = win_trades / total_trades * 100 if total_trades > 0 else 0
        
        print(f"回测完成")
        print(f"最终资金: {final_value:.2f}")
        print(f"总收益: {profit:.2f} ({profit_ratio:.2f}%)")
        print(f"夏普比率: {sharpe_ratio:.2f}" if sharpe_ratio else "夏普比率: N/A")
        print(f"最大回撤: {max_drawdown:.2f}%" if max_drawdown else "最大回撤: N/A")
        print(f"交易次数: {total_trades} (盈利: {win_trades}, 亏损: {lose_trades})")
        print(f"胜率: {win_rate:.2f}%")
        
        # 获取交易详情
        trade_details = []
        if hasattr(strat, 'trades'):
            for trade in strat.trades:
                trade_details.append({
                    'buy_date': trade['buy_date'].strftime('%Y-%m-%d'),
                    'buy_price': round(trade['buy_price'], 2),
                    'buy_signal_type': trade['buy_signal_type'],
                    'sell_date': trade['sell_date'].strftime('%Y-%m-%d') if trade['sell_date'] else None,
                    'sell_price': round(trade['sell_price'], 2) if trade['sell_price'] else None,
                    'sell_reason': trade['sell_reason'],
                    'quantity': trade['quantity'],
                    'stop_loss': round(trade['stop_loss'], 2),
                    'take_profit': round(trade['take_profit'], 2),
                    'profit': trade['profit'],
                    'profit_ratio': trade['profit_ratio']
                })
        
        # 打印交易详情
        print("\n=== 交易详情 ===")
        for i, trade in enumerate(trade_details, 1):
            print(f"\n交易 {i}:")
            print(f"  买入日期: {trade['buy_date']}")
            print(f"  买入价格: {trade['buy_price']:.2f}")
            print(f"  买入信号: {trade['buy_signal_type']}")
            print(f"  卖出日期: {trade['sell_date']}")
            print(f"  卖出价格: {trade['sell_price']:.2f}" if trade['sell_price'] else "  卖出价格: 未卖出")
            print(f"  卖出原因: {trade['sell_reason']}")
            print(f"  数量: {trade['quantity']}")
            print(f"  止损位: {trade['stop_loss']:.2f}")
            print(f"  止盈位: {trade['take_profit']:.2f}")
            print(f"  利润: {trade['profit']:.2f} ({trade['profit_ratio']:.2f}%)")
        
        return {
            'ts_code': ts_code,
            'start_date': start_date,
            'end_date': end_date,
            'initial_capital': INITIAL_CAPITAL,
            'final_value': round(final_value, 2),
            'profit': round(profit, 2),
            'profit_ratio': round(profit_ratio, 2),
            'sharpe_ratio': round(float(sharpe_ratio), 2) if sharpe_ratio else 0,
            'max_drawdown': round(float(max_drawdown), 2),
            'total_return': round(total_return, 2),
            'annual_return': round(annual_return, 2),
            'total_trades': total_trades,
            'win_trades': win_trades,
            'lose_trades': lose_trades,
            'win_rate': round(win_rate, 2),
            'model_type': 'chan',
            'trades': trade_details
        }
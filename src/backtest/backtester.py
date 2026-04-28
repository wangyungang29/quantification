import backtrader as bt
import pandas as pd
import numpy as np
from src.data.data_fetcher import DataFetcher
from src.features.feature_extractor import FeatureExtractor
from src.models.model_trainer import ModelTrainer
from config.config import INITIAL_CAPITAL, COMMISSION_RATE, SLIPPAGE, FEATURES

class PredictionStrategy(bt.Strategy):
    """基于预测结果的交易策略"""
    params = (
        ('model', None),
        ('feature_columns', None),  # 训练时的特征列名
        ('buy_threshold', 0.55),  # 降低买入阈值
        ('sell_threshold', 0.45),  # 降低卖出阈值
        ('rsi_threshold', 40),     # 放宽RSI阈值
        ('volume_ratio_threshold', 1.2),  # 降低量比要求
        ('max_consecutive_down', 7),  # 延长最大容忍天数
    )
    
    def __init__(self):
        self.dataclose = self.datas[0].close
        self.order = None
        self.trainer = ModelTrainer()  # 用于特征计算
        self.current_features = None  # 存储当前特征（含辅助指标）
        self.trades = []  # 记录交易历史
        self.signal_date = None  # 保存信号发生的日期
        self.signal_price = None  # 保存信号发生时的价格
        self.signal_shares = None  # 保存信号发生时的股数
    
    def next(self):
        if self.order:
            return
            
        if len(self) < 60:
            return
            
        self.current_features = self._build_full_features()
        if self.current_features is None:
            return
        
        if len(self.current_features) > 0:
            row = self.current_features.iloc[-1]
            rsi = row['rsi']
            volume_ratio = row['vol/Vol_MA5']
            consecutive_down = row['consecutive_down']
            consecutive_up = row['consecutive_up']
            current_date = self.datas[0].datetime.date(0)
            current_price = self.datas[0].close[0]
            
            buy_condition = (rsi < 40) and (volume_ratio > 1.2) and (consecutive_down >= 2)
            
            sell_condition = False
            if self.position.size > 0:
                if rsi > 60 or consecutive_up >= 2:
                    sell_condition = True
            
            if self.position.size == 0 and buy_condition:
                total_capital = self.broker.getvalue()
                trade_size = int((total_capital * 0.45) / current_price)
                if trade_size > 0:
                    self.buy(size=trade_size)
                    self.signal_date = current_date
                    self.signal_price = current_price
                    self.signal_shares = trade_size
                    print(f"信号-买入: 日期={current_date}, 价格={current_price:.2f}, 数量={trade_size}")
            elif self.position.size > 0 and sell_condition:
                self.sell(size=self.position.size)
                self.signal_date = current_date
                self.signal_price = current_price
                print(f"信号-卖出: 日期={current_date}, 价格={current_price:.2f}")
    
    def _build_full_features(self):
        """构建与训练时完全一致的特征（含辅助指标）"""
        # 获取最近60天数据（从当前位置往前60天）
        data = self.datas[0]
        size = 60  # 与训练时一致（用过去60天数据）

        # 从当前位置往前获取60天数据（包括当前bar）
        opens = []
        highs = []
        lows = []
        closes = []
        volumes = []
        dates = []

        for i in range(size):
            try:
                # 在backtrader中，-i表示从当前位置往前i天
                # i=0时，-0表示当前bar
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
        
        # 日期已经在上面的循环中获取
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
        
        # 反转数据顺序，使得最新日期在最后（与训练时一致）
        df = df.iloc[::-1]
        
        # 调用训练器的create_features计算所有特征（含连续下降天数、RSI等）
        try:
            df_features, _ = self.trainer.create_features(df)
            
            # 不再依赖模型特征列
            if df_features.empty:
                print("特征计算结果为空")
                return None

            # 不再打印特征最后3行，避免过多输出
            # print(f"特征最后3行: close={df_features['close'].iloc[-3:].tolist()}, rsi={df_features['rsi'].iloc[-3:].tolist()}, consecutive_down={df_features['consecutive_down'].iloc[-3:].tolist()}")
                
            return df_features  # 返回所有特征
        except Exception as e:
            print(f"特征计算错误: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _predict_up_probability(self, model_input_df):
        """修复预测输入格式"""
        try:
            # 确保列顺序与训练时一致
            print(f"模型输入特征列: {list(model_input_df.columns)}")
            print(f"模型期望特征列: {list(self.p.feature_columns)}")
            
            # 确保输入特征与模型训练时的特征一致
            model_input_df = model_input_df[self.p.feature_columns]
            print(f"调整后的输入特征列: {list(model_input_df.columns)}")
            
            # 打印输入特征值
            print(f"输入特征值: {model_input_df.values}")
            
            X = model_input_df.values
            proba = self.p.model.predict_proba(X)
            print(f"预测概率: {proba}")
            return proba[0][1]
        except Exception as e:
            print(f"预测错误: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _execute_trade(self, up_prob, consecutive_down, rsi, volume_ratio):
        """执行交易逻辑（综合策略）"""
        # 1. 获取交易信息
        current_price = self.dataclose[0]
        volume = self.datavolume[0] if hasattr(self, 'datavolume') else 0
        
        # 2. 从特征中获取连续上涨天数
        consecutive_up = 0
        if len(self.current_features) > 0 and 'consecutive_up' in self.current_features:
            consecutive_up = self.current_features['consecutive_up'].iloc[-1]
        
        # 计算价格角度（简化计算）
        price_angle = 0
        if len(self.current_features) > 0 and 'close' in self.current_features:
            close_prices = self.current_features['close']
            if len(close_prices) >= 3:
                # 计算最近两天的价格变化
                price_change1 = close_prices.iloc[-2] - close_prices.iloc[-3]
                price_change2 = close_prices.iloc[-1] - close_prices.iloc[-2]
                # 简化计算角度
                if price_change1 != 0:
                    price_angle = abs(price_change2 / price_change1)
        
        # 计算成交量变化率
        volume_change = 0
        if len(self.current_features) > 0 and 'volume' in self.current_features:
            volume_prices = self.current_features['volume']
            if len(volume_prices) >= 2:
                volume_change = (volume_prices.iloc[-1] - volume_prices.iloc[-2]) / volume_prices.iloc[-2]
        
        # 计算t-2的成交量变化
        volume_change_t2 = 0
        if len(self.current_features) > 0 and 'volume' in self.current_features:
            volume_prices = self.current_features['volume']
            if len(volume_prices) >= 3:
                volume_change_t2 = (volume_prices.iloc[-2] - volume_prices.iloc[-3]) / volume_prices.iloc[-3]
        
        # 计算换手率（假设总股本为10亿股）
        total_shares = 1000000000  # 假设总股本为10亿股
        turnover_rate = volume / total_shares * 100 if total_shares > 0 else 0
        
        # 计算MACD指标（简化计算）
        macd_golden_cross = False
        macd_red_bar = False
        if len(self.current_features) > 0 and 'macd' in self.current_features and 'macd_signal' in self.current_features and 'macd_hist' in self.current_features:
            macd = self.current_features['macd']
            macd_signal = self.current_features['macd_signal']
            macd_hist = self.current_features['macd_hist']
            if len(macd) >= 2 and len(macd_signal) >= 2 and len(macd_hist) >= 2:
                # MACD金叉：MACD从下方穿越信号线
                if macd.iloc[-1] > macd_signal.iloc[-1] and macd.iloc[-2] <= macd_signal.iloc[-2]:
                    macd_golden_cross = True
                # 红柱放大：MACD柱状图为正且增加
                if macd_hist.iloc[-1] > 0 and macd_hist.iloc[-1] > macd_hist.iloc[-2]:
                    macd_red_bar = True
        
        # 计算RSI指标（使用传入的rsi值）
        rsi_rebound = False
        if rsi is not None:
            # RSI低于30后反弹至40以上
            if len(self.current_features) > 0 and 'rsi' in self.current_features:
                rsi_values = self.current_features['rsi']
                if len(rsi_values) >= 2:
                    if rsi_values.iloc[-2] < 30 and rsi > 40:
                        rsi_rebound = True
        
        # 计算主力资金流向（简化计算，假设成交量增加且价格上涨为主力资金净流入）
        main_fund_inflow = False
        consecutive_fund_inflow_days = 0
        if len(self.current_features) > 0 and 'close' in self.current_features and 'volume' in self.current_features:
            close_prices = self.current_features['close']
            volume_prices = self.current_features['volume']
            if len(close_prices) >= 4 and len(volume_prices) >= 4:
                # 连续3日主力资金净流入
                for i in range(1, 4):
                    if i < len(close_prices) and i < len(volume_prices):
                        if close_prices.iloc[-i] > close_prices.iloc[-(i+1)] and volume_prices.iloc[-i] > volume_prices.iloc[-(i+1)]:
                            consecutive_fund_inflow_days += 1
                if consecutive_fund_inflow_days >= 3:
                    main_fund_inflow = True
        
        # 4. 综合策略的买入条件（与数据分析一致：RSI < 40 AND volume_ratio > 1.2 AND consecutive_down >= 2）
        buy_condition = (
            (rsi < 40)
            and (volume_ratio > 1.2)
            and (consecutive_down >= 2)
        )

        # 调试输出 - 只打印买入信号
        if buy_condition:
            print(f"日期={self.data.datetime.date(0)}, RSI={rsi:.1f}, 量比={volume_ratio:.2f}, 连跌={consecutive_down}, 买入={buy_condition}")
        
        # 5. 止盈止损条件（保留但不触发）
        short_term_profit = 0.15  # 15%止盈
        stop_loss = 0.08  # 8%止损
        
        # 6. 卖出条件（与数据分析脚本一致：RSI > 60 或 连续上涨 >= 2）
        sell_condition = False
        if self.position.size > 0:
            if rsi > 60 or consecutive_up >= 2:
                sell_condition = True
        
        # 7. 仓位管理
        cash = self.broker.getcash()
        total_capital = cash + (self.position.size * current_price if self.position.size > 0 else 0)
        current_position_ratio = (self.position.size * current_price) / total_capital if total_capital > 0 else 0
        
        # 确保在同一个时间点不会同时触发买入和卖出操作
        if self.position.size == 0 and buy_condition:
            sell_condition = False
        
        # 8. 执行交易
        action = "观望"
        if self.position.size == 0 and buy_condition:
            # 初始仓位：总资金的40-50%
            trade_size = int((total_capital * 0.45) / current_price)
            if trade_size < 1:
                trade_size = 1
            
            if cash >= current_price * trade_size:
                self.buy(size=trade_size)
                action = "买入"
                print(f"✅ 买入: 概率={up_prob:.2f}, RSI={rsi:.1f}, 量比={volume_ratio:.2f}, 换手率={turnover_rate:.2f}%, MACD金叉={macd_golden_cross}, MACD红柱={macd_red_bar}, RSI反弹={rsi_rebound}, 主力资金净流入={main_fund_inflow}, 数量={trade_size}")
            else:
                print(f"❌ 资金不足，无法买入")
        elif self.position.size > 0:
            # 加仓条件：股价突破2.00元且成交量持续放大，可追加10-15%
            if current_price > 2.00 and volume_change > 0 and current_position_ratio < 0.7:
                # 追加15-20%的资金
                add_size = int((total_capital * 0.175) / current_price)
                if add_size < 1:
                    add_size = 1
                
                if cash >= current_price * add_size:
                    self.buy(size=add_size)
                    action = "加仓"
                    print(f"✅ 加仓: 价格={current_price:.2f}, 成交量变化率={volume_change:.2f}, 追加数量={add_size}")
            elif sell_condition:
                self.sell(size=self.position.size)
                action = "卖出"
        
        # 9. 调试输出 - 只打印交易操作
        if action != "观望":
            try:
                print(f"📊 {self.data.datetime.date(0)}: 操作={action}, 价格={current_price:.2f}, 持仓比例={current_position_ratio:.2f}")
            except Exception as e:
                print(f"打印日期失败: {e}")
    
    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
        
        if order.status in [order.Completed]:
            action = "买入" if order.isbuy() else "卖出"
            print(f"{action}: 价格={order.executed.price:.2f}, 数量={order.executed.size}")
            
            # 记录交易信息
            try:
                # 使用信号日期和信号价格
                trade_date = self.signal_date if self.signal_date else self.data.datetime.date(0)
                trade_price = self.signal_price if self.signal_price else order.executed.price
                trade_info = {
                    'date': trade_date.strftime('%Y-%m-%d'),
                    'action': action,
                    'price': round(trade_price, 2),
                    'shares': order.executed.size,
                    'amount': round(trade_price * order.executed.size, 2),
                    'commission': round(order.executed.comm, 2),
                    'total': round(trade_price * order.executed.size + order.executed.comm, 2)
                }
                self.trades.append(trade_info)
                self.signal_date = None  # 清除信号日期
                self.signal_price = None  # 清除信号价格
                self.signal_shares = None  # 清除信号股数
                print(f"交易记录: {trade_info}")
            except Exception as e:
                print(f"记录交易信息失败: {e}")
        
        self.order = None

class Backtester:
    def __init__(self):
        self.fetcher = DataFetcher()
        self.trainer = ModelTrainer()
    
    def prepare_datafeed(self, ts_code, start_date, end_date):
        """准备回测数据"""
        # 获取股票数据
        # 扩展起始日期60天以覆盖回测预热期
        import datetime
        extended_start = (pd.to_datetime(start_date) - pd.Timedelta(days=90)).strftime('%Y%m%d')
        print(f"原始起始日期: {start_date}, 扩展起始日期: {extended_start}")
        df = self.fetcher.get_stock_history(ts_code, extended_start, end_date)
        
        if df.empty:
            print(f"无法获取股票 {ts_code} 的数据")
            return None
        
        # 检查数据长度是否足够
        if len(df) < 60:
            print(f"警告: 数据长度不足60天，当前长度: {len(df)}天")
            # 尝试获取更多数据
            import datetime
            current_date = datetime.datetime.now().strftime('%Y%m%d')
            # 计算起始日期，确保至少获取60天的数据
            new_start = (datetime.datetime.now() - datetime.timedelta(days=180)).strftime('%Y%m%d')
            df = self.fetcher.get_stock_history(ts_code, new_start, current_date)
            if len(df) < 60:
                print(f"错误: 无法获取足够的历史数据，当前长度: {len(df)}天")
                return None
        
        # 检查并打印数据列
        print(f"数据列: {list(df.columns)}")
        print(f"数据长度: {len(df)}天")
        
        # 确保日期列存在
        if 'date' not in df.columns and 'trade_date' in df.columns:
            print("将'trade_date'重命名为'date'")
            df.rename(columns={'trade_date': 'date'}, inplace=True)
        
        # 确保日期格式正确
        if 'date' in df.columns:
            print(f"日期列类型: {type(df['date'].iloc[0])}")
            if not pd.api.types.is_datetime64_any_dtype(df['date']):
                df['date'] = pd.to_datetime(df['date'])
                print("已转换日期格式")
            df.set_index('date', inplace=True)
            print(f"设置索引后的数据列: {list(df.columns)}")
        else:
            print("错误: 没有找到日期列")
            return None
        
        # 转换为backtrader数据格式
        # 当日期已经是索引时，不需要指定datetime参数
        data = bt.feeds.PandasData(
            dataname=df,
            open='open',
            high='high',
            low='low',
            close='close',
            volume='volume',
            openinterest=None
        )
        
        return data
    
    def run_backtest(self, ts_code, start_date, end_date, model_type='lightgbm'):
        """运行回测"""
        # 1. 加载模型
        model = self.trainer.load_model(f"{model_type}_golden_cross_model")
        if model is None:
            print("模型不存在，开始训练...")
            # 修复数据泄露：用回测期前的数据训练模型
            import datetime
            current_date = datetime.datetime.now().strftime('%Y%m%d')
            # 确保start_date不晚于当前日期
            if start_date > current_date:
                print(f"警告: 起始日期 {start_date} 晚于当前日期，将使用默认日期范围")
                start_date = (datetime.datetime.now() - datetime.timedelta(days=365)).strftime('%Y%m%d')
            train_end = (pd.to_datetime(start_date) - pd.Timedelta(days=1)).strftime('%Y%m%d')
            # 确保train_end不晚于当前日期
            if train_end > current_date:
                train_end = current_date
            # 确保训练数据的时间范围合理
            train_start = (pd.to_datetime(train_end) - pd.Timedelta(days=365*2)).strftime('%Y%m%d')
            df_train = self.fetcher.get_stock_history(ts_code, train_start, train_end)
            if df_train.empty:
                print("无法获取训练数据")
                return None
            try:
                model, _ = self.trainer.train_golden_cross_model(df_train, model_type)
                print("模型训练完成")
            except Exception as e:
                print(f"模型训练失败: {e}")
                import traceback
                traceback.print_exc()
                return None
        
        # 2. 获取特征列名
        feature_cols = []
        if hasattr(model, 'feature_names_in_'):
            feature_cols = model.feature_names_in_.tolist()
            print(f"模型特征列: {feature_cols}")
        else:
            feature_cols = FEATURES
            print(f"使用默认特征列: {feature_cols}")
        
        # 3. 准备数据
        data = self.prepare_datafeed(ts_code, start_date, end_date)
        if data is None:
            return None
        
        # 4. 运行回测
        cerebro = bt.Cerebro()
        
        # 添加数据
        cerebro.adddata(data)
        
        # 设置初始资金
        cerebro.broker.setcash(INITIAL_CAPITAL)
        
        # 设置佣金和滑点
        cerebro.broker.setcommission(commission=COMMISSION_RATE)
        cerebro.broker.set_slippage_fixed(SLIPPAGE)
        
        # 添加策略（传入模型和特征列）
        cerebro.addstrategy(
            PredictionStrategy,
            model=model,
            feature_columns=feature_cols,
            buy_threshold=0.4,  # 进一步降低买入阈值
            sell_threshold=0.45,  # 降低卖出阈值
            rsi_threshold=50,     # 进一步放宽RSI阈值
            volume_ratio_threshold=1.0,  # 进一步降低量比要求
            max_consecutive_down=7  # 延长最大容忍天数
        )
        
        # 添加分析器
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
        
        # 运行回测
        print(f"开始回测股票 {ts_code}...")
        results = cerebro.run()
        
        # 提取分析结果
        strat = results[0]
        sharpe = strat.analyzers.sharpe.get_analysis()
        drawdown = strat.analyzers.drawdown.get_analysis()
        returns = strat.analyzers.returns.get_analysis()
        trades = strat.analyzers.trades.get_analysis()
        
        # 计算回测指标
        total_trades = 0
        won_trades = 0
        
        # 优先使用我们自己记录的交易记录
        if results:
            strat = results[0]
            if hasattr(strat, 'trades') and len(strat.trades) > 0:
                # 每两次交易为一次完整交易（买入+卖出）
                total_trades = len(strat.trades) // 2
                
                # 计算胜率：比较买入成本和卖出收入
                for i in range(0, len(strat.trades), 2):
                    if i + 1 < len(strat.trades):
                        buy_trade = strat.trades[i]
                        sell_trade = strat.trades[i + 1]
                        if buy_trade['action'] == '买入' and sell_trade['action'] == '卖出':
                            # 买入成本（正值）和卖出收入（正值）的比较
                            buy_cost = buy_trade['price'] * buy_trade['shares']
                            sell_revenue = abs(sell_trade['price'] * sell_trade['shares'])
                            if sell_revenue > buy_cost:
                                won_trades += 1
        else:
            # 使用TradeAnalyzer的结果
            total_trades = trades.get('total', {}).get('total', 0)
            won_trades = trades.get('won', {}).get('total', 0)
        
        # 计算胜率，避免除以零
        win_rate = won_trades / total_trades if total_trades > 0 else 0
        
        # 处理夏普比率，确保不为None
        sharpe_ratio = sharpe.get('sharperatio', 0)
        if sharpe_ratio is None:
            sharpe_ratio = 0
        
        # 处理最大回撤
        max_drawdown = 0
        if 'max' in drawdown and 'drawdown' in drawdown['max']:
            max_drawdown = drawdown['max']['drawdown']
        
        # 处理总收益率
        total_return = 0
        if 'rtot' in returns:
            total_return = returns['rtot']
        else:
            total_return = (cerebro.broker.getvalue() - INITIAL_CAPITAL) / INITIAL_CAPITAL
        
        # 处理平均收益率
        average_return = returns.get('rnorm', 0)
        if average_return is None:
            average_return = 0
        
        # 获取交易记录
        trades = []
        if results:
            strat = results[0]
            if hasattr(strat, 'trades'):
                trades = strat.trades
                print(f"交易记录数量: {len(trades)}")
                for trade in trades:
                    print(trade)
        
        backtest_results = {
            'initial_capital': INITIAL_CAPITAL,
            'final_value': cerebro.broker.getvalue(),
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'average_return': average_return,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'trades': trades
        }
        
        # 打印回测结果
        print("\n回测结果:")
        print(f"初始资金: {INITIAL_CAPITAL}")
        print(f"最终资金: {cerebro.broker.getvalue():.2f}")
        print(f"总收益率: {backtest_results['total_return']:.2f}")
        print(f"夏普比率: {backtest_results['sharpe_ratio']:.2f}")
        print(f"最大回撤: {backtest_results['max_drawdown']:.2f}")
        print(f"平均收益率: {backtest_results['average_return']:.2f}")
        print(f"总交易次数: {backtest_results['total_trades']}")
        print(f"胜率: {backtest_results['win_rate']:.2f}")
        
        # 绘制回测曲线 - 仅在直接运行时使用，API调用时注释掉
        # cerebro.plot()
        
        return backtest_results

if __name__ == "__main__":
    # 测试回测模块
    backtester = Backtester()
    
    # 运行回测
    ts_code = '600519.SH'  # 贵州茅台
    start_date = '20230101'
    end_date = '20231231'
    
    results = backtester.run_backtest(ts_code, start_date, end_date)
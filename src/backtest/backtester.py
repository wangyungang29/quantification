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
        ('buy_threshold', 0.6),     # 训练时的买入概率阈值（0.6）
        ('sell_threshold', 0.6),    # 训练时的卖出概率阈值（0.6）
        ('rsi_threshold', 30),       # RSI超卖阈值（<30）
        ('volume_ratio_threshold', 1.5),  # 成交量比阈值（>1.5）
        ('max_consecutive_down', 5),  # 最大允许连续下降天数（≤5）
    )
    
    def __init__(self):
        self.dataclose = self.datas[0].close
        self.order = None
        self.trainer = ModelTrainer()  # 用于特征计算
        self.current_features = None  # 存储当前特征（含辅助指标）
    
    def next(self):
        # 检查是否有未完成的订单
        if self.order:
            return
            
        # 确保足够历史数据（训练时用60天）
        if len(self) < 60:
            return
            
        # 1. 构建完整特征（含辅助指标）
        self.current_features = self._build_full_features()
        if self.current_features is None:
            return
            
        # 2. 提取模型输入特征和辅助指标
        model_input = self.current_features[self.p.feature_columns]
        consecutive_down = self.current_features.get('consecutive_down', 0).iloc[-1]
        rsi = self.current_features.get('rsi', 50).iloc[-1]
        volume_ratio = self.current_features.get('vol/Vol_MA5', 1.0).iloc[-1]
        
        # 3. 模型预测
        up_prob = self._predict_up_probability(model_input)
        if up_prob is None:
            return
            
        # 4. 复现训练时的信号逻辑
        self._execute_trade(up_prob, consecutive_down, rsi, volume_ratio)
    
    def _build_full_features(self):
        """构建与训练时完全一致的特征（含辅助指标）"""
        # 获取最近60天数据
        data = self.datas[0]
        size = 60  # 与训练时一致（用过去60天数据）
        opens = data.open.get(size=size)
        highs = data.high.get(size=size)
        lows = data.low.get(size=size)
        closes = data.close.get(size=size)
        volumes = data.volume.get(size=size)
        
        # 日期处理（与训练时一致）
        try:
            # 尝试获取日期
            dates = []
            for i in range(size-1, -1, -1):
                try:
                    # 尝试获取日期对象
                    date_obj = data.datetime.date(-i)
                    dates.append(date_obj)
                except Exception as e:
                    # 如果失败，使用当前日期
                    import datetime
                    dates.append(datetime.datetime.now().date())
        except Exception as e:
            print(f"获取日期失败: {e}")
            # 如果日期获取失败，使用默认日期范围
            import datetime
            base_date = datetime.datetime.now().date()
            dates = [base_date - datetime.timedelta(days=i) for i in range(size-1, -1, -1)]
        
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
        
        # 调用训练器的create_features计算所有特征（含连续下降天数、RSI等）
        try:
            df_features, _ = self.trainer.create_features(df)
            
            # 打印特征匹配日志
            print(f"训练特征列: {self.p.feature_columns}")
            print(f"回测计算特征列: {list(df_features.columns)}")
            matching_features = [c for c in self.p.feature_columns if c in df_features.columns]
            print(f"匹配特征数: {len(matching_features)}")
            
            if df_features.empty:
                print("特征计算结果为空")
                return None
                
            return df_features  # 返回所有特征
        except Exception as e:
            print(f"特征计算错误: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _predict_up_probability(self, model_input):
        """模型预测上涨概率（与训练时一致）"""
        try:
            proba = self.p.model.predict_proba(model_input)
            return proba[0][1]  # 上涨概率（二分类模型的正类概率）
        except Exception as e:
            print(f"预测错误: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _execute_trade(self, up_prob, consecutive_down, rsi, volume_ratio):
        """复现训练时的交易逻辑"""
        # 买入条件（与generate_trading_signals完全一致）
        buy_condition = (
            (consecutive_down <= self.p.max_consecutive_down)  # ≤5天
            and (up_prob > self.p.buy_threshold)  # 概率>0.6
            and (
                (rsi < self.p.rsi_threshold)  # RSI<30（超卖）
                or (volume_ratio > self.p.volume_ratio_threshold)  # 成交量比>1.5
            )
        )
        
        # 卖出条件（与generate_trading_signals一致）
        sell_condition = (
            (up_prob < self.p.sell_threshold)  # 概率<0.6（下跌概率高）
            or (consecutive_down > self.p.max_consecutive_down)  # 连续下降>5天
        )
        
        # 执行交易
        action = "观望"
        if not self.position and buy_condition:
            self.order = self.buy(size=100)
            action = "买入"
            print(f"买入: 概率={up_prob:.2f}, 连续下降={consecutive_down}天, RSI={rsi:.1f}, 量比={volume_ratio:.2f}")
            # 模拟训练时的信号输出
            print(f"⭐ 预测上涨概率{up_prob:.2%}，连续下降{consecutive_down}天，RSI={rsi:.1f}，量比={volume_ratio:.2f}，建议买入")
        elif self.position and sell_condition:
            self.order = self.sell(size=100)
            action = "卖出"
            print(f"卖出: 概率={up_prob:.2f}, 连续下降={consecutive_down}天")
            # 模拟训练时的信号输出
            print(f"❌ 预测下跌概率{(1-up_prob):.2%}，连续下降{consecutive_down}天，建议卖出")
        
        # 打印交易信号
        try:
            print(f"日期: {self.data.datetime.date(0)}, 上涨概率: {up_prob:.4f}, 操作: {action}")
        except Exception as e:
            print(f"打印日期失败: {e}")
    
    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
        
        if order.status in [order.Completed]:
            action = "买入" if order.isbuy() else "卖出"
            print(f"{action}: 价格={order.executed.price:.2f}, 数量={order.executed.size}")
        
        self.order = None

class Backtester:
    def __init__(self):
        self.fetcher = DataFetcher()
        self.trainer = ModelTrainer()
    
    def prepare_datafeed(self, ts_code, start_date, end_date):
        """准备回测数据"""
        # 获取股票数据
        df = self.fetcher.get_stock_history(ts_code, start_date, end_date)
        
        if df.empty:
            print(f"无法获取股票 {ts_code} 的数据")
            return None
        
        # 检查并打印数据列
        print(f"数据列: {list(df.columns)}")
        
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
            # 训练模型
            df = self.fetcher.get_stock_history(ts_code, '20200101', end_date)
            if df.empty:
                print("无法获取训练数据")
                return None
            try:
                model, _ = self.trainer.train_golden_cross_model(df, model_type)
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
            buy_threshold=0.6,          # 训练时的买入阈值
            sell_threshold=0.6,         # 训练时的卖出阈值
            rsi_threshold=30,           # 训练时的RSI阈值
            volume_ratio_threshold=1.5,  # 训练时的量比阈值
            max_consecutive_down=5       # 训练时的最大连续下降天数
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
        
        backtest_results = {
            'initial_capital': INITIAL_CAPITAL,
            'final_value': cerebro.broker.getvalue(),
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'average_return': average_return,
            'total_trades': total_trades,
            'win_rate': win_rate
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
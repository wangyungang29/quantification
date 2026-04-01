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
        ('feature_extractor', None),
        ('buy_threshold', 0.6),  # 买入阈值
        ('sell_threshold', 0.4),  # 卖出阈值
    )
    
    def __init__(self):
        self.dataclose = self.datas[0].close
        self.order = None
        self.buy_price = 0
    
    def next(self):
        # 检查是否有未完成的订单
        if self.order:
            return
        
        # 构建特征数据
        data = self.datas[0]
        df = pd.DataFrame({
            'open': data.open.get(size=60),
            'high': data.high.get(size=60),
            'low': data.low.get(size=60),
            'close': data.close.get(size=60),
            'volume': data.volume.get(size=60),
            'amount': data.volume.get(size=60) * data.close.get(size=60)
        })
        
        # 提取特征
        if len(df) >= 60:  # 确保有足够的数据计算特征
            df_features = self.params.feature_extractor.extract_features(df)
            if not df_features.empty:
                latest_features = df_features.tail(1)[FEATURES]
                
                # 预测
                prob = self.params.model.predict_proba(latest_features)[0]
                up_prob = prob[2]  # 上涨概率
                down_prob = prob[0]  # 下跌概率
                
                # 交易逻辑
                if not self.position:
                    # 没有持仓，判断是否买入
                    if up_prob > self.params.buy_threshold:
                        # 买入
                        self.order = self.buy(size=100)  # 买入100股
                else:
                    # 有持仓，判断是否卖出
                    if down_prob > self.params.sell_threshold:
                        # 卖出
                        self.order = self.sell(size=100)  # 卖出100股
    
    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
        
        if order.status in [order.Completed]:
            if order.isbuy():"""
                self.buy_price = order.executed.price
                print(f"买入: 价格={order.executed.price}, 数量={order.executed.size}")
            else:
                print(f"卖出: 价格={order.executed.price}, 数量={order.executed.size}")
        
        self.order = None

class Backtester:
    def __init__(self):
        self.fetcher = DataFetcher()
        self.extractor = FeatureExtractor()
        self.trainer = ModelTrainer()
    
    def prepare_datafeed(self, ts_code, start_date, end_date):
        """准备回测数据"""
        # 获取股票数据
        df = self.fetcher.get_stock_history(ts_code, start_date, end_date)
        
        if df.empty:
            print(f"无法获取股票 {ts_code} 的数据")
            return None
        
        # 转换为backtrader数据格式
        data = bt.feeds.PandasData(
            dataname=df,
            datetime='date',
            open='open',
            high='high',
            low='low',
            close='close',
            volume='volume',
            openinterest=None
        )
        
        return data
    
    def run_backtest(self, ts_code, start_date, end_date, model_type='xgboost'):
        """运行回测"""
        # 准备数据
        data = self.prepare_datafeed(ts_code, start_date, end_date)
        if data is None:
            return None
        
        # 加载模型
        model = self.trainer.load_model(f"{model_type}_model")
        if model is None:
            print("模型不存在，请先训练模型")
            return None
        
        # 创建回测引擎
        cerebro = bt.Cerebro()
        
        # 添加数据
        cerebro.adddata(data)
        
        # 设置初始资金
        cerebro.broker.setcash(INITIAL_CAPITAL)
        
        # 设置佣金和滑点
        cerebro.broker.setcommission(commission=COMMISSION_RATE)
        cerebro.broker.set_slippage_fixed(SLIPPAGE)
        
        # 添加策略
        cerebro.addstrategy(PredictionStrategy, model=model, feature_extractor=self.extractor)
        
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
        backtest_results = {
            'initial_capital': INITIAL_CAPITAL,
            'final_value': cerebro.broker.getvalue(),
            'total_return': (cerebro.broker.getvalue() - INITIAL_CAPITAL) / INITIAL_CAPITAL,
            'sharpe_ratio': sharpe.get('sharperatio', 0),
            'max_drawdown': drawdown.get('max', {}).get('drawdown', 0),
            'average_return': returns.get('rnorm', 0),
            'total_trades': trades.get('total', {}).get('total', 0),
            'win_rate': trades.get('won', {}).get('total', 0) / trades.get('total', {}).get('total', 1)
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
        
        # 绘制回测曲线
        cerebro.plot()
        
        return backtest_results

if __name__ == "__main__":
    # 测试回测模块
    backtester = Backtester()
    
    # 运行回测
    ts_code = '600519.SH'  # 贵州茅台
    start_date = '20230101'
    end_date = '20231231'
    
    results = backtester.run_backtest(ts_code, start_date, end_date)
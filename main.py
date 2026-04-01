import argparse
import os
from src.data.data_fetcher import DataFetcher
from src.features.feature_extractor import FeatureExtractor
from src.models.model_trainer import ModelTrainer
from src.prediction.predictor import Predictor
from src.backtest.backtester import Backtester

class QuantSystem:
    def __init__(self):
        self.fetcher = DataFetcher()
        self.extractor = FeatureExtractor()
        self.trainer = ModelTrainer()
        self.predictor = Predictor()
        self.backtester = Backtester()
    
    def train_model(self, ts_code, model_type='xgboost'):
        """训练模型"""
        print(f"开始训练 {model_type} 模型...")
        
        # 获取股票数据
        df = self.fetcher.get_stock_history(ts_code)
        
        if df.empty:
            print(f"无法获取股票 {ts_code} 的数据")
            return
        
        # 提取特征
        df_features = self.extractor.extract_features(df)
        
        if df_features.empty:
            print(f"无法提取股票 {ts_code} 的特征")
            return
        
        # 创建标签
        df_labeled = self.extractor.create_labels(df_features)
        
        if df_labeled.empty:
            print(f"无法创建标签")
            return
        
        # 训练模型
        model, metrics = self.trainer.train_and_evaluate(df_labeled, model_type=model_type)
        
        print(f"模型训练完成，准确率: {metrics['accuracy']:.4f}")
    
    def predict_stock(self, ts_code, model_type='xgboost'):
        """预测股票"""
        print(f"开始预测股票 {ts_code}...")
        
        # 预测
        result = self.predictor.predict_stock(ts_code, model_type=model_type)
        
        if result:
            report = self.predictor.generate_prediction_report(result)
            print(report)
        else:
            print("预测失败")
    
    def backtest_model(self, ts_code, start_date, end_date, model_type='xgboost'):
        """回测模型"""
        print(f"开始回测股票 {ts_code}...")
        
        # 运行回测
        results = self.backtester.run_backtest(ts_code, start_date, end_date, model_type=model_type)
        
        if results:
            print("回测完成")
        else:
            print("回测失败")
    
    def fetch_data(self, ts_code):
        """获取股票数据"""
        print(f"开始获取股票 {ts_code} 的数据...")
        
        # 获取数据
        df = self.fetcher.get_stock_history(ts_code)
        
        if not df.empty:
            # 保存数据
            self.fetcher.save_data(df, ts_code)
            print(f"数据获取完成，共 {len(df)} 条记录")
        else:
            print("数据获取失败")

def main():
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='量化交易系统')
    
    # 子命令
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # 训练模型命令
    train_parser = subparsers.add_parser('train', help='训练模型')
    train_parser.add_argument('ts_code', help='股票代码，如 600519.SH')
    train_parser.add_argument('--model', choices=['xgboost', 'lightgbm'], default='xgboost', help='模型类型')
    
    # 预测股票命令
    predict_parser = subparsers.add_parser('predict', help='预测股票')
    predict_parser.add_argument('ts_code', help='股票代码，如 600519.SH')
    predict_parser.add_argument('--model', choices=['xgboost', 'lightgbm'], default='xgboost', help='模型类型')
    
    # 回测模型命令
    backtest_parser = subparsers.add_parser('backtest', help='回测模型')
    backtest_parser.add_argument('ts_code', help='股票代码，如 600519.SH')
    backtest_parser.add_argument('start_date', help='开始日期，如 20230101')
    backtest_parser.add_argument('end_date', help='结束日期，如 20231231')
    backtest_parser.add_argument('--model', choices=['xgboost', 'lightgbm'], default='xgboost', help='模型类型')
    
    # 获取数据命令
    fetch_parser = subparsers.add_parser('fetch', help='获取股票数据')
    fetch_parser.add_argument('ts_code', help='股票代码，如 600519.SH')
    
    # 解析参数
    args = parser.parse_args()
    
    # 创建量化系统实例
    quant_system = QuantSystem()
    
    # 执行命令
    if args.command == 'train':
        quant_system.train_model(args.ts_code, model_type=args.model)
    elif args.command == 'predict':
        quant_system.predict_stock(args.ts_code, model_type=args.model)
    elif args.command == 'backtest':
        quant_system.backtest_model(args.ts_code, args.start_date, args.end_date, model_type=args.model)
    elif args.command == 'fetch':
        quant_system.fetch_data(args.ts_code)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
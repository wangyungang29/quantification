import pandas as pd
import numpy as np
from src.data.data_fetcher import DataFetcher
from src.features.feature_extractor import FeatureExtractor
from src.models.model_trainer import ModelTrainer
from config.config import FEATURES, PREDICT_DAYS

class Predictor:
    def __init__(self):
        self.fetcher = DataFetcher()
        self.extractor = FeatureExtractor()
        self.trainer = ModelTrainer()
    
    def predict_stock(self, ts_code, model_type='xgboost'):
        """预测股票涨跌"""
        # 获取最新股票数据
        df = self.fetcher.get_stock_history(ts_code)
        
        if df.empty:
            print(f"无法获取股票 {ts_code} 的数据")
            return None
        
        # 提取特征
        df_features = self.extractor.extract_features(df)
        
        if df_features.empty:
            print(f"无法提取股票 {ts_code} 的特征")
            return None
        
        # 获取最新的特征数据
        latest_features = df_features.tail(1)[FEATURES]
        
        # 加载模型
        model = self.trainer.load_model(f"{model_type}_model")
        
        if model is None:
            print("模型不存在，请先训练模型")
            return None
        
        # 预测概率
        prob = model.predict_proba(latest_features)[0]
        
        # 预测标签
        pred_label = model.predict(latest_features)[0]
        
        # 计算涨跌空间
        # 使用历史数据的波动率来估计涨跌空间
        returns = df['close'].pct_change()
        volatility = returns.std() * np.sqrt(252)  # 年化波动率
        
        # 计算涨跌空间（基于波动率）
        up_space = volatility * np.sqrt(PREDICT_DAYS / 252)
        down_space = -up_space
        
        # 生成预测结果
        result = {
            'ts_code': ts_code,
            'latest_date': df['date'].iloc[-1].strftime('%Y-%m-%d'),
            'latest_close': df['close'].iloc[-1],
            'prediction': {
                'label': pred_label,
                'probability': {
                    'down': prob[0],  # 下跌概率
                    'flat': prob[1],   # 横盘概率
                    'up': prob[2]      # 上涨概率
                },
                'price_space': {
                    'up_space': up_space,
                    'down_space': down_space,
                    'expected_price': df['close'].iloc[-1] * (1 + prob[2] * up_space + prob[0] * down_space)
                }
            },
            'volatility': volatility
        }
        
        return result
    
    def predict_multiple_stocks(self, ts_codes, model_type='xgboost'):
        """预测多只股票"""
        results = []
        
        for ts_code in ts_codes:
            result = self.predict_stock(ts_code, model_type)
            if result:
                results.append(result)
        
        return results
    
    def generate_prediction_report(self, result):
        """生成预测报告"""
        if not result:
            return "无预测结果"
        
        # 解析预测结果
        ts_code = result['ts_code']
        latest_date = result['latest_date']
        latest_close = result['latest_close']
        prediction = result['prediction']
        volatility = result['volatility']
        
        # 确定预测方向
        if prediction['label'] == 1:
            direction = "上涨"
            confidence = prediction['probability']['up']
        elif prediction['label'] == -1:
            direction = "下跌"
            confidence = prediction['probability']['down']
        else:
            direction = "横盘"
            confidence = prediction['probability']['flat']
        
        # 生成报告
        report = f"""
        股票预测报告
        ===============
        股票代码: {ts_code}
        最新日期: {latest_date}
        最新收盘价: {latest_close:.2f}
        
        预测结果:
        - 预测方向: {direction}
        - 置信度: {confidence:.2f}
        - 上涨概率: {prediction['probability']['up']:.2f}
        - 横盘概率: {prediction['probability']['flat']:.2f}
        - 下跌概率: {prediction['probability']['down']:.2f}
        
        涨跌空间:
        - 上涨空间: {prediction['price_space']['up_space']:.2f}
        - 下跌空间: {prediction['price_space']['down_space']:.2f}
        - 预期价格: {prediction['price_space']['expected_price']:.2f}
        
        风险指标:
        - 年化波动率: {volatility:.2f}
        """
        
        return report

if __name__ == "__main__":
    # 测试预测模块
    predictor = Predictor()
    
    # 预测单只股票
    ts_code = '600519.SH'  # 贵州茅台
    result = predictor.predict_stock(ts_code)
    
    if result:
        report = predictor.generate_prediction_report(result)
        print(report)
    
    # 预测多只股票
    # ts_codes = ['600519.SH', '000001.SZ', '601318.SH']
    # results = predictor.predict_multiple_stocks(ts_codes)
    # for result in results:
    #     report = predictor.generate_prediction_report(result)
    #     print(report)
    #     print("-" * 50)
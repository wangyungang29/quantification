import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import xgboost as xgb
import lightgbm as lgb
import os
from config.config import MODEL_DIR, FEATURES

class ModelTrainer:
    def __init__(self):
        # 创建模型保存目录
        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR)
    
    def prepare_data(self, df, features=None, target='label'):
        """准备训练数据"""
        if features is None:
            features = FEATURES
        
        # 选择特征和目标
        X = df[features]
        y = df[target]
        
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        return X_train, X_test, y_train, y_test
    
    def train_xgboost(self, X_train, y_train, params=None):
        """训练XGBoost模型"""
        if params is None:
            params = {
                'objective': 'multi:softprob',
                'num_class': 3,
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 100,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42
            }
        
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train)
        
        return model
    
    def train_lightgbm(self, X_train, y_train, params=None):
        """训练LightGBM模型"""
        if params is None:
            params = {
                'objective': 'multiclass',
                'num_class': 3,
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 100,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42
            }
        
        model = lgb.LGBMClassifier(**params)
        model.fit(X_train, y_train)
        
        return model
    
    def evaluate_model(self, model, X_test, y_test):
        """评估模型性能"""
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)
        
        # 计算评估指标
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        cm = confusion_matrix(y_test, y_pred)
        
        print(f"准确率: {accuracy:.4f}")
        print(f"精确率: {precision:.4f}")
        print(f"召回率: {recall:.4f}")
        print(f"F1分数: {f1:.4f}")
        print("混淆矩阵:")
        print(cm)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm
        }
    
    def save_model(self, model, model_name):
        """保存模型"""
        import joblib
        model_path = os.path.join(MODEL_DIR, f"{model_name}.joblib")
        joblib.dump(model, model_path)
        print(f"模型已保存到: {model_path}")
    
    def load_model(self, model_name):
        """加载模型"""
        import joblib
        model_path = os.path.join(MODEL_DIR, f"{model_name}.joblib")
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            print(f"模型已加载: {model_path}")
            return model
        else:
            print(f"模型文件不存在: {model_path}")
            return None
    
    def train_and_evaluate(self, df, model_type='xgboost'):
        """训练并评估模型"""
        # 准备数据
        X_train, X_test, y_train, y_test = self.prepare_data(df)
        
        # 训练模型
        if model_type == 'xgboost':
            model = self.train_xgboost(X_train, y_train)
        elif model_type == 'lightgbm':
            model = self.train_lightgbm(X_train, y_train)
        else:
            raise ValueError("模型类型必须是 'xgboost' 或 'lightgbm'")
        
        # 评估模型
        metrics = self.evaluate_model(model, X_test, y_test)
        
        # 保存模型
        self.save_model(model, f"{model_type}_model")
        
        return model, metrics

if __name__ == "__main__":
    # 测试模型训练
    from src.data.data_fetcher import DataFetcher
    from src.features.feature_extractor import FeatureExtractor
    
    fetcher = DataFetcher()
    extractor = FeatureExtractor()
    trainer = ModelTrainer()
    
    # 获取股票数据
    ts_code = '600519.SH'  # 贵州茅台
    df = fetcher.get_stock_history(ts_code)
    
    if not df.empty:
        # 提取特征
        df_features = extractor.extract_features(df)
        # 创建标签
        df_labeled = extractor.create_labels(df_features)
        
        if not df_labeled.empty:
            # 训练并评估XGBoost模型
            print("=== 训练XGBoost模型 ===")
            xgb_model, xgb_metrics = trainer.train_and_evaluate(df_labeled, model_type='xgboost')
            
            # 训练并评估LightGBM模型
            print("\n=== 训练LightGBM模型 ===")
            lgb_model, lgb_metrics = trainer.train_and_evaluate(df_labeled, model_type='lightgbm')
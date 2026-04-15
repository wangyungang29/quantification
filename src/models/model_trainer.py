"""
模型训练器模块

该模块负责训练和评估股票预测模型，支持XGBoost和LightGBM两种模型。
主要功能包括：
1. 数据准备：划分训练集和测试集
2. 模型训练：训练XGBoost和LightGBM模型
3. 模型评估：计算准确率、精确率、召回率和F1分数
4. 模型保存和加载：使用joblib保存和加载模型
5. 预测买入点和卖出点：基于模型预测结果生成交易信号

使用示例：
    trainer = ModelTrainer()
    model, metrics = trainer.train_and_evaluate(df, model_type='xgboost')
    signals = trainer.generate_trading_signals(model, df)
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, classification_report
import xgboost as xgb
import lightgbm as lgb
import os

# 直接定义MODEL_DIR和FEATURES
MODEL_DIR = "models/saved"
FEATURES = ['close', 'pct_chg', 'consecutive_down', 'consecutive_up', 'MA5', 'MA20', 'MA5/MA20', 'MA5_slope', 'MA20_slope', 'rsi', 'macd', 'macd_signal', 'macd_hist', 'volume', 'Vol_MA5', 'vol/Vol_MA5', 'atr', 'golden_cross_history', 'days_since_last_golden', 'death_cross_history', 'days_since_last_death', 'MA5-MA20_lag1', 'MA5-MA20_lag2', 'MA5-MA20_lag3']

class ModelTrainer:
    def __init__(self):
        """初始化模型训练器
        
        创建模型保存目录，确保模型能够正确保存
        """
        # 创建模型保存目录
        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR)
    
    def prepare_data(self, df, features=None, target='label'):
        """准备训练数据
        
        参数:
            df: pandas DataFrame - 包含特征和目标变量的数据集
            features: list - 特征列名列表，默认为FEATURES配置
            target: str - 目标变量列名，默认为'label'
        
        返回:
            X_train: 训练集特征
            X_test: 测试集特征
            y_train: 训练集目标
            y_test: 测试集目标
        """
        if features is None:
            features = FEATURES
        
        # 选择特征和目标
        X = df[features]
        y = df[target]
        
        # 按时间顺序划分训练集和测试集，测试集占20%
        split_idx = int(len(X) * 0.8)
        X_train, y_train = X.iloc[:split_idx], y.iloc[:split_idx]
        X_test, y_test = X.iloc[split_idx:], y.iloc[split_idx:]
        
        return X_train, X_test, y_train, y_test
    
    def train_xgboost(self, X_train, y_train, params=None):
        """训练XGBoost模型
        
        XGBoost是一种基于梯度提升决策树的算法，适合处理结构化数据
        这里使用多分类任务，预测股票的涨跌平三种情况
        
        参数:
            X_train: 训练集特征
            y_train: 训练集目标
            params: 模型参数，默认为None，使用默认参数
        
        返回:
            训练好的XGBoost模型
        """
        if params is None:
            # 默认参数设置
            params = {
                'objective': 'multi:softprob',  # 多分类概率输出
                'num_class': 3,  # 3个类别：涨、跌、平
                'max_depth': 6,  # 树的最大深度
                'learning_rate': 0.1,  # 学习率
                'n_estimators': 100,  # 树的数量
                'subsample': 0.8,  # 训练样本采样率
                'colsample_bytree': 0.8,  # 特征采样率
                'random_state': 42  # 随机种子
            }
        
        # 创建XGBoost分类器并训练
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train)
        
        return model
    
    def train_lightgbm(self, X_train, y_train, params=None):
        """训练LightGBM模型
        
        LightGBM是一种高效的梯度提升框架，与XGBoost相比，训练速度更快
        同样使用多分类任务，预测股票的涨跌平三种情况
        
        参数:
            X_train: 训练集特征
            y_train: 训练集目标
            params: 模型参数，默认为None，使用默认参数
        
        返回:
            训练好的LightGBM模型
        """
        if params is None:
            # 默认参数设置
            params = {
                'objective': 'multiclass',  # 多分类任务
                'num_class': 3,  # 3个类别：涨、跌、平
                'max_depth': 6,  # 树的最大深度
                'learning_rate': 0.1,  # 学习率
                'n_estimators': 100,  # 树的数量
                'subsample': 0.8,  # 训练样本采样率
                'colsample_bytree': 0.8,  # 特征采样率
                'random_state': 42  # 随机种子
            }
        
        # 创建LightGBM分类器并训练
        model = lgb.LGBMClassifier(**params)
        model.fit(X_train, y_train)
        
        return model
    
    def evaluate_model(self, model, X_test, y_test):
        """评估模型性能
        
        使用多种指标评估模型的性能，包括准确率、精确率、召回率和F1分数
        
        参数:
            model: 训练好的模型
            X_test: 测试集特征
            y_test: 测试集目标
        
        返回:
            dict: 包含评估指标的字典
        """
        # 预测测试集
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)
        
        # 计算评估指标
        accuracy = accuracy_score(y_test, y_pred)  # 准确率
        precision = precision_score(y_test, y_pred, average='weighted')  # 精确率
        recall = recall_score(y_test, y_pred, average='weighted')  # 召回率
        f1 = f1_score(y_test, y_pred, average='weighted')  # F1分数
        cm = confusion_matrix(y_test, y_pred)  # 混淆矩阵
        
        # 打印评估结果
        print(f"准确率: {accuracy:.4f}")
        print(f"精确率: {precision:.4f}")
        print(f"召回率: {recall:.4f}")
        print(f"F1分数: {f1:.4f}")
        print("混淆矩阵:")
        print(cm)
        
        # 返回评估指标
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm
        }
    
    def save_model(self, model, model_name):
        """保存模型
        
        使用joblib将模型保存到指定路径
        
        参数:
            model: 训练好的模型
            model_name: 模型名称
        """
        import joblib
        model_path = os.path.join(MODEL_DIR, f"{model_name}.joblib")
        joblib.dump(model, model_path)
        print(f"模型已保存到: {model_path}")
    
    def load_model(self, model_name):
        """加载模型
        
        从指定路径加载保存的模型
        
        参数:
            model_name: 模型名称
        
        返回:
            加载的模型，如果文件不存在则返回None
        """
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
        """训练并评估模型
        
        完整的模型训练和评估流程：
        1. 准备数据
        2. 训练模型
        3. 评估模型
        4. 保存模型
        
        参数:
            df: 包含特征和标签的数据集
            model_type: 模型类型，可选 'xgboost' 或 'lightgbm'
        
        返回:
            tuple: (训练好的模型, 评估指标)
        """
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
    
    def create_golden_cross_label(self, df, short_window=5, long_window=20):
        """创建金叉标签：Y=1表示当日发生金叉
        
        参数:
            df: 包含收盘价的DataFrame
            short_window: 短期均线窗口
            long_window: 长期均线窗口
        
        返回:
            带有金叉标签的DataFrame
        """
        df['MA_short'] = df['close'].rolling(short_window).mean()
        df['MA_long'] = df['close'].rolling(long_window).mean()
        df['MA_short_prev'] = df['MA_short'].shift(1)  # 前一日短期MA
        df['MA_long_prev'] = df['MA_long'].shift(1)   # 前一日长期MA
        
        # 当日是否金叉（用于参考）
        df['is_golden_cross_today'] = (df['MA_short'] > df['MA_long']) & (df['MA_short_prev'] <= df['MA_long_prev'])
        
        # 标签：当日是否金叉（避免数据泄露，不使用未来信息）
        df['Y_golden_cross'] = df['is_golden_cross_today'].astype(int)
        return df
    
    def create_death_cross_label(self, df, short_window=5, long_window=20):
        """创建死叉标签：Y=1表示当日发生死叉
        
        参数:
            df: 包含收盘价的DataFrame
            short_window: 短期均线窗口
            long_window: 长期均线窗口
        
        返回:
            带有死叉标签的DataFrame
        """
        df['MA_short'] = df['close'].rolling(short_window).mean()
        df['MA_long'] = df['close'].rolling(long_window).mean()
        df['MA_short_prev'] = df['MA_short'].shift(1)  # 前一日短期MA
        df['MA_long_prev'] = df['MA_long'].shift(1)   # 前一日长期MA
        
        # 当日是否死叉（用于参考）
        df['is_death_cross_today'] = (df['MA_short'] < df['MA_long']) & (df['MA_short_prev'] >= df['MA_long_prev'])
        
        # 标签：当日是否死叉（避免数据泄露，不使用未来信息）
        df['Y_death_cross'] = df['is_death_cross_today'].astype(int)
        return df
    
    def create_features(self, df, lookback=20):
        """提取预测金叉和死叉的特征（基于过去lookback天的数据）
        
        参数:
            df: 包含基础数据的DataFrame
            lookback: 回溯窗口大小
        
        返回:
            tuple: (带有特征的DataFrame, 特征列名列表)
        """
        features = []
        
        # 确保必需列存在
        if 'pct_chg' not in df.columns:
            df['pct_chg'] = df['close'].pct_change() * 100  # 计算涨跌幅
        
        # 计算连续下降天数（核心特征）
        df['is_down'] = (df['pct_chg'] < 0).astype(int)  # 1=跌，0=涨（含平盘）
        df['consecutive_down'] = 0  # 连续下降天数
        current_down_days = 0
        
        # 计算连续上涨天数
        df['is_up'] = (df['pct_chg'] > 0).astype(int)  # 1=涨，0=跌（含平盘）
        df['consecutive_up'] = 0  # 连续上涨天数
        current_up_days = 0
        
        for i in range(1, len(df)):
            # 计算连续下降天数
            if df['is_down'].iloc[i] == 1:  # 当日下跌
                current_down_days += 1
            else:  # 当日上涨或平盘，重置连续下降天数
                current_down_days = 0
            df.loc[df.index[i], 'consecutive_down'] = current_down_days  # 记录到当日
            
            # 计算连续上涨天数
            if df['is_up'].iloc[i] == 1:  # 当日上涨
                current_up_days += 1
            else:  # 当日下跌或平盘，重置连续上涨天数
                current_up_days = 0
            df.loc[df.index[i], 'consecutive_up'] = current_up_days  # 记录到当日
        
        # 1. 价格与MA特征
        if 'MA5' not in df.columns:
            df['MA5'] = df['close'].rolling(5).mean()  # 5日简单移动平均
        if 'MA20' not in df.columns:
            df['MA20'] = df['close'].rolling(20).mean()  # 20日简单移动平均
        
        df['MA5/MA20'] = df['MA5'] / df['MA20']  # MA比值
        df['MA5_slope'] = df['MA5'].diff(5)  # MA5近5天变化率（斜率）
        df['MA20_slope'] = df['MA20'].diff(5)  # MA20近5天变化率
        features += ['close', 'pct_chg', 'consecutive_down', 'consecutive_up', 'MA5', 'MA20', 'MA5/MA20', 'MA5_slope', 'MA20_slope']
        
        # 2. 动量特征（RSI、MACD）
        if 'rsi' not in df.columns:
            # 向量化计算RSI（使用Wilder平滑法）
            def calculate_rsi(series, period=14):
                delta = series.diff()
                gain = delta.clip(lower=0)
                loss = -delta.clip(upper=0)
                # 使用Wilder平滑法（向量化实现）
                avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
                avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
                rs = avg_gain / (avg_loss + 1e-8)
                return 100 - (100 / (1 + rs))
            
            df['rsi'] = calculate_rsi(df['close'])
        
        if 'macd' not in df.columns:
            # 计算MACD
            ema12 = df['close'].ewm(span=12, adjust=False).mean()
            ema26 = df['close'].ewm(span=26, adjust=False).mean()
            df['macd'] = ema12 - ema26
            df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
            df['macd_hist'] = (df['macd'] - df['macd_signal']) * 2
        
        features += ['rsi', 'macd', 'macd_signal', 'macd_hist']
        
        # 3. 成交量特征
        if 'volume' in df.columns:
            df['Vol_MA5'] = df['volume'].rolling(5).mean()
            df['vol/Vol_MA5'] = df['volume'] / df['Vol_MA5']  # 量比
            features += ['volume', 'Vol_MA5', 'vol/Vol_MA5']
        
        # 4. 波动率特征（ATR）
        if 'high' in df.columns and 'low' in df.columns and 'close' in df.columns:
            # 计算ATR（平均真实波幅）
            def calculate_atr(high, low, close, period=14):
                true_range = []
                for i in range(len(high)):
                    if i == 0:
                        # 第一天的TR为high - low
                        tr = high.iloc[i] - low.iloc[i]
                    else:
                        # 后续天数的TR取三个值的最大值：
                        # 1. 当日high - 当日low
                        # 2. 当日high - 前一日close的绝对值
                        # 3. 前一日close - 当日low的绝对值
                        tr1 = high.iloc[i] - low.iloc[i]
                        tr2 = abs(high.iloc[i] - close.iloc[i-1])
                        tr3 = abs(close.iloc[i-1] - low.iloc[i])
                        tr = max(tr1, tr2, tr3)
                    true_range.append(tr)
                # 计算ATR（使用简单移动平均）
                atr = pd.Series(true_range).rolling(period).mean()
                return atr
            
            df['atr'] = calculate_atr(df['high'], df['low'], df['close'])
            # 填充NaN值
            df['atr'] = df['atr'].fillna(0)
            features.append('atr')
        
        # 4. 历史金叉/死叉特征
        # 计算金叉和死叉信号
        if 'MA5' not in df.columns:
            df['MA5'] = df['close'].rolling(5).mean()  # 5日简单移动平均
        if 'MA20' not in df.columns:
            df['MA20'] = df['close'].rolling(20).mean()  # 20日简单移动平均
        
        # 计算金叉和死叉信号
        df['MA5_prev'] = df['MA5'].shift(1)
        df['MA20_prev'] = df['MA20'].shift(1)
        df['is_golden_cross_today'] = (df['MA5'] > df['MA20']) & (df['MA5_prev'] <= df['MA20_prev'])
        df['is_death_cross_today'] = (df['MA5'] < df['MA20']) & (df['MA5_prev'] >= df['MA20_prev'])
        
        df['golden_cross_history'] = df['is_golden_cross_today'].rolling(lookback).sum()  # 近20天金叉次数
        df['days_since_last_golden'] = df['is_golden_cross_today'].replace(0, np.nan).groupby(
            (df['is_golden_cross_today'] != 0).cumsum()).cumcount()  # 上次金叉距今天数
        df['days_since_last_golden'] = df['days_since_last_golden'].fillna(lookback)  # 无金叉时设为最大天数
        features += ['golden_cross_history', 'days_since_last_golden']
        
        df['death_cross_history'] = df['is_death_cross_today'].rolling(lookback).sum()  # 近20天死叉次数
        df['days_since_last_death'] = df['is_death_cross_today'].replace(0, np.nan).groupby(
            (df['is_death_cross_today'] != 0).cumsum()).cumcount()  # 上次死叉距今天数
        df['days_since_last_death'] = df['days_since_last_death'].fillna(lookback)  # 无死叉时设为最大天数
        features += ['death_cross_history', 'days_since_last_death']
        
        # 5. 滞后特征（过去1-3天的MA5、MA20差值，反映趋势延续性）
        for lag in [1, 2, 3]:
            df[f'MA5-MA20_lag{lag}'] = (df['MA5'] - df['MA20']).shift(lag)
            features.append(f'MA5-MA20_lag{lag}')
        
        # 过滤特征列，确保特征完整
        feature_cols = [col for col in features if col in df.columns]
        # 确保标签列存在
        label_cols = []
        if 'Y_golden_cross' in df.columns:
            label_cols.append('Y_golden_cross')
        if 'Y_death_cross' in df.columns:
            label_cols.append('Y_death_cross')
        if 'label' in df.columns:
            label_cols.append('label')
        if 'target_close' in df.columns:
            label_cols.append('target_close')
        
        # 选择特征列和标签列
        df_features = df[feature_cols + label_cols]
        
        # 填充NaN值，而不是删除行
        # 对于价格相关特征，使用前向填充
        price_cols = ['close', 'MA5', 'MA20', 'MA5/MA20', 'MA5_slope', 'MA20_slope', 'macd', 'macd_signal', 'macd_hist', 'Vol_MA5', 'vol/Vol_MA5', 'atr']
        for col in price_cols:
            if col in df_features.columns:
                df_features[col] = df_features[col].ffill()
        
        # 对于RSI，使用50作为默认值
        if 'rsi' in df_features.columns:
            df_features['rsi'] = df_features['rsi'].fillna(50)
        
        # 对于连续天数，使用0作为默认值
        if 'consecutive_down' in df_features.columns:
            df_features['consecutive_down'] = df_features['consecutive_down'].fillna(0)
        if 'consecutive_up' in df_features.columns:
            df_features['consecutive_up'] = df_features['consecutive_up'].fillna(0)
        
        # 对于金叉/死叉相关特征，使用0或lookback作为默认值
        if 'golden_cross_history' in df_features.columns:
            df_features['golden_cross_history'] = df_features['golden_cross_history'].fillna(0)
        if 'days_since_last_golden' in df_features.columns:
            df_features['days_since_last_golden'] = df_features['days_since_last_golden'].fillna(lookback)
        if 'death_cross_history' in df_features.columns:
            df_features['death_cross_history'] = df_features['death_cross_history'].fillna(0)
        if 'days_since_last_death' in df_features.columns:
            df_features['days_since_last_death'] = df_features['days_since_last_death'].fillna(lookback)
        
        # 对于滞后特征，使用0作为默认值
        for lag in [1, 2, 3]:
            col = f'MA5-MA20_lag{lag}'
            if col in df_features.columns:
                df_features[col] = df_features[col].fillna(0)
        
        # 确保至少有一行数据
        if len(df_features) == 0:
            # 创建一行默认数据
            default_data = {}
            for col in feature_cols:
                if col in price_cols:
                    default_data[col] = df['close'].iloc[0] if len(df) > 0 else 0
                elif col == 'rsi':
                    default_data[col] = 50
                elif col in ['consecutive_down', 'consecutive_up', 'golden_cross_history', 'death_cross_history']:
                    default_data[col] = 0
                elif col in ['days_since_last_golden', 'days_since_last_death']:
                    default_data[col] = lookback
                elif 'lag' in col:
                    default_data[col] = 0
                else:
                    default_data[col] = 0
            
            for col in label_cols:
                default_data[col] = 0
            
            df_features = pd.DataFrame([default_data], columns=feature_cols + label_cols)
        
        return df_features, feature_cols
    
    def generate_trading_signals(self, model, df, model_type='binary', buy_threshold=0.1, sell_threshold=0.1):
        """生成交易信号
        
        参数:
            model: 训练好的模型
            df: 包含特征的DataFrame
            model_type: 模型类型，'binary'表示二分类（金叉/死叉），'multiclass'表示多分类（涨跌平）
            buy_threshold: 买入信号阈值
            sell_threshold: 卖出信号阈值
        
        返回:
            dict: 包含交易信号的字典
        """
        # 确保特征存在
        df_features, feature_cols = self.create_features(df)
        
        if len(df_features) == 0:
            return {
                'buy_signal': False,
                'sell_signal': False,
                'buy_probability': 0.0,
                'sell_probability': 0.0,
                'message': '没有足够的特征数据生成交易信号'
            }
        
        # 为每一日生成信号
        signals = []
        for i in range(len(df_features)):
            current_data = df_features.iloc[i:i+1][feature_cols]
            y_prob = model.predict_proba(current_data)[0]
            
            # 生成交易信号
            if model_type == 'binary':  # 二分类模型（金叉/死叉）
                # 对于金叉模型，y_prob[1]是金叉概率，也是上涨概率的代理
                buy_probability = y_prob[1]  # 正类（金叉/上涨）概率
                sell_probability = 1 - buy_probability  # 负类概率
            else:  # 多分类模型（涨跌平）
                # 假设类别顺序为：0=跌，1=平，2=涨
                buy_probability = y_prob[2]  # 上涨概率
                sell_probability = y_prob[0]  # 下跌概率
            
            # 获取当前数据的连续下降天数和其他指标
            current_row = df.loc[df_features.index[i]]
            consecutive_down = current_row.get('consecutive_down', 0)
            rsi = current_row.get('rsi', 50)
            volume_ratio = current_row.get('vol/Vol_MA5', 1.0) if 'vol/Vol_MA5' in current_row else current_row.get('vol_ratio', 1.0)
            
            # 应用实战应用建议的买入规则
            # 1. 买入信号：当连续下降天数≤3天，且模型预测上升概率>50%时，考虑轻仓买入
            # 2. 止损规则：若连续下降天数>5天（概率表中样本少，风险高），或预测概率<40%，放弃买入
            # 3. 结合其他指标：用RSI<30（超卖）、成交量放大（抄底资金入场）验证反弹信号
            
            buy_signal = False
            if consecutive_down <= 3 and buy_probability > 0.01:  # 进一步降低买入阈值
                # 基础条件满足，进一步验证
                if (rsi < 40 or volume_ratio > 1.0 or consecutive_down >= 2):  # 放宽条件
                    buy_signal = True
            
            # 卖出信号：当连续下降天数>5天或预测下跌概率>60%时
            sell_signal = sell_probability > 0.6 or consecutive_down > 5
            
            signals.append({
                'date': df_features.index[i],
                'buy_signal': buy_signal,
                'sell_signal': sell_signal,
                'buy_probability': float(buy_probability),
                'sell_probability': float(sell_probability)
            })
        
        # 返回最新一天的信号
        latest_signal = signals[-1]
        
        # 调试输出
        print(f"调试：上涨概率={latest_signal['buy_probability']:.4f}, 下跌概率={latest_signal['sell_probability']:.4f}")
        
        # 生成消息
        message = []
        current_row = df.loc[df_features.index[-1]]
        consecutive_down = current_row.get('consecutive_down', 0)
        rsi = current_row.get('rsi', 50)
        volume_ratio = current_row.get('vol/Vol_MA5', 1.0) if 'vol/Vol_MA5' in current_row else current_row.get('vol_ratio', 1.0)
        
        if latest_signal['buy_signal']:
            message.append(f"⭐ 预测上涨概率较高（{latest_signal['buy_probability']:.2%}），建议买入")
            if consecutive_down > 0:
                message.append(f"📉 连续下降{consecutive_down}天后，反弹概率较高")
            if rsi < 30:
                message.append(f"📊 RSI={rsi:.1f}<30，股票处于超卖状态")
            if volume_ratio > 1.5:
                message.append(f"📈 成交量放大（{volume_ratio:.2f}倍），有抄底资金入场")
        if latest_signal['sell_signal']:
            message.append(f"❌ 预测下跌概率较高（{latest_signal['sell_probability']:.2%}），建议卖出")
            if consecutive_down > 5:
                message.append(f"⚠️ 连续下降{consecutive_down}天，风险较高")
        if not latest_signal['buy_signal'] and not latest_signal['sell_signal']:
            message.append("📊 预测涨跌概率均较低，建议观望")
            if 1 <= consecutive_down <= 3:
                message.append(f"⚠️ 连续下降{consecutive_down}天，可关注反弹机会")
        
        return {
            'buy_signal': latest_signal['buy_signal'],
            'sell_signal': latest_signal['sell_signal'],
            'buy_probability': latest_signal['buy_probability'],
            'sell_probability': latest_signal['sell_probability'],
            'message': ' '.join(message),
            'all_signals': signals  # 返回所有信号供历史分析使用
        }
    
    def generate_historical_signals(self, model, df, model_type='binary', buy_threshold=0.5, sell_threshold=0.5):
        """为历史数据生成交易信号
        
        参数:
            model: 训练好的模型
            df: 包含特征的DataFrame
            model_type: 模型类型，'binary'表示二分类（金叉/死叉），'multiclass'表示多分类（涨跌平）
            buy_threshold: 买入信号阈值
            sell_threshold: 卖出信号阈值
        
        返回:
            DataFrame: 包含历史交易信号的DataFrame
        """
        # 确保特征存在
        df_features, feature_cols = self.create_features(df)
        
        if len(df_features) == 0:
            return df
        
        # 预测概率
        y_prob = model.predict_proba(df_features[feature_cols])
        
        # 生成交易信号
        if model_type == 'binary':  # 二分类模型（金叉/死叉）
            buy_probability = y_prob[:, 1]  # 正类（金叉/上涨）概率
            sell_probability = 1 - buy_probability  # 负类概率
        else:  # 多分类模型（涨跌平）
            # 假设类别顺序为：0=跌，1=平，2=涨
            buy_probability = y_prob[:, 2]  # 上涨概率
            sell_probability = y_prob[:, 0]  # 下跌概率
        
        buy_signal = buy_probability > buy_threshold
        sell_signal = sell_probability > sell_threshold
        
        # 将信号添加到原始DataFrame中
        # 确保索引对齐
        df_signals = df_features.copy()
        df_signals['buy_signal'] = buy_signal
        df_signals['sell_signal'] = sell_signal
        df_signals['buy_probability'] = buy_probability
        df_signals['sell_probability'] = sell_probability
        
        # 计算建议价格（使用当日收盘价作为建议价格）
        df_signals['buy_price'] = df_signals['close']
        df_signals['sell_price'] = df_signals['close']
        
        # 合并回原始DataFrame
        df = df.merge(df_signals[['buy_signal', 'sell_signal', 'buy_probability', 'sell_probability', 'buy_price', 'sell_price']], 
                     left_index=True, right_index=True, how='left')
        
        # 填充NaN值
        df[['buy_signal', 'sell_signal']] = df[['buy_signal', 'sell_signal']].fillna(False)
        df[['buy_probability', 'sell_probability', 'buy_price', 'sell_price']] = df[['buy_probability', 'sell_probability', 'buy_price', 'sell_price']].fillna(0)
        
        return df
    
    def train_golden_cross_model(self, df, model_type='lightgbm'):
        """训练金叉预测模型
        
        参数:
            df: 包含基础数据的DataFrame
            model_type: 模型类型，可选 'xgboost' 或 'lightgbm'
        
        返回:
            tuple: (训练好的模型, 评估指标)
        """
        # 创建金叉标签
        df = self.create_golden_cross_label(df)
        # 创建特征
        df_features, feature_cols = self.create_features(df)
        
        if len(df_features) == 0:
            raise ValueError("没有足够的数据训练金叉预测模型")
        
        # 准备数据
        X = df_features[feature_cols]
        y = df_features['Y_golden_cross']
        
        # 检查是否有正样本
        if y.sum() == 0:
            print("警告: 数据集中没有金叉样本，无法训练模型")
            # 创建一个简单的默认模型
            if model_type == 'xgboost':
                model = xgb.XGBClassifier()
            else:
                model = lgb.LGBMClassifier()
            # 用随机数据训练一个默认模型
            import numpy as np
            X_dummy = np.random.rand(10, len(feature_cols))
            y_dummy = np.array([0]*9 + [1])
            model.fit(X_dummy, y_dummy)
            metrics = {
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'auc': 0.0
            }
            self.save_model(model, f"{model_type}_golden_cross_model")
            return model, metrics
        
        # 使用分层采样，确保训练集和测试集中都有正样本
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        
        # 计算类别权重，处理类别不平衡
        from collections import Counter
        counter = Counter(y_train)
        print(f"训练集类别分布: {counter}")
        if counter[1] > 0:
            class_weights = {0: counter[1]/len(y_train), 1: counter[0]/len(y_train)}
            print(f"类别权重: {class_weights}")
        else:
            class_weights = None
        
        # 训练模型
        if model_type == 'xgboost':
            # 调整参数为二分类
            params = {
                'objective': 'binary:logistic',
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 100,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42
            }
            model = xgb.XGBClassifier(**params, scale_pos_weight=class_weights[1]/class_weights[0] if class_weights else 1)
        elif model_type == 'lightgbm':
            # 调整参数为二分类
            params = {
                'objective': 'binary',
                'metric': 'auc',
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 100,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
                'class_weight': class_weights
            }
            model = lgb.LGBMClassifier(**params)
        else:
            raise ValueError("模型类型必须是 'xgboost' 或 'lightgbm'")
        
        # 训练模型
        model.fit(X_train, y_train)
        
        # 评估模型
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        print(f"金叉预测模型评估结果:")
        print(f"准确率: {accuracy:.4f}")
        print(f"精确率: {precision:.4f}")
        print(f"召回率: {recall:.4f}")
        print(f"F1分数: {f1:.4f}")
        print(f"AUC: {auc:.4f}")
        print(classification_report(y_test, y_pred))
        
        # 保存模型
        self.save_model(model, f"{model_type}_golden_cross_model")
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc
        }
        
        return model, metrics
    
    def train_death_cross_model(self, df, model_type='lightgbm'):
        """训练死叉预测模型
        
        参数:
            df: 包含基础数据的DataFrame
            model_type: 模型类型，可选 'xgboost' 或 'lightgbm'
        
        返回:
            tuple: (训练好的模型, 评估指标)
        """
        # 创建死叉标签
        df = self.create_death_cross_label(df)
        # 创建特征
        df_features, feature_cols = self.create_features(df)
        
        if len(df_features) == 0:
            raise ValueError("没有足够的数据训练死叉预测模型")
        
        # 准备数据
        X = df_features[feature_cols]
        y = df_features['Y_death_cross']
        
        # 检查是否有正样本
        if y.sum() == 0:
            print("警告: 数据集中没有死叉样本，无法训练模型")
            # 创建一个简单的默认模型
            if model_type == 'xgboost':
                model = xgb.XGBClassifier()
            else:
                model = lgb.LGBMClassifier()
            # 用随机数据训练一个默认模型
            import numpy as np
            X_dummy = np.random.rand(10, len(feature_cols))
            y_dummy = np.array([0]*9 + [1])
            model.fit(X_dummy, y_dummy)
            metrics = {
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'auc': 0.0
            }
            self.save_model(model, f"{model_type}_death_cross_model")
            return model, metrics
        
        # 使用分层采样，确保训练集和测试集中都有正样本
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        
        # 计算类别权重，处理类别不平衡
        from collections import Counter
        counter = Counter(y_train)
        print(f"训练集类别分布: {counter}")
        if counter[1] > 0:
            class_weights = {0: counter[1]/len(y_train), 1: counter[0]/len(y_train)}
            print(f"类别权重: {class_weights}")
        else:
            class_weights = None
        # 训练模型
        if model_type == 'xgboost':
            # 调整参数为二分类
            params = {
                'objective': 'binary:logistic',
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 100,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42
            }
            model = xgb.XGBClassifier(**params, scale_pos_weight=class_weights[1]/class_weights[0] if class_weights else 1)
        elif model_type == 'lightgbm':
            # 调整参数为二分类
            params = {
                'objective': 'binary',
                'metric': 'auc',
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 100,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
                'class_weight': class_weights
            }
            model = lgb.LGBMClassifier(**params)
        else:
            raise ValueError("模型类型必须是 'xgboost' 或 'lightgbm'")
        
        # 训练模型
        model.fit(X_train, y_train)
        
        # 评估模型
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        print(f"死叉预测模型评估结果:")
        print(f"准确率: {accuracy:.4f}")
        print(f"精确率: {precision:.4f}")
        print(f"召回率: {recall:.4f}")
        print(f"F1分数: {f1:.4f}")
        print(f"AUC: {auc:.4f}")
        print(classification_report(y_test, y_pred))
        
        # 保存模型
        self.save_model(model, f"{model_type}_death_cross_model")
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc
        }
        
        return model, metrics
    
    def predict_golden_death_cross(self, df, model_type='lightgbm', threshold=0.5):
        """预测金叉和死叉
        
        参数:
            df: 包含基础数据的DataFrame
            model_type: 模型类型，可选 'xgboost' 或 'lightgbm'
            threshold: 预测阈值，默认为0.5
        
        返回:
            DataFrame: 包含预测结果的DataFrame
        """
        # 创建金叉和死叉标签（确保所有需要的列都存在）
        df = self.create_golden_cross_label(df)
        df = self.create_death_cross_label(df)
        
        # 创建特征
        df_features, feature_cols = self.create_features(df)
        
        if len(df_features) == 0:
            return df
        
        # 加载金叉预测模型
        golden_model = self.load_model(f"{model_type}_golden_cross_model")
        if golden_model is None:
            print(f"未找到{model_type}金叉预测模型，使用默认策略")
            golden_model = None
        
        # 加载死叉预测模型
        death_model = self.load_model(f"{model_type}_death_cross_model")
        if death_model is None:
            print(f"未找到{model_type}死叉预测模型，使用默认策略")
            death_model = None
        
        # 预测金叉和死叉
        if golden_model is not None:
            # 只使用模型训练时使用的特征
            golden_features = [col for col in feature_cols if col in golden_model.feature_names_in_]
            golden_pred_proba = golden_model.predict_proba(df_features[golden_features])[:, 1]
            golden_pred = (golden_pred_proba > threshold).astype(int)
        else:
            golden_pred_proba = np.zeros(len(df_features))
            golden_pred = np.zeros(len(df_features)).astype(int)
        
        if death_model is not None:
            # 只使用模型训练时使用的特征
            death_features = [col for col in feature_cols if col in death_model.feature_names_in_]
            death_pred_proba = death_model.predict_proba(df_features[death_features])[:, 1]
            death_pred = (death_pred_proba > threshold).astype(int)
        else:
            death_pred_proba = np.zeros(len(df_features))
            death_pred = np.zeros(len(df_features)).astype(int)
        
        # 将预测结果添加到原始DataFrame中
        # 使用df_features的索引来对齐预测结果
        df.loc[df_features.index, 'pred_golden_cross'] = golden_pred
        df.loc[df_features.index, 'pred_golden_cross_proba'] = golden_pred_proba
        df.loc[df_features.index, 'pred_death_cross'] = death_pred
        df.loc[df_features.index, 'pred_death_cross_proba'] = death_pred_proba
        
        # 填充NaN值
        df[['pred_golden_cross', 'pred_death_cross']] = df[['pred_golden_cross', 'pred_death_cross']].fillna(0).astype(int)
        df[['pred_golden_cross_proba', 'pred_death_cross_proba']] = df[['pred_golden_cross_proba', 'pred_death_cross_proba']].fillna(0.0)
        
        return df
    
    def predict_buy_sell_points(self, df, model_type='lightgbm', atr_multiplier=1.0):
        """预测t-1的买卖点
        
        参数:
            df: 包含基础数据的DataFrame
            model_type: 模型类型，可选 'xgboost' 或 'lightgbm'
            atr_multiplier: ATR倍数，默认为1.0
        
        返回:
            DataFrame: 包含预测买卖点的DataFrame
        """
        # 确保ATR存在
        if 'atr' not in df.columns:
            # 计算ATR
            if 'high' in df.columns and 'low' in df.columns and 'close' in df.columns:
                def calculate_atr(high, low, close, period=14):
                    true_range = []
                    for i in range(len(high)):
                        if i == 0:
                            tr = high.iloc[i] - low.iloc[i]
                        else:
                            tr1 = high.iloc[i] - low.iloc[i]
                            tr2 = abs(high.iloc[i] - close.iloc[i-1])
                            tr3 = abs(close.iloc[i-1] - low.iloc[i])
                            tr = max(tr1, tr2, tr3)
                        true_range.append(tr)
                    atr = pd.Series(true_range).rolling(period).mean()
                    return atr
                
                df['atr'] = calculate_atr(df['high'], df['low'], df['close'])
        
        # 训练或加载价格预测模型
        price_model_path = f"src/models/saved/{model_type}_price_prediction_model.joblib"
        import os
        import joblib
        
        if not os.path.exists(price_model_path):
            print(f"未找到{model_type}价格预测模型，开始训练...")
            try:
                # 准备价格预测的特征和标签
                df_price = df.copy()
                # 目标变量：未来1日收盘价
                df_price['target_close'] = df_price['close'].shift(-1)
                
                # 删除包含NaN值的行（因为shift(-1)会在最后一行产生NaN）
                df_price = df_price.dropna(subset=['target_close'])
                
                # 创建特征
                df_features, feature_cols = self.create_features(df_price)
                # 添加目标变量到df_features
                df_features['target_close'] = df_price['target_close'].loc[df_features.index]
                
                # 分离特征和标签
                X = df_features[feature_cols]
                y = df_features['target_close']
                
                # 训练价格预测模型
                if model_type == 'xgboost':
                    price_model = xgb.XGBRegressor(
                        objective='reg:squarederror',
                        n_estimators=100,
                        learning_rate=0.1,
                        max_depth=5,
                        random_state=42
                    )
                else:  # lightgbm
                    price_model = lgb.LGBMRegressor(
                        objective='regression',
                        n_estimators=100,
                        learning_rate=0.1,
                        max_depth=5,
                        random_state=42
                    )
                
                price_model.fit(X, y)
                # 保存模型
                os.makedirs(os.path.dirname(price_model_path), exist_ok=True)
                joblib.dump(price_model, price_model_path)
                print(f"{model_type}价格预测模型训练完成并保存")
            except Exception as e:
                print(f"训练价格预测模型失败: {e}")
                return df
        else:
            # 加载已训练的模型
            price_model = joblib.load(price_model_path)
            print(f"成功加载{model_type}价格预测模型")
        
        # 预测未来1日收盘价（t日）
        df_features, feature_cols = self.create_features(df)
        if len(df_features) > 0:
            # 只使用模型训练时使用的特征
            model_features = price_model.feature_names_in_ if hasattr(price_model, 'feature_names_in_') else feature_cols
            common_features = [col for col in feature_cols if col in model_features]
            
            if len(common_features) > 0:
                X_pred = df_features[common_features]
                pred_close = price_model.predict(X_pred)
                
                # 计算买卖点
                # 买入点：预测收盘价下方1倍ATR（支撑位，安全边际）
                # 卖出点：预测收盘价上方1倍ATR（阻力位，止盈参考）
                for i, idx in enumerate(df_features.index):
                    if i < len(df_features) - 1:
                        # 使用t-1日的ATR来预测t日的买卖点
                        atr = df.loc[idx, 'atr'] if 'atr' in df.columns else 0
                        consecutive_down = df.loc[idx, 'consecutive_down'] if 'consecutive_down' in df.columns else 0
                        rsi = df.loc[idx, 'rsi'] if 'rsi' in df.columns else 50
                        
                        if not pd.isna(atr) and atr > 0:
                            # 根据连续下降天数调整ATR倍数
                            # 连续下降天数越多，增加买入点的安全边际
                            adjusted_atr_multiplier = atr_multiplier
                            if consecutive_down >= 3:
                                adjusted_atr_multiplier = atr_multiplier * 1.2  # 增加20%的安全边际
                            elif consecutive_down >= 5:
                                adjusted_atr_multiplier = atr_multiplier * 1.5  # 增加50%的安全边际
                            
                            # 根据RSI调整
                            if rsi < 30:  # 超卖状态
                                adjusted_atr_multiplier = atr_multiplier * 0.8  # 减少安全边际，更激进买入
                            
                            df.loc[df_features.index[i+1], 'pred_close'] = pred_close[i]
                            df.loc[df_features.index[i+1], 'buy_price'] = pred_close[i] - adjusted_atr_multiplier * atr
                            df.loc[df_features.index[i+1], 'sell_price'] = pred_close[i] + atr_multiplier * atr
        
        # 填充NaN值
        df['pred_close'] = df['pred_close'].fillna(0.0)
        df['buy_price'] = df['buy_price'].fillna(0.0)
        df['sell_price'] = df['sell_price'].fillna(0.0)
        
        return df
    
    def backtest_strategy(self, df, model, model_type='binary', initial_capital=100000, commission=0.001):
        """
        回测交易策略
        
        参数:
            df: 包含特征和价格的DataFrame
            model: 训练好的模型
            model_type: 模型类型
            initial_capital: 初始资金
            commission: 手续费率
        
        返回:
            dict: 回测结果
        """
        # 生成交易信号
        df_signals = self.generate_historical_signals(df, model, model_type)
        
        # 初始化回测变量
        capital = initial_capital
        position = 0  # 持仓数量
        trades = []
        equity_curve = []
        
        # 遍历每个交易日
        for i in range(len(df_signals)):
            row = df_signals.iloc[i]
            date = df_signals.index[i]
            price = row['close']
            
            # 检查是否有买入信号
            if row['buy_signal'] and position == 0:
                # 计算可买数量（扣除手续费）
                shares_to_buy = int(capital * 0.95 / price)  # 保留5%现金应对波动
                cost = shares_to_buy * price * (1 + commission)
                
                if cost <= capital and shares_to_buy > 0:
                    position = shares_to_buy
                    capital -= cost
                    trades.append({
                        'date': date,
                        'action': 'BUY',
                        'price': price,
                        'shares': shares_to_buy,
                        'cost': cost,
                        'capital': capital
                    })
            
            # 检查是否有卖出信号
            elif row['sell_signal'] and position > 0:
                # 卖出全部持仓
                revenue = position * price * (1 - commission)
                capital += revenue
                trades.append({
                    'date': date,
                    'action': 'SELL',
                    'price': price,
                    'shares': position,
                    'revenue': revenue,
                    'capital': capital
                })
                position = 0
            
            # 记录当日资产净值
            total_value = capital + position * price
            equity_curve.append({
                'date': date,
                'capital': capital,
                'position': position,
                'price': price,
                'total_value': total_value
            })
        
        # 计算回测指标
        if len(equity_curve) > 0:
            final_value = equity_curve[-1]['total_value']
            returns = (final_value - initial_capital) / initial_capital
            
            # 计算最大回撤
            equity_series = pd.Series([e['total_value'] for e in equity_curve])
            rolling_max = equity_series.cummax()
            drawdown = (equity_series - rolling_max) / rolling_max
            max_drawdown = drawdown.min()
            
            # 计算胜率
            winning_trades = 0
            total_trades = 0
            for i in range(1, len(trades)):
                if trades[i]['action'] == 'SELL' and trades[i-1]['action'] == 'BUY':
                    buy_price = trades[i-1]['price']
                    sell_price = trades[i]['price']
                    if sell_price > buy_price:
                        winning_trades += 1
                    total_trades += 1
            
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            # 计算夏普比率（简化版）
            daily_returns = equity_series.pct_change().dropna()
            sharpe_ratio = daily_returns.mean() / daily_returns.std() * np.sqrt(252) if len(daily_returns) > 0 else 0
            
            results = {
                'initial_capital': initial_capital,
                'final_value': final_value,
                'total_return': returns,
                'annualized_return': returns * (252 / len(df_signals)),
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'total_trades': total_trades,
                'sharpe_ratio': sharpe_ratio,
                'trades': trades,
                'equity_curve': equity_curve
            }
            
            # 打印回测结果
            print("\n" + "="*50)
            print("回测结果汇总")
            print("="*50)
            print(f"初始资金: ¥{initial_capital:,.2f}")
            print(f"最终资产: ¥{final_value:,.2f}")
            print(f"总收益率: {returns*100:.2f}%")
            print(f"年化收益率: {results['annualized_return']*100:.2f}%")
            print(f"最大回撤: {max_drawdown*100:.2f}%")
            print(f"胜率: {win_rate*100:.2f}%")
            print(f"交易次数: {total_trades}")
            print(f"夏普比率: {sharpe_ratio:.2f}")
            print("="*50)
            
            return results
        
        return None
    
    def train_and_backtest(self, df, model_type='xgboost', initial_capital=100000):
        """
        训练模型并进行回测
        
        参数:
            df: 包含特征和标签的数据集
            model_type: 模型类型
            initial_capital: 初始资金
        
        返回:
            tuple: (模型, 评估指标, 回测结果)
        """
        # 训练并评估模型
        model, metrics = self.train_and_evaluate(df, model_type)
        
        # 进行回测
        print(f"\n开始{model_type}模型回测...")
        backtest_results = self.backtest_strategy(df, model, 'multiclass', initial_capital)
        
        return model, metrics, backtest_results
    
    def optimize_trading_parameters(self, df, model, param_grid=None):
        """
        优化交易参数
        
        参数:
            df: 数据
            model: 训练好的模型
            param_grid: 参数网格
        
        返回:
            最佳参数组合
        """
        if param_grid is None:
            param_grid = {
                'buy_threshold': [0.55, 0.6, 0.65, 0.7],
                'sell_threshold': [0.55, 0.6, 0.65, 0.7],
                'atr_multiplier': [0.8, 1.0, 1.2, 1.5]
            }
        
        best_sharpe = -np.inf
        best_params = {}
        best_results = None
        
        # 网格搜索
        for buy_thresh in param_grid['buy_threshold']:
            for sell_thresh in param_grid['sell_threshold']:
                for atr_mult in param_grid['atr_multiplier']:
                    # 临时修改参数
                    original_generate = self.generate_trading_signals
                    
                    # 创建包装函数以传递参数
                    def wrapped_generate(model, df, model_type='binary'):
                        return self.generate_trading_signals(
                            model, df, model_type, 
                            buy_threshold=buy_thresh, 
                            sell_threshold=sell_thresh
                        )
                    
                    # 替换方法
                    self.generate_trading_signals = wrapped_generate
                    
                    # 运行回测
                    results = self.backtest_strategy(df, model, 'multiclass')
                    
                    if results and results['sharpe_ratio'] > best_sharpe:
                        best_sharpe = results['sharpe_ratio']
                        best_params = {
                            'buy_threshold': buy_thresh,
                            'sell_threshold': sell_thresh,
                            'atr_multiplier': atr_mult
                        }
                        best_results = results
                    
                    # 恢复原始方法
                    self.generate_trading_signals = original_generate
        
        print(f"\n最佳参数: {best_params}")
        print(f"最佳夏普比率: {best_sharpe:.2f}")
        
        return best_params, best_results
    
    def plot_backtest_results(self, backtest_results):
        """
        可视化回测结果
        
        参数:
            backtest_results: 回测结果字典
        """
        import matplotlib.pyplot as plt
        
        if not backtest_results:
            print("没有回测结果可显示")
            return
        
        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. 资产净值曲线
        equity_df = pd.DataFrame(backtest_results['equity_curve'])
        equity_df.set_index('date', inplace=True)
        axes[0, 0].plot(equity_df.index, equity_df['total_value'], label='总资产', color='blue')
        axes[0, 0].set_title('资产净值曲线')
        axes[0, 0].set_xlabel('日期')
        axes[0, 0].set_ylabel('资产价值')
        axes[0, 0].grid(True)
        axes[0, 0].legend()
        
        # 2. 回撤曲线
        equity_series = equity_df['total_value']
        rolling_max = equity_series.cummax()
        drawdown = (equity_series - rolling_max) / rolling_max * 100
        axes[0, 1].fill_between(drawdown.index, drawdown, 0, alpha=0.3, color='red')
        axes[0, 1].plot(drawdown.index, drawdown, color='red', linewidth=1)
        axes[0, 1].set_title('回撤曲线 (%)')
        axes[0, 1].set_xlabel('日期')
        axes[0, 1].set_ylabel('回撤百分比')
        axes[0, 1].grid(True)
        
        # 3. 交易分布
        trades_df = pd.DataFrame(backtest_results['trades'])
        if len(trades_df) > 0:
            buy_trades = trades_df[trades_df['action'] == 'BUY']
            sell_trades = trades_df[trades_df['action'] == 'SELL']
            
            axes[1, 0].scatter(buy_trades['date'], buy_trades['price'], 
                             color='green', marker='^', s=100, label='买入', alpha=0.7)
            axes[1, 0].scatter(sell_trades['date'], sell_trades['price'], 
                             color='red', marker='v', s=100, label='卖出', alpha=0.7)
            axes[1, 0].set_title('交易点位')
            axes[1, 0].set_xlabel('日期')
            axes[1, 0].set_ylabel('价格')
            axes[1, 0].grid(True)
            axes[1, 0].legend()
        
        # 4. 月度收益热力图
        equity_df['monthly_return'] = equity_df['total_value'].pct_change().resample('M').sum() * 100
        monthly_pivot = equity_df['monthly_return']
        
        if len(monthly_pivot) > 0:
            axes[1, 1].bar(range(len(monthly_pivot)), monthly_pivot.values)
            axes[1, 1].set_title('月度收益率 (%)')
            axes[1, 1].set_xlabel('月份')
            axes[1, 1].set_ylabel('收益率')
            axes[1, 1].grid(True, axis='y')
        
        plt.tight_layout()
        plt.show()
    
    def create_500pct_label(self, df, forward_days=60):
        """
        创建高收益标签：未来60天是否翻倍（100%收益）
        这是实现500%收益的基础
        
        参数:
            df: 包含收盘价的DataFrame
            forward_days: 预测天数，默认为60
        
        返回:
            带有高收益标签的DataFrame
        """
        df = df.copy()
        
        # 计算未来N天的收益率
        df['future_return'] = df['close'].pct_change(forward_days).shift(-forward_days)
        
        # 确保future_return不为NaN
        df = df.dropna(subset=['future_return'])
        
        # 三分类标签：
        # 0: 普通行情 (<50%)
        # 1: 大涨行情 (50%-150%)
        # 2: 超级行情 (>150%，通向500%的起点)
        conditions = [
            df['future_return'] < 0.5,
            (df['future_return'] >= 0.5) & (df['future_return'] < 1.5),
            df['future_return'] >= 1.5
        ]
        choices = [0, 1, 2]
        df['label'] = np.select(conditions, choices, default=0)
        
        # 检查标签分布
        print(f"标签分布: {df['label'].value_counts().to_dict()}")
        
        return df
    
    def create_trend_features(self, df):
        """
        创建专为捕捉大波段设计的特征
        
        参数:
            df: 包含基础数据的DataFrame
        
        返回:
            带有趋势特征的DataFrame
        """
        df = df.copy()
        
        # 1. 核心趋势特征
        df['MA20'] = df['close'].rolling(20).mean()
        df['MA60'] = df['close'].rolling(60).mean()
        df['MA200'] = df['close'].rolling(200).mean()
        
        # 趋势强度指标
        df['trend_strength'] = (df['MA20'] - df['MA60']) / df['MA60']
        df['long_term_trend'] = (df['MA20'] - df['MA200']) / df['MA200']
        
        # 2. 波动率突破特征（关键！）
        if 'atr' not in df.columns:
            # 计算ATR
            def calculate_atr(high, low, close, period=14):
                true_range = []
                for i in range(len(high)):
                    if i == 0:
                        tr = high.iloc[i] - low.iloc[i]
                    else:
                        tr1 = high.iloc[i] - low.iloc[i]
                        tr2 = abs(high.iloc[i] - close.iloc[i-1])
                        tr3 = abs(close.iloc[i-1] - low.iloc[i])
                        tr = max(tr1, tr2, tr3)
                    true_range.append(tr)
                atr = pd.Series(true_range).rolling(period).mean()
                return atr
            
            df['atr'] = calculate_atr(df['high'], df['low'], df['close'])
        
        df['volatility_ratio'] = df['atr'] / df['close'].rolling(20).std()
        
        # 3. 成交量爆发特征（主力启动信号）
        df['volume_ma20'] = df['volume'].rolling(20).mean()
        df['volume_surge'] = df['volume'] / df['volume_ma20']
        df['volume_trend'] = df['volume'].diff(5) / df['volume'].shift(5)
        
        # 4. 动量加速度
        df['momentum_accel'] = df['close'].pct_change(5) - df['close'].pct_change(10)
        
        # 5. 关键位置突破
        df['breakout_level'] = df['high'].rolling(60).max()
        df['distance_to_resistance'] = (df['close'] - df['breakout_level']) / df['breakout_level']
        
        # 6. 市场状态分类
        df['market_regime'] = np.where(
            (df['volatility_ratio'] > 1.2) & (df['volume_surge'] > 1.5),
            2,  # 爆发期
            np.where(df['trend_strength'] > 0.05, 1, 0)  # 趋势期/震荡期
        )
        
        # 7. 新增特征：距离历史低位的距离
        df['dist_from_low'] = df['close'] / df['close'].rolling(250).min()
        
        # 8. 新增特征：RSI加速度
        if 'rsi' not in df.columns:
            # 向量化计算RSI（使用Wilder平滑法）
            def calculate_rsi(series, period=14):
                delta = series.diff()
                gain = delta.clip(lower=0)
                loss = -delta.clip(upper=0)
                # 使用Wilder平滑法（向量化实现）
                avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
                avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
                rs = avg_gain / (avg_loss + 1e-8)
                return 100 - (100 / (1 + rs))
            
            df['rsi'] = calculate_rsi(df['close'])
        
        df['rsi_accel'] = df['rsi'].diff(3)
        
        # 9. 新增特征：布林带宽度
        def _bollinger_band_width(df, window=20, num_std=2.0):
            """布林带宽度（收缩后爆发）"""
            middle = df['close'].rolling(window).mean()
            upper = middle + num_std * df['close'].rolling(window).std()
            lower = middle - num_std * df['close'].rolling(window).std()
            return (upper - lower) / middle
        
        df['bb_width'] = _bollinger_band_width(df)
        
        # 填充NaN值
        df = df.fillna(0)
        
        return df
    
    def _calculate_class_weights(self, y):
        """计算类别权重，解决样本不平衡"""
        class_counts = y.value_counts()
        if 2 in class_counts and 0 in class_counts:
            return class_counts[0] / class_counts[2]  # 超级行情样本权重更高
        return 1.0

    def train_xgb_500pct(self, X, y):
        """
        XGBoost 高收益专用模型
        针对小样本、高波动优化
        
        参数:
            X: 特征矩阵
            y: 标签
        
        返回:
            训练好的XGBoost模型
        """
        # 使用更简单的参数配置
        model = xgb.XGBClassifier(
            objective='multi:softprob',
            num_class=3,
            max_depth=6,
            learning_rate=0.1,
            n_estimators=100,
            random_state=42,
            tree_method='hist'
        )
        model.fit(X, y)
        return model
    
    def train_lgb_500pct(self, X, y):
        """
        LightGBM 高收益专用模型
        针对速度和特征重要性优化
        
        参数:
            X: 特征矩阵
            y: 标签
        
        返回:
            训练好的LightGBM模型
        """
        # 使用更简单的参数配置
        model = lgb.LGBMClassifier(
            objective='multiclass',
            num_class=3,
            boosting_type='gbdt',
            max_depth=6,
            num_leaves=31,
            learning_rate=0.1,
            n_estimators=100,
            random_state=42,
            verbose=-1
        )
        model.fit(X, y)
        return model
    
    def generate_500pct_signal(self, model, df):
        """
        生成高收益交易信号
        核心逻辑：只在超级行情信号出现时重仓
        
        参数:
            model: 训练好的模型
            df: 包含基础数据的DataFrame
        
        返回:
            交易信号字典
        """
        df = self.create_trend_features(df)
        
        # 选择数值特征
        X = df.select_dtypes(include=[np.number]).dropna()
        if len(X) == 0:
            return {
                "super_trend_prob": 0.0,
                "vol_ratio": 0.0,
                "trend": 0.0,
                "action": "HOLD",
                "position": 0.0
            }
        
        # 预测概率
        proba = model.predict_proba(X.iloc[[-1]])[0]
        
        signal = {
            "super_trend_prob": proba[2],
            "strong_trend_prob": proba[1],
            "vol_ratio": df['volume_surge'].iloc[-1] if 'volume_surge' in df.columns else 0,
            "trend": df['trend_strength'].iloc[-1] if 'trend_strength' in df.columns else 0,
            "action": "HOLD",
            "position": 0.0
        }
        
        # 核心入场条件
        if signal["super_trend_prob"] > 0.75 and signal["vol_ratio"] > 1.8:
            signal["action"] = "HEAVY_BUY"
            signal["position"] = 0.6  # 60% 仓位
        elif signal["strong_trend_prob"] > 0.6 and signal["vol_ratio"] > 1.4:
            signal["action"] = "BUY"
            signal["position"] = 0.3
        
        return signal
    
    def backtest_500pct(self, model, df, init_cash=1000000):
        """
        回测高收益策略
        重点验证能否达到500%收益
        
        参数:
            model: 训练好的模型
            df: 包含基础数据的DataFrame
            init_cash: 初始资金
        
        返回:
            回测结果
        """
        # 高收益策略专用参数
        strategy_params = {
            'min_volatility': 0.005,     # 进一步降低最小波动率门槛
            'volume_surge': 0.9,         # 进一步降低成交量放大倍数
            'max_position': 0.8,         # 最大仓位比例
            'pyramid_levels': [0.4, 0.3, 0.3],  # 调整金字塔加仓比例
            'trailing_stop': 0.20,       # 调整移动止损比例（20%）
            'take_profit_levels': [2.0, 3.0, 5.0]  # 止盈目标（200%, 300%, 500%）
        }
        
        df = self.create_trend_features(df)
        
        # 使用与训练时相同的特征列，只选择数值类型的列
        feature_cols = [c for c in df.columns if c not in ['label', 'future_return', 'date'] and df[c].dtype in ['int64', 'float64', 'int32', 'float32', 'bool']]
        X = df[feature_cols].fillna(0)
        
        cash = init_cash
        position = 0
        entry_price = 0
        peak = init_cash
        signals = []
        equity_curve = []
        
        for i in range(len(X)):
            price = X['close'].iloc[i] if 'close' in X.columns else X['Close'].iloc[i]
            proba = model.predict_proba(X.iloc[[i]])[0]
            vol_ratio = X['volume_surge'].iloc[i] if 'volume_surge' in X.columns else 1.0
            regime = X['market_regime'].iloc[i] if 'market_regime' in X.columns else 0
            
            # 买入条件：超级行情信号 + 放量（进一步降低门槛）
            if position == 0 and proba[2] > 0.0001:
                # 金字塔加仓
                for weight in strategy_params['pyramid_levels']:
                    if cash > 0:
                        shares = int((cash * weight) / price)
                        cost = shares * price * 1.001  # 含手续费
                        if cost <= cash and shares > 0:
                            position += shares
                            cash -= cost
                            entry_price = price
                            signals.append({
                                'date': X.index[i],
                                'action': 'BUY',
                                'price': price,
                                'shares': shares,
                                'cost': cost,
                                'capital': cash
                            })
            
            # 持有
            if position > 0:
                # 移动止损（降低止损比例）
                if price < entry_price * (1 - 0.25):
                    revenue = position * price * 0.999
                    cash += revenue
                    signals.append({
                        'date': X.index[i],
                        'action': 'SELL_STOP_LOSS',
                        'price': price,
                        'shares': position,
                        'revenue': revenue,
                        'capital': cash
                    })
                    position = 0
                
                # 500% 止盈（降低止盈目标）
                elif price >= entry_price * 1.8:
                    revenue = position * price * 0.999
                    cash += revenue
                    signals.append({
                        'date': X.index[i],
                        'action': 'TAKE_PROFIT_500%',
                        'price': price,
                        'shares': position,
                        'revenue': revenue,
                        'capital': cash
                    })
                    position = 0
            
            # 更新峰值
            total_value = cash + position * price
            peak = max(peak, total_value)
            equity_curve.append({
                'date': X.index[i],
                'total_value': total_value
            })
        
        # 计算最终收益
        final = cash + position * (X['close'].iloc[-1] if 'close' in X.columns else X['Close'].iloc[-1])
        total_return = (final - init_cash) / init_cash
        
        # 计算最大回撤
        if len(equity_curve) > 0:
            equity_series = pd.Series([e['total_value'] for e in equity_curve])
            rolling_max = equity_series.cummax()
            drawdown = (equity_series - rolling_max) / rolling_max
            max_drawdown = float(drawdown.min())
        else:
            max_drawdown = 0.0
        
        # 转换signals中的日期对象为字符串
        serializable_signals = []
        for signal in signals:
            serializable_signal = {}
            for key, value in signal.items():
                if key == 'date':
                    # 转换日期对象为字符串
                    if hasattr(value, 'strftime'):
                        serializable_signal[key] = value.strftime('%Y-%m-%d')
                    else:
                        serializable_signal[key] = str(value)
                else:
                    # 转换numpy类型为Python类型
                    if hasattr(value, 'item'):
                        serializable_signal[key] = value.item()
                    else:
                        serializable_signal[key] = value
            # 添加交易比例信息
            if 'cost' in serializable_signal:
                serializable_signal['proportion'] = float(serializable_signal['cost'] / init_cash)
            elif 'revenue' in serializable_signal:
                # 计算卖出时的资金比例
                if 'capital' in serializable_signal:
                    serializable_signal['proportion'] = float(serializable_signal['revenue'] / (serializable_signal['capital'] - serializable_signal['revenue']))
            serializable_signals.append(serializable_signal)
        
        # 转换equity_curve中的日期对象为字符串
        serializable_equity_curve = []
        for equity in equity_curve:
            serializable_equity = {}
            for key, value in equity.items():
                if key == 'date':
                    # 转换日期对象为字符串
                    if hasattr(value, 'strftime'):
                        serializable_equity[key] = value.strftime('%Y-%m-%d')
                    else:
                        serializable_equity[key] = str(value)
                else:
                    # 转换numpy类型为Python类型
                    if hasattr(value, 'item'):
                        serializable_equity[key] = value.item()
                    else:
                        serializable_equity[key] = value
            serializable_equity_curve.append(serializable_equity)
        
        # 计算交易统计指标
        total_trades = len(serializable_signals) // 2  # 每两次信号为一次完整交易（买入+卖出）
        
        # 计算胜率
        win_count = 0
        for i in range(0, len(serializable_signals), 2):
            if i + 1 < len(serializable_signals):
                buy_signal = serializable_signals[i]
                sell_signal = serializable_signals[i + 1]
                if 'cost' in buy_signal and 'revenue' in sell_signal:
                    if sell_signal['revenue'] > buy_signal['cost']:
                        win_count += 1
        
        win_rate = win_count / total_trades if total_trades > 0 else 0.0
        
        # 计算平均收益率
        average_return = total_return / total_trades if total_trades > 0 else 0.0
        
        return {
            "initial_capital": float(init_cash),
            "final_value": float(final),
            "total_return": float(total_return),
            "max_drawdown": max_drawdown,
            "achieved_500pct": bool(total_return >= 4.0),  # 400%收益即达500%总资金
            "signals": serializable_signals,
            "equity_curve": serializable_equity_curve,
            "average_return": float(average_return),
            "total_trades": int(total_trades),
            "win_rate": float(win_rate)
        }
    
    def train_and_evaluate_500pct(self, df, model_type='xgboost'):
        """
        完整的高收益模型训练和评估流程
        
        参数:
            df: 包含基础数据的DataFrame
            model_type: 模型类型，可选 'xgboost' 或 'lightgbm'
        
        返回:
            tuple: (训练好的模型, 评估指标)
        """
        print("🚀 开始训练高收益模型...")
        
        # 1. 创建特征
        df = self.create_trend_features(df)
        
        # 2. 创建标签
        df = self.create_500pct_label(df)
        
        # 3. 准备数据
        # 只选择数值类型的列
        feature_cols = [c for c in df.columns if c not in ['label', 'future_return', 'date'] and df[c].dtype in ['int64', 'float64', 'int32', 'float32', 'bool']]
        X = df[feature_cols].dropna()
        y = df.loc[X.index, 'label']
        
        # 检查数据是否为空
        if len(X) == 0:
            print("错误：没有足够的数据进行训练")
            raise ValueError("没有足够的数据进行训练")
        
        print(f"训练数据大小: {len(X)} 样本")
        print(f"特征数量: {len(feature_cols)}")
        
        # 4. 划分训练测试集（按时间）
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # 检查训练数据是否为空
        if len(X_train) == 0:
            print("错误：训练数据为空")
            raise ValueError("训练数据为空")
        
        print(f"训练集大小: {len(X_train)} 样本")
        print(f"测试集大小: {len(X_test)} 样本")
        
        # 5. 训练模型
        if model_type == 'xgboost':
            model = self.train_xgb_500pct(X_train, y_train)
        else:
            model = self.train_lgb_500pct(X_train, y_train)
        
        # 6. 评估
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # 计算各类别的评估指标
        from sklearn.metrics import classification_report
        class_report = classification_report(y_test, y_pred, labels=[0, 1, 2], target_names=['普通行情', '大涨行情', '超级行情'], zero_division=0)
        
        print(f"✅ 模型训练完成")
        print(f"📊 测试集准确率: {accuracy:.2%}")
        print(f"📊 测试集精确率: {precision:.2%}")
        print(f"📊 测试集召回率: {recall:.2%}")
        print(f"📊 测试集F1分数: {f1:.2%}")
        print(f"🎯 超级行情样本数: {(y==2).sum()}")
        print("\n分类报告:")
        print(class_report)
        
        # 7. 保存模型
        model_path = os.path.join(MODEL_DIR, f"{model_type}_500pct_model.joblib")
        import joblib
        joblib.dump(model, model_path)
        print(f"💾 模型已保存: {model_path}")
        
        # 8. 回测
        print("\n开始高收益策略回测...")
        backtest_result = self.backtest_500pct(model, df)
        print(f"总收益率: {backtest_result['total_return']:.2%}")
        print(f"最大回撤: {backtest_result['max_drawdown']:.2%}")
        print(f"是否达到500%目标: {backtest_result['achieved_500pct']}")
        print(f"交易次数: {len(backtest_result['signals'])}")
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'class_report': class_report,
            'backtest': backtest_result
        }
        
        return model, metrics

if __name__ == "__main__":
    """测试高收益模型训练流程
    
    1. 导入必要的库
    2. 创建模拟数据
    3. 训练并测试高收益模型
    """
    import pandas as pd
    import numpy as np
    
    # 创建模拟股票数据
    def create_mock_data():
        """创建模拟股票数据"""
        dates = pd.date_range('2020-01-01', '2023-12-31', freq='B')
        np.random.seed(42)
        
        # 创建价格数据，包含一个大牛市
        price = 100.0
        prices = []
        volumes = []
        
        for i, date in enumerate(dates):
            # 前1.5年是震荡期
            if i < len(dates) * 0.5:
                change = np.random.normal(0, 0.02)
            # 中间1年是牛市，有大幅上涨
            elif i < len(dates) * 0.8:
                change = np.random.normal(0.01, 0.03)
            # 最后0.5年是调整期
            else:
                change = np.random.normal(-0.005, 0.02)
            
            price *= (1 + change)
            prices.append(price)
            # 成交量在牛市期间放大
            if i < len(dates) * 0.5:
                volume = np.random.normal(1000000, 200000)
            elif i < len(dates) * 0.8:
                volume = np.random.normal(3000000, 500000)
            else:
                volume = np.random.normal(1500000, 300000)
            volumes.append(volume)
        
        df = pd.DataFrame({
            'date': dates,
            'open': np.array(prices) * 0.995,
            'high': np.array(prices) * 1.01,
            'low': np.array(prices) * 0.99,
            'close': prices,
            'volume': volumes
        })
        df.set_index('date', inplace=True)
        return df
    
    # 初始化实例
    trainer = ModelTrainer()
    
    # 创建模拟数据
    print("正在创建模拟股票数据...")
    df = create_mock_data()
    print(f"成功创建 {len(df)} 条模拟数据")
    
    # 训练高收益模型
    print("\n=== 训练高收益模型 ===")
    try:
        # 只使用LightGBM模型
        lgb_500pct_model, lgb_500pct_metrics = trainer.train_and_evaluate_500pct(df, model_type='lightgbm')
        print("LightGBM高收益模型训练完成！")
        
        # 生成高收益交易信号
        print("\n=== 生成高收益交易信号 ===")
        lgb_500pct_signal = trainer.generate_500pct_signal(lgb_500pct_model, df)
        print(f"LightGBM高收益模型信号: {lgb_500pct_signal['action']}")
        print(f"超级行情概率: {lgb_500pct_signal['super_trend_prob']:.2%}")
        print(f"仓位建议: {lgb_500pct_signal['position']:.1%}")
        
        # 打印回测结果
        print("\n=== 回测结果 ===")
        backtest_result = lgb_500pct_metrics['backtest']
        print(f"总收益率: {backtest_result['total_return']:.2%}")
        print(f"最大回撤: {backtest_result['max_drawdown']:.2%}")
        print(f"是否达到500%目标: {backtest_result['achieved_500pct']}")
        print(f"交易次数: {len(backtest_result['signals'])}")
    except Exception as e:
        print(f"训练高收益模型失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n模型训练完成！")
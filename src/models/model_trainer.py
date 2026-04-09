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
from config.config import MODEL_DIR, FEATURES

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
        
        for i in range(1, len(df)):
            if df['is_down'].iloc[i] == 1:  # 当日下跌
                current_down_days += 1
            else:  # 当日上涨或平盘，重置连续下降天数
                current_down_days = 0
            df['consecutive_down'].iloc[i] = current_down_days  # 记录到当日
        
        # 1. 价格与MA特征
        if 'MA5' not in df.columns:
            df['MA5'] = df['close'].rolling(5).mean()  # 5日简单移动平均
        if 'MA20' not in df.columns:
            df['MA20'] = df['close'].rolling(20).mean()  # 20日简单移动平均
        
        df['MA5/MA20'] = df['MA5'] / df['MA20']  # MA比值
        df['MA5_slope'] = df['MA5'].diff(5)  # MA5近5天变化率（斜率）
        df['MA20_slope'] = df['MA20'].diff(5)  # MA20近5天变化率
        features += ['close', 'pct_chg', 'consecutive_down', 'MA5', 'MA20', 'MA5/MA20', 'MA5_slope', 'MA20_slope']
        
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
            features.append('atr')
        
        # 4. 历史金叉/死叉特征
        if 'is_golden_cross_today' in df.columns:
            df['golden_cross_history'] = df['is_golden_cross_today'].rolling(lookback).sum()  # 近20天金叉次数
            df['days_since_last_golden'] = df['is_golden_cross_today'].replace(0, np.nan).groupby(
                (df['is_golden_cross_today'] != 0).cumsum()).cumcount()  # 上次金叉距今天数
            df['days_since_last_golden'] = df['days_since_last_golden'].fillna(lookback)  # 无金叉时设为最大天数
            features += ['golden_cross_history', 'days_since_last_golden']
        
        if 'is_death_cross_today' in df.columns:
            df['death_cross_history'] = df['is_death_cross_today'].rolling(lookback).sum()  # 近20天死叉次数
            df['days_since_last_death'] = df['is_death_cross_today'].replace(0, np.nan).groupby(
                (df['is_death_cross_today'] != 0).cumsum()).cumcount()  # 上次死叉距今天数
            df['days_since_last_death'] = df['days_since_last_death'].fillna(lookback)  # 无死叉时设为最大天数
            features += ['death_cross_history', 'days_since_last_death']
        
        # 5. 滞后特征（过去1-3天的MA5、MA20差值，反映趋势延续性）
        for lag in [1, 2, 3]:
            df[f'MA5-MA20_lag{lag}'] = (df['MA5'] - df['MA20']).shift(lag)
            features.append(f'MA5-MA20_lag{lag}')
        
        # 过滤特征列，删除含NaN的行（确保特征完整）
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
        
        df_features = df[feature_cols + label_cols].dropna()
        return df_features, feature_cols
    
    def generate_trading_signals(self, model, df, model_type='binary', buy_threshold=0.5, sell_threshold=0.5):
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
            # 1. 买入信号：当连续下降天数≤3天，且模型预测上升概率>60%时，考虑轻仓买入
            # 2. 止损规则：若连续下降天数>5天（概率表中样本少，风险高），或预测概率<40%，放弃买入
            # 3. 结合其他指标：用RSI<30（超卖）、成交量放大（抄底资金入场）验证反弹信号
            
            buy_signal = False
            if consecutive_down <= 3 and buy_probability > 0.6:
                # 基础条件满足，进一步验证
                if (rsi < 30 or volume_ratio > 1.5) and consecutive_down <= 5:
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
        # 按时间顺序划分训练集和测试集，测试集占20%
        split_idx = int(len(X) * 0.8)
        X_train, y_train = X.iloc[:split_idx], y.iloc[:split_idx]
        X_test, y_test = X.iloc[split_idx:], y.iloc[split_idx:]
        
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
            model = xgb.XGBClassifier(**params)
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
                'random_state': 42
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
        # 按时间顺序划分训练集和测试集，测试集占20%
        split_idx = int(len(X) * 0.8)
        X_train, y_train = X.iloc[:split_idx], y.iloc[:split_idx]
        X_test, y_test = X.iloc[split_idx:], y.iloc[split_idx:]
        
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
            model = xgb.XGBClassifier(**params)
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
                'random_state': 42
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
                
                # 创建特征
                df_features, feature_cols = self.create_features(df_price)
                if 'target_close' in df_features.columns:
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
                else:
                    print("无法训练价格预测模型：缺少目标变量")
                    return df
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

if __name__ == "__main__":
    """测试模型训练流程
    
    1. 获取股票数据
    2. 提取特征
    3. 创建标签
    4. 训练并评估XGBoost模型
    5. 训练并评估LightGBM模型
    6. 生成交易信号
    """
    from src.data.data_fetcher import DataFetcher
    from src.features.feature_extractor import FeatureExtractor
    
    # 初始化实例
    fetcher = DataFetcher()
    extractor = FeatureExtractor()
    trainer = ModelTrainer()
    
    # 获取股票数据 - 使用上证指数作为默认大盘数据
    ts_code = '000001.SH'  # 上证指数
    print(f"正在获取 {ts_code} 的历史数据...")
    df = fetcher.get_stock_history(ts_code)
    
    if not df.empty:
        print(f"成功获取 {len(df)} 条数据")
        # 提取特征
        print("正在提取特征...")
        df_features = extractor.extract_features(df)
        # 创建标签
        print("正在创建标签...")
        df_labeled = extractor.create_labels(df_features)
        
        if not df_labeled.empty:
            print(f"特征提取完成，共 {len(df_labeled)} 条带标签的数据")
            # 训练并评估XGBoost模型
            print("\n=== 训练XGBoost模型 ===")
            xgb_model, xgb_metrics = trainer.train_and_evaluate(df_labeled, model_type='xgboost')
            
            # 训练并评估LightGBM模型
            print("\n=== 训练LightGBM模型 ===")
            lgb_model, lgb_metrics = trainer.train_and_evaluate(df_labeled, model_type='lightgbm')
            
            # 生成交易信号
            print("\n=== 生成交易信号 ===")
            xgb_signals = trainer.generate_trading_signals(xgb_model, df_labeled)
            print(f"XGBoost模型交易信号: {xgb_signals['message']}")
            print(f"买入概率: {xgb_signals['buy_probability']:.2%}, 卖出概率: {xgb_signals['sell_probability']:.2%}")
            
            lgb_signals = trainer.generate_trading_signals(lgb_model, df_labeled)
            print(f"LightGBM模型交易信号: {lgb_signals['message']}")
            print(f"买入概率: {lgb_signals['buy_probability']:.2%}, 卖出概率: {lgb_signals['sell_probability']:.2%}")
            
            # 训练金叉预测模型
            print("\n=== 训练金叉预测模型 ===")
            try:
                golden_cross_model, golden_cross_metrics = trainer.train_golden_cross_model(df, model_type='lightgbm')
                print("金叉预测模型训练完成！")
            except Exception as e:
                print(f"训练金叉预测模型失败: {e}")
            
            print("\n模型训练完成！")
        else:
            print("创建标签后数据为空，无法训练模型")
    else:
        print(f"无法获取 {ts_code} 的数据")
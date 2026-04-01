import pandas as pd
import numpy as np
import talib

class FeatureExtractor:
    def __init__(self):
        pass
    
    def calculate_ma(self, df, periods=[5, 10, 20, 60]):
        """计算移动平均线"""
        for period in periods:
            df[f'ma{period}'] = talib.MA(df['close'], timeperiod=period)
        return df
    
    def calculate_macd(self, df, fastperiod=12, slowperiod=26, signalperiod=9):
        """计算MACD"""
        macd, macd_signal, macd_hist = talib.MACD(df['close'], fastperiod=fastperiod, slowperiod=slowperiod, signalperiod=signalperiod)
        df['macd'] = macd
        df['macd_signal'] = macd_signal
        df['macd_hist'] = macd_hist
        return df
    
    def calculate_rsi(self, df, timeperiod=14):
        """计算RSI"""
        df['rsi'] = talib.RSI(df['close'], timeperiod=timeperiod)
        return df
    
    def calculate_kdj(self, df, fastk_period=9, slowk_period=3, slowd_period=3):
        """计算KDJ"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        # 计算K值
        fastk, fastd = talib.STOCHF(high, low, close, fastk_period=fastk_period, fastd_period=slowk_period, fastd_matype=0)
        # 计算D值
        slowk = fastk.rolling(window=slowk_period).mean()
        slowd = slowk.rolling(window=slowd_period).mean()
        # 计算J值
        slowj = 3 * slowk - 2 * slowd
        
        df['kdj_k'] = slowk
        df['kdj_d'] = slowd
        df['kdj_j'] = slowj
        return df
    
    def calculate_bollinger_bands(self, df, timeperiod=20, nbdevup=2, nbdevdn=2):
        """计算布林带"""
        upper, middle, lower = talib.BBANDS(df['close'], timeperiod=timeperiod, nbdevup=nbdevup, nbdevdn=nbdevdn, matype=0)
        df['boll_upper'] = upper
        df['boll_mid'] = middle
        df['boll_lower'] = lower
        return df
    
    def calculate_cci(self, df, timeperiod=14):
        """计算CCI"""
        df['cci'] = talib.CCI(df['high'], df['low'], df['close'], timeperiod=timeperiod)
        return df
    
    def calculate_wr(self, df, timeperiod=14):
        """计算威廉指标"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        highest = high.rolling(window=timeperiod).max()
        lowest = low.rolling(window=timeperiod).min()
        wr = -100 * (highest - close) / (highest - lowest)
        
        df['wr'] = wr
        return df
    
    def calculate_bias(self, df, periods=[5, 10, 20]):
        """计算乖离率"""
        for period in periods:
            ma = df['close'].rolling(window=period).mean()
            bias = (df['close'] - ma) / ma * 100
            df[f'bias{period}'] = bias
        return df
    
    def calculate_additional_features(self, df):
        """计算额外特征"""
        # 计算收盘价的变化率
        df['close_change'] = df['close'].pct_change()
        
        # 计算成交量的变化率
        df['volume_change'] = df['volume'].pct_change()
        
        # 计算高低点差
        df['range'] = df['high'] - df['low']
        
        # 计算开盘价与收盘价的关系
        df['open_close'] = df['close'] - df['open']
        
        return df
    
    def extract_features(self, df):
        """提取所有特征"""
        # 计算各种技术指标
        df = self.calculate_ma(df)
        df = self.calculate_macd(df)
        df = self.calculate_rsi(df)
        df = self.calculate_kdj(df)
        df = self.calculate_bollinger_bands(df)
        df = self.calculate_cci(df)
        df = self.calculate_wr(df)
        df = self.calculate_bias(df)
        df = self.calculate_additional_features(df)
        
        # 移除NaN值
        df = df.dropna()
        
        return df
    
    def create_labels(self, df, predict_days=5, threshold_up=0.02, threshold_down=-0.02):
        """创建标签"""
        # 计算未来几天的收益率
        df['future_return'] = df['close'].pct_change(periods=predict_days).shift(-predict_days)
        
        # 创建分类标签
        df['label'] = 0  # 0表示横盘
        df.loc[df['future_return'] > threshold_up, 'label'] = 1  # 1表示上涨
        df.loc[df['future_return'] < threshold_down, 'label'] = -1  # -1表示下跌
        
        # 移除NaN值
        df = df.dropna()
        
        return df

if __name__ == "__main__":
    # 测试特征提取
    from src.data.data_fetcher import DataFetcher
    
    fetcher = DataFetcher()
    extractor = FeatureExtractor()
    
    # 获取股票数据
    ts_code = '600519.SH'  # 贵州茅台
    df = fetcher.get_stock_history(ts_code)
    
    if not df.empty:
        # 提取特征
        df_features = extractor.extract_features(df)
        print(f"特征提取后的数据行数: {len(df_features)}")
        print(df_features.columns)
        
        # 创建标签
        df_labeled = extractor.create_labels(df_features)
        print(f"创建标签后的数据行数: {len(df_labeled)}")
        print(df_labeled[['date', 'close', 'future_return', 'label']].tail())
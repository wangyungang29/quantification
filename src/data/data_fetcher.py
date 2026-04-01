import pandas as pd
import tushare as ts
from config.config import TUSHARE_TOKEN, DATA_DIR, HISTORY_DAYS
import os

class DataFetcher:
    def __init__(self):
        # 初始化tushare - 直接使用token初始化，避免写入文件
        self.pro = ts.pro_api(TUSHARE_TOKEN)
    
    def get_stock_list(self):
        """获取股票列表"""
        try:
            stock_list = self.pro.stock_basic(exchange='', list_status='L', fields='ts_code, symbol, name, area, industry, market, list_date')
            return stock_list
        except Exception as e:
            print(f"获取股票列表失败: {e}")
            return pd.DataFrame()
    
    def get_stock_history(self, ts_code, start_date=None, end_date=None, days=HISTORY_DAYS):
        """获取股票历史数据"""
        try:
            if not start_date:
                # 计算起始日期
                import datetime
                end = end_date if end_date else datetime.datetime.now().strftime('%Y%m%d')
                start = (datetime.datetime.now() - datetime.timedelta(days=days)).strftime('%Y%m%d')
            else:
                start = start_date
                end = end_date if end_date else datetime.datetime.now().strftime('%Y%m%d')
            
            # 获取日线数据
            df = self.pro.daily(ts_code=ts_code, start_date=start, end_date=end)
            
            # 按日期排序
            df = df.sort_values('trade_date').reset_index(drop=True)
            
            # 重命名列
            df.rename(columns={
                'trade_date': 'date',
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'vol': 'volume',
                'amount': 'amount'
            }, inplace=True)
            
            # 转换日期格式
            df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
            
            return df
        except Exception as e:
            print(f"获取股票 {ts_code} 历史数据失败: {e}")
            return pd.DataFrame()
    
    def save_data(self, df, ts_code):
        """保存数据到本地"""
        if not os.path.exists(DATA_DIR):
            os.makedirs(DATA_DIR)
        
        # 保存为CSV格式
        csv_path = os.path.join(DATA_DIR, f"{ts_code}.csv")
        df.to_csv(csv_path, index=False)
        print(f"数据已保存到: {csv_path}")
        
        # 保存为JSON格式
        json_path = os.path.join(DATA_DIR, f"{ts_code}.json")
        # 转换日期格式
        df_json = df.copy()
        df_json['date'] = df_json['date'].dt.strftime('%Y-%m-%d')
        df_json.to_json(json_path, orient='records', force_ascii=False, indent=2)
        print(f"数据已保存为JSON格式到: {json_path}")
    
    def load_data(self, ts_code):
        """从本地加载数据"""
        file_path = os.path.join(DATA_DIR, f"{ts_code}.csv")
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            df['date'] = pd.to_datetime(df['date'])
            return df
        else:
            return pd.DataFrame()

if __name__ == "__main__":
    fetcher = DataFetcher()
    # 测试获取股票列表
    stock_list = fetcher.get_stock_list()
    print(f"股票列表数量: {len(stock_list)}")
    print(stock_list.head())
    
    # 测试获取单只股票数据
    if not stock_list.empty:
        ts_code = stock_list.iloc[0]['ts_code']
        df = fetcher.get_stock_history(ts_code)
        print(f"{ts_code} 历史数据行数: {len(df)}")
        print(df.head())
        # 保存数据
        fetcher.save_data(df, ts_code)
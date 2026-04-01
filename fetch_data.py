import pandas as pd
import tushare as ts
import os
import datetime
import json

# 配置
TUSHARE_TOKEN = '68e431d47c0319d7bdea7cd1daf164392d22c8a9216d99476354be78'
DATA_DIR = 'data'
HISTORY_DAYS = 365

# 初始化tushare - 直接使用token初始化，避免写入文件
pro = ts.pro_api(TUSHARE_TOKEN)

def get_stock_history(ts_code, start_date=None, end_date=None, days=HISTORY_DAYS):
    """获取股票历史数据"""
    try:
        if not start_date:
            # 计算起始日期
            end = end_date if end_date else datetime.datetime.now().strftime('%Y%m%d')
            start = (datetime.datetime.now() - datetime.timedelta(days=days)).strftime('%Y%m%d')
        else:
            start = start_date
            end = end_date if end_date else datetime.datetime.now().strftime('%Y%m%d')
        
        # 获取日线数据
        print(f"正在获取股票 {ts_code} 的数据，时间范围：{start} 到 {end}")
        df = pro.daily(ts_code=ts_code, start_date=start, end_date=end)
        
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

def save_data(df, ts_code):
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

if __name__ == "__main__":
    # 测试获取供销大集的数据
    ts_code = '000564.SZ'  # 供销大集
    df = get_stock_history(ts_code)
    
    if not df.empty:
        save_data(df, ts_code)
        print(f"获取成功，共 {len(df)} 条记录")
        print(f"最新日期: {df['date'].iloc[-1].strftime('%Y-%m-%d')}")
        print(f"最新收盘价: {df['close'].iloc[-1]}")
    else:
        print("获取失败")

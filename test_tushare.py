import tushare as ts
import pandas as pd
import os
import json

# 配置
TUSHARE_TOKEN = '68e431d47c0319d7bdea7cd1daf164392d22c8a9216d99476354be78'
DATA_DIR = 'data'

# 初始化tushare - 直接使用token初始化，避免写入文件
print("正在初始化tushare...")
pro = ts.pro_api(TUSHARE_TOKEN)
print("tushare初始化完成")

# 测试获取股票数据
ts_code = '000564.SZ'  # 供销大集
print(f"\n正在获取股票 {ts_code} 的数据...")

try:
    # 获取日线数据
    df = pro.daily(ts_code=ts_code, start_date='20260301', end_date='20260401')
    print(f"获取成功，共 {len(df)} 条记录")
    print("数据预览:")
    print(df.head())
    
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
    
    # 保存数据
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    
    # 保存为CSV
    csv_path = os.path.join(DATA_DIR, f"{ts_code}.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n数据已保存到: {csv_path}")
    
    # 保存为JSON
    json_path = os.path.join(DATA_DIR, f"{ts_code}.json")
    df_json = df.copy()
    df_json['date'] = df_json['date'].dt.strftime('%Y-%m-%d')
    df_json.to_json(json_path, orient='records', force_ascii=False, indent=2)
    print(f"数据已保存为JSON格式到: {json_path}")
    
    # 打印最新数据
    print("\n最新数据:")
    latest_data = df.tail(1)
    print(f"日期: {latest_data['date'].iloc[0].strftime('%Y-%m-%d')}")
    print(f"收盘价: {latest_data['close'].iloc[0]}")
    
except Exception as e:
    print(f"获取数据失败: {e}")
    # 打印更详细的错误信息
    import traceback
    traceback.print_exc()

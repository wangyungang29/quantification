import tushare as ts
import pandas as pd

# 配置
TUSHARE_TOKEN = '68e431d47c0319d7bdea7cd1daf164392d22c8a9216d99476354be78'

# 初始化tushare
print("正在初始化tushare...")
pro = ts.pro_api(TUSHARE_TOKEN)
print("tushare初始化完成")

# 测试获取股票数据
ts_code = '000564.SZ'  # 供销大集
print(f"\n正在获取股票 {ts_code} 的数据...")

try:
    # 使用文档中推荐的方式获取数据
    df = pro.daily(ts_code=ts_code, start_date='20260301', end_date='20260401')
    print(f"获取成功，共 {len(df)} 条记录")
    print("数据预览:")
    print(df.head())
    
    # 打印最新数据
    if not df.empty:
        latest_data = df.sort_values('trade_date').tail(1)
        print("\n最新数据:")
        print(f"日期: {latest_data['trade_date'].iloc[0]}")
        print(f"收盘价: {latest_data['close'].iloc[0]}")
    
except Exception as e:
    print(f"获取数据失败: {e}")
    # 打印更详细的错误信息
    import traceback
    traceback.print_exc()

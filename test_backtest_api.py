#!/usr/bin/env python3
"""
测试普通策略回测API
"""

import requests
import json

# 测试数据
test_data = {
    "ts_code": "600519.SH",  # 贵州茅台
    "model_type": "lightgbm",
    "start_date": "20230101",
    "end_date": "20231231"
}

# 发送请求
try:
    response = requests.post(
        "http://localhost:5001/api/backtest",
        json=test_data,
        timeout=60
    )
    
    if response.status_code == 200:
        result = response.json()
        print("✅ 回测成功！")
        print(f"初始资金: {result['initial_capital']:.2f}")
        print(f"最终资金: {result['final_value']:.2f}")
        print(f"总收益率: {result['total_return']:.2%}")
        print(f"夏普比率: {result['sharpe_ratio']:.2f}")
        print(f"最大回撤: {result['max_drawdown']:.2%}")
        print(f"平均收益率: {result['average_return']:.2%}")
        print(f"总交易次数: {result['total_trades']}")
        print(f"胜率: {result['win_rate']:.2%}")
        
        # 打印交易记录
        if result['trades']:
            print("\n交易记录:")
            for trade in result['trades'][:10]:  # 只打印前10条
                print(f"{trade['date']}: {trade['action']} - 价格={trade['price']:.2f}, 数量={trade['shares']}")
        else:
            print("\n❌ 没有交易记录")
        
    else:
        print(f"❌ 请求失败: {response.status_code}")
        print(f"错误信息: {response.text}")
        
except Exception as e:
    print(f"❌ 发生错误: {e}")
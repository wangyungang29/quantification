#!/usr/bin/env python3
"""
测试高收益策略回测API（使用用户指定的参数）
"""

import requests
import json

# 测试数据（用户指定的参数）
test_data = {
    "ts_code": "000564.SZ",  # 供销大集
    "model_type": "lightgbm"
}

# 发送请求
try:
    response = requests.post(
        "http://localhost:5001/api/backtest_500pct",
        json=test_data,
        timeout=120
    )
    
    if response.status_code == 200:
        result = response.json()
        print("✅ 高收益策略回测成功！")
        print(f"股票代码: {test_data['ts_code']}")
        print(f"初始资金: {result.get('initial_capital', 'N/A')}")
        print(f"最终资金: {result.get('final_value', 'N/A')}")
        print(f"总收益率: {result.get('total_return', 'N/A')}")
        print(f"夏普比率: {result.get('sharpe_ratio', 'N/A')}")
        print(f"最大回撤: {result.get('max_drawdown', 'N/A')}")
        print(f"平均收益率: {result.get('average_return', 'N/A')}")
        print(f"总交易次数: {result.get('total_trades', 'N/A')}")
        print(f"胜率: {result.get('win_rate', 'N/A')}")
        
        # 打印交易记录
        if result.get('trades'):
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
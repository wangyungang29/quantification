import requests
import json

# 测试高收益策略回测API
url = 'http://localhost:5001/api/backtest_500pct'
data = {
    'ts_code': '000564.SZ',
    'model_type': 'xgboost'
}

try:
    print("发送请求到高收益策略回测API...")
    response = requests.post(url, json=data, timeout=300)
    print(f"状态码: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print("回测成功！")
        print(f"总收益率: {result.get('total_return', 0):.2%}")
        print(f"最大回撤: {result.get('max_drawdown', 0):.2%}")
        print(f"是否达到500%目标: {result.get('achieved_500pct', False)}")
        print(f"交易次数: {len(result.get('signals', []))}")
    else:
        print(f"请求失败: {response.text}")
except Exception as e:
    print(f"发生错误: {e}")
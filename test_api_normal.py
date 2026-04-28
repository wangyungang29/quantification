import requests
import json

# 测试普通策略回测API
def test_normal_strategy_backtest():
    url = 'http://localhost:5001/api/backtest'
    data = {
        'ts_code': '600519.SH',  # 贵州茅台
        'model_type': 'xgboost',
        'start_date': '20200101',
        'end_date': '20201231'
    }
    
    print("发送请求到普通策略回测API...")
    try:
        response = requests.post(url, json=data, timeout=300)
        if response.status_code == 200:
            result = response.json()
            print("回测成功！")
            print(f"总收益率: {result.get('total_return', 0):.2%}")
            print(f"最大回撤: {result.get('max_drawdown', 0):.2%}")
            print(f"交易次数: {result.get('total_trades', 0)}")
            print(f"胜率: {result.get('win_rate', 0):.2%}")
            print(f"平均收益率: {result.get('average_return', 0):.2%}")
        else:
            print(f"回测失败，状态码: {response.status_code}")
            print(f"错误信息: {response.text}")
    except Exception as e:
        print(f"发生错误: {e}")

if __name__ == "__main__":
    test_normal_strategy_backtest()
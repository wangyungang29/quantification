#!/usr/bin/env python3
"""
分析普通策略回测结果，找出导致负收益率的原因
"""

import requests
import json
import pandas as pd
import matplotlib.pyplot as plt

# 测试数据（用户指定的参数）
test_data = {
    "ts_code": "000564.SZ",  # 供销大集
    "model_type": "lightgbm",
    "start_date": "20260115",
    "end_date": "20260415"
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
        print("✅ 普通策略回测成功！")
        print(f"股票代码: {test_data['ts_code']}")
        print(f"时间范围: {test_data['start_date']} 到 {test_data['end_date']}")
        print(f"初始资金: {result['initial_capital']:.2f}")
        print(f"最终资金: {result['final_value']:.2f}")
        print(f"总收益率: {result['total_return']:.2%}")
        print(f"夏普比率: {result['sharpe_ratio']:.2f}")
        print(f"最大回撤: {result['max_drawdown']:.2%}")
        print(f"平均收益率: {result['average_return']:.2%}")
        print(f"总交易次数: {result['total_trades']}")
        print(f"胜率: {result['win_rate']:.2%}")
        
        # 分析交易记录
        if result['trades']:
            print("\n交易记录分析:")
            
            # 转换交易记录为DataFrame
            trades_df = pd.DataFrame(result['trades'])
            
            # 计算每次交易的盈亏
            buy_trades = trades_df[trades_df['action'] == '买入']
            sell_trades = trades_df[trades_df['action'] == '卖出']
            
            if len(buy_trades) == len(sell_trades):
                print(f"买入次数: {len(buy_trades)}")
                print(f"卖出次数: {len(sell_trades)}")
                
                # 计算每次交易的盈亏
                profits = []
                for i in range(len(buy_trades)):
                    buy_price = buy_trades.iloc[i]['price']
                    sell_price = sell_trades.iloc[i]['price']
                    shares = buy_trades.iloc[i]['shares']
                    profit = (sell_price - buy_price) * shares
                    profits.append(profit)
                    print(f"交易 {i+1}: 买入价格={buy_price:.2f}, 卖出价格={sell_price:.2f}, 盈亏={profit:.2f}")
                
                # 计算总盈亏和平均盈亏
                total_profit = sum(profits)
                avg_profit = total_profit / len(profits) if profits else 0
                print(f"\n总盈亏: {total_profit:.2f}")
                print(f"平均盈亏: {avg_profit:.2f}")
                
                # 计算盈利和亏损的交易次数
                winning_trades = sum(1 for p in profits if p > 0)
                losing_trades = sum(1 for p in profits if p < 0)
                print(f"盈利交易次数: {winning_trades}")
                print(f"亏损交易次数: {losing_trades}")
                
                # 计算平均持仓时间
                buy_dates = pd.to_datetime(buy_trades['date'])
                sell_dates = pd.to_datetime(sell_trades['date'])
                holding_periods = (sell_dates - buy_dates).dt.days
                avg_holding_period = holding_periods.mean()
                print(f"平均持仓时间: {avg_holding_period:.1f} 天")
                
            else:
                print("交易记录不完整，无法计算盈亏")
        else:
            print("\n❌ 没有交易记录")
        
    else:
        print(f"❌ 请求失败: {response.status_code}")
        print(f"错误信息: {response.text}")
        
except Exception as e:
    print(f"❌ 发生错误: {e}")
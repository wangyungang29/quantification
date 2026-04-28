#!/usr/bin/env python3
"""
测试优化后的/predict接口
"""

import requests
import json

# 测试数据
test_data = {
    "ts_code": "600519.SH",  # 贵州茅台
    "model_type": "lightgbm"
}

# 发送请求
try:
    response = requests.post(
        "http://localhost:5001/api/predict",
        json=test_data,
        timeout=30
    )
    
    if response.status_code == 200:
        result = response.json()
        print("✅ 预测成功！")
        print(f"股票代码: {result['ts_code']}")
        print(f"最新日期: {result['latest_date']}")
        print(f"最新价格: {result['latest_close']:.2f}")
        print(f"预测标签: {result['prediction']['label']}")
        print(f"预测结果: {'上涨' if result['prediction']['label'] == 1 else '下跌' if result['prediction']['label'] == -1 else '横盘'}")
        print(f"上涨概率: {result['prediction']['probability']['up']:.2%}")
        print(f"横盘概率: {result['prediction']['probability']['flat']:.2%}")
        print(f"下跌概率: {result['prediction']['probability']['down']:.2%}")
        print(f"预期价格: {result['prediction']['price_space']['expected_price']:.2f}")
        print(f"波动率: {result['volatility']:.2%}")
        
        # 打印技术指标
        print("\n📊 技术指标:")
        # 从历史数据中获取最新的技术指标
        if result.get('history_data'):
            latest_data = result['history_data'][-1]
            print(f"MA5: {latest_data.get('ma5', 'N/A'):.2f}")
            print(f"MA20: {latest_data.get('ma20', 'N/A'):.2f}")
            print(f"MA60: {latest_data.get('ma60', 'N/A'):.2f}")
            print(f"RSI: {latest_data.get('rsi', 'N/A'):.1f}")
            print(f"MACD: {latest_data.get('macd', 'N/A'):.4f}")
            print(f"MACD Signal: {latest_data.get('macd_signal', 'N/A'):.4f}")
            print(f"MACD Hist: {latest_data.get('macd_hist', 'N/A'):.4f}")
            print(f"布林带: 上轨={latest_data.get('boll_upper', 'N/A'):.2f}, 中轨={latest_data.get('boll_mid', 'N/A'):.2f}, 下轨={latest_data.get('boll_lower', 'N/A'):.2f}")
        
        # 打印交易信号
        print("\n📈 交易信号:")
        signals = result.get('trading_signals', {})
        print(f"买入信号: {signals.get('buy_signal', 'N/A')}")
        print(f"卖出信号: {signals.get('sell_signal', 'N/A')}")
        print(f"高收益策略信号: {signals.get('high_return_action', 'N/A')}")
        print(f"高收益策略仓位: {signals.get('high_return_position', 'N/A'):.1%}")
        
        # 打印历史数据
        print("\n📅 历史数据 (最近5天):")
        history = result.get('history_data', [])[:5]
        for item in history:
            print(f"{item['date']}: 收盘价={item['close']:.2f}, 买入信号={item['buy_signal']}, 卖出信号={item['sell_signal']}")
        
    else:
        print(f"❌ 请求失败: {response.status_code}")
        print(f"错误信息: {response.text}")
        
except Exception as e:
    print(f"❌ 发生错误: {e}")
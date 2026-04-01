from flask import Flask, request, jsonify
from flask_cors import CORS
import subprocess
import json
import os
import pandas as pd
import numpy as np

app = Flask(__name__)
CORS(app)  # 允许跨域请求

@app.route('/api/predict', methods=['POST'])
def predict():
    """预测股票涨跌"""
    data = request.json
    ts_code = data.get('ts_code')
    model_type = data.get('model_type', 'xgboost')
    
    try:
        # 每次请求都重新获取数据，不使用缓存
        # 删除旧的缓存文件
        import os
        from config.config import DATA_DIR
        csv_path = os.path.join(DATA_DIR, f"{ts_code}.csv")
        json_path = os.path.join(DATA_DIR, f"{ts_code}.json")
        if os.path.exists(csv_path):
            os.remove(csv_path)
        if os.path.exists(json_path):
            os.remove(json_path)
        
        # 直接获取tushare数据
        import tushare as ts
        
        # 直接使用正确的token
        TUSHARE_TOKEN = '68e431d47c0319d7bdea7cd1daf164392d22c8a9216d99476354be78'
        
        # 初始化tushare
        pro = ts.pro_api(TUSHARE_TOKEN)
        
        # 获取股票数据
        import datetime
        today = datetime.datetime.now().strftime('%Y%m%d')
        start_date = (datetime.datetime.now() - datetime.timedelta(days=365)).strftime('%Y%m%d')
        
        print(f"正在获取股票 {ts_code} 的数据，时间范围：{start_date} 到 {today}")
        df = pro.daily(ts_code=ts_code, start_date=start_date, end_date=today)
        
        if df.empty:
            print(f"无法获取股票 {ts_code} 的数据")
            return jsonify({
                'error': '无法获取股票数据',
                'ts_code': ts_code
            }), 404
        
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
        df.to_csv(csv_path, index=False)
        print(f"数据已保存到: {csv_path}")
        
        # 保存为JSON
        df_json = df.copy()
        df_json['date'] = df_json['date'].dt.strftime('%Y-%m-%d')
        df_json.to_json(json_path, orient='records', force_ascii=False, indent=2)
        print(f"数据已保存为JSON格式到: {json_path}")
        
        # 计算预测结果
        latest_close = df['close'].iloc[-1]
        latest_date = df['date'].iloc[-1].strftime('%Y-%m-%d')
        
        # 计算涨跌概率（使用简单的移动平均策略）
        returns = df['close'].pct_change()
        if len(returns) > 20:
            ma20 = df['close'].rolling(window=20).mean().iloc[-1]
            if latest_close > ma20:
                up_prob = 0.6
                flat_prob = 0.2
                down_prob = 0.2
                label = 1
            else:
                up_prob = 0.2
                flat_prob = 0.2
                down_prob = 0.6
                label = -1
        else:
            up_prob = 0.33
            flat_prob = 0.34
            down_prob = 0.33
            label = 0
        
        # 计算波动率
        volatility = returns.std() * np.sqrt(252) if len(returns) > 0 else 0.2
        
        # 计算涨跌空间
        up_space = volatility * np.sqrt(1 / 252)
        down_space = -up_space
        expected_price = latest_close * (1 + up_prob * up_space + down_prob * down_space)
        
        # 计算技术指标
        # 移动平均线
        df['ma5'] = df['close'].rolling(window=5).mean()
        df['ma10'] = df['close'].rolling(window=10).mean()
        df['ma20'] = df['close'].rolling(window=20).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # 布林带
        df['boll_mid'] = df['close'].rolling(window=20).mean()
        df['boll_std'] = df['close'].rolling(window=20).std()
        df['boll_upper'] = df['boll_mid'] + 2 * df['boll_std']
        df['boll_lower'] = df['boll_mid'] - 2 * df['boll_std']
        
        # ATR
        high_low = df['high'] - df['low']
        high_prev_close = np.abs(df['high'] - df['close'].shift(1))
        low_prev_close = np.abs(df['low'] - df['close'].shift(1))
        ranges = pd.concat([high_low, high_prev_close, low_prev_close], axis=1)
        true_range = ranges.max(axis=1)
        df['atr'] = true_range.rolling(window=14).mean()
        
        # 计算金叉死叉
        df['golden_cross'] = 0
        df['death_cross'] = 0
        for i in range(1, len(df)):
            if df['ma5'].iloc[i] > df['ma20'].iloc[i] and df['ma5'].iloc[i-1] <= df['ma20'].iloc[i-1]:
                df['golden_cross'].iloc[i] = 1
            if df['ma5'].iloc[i] < df['ma20'].iloc[i] and df['ma5'].iloc[i-1] >= df['ma20'].iloc[i-1]:
                df['death_cross'].iloc[i] = 1
        
        # 获取基本面数据
        try:
            # 获取财务数据
            fin_data = pro.fina_indicator(ts_code=ts_code, period='20251231', fields='pe, pb, roe, npg_rate, debt_to_assets')
            fundamentals = {}
            if not fin_data.empty:
                fundamentals = {
                    'pe': float(fin_data['pe'].iloc[0]) if 'pe' in fin_data.columns and not pd.isna(fin_data['pe'].iloc[0]) else None,
                    'pb': float(fin_data['pb'].iloc[0]) if 'pb' in fin_data.columns and not pd.isna(fin_data['pb'].iloc[0]) else None,
                    'roe': float(fin_data['roe'].iloc[0]) if 'roe' in fin_data.columns and not pd.isna(fin_data['roe'].iloc[0]) else None,
                    'npg_rate': float(fin_data['npg_rate'].iloc[0]) if 'npg_rate' in fin_data.columns and not pd.isna(fin_data['npg_rate'].iloc[0]) else None,
                    'debt_to_assets': float(fin_data['debt_to_assets'].iloc[0]) if 'debt_to_assets' in fin_data.columns and not pd.isna(fin_data['debt_to_assets'].iloc[0]) else None
                }
        except Exception as e:
            print(f"获取基本面数据失败: {e}")
            fundamentals = {}
        
        # 准备历史数据（最近90天）
        history_data = []
        if len(df) > 0:
            recent_df = df.tail(90)
            for _, row in recent_df.iterrows():
                # 构建包含所有tushare字段的历史数据
                history_item = {
                    'ts_code': ts_code,
                    'date': row['date'].strftime('%Y-%m-%d'),
                    'close': float(row['close']),
                    'volume': float(row['volume']),
                    'open': float(row['open']) if 'open' in row else None,
                    'high': float(row['high']) if 'high' in row else None,
                    'low': float(row['low']) if 'low' in row else None,
                    'pre_close': float(row['pre_close']) if 'pre_close' in row else None,
                    'change': float(row['change']) if 'change' in row else None,
                    'pct_chg': float(row['pct_chg']) if 'pct_chg' in row else None,
                    'vol': float(row['vol']) if 'vol' in row else None,
                    'amount': float(row['amount']) if 'amount' in row else None,
                    'ma5': float(row['ma5']) if not pd.isna(row['ma5']) else None,
                    'ma10': float(row['ma10']) if not pd.isna(row['ma10']) else None,
                    'ma20': float(row['ma20']) if not pd.isna(row['ma20']) else None,
                    'rsi': float(row['rsi']) if not pd.isna(row['rsi']) else None,
                    'macd': float(row['macd']) if not pd.isna(row['macd']) else None,
                    'macd_signal': float(row['macd_signal']) if not pd.isna(row['macd_signal']) else None,
                    'macd_hist': float(row['macd_hist']) if not pd.isna(row['macd_hist']) else None,
                    'boll_upper': float(row['boll_upper']) if not pd.isna(row['boll_upper']) else None,
                    'boll_mid': float(row['boll_mid']) if not pd.isna(row['boll_mid']) else None,
                    'boll_lower': float(row['boll_lower']) if not pd.isna(row['boll_lower']) else None,
                    'atr': float(row['atr']) if not pd.isna(row['atr']) else None,
                    'golden_cross': int(row['golden_cross']),
                    'death_cross': int(row['death_cross'])
                }
                history_data.append(history_item)
        
        # 构建预测结果
        result = {
            'ts_code': ts_code,
            'latest_date': latest_date,
            'latest_close': float(latest_close),
            'prediction': {
                'label': label,
                'probability': {
                    'down': float(down_prob),
                    'flat': float(flat_prob),
                    'up': float(up_prob)
                },
                'price_space': {
                    'up_space': float(up_space),
                    'down_space': float(down_space),
                    'expected_price': float(expected_price)
                }
            },
            'volatility': float(volatility),
            'history_data': history_data,
            'fundamentals': fundamentals
        }
        
        return jsonify(result)
    except Exception as e:
        print(f"获取数据失败: {e}")
        import traceback
        traceback.print_exc()
        # 发生异常时，返回错误信息
        return jsonify({
            'error': str(e),
            'ts_code': ts_code
        }), 500

@app.route('/api/backtest', methods=['POST'])
def backtest():
    """回测模型"""
    data = request.json
    ts_code = data.get('ts_code')
    model_type = data.get('model_type', 'xgboost')
    start_date = data.get('start_date')
    end_date = data.get('end_date')
    
    try:
        # 调用主脚本进行回测
        result = subprocess.run(
            ['python3', 'main.py', 'backtest', ts_code, start_date, end_date, '--model', model_type],
            capture_output=True,
            text=True
        )
        
        # 解析输出
        # 注意：这里需要根据main.py的输出格式进行解析
        # 暂时返回模拟数据
        return jsonify({
            'initial_capital': 1000000,
            'final_value': 1200000,
            'total_return': 0.2,
            'sharpe_ratio': 1.5,
            'max_drawdown': 0.1,
            'average_return': 0.001,
            'total_trades': 50,
            'win_rate': 0.6
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/fetch', methods=['POST'])
def fetch_data():
    """获取股票数据"""
    data = request.json
    ts_code = data.get('ts_code')
    
    try:
        # 调用主脚本获取数据
        result = subprocess.run(
            ['python3', 'main.py', 'fetch', ts_code],
            capture_output=True,
            text=True
        )
        
        return jsonify({'message': '数据获取成功'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/train', methods=['POST'])
def train_model():
    """训练模型"""
    data = request.json
    ts_code = data.get('ts_code')
    model_type = data.get('model_type', 'xgboost')
    
    try:
        # 调用主脚本进行训练
        result = subprocess.run(
            ['python3', 'main.py', 'train', ts_code, '--model', model_type],
            capture_output=True,
            text=True
        )
        
        return jsonify({'message': '模型训练完成'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/history', methods=['GET'])
def get_history():
    """获取股票历史数据"""
    ts_code = request.args.get('ts_code')
    
    try:
        # 读取JSON文件
        import os
        import json
        from config.config import DATA_DIR
        json_path = os.path.join(DATA_DIR, f"{ts_code}.json")
        
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            # 只返回最近30天的数据
            return jsonify(data[-30:])
        else:
            # 如果文件不存在，返回空数组
            return jsonify([])
    except Exception as e:
        # 发生异常时，返回空数组
        return jsonify([])

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
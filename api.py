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
    """基于缠论的股票预测"""
    data = request.json
    ts_code = data.get('ts_code')
    
    try:
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
        TUSHARE_TOKEN = '68e431d47c0319d7bdea7cd1daf164392d22c8a9216d99476354be78'
        pro = ts.pro_api(TUSHARE_TOKEN)
        
        # 获取股票数据（至少需要200天数据用于缠论分析）
        import datetime
        today = datetime.datetime.now().strftime('%Y%m%d')
        start_date = (datetime.datetime.now() - datetime.timedelta(days=500)).strftime('%Y%m%d')
        
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
        df.to_csv(csv_path, index=False)
        print(f"数据已保存到: {csv_path}")
        
        df_json = df.copy()
        df_json['date'] = df_json['date'].dt.strftime('%Y-%m-%d')
        df_json.to_json(json_path, orient='records', force_ascii=False, indent=2)
        print(f"数据已保存为JSON格式到: {json_path}")
        
        # 计算预测结果
        latest_close = df['close'].iloc[-1]
        latest_date = df['date'].iloc[-1].strftime('%Y-%m-%d')
        
        # 计算技术指标
        df['ma5'] = df['close'].rolling(window=5).mean()
        df['ma10'] = df['close'].rolling(window=10).mean()
        df['ma20'] = df['close'].rolling(window=20).mean()
        df['ema12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['ema26'] = df['close'].ewm(span=26, adjust=False).mean()
        df['ma60'] = df['close'].rolling(window=60).mean()
        
        # RSI
        def calculate_rsi(series, period=14):
            delta = series.diff()
            gain = delta.clip(lower=0)
            loss = -delta.clip(upper=0)
            avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
            avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
            rs = avg_gain / (avg_loss + 1e-8)
            return 100 - (100 / (1 + rs))
        
        df['rsi'] = calculate_rsi(df['close'], period=14)
        
        # MACD
        def calculate_macd(close_series, fast=12, slow=26, signal=9):
            ema_fast = close_series.ewm(span=fast, adjust=False).mean()
            ema_slow = close_series.ewm(span=slow, adjust=False).mean()
            dif = ema_fast - ema_slow
            dea = dif.ewm(span=signal, adjust=False).mean()
            macd_hist = (dif - dea) * 2
            return dif, dea, macd_hist
        
        df['macd'], df['macd_signal'], df['macd_hist'] = calculate_macd(df['close'])
        
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
        df['golden_cross'] = 0.0
        df['death_cross'] = 0.0
        for i in range(1, len(df)):
            try:
                if not pd.isna(df['ma5'].iloc[i]) and not pd.isna(df['ma20'].iloc[i]) and not pd.isna(df['ma5'].iloc[i-1]) and not pd.isna(df['ma20'].iloc[i-1]):
                    if df['ma5'].iloc[i] > df['ma20'].iloc[i] and df['ma5'].iloc[i-1] <= df['ma20'].iloc[i-1]:
                        df.loc[df.index[i], 'golden_cross'] = 1.0
                    if df['ma5'].iloc[i] < df['ma20'].iloc[i] and df['ma5'].iloc[i-1] >= df['ma20'].iloc[i-1]:
                        df.loc[df.index[i], 'death_cross'] = 1.0
            except:
                continue
        
        # 使用缠论解析器分析
        from src.models.chan_parser import ChanParser
        parser = ChanParser()
        chan_result = parser.analyze(df)
        
        # 初始化信号列
        df['buy_signal'] = 0.0
        df['sell_signal'] = 0.0
        df['buy_price'] = 0.0
        df['sell_price'] = 0.0
        df['signal_type'] = ''
        df['stop_loss'] = 0.0
        
        # 根据缠论信号更新数据
        signals = chan_result.get('signals', [])
        for signal in signals:
            idx = signal['index']
            if idx < len(df):
                if signal['type'] in ['一买', '二买', '三买']:
                    df.loc[df.index[idx], 'buy_signal'] = 1.0
                    df.loc[df.index[idx], 'buy_price'] = signal['price']
                    df.loc[df.index[idx], 'signal_type'] = signal['type']
                    df.loc[df.index[idx], 'stop_loss'] = signal.get('stop_loss', signal['price'])
                elif signal['type'] in ['一卖', '二卖', '三卖']:
                    df.loc[df.index[idx], 'sell_signal'] = 1.0
                    df.loc[df.index[idx], 'sell_price'] = signal['price']
                    df.loc[df.index[idx], 'signal_type'] = signal['type']
        
        # 计算基础概率（基于缠论信号）
        buy_signal_count = sum(1 for s in signals if s['type'] in ['一买', '二买', '三买'])
        sell_signal_count = sum(1 for s in signals if s['type'] in ['一卖', '二卖', '三卖'])
        total_signals = buy_signal_count + sell_signal_count
        
        if total_signals > 0:
            up_prob = buy_signal_count / total_signals
            down_prob = sell_signal_count / total_signals
            flat_prob = 0.1
        else:
            # 如果没有缠论信号，使用技术指标计算概率
            up_prob = 0.5
            down_prob = 0.4
            flat_prob = 0.1
        
        # 调整概率
        up_prob = max(0.1, min(0.9, up_prob))
        down_prob = max(0.1, min(0.8, down_prob))
        flat_prob = max(0.1, min(0.3, flat_prob))
        
        # 归一化
        total = up_prob + down_prob + flat_prob
        up_prob /= total
        down_prob /= total
        flat_prob /= total
        
        label = 1 if up_prob > down_prob else (-1 if down_prob > up_prob else 0)
        
        # 计算涨跌空间
        returns = df['close'].pct_change()
        volatility = returns.std() * np.sqrt(252) if len(returns) > 0 else 0.2
        up_space = volatility * np.sqrt(1 / 252)
        down_space = -up_space
        expected_price = latest_close * (1 + up_prob * up_space + down_prob * down_space)
        
        # 准备历史数据（最近90天）
        history_data = []
        if len(df) > 0:
            recent_df = df.tail(90)
            for _, row in recent_df.iterrows():
                volume_value = float(row['volume']) if 'volume' in row else 0
                history_item = {
                    'ts_code': ts_code,
                    'date': row['date'].strftime('%Y-%m-%d'),
                    'close': float(row['close']),
                    'volume': volume_value,
                    'open': float(row['open']) if 'open' in row else None,
                    'high': float(row['high']) if 'high' in row else None,
                    'low': float(row['low']) if 'low' in row else None,
                    'pre_close': float(row['pre_close']) if 'pre_close' in row else None,
                    'change': float(row['change']) if 'change' in row else None,
                    'pct_chg': float(row['pct_chg']) if 'pct_chg' in row else None,
                    'vol': volume_value,
                    'amount': float(row['amount']) if 'amount' in row else None,
                    'ma5': float(row['ma5']) if not pd.isna(row['ma5']) else None,
                    'ma10': float(row['ma10']) if not pd.isna(row['ma10']) else None,
                    'ma20': float(row['ma20']) if not pd.isna(row['ma20']) else None,
                    'ema12': float(row['ema12']) if not pd.isna(row['ema12']) else None,
                    'ema26': float(row['ema26']) if not pd.isna(row['ema26']) else None,
                    'rsi': float(row['rsi']) if not pd.isna(row['rsi']) else None,
                    'macd': float(row['macd']) if not pd.isna(row['macd']) else None,
                    'macd_signal': float(row['macd_signal']) if not pd.isna(row['macd_signal']) else None,
                    'macd_hist': float(row['macd_hist']) if not pd.isna(row['macd_hist']) else None,
                    'boll_upper': float(row['boll_upper']) if not pd.isna(row['boll_upper']) else None,
                    'boll_mid': float(row['boll_mid']) if not pd.isna(row['boll_mid']) else None,
                    'boll_lower': float(row['boll_lower']) if not pd.isna(row['boll_lower']) else None,
                    'ma60': float(row['ma60']) if not pd.isna(row['ma60']) else None,
                    'atr': float(row['atr']) if not pd.isna(row['atr']) else None,
                    'golden_cross': int(float(row['golden_cross'])) if 'golden_cross' in row else 0,
                    'death_cross': int(float(row['death_cross'])) if 'death_cross' in row else 0,
                    'buy_signal': bool(float(row['buy_signal'])) if 'buy_signal' in row else False,
                    'sell_signal': bool(float(row['sell_signal'])) if 'sell_signal' in row else False,
                    'buy_price': float(row['buy_price']) if 'buy_price' in row and not pd.isna(row['buy_price']) else 0,
                    'sell_price': float(row['sell_price']) if 'sell_price' in row and not pd.isna(row['sell_price']) else 0,
                    'signal_type': str(row['signal_type']) if 'signal_type' in row else '',
                    'stop_loss': float(row['stop_loss']) if 'stop_loss' in row else 0
                }
                history_data.append(history_item)
        
        # 获取最新缠论信号
        latest_signal = parser.get_latest_signal()
        
        # 生成交易信号
        buy_signal = False
        sell_signal = False
        message = []
        
        if latest_signal:
            if latest_signal['type'] in ['一买', '二买', '三买']:
                buy_signal = True
                message.append(f"⭐ {latest_signal['type']}：{latest_signal['description']}")
                message.append(f"入场价: {latest_signal['price']:.2f}")
                message.append(f"止损位: {latest_signal.get('stop_loss', latest_signal['price']):.2f}")
            elif latest_signal['type'] in ['一卖', '二卖', '三卖']:
                sell_signal = True
                message.append(f"❌ {latest_signal['type']}")
        else:
            # 如果没有缠论信号，使用技术指标判断
            if df['ma5'].iloc[-1] > df['ma20'].iloc[-1] and df['rsi'].iloc[-1] < 50:
                buy_signal = True
                message.append("📈 均线多头排列，RSI处于低位，建议关注买入机会")
            elif df['ma5'].iloc[-1] < df['ma20'].iloc[-1] and df['rsi'].iloc[-1] > 50:
                sell_signal = True
                message.append("📉 均线空头排列，RSI处于高位，建议关注卖出机会")
            else:
                message.append("📊 当前无明确信号，建议观望")
        
        trading_signals = {
            'buy_signal': buy_signal,
            'sell_signal': sell_signal,
            'buy_probability': float(up_prob),
            'sell_probability': float(down_prob),
            'message': ' '.join(message),
            'latest_signal': latest_signal
        }
        
        # 构建中枢信息
        zhongshus = chan_result.get('zhongshus', [])
        zhongshu_info = []
        for zs in zhongshus[-3:]:  # 只返回最近3个中枢
            zhongshu_info.append({
                'high': float(zs['high']),
                'low': float(zs['low']),
                'center': float(zs['center']),
                'range': float(zs['range']),
                'type': zs['type']
            })
        
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
            'trading_signals': trading_signals,
            'history_data': history_data,
            'zhongshus': zhongshu_info,
            'fundamentals': {}
        }
        
        return jsonify(result)
    except Exception as e:
        print(f"获取数据失败: {e}")
        import traceback
        traceback.print_exc()
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
        # 直接使用Backtester进行回测
        from src.backtest.backtester import Backtester
        backtester = Backtester()
        
        # 运行回测
        results = backtester.run_backtest(ts_code, start_date, end_date, model_type=model_type)
        
        if results:
            return jsonify(results)
        else:
            # 如果回测失败，返回错误信息
            return jsonify({'error': '回测失败，请先训练模型'}), 400
    except Exception as e:
        print(f"回测失败: {e}")
        import traceback
        traceback.print_exc()
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
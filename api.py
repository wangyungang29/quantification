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
        start_date = (datetime.datetime.now() - datetime.timedelta(days=445)).strftime('%Y%m%d')  # 多拿80天数据（20天+60天）
        
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
        df['ma5'] = df['close'].rolling(window=5).mean()  # SMA_5
        df['ma10'] = df['close'].rolling(window=10).mean()
        df['ma20'] = df['close'].rolling(window=20).mean()  # SMA_20
        df['ema12'] = df['close'].ewm(span=12, adjust=False).mean()  # EMA_12
        df['ema26'] = df['close'].ewm(span=26, adjust=False).mean()  # EMA_26
        
        # RSI (使用Wilder平滑)
        def calculate_rsi(series, period=14):
            delta = series.diff()
            gain = delta.clip(lower=0)
            loss = -delta.clip(upper=0)
            
            # 使用指数移动平均（Wilder平滑）
            avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
            avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
            
            rs = avg_gain / (avg_loss + 1e-8)
            rsi = 100 - (100 / (1 + rs))
            return rsi
        
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
        
        # 60日均线
        df['ma60'] = df['close'].rolling(window=60).mean()
        
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
            except Exception as e:
                print(f"计算金叉死叉时出错: {e}")
                continue
        
        # 为历史数据生成交易信号
        # 使用模型预测来生成买入和卖出信号
        df['buy_signal'] = 0.0
        df['sell_signal'] = 0.0
        df['buy_price'] = 0.0
        df['sell_price'] = 0.0
        df['buy_probability'] = 0.0
        df['sell_probability'] = 0.0
        
        # 高收益策略信号
        df['high_return_action'] = "HOLD"
        df['high_return_position'] = 0.0
        df['super_trend_prob'] = 0.0
        
        # 使用模型预测金叉和死叉（必须在构建history_data之前调用）
        try:
            from src.models.model_trainer import ModelTrainer
            trainer = ModelTrainer()
            
            # 检查是否存在训练好的模型，如果不存在则训练
            golden_model_path = f"src/models/saved/{model_type}_golden_cross_model.joblib"
            death_model_path = f"src/models/saved/{model_type}_death_cross_model.joblib"
            price_model_path = f"src/models/saved/{model_type}_price_prediction_model.joblib"
            high_return_model_path = f"src/models/saved/{model_type}_500pct_model.joblib"
            
            import os
            if not os.path.exists(golden_model_path) or not os.path.exists(death_model_path):
                print(f"未找到训练好的模型，开始训练...")
                try:
                    # 训练金叉预测模型
                    print('训练金叉预测模型...')
                    trainer.train_golden_cross_model(df, model_type=model_type)
                    print('金叉预测模型训练完成')
                    
                    # 训练死叉预测模型
                    print('训练死叉预测模型...')
                    trainer.train_death_cross_model(df, model_type=model_type)
                    print('死叉预测模型训练完成')
                except Exception as e:
                    print(f"模型训练失败: {e}")
            
            # 训练高收益模型
            if not os.path.exists(high_return_model_path):
                print(f"未找到高收益模型，开始训练...")
                try:
                    trainer.train_and_evaluate_500pct(df, model_type=model_type)
                    print('高收益模型训练完成')
                except Exception as e:
                    print(f"高收益模型训练失败: {e}")
            
            # 预测金叉和死叉
            df = trainer.predict_golden_death_cross(df, model_type=model_type, threshold=0.5)
            print(f"成功预测金叉和死叉")
            
            # 预测t-1的买卖点
            df = trainer.predict_buy_sell_points(df, model_type=model_type, atr_multiplier=1.0)
            print(f"成功预测买卖点")
            
            # 生成高收益策略信号
            try:
                high_return_model = trainer.load_model(f"{model_type}_500pct_model")
                if high_return_model is not None:
                    high_return_signal = trainer.generate_500pct_signal(high_return_model, df)
                    df.loc[df.index[-1], 'high_return_action'] = high_return_signal['action']
                    df.loc[df.index[-1], 'high_return_position'] = high_return_signal['position']
                    df.loc[df.index[-1], 'super_trend_prob'] = high_return_signal['super_trend_prob']
                    print(f"高收益策略信号: {high_return_signal['action']}")
            except Exception as e:
                print(f"生成高收益策略信号失败: {e}")
            
            # 为历史数据生成模型交易信号
            # 加载金叉预测模型用于生成信号
            golden_model = trainer.load_model(f"{model_type}_golden_cross_model")
            if golden_model is not None:
                print("使用模型为历史数据生成交易信号...")
                
                # 首先为整个df计算特征（包括连续下降天数等）
                df_features_full, feature_cols_full = trainer.create_features(df)
                if len(df_features_full) > 0:
                    # 将连续下降天数等特征合并回原始df
                    if 'consecutive_down' in df_features_full.columns:
                        df.loc[df_features_full.index, 'consecutive_down'] = df_features_full['consecutive_down']
                    if 'consecutive_up' in df_features_full.columns:
                        df.loc[df_features_full.index, 'consecutive_up'] = df_features_full['consecutive_up']
                    if 'vol/Vol_MA5' in df_features_full.columns:
                        df.loc[df_features_full.index, 'vol/Vol_MA5'] = df_features_full['vol/Vol_MA5']
                    if 'rsi' in df_features_full.columns:
                        df.loc[df_features_full.index, 'rsi'] = df_features_full['rsi']

                
                # 为每一行生成信号
                for i in range(len(df)):
                    try:
                        # 获取当前行在df_features_full中的索引
                        if i >= len(df_features_full):
                            continue
                            
                        current_idx = df_features_full.index[i]
                        
                        # 创建特征（用于模型预测）
                        df_features, feature_cols = trainer.create_features(df.iloc[:i+1])
                        if len(df_features) > 0:
                            # 只使用模型训练时使用的特征
                            model_features = [col for col in feature_cols if col in golden_model.feature_names_in_]
                            if len(model_features) > 0:
                                # 获取当前数据
                                current_data = df_features.iloc[-1:][model_features]
                                # 预测概率
                                y_prob = golden_model.predict_proba(current_data)[0]
                                buy_probability = y_prob[1]  # 正类（金叉/上涨）概率
                                sell_probability = 1 - buy_probability  # 负类概率
                                
                                # 获取当前数据的连续下降天数和其他指标（从预先计算的df_features_full中获取）
                                # 这样确保使用的是基于t-1之前数据统计的连续下降天数
                                consecutive_down = df_features_full.loc[current_idx, 'consecutive_down'] if 'consecutive_down' in df_features_full.columns else 0
                                rsi = df_features_full.loc[current_idx, 'rsi'] if 'rsi' in df_features_full.columns else 50
                                volume_ratio = df_features_full.loc[current_idx, 'vol/Vol_MA5'] if 'vol/Vol_MA5' in df_features_full.columns else 1.0
                                
                                # 应用实战应用建议的买入规则
                                buy_signal = False
                                if consecutive_down <= 3 and buy_probability > 0.6:
                                    if (rsi < 30 or volume_ratio > 1.5) and consecutive_down <= 5:
                                        buy_signal = True
                                
                                # 卖出信号：当连续下降天数>5天或预测下跌概率>60%时
                                sell_signal = sell_probability > 0.6 or consecutive_down > 5
                                
                                # 更新信号
                                df.loc[df.index[i], 'buy_signal'] = 1.0 if buy_signal else 0.0
                                df.loc[df.index[i], 'sell_signal'] = 1.0 if sell_signal else 0.0
                                df.loc[df.index[i], 'buy_probability'] = buy_probability
                                df.loc[df.index[i], 'sell_probability'] = sell_probability
                                
                                # 调试输出
                                # if i % 10 == 0 or consecutive_down > 0:  # 每10行或连续下降天数>0时输出调试信息
                                    # print(f"DEBUG 第{i}行 (日期{df.iloc[i]['date']}): consecutive_down={consecutive_down}, rsi={rsi:.2f}, volume_ratio={volume_ratio:.2f}, buy_prob={buy_probability:.2f}, buy_signal={buy_signal}")
                    except Exception as e:
                        print(f"为历史数据生成交易信号时出错 (第{i}行): {e}")
                        import traceback
                        traceback.print_exc()
                        continue
            else:
                print("未找到模型，使用金叉死叉规则生成交易信号")
                # 回退到原始的金叉死叉规则
                for i in range(1, len(df)):
                    try:
                        golden_cross_val = float(df['golden_cross'].iloc[i])
                        death_cross_val = float(df['death_cross'].iloc[i])
                        if golden_cross_val == 1.0:
                            df.loc[df.index[i], 'buy_signal'] = 1.0
                            df.loc[df.index[i], 'buy_price'] = df['close'].iloc[i]
                        elif death_cross_val == 1.0:
                            df.loc[df.index[i], 'sell_signal'] = 1.0
                            df.loc[df.index[i], 'sell_price'] = df['close'].iloc[i]
                    except Exception as e:
                        print(f"生成交易信号时出错: {e}")
                        continue
        except Exception as e:
            print(f"预测金叉和死叉失败: {e}")
            # 如果预测失败，使用默认值
            df['pred_golden_cross'] = 0
            df['pred_golden_cross_proba'] = 0.0
            df['pred_death_cross'] = 0
            df['pred_death_cross_proba'] = 0.0
            df['pred_close'] = 0.0
            df['buy_price'] = 0.0
            df['sell_price'] = 0.0
            df['buy_probability'] = 0.0
            df['sell_probability'] = 0.0
            df['high_return_action'] = "HOLD"
            df['high_return_position'] = 0.0
            df['super_trend_prob'] = 0.0
        
        # 基本面数据功能已删除（需要Tushare 2000积分以上权限）
        fundamentals = {}
        
        # 准备历史数据（最近90天）
        history_data = []
        if len(df) > 0:
            recent_df = df.tail(90)
            for _, row in recent_df.iterrows():
                # 构建包含所有tushare字段的历史数据
                # 使用volume字段作为vol的值，因为tushare返回的是volume字段
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
                    'ma5': float(row['ma5']) if not pd.isna(row['ma5']) else None,  # SMA_5
                    'ma10': float(row['ma10']) if not pd.isna(row['ma10']) else None,
                    'ma20': float(row['ma20']) if not pd.isna(row['ma20']) else None,  # SMA_20
                    'ema12': float(row['ema12']) if not pd.isna(row['ema12']) else None,  # EMA_12
                    'ema26': float(row['ema26']) if not pd.isna(row['ema26']) else None,  # EMA_26
                    'rsi': float(row['rsi']) if not pd.isna(row['rsi']) else None,
                    'macd': float(row['macd']) if not pd.isna(row['macd']) else None,  # DIF
                    'macd_signal': float(row['macd_signal']) if not pd.isna(row['macd_signal']) else None,  # DEA
                    'macd_hist': float(row['macd_hist']) if not pd.isna(row['macd_hist']) else None,  # MACD Hist
                    'boll_upper': float(row['boll_upper']) if not pd.isna(row['boll_upper']) else None,
                    'boll_mid': float(row['boll_mid']) if not pd.isna(row['boll_mid']) else None,
                    'boll_lower': float(row['boll_lower']) if not pd.isna(row['boll_lower']) else None,
                    'ma60': float(row['ma60']) if not pd.isna(row['ma60']) else None,  # 60日均线
                    'atr': float(row['atr']) if not pd.isna(row['atr']) else None,
                    'golden_cross': int(float(row['golden_cross'])) if 'golden_cross' in row else 0,
                    'death_cross': int(float(row['death_cross'])) if 'death_cross' in row else 0,
                    'buy_signal': bool(float(row['buy_signal'])) if 'buy_signal' in row else False,
                    'sell_signal': bool(float(row['sell_signal'])) if 'sell_signal' in row else False,
                    'buy_price': float(row['buy_price']) if 'buy_price' in row and not pd.isna(row['buy_price']) else 0,
                    'sell_price': float(row['sell_price']) if 'sell_price' in row and not pd.isna(row['sell_price']) else 0,
                    'pred_close': float(row['pred_close']) if 'pred_close' in row and not pd.isna(row['pred_close']) else 0,
                    'consecutive_down': int(row['consecutive_down']) if 'consecutive_down' in row else 0,
                    'consecutive_up': int(row['consecutive_up']) if 'consecutive_up' in row else 0,
                    'pred_golden_cross': bool(float(row['pred_golden_cross'])) if 'pred_golden_cross' in row else False,
                    'pred_golden_cross_proba': float(row['pred_golden_cross_proba']) if 'pred_golden_cross_proba' in row else 0.0,
                    'pred_death_cross': bool(float(row['pred_death_cross'])) if 'pred_death_cross' in row else False,
                    'buy_probability': float(row['buy_probability']) if 'buy_probability' in row else 0.0,
                    'sell_probability': float(row['sell_probability']) if 'sell_probability' in row else 0.0,
                    'pred_death_cross_proba': float(row['pred_death_cross_proba']) if 'pred_death_cross_proba' in row else 0.0,
                    'high_return_action': row['high_return_action'] if 'high_return_action' in row else "HOLD",
                    'high_return_position': float(row['high_return_position']) if 'high_return_position' in row else 0.0,
                    'super_trend_prob': float(row['super_trend_prob']) if 'super_trend_prob' in row else 0.0
                }
                history_data.append(history_item)
        
        # 生成交易信号
        from src.models.model_trainer import ModelTrainer
        trainer = ModelTrainer()
        
        # 准备用于生成交易信号的数据
        signal_data = df.copy()
        if 'pct_chg' not in signal_data.columns:
            signal_data['pct_chg'] = signal_data['close'].pct_change() * 100
        
        # 生成交易信号
        # 这里使用简单的概率阈值生成信号，实际应用中可以使用训练好的模型
        buy_probability = up_prob
        sell_probability = down_prob
        buy_threshold = 0.6
        sell_threshold = 0.6
        
        buy_signal = buy_probability > buy_threshold
        sell_signal = sell_probability > sell_threshold
        
        # 生成消息
        message = []
        if buy_signal:
            message.append(f"⭐ 预测上涨概率较高（{buy_probability:.2%}），建议买入")
        if sell_signal:
            message.append(f"❌ 预测下跌概率较高（{sell_probability:.2%}），建议卖出")
        if not buy_signal and not sell_signal:
            message.append("📊 预测涨跌概率均较低，建议观望")
        
        # 获取高收益策略信号
        high_return_action = df['high_return_action'].iloc[-1] if 'high_return_action' in df.columns else "HOLD"
        high_return_position = df['high_return_position'].iloc[-1] if 'high_return_position' in df.columns else 0.0
        super_trend_prob = df['super_trend_prob'].iloc[-1] if 'super_trend_prob' in df.columns else 0.0
        
        # 添加高收益策略消息
        if high_return_action == "HEAVY_BUY":
            message.append(f"🚀 高收益策略：强烈买入，建议仓位 {high_return_position:.1%}")
        elif high_return_action == "BUY":
            message.append(f"📈 高收益策略：买入，建议仓位 {high_return_position:.1%}")
        elif high_return_action == "HOLD":
            message.append(f"📊 高收益策略：观望")
        
        trading_signals = {
            'buy_signal': buy_signal,
            'sell_signal': sell_signal,
            'buy_probability': float(buy_probability),
            'sell_probability': float(sell_probability),
            'message': ' '.join(message),
            'high_return_action': high_return_action,
            'high_return_position': float(high_return_position),
            'super_trend_prob': float(super_trend_prob)
        }
        
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
            'trading_signals': trading_signals,  # 新增交易信号
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

@app.route('/api/backtest_500pct', methods=['POST'])
def backtest_500pct():
    """回测高收益策略"""
    data = request.json
    ts_code = data.get('ts_code')
    model_type = data.get('model_type', 'xgboost')
    
    try:
        # 直接获取tushare数据
        import tushare as ts
        
        # 直接使用正确的token
        TUSHARE_TOKEN = '68e431d47c0319d7bdea7cd1daf164392d22c8a9216d99476354be78'
        
        # 初始化tushare
        pro = ts.pro_api(TUSHARE_TOKEN)
        
        # 获取股票数据
        import datetime
        today = datetime.datetime.now().strftime('%Y%m%d')
        start_date = (datetime.datetime.now() - datetime.timedelta(days=1000)).strftime('%Y%m%d')  # 多拿数据
        
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
        
        # 使用ModelTrainer进行高收益策略回测
        from src.models.model_trainer import ModelTrainer
        trainer = ModelTrainer()
        
        # 删除非数值列，避免模型训练错误
        if 'ts_code' in df.columns:
            df = df.drop('ts_code', axis=1)
        
        # 训练高收益模型
        model, metrics = trainer.train_and_evaluate_500pct(df, model_type=model_type)
        
        # 回测结果
        backtest_result = metrics['backtest']
        
        return jsonify(backtest_result)
    except Exception as e:
        print(f"高收益策略回测失败: {e}")
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

@app.route('/api/train_500pct', methods=['POST'])
def train_500pct_model():
    """训练高收益模型"""
    data = request.json
    ts_code = data.get('ts_code')
    model_type = data.get('model_type', 'xgboost')
    
    try:
        # 直接获取tushare数据
        import tushare as ts
        
        # 直接使用正确的token
        TUSHARE_TOKEN = '68e431d47c0319d7bdea7cd1daf164392d22c8a9216d99476354be78'
        
        # 初始化tushare
        pro = ts.pro_api(TUSHARE_TOKEN)
        
        # 获取股票数据
        import datetime
        today = datetime.datetime.now().strftime('%Y%m%d')
        start_date = (datetime.datetime.now() - datetime.timedelta(days=1000)).strftime('%Y%m%d')  # 多拿数据
        
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
        
        # 使用ModelTrainer训练高收益模型
        from src.models.model_trainer import ModelTrainer
        trainer = ModelTrainer()
        
        # 训练高收益模型
        model, metrics = trainer.train_and_evaluate_500pct(df, model_type=model_type)
        
        return jsonify({
            'message': '高收益模型训练完成',
            'metrics': metrics
        })
    except Exception as e:
        print(f"高收益模型训练失败: {e}")
        import traceback
        traceback.print_exc()
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
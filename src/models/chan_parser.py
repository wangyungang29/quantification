"""
缠论解析器模块
基于缠论（Chan Theory）实现完整的行情分析功能
日线级别优化版：严格过滤中继底、二买确认、均线过滤
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from enum import Enum


class SignalType(Enum):
    """信号类型"""
    BUY_1 = '一买'
    BUY_2 = '二买'
    BUY_3 = '三买'
    SELL_1 = '一卖'
    SELL_2 = '二卖'
    SELL_3 = '三卖'


class ChanParser:
    """缠论解析器（日线级别优化版）"""

    def __init__(self, params: Dict = None):
        """初始化缠论解析器"""
        if params is None:
            params = self._default_params()

        self.params = params
        self.bars = None
        self.std_bars = None
        self.fengs = None
        self.pens = None
        self.segments = None
        self.zhongshus = None
        self.signals = None
        self.ma5 = None
        self.ma10 = None
        self.price_channels = None

    def _default_params(self) -> Dict:
        """默认参数"""
        return {
            "min_bars_pen": 3,
            "segment_strength": 2,
            "central_overlap": 3,
            "macd_threshold": 0.5,
            "include_gap": False,
            "pen_break_ratio": 0.5,
            "stop_loss_ratio": 0.08,
            "ma_short": 5,
            "ma_long": 10,
        }

    def _calculate_ma(self, df: pd.DataFrame) -> Tuple[Optional[pd.Series], Optional[pd.Series]]:
        """计算均线"""
        if df is None or len(df) < self.params['ma_long']:
            return None, None

        ma_short = df['close'].rolling(window=self.params['ma_short']).mean()
        ma_long = df['close'].rolling(window=self.params['ma_long']).mean()

        self.ma5 = ma_short
        self.ma10 = ma_long

        return ma_short, ma_long

    def _check_price_channel(self, df: pd.DataFrame, current_idx: int, lookback: int = 20) -> Dict:
        """检查价格通道，判断是否处于下降通道"""
        if df is None or current_idx < lookback:
            return {'is_downtrend': False, 'channel_high': 0, 'channel_low': 0}

        start_idx = max(0, current_idx - lookback)
        segment = df.iloc[start_idx:current_idx + 1].copy()

        if len(segment) < 5:
            return {'is_downtrend': False, 'channel_high': 0, 'channel_low': 0}

        highs = segment['high'].values
        lows = segment['low'].values

        highs_trend = np.polyfit(range(len(highs)), highs, 1)[0]
        lows_trend = np.polyfit(range(len(lows)), lows, 1)[0]

        is_downtrend = highs_trend < 0 and lows_trend < 0

        channel_high = max(highs[-5:]) if len(highs) >= 5 else max(highs)
        channel_low = min(lows[-5:]) if len(lows) >= 5 else min(lows)

        self.price_channels = {'is_downtrend': is_downtrend, 'channel_high': channel_high, 'channel_low': channel_low}

        return self.price_channels

    def _check_ma_filter(self, current_price: float, current_idx: int) -> bool:
        """检查均线过滤：收盘价必须站上短期或长期均线"""
        if self.ma5 is None or self.ma10 is None:
            return True

        if current_idx >= len(self.ma5) or current_idx >= len(self.ma10):
            return True

        ma5_val = self.ma5.iloc[current_idx]
        ma10_val = self.ma10.iloc[current_idx]

        if pd.isna(ma5_val) or pd.isna(ma10_val):
            return True

        return current_price >= ma5_val or current_price >= ma10_val

    def _check_ma_filter_at_idx(self, idx: int) -> bool:
        """检查特定索引处的均线状态"""
        if self.ma5 is None or self.ma10 is None:
            return True

        if idx >= len(self.ma5) or idx >= len(self.ma10):
            return True

        ma5_val = self.ma5.iloc[idx]
        ma10_val = self.ma10.iloc[idx]

        if pd.isna(ma5_val) or pd.isna(ma10_val):
            return True

        return True

    def process_kline(self, df: pd.DataFrame) -> pd.DataFrame:
        """处理K线包含关系，生成标准化K线"""
        if df is None or df.empty:
            return pd.DataFrame()

        self.bars = df.copy()
        self.bars.reset_index(drop=True, inplace=True)

        self._calculate_ma(df)

        std_bars = []
        current_high = df.iloc[0]['high']
        current_low = df.iloc[0]['low']
        current_open = df.iloc[0]['open']
        current_close = df.iloc[0]['close']

        for i in range(1, len(df)):
            bar = df.iloc[i]

            if bar['high'] <= current_high and bar['low'] >= current_low:
                current_high = max(current_high, bar['high'])
                current_low = min(current_low, bar['low'])
                current_close = bar['close']
            elif bar['high'] >= current_high and bar['low'] <= current_low:
                current_high = max(current_high, bar['high'])
                current_low = min(current_low, bar['low'])
                current_close = bar['close']
            else:
                std_bars.append({
                    'high': current_high,
                    'low': current_low,
                    'open': current_open,
                    'close': current_close
                })
                current_high = bar['high']
                current_low = bar['low']
                current_open = bar['open']
                current_close = bar['close']

        std_bars.append({
            'high': current_high,
            'low': current_low,
            'open': current_open,
            'close': current_close
        })

        self.std_bars = pd.DataFrame(std_bars)
        return self.std_bars

    def identify_feng(self) -> List[Dict]:
        """识别顶分型和底分型"""
        if self.std_bars is None or len(self.std_bars) < 3:
            return []

        fengs = []

        for i in range(1, len(self.std_bars) - 1):
            prev = self.std_bars.iloc[i-1]
            curr = self.std_bars.iloc[i]
            next_bar = self.std_bars.iloc[i+1]

            if curr['high'] > prev['high'] and curr['high'] > next_bar['high']:
                fengs.append({
                    'index': i,
                    'type': '顶分型',
                    'high': curr['high'],
                    'low': min(prev['low'], curr['low'], next_bar['low']),
                    'bars': [i-1, i, i+1]
                })
            elif curr['low'] < prev['low'] and curr['low'] < next_bar['low']:
                fengs.append({
                    'index': i,
                    'type': '底分型',
                    'low': curr['low'],
                    'high': max(prev['high'], curr['high'], next_bar['high']),
                    'bars': [i-1, i, i+1]
                })

        self.fengs = fengs
        return fengs

    def generate_pens(self) -> List[Dict]:
        """连接分型生成笔"""
        if self.fengs is None or len(self.fengs) < 2:
            return []

        pens = []
        start_feng = None

        for i, feng in enumerate(self.fengs):
            if start_feng is None:
                start_feng = feng
                continue

            if start_feng['type'] != feng['type']:
                if self._validate_pen(start_feng, feng):
                    pen = {
                        'start_index': start_feng['index'],
                        'end_index': feng['index'],
                        'start_type': start_feng['type'],
                        'end_type': feng['type'],
                        'direction': 'up' if start_feng['type'] == '底分型' else 'down',
                        'high': max(start_feng['high'], feng['high']),
                        'low': min(start_feng['low'], feng['low']),
                        'length': abs(feng['high'] - feng['low']),
                        'bars_count': feng['index'] - start_feng['index']
                    }
                    pens.append(pen)

                start_feng = feng

        self.pens = pens
        return pens

    def _validate_pen(self, start_feng: Dict, end_feng: Dict) -> bool:
        """验证笔的有效性"""
        bars_count = end_feng['index'] - start_feng['index']
        if bars_count < self.params['min_bars_pen']:
            return False

        if start_feng['type'] == '底分型':
            if end_feng['high'] <= start_feng['high']:
                return False
        else:
            if end_feng['low'] >= start_feng['low']:
                return False

        return True

    def generate_segments(self) -> List[Dict]:
        """由笔构成线段"""
        if self.pens is None or len(self.pens) < 2:
            return []

        segments = []
        current_segment = None

        for pen in self.pens:
            if current_segment is None:
                current_segment = {
                    'start_index': pen['start_index'],
                    'end_index': pen['end_index'],
                    'direction': pen['direction'],
                    'pens': [pen],
                    'high': pen['high'],
                    'low': pen['low']
                }
                continue

            if pen['direction'] == current_segment['direction']:
                current_segment['end_index'] = pen['end_index']
                current_segment['pens'].append(pen)
                current_segment['high'] = max(current_segment['high'], pen['high'])
                current_segment['low'] = min(current_segment['low'], pen['low'])
            else:
                if self._check_segment_break(current_segment, pen):
                    segments.append(current_segment)
                    current_segment = {
                        'start_index': pen['start_index'],
                        'end_index': pen['end_index'],
                        'direction': pen['direction'],
                        'pens': [pen],
                        'high': pen['high'],
                        'low': pen['low']
                    }
                else:
                    current_segment['pens'].append(pen)
                    current_segment['end_index'] = pen['end_index']
                    if pen['direction'] == 'up':
                        current_segment['high'] = max(current_segment['high'], pen['high'])
                    else:
                        current_segment['low'] = min(current_segment['low'], pen['low'])

        if current_segment:
            segments.append(current_segment)

        self.segments = segments
        return segments

    def _check_segment_break(self, segment: Dict, pen: Dict) -> bool:
        """检查线段是否被破坏"""
        if len(segment['pens']) < self.params['segment_strength']:
            return False

        if segment['direction'] == 'up':
            return pen['low'] < segment['low']
        else:
            return pen['high'] > segment['high']

    def identify_zhongshu(self) -> List[Dict]:
        """识别中枢"""
        if self.segments is None or len(self.segments) < 3:
            return []

        zhongshus = []
        i = 0

        while i < len(self.segments) - 2:
            seg1 = self.segments[i]
            seg2 = self.segments[i+1]
            seg3 = self.segments[i+2]

            if (seg1['direction'] == 'up' and seg2['direction'] == 'down' and seg3['direction'] == 'up') or \
               (seg1['direction'] == 'down' and seg2['direction'] == 'up' and seg3['direction'] == 'down'):

                high = min(seg1['high'], seg2['high'], seg3['high'])
                low = max(seg1['low'], seg2['low'], seg3['low'])

                if high > low:
                    zhongshu = {
                        'start_index': seg1['start_index'],
                        'end_index': seg3['end_index'],
                        'high': high,
                        'low': low,
                        'center': (high + low) / 2,
                        'range': high - low,
                        'segments': [seg1, seg2, seg3],
                        'type': '扩展' if seg3['direction'] == seg1['direction'] else '新生'
                    }
                    zhongshus.append(zhongshu)
                    i += 3
                else:
                    i += 1
            else:
                i += 1

        self.zhongshus = zhongshus
        return zhongshus

    def detect_buy_signals(self) -> List[Dict]:
        """
        识别买点信号
        优化版逻辑：
        1. 一买：线段从下跌转为上涨（但下降通道中不买入）
        2. 二买：回调后再次上涨形成
        3. 过滤中继底：下降通道中不买入
        """
        signals = []

        if self.segments is None or self.bars is None:
            return signals

        for i in range(len(self.segments) - 1):
            seg1 = self.segments[i]
            seg2 = self.segments[i+1]

            if seg1['direction'] == 'down' and seg2['direction'] == 'up':
                seg2_end_idx = seg2['end_index']

                if seg2_end_idx >= len(self.bars):
                    continue

                channel_info = self._check_price_channel(self.bars, seg2_end_idx)

                if channel_info['is_downtrend']:
                    signals.append({
                        'type': SignalType.BUY_1.value,
                        'index': seg2_end_idx,
                        'price': seg2['low'],
                        'stop_loss': seg1['low'],
                        'description': '一买-下降通道-中继底',
                        'is_valid': False,
                        'filter_reason': '中继底过滤'
                    })
                    continue

                ma_ok = self._check_ma_filter(seg2['low'], seg2_end_idx)

                signals.append({
                    'type': SignalType.BUY_1.value,
                    'index': seg2_end_idx,
                    'price': seg2['low'],
                    'stop_loss': seg1['low'],
                    'description': '一买-有效信号' if ma_ok else '一买-均线未站上',
                    'is_valid': ma_ok,
                    'channel_check': channel_info
                })

        self._detect_buy2_signals(signals)
        self._detect_buy3_signals(signals)

        self.signals = signals
        return signals

    def _detect_buy2_signals(self, signals: List[Dict]):
        """
        识别二买信号（简化宽松版）
        二买逻辑：上涨回调后再次上涨，回调低点不跌破前一波低点
        """
        if self.segments is None or self.bars is None or self.fengs is None:
            return

        for i in range(len(self.segments) - 1):
            seg = self.segments[i]

            if seg['direction'] != 'up' or i == 0:
                continue

            prev_seg = self.segments[i - 1]
            if prev_seg['direction'] != 'down':
                continue

            base_low = prev_seg['low']
            seg_end_idx = min(seg['end_index'], len(self.bars) - 1)

            for j in range(seg['start_index'], seg_end_idx + 1):
                if j + 2 >= len(self.bars):
                    continue

                bar_j = self.bars.iloc[j]
                bar_j1 = self.bars.iloc[j + 1]
                bar_j2 = self.bars.iloc[j + 2]

                if bar_j['close'] < bar_j1['close'] and bar_j1['close'] > bar_j2['close']:
                    callback_low = bar_j1['low']

                    if callback_low < base_low:
                        continue

                    entry_idx = j + 2
                    if entry_idx >= len(self.bars):
                        continue

                    entry_price = bar_j2['close']
                    channel_info = self._check_price_channel(self.bars, entry_idx)
                    ma_ok = self._check_ma_filter(entry_price, entry_idx)

                    if ma_ok and not channel_info['is_downtrend']:
                        signals.append({
                            'type': SignalType.BUY_2.value,
                            'index': j + 1,
                            'price': base_low,
                            'stop_loss': base_low,
                            'description': '二买-有效信号',
                            'is_valid': True,
                            'base_low': base_low,
                            'callback_low': callback_low,
                            'entry_price': entry_price,
                            'entry_idx': entry_idx
                        })
                    break

    def _detect_buy3_signals(self, signals: List[Dict]):
        """识别三买（中枢上沿不回踩）"""
        if not self.zhongshus or self.segments is None or self.bars is None:
            return

        for zs in self.zhongshus:
            for i, seg in enumerate(self.segments):
                if seg['start_index'] > zs['end_index'] and seg['direction'] == 'up':
                    if i + 1 < len(self.segments):
                        next_seg = self.segments[i+1]
                        if next_seg['direction'] == 'down' and next_seg['low'] > zs['high']:
                            current_idx = next_seg['end_index']

                            if current_idx >= len(self.bars):
                                continue

                            current_price = self.bars.iloc[current_idx]['close']

                            ma_ok = self._check_ma_filter(current_price, current_idx)
                            channel_info = self._check_price_channel(self.bars, current_idx)

                            if ma_ok and not channel_info['is_downtrend']:
                                signals.append({
                                    'type': SignalType.BUY_3.value,
                                    'index': next_seg['end_index'],
                                    'price': next_seg['low'],
                                    'stop_loss': zs['high'],
                                    'description': '三买-有效信号',
                                    'is_valid': True,
                                    'zhongshu_high': zs['high'],
                                    'zhongshu_low': zs['low']
                                })
                            else:
                                signals.append({
                                    'type': SignalType.BUY_3.value,
                                    'index': next_seg['end_index'],
                                    'price': next_seg['low'],
                                    'stop_loss': zs['high'],
                                    'description': '三买-条件未满足',
                                    'is_valid': False,
                                    'zhongshu_high': zs['high'],
                                    'zhongshu_low': zs['low']
                                })
                    break

    def _check_beichi(self, down_seg: Dict, up_seg: Dict) -> bool:
        """检查背驰"""
        if self.bars is None or len(self.bars) < 14:
            return False

        df = self.bars.copy()
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        macd_signal = macd.ewm(span=9, adjust=False).mean()
        macd_hist = (macd - macd_signal) * 2

        down_start = down_seg['start_index']
        down_end = down_seg['end_index']
        up_end = up_seg['end_index']

        down_macd_sum = abs(macd_hist.iloc[down_start:down_end].sum())
        up_macd_sum = macd_hist.iloc[down_end:up_end].sum()

        return up_macd_sum > down_macd_sum * self.params['macd_threshold']

    def analyze(self, df: pd.DataFrame) -> Dict:
        """完整分析流程"""
        result = {}

        result['std_bars'] = self.process_kline(df)
        result['fengs'] = self.identify_feng()
        result['pens'] = self.generate_pens()
        result['segments'] = self.generate_segments()
        result['zhongshus'] = self.identify_zhongshu()
        result['signals'] = self.detect_buy_signals()

        return result

    def get_latest_signal(self) -> Optional[Dict]:
        """获取最新有效信号"""
        if self.signals is None or len(self.signals) == 0:
            return None

        valid_signals = [s for s in self.signals if s.get('is_valid', False)]

        if not valid_signals:
            return None

        return sorted(valid_signals, key=lambda x: x['index'])[-1]

    def get_valid_signals(self) -> List[Dict]:
        """获取所有有效信号"""
        if self.signals is None:
            return []

        return [s for s in self.signals if s.get('is_valid', False)]
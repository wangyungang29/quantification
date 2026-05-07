"""
缠论解析器模块
基于缠论（Chan Theory）实现完整的行情分析功能
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
    """缠论解析器"""

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

    def _default_params(self) -> Dict:
        """默认参数"""
        return {
            "min_bars_pen": 3,        # 一笔最少K线数（降低要求）
            "segment_strength": 2,    # 线段破坏所需笔数（降低要求）
            "central_overlap": 3,     # 中枢最小重叠段
            "macd_threshold": 0.3,    # 背驰判定阈值（降低要求）
            "include_gap": False,     # 是否包含跳空缺口
            "pen_break_ratio": 0.3,   # 笔破坏比例（降低要求）
        }

    def process_kline(self, df: pd.DataFrame) -> pd.DataFrame:
        """处理K线包含关系，生成标准化K线"""
        if df is None or df.empty:
            return pd.DataFrame()

        self.bars = df.copy()
        self.bars.reset_index(drop=True, inplace=True)

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
        if self.pens is None or len(self.pens) < 2:  # 降低要求
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
        """识别买点信号（一买、二买、三买）"""
        signals = []

        if self.segments is None:
            return signals

        # 简化的一买识别：线段从下跌转为上涨
        for i in range(len(self.segments) - 1):
            seg1 = self.segments[i]
            seg2 = self.segments[i+1]

            if seg1['direction'] == 'down' and seg2['direction'] == 'up':
                signals.append({
                    'type': SignalType.BUY_1.value,
                    'index': seg2['start_index'],
                    'price': seg2['low'],
                    'stop_loss': seg1['low'],
                    'description': '下跌转上涨一买'
                })

        # 简化的二买识别：回调后再次上涨
        for i in range(len(self.segments) - 2):
            seg1 = self.segments[i]
            seg2 = self.segments[i+1]
            seg3 = self.segments[i+2]

            if seg1['direction'] == 'down' and seg2['direction'] == 'up' and seg3['direction'] == 'down':
                signals.append({
                    'type': SignalType.BUY_2.value,
                    'index': seg3['end_index'],
                    'price': seg3['low'],
                    'stop_loss': seg1['low'],
                    'description': '回调二买'
                })

        # 识别三买（中枢上沿不回踩）
        if self.zhongshus:
            for zs in self.zhongshus:
                for i, seg in enumerate(self.segments):
                    if seg['start_index'] > zs['end_index'] and seg['direction'] == 'up':
                        if i + 1 < len(self.segments):
                            next_seg = self.segments[i+1]
                            if next_seg['direction'] == 'down' and next_seg['low'] > zs['high']:
                                signals.append({
                                    'type': SignalType.BUY_3.value,
                                    'index': next_seg['end_index'],
                                    'price': next_seg['low'],
                                    'stop_loss': zs['high'],
                                    'description': '中枢上沿不回踩三买',
                                    'zhongshu_high': zs['high'],
                                    'zhongshu_low': zs['low']
                                })
                        break

        # 如果没有线段信号，尝试使用笔信号
        if not signals and self.pens:
            for i in range(len(self.pens) - 1):
                pen1 = self.pens[i]
                pen2 = self.pens[i+1]
                
                if pen1['direction'] == 'down' and pen2['direction'] == 'up':
                    signals.append({
                        'type': SignalType.BUY_1.value,
                        'index': pen2['start_index'],
                        'price': pen2['low'],
                        'stop_loss': pen1['low'],
                        'description': '笔级别一买'
                    })

        self.signals = signals
        return signals

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
        """获取最新信号"""
        if self.signals is None or len(self.signals) == 0:
            return None

        return sorted(self.signals, key=lambda x: x['index'])[-1]
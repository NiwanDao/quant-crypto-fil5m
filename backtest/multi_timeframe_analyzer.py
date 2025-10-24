"""
多时间框架分析模块
包含不同时间周期的趋势分析、支撑阻力识别、多周期信号融合等
"""

import pandas as pd
import numpy as np
import talib
from typing import Dict, List, Tuple, Optional
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class MultiTimeframeAnalyzer:
    """多时间框架分析器"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.timeframes = ['15m', '1h', '4h', '1d']  # 支持的时间框架
        self.resampled_data = {}
        
    def resample_data(self, data: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """重采样数据到指定时间框架"""
        if timeframe == '15m':
            return data
        elif timeframe == '1h':
            return data.resample('1H').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()
        elif timeframe == '4h':
            return data.resample('4H').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()
        elif timeframe == '1d':
            return data.resample('1D').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()
        else:
            raise ValueError(f"不支持的时间框架: {timeframe}")
    
    def calculate_trend_strength(self, data: pd.DataFrame, period: int = 20) -> pd.Series:
        """计算趋势强度"""
        close = data['close']
        
        # ADX指标
        adx = talib.ADX(data['high'].values, data['low'].values, close.values, timeperiod=period)
        adx = pd.Series(adx, index=close.index)
        
        # 趋势方向
        sma_short = close.rolling(period//2).mean()
        sma_long = close.rolling(period).mean()
        trend_direction = np.where(sma_short > sma_long, 1, -1)
        
        # 趋势强度 = ADX * 方向
        trend_strength = adx * trend_direction
        
        return trend_strength.fillna(0)
    
    def identify_support_resistance(self, data: pd.DataFrame, window: int = 20) -> Tuple[pd.Series, pd.Series]:
        """识别支撑和阻力位"""
        high = data['high']
        low = data['low']
        close = data['close']
        
        # 使用滚动窗口找极值点
        resistance = high.rolling(window, center=True).max()
        support = low.rolling(window, center=True).min()
        
        # 过滤掉不是真正极值的点
        resistance_mask = high == resistance
        support_mask = low == support
        
        # 计算支撑阻力强度
        resistance_strength = pd.Series(0, index=close.index)
        support_strength = pd.Series(0, index=close.index)
        
        for i in range(window, len(close) - window):
            if resistance_mask.iloc[i]:
                # 计算该阻力位的测试次数
                recent_highs = high.iloc[i-window:i+window]
                touches = (recent_highs >= resistance.iloc[i] * 0.99).sum()
                resistance_strength.iloc[i] = touches
            
            if support_mask.iloc[i]:
                # 计算该支撑位的测试次数
                recent_lows = low.iloc[i-window:i+window]
                touches = (recent_lows <= support.iloc[i] * 1.01).sum()
                support_strength.iloc[i] = touches
        
        return support_strength, resistance_strength
    
    def calculate_momentum_indicators(self, data: pd.DataFrame) -> Dict:
        """计算动量指标"""
        close = data['close']
        high = data['high']
        low = data['low']
        volume = data['volume']
        
        indicators = {}
        
        # RSI
        indicators['rsi'] = pd.Series(talib.RSI(close.values, timeperiod=14), index=close.index)
        
        # MACD
        macd, macd_signal, macd_hist = talib.MACD(close.values)
        indicators['macd'] = pd.Series(macd, index=close.index)
        indicators['macd_signal'] = pd.Series(macd_signal, index=close.index)
        indicators['macd_hist'] = pd.Series(macd_hist, index=close.index)
        
        # 随机指标
        stoch_k, stoch_d = talib.STOCH(high.values, low.values, close.values)
        indicators['stoch_k'] = pd.Series(stoch_k, index=close.index)
        indicators['stoch_d'] = pd.Series(stoch_d, index=close.index)
        
        # 威廉指标
        indicators['williams_r'] = pd.Series(talib.WILLR(high.values, low.values, close.values), index=close.index)
        
        # 成交量指标
        indicators['obv'] = pd.Series(talib.OBV(close.values, volume.values), index=close.index)
        
        return indicators
    
    def calculate_volatility_indicators(self, data: pd.DataFrame) -> Dict:
        """计算波动率指标"""
        close = data['close']
        high = data['high']
        low = data['low']
        
        indicators = {}
        
        # ATR
        indicators['atr'] = pd.Series(talib.ATR(high.values, low.values, close.values), index=close.index)
        
        # 布林带
        bb_upper, bb_middle, bb_lower = talib.BBANDS(close.values)
        indicators['bb_upper'] = pd.Series(bb_upper, index=close.index)
        indicators['bb_middle'] = pd.Series(bb_middle, index=close.index)
        indicators['bb_lower'] = pd.Series(bb_lower, index=close.index)
        indicators['bb_width'] = (indicators['bb_upper'] - indicators['bb_lower']) / indicators['bb_middle']
        
        # 历史波动率
        returns = close.pct_change()
        indicators['hv_20'] = returns.rolling(20).std() * np.sqrt(252 * 24 * 4)
        indicators['hv_50'] = returns.rolling(50).std() * np.sqrt(252 * 24 * 4)
        
        return indicators
    
    def analyze_market_structure(self, data: pd.DataFrame) -> Dict:
        """分析市场结构"""
        close = data['close']
        high = data['high']
        low = data['low']
        
        # 识别高低点
        highs = high.rolling(5, center=True).max() == high
        lows = low.rolling(5, center=True).min() == low
        
        # 计算高低点序列
        high_points = close[highs]
        low_points = close[lows]
        
        # 趋势结构分析
        structure = {
            'higher_highs': 0,
            'lower_highs': 0,
            'higher_lows': 0,
            'lower_lows': 0
        }
        
        if len(high_points) >= 2:
            recent_highs = high_points.tail(2)
            if recent_highs.iloc[-1] > recent_highs.iloc[-2]:
                structure['higher_highs'] = 1
            else:
                structure['lower_highs'] = 1
        
        if len(low_points) >= 2:
            recent_lows = low_points.tail(2)
            if recent_lows.iloc[-1] > recent_lows.iloc[-2]:
                structure['higher_lows'] = 1
            else:
                structure['lower_lows'] = 1
        
        return structure
    
    def calculate_timeframe_alignment(self, data: pd.DataFrame) -> pd.Series:
        """计算多时间框架对齐度"""
        alignment_scores = []
        
        for tf in self.timeframes:
            if tf == '15m':
                tf_data = data
            else:
                tf_data = self.resample_data(data, tf)
            
            # 计算该时间框架的趋势
            trend_strength = self.calculate_trend_strength(tf_data)
            
            # 对齐到原始时间框架
            if tf != '15m':
                trend_strength = trend_strength.reindex(data.index, method='ffill')
            
            alignment_scores.append(trend_strength)
        
        # 计算对齐度（所有时间框架趋势方向一致的程度）
        alignment_df = pd.DataFrame(alignment_scores).T
        alignment_score = alignment_df.apply(lambda x: len(x[x > 0]) == len(x) or len(x[x < 0]) == len(x), axis=1)
        
        return alignment_score.astype(float)
    
    def generate_multi_timeframe_signals(self, data: pd.DataFrame) -> Dict:
        """生成多时间框架信号"""
        print("🔄 生成多时间框架信号...")
        
        signals = {}
        
        # 主时间框架（15分钟）信号
        main_signals = self._generate_single_timeframe_signals(data)
        signals['main'] = main_signals
        
        # 其他时间框架信号
        for tf in ['1h', '4h', '1d']:
            tf_data = self.resample_data(data, tf)
            tf_signals = self._generate_single_timeframe_signals(tf_data)
            
            # 对齐到主时间框架
            for signal_type in tf_signals:
                aligned_signal = tf_signals[signal_type].reindex(data.index, method='ffill')
                signals[f'{tf}_{signal_type}'] = aligned_signal
        
        return signals
    
    def _generate_single_timeframe_signals(self, data: pd.DataFrame) -> Dict:
        """生成单个时间框架的信号"""
        close = data['close']
        
        # 趋势信号
        trend_strength = self.calculate_trend_strength(data)
        trend_signal = np.where(trend_strength > 25, 1, np.where(trend_strength < -25, -1, 0))
        
        # 动量信号
        momentum = self.calculate_momentum_indicators(data)
        momentum_signal = np.where(
            (momentum['rsi'] > 50) & (momentum['macd'] > momentum['macd_signal']), 1,
            np.where(
                (momentum['rsi'] < 50) & (momentum['macd'] < momentum['macd_signal']), -1, 0
            )
        )
        
        # 波动率信号
        volatility = self.calculate_volatility_indicators(data)
        vol_signal = np.where(volatility['bb_width'] > volatility['bb_width'].rolling(20).mean() * 1.2, 1, 0)
        
        # 市场结构信号
        structure = self.analyze_market_structure(data)
        structure_signal = 1 if structure['higher_highs'] and structure['higher_lows'] else -1 if structure['lower_highs'] and structure['lower_lows'] else 0
        
        return {
            'trend': pd.Series(trend_signal, index=close.index),
            'momentum': pd.Series(momentum_signal, index=close.index),
            'volatility': pd.Series(vol_signal, index=close.index),
            'structure': pd.Series(structure_signal, index=close.index)
        }
    
    def combine_multi_timeframe_signals(self, signals: Dict, weights: Optional[Dict] = None) -> Tuple[pd.Series, pd.Series]:
        """组合多时间框架信号"""
        if weights is None:
            weights = {
                'main_trend': 0.3,
                'main_momentum': 0.2,
                '1h_trend': 0.2,
                '4h_trend': 0.2,
                '1d_trend': 0.1
            }
        
        # 计算加权信号
        combined_buy = pd.Series(0, index=list(signals.values())[0].index)
        combined_sell = pd.Series(0, index=list(signals.values())[0].index)
        
        for signal_name, signal_data in signals.items():
            if isinstance(signal_data, dict):
                for sub_signal_name, sub_signal in signal_data.items():
                    weight_key = f"{signal_name}_{sub_signal_name}" if signal_name != 'main' else f"main_{sub_signal_name}"
                    weight = weights.get(weight_key, 0.1)
                    
                    if sub_signal_name in ['trend', 'momentum']:
                        combined_buy += (sub_signal > 0) * weight
                        combined_sell += (sub_signal < 0) * weight
        
        # 归一化
        total_weight = sum(weights.values())
        combined_buy = combined_buy / total_weight
        combined_sell = combined_sell / total_weight
        
        return combined_buy, combined_sell
    
    def calculate_signal_confidence(self, signals: Dict) -> pd.Series:
        """计算信号置信度"""
        # 收集所有信号
        all_signals = []
        for signal_group in signals.values():
            if isinstance(signal_group, dict):
                for signal in signal_group.values():
                    all_signals.append(signal)
            else:
                all_signals.append(signal_group)
        
        # 计算信号一致性
        signal_df = pd.DataFrame(all_signals).T
        signal_consistency = signal_df.apply(lambda x: len(x[x > 0]) / len(x) if len(x) > 0 else 0, axis=1)
        
        # 计算信号强度
        signal_strength = signal_df.abs().mean(axis=1)
        
        # 综合置信度
        confidence = signal_consistency * signal_strength
        
        return confidence.fillna(0)
    
    def analyze_timeframe_correlation(self, data: pd.DataFrame) -> Dict:
        """分析时间框架相关性"""
        correlations = {}
        
        # 计算不同时间框架的收益率
        returns_by_tf = {}
        for tf in self.timeframes:
            if tf == '15m':
                tf_data = data
            else:
                tf_data = self.resample_data(data, tf)
            
            tf_returns = tf_data['close'].pct_change()
            returns_by_tf[tf] = tf_returns
        
        # 计算相关性矩阵
        returns_df = pd.DataFrame(returns_by_tf)
        correlation_matrix = returns_df.corr()
        
        # 计算平均相关性
        avg_correlation = correlation_matrix.mean().mean()
        
        return {
            'correlation_matrix': correlation_matrix,
            'average_correlation': avg_correlation,
            'max_correlation': correlation_matrix.max().max(),
            'min_correlation': correlation_matrix.min().min()
        }

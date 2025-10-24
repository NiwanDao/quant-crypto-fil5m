"""
实时数据获取和特征计算模块
支持多种数据源和实时特征计算
"""

import pandas as pd
import numpy as np
import requests
import websocket
import threading
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
import talib
import warnings

warnings.filterwarnings('ignore')

class DataFetcher:
    """实时数据获取器"""
    
    def __init__(self, exchange_interface, config: Dict):
        self.exchange = exchange_interface
        self.config = config
        self.logger = logging.getLogger('DataFetcher')
        
        # 数据缓存
        self.data_cache = {}
        self.latest_data = {}
        
        # WebSocket连接
        self.ws_connections = {}
        self.is_running = False
        
    def get_historical_data(self, symbol: str, timeframe: str = '15m', limit: int = 1000) -> pd.DataFrame:
        """获取历史数据"""
        try:
            df = self.exchange.get_market_data(symbol, timeframe, limit)
            
            if not df.empty:
                # 计算技术指标
                df = self._calculate_technical_indicators(df)
                
                # 缓存数据
                cache_key = f"{symbol}_{timeframe}"
                self.data_cache[cache_key] = df
                
                self.logger.info(f"✅ 获取历史数据: {symbol} {timeframe} {len(df)} 条记录")
            
            return df
            
        except Exception as e:
            self.logger.error(f"❌ 获取历史数据失败: {str(e)}")
            return pd.DataFrame()
    
    def get_realtime_data(self, symbol: str, timeframe: str = '15m') -> pd.DataFrame:
        """获取实时数据（包含最新K线）"""
        try:
            # 获取最新数据
            df = self.get_historical_data(symbol, timeframe, 100)
            
            if not df.empty:
                # 更新缓存
                cache_key = f"{symbol}_{timeframe}"
                self.latest_data[cache_key] = df.iloc[-1].copy()
                
                return df
            
            return pd.DataFrame()
            
        except Exception as e:
            self.logger.error(f"❌ 获取实时数据失败: {str(e)}")
            return pd.DataFrame()
    
    def start_websocket_stream(self, symbol: str, callback: Callable):
        """启动WebSocket数据流"""
        try:
            # 这里需要根据具体交易所实现WebSocket连接
            # 示例使用币安WebSocket
            if hasattr(self.exchange, 'start_websocket'):
                self.exchange.start_websocket(symbol, callback)
            else:
                self.logger.warning("⚠️ 当前交易所不支持WebSocket")
                
        except Exception as e:
            self.logger.error(f"❌ 启动WebSocket失败: {str(e)}")
    
    def _calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算技术指标"""
        try:
            if len(df) < 20:
                return df
            
            # 基础价格数据
            high = df['high'].values
            low = df['low'].values
            close = df['close'].values
            volume = df['volume'].values
            
            # 移动平均线
            df['sma_5'] = talib.SMA(close, timeperiod=5)
            df['sma_10'] = talib.SMA(close, timeperiod=10)
            df['sma_20'] = talib.SMA(close, timeperiod=20)
            df['sma_50'] = talib.SMA(close, timeperiod=50)
            df['sma_100'] = talib.SMA(close, timeperiod=100)
            
            # 指数移动平均线
            df['ema_12'] = talib.EMA(close, timeperiod=12)
            df['ema_26'] = talib.EMA(close, timeperiod=26)
            df['ema_50'] = talib.EMA(close, timeperiod=50)
            
            # 模型需要的特征名称映射
            df['ema20'] = df['ema_12']  # 使用ema_12作为ema20的近似
            df['ema50'] = df['ema_50']
            
            # MACD
            macd, macd_signal, macd_hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
            df['macd'] = macd
            df['macd_signal'] = macd_signal
            df['macd_sig'] = macd_signal  # 模型需要的特征名
            df['macd_histogram'] = macd_hist
            
            # RSI
            df['rsi'] = talib.RSI(close, timeperiod=14)
            df['rsi_6'] = talib.RSI(close, timeperiod=6)
            df['rsi_21'] = talib.RSI(close, timeperiod=21)
            
            # 布林带
            bb_upper, bb_middle, bb_lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
            df['bb_upper'] = bb_upper
            df['bb_middle'] = bb_middle
            df['bb_lower'] = bb_lower
            df['bb_width'] = (bb_upper - bb_lower) / bb_middle
            df['bbw'] = df['bb_width']  # 模型需要的特征名
            df['bb_position'] = (close - bb_lower) / (bb_upper - bb_lower)
            
            # KDJ指标
            k, d = talib.STOCH(high, low, close, fastk_period=9, slowk_period=3, slowd_period=3)
            df['kdj_k'] = k
            df['kdj_d'] = d
            df['kdj_j'] = 3 * k - 2 * d
            df['kdj_kd_diff'] = k - d  # KDJ K-D差值
            
            # 威廉指标
            df['williams_r'] = talib.WILLR(high, low, close, timeperiod=14)
            
            # CCI指标
            df['cci'] = talib.CCI(high, low, close, timeperiod=14)
            
            # ATR指标
            df['atr'] = talib.ATR(high, low, close, timeperiod=14)
            df['atr_percent'] = df['atr'] / close
            
            # 成交量指标
            df['volume_sma'] = talib.SMA(volume, timeperiod=20)
            df['volume_ratio'] = volume / df['volume_sma']
            df['volume_rate'] = volume / talib.SMA(volume, timeperiod=5)  # 成交量变化率
            
            # OBV指标
            df['obv'] = talib.OBV(close, volume)
            df['obv_ema'] = talib.EMA(df['obv'], timeperiod=10)
            
            # VWAP (使用价格和成交量的加权平均)
            df['vwap'] = (df['close'] * df['volume']).rolling(20).sum() / df['volume'].rolling(20).sum()
            
            # VPT (Volume Price Trend)
            df['vpt'] = (df['volume'] * df['returns']).cumsum()
            
            # VWAP deviation
            df['vwap_dev'] = (df['close'] - df['vwap']) / df['vwap']
            
            # 价格位置
            df['price_position'] = (close - talib.MIN(low, 20)) / (talib.MAX(high, 20) - talib.MIN(low, 20))
            
            # 波动率
            df['volatility'] = talib.ATR(high, low, close, timeperiod=20) / close
            df['expansion'] = df['volatility']  # 模型需要的特征名
            
            # Volatility momentum
            df['vol_mom'] = df['volatility'].pct_change()
            
            # 趋势强度
            df['trend_strength'] = abs(df['sma_20'] - df['sma_50']) / df['sma_50']
            
            # 动量指标
            df['momentum'] = talib.MOM(close, timeperiod=10)
            df['roc'] = talib.ROC(close, timeperiod=10)
            
            # 模型需要的动量特征
            df['impact_mom'] = df['momentum']  # 使用momentum作为impact_mom
            df['mom3'] = talib.MOM(close, timeperiod=3)
            df['mom6'] = talib.MOM(close, timeperiod=6)
            
            # 支撑阻力
            df['support'] = talib.MIN(low, 20)
            df['resistance'] = talib.MAX(high, 20)
            
            # 缺口检测
            df['gap_up'] = (df['low'] > df['high'].shift(1)).astype(int)
            df['gap_down'] = (df['high'] < df['low'].shift(1)).astype(int)
            
            # 价格模式识别
            df['doji'] = talib.CDLDOJI(df['open'], high, low, close)
            df['hammer'] = talib.CDLHAMMER(df['open'], high, low, close)
            df['shooting_star'] = talib.CDLSHOOTINGSTAR(df['open'], high, low, close)
            df['engulfing'] = talib.CDLENGULFING(df['open'], high, low, close)
            
            # 市场状态
            df['market_regime'] = self._calculate_market_regime(df)
            
            # 添加returns特征（模型需要的ret）
            df['ret'] = df['returns'] if 'returns' in df.columns else df['close'].pct_change()
            
            # 特征工程
            df = self._create_advanced_features(df)
            
            return df
            
        except Exception as e:
            self.logger.error(f"❌ 技术指标计算失败: {str(e)}")
            return df
    
    def _calculate_market_regime(self, df: pd.DataFrame) -> pd.Series:
        """计算市场状态"""
        try:
            regime = pd.Series('neutral', index=df.index)
            
            # 趋势判断
            uptrend = (df['sma_20'] > df['sma_50']) & (df['trend_strength'] > 0.02)
            downtrend = (df['sma_20'] < df['sma_50']) & (df['trend_strength'] > 0.02)
            
            # 波动率判断
            high_vol = df['volatility'] > df['volatility'].quantile(0.8)
            low_vol = df['volatility'] < df['volatility'].quantile(0.2)
            
            # 成交量判断
            high_volume = df['volume_ratio'] > 1.5
            low_volume = df['volume_ratio'] < 0.5
            
            # 状态分类
            regime[uptrend & high_vol] = 'uptrend_high_vol'
            regime[uptrend & low_vol] = 'uptrend_low_vol'
            regime[downtrend & high_vol] = 'downtrend_high_vol'
            regime[downtrend & low_vol] = 'downtrend_low_vol'
            regime[high_vol & ~uptrend & ~downtrend] = 'sideways_high_vol'
            regime[low_vol & ~uptrend & ~downtrend] = 'sideways_low_vol'
            
            return regime
            
        except Exception as e:
            self.logger.error(f"❌ 市场状态计算失败: {str(e)}")
            return pd.Series('neutral', index=df.index)
    
    def _create_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建高级特征"""
        try:
            # 价格变化特征
            df['price_change'] = df['close'].pct_change()
            df['price_change_abs'] = df['price_change'].abs()
            df['price_change_2'] = df['close'].pct_change(2)
            df['price_change_5'] = df['close'].pct_change(5)
            
            # 成交量特征
            df['volume_change'] = df['volume'].pct_change()
            df['volume_price_trend'] = df['volume'] * df['price_change']
            
            # 技术指标组合
            df['macd_rsi'] = df['macd'] * df['rsi'] / 100
            df['bb_rsi'] = df['bb_position'] * df['rsi'] / 100
            df['trend_volume'] = df['trend_strength'] * df['volume_ratio']
            
            # 多时间框架特征
            df['sma_ratio_5_20'] = df['sma_5'] / df['sma_20']
            df['sma_ratio_20_50'] = df['sma_20'] / df['sma_50']
            df['ema_ratio_12_26'] = df['ema_12'] / df['ema_26']
            
            # 波动率特征
            df['volatility_ma'] = df['volatility'].rolling(20).mean()
            df['volatility_ratio'] = df['volatility'] / df['volatility_ma']
            
            # 动量特征
            df['momentum_ma'] = df['momentum'].rolling(10).mean()
            df['momentum_ratio'] = df['momentum'] / df['momentum_ma']
            
            # 相对强度
            df['relative_strength'] = df['close'] / df['sma_50']
            df['relative_volume'] = df['volume'] / df['volume_sma']
            
            # 价格位置特征
            df['price_bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            df['price_sma_position'] = (df['close'] - df['sma_20']) / df['sma_20']
            
            # 时间特征
            df['hour'] = df.index.hour
            df['day_of_week'] = df.index.dayofweek
            df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
            
            # 市场微观结构
            df['spread'] = df['high'] - df['low']
            df['spread_percent'] = df['spread'] / df['close']
            df['body_size'] = abs(df['close'] - df['open'])
            df['body_percent'] = df['body_size'] / df['spread']
            
            # 缺口特征
            df['gap_size'] = df['open'] - df['close'].shift(1)
            df['gap_percent'] = df['gap_size'] / df['close'].shift(1)
            
            # 支撑阻力强度
            df['support_strength'] = (df['close'] - df['support']) / df['close']
            df['resistance_strength'] = (df['resistance'] - df['close']) / df['close']
            
            return df
            
        except Exception as e:
            self.logger.error(f"❌ 高级特征创建失败: {str(e)}")
            return df
    
    def get_latest_features(self, symbol: str, timeframe: str = '15m') -> Dict:
        """获取最新特征数据"""
        try:
            cache_key = f"{symbol}_{timeframe}"
            
            if cache_key in self.latest_data:
                return self.latest_data[cache_key].to_dict()
            
            # 获取最新数据
            df = self.get_realtime_data(symbol, timeframe)
            
            if not df.empty:
                return df.iloc[-1].to_dict()
            
            return {}
            
        except Exception as e:
            self.logger.error(f"❌ 获取最新特征失败: {str(e)}")
            return {}
    
    def calculate_feature_importance(self, df: pd.DataFrame, target_col: str = 'returns') -> pd.DataFrame:
        """计算特征重要性"""
        try:
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.feature_selection import mutual_info_regression
            
            # 选择数值特征
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            feature_cols = [col for col in numeric_cols if col != target_col and not col.startswith('y')]
            
            # 准备数据
            X = df[feature_cols].fillna(0)
            y = df[target_col].fillna(0)
            
            # 随机森林特征重要性
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(X, y)
            
            # 互信息
            mi_scores = mutual_info_regression(X, y, random_state=42)
            
            # 创建重要性DataFrame
            importance_df = pd.DataFrame({
                'feature': feature_cols,
                'rf_importance': rf.feature_importances_,
                'mutual_info': mi_scores
            })
            
            importance_df = importance_df.sort_values('rf_importance', ascending=False)
            
            return importance_df
            
        except Exception as e:
            self.logger.error(f"❌ 特征重要性计算失败: {str(e)}")
            return pd.DataFrame()
    
    def detect_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """检测数据异常"""
        try:
            from sklearn.ensemble import IsolationForest
            
            # 选择关键特征
            key_features = ['close', 'volume', 'volatility', 'rsi', 'macd']
            available_features = [col for col in key_features if col in df.columns]
            
            if len(available_features) < 3:
                return df
            
            # 准备数据
            X = df[available_features].fillna(0)
            
            # 异常检测
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            anomalies = iso_forest.fit_predict(X)
            
            df['is_anomaly'] = (anomalies == -1).astype(int)
            df['anomaly_score'] = iso_forest.decision_function(X)
            
            return df
            
        except Exception as e:
            self.logger.error(f"❌ 异常检测失败: {str(e)}")
            return df
    
    def validate_data_quality(self, df: pd.DataFrame) -> Dict:
        """验证数据质量"""
        try:
            quality_report = {
                'total_records': len(df),
                'missing_values': df.isnull().sum().sum(),
                'duplicate_records': df.duplicated().sum(),
                'zero_volume_records': (df['volume'] == 0).sum(),
                'negative_price_records': (df['close'] <= 0).sum(),
                'data_completeness': (1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
            }
            
            # 数据连续性检查
            if 'timestamp' in df.columns:
                time_diff = df['timestamp'].diff().dt.total_seconds()
                expected_interval = 15 * 60  # 15分钟
                quality_report['time_gaps'] = (time_diff > expected_interval * 1.5).sum()
            
            return quality_report
            
        except Exception as e:
            self.logger.error(f"❌ 数据质量验证失败: {str(e)}")
            return {}


class RealTimeDataProcessor:
    """实时数据处理器"""
    
    def __init__(self, data_fetcher: DataFetcher):
        self.data_fetcher = data_fetcher
        self.logger = logging.getLogger('RealTimeProcessor')
        
        # 数据流
        self.data_streams = {}
        self.callbacks = {}
        
    def start_data_stream(self, symbol: str, timeframe: str, callback: Callable):
        """启动数据流"""
        try:
            stream_key = f"{symbol}_{timeframe}"
            
            # 注册回调
            self.callbacks[stream_key] = callback
            
            # 启动数据获取线程
            thread = threading.Thread(
                target=self._data_stream_worker,
                args=(symbol, timeframe),
                daemon=True
            )
            thread.start()
            
            self.logger.info(f"✅ 数据流启动: {symbol} {timeframe}")
            
        except Exception as e:
            self.logger.error(f"❌ 启动数据流失败: {str(e)}")
    
    def _data_stream_worker(self, symbol: str, timeframe: str):
        """数据流工作线程"""
        try:
            while True:
                # 获取最新数据
                df = self.data_fetcher.get_realtime_data(symbol, timeframe)
                
                if not df.empty:
                    # 调用回调函数
                    stream_key = f"{symbol}_{timeframe}"
                    if stream_key in self.callbacks:
                        self.callbacks[stream_key](df)
                
                # 等待下一个周期
                time.sleep(15 * 60)  # 15分钟
                
        except Exception as e:
            self.logger.error(f"❌ 数据流工作线程错误: {str(e)}")
    
    def stop_data_stream(self, symbol: str, timeframe: str):
        """停止数据流"""
        stream_key = f"{symbol}_{timeframe}"
        if stream_key in self.callbacks:
            del self.callbacks[stream_key]
            self.logger.info(f"✅ 数据流停止: {symbol} {timeframe}")


# 使用示例
if __name__ == '__main__':
    # 配置示例
    config = {
        'symbol': 'FIL/USDT',
        'timeframe': '15m'
    }
    
    # 创建数据获取器（需要传入exchange_interface）
    # data_fetcher = DataFetcher(exchange_interface, config)
    
    # 获取历史数据
    # df = data_fetcher.get_historical_data('FIL/USDT', '15m', 100)
    # print(f"获取到 {len(df)} 条历史数据")
    
    # 计算特征重要性
    # importance = data_fetcher.calculate_feature_importance(df)
    # print("特征重要性:", importance.head())
    
    print("数据获取模块已准备就绪")

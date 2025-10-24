"""
动态阈值调整模块
根据市场状态、波动性和信号强度动态调整交易阈值
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from datetime import datetime, timedelta
import json
import os

class DynamicThresholdManager:
    """动态阈值管理器"""
    
    def __init__(self, config_path: str = 'conf/config.yml'):
        self.config_path = config_path
        self.thresholds_history = []
        self.market_state_history = []
        
    def load_base_thresholds(self) -> Dict:
        """加载基础阈值"""
        try:
            with open('models/optimal_thresholds.json', 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            # 如果优化阈值不存在，使用默认值
            return {
                'buy_threshold': 0.60,
                'sell_threshold': 0.40
            }
    
    def assess_market_volatility(self, atr: float, price: float, lookback_periods: int = 20) -> str:
        """评估市场波动性"""
        atr_pct = (atr / price) * 100
        
        if atr_pct < 1.0:
            return "low"
        elif atr_pct < 2.5:
            return "medium"
        else:
            return "high"
    
    def assess_market_trend(self, prices: pd.Series, lookback_periods: int = 20) -> str:
        """评估市场趋势"""
        if len(prices) < lookback_periods:
            return "neutral"
        
        recent_prices = prices.tail(lookback_periods)
        price_change = (recent_prices.iloc[-1] - recent_prices.iloc[0]) / recent_prices.iloc[0]
        
        if price_change > 0.02:  # 2%以上上涨
            return "bullish"
        elif price_change < -0.02:  # 2%以上下跌
            return "bearish"
        else:
            return "neutral"
    
    def assess_market_regime(self, prices: pd.Series, volumes: pd.Series, 
                           lookback_periods: int = 50) -> str:
        """评估市场状态（趋势/震荡）"""
        if len(prices) < lookback_periods:
            return "unknown"
        
        recent_prices = prices.tail(lookback_periods)
        recent_volumes = volumes.tail(lookback_periods)
        
        # 计算价格波动性
        price_volatility = recent_prices.pct_change().std()
        
        # 计算成交量波动性
        volume_volatility = recent_volumes.pct_change().std()
        
        # 计算价格趋势强度
        price_trend = abs(recent_prices.pct_change().mean())
        
        # 判断市场状态
        if price_volatility > 0.03 and volume_volatility > 0.5:
            return "trending"  # 趋势市场
        elif price_volatility < 0.02 and volume_volatility < 0.3:
            return "ranging"   # 震荡市场
        else:
            return "mixed"     # 混合状态
    
    def calculate_adaptive_thresholds(self, 
                                   base_buy_threshold: float,
                                   base_sell_threshold: float,
                                   market_volatility: str,
                                   market_trend: str,
                                   market_regime: str,
                                   signal_strength: float,
                                   recent_performance: Optional[Dict] = None) -> Tuple[float, float]:
        """计算自适应阈值"""
        
        # 基础阈值
        buy_threshold = base_buy_threshold
        sell_threshold = base_sell_threshold
        
        # 1. 根据市场波动性调整
        volatility_adjustments = {
            "low": {"buy": -0.05, "sell": -0.05},      # 低波动时降低阈值
            "medium": {"buy": 0.0, "sell": 0.0},       # 中等波动时保持
            "high": {"buy": 0.05, "sell": 0.05}        # 高波动时提高阈值
        }
        
        vol_adj = volatility_adjustments.get(market_volatility, {"buy": 0.0, "sell": 0.0})
        buy_threshold += vol_adj["buy"]
        sell_threshold += vol_adj["sell"]
        
        # 2. 根据市场趋势调整
        trend_adjustments = {
            "bullish": {"buy": -0.03, "sell": 0.03},   # 牛市时降低买入阈值，提高卖出阈值
            "bearish": {"buy": 0.03, "sell": -0.03},  # 熊市时提高买入阈值，降低卖出阈值
            "neutral": {"buy": 0.0, "sell": 0.0}       # 中性时保持
        }
        
        trend_adj = trend_adjustments.get(market_trend, {"buy": 0.0, "sell": 0.0})
        buy_threshold += trend_adj["buy"]
        sell_threshold += trend_adj["sell"]
        
        # 3. 根据市场状态调整
        regime_adjustments = {
            "trending": {"buy": -0.02, "sell": 0.02},  # 趋势市场时更激进
            "ranging": {"buy": 0.02, "sell": -0.02},   # 震荡市场时更保守
            "mixed": {"buy": 0.0, "sell": 0.0}         # 混合状态时保持
        }
        
        regime_adj = regime_adjustments.get(market_regime, {"buy": 0.0, "sell": 0.0})
        buy_threshold += regime_adj["buy"]
        sell_threshold += regime_adj["sell"]
        
        # 4. 根据信号强度调整
        strength_factor = 0.5 + (signal_strength * 0.5)  # 0.5-1.0
        buy_threshold *= strength_factor
        sell_threshold *= strength_factor
        
        # 5. 根据近期表现调整（如果有历史数据）
        if recent_performance:
            win_rate = recent_performance.get('win_rate', 0.5)
            if win_rate > 0.6:  # 高胜率时更激进
                buy_threshold -= 0.02
                sell_threshold -= 0.02
            elif win_rate < 0.4:  # 低胜率时更保守
                buy_threshold += 0.02
                sell_threshold += 0.02
        
        # 6. 确保阈值在合理范围内
        buy_threshold = np.clip(buy_threshold, 0.3, 0.9)
        sell_threshold = np.clip(sell_threshold, 0.1, 0.7)
        
        return buy_threshold, sell_threshold
    
    def get_dynamic_thresholds(self, 
                             current_price: float,
                             atr: float,
                             prices: pd.Series,
                             volumes: pd.Series,
                             signal_strength: float,
                             recent_performance: Optional[Dict] = None) -> Dict:
        """获取动态阈值"""
        
        # 加载基础阈值
        base_thresholds = self.load_base_thresholds()
        base_buy = base_thresholds['buy_threshold']
        base_sell = base_thresholds['sell_threshold']
        
        # 评估市场状态
        market_volatility = self.assess_market_volatility(atr, current_price)
        market_trend = self.assess_market_trend(prices)
        market_regime = self.assess_market_regime(prices, volumes)
        
        # 计算自适应阈值
        buy_threshold, sell_threshold = self.calculate_adaptive_thresholds(
            base_buy, base_sell, market_volatility, market_trend, 
            market_regime, signal_strength, recent_performance
        )
        
        # 记录阈值历史
        threshold_record = {
            'timestamp': datetime.now().isoformat(),
            'buy_threshold': buy_threshold,
            'sell_threshold': sell_threshold,
            'market_volatility': market_volatility,
            'market_trend': market_trend,
            'market_regime': market_regime,
            'signal_strength': signal_strength,
            'base_buy': base_buy,
            'base_sell': base_sell
        }
        
        self.thresholds_history.append(threshold_record)
        
        # 保持历史记录在合理范围内
        if len(self.thresholds_history) > 1000:
            self.thresholds_history = self.thresholds_history[-500:]
        
        return {
            'buy_threshold': buy_threshold,
            'sell_threshold': sell_threshold,
            'market_volatility': market_volatility,
            'market_trend': market_trend,
            'market_regime': market_regime,
            'adjustments': {
                'volatility_adj': market_volatility,
                'trend_adj': market_trend,
                'regime_adj': market_regime,
                'strength_factor': signal_strength
            }
        }
    
    def get_threshold_statistics(self, lookback_days: int = 7) -> Dict:
        """获取阈值调整统计信息"""
        if not self.thresholds_history:
            return {}
        
        # 过滤最近的数据
        cutoff_date = datetime.now() - timedelta(days=lookback_days)
        recent_history = [
            record for record in self.thresholds_history
            if datetime.fromisoformat(record['timestamp']) > cutoff_date
        ]
        
        if not recent_history:
            return {}
        
        buy_thresholds = [r['buy_threshold'] for r in recent_history]
        sell_thresholds = [r['sell_threshold'] for r in recent_history]
        
        return {
            'buy_threshold_stats': {
                'mean': np.mean(buy_thresholds),
                'std': np.std(buy_thresholds),
                'min': np.min(buy_thresholds),
                'max': np.max(buy_thresholds)
            },
            'sell_threshold_stats': {
                'mean': np.mean(sell_thresholds),
                'std': np.std(sell_thresholds),
                'min': np.min(sell_thresholds),
                'max': np.max(sell_thresholds)
            },
            'market_volatility_dist': {
                vol: sum(1 for r in recent_history if r['market_volatility'] == vol)
                for vol in ['low', 'medium', 'high']
            },
            'market_trend_dist': {
                trend: sum(1 for r in recent_history if r['market_trend'] == trend)
                for trend in ['bullish', 'bearish', 'neutral']
            },
            'market_regime_dist': {
                regime: sum(1 for r in recent_history if r['market_regime'] == regime)
                for regime in ['trending', 'ranging', 'mixed']
            }
        }
    
    def save_threshold_history(self, filepath: str = 'models/threshold_history.json'):
        """保存阈值历史"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.thresholds_history, f, indent=2)
    
    def load_threshold_history(self, filepath: str = 'models/threshold_history.json'):
        """加载阈值历史"""
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                self.thresholds_history = json.load(f)


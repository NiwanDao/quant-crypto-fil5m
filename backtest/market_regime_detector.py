"""
市场状态识别和适应性策略模块
包含市场状态分类、策略切换、环境适应性等功能
"""

import pandas as pd
import numpy as np
import talib
from typing import Dict, List, Tuple, Optional
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

class MarketRegimeDetector:
    """市场状态识别器"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.regime_models = {}
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=3)
        
    def calculate_market_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算市场特征"""
        print("🔍 计算市场特征...")
        
        close = data['close']
        high = data['high']
        low = data['low']
        volume = data['volume']
        
        features = pd.DataFrame(index=close.index)
        
        # 价格特征
        features['returns'] = close.pct_change()
        features['volatility'] = features['returns'].rolling(20).std()
        features['price_momentum'] = close.pct_change(20)
        
        # 技术指标
        features['rsi'] = pd.Series(talib.RSI(close.values, timeperiod=14), index=close.index)
        features['macd'] = pd.Series(talib.MACD(close.values)[0], index=close.index)
        features['bb_width'] = self._calculate_bb_width(close)
        features['atr'] = pd.Series(talib.ATR(high.values, low.values, close.values), index=close.index)
        
        # 趋势特征
        features['trend_strength'] = self._calculate_trend_strength(close)
        features['trend_direction'] = np.where(close > close.rolling(20).mean(), 1, -1)
        
        # 成交量特征
        features['volume_ratio'] = volume / volume.rolling(20).mean()
        features['price_volume_trend'] = (close.pct_change() * volume).rolling(10).sum()
        
        # 波动率特征
        features['volatility_ratio'] = features['volatility'] / features['volatility'].rolling(50).mean()
        features['volatility_regime'] = self._classify_volatility_regime(features['volatility'])
        
        # 市场结构特征
        features['market_structure'] = self._analyze_market_structure(high, low, close)
        
        # 流动性特征
        features['liquidity'] = self._calculate_liquidity(volume, close)
        
        return features.dropna()
    
    def _calculate_bb_width(self, close: pd.Series) -> pd.Series:
        """计算布林带宽度"""
        bb_upper, bb_middle, bb_lower = talib.BBANDS(close.values)
        bb_width = (bb_upper - bb_lower) / bb_middle
        return pd.Series(bb_width, index=close.index)
    
    def _calculate_trend_strength(self, close: pd.Series) -> pd.Series:
        """计算趋势强度"""
        # ADX指标
        high = close.rolling(14).max()
        low = close.rolling(14).min()
        adx = talib.ADX(high.values, low.values, close.values, timeperiod=14)
        return pd.Series(adx, index=close.index)
    
    def _classify_volatility_regime(self, volatility: pd.Series) -> pd.Series:
        """分类波动率状态"""
        # 使用分位数分类
        low_threshold = volatility.rolling(100).quantile(0.33)
        high_threshold = volatility.rolling(100).quantile(0.67)
        
        regime = pd.Series('medium', index=volatility.index)
        regime[volatility < low_threshold] = 'low'
        regime[volatility > high_threshold] = 'high'
        
        return regime
    
    def _analyze_market_structure(self, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """分析市场结构"""
        # 识别高低点
        highs = high.rolling(5, center=True).max() == high
        lows = low.rolling(5, center=True).min() == low
        
        # 计算结构强度
        structure_strength = pd.Series(0, index=close.index)
        
        for i in range(10, len(close)):
            recent_highs = high[highs].iloc[max(0, i-20):i]
            recent_lows = low[lows].iloc[max(0, i-20):i]
            
            if len(recent_highs) >= 2 and len(recent_lows) >= 2:
                # 计算高低点的一致性
                high_trend = 1 if recent_highs.iloc[-1] > recent_highs.iloc[-2] else -1
                low_trend = 1 if recent_lows.iloc[-1] > recent_lows.iloc[-2] else -1
                
                structure_strength.iloc[i] = (high_trend + low_trend) / 2
        
        return structure_strength
    
    def _calculate_liquidity(self, volume: pd.Series, close: pd.Series) -> pd.Series:
        """计算流动性指标"""
        # 基于成交量的流动性
        volume_ma = volume.rolling(20).mean()
        liquidity = volume / volume_ma
        
        # 价格冲击估计
        price_impact = 1 / liquidity
        
        return price_impact
    
    def detect_market_regimes(self, features: pd.DataFrame, n_regimes: int = 4) -> pd.Series:
        """检测市场状态"""
        print(f"🎯 检测市场状态（{n_regimes}个状态）...")
        
        # 选择用于聚类的特征
        clustering_features = [
            'volatility', 'trend_strength', 'rsi', 'bb_width', 
            'volume_ratio', 'volatility_ratio', 'liquidity'
        ]
        
        # 标准化特征
        X = features[clustering_features].fillna(0)
        X_scaled = self.scaler.fit_transform(X)
        
        # 降维
        X_pca = self.pca.fit_transform(X_scaled)
        
        # K-means聚类
        kmeans = KMeans(n_clusters=n_regimes, random_state=42, n_init=10)
        regime_labels = kmeans.fit_predict(X_pca)
        
        # 创建状态标签
        regime_mapping = {
            0: 'trending_high_vol',
            1: 'trending_low_vol', 
            2: 'sideways_high_vol',
            3: 'sideways_low_vol'
        }
        
        regimes = pd.Series([regime_mapping.get(label, 'unknown') for label in regime_labels], 
                           index=features.index)
        
        return regimes
    
    def calculate_regime_transition_matrix(self, regimes: pd.Series) -> pd.DataFrame:
        """计算状态转移矩阵"""
        regime_changes = regimes != regimes.shift(1)
        transitions = []
        
        for i in range(1, len(regimes)):
            if regime_changes.iloc[i]:
                from_regime = regimes.iloc[i-1]
                to_regime = regimes.iloc[i]
                transitions.append((from_regime, to_regime))
        
        # 创建转移矩阵
        unique_regimes = regimes.unique()
        transition_matrix = pd.DataFrame(0, index=unique_regimes, columns=unique_regimes)
        
        for from_regime, to_regime in transitions:
            transition_matrix.loc[from_regime, to_regime] += 1
        
        # 转换为概率
        transition_matrix = transition_matrix.div(transition_matrix.sum(axis=1), axis=0)
        
        return transition_matrix.fillna(0)
    
    def calculate_regime_performance(self, regimes: pd.Series, returns: pd.Series) -> Dict:
        """计算各状态下的表现"""
        regime_performance = {}
        
        for regime in regimes.unique():
            mask = regimes == regime
            regime_returns = returns[mask]
            
            if len(regime_returns) > 0:
                regime_performance[regime] = {
                    'count': len(regime_returns),
                    'mean_return': regime_returns.mean(),
                    'volatility': regime_returns.std(),
                    'sharpe_ratio': regime_returns.mean() / regime_returns.std() if regime_returns.std() > 0 else 0,
                    'max_return': regime_returns.max(),
                    'min_return': regime_returns.min(),
                    'positive_ratio': (regime_returns > 0).mean()
                }
        
        return regime_performance
    
    def generate_regime_adaptive_signals(self, data: pd.DataFrame, regimes: pd.Series) -> Dict:
        """生成状态自适应信号"""
        print("🔄 生成状态自适应信号...")
        
        signals = {}
        
        for regime in regimes.unique():
            mask = regimes == regime
            regime_data = data[mask]
            
            if len(regime_data) == 0:
                continue
            
            # 根据状态调整策略参数
            if 'trending' in regime:
                signals[regime] = self._generate_trend_following_signals(regime_data)
            elif 'sideways' in regime:
                signals[regime] = self._generate_mean_reversion_signals(regime_data)
            else:
                signals[regime] = self._generate_neutral_signals(regime_data)
        
        return signals
    
    def _generate_trend_following_signals(self, data: pd.DataFrame) -> Dict:
        """生成趋势跟踪信号"""
        close = data['close']
        
        # 移动平均线信号
        sma_short = close.rolling(10).mean()
        sma_long = close.rolling(20).mean()
        ma_signal = np.where(sma_short > sma_long, 1, -1)
        
        # MACD信号
        macd, macd_signal, macd_hist = talib.MACD(close.values)
        macd_signal_vals = np.where(macd > macd_signal, 1, -1)
        
        # 趋势强度信号
        trend_strength = self._calculate_trend_strength(close)
        trend_signal = np.where(trend_strength > 25, 1, np.where(trend_strength < -25, -1, 0))
        
        return {
            'ma_signal': pd.Series(ma_signal, index=close.index),
            'macd_signal': pd.Series(macd_signal_vals, index=close.index),
            'trend_signal': trend_signal
        }
    
    def _generate_mean_reversion_signals(self, data: pd.DataFrame) -> Dict:
        """生成均值回归信号"""
        close = data['close']
        
        # RSI信号
        rsi = talib.RSI(close.values, timeperiod=14)
        rsi_signal = np.where(rsi < 30, 1, np.where(rsi > 70, -1, 0))
        
        # 布林带信号
        bb_upper, bb_middle, bb_lower = talib.BBANDS(close.values)
        bb_signal = np.where(close < bb_lower, 1, np.where(close > bb_upper, -1, 0))
        
        # 价格偏离信号
        price_deviation = (close - close.rolling(20).mean()) / close.rolling(20).std()
        deviation_signal = np.where(price_deviation < -2, 1, np.where(price_deviation > 2, -1, 0))
        
        return {
            'rsi_signal': pd.Series(rsi_signal, index=close.index),
            'bb_signal': pd.Series(bb_signal, index=close.index),
            'deviation_signal': pd.Series(deviation_signal, index=close.index)
        }
    
    def _generate_neutral_signals(self, data: pd.DataFrame) -> Dict:
        """生成中性信号"""
        close = data['close']
        
        # 低波动率环境下的保守信号
        volatility = close.pct_change().rolling(20).std()
        low_vol_mask = volatility < volatility.quantile(0.3)
        
        # 只在低波动率时给出信号
        neutral_signal = np.where(low_vol_mask, 0, 0)
        
        return {
            'neutral_signal': pd.Series(neutral_signal, index=close.index)
        }
    
    def calculate_regime_stability(self, regimes: pd.Series, window: int = 50) -> pd.Series:
        """计算状态稳定性"""
        # 计算滚动窗口内的状态一致性
        stability = regimes.rolling(window).apply(lambda x: len(x.unique()) == 1, raw=False)
        
        return stability.fillna(0)
    
    def predict_next_regime(self, current_regime: str, transition_matrix: pd.DataFrame) -> Dict:
        """预测下一个状态"""
        if current_regime not in transition_matrix.index:
            return {'predicted_regime': 'unknown', 'probability': 0}
        
        # 获取转移概率
        probabilities = transition_matrix.loc[current_regime]
        predicted_regime = probabilities.idxmax()
        probability = probabilities.max()
        
        return {
            'predicted_regime': predicted_regime,
            'probability': probability,
            'all_probabilities': probabilities.to_dict()
        }
    
    def optimize_regime_parameters(self, regimes: pd.Series, returns: pd.Series) -> Dict:
        """优化各状态下的参数"""
        print("⚙️ 优化状态参数...")
        
        optimized_params = {}
        
        for regime in regimes.unique():
            mask = regimes == regime
            regime_returns = returns[mask]
            
            if len(regime_returns) < 30:
                continue
            
            # 计算最优参数
            volatility = regime_returns.std()
            mean_return = regime_returns.mean()
            
            # 基于夏普比率优化
            sharpe_ratio = mean_return / volatility if volatility > 0 else 0
            
            # 风险调整
            risk_multiplier = min(2.0, max(0.5, 1 / volatility)) if volatility > 0 else 1.0
            
            optimized_params[regime] = {
                'risk_multiplier': risk_multiplier,
                'position_size': min(1.0, sharpe_ratio * 0.1),
                'stop_loss': volatility * 2,
                'take_profit': volatility * 3,
                'sharpe_ratio': sharpe_ratio
            }
        
        return optimized_params
    
    def create_regime_visualization(self, data: pd.DataFrame, regimes: pd.Series, 
                                  features: pd.DataFrame) -> None:
        """创建状态可视化"""
        print("📊 创建状态可视化...")
        
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        fig, axes = plt.subplots(3, 2, figsize=(20, 15))
        
        # 1. 价格和状态
        ax1 = axes[0, 0]
        close = data['close']
        ax1.plot(close.index, close.values, alpha=0.7, linewidth=1)
        
        # 为每个状态着色
        regime_colors = {
            'trending_high_vol': 'red',
            'trending_low_vol': 'green',
            'sideways_high_vol': 'orange',
            'sideways_low_vol': 'blue'
        }
        
        for regime in regimes.unique():
            mask = regimes == regime
            if mask.any():
                ax1.scatter(close.index[mask], close.values[mask], 
                           c=regime_colors.get(regime, 'gray'), 
                           label=regime, alpha=0.6, s=10)
        
        ax1.set_title('价格走势与市场状态')
        ax1.set_ylabel('价格')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 状态分布
        ax2 = axes[0, 1]
        regime_counts = regimes.value_counts()
        ax2.pie(regime_counts.values, labels=regime_counts.index, autopct='%1.1f%%')
        ax2.set_title('状态分布')
        
        # 3. 波动率分析
        ax3 = axes[1, 0]
        volatility = features['volatility']
        for regime in regimes.unique():
            mask = regimes == regime
            if mask.any():
                ax3.hist(volatility[mask], alpha=0.6, label=regime, bins=20)
        ax3.set_title('各状态波动率分布')
        ax3.set_xlabel('波动率')
        ax3.set_ylabel('频次')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. 趋势强度分析
        ax4 = axes[1, 1]
        trend_strength = features['trend_strength']
        for regime in regimes.unique():
            mask = regimes == regime
            if mask.any():
                ax4.hist(trend_strength[mask], alpha=0.6, label=regime, bins=20)
        ax4.set_title('各状态趋势强度分布')
        ax4.set_xlabel('趋势强度')
        ax4.set_ylabel('频次')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. 状态转移热力图
        ax5 = axes[2, 0]
        transition_matrix = self.calculate_regime_transition_matrix(regimes)
        sns.heatmap(transition_matrix, annot=True, cmap='Blues', ax=ax5)
        ax5.set_title('状态转移矩阵')
        
        # 6. 时间序列状态
        ax6 = axes[2, 1]
        regime_numeric = pd.Categorical(regimes).codes
        ax6.plot(close.index, regime_numeric, marker='o', markersize=2, alpha=0.7)
        ax6.set_title('状态时间序列')
        ax6.set_ylabel('状态编号')
        ax6.set_xlabel('时间')
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('backtest/market_regime_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print("✅ 状态可视化完成")

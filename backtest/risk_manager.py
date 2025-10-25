"""
高级风险管理模块
包含动态止损、仓位管理、风险预算等功能
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
import talib
from scipy import stats

class AdvancedRiskManager:
    """高级风险管理器"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.risk_params = config.get('risk', {})
        
    def calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, 
                     period: int = 14) -> pd.Series:
        """计算平均真实波幅"""
        return pd.Series(talib.ATR(high.values, low.values, close.values, timeperiod=period),
                        index=close.index)
    
    def calculate_dynamic_stop_loss(self, entry_price: float, atr: float, 
                                  trend_direction: str, volatility_regime: str) -> float:
        """动态止损计算"""
        # 基础止损倍数
        base_multiplier = self.risk_params.get('atr_stop_mult', 1.5)
        
        # 根据趋势调整
        if trend_direction == 'uptrend':
            multiplier = base_multiplier * 0.8  # 趋势中放宽止损
        elif trend_direction == 'downtrend':
            multiplier = base_multiplier * 1.2  # 下跌中收紧止损
        else:
            multiplier = base_multiplier
        
        # 根据波动率调整
        if volatility_regime == 'high_volatility':
            multiplier *= 1.5
        elif volatility_regime == 'low_volatility':
            multiplier *= 0.8
        
        return entry_price - (atr * multiplier)
    
    def calculate_position_size_kelly(self, win_rate: float, avg_win: float, 
                                    avg_loss: float, current_equity: float) -> float:
        """凯利公式计算最优仓位"""
        if avg_loss == 0:
            return 0
        
        # 凯利公式: f = (bp - q) / b
        # b = 平均盈利/平均亏损, p = 胜率, q = 1-胜率
        b = abs(avg_win / avg_loss)
        p = win_rate
        q = 1 - win_rate
        
        kelly_fraction = (b * p - q) / b
        
        # 限制凯利比例，避免过度杠杆
        kelly_fraction = max(0, min(kelly_fraction, 0.25))  # 最大25%
        
        return current_equity * kelly_fraction
    
    def calculate_position_size_volatility_target(self, target_volatility: float, 
                                                current_volatility: float, 
                                                current_equity: float) -> float:
        """基于波动率目标的仓位计算"""
        if current_volatility == 0:
            return 0
        
        # 波动率调整系数
        vol_adjustment = target_volatility / current_volatility
        
        # 基础仓位
        base_position = current_equity * 0.1  # 10%基础仓位
        
        return base_position * vol_adjustment
    
    def calculate_correlation_risk(self, returns: pd.Series, 
                                 market_returns: pd.Series) -> float:
        """计算相关性风险"""
        if len(returns) < 30 or len(market_returns) < 30:
            return 0
        
        # 计算滚动相关性
        correlation = returns.rolling(30).corr(market_returns).iloc[-1]
        
        if pd.isna(correlation):
            return 0
        
        # 高相关性增加风险
        return abs(correlation)
    
    def calculate_liquidity_risk(self, volume: pd.Series, price: pd.Series) -> pd.Series:
        """计算流动性风险"""
        # 基于成交量的流动性指标
        volume_ma = volume.rolling(20).mean()
        current_volume = volume
        
        # 流动性比率
        liquidity_ratio = current_volume / volume_ma
        
        # 价格冲击估计（简化）
        price_impact = 1 / liquidity_ratio
        
        return price_impact
    
    def calculate_market_stress(self, returns: pd.Series, volatility: pd.Series) -> pd.Series:
        """计算市场压力指标"""
        # VIX类似指标
        stress_indicator = volatility * abs(returns)
        
        # 滚动标准化
        stress_normalized = (stress_indicator - stress_indicator.rolling(50).mean()) / stress_indicator.rolling(50).std()
        
        return stress_normalized.fillna(0)
    
    def calculate_portfolio_var(self, returns: pd.Series, confidence_level: float = 0.95) -> float:
        """计算投资组合VaR"""
        if len(returns) < 30:
            return 0
        
        # 历史模拟法
        var_percentile = (1 - confidence_level) * 100
        var = np.percentile(returns, var_percentile)
        
        return abs(var)
    
    def calculate_expected_shortfall(self, returns: pd.Series, confidence_level: float = 0.95) -> float:
        """计算期望损失（CVaR）"""
        if len(returns) < 30:
            return 0
        
        var_percentile = (1 - confidence_level) * 100
        var_threshold = np.percentile(returns, var_percentile)
        
        # 计算超过VaR的损失期望
        tail_losses = returns[returns <= var_threshold]
        
        if len(tail_losses) == 0:
            return 0
        
        return abs(tail_losses.mean())
    
    def calculate_maximum_adverse_excursion(self, prices: pd.Series, 
                                         entry_prices: pd.Series) -> pd.Series:
        """计算最大不利偏移（MAE）"""
        mae = pd.Series(index=prices.index, dtype=float)
        
        for i, (timestamp, price) in enumerate(prices.items()):
            if i == 0:
                mae.iloc[i] = 0
                continue
            
            # 找到最近的入场价格
            recent_entries = entry_prices[entry_prices.index <= timestamp]
            if len(recent_entries) == 0:
                mae.iloc[i] = 0
                continue
            
            last_entry = recent_entries.iloc[-1]
            
            # 计算从入场到当前的最大不利偏移
            price_changes = prices.iloc[:i+1] - last_entry
            mae.iloc[i] = price_changes.min()
        
        return mae
    
    def calculate_maximum_favorable_excursion(self, prices: pd.Series, 
                                            entry_prices: pd.Series) -> pd.Series:
        """计算最大有利偏移（MFE）"""
        mfe = pd.Series(index=prices.index, dtype=float)
        
        for i, (timestamp, price) in enumerate(prices.items()):
            if i == 0:
                mfe.iloc[i] = 0
                continue
            
            # 找到最近的入场价格
            recent_entries = entry_prices[entry_prices.index <= timestamp]
            if len(recent_entries) == 0:
                mfe.iloc[i] = 0
                continue
            
            last_entry = recent_entries.iloc[-1]
            
            # 计算从入场到当前的最大有利偏移
            price_changes = prices.iloc[:i+1] - last_entry
            mfe.iloc[i] = price_changes.max()
        
        return mfe
    
    def calculate_risk_metrics(self, data: pd.DataFrame, returns: pd.Series) -> Dict:
        """计算综合风险指标"""
        print("📊 计算高级风险指标...")
        
        # 基础风险指标
        volatility = returns.std() * np.sqrt(252 * 24 * 4)  # 年化波动率
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252 * 24 * 4) if returns.std() > 0 else 0
        
        # VaR和CVaR
        var_95 = self.calculate_portfolio_var(returns, 0.95)
        var_99 = self.calculate_portfolio_var(returns, 0.99)
        cvar_95 = self.calculate_expected_shortfall(returns, 0.95)
        
        # 最大回撤
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # 偏度和峰度
        skewness = returns.skew()
        kurtosis = returns.kurtosis()
        
        # 市场压力指标
        market_stress = self.calculate_market_stress(returns, returns.rolling(20).std())
        
        # 流动性风险
        if 'volume' in data.columns:
            liquidity_risk = self.calculate_liquidity_risk(data['volume'], data['close'])
        else:
            liquidity_risk = pd.Series(0, index=data.index)
        
        return {
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'var_95': var_95,
            'var_99': var_99,
            'cvar_95': cvar_95,
            'max_drawdown': max_drawdown,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'market_stress': market_stress,
            'liquidity_risk': liquidity_risk
        }
    
    def generate_risk_adjusted_signals(self, signals: pd.Series, risk_metrics: Dict) -> pd.Series:
        """生成风险调整后的信号"""
        # 基于市场压力的信号过滤
        market_stress = risk_metrics['market_stress']
        stress_threshold = 2.0  # 2倍标准差
        
        # 高压力时期减少信号
        adjusted_signals = signals.copy()
        high_stress_mask = market_stress > stress_threshold
        adjusted_signals[high_stress_mask] = False
        
        # 基于流动性的信号调整
        liquidity_risk = risk_metrics['liquidity_risk']
        low_liquidity_mask = liquidity_risk > liquidity_risk.quantile(0.8)
        adjusted_signals[low_liquidity_mask] = False
        
        return adjusted_signals
    
    def calculate_dynamic_position_sizing(self, signals: pd.Series, risk_metrics: Dict, 
                                        current_equity: float) -> pd.Series:
        """动态仓位计算"""
        # 基础仓位
        base_size = self.config['backtest']['fixed_cash_per_trade']
        
        # 风险调整
        volatility = risk_metrics['volatility']
        max_risk_per_trade = self.risk_params.get('max_risk_per_trade', 0.005)
        
        # 基于波动率的调整
        vol_adjustment = min(1.0, max_risk_per_trade / volatility)
        
        # 基于市场压力的调整
        market_stress = risk_metrics['market_stress']
        stress_adjustment = np.where(market_stress > 1.0, 0.5, 1.0)
        
        # 基于流动性的调整
        liquidity_risk = risk_metrics['liquidity_risk']
        liquidity_adjustment = np.where(liquidity_risk > liquidity_risk.quantile(0.7), 0.7, 1.0)
        
        # 综合调整
        position_size = base_size * vol_adjustment * stress_adjustment * liquidity_adjustment
        
        return pd.Series(position_size, index=signals.index)

"""
风险管理和仓位控制模块
实现多层次风险管理策略
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import json

class RiskLevel(Enum):
    """风险等级"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class PositionSizeMethod(Enum):
    """仓位计算方法"""
    FIXED = "fixed"
    PERCENTAGE = "percentage"
    VOLATILITY = "volatility"
    KELLY = "kelly"
    RISK_PARITY = "risk_parity"

@dataclass
class RiskMetrics:
    """风险指标"""
    portfolio_value: float
    max_drawdown: float
    current_drawdown: float
    volatility: float
    sharpe_ratio: float
    var_95: float
    var_99: float
    max_loss: float
    win_rate: float
    profit_factor: float
    risk_score: float

@dataclass
class PositionRisk:
    """持仓风险"""
    symbol: str
    position_size: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    risk_amount: float
    stop_loss: float
    take_profit: float
    risk_reward_ratio: float

class RiskManager:
    """风险管理器"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger('RiskManager')
        
        # 风险参数
        self.max_portfolio_risk = config.get('max_portfolio_risk', 0.02)  # 最大投资组合风险2%
        self.max_position_risk = config.get('max_position_risk', 0.005)  # 最大单笔风险0.5%
        self.max_drawdown_limit = config.get('max_drawdown_limit', 0.15)  # 最大回撤限制15%
        self.volatility_limit = config.get('volatility_limit', 0.3)  # 波动率限制30%
        
        # 仓位管理参数
        self.position_size_method = PositionSizeMethod(config.get('position_size_method', 'volatility'))
        self.base_position_size = config.get('base_position_size', 0.1)  # 基础仓位10%
        self.max_position_size = config.get('max_position_size', 0.2)  # 最大仓位20%
        
        # 止损止盈参数
        self.stop_loss_atr_mult = config.get('stop_loss_atr_mult', 2.0)
        self.take_profit_atr_mult = config.get('take_profit_atr_mult', 3.0)
        self.trailing_stop = config.get('trailing_stop', True)
        
        # 风险历史
        self.risk_history = []
        self.portfolio_history = []
        
    def calculate_portfolio_risk(self, positions: Dict, current_prices: Dict, 
                                portfolio_value: float) -> RiskMetrics:
        """计算投资组合风险指标"""
        try:
            # 基础指标
            total_exposure = sum(pos['amount'] * current_prices.get(symbol, 0) 
                               for symbol, pos in positions.items())
            
            # 计算收益率历史（简化版）
            if len(self.portfolio_history) > 1:
                returns = pd.Series(self.portfolio_history).pct_change().dropna()
                
                # 波动率
                volatility = returns.std() * np.sqrt(252 * 24 * 4)  # 年化
                
                # 夏普比率
                sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252 * 24 * 4) if returns.std() > 0 else 0
                
                # VaR计算
                var_95 = np.percentile(returns, 5)
                var_99 = np.percentile(returns, 1)
                
                # 最大回撤
                cumulative = (1 + returns).cumprod()
                running_max = cumulative.expanding().max()
                drawdown = (cumulative - running_max) / running_max
                max_drawdown = drawdown.min()
                current_drawdown = drawdown.iloc[-1] if len(drawdown) > 0 else 0
                
            else:
                volatility = 0.0
                sharpe_ratio = 0.0
                var_95 = 0.0
                var_99 = 0.0
                max_drawdown = 0.0
                current_drawdown = 0.0
            
            # 计算风险评分
            risk_score = self._calculate_risk_score(
                current_drawdown, volatility, total_exposure, portfolio_value
            )
            
            return RiskMetrics(
                portfolio_value=portfolio_value,
                max_drawdown=max_drawdown,
                current_drawdown=current_drawdown,
                volatility=volatility,
                sharpe_ratio=sharpe_ratio,
                var_95=var_95,
                var_99=var_99,
                max_loss=0.0,  # 需要历史数据计算
                win_rate=0.0,  # 需要历史数据计算
                profit_factor=0.0,  # 需要历史数据计算
                risk_score=risk_score
            )
            
        except Exception as e:
            self.logger.error(f"❌ 投资组合风险计算失败: {str(e)}")
            return RiskMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0)
    
    def _calculate_risk_score(self, drawdown: float, volatility: float, 
                             exposure: float, portfolio_value: float) -> float:
        """计算风险评分 (0-1, 1为最高风险)"""
        try:
            # 回撤风险
            drawdown_risk = min(abs(drawdown) / self.max_drawdown_limit, 1.0)
            
            # 波动率风险
            volatility_risk = min(volatility / self.volatility_limit, 1.0)
            
            # 杠杆风险
            leverage_risk = min(exposure / portfolio_value, 1.0) if portfolio_value > 0 else 0
            
            # 综合风险评分
            risk_score = (drawdown_risk * 0.4 + volatility_risk * 0.3 + leverage_risk * 0.3)
            
            return min(risk_score, 1.0)
            
        except Exception as e:
            self.logger.error(f"❌ 风险评分计算失败: {str(e)}")
            return 1.0
    
    def calculate_position_size(self, symbol: str, signal_strength: float, 
                              current_price: float, atr: float, 
                              portfolio_value: float, available_cash: float) -> float:
        """计算仓位大小"""
        try:
            if self.position_size_method == PositionSizeMethod.FIXED:
                return self._calculate_fixed_position_size(available_cash, current_price)
            
            elif self.position_size_method == PositionSizeMethod.PERCENTAGE:
                return self._calculate_percentage_position_size(
                    portfolio_value, current_price, signal_strength
                )
            
            elif self.position_size_method == PositionSizeMethod.VOLATILITY:
                return self._calculate_volatility_position_size(
                    atr, current_price, portfolio_value, signal_strength
                )
            
            elif self.position_size_method == PositionSizeMethod.KELLY:
                return self._calculate_kelly_position_size(
                    signal_strength, portfolio_value, current_price
                )
            
            elif self.position_size_method == PositionSizeMethod.RISK_PARITY:
                return self._calculate_risk_parity_position_size(
                    atr, current_price, portfolio_value
                )
            
            else:
                return self._calculate_fixed_position_size(available_cash, current_price)
                
        except Exception as e:
            self.logger.error(f"❌ 仓位计算失败: {str(e)}")
            return 0.0
    
    def _calculate_fixed_position_size(self, available_cash: float, current_price: float) -> float:
        """固定金额仓位计算"""
        fixed_amount = self.config.get('fixed_cash_per_trade', 1000)
        position_size = min(fixed_amount, available_cash) / current_price
        return min(position_size, available_cash / current_price)
    
    def _calculate_percentage_position_size(self, portfolio_value: float, 
                                          current_price: float, signal_strength: float) -> float:
        """百分比仓位计算"""
        base_percentage = self.base_position_size
        strength_multiplier = np.clip(signal_strength, 0.5, 1.5)
        
        position_percentage = base_percentage * strength_multiplier
        position_percentage = min(position_percentage, self.max_position_size)
        
        position_value = portfolio_value * position_percentage
        return position_value / current_price
    
    def _calculate_volatility_position_size(self, atr: float, current_price: float,
                                         portfolio_value: float, signal_strength: float) -> float:
        """基于波动率的仓位计算"""
        # 目标风险金额
        target_risk = portfolio_value * self.max_position_risk
        
        # 基于ATR的止损距离
        stop_distance = atr * self.stop_loss_atr_mult
        
        # 计算仓位大小
        position_size = target_risk / stop_distance
        
        # 应用信号强度调整
        position_size *= np.clip(signal_strength, 0.5, 1.5)
        
        # 限制最大仓位
        max_position_value = portfolio_value * self.max_position_size
        max_position_size = max_position_value / current_price
        
        return min(position_size, max_position_size)
    
    def _calculate_kelly_position_size(self, signal_strength: float, 
                                    portfolio_value: float, current_price: float) -> float:
        """凯利公式仓位计算"""
        # 简化的凯利公式
        # f = (bp - q) / b
        # 其中 b = 赔率, p = 胜率, q = 败率
        
        # 使用信号强度作为胜率估计
        win_rate = signal_strength
        loss_rate = 1 - win_rate
        
        # 假设赔率为2:1
        odds = 2.0
        
        # 凯利比例
        kelly_fraction = (odds * win_rate - loss_rate) / odds
        kelly_fraction = max(0, min(kelly_fraction, 0.25))  # 限制在0-25%
        
        position_value = portfolio_value * kelly_fraction
        return position_value / current_price
    
    def _calculate_risk_parity_position_size(self, atr: float, current_price: float,
                                          portfolio_value: float) -> float:
        """风险平价仓位计算"""
        # 目标风险贡献
        target_risk_contribution = portfolio_value * self.max_position_risk
        
        # 基于ATR的风险度量
        position_risk = atr / current_price
        
        # 计算仓位大小
        position_size = target_risk_contribution / position_risk
        
        return min(position_size, portfolio_value * self.max_position_size / current_price)
    
    def calculate_stop_loss_take_profit(self, entry_price: float, atr: float, 
                                      side: str) -> Tuple[float, float]:
        """计算止损止盈价格"""
        try:
            if side.lower() == 'buy':
                # 多头止损止盈
                stop_loss = entry_price - (atr * self.stop_loss_atr_mult)
                take_profit = entry_price + (atr * self.take_profit_atr_mult)
            else:
                # 空头止损止盈
                stop_loss = entry_price + (atr * self.stop_loss_atr_mult)
                take_profit = entry_price - (atr * self.take_profit_atr_mult)
            
            return stop_loss, take_profit
            
        except Exception as e:
            self.logger.error(f"❌ 止损止盈计算失败: {str(e)}")
            return entry_price, entry_price
    
    def check_risk_limits(self, risk_metrics: RiskMetrics, 
                         new_position_risk: float) -> Tuple[bool, List[str]]:
        """检查风险限制"""
        violations = []
        
        # 检查最大回撤
        if risk_metrics.current_drawdown < -self.max_drawdown_limit:
            violations.append(f"回撤超限: {risk_metrics.current_drawdown:.2%} > {self.max_drawdown_limit:.2%}")
        
        # 检查波动率
        if risk_metrics.volatility > self.volatility_limit:
            violations.append(f"波动率超限: {risk_metrics.volatility:.2%} > {self.volatility_limit:.2%}")
        
        # 检查单笔风险
        if new_position_risk > self.max_position_risk:
            violations.append(f"单笔风险超限: {new_position_risk:.2%} > {self.max_position_risk:.2%}")
        
        # 检查投资组合风险
        if risk_metrics.risk_score > 0.8:
            violations.append(f"投资组合风险过高: {risk_metrics.risk_score:.2f}")
        
        return len(violations) == 0, violations
    
    def update_trailing_stop(self, position: Dict, current_price: float, 
                           atr: float) -> float:
        """更新跟踪止损"""
        try:
            if not self.trailing_stop:
                return position.get('stop_loss', 0)
            
            entry_price = position.get('entry_price', 0)
            current_stop = position.get('stop_loss', 0)
            side = position.get('side', 'buy')
            
            if side.lower() == 'buy':
                # 多头跟踪止损
                new_stop = current_price - (atr * self.stop_loss_atr_mult)
                return max(new_stop, current_stop)  # 只能向上调整
            else:
                # 空头跟踪止损
                new_stop = current_price + (atr * self.stop_loss_atr_mult)
                return min(new_stop, current_stop)  # 只能向下调整
                
        except Exception as e:
            self.logger.error(f"❌ 跟踪止损更新失败: {str(e)}")
            return position.get('stop_loss', 0)
    
    def get_risk_level(self, risk_metrics: RiskMetrics) -> RiskLevel:
        """获取风险等级"""
        try:
            if risk_metrics.risk_score >= 0.8:
                return RiskLevel.CRITICAL
            elif risk_metrics.risk_score >= 0.6:
                return RiskLevel.HIGH
            elif risk_metrics.risk_score >= 0.4:
                return RiskLevel.MEDIUM
            else:
                return RiskLevel.LOW
                
        except Exception as e:
            self.logger.error(f"❌ 风险等级计算失败: {str(e)}")
            return RiskLevel.HIGH
    
    def should_reduce_position(self, risk_metrics: RiskMetrics, 
                              position_risk: float) -> bool:
        """判断是否应该减仓"""
        try:
            # 高风险时减仓
            if risk_metrics.risk_score > 0.7:
                return True
            
            # 回撤过大时减仓
            if risk_metrics.current_drawdown < -0.1:  # 回撤超过10%
                return True
            
            # 单笔风险过大时减仓
            if position_risk > self.max_position_risk * 1.5:
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"❌ 减仓判断失败: {str(e)}")
            return True
    
    def should_stop_trading(self, risk_metrics: RiskMetrics) -> bool:
        """判断是否应该停止交易"""
        try:
            # 风险等级为CRITICAL时停止交易
            if self.get_risk_level(risk_metrics) == RiskLevel.CRITICAL:
                return True
            
            # 回撤超过限制时停止交易
            if risk_metrics.current_drawdown < -self.max_drawdown_limit:
                return True
            
            # 投资组合价值过低时停止交易
            if risk_metrics.portfolio_value < self.config.get('min_portfolio_value', 1000):
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"❌ 停止交易判断失败: {str(e)}")
            return True
    
    def update_portfolio_history(self, portfolio_value: float):
        """更新投资组合历史"""
        try:
            self.portfolio_history.append(portfolio_value)
            
            # 保持最近1000个记录
            if len(self.portfolio_history) > 1000:
                self.portfolio_history = self.portfolio_history[-1000:]
                
        except Exception as e:
            self.logger.error(f"❌ 投资组合历史更新失败: {str(e)}")
    
    def get_risk_report(self, risk_metrics: RiskMetrics) -> Dict:
        """生成风险报告"""
        try:
            risk_level = self.get_risk_level(risk_metrics)
            
            report = {
                'timestamp': datetime.now().isoformat(),
                'risk_level': risk_level.value,
                'risk_score': risk_metrics.risk_score,
                'portfolio_value': risk_metrics.portfolio_value,
                'current_drawdown': risk_metrics.current_drawdown,
                'max_drawdown': risk_metrics.max_drawdown,
                'volatility': risk_metrics.volatility,
                'sharpe_ratio': risk_metrics.sharpe_ratio,
                'var_95': risk_metrics.var_95,
                'var_99': risk_metrics.var_99,
                'recommendations': self._get_risk_recommendations(risk_metrics, risk_level)
            }
            
            return report
            
        except Exception as e:
            self.logger.error(f"❌ 风险报告生成失败: {str(e)}")
            return {}
    
    def _get_risk_recommendations(self, risk_metrics: RiskMetrics, 
                                risk_level: RiskLevel) -> List[str]:
        """获取风险建议"""
        recommendations = []
        
        if risk_level == RiskLevel.CRITICAL:
            recommendations.append("立即停止交易，减少仓位")
            recommendations.append("检查市场环境，等待更好的机会")
        elif risk_level == RiskLevel.HIGH:
            recommendations.append("减少仓位大小")
            recommendations.append("提高止损标准")
            recommendations.append("增加现金比例")
        elif risk_level == RiskLevel.MEDIUM:
            recommendations.append("保持当前仓位")
            recommendations.append("密切关注市场变化")
        else:  # LOW
            recommendations.append("可以适当增加仓位")
            recommendations.append("继续监控风险指标")
        
        return recommendations
    
    def save_risk_history(self, filepath: str = 'logs/risk_history.json'):
        """保存风险历史"""
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.risk_history, f, indent=2, ensure_ascii=False, default=str)
                
            self.logger.info(f"✅ 风险历史已保存: {filepath}")
            
        except Exception as e:
            self.logger.error(f"❌ 风险历史保存失败: {str(e)}")


# 使用示例
if __name__ == '__main__':
    # 配置示例
    config = {
        'max_portfolio_risk': 0.02,
        'max_position_risk': 0.005,
        'max_drawdown_limit': 0.15,
        'volatility_limit': 0.3,
        'position_size_method': 'volatility',
        'base_position_size': 0.1,
        'max_position_size': 0.2,
        'stop_loss_atr_mult': 2.0,
        'take_profit_atr_mult': 3.0,
        'trailing_stop': True,
        'min_portfolio_value': 1000
    }
    
    # 创建风险管理器
    risk_manager = RiskManager(config)
    
    # 示例：计算仓位大小
    position_size = risk_manager.calculate_position_size(
        symbol='FIL/USDT',
        signal_strength=0.8,
        current_price=5.0,
        atr=0.1,
        portfolio_value=10000,
        available_cash=5000
    )
    
    print(f"建议仓位大小: {position_size:.6f}")
    
    # 示例：计算止损止盈
    stop_loss, take_profit = risk_manager.calculate_stop_loss_take_profit(
        entry_price=5.0,
        atr=0.1,
        side='buy'
    )
    
    print(f"止损价格: {stop_loss:.4f}")
    print(f"止盈价格: {take_profit:.4f}")
    
    print("风险管理模块已准备就绪")

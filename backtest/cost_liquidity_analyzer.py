"""
交易成本和流动性分析模块
包含交易成本计算、滑点分析、流动性评估等功能
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class CostLiquidityAnalyzer:
    """交易成本和流动性分析器"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.cost_params = config.get('fees_slippage', {})
        
    def calculate_dynamic_fees(self, volume: pd.Series, price: pd.Series, 
                             trade_size: pd.Series) -> pd.Series:
        """计算动态交易费用"""
        # 基础费率
        base_fee_rate = self.cost_params.get('taker_fee_bps', 6) / 10000
        
        # 基于交易量的费率调整
        volume_tier = self._get_volume_tier(volume)
        fee_adjustment = self._calculate_fee_adjustment(volume_tier)
        
        # 基于交易规模的费率调整
        size_adjustment = self._calculate_size_adjustment(trade_size, price)
        
        # 综合费率
        dynamic_fee_rate = base_fee_rate * fee_adjustment * size_adjustment
        
        return dynamic_fee_rate
    
    def _get_volume_tier(self, volume: pd.Series) -> pd.Series:
        """获取交易量等级"""
        # 基于历史交易量的分位数
        volume_quantiles = volume.rolling(30).quantile([0.2, 0.4, 0.6, 0.8])
        
        tier = pd.Series('low', index=volume.index)
        tier[volume > volume_quantiles.iloc[:, 3]] = 'very_high'
        tier[volume > volume_quantiles.iloc[:, 2]] = 'high'
        tier[volume > volume_quantiles.iloc[:, 1]] = 'medium'
        tier[volume > volume_quantiles.iloc[:, 0]] = 'low_medium'
        
        return tier
    
    def _calculate_fee_adjustment(self, volume_tier: pd.Series) -> pd.Series:
        """计算基于交易量的费率调整"""
        adjustments = {
            'very_high': 0.7,  # 大交易量享受折扣
            'high': 0.8,
            'medium': 0.9,
            'low_medium': 1.0,
            'low': 1.1  # 小交易量可能加价
        }
        
        return volume_tier.map(adjustments).fillna(1.0)
    
    def _calculate_size_adjustment(self, trade_size: pd.Series, price: pd.Series) -> pd.Series:
        """计算基于交易规模的费率调整"""
        # 交易金额
        trade_value = trade_size * price
        
        # 基于交易金额的调整
        size_adjustment = pd.Series(1.0, index=trade_size.index)
        
        # 大额交易可能享受折扣
        large_trade_mask = trade_value > trade_value.quantile(0.8)
        size_adjustment[large_trade_mask] = 0.9
        
        # 小额交易可能加价
        small_trade_mask = trade_value < trade_value.quantile(0.2)
        size_adjustment[small_trade_mask] = 1.1
        
        return size_adjustment
    
    def calculate_slippage(self, volume: pd.Series, price: pd.Series, 
                          trade_size: pd.Series, market_impact: bool = True) -> pd.Series:
        """计算滑点"""
        # 基础滑点
        base_slippage = self.cost_params.get('base_slippage_bps', 1) / 10000
        
        # 市场冲击滑点
        if market_impact:
            market_impact_slippage = self._calculate_market_impact_slippage(
                volume, price, trade_size
            )
        else:
            market_impact_slippage = pd.Series(0, index=volume.index)
        
        # 流动性滑点
        liquidity_slippage = self._calculate_liquidity_slippage(volume, trade_size)
        
        # 波动率滑点
        volatility_slippage = self._calculate_volatility_slippage(price)
        
        # 综合滑点
        total_slippage = base_slippage + market_impact_slippage + liquidity_slippage + volatility_slippage
        
        return total_slippage.clip(lower=0)
    
    def _calculate_market_impact_slippage(self, volume: pd.Series, price: pd.Series, 
                                        trade_size: pd.Series) -> pd.Series:
        """计算市场冲击滑点"""
        # 交易量占市场成交量的比例
        volume_ratio = trade_size / volume
        
        # 市场冲击系数（简化模型）
        impact_coefficient = 0.1  # 可调整参数
        
        # 市场冲击滑点
        market_impact = volume_ratio * impact_coefficient
        
        return market_impact.fillna(0)
    
    def _calculate_liquidity_slippage(self, volume: pd.Series, trade_size: pd.Series) -> pd.Series:
        """计算流动性滑点"""
        # 流动性指标
        liquidity_ratio = volume / volume.rolling(20).mean()
        
        # 流动性滑点（流动性越低，滑点越高）
        liquidity_slippage = np.where(
            liquidity_ratio < 0.5, 0.002,  # 低流动性
            np.where(liquidity_ratio < 1.0, 0.001, 0.0005)  # 正常流动性
        )
        
        return pd.Series(liquidity_slippage, index=volume.index)
    
    def _calculate_volatility_slippage(self, price: pd.Series) -> pd.Series:
        """计算波动率滑点"""
        # 价格波动率
        returns = price.pct_change()
        volatility = returns.rolling(20).std()
        
        # 波动率滑点
        volatility_slippage = volatility * 0.5  # 可调整参数
        
        return volatility_slippage.fillna(0)
    
    def calculate_liquidity_metrics(self, volume: pd.Series, price: pd.Series, 
                                  high: pd.Series, low: pd.Series) -> Dict:
        """计算流动性指标"""
        print("💧 计算流动性指标...")
        
        # 基础流动性指标
        liquidity_metrics = {}
        
        # 1. 成交量流动性
        volume_ma = volume.rolling(20).mean()
        liquidity_metrics['volume_liquidity'] = volume / volume_ma
        
        # 2. 价格冲击流动性
        price_impact = self._calculate_price_impact(volume, price)
        liquidity_metrics['price_impact'] = price_impact
        
        # 3. 买卖价差估计
        bid_ask_spread = self._estimate_bid_ask_spread(high, low, price)
        liquidity_metrics['bid_ask_spread'] = bid_ask_spread
        
        # 4. 流动性风险
        liquidity_risk = self._calculate_liquidity_risk(volume, price)
        liquidity_metrics['liquidity_risk'] = liquidity_risk
        
        # 5. 市场深度估计
        market_depth = self._estimate_market_depth(volume, price)
        liquidity_metrics['market_depth'] = market_depth
        
        return liquidity_metrics
    
    def _calculate_price_impact(self, volume: pd.Series, price: pd.Series) -> pd.Series:
        """计算价格冲击"""
        # 简化的价格冲击模型
        volume_ratio = volume / volume.rolling(50).mean()
        price_impact = 1 / volume_ratio  # 成交量越大，价格冲击越小
        
        return price_impact.fillna(1)
    
    def _estimate_bid_ask_spread(self, high: pd.Series, low: pd.Series, 
                               close: pd.Series) -> pd.Series:
        """估计买卖价差"""
        # 基于高低价差的价差估计
        daily_range = high - low
        estimated_spread = daily_range / close * 0.1  # 假设价差为日波动的10%
        
        return estimated_spread.fillna(0)
    
    def _calculate_liquidity_risk(self, volume: pd.Series, price: pd.Series) -> pd.Series:
        """计算流动性风险"""
        # 成交量波动率
        volume_volatility = volume.pct_change().rolling(20).std()
        
        # 价格波动率
        price_volatility = price.pct_change().rolling(20).std()
        
        # 流动性风险 = 成交量波动率 * 价格波动率
        liquidity_risk = volume_volatility * price_volatility
        
        return liquidity_risk.fillna(0)
    
    def _estimate_market_depth(self, volume: pd.Series, price: pd.Series) -> pd.Series:
        """估计市场深度"""
        # 基于成交量的市场深度估计
        volume_ma = volume.rolling(20).mean()
        market_depth = volume_ma * price  # 市场深度 = 平均成交量 * 价格
        
        return market_depth.fillna(0)
    
    def calculate_transaction_costs(self, trade_size: pd.Series, price: pd.Series, 
                                 volume: pd.Series, high: pd.Series, low: pd.Series) -> Dict:
        """计算综合交易成本"""
        print("💰 计算综合交易成本...")
        
        # 交易费用
        fees = self.calculate_dynamic_fees(volume, price, trade_size)
        total_fees = trade_size * price * fees
        
        # 滑点成本
        slippage = self.calculate_slippage(volume, price, trade_size)
        slippage_cost = trade_size * price * slippage
        
        # 流动性成本
        liquidity_metrics = self.calculate_liquidity_metrics(volume, price, high, low)
        liquidity_cost = trade_size * price * liquidity_metrics['price_impact'] * 0.001
        
        # 总成本
        total_cost = total_fees + slippage_cost + liquidity_cost
        
        return {
            'fees': total_fees,
            'slippage_cost': slippage_cost,
            'liquidity_cost': liquidity_cost,
            'total_cost': total_cost,
            'cost_ratio': total_cost / (trade_size * price),
            'liquidity_metrics': liquidity_metrics
        }
    
    def optimize_trade_timing(self, signals: pd.Series, volume: pd.Series, 
                            price: pd.Series, liquidity_metrics: Dict) -> pd.Series:
        """优化交易时机"""
        print("⏰ 优化交易时机...")
        
        # 流动性过滤
        high_liquidity = liquidity_metrics['volume_liquidity'] > 1.0
        low_impact = liquidity_metrics['price_impact'] < liquidity_metrics['price_impact'].quantile(0.7)
        
        # 波动率过滤
        price_volatility = price.pct_change().rolling(20).std()
        low_volatility = price_volatility < price_volatility.quantile(0.6)
        
        # 综合过滤条件
        optimal_timing = signals & high_liquidity & low_impact & low_volatility
        
        return optimal_timing
    
    def calculate_portfolio_liquidity_risk(self, positions: pd.Series, 
                                        liquidity_metrics: Dict) -> pd.Series:
        """计算投资组合流动性风险"""
        # 基于持仓和流动性的风险计算
        position_liquidity_risk = positions * liquidity_metrics['liquidity_risk']
        
        return position_liquidity_risk.fillna(0)
    
    def generate_liquidity_adjusted_signals(self, signals: pd.Series, 
                                          liquidity_metrics: Dict) -> pd.Series:
        """生成流动性调整后的信号"""
        # 流动性阈值
        liquidity_threshold = liquidity_metrics['volume_liquidity'].quantile(0.3)
        impact_threshold = liquidity_metrics['price_impact'].quantile(0.7)
        
        # 流动性过滤
        liquidity_filter = (
            liquidity_metrics['volume_liquidity'] > liquidity_threshold
        ) & (
            liquidity_metrics['price_impact'] < impact_threshold
        )
        
        # 调整后的信号
        adjusted_signals = signals & liquidity_filter
        
        return adjusted_signals
    
    def calculate_cost_attribution(self, trades: pd.DataFrame) -> Dict:
        """计算成本归因分析"""
        if len(trades) == 0:
            return {}
        
        # 按时间分析成本
        trades['cost_ratio'] = trades['fees'] / (trades['size'] * trades['price'])
        
        # 成本统计
        cost_stats = {
            'avg_cost_ratio': trades['cost_ratio'].mean(),
            'max_cost_ratio': trades['cost_ratio'].max(),
            'min_cost_ratio': trades['cost_ratio'].min(),
            'cost_volatility': trades['cost_ratio'].std()
        }
        
        # 按交易规模分析成本
        size_buckets = pd.cut(trades['size'], bins=5, labels=['very_small', 'small', 'medium', 'large', 'very_large'])
        cost_by_size = trades.groupby(size_buckets)['cost_ratio'].agg(['mean', 'std', 'count'])
        
        cost_stats['cost_by_size'] = cost_by_size.to_dict()
        
        return cost_stats
    
    def create_cost_analysis_visualization(self, cost_data: Dict, 
                                         liquidity_metrics: Dict) -> None:
        """创建成本分析可视化"""
        print("📊 创建成本分析可视化...")
        
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. 成本构成
        ax1 = axes[0, 0]
        cost_components = ['fees', 'slippage_cost', 'liquidity_cost']
        cost_values = [cost_data.get(comp, 0) for comp in cost_components]
        ax1.pie(cost_values, labels=cost_components, autopct='%1.1f%%')
        ax1.set_title('交易成本构成')
        
        # 2. 流动性指标时间序列
        ax2 = axes[0, 1]
        ax2.plot(liquidity_metrics['volume_liquidity'], label='成交量流动性', alpha=0.7)
        ax2.plot(liquidity_metrics['price_impact'], label='价格冲击', alpha=0.7)
        ax2.set_title('流动性指标')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 买卖价差分布
        ax3 = axes[0, 2]
        ax3.hist(liquidity_metrics['bid_ask_spread'], bins=30, alpha=0.7, edgecolor='black')
        ax3.set_title('买卖价差分布')
        ax3.set_xlabel('价差')
        ax3.set_ylabel('频次')
        ax3.grid(True, alpha=0.3)
        
        # 4. 流动性风险分析
        ax4 = axes[1, 0]
        ax4.scatter(liquidity_metrics['volume_liquidity'], 
                   liquidity_metrics['liquidity_risk'], alpha=0.6)
        ax4.set_xlabel('成交量流动性')
        ax4.set_ylabel('流动性风险')
        ax4.set_title('流动性风险分析')
        ax4.grid(True, alpha=0.3)
        
        # 5. 市场深度分析
        ax5 = axes[1, 1]
        ax5.plot(liquidity_metrics['market_depth'], alpha=0.7)
        ax5.set_title('市场深度')
        ax5.set_ylabel('市场深度')
        ax5.grid(True, alpha=0.3)
        
        # 6. 成本效率分析
        ax6 = axes[1, 2]
        if 'cost_ratio' in cost_data:
            ax6.hist(cost_data['cost_ratio'], bins=30, alpha=0.7, edgecolor='black')
            ax6.set_title('成本比率分布')
            ax6.set_xlabel('成本比率')
            ax6.set_ylabel('频次')
        else:
            ax6.text(0.5, 0.5, '无成本数据', ha='center', va='center', transform=ax6.transAxes)
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('backtest/cost_liquidity_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print("✅ 成本分析可视化完成")

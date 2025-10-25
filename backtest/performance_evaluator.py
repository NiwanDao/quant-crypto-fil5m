"""
综合性能评估和可视化模块
包含多维度性能指标、风险评估、可视化分析等功能
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import warnings
from scipy import stats
from sklearn.metrics import roc_auc_score, precision_recall_curve, f1_score
import json

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class PerformanceEvaluator:
    """综合性能评估器"""
    
    def __init__(self, config: Dict):
        self.config = config
        
    def calculate_comprehensive_metrics(self, returns: pd.Series, 
                                      benchmark_returns: Optional[pd.Series] = None) -> Dict:
        """计算综合性能指标"""
        print("📊 计算综合性能指标...")
        
        metrics = {}
        
        # 基础收益指标
        metrics.update(self._calculate_return_metrics(returns))
        
        # 风险指标
        metrics.update(self._calculate_risk_metrics(returns))
        
        # 风险调整收益指标
        metrics.update(self._calculate_risk_adjusted_metrics(returns))
        
        # 基准比较指标
        if benchmark_returns is not None:
            metrics.update(self._calculate_benchmark_metrics(returns, benchmark_returns))
        
        # 交易指标
        metrics.update(self._calculate_trading_metrics(returns))
        
        # 稳定性指标
        metrics.update(self._calculate_stability_metrics(returns))
        
        return metrics
    
    def _calculate_return_metrics(self, returns: pd.Series) -> Dict:
        """计算收益指标"""
        total_return = (1 + returns).prod() - 1
        annualized_return = (1 + returns.mean()) ** (252 * 24 * 4) - 1  # 15分钟数据
        cumulative_return = (1 + returns).cumprod()
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'cumulative_return': cumulative_return.iloc[-1],
            'mean_return': returns.mean(),
            'median_return': returns.median()
        }
    
    def _calculate_risk_metrics(self, returns: pd.Series) -> Dict:
        """计算风险指标"""
        volatility = returns.std() * np.sqrt(252 * 24 * 4)  # 年化波动率
        downside_volatility = returns[returns < 0].std() * np.sqrt(252 * 24 * 4)
        
        # VaR计算
        var_95 = np.percentile(returns, 5)
        var_99 = np.percentile(returns, 1)
        
        # CVaR计算
        cvar_95 = returns[returns <= var_95].mean()
        cvar_99 = returns[returns <= var_99].mean()
        
        # 最大回撤
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # 回撤持续时间
        drawdown_duration = self._calculate_drawdown_duration(drawdown)
        
        return {
            'volatility': volatility,
            'downside_volatility': downside_volatility,
            'var_95': var_95,
            'var_99': var_99,
            'cvar_95': cvar_95,
            'cvar_99': cvar_99,
            'max_drawdown': max_drawdown,
            'max_drawdown_duration': drawdown_duration,
            'skewness': returns.skew(),
            'kurtosis': returns.kurtosis()
        }
    
    def _calculate_risk_adjusted_metrics(self, returns: pd.Series) -> Dict:
        """计算风险调整收益指标"""
        # 夏普比率
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252 * 24 * 4) if returns.std() > 0 else 0
        
        # 索提诺比率
        downside_returns = returns[returns < 0]
        sortino_ratio = returns.mean() / downside_returns.std() * np.sqrt(252 * 24 * 4) if len(downside_returns) > 0 and downside_returns.std() > 0 else 0
        
        # 卡玛比率
        max_drawdown = abs(self._calculate_max_drawdown(returns))
        calmar_ratio = returns.mean() * 252 * 24 * 4 / max_drawdown if max_drawdown > 0 else 0
        
        # 信息比率（相对于无风险利率）
        risk_free_rate = 0.02  # 假设无风险利率2%
        excess_returns = returns - risk_free_rate / (252 * 24 * 4)
        information_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252 * 24 * 4) if excess_returns.std() > 0 else 0
        
        return {
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'information_ratio': information_ratio
        }
    
    def _calculate_benchmark_metrics(self, returns: pd.Series, benchmark_returns: pd.Series) -> Dict:
        """计算基准比较指标"""
        # 对齐数据
        aligned_data = pd.DataFrame({
            'strategy': returns,
            'benchmark': benchmark_returns
        }).dropna()
        
        if len(aligned_data) == 0:
            return {}
        
        strategy_returns = aligned_data['strategy']
        benchmark_returns = aligned_data['benchmark']
        
        # 超额收益
        excess_returns = strategy_returns - benchmark_returns
        
        # 跟踪误差
        tracking_error = excess_returns.std() * np.sqrt(252 * 24 * 4)
        
        # 信息比率
        information_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252 * 24 * 4) if excess_returns.std() > 0 else 0
        
        # Beta系数
        beta = np.cov(strategy_returns, benchmark_returns)[0, 1] / np.var(benchmark_returns) if np.var(benchmark_returns) > 0 else 0
        
        # Alpha系数
        alpha = strategy_returns.mean() - beta * benchmark_returns.mean()
        
        # 相关性
        correlation = strategy_returns.corr(benchmark_returns)
        
        return {
            'tracking_error': tracking_error,
            'information_ratio': information_ratio,
            'beta': beta,
            'alpha': alpha,
            'correlation': correlation,
            'excess_return': excess_returns.mean() * 252 * 24 * 4
        }
    
    def _calculate_trading_metrics(self, returns: pd.Series) -> Dict:
        """计算交易指标"""
        # 胜率
        win_rate = (returns > 0).mean()
        
        # 平均盈利和亏损
        winning_returns = returns[returns > 0]
        losing_returns = returns[returns < 0]
        
        avg_win = winning_returns.mean() if len(winning_returns) > 0 else 0
        avg_loss = losing_returns.mean() if len(losing_returns) > 0 else 0
        
        # 盈亏比
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
        
        # 最大连续盈利和亏损
        consecutive_wins, consecutive_losses = self._calculate_consecutive_periods(returns)
        
        return {
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_consecutive_wins': consecutive_wins,
            'max_consecutive_losses': consecutive_losses
        }
    
    def _calculate_stability_metrics(self, returns: pd.Series) -> Dict:
        """计算稳定性指标"""
        # 滚动夏普比率
        rolling_sharpe = returns.rolling(252 * 24 * 4).apply(
            lambda x: x.mean() / x.std() * np.sqrt(252 * 24 * 4) if x.std() > 0 else 0
        )
        
        # 夏普比率稳定性
        sharpe_stability = 1 / rolling_sharpe.std() if rolling_sharpe.std() > 0 else 0
        
        # 收益稳定性
        return_stability = 1 / returns.std() if returns.std() > 0 else 0
        
        # 回撤稳定性
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        drawdown_stability = 1 / drawdown.std() if drawdown.std() > 0 else 0
        
        return {
            'sharpe_stability': sharpe_stability,
            'return_stability': return_stability,
            'drawdown_stability': drawdown_stability,
            'rolling_sharpe_mean': rolling_sharpe.mean(),
            'rolling_sharpe_std': rolling_sharpe.std()
        }
    
    def _calculate_drawdown_duration(self, drawdown: pd.Series) -> int:
        """计算最大回撤持续时间"""
        in_drawdown = drawdown < 0
        drawdown_periods = []
        current_period = 0
        
        for is_dd in in_drawdown:
            if is_dd:
                current_period += 1
            else:
                if current_period > 0:
                    drawdown_periods.append(current_period)
                current_period = 0
        
        if current_period > 0:
            drawdown_periods.append(current_period)
        
        return max(drawdown_periods) if drawdown_periods else 0
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """计算最大回撤"""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    def _calculate_consecutive_periods(self, returns: pd.Series) -> Tuple[int, int]:
        """计算最大连续盈利和亏损期数"""
        max_consecutive_wins = 0
        max_consecutive_losses = 0
        current_wins = 0
        current_losses = 0
        
        for ret in returns:
            if ret > 0:
                current_wins += 1
                current_losses = 0
                max_consecutive_wins = max(max_consecutive_wins, current_wins)
            else:
                current_losses += 1
                current_wins = 0
                max_consecutive_losses = max(max_consecutive_losses, current_losses)
        
        return max_consecutive_wins, max_consecutive_losses
    
    def create_comprehensive_visualization(self, data: pd.DataFrame, returns: pd.Series, 
                                         portfolio_value: pd.Series, signals: Dict) -> None:
        """创建综合可视化分析"""
        print("📊 创建综合可视化分析...")
        
        # 创建大图
        fig = plt.figure(figsize=(24, 20))
        
        # 1. 价格和信号图
        ax1 = plt.subplot(4, 3, 1)
        plt.plot(data.index, data['close'], label='价格', alpha=0.7, linewidth=1)
        
        # 买入信号
        if 'buy_signals' in signals:
            buy_points = data[signals['buy_signals']]
            plt.scatter(buy_points.index, buy_points['close'], 
                       color='green', marker='^', s=30, label='买入信号', alpha=0.8)
        
        # 卖出信号
        if 'sell_signals' in signals:
            sell_points = data[signals['sell_signals']]
            plt.scatter(sell_points.index, sell_points['close'], 
                       color='red', marker='v', s=30, label='卖出信号', alpha=0.8)
        
        plt.title('价格走势与交易信号', fontsize=12, fontweight='bold')
        plt.ylabel('价格')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 2. 投资组合价值
        ax2 = plt.subplot(4, 3, 2)
        plt.plot(portfolio_value.index, portfolio_value, label='投资组合价值', 
                color='blue', linewidth=2)
        plt.axhline(y=portfolio_value.iloc[0], color='red', linestyle='--', 
                   alpha=0.7, label='初始资金')
        plt.title('投资组合价值变化', fontsize=12, fontweight='bold')
        plt.ylabel('价值')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 3. 收益分布
        ax3 = plt.subplot(4, 3, 3)
        plt.hist(returns, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        plt.axvline(x=0, color='red', linestyle='--', label='零收益线')
        plt.title('收益分布', fontsize=12, fontweight='bold')
        plt.xlabel('收益率')
        plt.ylabel('频次')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 4. 累积收益
        ax4 = plt.subplot(4, 3, 4)
        cumulative_returns = (1 + returns).cumprod()
        plt.plot(cumulative_returns.index, cumulative_returns, 
                label='累积收益', color='green', linewidth=2)
        plt.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='初始值')
        plt.title('累积收益曲线', fontsize=12, fontweight='bold')
        plt.ylabel('累积收益')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 5. 回撤分析
        ax5 = plt.subplot(4, 3, 5)
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        plt.fill_between(drawdown.index, 0, drawdown, alpha=0.3, color='red', label='回撤')
        plt.title('回撤分析', fontsize=12, fontweight='bold')
        plt.ylabel('回撤 %')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 6. 滚动夏普比率
        ax6 = plt.subplot(4, 3, 6)
        rolling_sharpe = returns.rolling(252 * 24 * 4).apply(
            lambda x: x.mean() / x.std() * np.sqrt(252 * 24 * 4) if x.std() > 0 else 0
        )
        plt.plot(rolling_sharpe.index, rolling_sharpe, label='滚动夏普比率', color='purple')
        plt.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='夏普比率=1')
        plt.title('滚动夏普比率', fontsize=12, fontweight='bold')
        plt.ylabel('夏普比率')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 7. 月度收益热力图
        ax7 = plt.subplot(4, 3, 7)
        monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
        monthly_returns_pivot = monthly_returns.groupby([monthly_returns.index.year, monthly_returns.index.month]).sum()
        if len(monthly_returns_pivot) > 0:
            sns.heatmap(monthly_returns_pivot.unstack(), annot=True, fmt='.2%', 
                       cmap='RdYlGn', center=0, ax=ax7)
        plt.title('月度收益热力图', fontsize=12, fontweight='bold')
        
        # 8. 风险收益散点图
        ax8 = plt.subplot(4, 3, 8)
        volatility = returns.rolling(20).std() * np.sqrt(252 * 24 * 4)
        rolling_return = returns.rolling(20).mean() * 252 * 24 * 4
        plt.scatter(volatility, rolling_return, alpha=0.6, s=20)
        plt.xlabel('波动率')
        plt.ylabel('年化收益率')
        plt.title('风险收益散点图', fontsize=12, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # 9. 收益自相关分析
        ax9 = plt.subplot(4, 3, 9)
        from statsmodels.tsa.stattools import acf
        autocorr = acf(returns.dropna(), nlags=20)
        plt.plot(autocorr, marker='o')
        plt.axhline(y=0, color='red', linestyle='--', alpha=0.7)
        plt.title('收益自相关分析', fontsize=12, fontweight='bold')
        plt.xlabel('滞后期')
        plt.ylabel('自相关系数')
        plt.grid(True, alpha=0.3)
        
        # 10. 分位数分析
        ax10 = plt.subplot(4, 3, 10)
        quantiles = np.arange(0.01, 1.0, 0.01)
        return_quantiles = np.percentile(returns, quantiles * 100)
        plt.plot(quantiles, return_quantiles, linewidth=2)
        plt.axhline(y=0, color='red', linestyle='--', alpha=0.7)
        plt.title('收益分位数分析', fontsize=12, fontweight='bold')
        plt.xlabel('分位数')
        plt.ylabel('收益率')
        plt.grid(True, alpha=0.3)
        
        # 11. 波动率聚类分析
        ax11 = plt.subplot(4, 3, 11)
        volatility = returns.rolling(20).std()
        plt.plot(volatility.index, volatility, alpha=0.7, label='波动率')
        plt.axhline(y=volatility.mean(), color='red', linestyle='--', alpha=0.7, label='平均波动率')
        plt.title('波动率时间序列', fontsize=12, fontweight='bold')
        plt.ylabel('波动率')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 12. 性能指标雷达图
        ax12 = plt.subplot(4, 3, 12, projection='polar')
        metrics = self.calculate_comprehensive_metrics(returns)
        
        # 选择关键指标进行雷达图展示
        radar_metrics = {
            '夏普比率': min(metrics.get('sharpe_ratio', 0), 3),
            '索提诺比率': min(metrics.get('sortino_ratio', 0), 3),
            '卡玛比率': min(metrics.get('calmar_ratio', 0), 3),
            '胜率': metrics.get('win_rate', 0) * 100,
            '盈亏比': min(metrics.get('profit_factor', 0), 5),
            '最大回撤': abs(metrics.get('max_drawdown', 0)) * 100
        }
        
        angles = np.linspace(0, 2 * np.pi, len(radar_metrics), endpoint=False).tolist()
        values = list(radar_metrics.values())
        angles += angles[:1]  # 闭合图形
        values += values[:1]
        
        ax12.plot(angles, values, 'o-', linewidth=2, label='策略表现')
        ax12.fill(angles, values, alpha=0.25)
        ax12.set_xticks(angles[:-1])
        ax12.set_xticklabels(radar_metrics.keys())
        ax12.set_title('性能指标雷达图', fontsize=12, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig('backtest/comprehensive_performance_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print("✅ 综合可视化分析完成")
    
    def generate_performance_report(self, metrics: Dict, config: Dict) -> str:
        """生成性能报告"""
        print("📝 生成性能报告...")
        
        report = []
        report.append("# 综合回测性能报告")
        report.append("=" * 50)
        report.append("")
        
        # 基本信息
        report.append("## 基本信息")
        report.append(f"- 回测期间: {config.get('backtest', {}).get('start_date', 'N/A')} 至 {config.get('backtest', {}).get('end_date', 'N/A')}")
        report.append(f"- 初始资金: {config.get('backtest', {}).get('initial_cash', 'N/A')}")
        report.append(f"- 交易对: {config.get('symbol', 'N/A')}")
        report.append("")
        
        # 收益指标
        report.append("## 收益指标")
        report.append(f"- 总收益率: {metrics.get('total_return', 0):.2%}")
        report.append(f"- 年化收益率: {metrics.get('annualized_return', 0):.2%}")
        report.append(f"- 平均收益率: {metrics.get('mean_return', 0):.4f}")
        report.append("")
        
        # 风险指标
        report.append("## 风险指标")
        report.append(f"- 年化波动率: {metrics.get('volatility', 0):.2%}")
        report.append(f"- 最大回撤: {metrics.get('max_drawdown', 0):.2%}")
        report.append(f"- VaR(95%): {metrics.get('var_95', 0):.4f}")
        report.append(f"- CVaR(95%): {metrics.get('cvar_95', 0):.4f}")
        report.append("")
        
        # 风险调整收益
        report.append("## 风险调整收益指标")
        report.append(f"- 夏普比率: {metrics.get('sharpe_ratio', 0):.2f}")
        report.append(f"- 索提诺比率: {metrics.get('sortino_ratio', 0):.2f}")
        report.append(f"- 卡玛比率: {metrics.get('calmar_ratio', 0):.2f}")
        report.append("")
        
        # 交易指标
        report.append("## 交易指标")
        report.append(f"- 胜率: {metrics.get('win_rate', 0):.2%}")
        report.append(f"- 平均盈利: {metrics.get('avg_win', 0):.4f}")
        report.append(f"- 平均亏损: {metrics.get('avg_loss', 0):.4f}")
        report.append(f"- 盈亏比: {metrics.get('profit_factor', 0):.2f}")
        report.append("")
        
        # 稳定性指标
        report.append("## 稳定性指标")
        report.append(f"- 夏普比率稳定性: {metrics.get('sharpe_stability', 0):.2f}")
        report.append(f"- 收益稳定性: {metrics.get('return_stability', 0):.2f}")
        report.append(f"- 回撤稳定性: {metrics.get('drawdown_stability', 0):.2f}")
        report.append("")
        
        # 风险提示
        report.append("## 风险提示")
        if metrics.get('max_drawdown', 0) < -0.2:
            report.append("⚠️ 最大回撤超过20%，风险较高")
        if metrics.get('sharpe_ratio', 0) < 1.0:
            report.append("⚠️ 夏普比率低于1.0，风险调整收益不佳")
        if metrics.get('win_rate', 0) < 0.5:
            report.append("⚠️ 胜率低于50%，交易策略需要优化")
        
        report_text = "\n".join(report)
        
        # 保存报告
        with open('backtest/performance_report.txt', 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        return report_text
    
    def save_metrics_to_json(self, metrics: Dict) -> None:
        """保存指标到JSON文件"""
        # 处理numpy类型
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, pd.Series):
                return obj.tolist()
            return obj
        
        # 递归转换
        def recursive_convert(d):
            if isinstance(d, dict):
                return {k: recursive_convert(v) for k, v in d.items()}
            elif isinstance(d, list):
                return [recursive_convert(v) for v in d]
            else:
                return convert_numpy(d)
        
        converted_metrics = recursive_convert(metrics)
        
        with open('backtest/detailed_metrics.json', 'w', encoding='utf-8') as f:
            json.dump(converted_metrics, f, indent=2, ensure_ascii=False)
        
        print("✅ 详细指标已保存到 detailed_metrics.json")

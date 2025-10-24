"""
简化版多方考虑回测系统
基于 feat_2025_3_to_2025_6.parquet 数据
包含核心功能，确保稳定运行
"""

import pandas as pd
import numpy as np
import vectorbt as vbt
import lightgbm as lgb
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import json
import os
import warnings
from datetime import datetime
from sklearn.metrics import roc_auc_score, f1_score
import talib

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class SimpleComprehensiveBacktester:
    """简化版综合回测系统"""
    
    def __init__(self, data_path: str, config_path: str = 'conf/config.yml'):
        self.data_path = data_path
        self.config_path = config_path
        self.config = self._load_config()
        self.data = None
        self.model = None
        self.results = {}
        
    def _load_config(self):
        """加载配置文件"""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def load_data(self):
        """加载数据"""
        print("📊 加载数据...")
        self.data = pd.read_parquet(self.data_path)
        
        # 确保时间索引
        if 'ts' in self.data.columns:
            self.data.set_index('ts', inplace=True)
        
        print(f"✅ 数据加载完成: {len(self.data)} 条记录")
        print(f"📅 时间范围: {self.data.index.min()} 到 {self.data.index.max()}")
        
        return self.data
    
    def load_model(self):
        """加载模型"""
        print("🤖 加载模型...")
        
        try:
            # 尝试加载集成模型
            ensemble_models = joblib.load('models/ensemble_models.pkl')
            self.model = ensemble_models
            print("✅ 集成模型加载成功")
        except:
            try:
                # 回退到优化模型
                self.model = joblib.load('models/lgb_trend_optimized.pkl')
                print("✅ 优化模型加载成功")
            except:
                # 回退到基础模型
                self.model = joblib.load('models/lgb_trend.pkl')
                print("✅ 基础模型加载成功")
        
        return self.model
    
    def calculate_market_regime(self):
        """计算市场状态"""
        print("🔍 分析市场状态...")
        
        close = self.data['close']
        returns = self.data['returns']
        
        # 计算技术指标
        sma_20 = close.rolling(20).mean()
        sma_50 = close.rolling(50).mean()
        volatility = returns.rolling(20).std()
        
        # 趋势强度
        trend_strength = abs(close - sma_20) / sma_20
        
        # 市场状态分类
        regime = pd.Series('neutral', index=close.index)
        regime[(sma_20 > sma_50) & (trend_strength > 0.02)] = 'uptrend'
        regime[(sma_20 < sma_50) & (trend_strength > 0.02)] = 'downtrend'
        regime[(trend_strength < 0.01) & (volatility < volatility.quantile(0.3))] = 'sideways_low_vol'
        regime[(trend_strength < 0.01) & (volatility > volatility.quantile(0.7))] = 'sideways_high_vol'
        regime[volatility > volatility.quantile(0.8)] = 'high_volatility'
        
        return regime
    
    def calculate_risk_metrics(self):
        """计算风险指标"""
        print("📊 计算风险指标...")
        
        close = self.data['close']
        returns = self.data['returns']
        
        # 基础风险指标
        volatility = returns.std() * np.sqrt(252 * 24 * 4)  # 年化波动率
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252 * 24 * 4)
        
        # VaR计算
        var_95 = np.percentile(returns, 5)
        var_99 = np.percentile(returns, 1)
        
        # 最大回撤
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        return {
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'var_95': var_95,
            'var_99': var_99,
            'max_drawdown': max_drawdown
        }
    
    def generate_signals(self):
        """生成交易信号"""
        print("🎯 生成交易信号...")
        
        # 特征列
        features = [c for c in self.data.columns if c not in ['y']]
        
        # 模型预测
        if isinstance(self.model, list):
            # 集成模型
            predictions = []
            for model in self.model:
                pred = model.predict_proba(self.data[features])[:, 1]
                predictions.append(pred)
            prob_up = np.mean(predictions, axis=0)
        else:
            # 单个模型
            prob_up = self.model.predict_proba(self.data[features])[:, 1]
        
        prob_down = 1 - prob_up
        
        # 获取阈值
        buy_threshold = self.config['model']['proba_threshold']
        sell_threshold = self.config['model']['sell_threshold']
        
        # 市场状态
        market_regime = self.calculate_market_regime()
        
        # 动态调整阈值
        dynamic_buy_threshold = np.full(len(prob_up), buy_threshold)
        dynamic_sell_threshold = np.full(len(prob_up), sell_threshold)
        
        # 根据市场状态调整
        for regime in market_regime.unique():
            mask = market_regime == regime
            if regime == 'uptrend':
                dynamic_buy_threshold[mask] *= 0.9
                dynamic_sell_threshold[mask] *= 1.1
            elif regime == 'downtrend':
                dynamic_buy_threshold[mask] *= 1.1
                dynamic_sell_threshold[mask] *= 0.9
            elif regime == 'high_volatility':
                dynamic_buy_threshold[mask] *= 1.2
                dynamic_sell_threshold[mask] *= 0.8
        
        # 生成信号
        buy_signals = (prob_up > dynamic_buy_threshold) & (market_regime != 'high_volatility')
        sell_signals = (prob_down > dynamic_sell_threshold) & (market_regime != 'high_volatility')
        
        # 信号强度
        signal_strength = np.abs(prob_up - prob_down)
        
        return buy_signals, sell_signals, signal_strength, market_regime
    
    def calculate_position_size(self, signal_strength, risk_metrics):
        """计算动态仓位"""
        print("💰 计算动态仓位...")
        
        # 基础仓位
        base_size = self.config['backtest']['fixed_cash_per_trade'] / self.data['close']
        
        # 基于信号强度的调整
        strength_multiplier = np.clip(signal_strength, 0.3, 1.0)
        
        # 基于风险的调整
        volatility = risk_metrics['volatility']
        risk_multiplier = np.clip(0.1 / volatility, 0.5, 2.0)
        
        # 综合仓位
        position_size = base_size * strength_multiplier * risk_multiplier
        
        return np.clip(position_size, 0, None)
    
    def run_backtest(self):
        """运行回测"""
        print("🔄 运行回测...")
        
        # 生成信号
        buy_signals, sell_signals, signal_strength, market_regime = self.generate_signals()
        
        # 计算风险指标
        risk_metrics = self.calculate_risk_metrics()
        
        # 计算仓位
        position_size = self.calculate_position_size(signal_strength, risk_metrics)
        
        # 交易成本
        fees = self.config['fees_slippage']['taker_fee_bps'] / 1e4
        slippage = self.config['fees_slippage']['base_slippage_bps'] / 1e4
        
        # 运行回测
        pf = vbt.Portfolio.from_signals(
            close=self.data['close'],
            entries=buy_signals,
            exits=sell_signals,
            fees=fees,
            slippage=slippage,
            init_cash=self.config['backtest']['initial_cash'],
            size=position_size
        )
        
        return pf, buy_signals, sell_signals, signal_strength, market_regime
    
    def analyze_performance(self, pf):
        """分析性能"""
        print("📈 分析性能指标...")
        
        # 基础统计
        stats = pf.stats()
        
        # 交易分析
        trades = pf.trades.records_readable
        if len(trades) > 0:
            win_rate = len(trades[trades['PnL'] > 0]) / len(trades)
            avg_win = trades[trades['PnL'] > 0]['PnL'].mean() if len(trades[trades['PnL'] > 0]) > 0 else 0
            avg_loss = trades[trades['PnL'] < 0]['PnL'].mean() if len(trades[trades['PnL'] < 0]) > 0 else 0
            profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
        else:
            win_rate = avg_win = avg_loss = profit_factor = 0
        
        # 风险调整收益
        returns = pf.returns()
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252 * 24 * 4) if returns.std() > 0 else 0
        
        # 最大回撤
        max_drawdown = pf.max_drawdown()
        
        return {
            'total_return': stats['Total Return [%]'],
            'annualized_return': stats['Annualized Return [%]'],
            'volatility': stats['Annualized Volatility [%]'],
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_trades': len(trades),
            'avg_win': avg_win,
            'avg_loss': avg_loss
        }
    
    def create_visualizations(self, pf, buy_signals, sell_signals, signal_strength, market_regime):
        """创建可视化"""
        print("📊 生成可视化图表...")
        
        # 创建综合图表
        fig, axes = plt.subplots(4, 2, figsize=(20, 16))
        
        # 1. 价格和信号
        ax1 = axes[0, 0]
        plt.sca(ax1)
        plt.plot(self.data.index, self.data['close'], label='价格', alpha=0.7, linewidth=1)
        
        # 买入信号
        buy_points = self.data[buy_signals]
        plt.scatter(buy_points.index, buy_points['close'], 
                   color='green', marker='^', s=30, label='买入信号', alpha=0.8)
        
        # 卖出信号
        sell_points = self.data[sell_signals]
        plt.scatter(sell_points.index, sell_points['close'], 
                   color='red', marker='v', s=30, label='卖出信号', alpha=0.8)
        
        plt.title('价格走势与交易信号', fontsize=12, fontweight='bold')
        plt.ylabel('价格')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 2. 投资组合价值
        ax2 = axes[0, 1]
        plt.sca(ax2)
        portfolio_value = pf.value()
        plt.plot(portfolio_value.index, portfolio_value, label='投资组合价值', 
                color='blue', linewidth=2)
        plt.axhline(y=self.config['backtest']['initial_cash'], color='red', 
                   linestyle='--', alpha=0.7, label='初始资金')
        plt.title('投资组合价值变化', fontsize=12, fontweight='bold')
        plt.ylabel('价值')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 3. 回撤分析
        ax3 = axes[1, 0]
        plt.sca(ax3)
        drawdown = pf.drawdowns.records_readable
        if len(drawdown) > 0:
            plt.fill_between(drawdown['Start'], 0, drawdown['Drawdown'], 
                           alpha=0.3, color='red', label='回撤')
        plt.title('回撤分析', fontsize=12, fontweight='bold')
        plt.ylabel('回撤 %')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 4. 信号强度分布
        ax4 = axes[1, 1]
        plt.sca(ax4)
        plt.hist(signal_strength, bins=30, alpha=0.7, color='purple', edgecolor='black')
        plt.axvline(x=0.5, color='red', linestyle='--', label='最小信号强度')
        plt.title('信号强度分布', fontsize=12, fontweight='bold')
        plt.xlabel('信号强度')
        plt.ylabel('频次')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 5. 市场状态分析
        ax5 = axes[2, 0]
        plt.sca(ax5)
        regime_colors = {
            'uptrend': 'green',
            'downtrend': 'red', 
            'sideways_low_vol': 'blue',
            'sideways_high_vol': 'orange',
            'high_volatility': 'purple',
            'neutral': 'gray'
        }
        
        for regime in market_regime.unique():
            mask = market_regime == regime
            plt.scatter(self.data.index[mask], self.data['close'][mask], 
                       c=regime_colors.get(regime, 'gray'), 
                       label=regime, alpha=0.6, s=10)
        
        plt.title('市场状态识别', fontsize=12, fontweight='bold')
        plt.ylabel('价格')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        # 6. 收益分布
        ax6 = axes[2, 1]
        plt.sca(ax6)
        returns = pf.returns()
        plt.hist(returns, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        plt.axvline(x=0, color='red', linestyle='--', label='零收益线')
        plt.title('收益分布', fontsize=12, fontweight='bold')
        plt.xlabel('收益率')
        plt.ylabel('频次')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 7. 累积收益
        ax7 = axes[3, 0]
        plt.sca(ax7)
        cumulative_returns = (1 + returns).cumprod()
        plt.plot(cumulative_returns.index, cumulative_returns, 
                label='累积收益', color='green', linewidth=2)
        plt.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='初始值')
        plt.title('累积收益曲线', fontsize=12, fontweight='bold')
        plt.ylabel('累积收益')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 8. 滚动夏普比率
        ax8 = axes[3, 1]
        plt.sca(ax8)
        rolling_sharpe = returns.rolling(252 * 24 * 4).apply(
            lambda x: x.mean() / x.std() * np.sqrt(252 * 24 * 4) if x.std() > 0 else 0
        )
        plt.plot(rolling_sharpe.index, rolling_sharpe, label='滚动夏普比率', color='purple')
        plt.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='夏普比率=1')
        plt.title('滚动夏普比率', fontsize=12, fontweight='bold')
        plt.ylabel('夏普比率')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('backtest/simple_comprehensive_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # 创建权益曲线图
        plt.figure(figsize=(15, 8))
        portfolio_value = pf.value()
        plt.plot(portfolio_value.index, portfolio_value, label='投资组合价值', 
                linewidth=2, color='blue')
        plt.axhline(y=self.config['backtest']['initial_cash'], color='red', 
                   linestyle='--', alpha=0.7, label='初始资金')
        plt.title('投资组合价值曲线', fontsize=16, fontweight='bold')
        plt.xlabel('时间')
        plt.ylabel('价值')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('backtest/simple_equity_curve.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def save_results(self, pf, performance, buy_signals, sell_signals, signal_strength, market_regime):
        """保存结果"""
        print("💾 保存回测结果...")
        
        # 创建输出目录
        os.makedirs('backtest', exist_ok=True)
        
        # 保存统计报告
        stats_df = pf.stats().to_frame('value')
        stats_df.to_csv('backtest/simple_stats.csv')
        
        # 保存交易记录
        if len(pf.trades.records_readable) > 0:
            trades_df = pf.trades.records_readable
            trades_df.to_csv('backtest/simple_trades.csv', index=False)
        
        # 保存信号数据
        signal_data = pd.DataFrame({
            'timestamp': self.data.index,
            'close': self.data['close'],
            'buy_signal': buy_signals,
            'sell_signal': sell_signals,
            'signal_strength': signal_strength,
            'market_regime': market_regime
        })
        signal_data.to_csv('backtest/simple_signals.csv', index=False)
        
        # 保存性能报告
        with open('backtest/simple_performance_report.json', 'w', encoding='utf-8') as f:
            json.dump(performance, f, indent=2, ensure_ascii=False)
        
        print("✅ 结果保存完成")
    
    def run_comprehensive_backtest(self):
        """运行综合回测"""
        print("🚀 开始多方考虑的综合回测...")
        print("=" * 60)
        
        # 1. 加载数据
        self.load_data()
        
        # 2. 加载模型
        self.load_model()
        
        # 3. 运行回测
        pf, buy_signals, sell_signals, signal_strength, market_regime = self.run_backtest()
        
        # 4. 性能分析
        performance = self.analyze_performance(pf)
        
        # 5. 创建可视化
        self.create_visualizations(pf, buy_signals, sell_signals, signal_strength, market_regime)
        
        # 6. 保存结果
        self.save_results(pf, performance, buy_signals, sell_signals, signal_strength, market_regime)
        
        # 7. 输出结果
        print("\n" + "=" * 60)
        print("📊 回测结果汇总")
        print("=" * 60)
        
        print(f"📈 总收益率: {performance['total_return']:.2f}%")
        print(f"📊 年化收益率: {performance['annualized_return']:.2f}%")
        print(f"📉 最大回撤: {performance['max_drawdown']:.2f}%")
        print(f"⚡ 夏普比率: {performance['sharpe_ratio']:.2f}")
        print(f"🎯 胜率: {performance['win_rate']:.2f}")
        print(f"💰 盈亏比: {performance['profit_factor']:.2f}")
        print(f"📊 总交易次数: {performance['total_trades']}")
        
        print(f"\n📁 生成的文件:")
        print(f"  - backtest/simple_stats.csv: 详细统计")
        print(f"  - backtest/simple_trades.csv: 交易记录")
        print(f"  - backtest/simple_signals.csv: 信号数据")
        print(f"  - backtest/simple_performance_report.json: 性能报告")
        print(f"  - backtest/simple_equity_curve.png: 权益曲线")
        print(f"  - backtest/simple_comprehensive_analysis.png: 综合分析图")
        
        print("\n✅ 综合回测完成！")
        
        return pf, performance

def main():
    """主函数"""
    print("🚀 启动简化版多方考虑回测系统")
    print("基于 feat_2025_3_to_2025_6.parquet 数据")
    print("=" * 60)
    
    # 创建回测器
    backtester = SimpleComprehensiveBacktester('data/feat_2025_3_to_2025_6.parquet')
    
    try:
        # 运行综合回测
        pf, performance = backtester.run_comprehensive_backtest()
        
        print("\n🎉 回测分析完成！")
        print("请查看 backtest/ 目录下的详细报告和可视化图表")
        
    except Exception as e:
        print(f"❌ 回测过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()

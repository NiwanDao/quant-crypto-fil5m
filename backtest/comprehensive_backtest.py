"""
多方考虑的综合回测系统
基于 feat_2025_3_to_2025_6.parquet 数据
包含风险管理、市场状态识别、多时间框架分析等
"""

import pandas as pd
import numpy as np
import vectorbt as vbt
import lightgbm as lgb
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
import yaml
import json
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import roc_auc_score, precision_recall_curve, f1_score
from scipy import stats
import talib

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class ComprehensiveBacktester:
    """多方考虑的综合回测系统"""
    
    def __init__(self, data_path: str, config_path: str = 'conf/config.yml'):
        self.data_path = data_path
        self.config_path = config_path
        self.config = self._load_config()
        self.data = None
        self.features = None
        self.model = None
        self.results = {}
        
    def _load_config(self) -> Dict:
        """加载配置文件"""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def load_data(self) -> pd.DataFrame:
        """加载和预处理数据"""
        print("📊 加载数据...")
        self.data = pd.read_parquet(self.data_path)
        
        # 确保时间索引
        if 'ts' in self.data.columns:
            self.data.set_index('ts', inplace=True)
        
        # 特征列
        self.features = [c for c in self.data.columns if c not in ['y']]
        
        print(f"✅ 数据加载完成: {len(self.data)} 条记录, {len(self.features)} 个特征")
        print(f"📅 时间范围: {self.data.index.min()} 到 {self.data.index.max()}")
        
        return self.data
    
    def load_model(self) -> object:
        """加载训练好的模型"""
        print("🤖 加载模型...")
        
        # 尝试加载集成模型
        try:
            ensemble_models = joblib.load('models/ensemble_models.pkl')
            self.model = ensemble_models
            print("✅ 集成模型加载成功")
        except:
            # 回退到单个模型
            try:
                self.model = joblib.load('models/lgb_trend_optimized.pkl')
                print("✅ 优化模型加载成功")
            except:
                self.model = joblib.load('models/lgb_trend.pkl')
                print("✅ 基础模型加载成功")
        
        return self.model
    
    def calculate_market_regime(self, window: int = 20) -> pd.Series:
        """识别市场状态（趋势/震荡/高波动/低波动）"""
        print("🔍 分析市场状态...")
        
        close = self.data['close']
        returns = self.data['returns']
        
        # 计算技术指标
        sma_20 = close.rolling(window).mean()
        sma_50 = close.rolling(50).mean()
        volatility = returns.rolling(window).std()
        rsi = talib.RSI(close.values, timeperiod=14)
        rsi = pd.Series(rsi, index=close.index)
        
        # 趋势强度
        trend_strength = abs(close - sma_20) / sma_20
        
        # 市场状态分类
        regime = pd.Series('neutral', index=close.index)
        
        # 趋势市场
        regime[(sma_20 > sma_50) & (trend_strength > 0.02)] = 'uptrend'
        regime[(sma_20 < sma_50) & (trend_strength > 0.02)] = 'downtrend'
        
        # 震荡市场
        regime[(trend_strength < 0.01) & (volatility < volatility.quantile(0.3))] = 'sideways_low_vol'
        regime[(trend_strength < 0.01) & (volatility > volatility.quantile(0.7))] = 'sideways_high_vol'
        
        # 高波动市场
        regime[volatility > volatility.quantile(0.8)] = 'high_volatility'
        
        return regime
    
    def calculate_risk_metrics(self) -> Dict:
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
        
        # 偏度和峰度
        skewness = returns.skew()
        kurtosis = returns.kurtosis()
        
        return {
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'var_95': var_95,
            'var_99': var_99,
            'max_drawdown': max_drawdown,
            'skewness': skewness,
            'kurtosis': kurtosis
        }
    
    def generate_signals(self) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """生成交易信号"""
        print("🎯 生成交易信号...")
        
        # 模型预测
        if isinstance(self.model, list):
            # 集成模型
            predictions = []
            for model in self.model:
                pred = model.predict_proba(self.data[self.features])[:, 1]
                predictions.append(pred)
            prob_up = np.mean(predictions, axis=0)
        else:
            # 单个模型
            prob_up = self.model.predict_proba(self.data[self.features])[:, 1]
        
        prob_down = 1 - prob_up
        
        # 获取阈值
        buy_threshold = self.config['model']['proba_threshold']
        sell_threshold = self.config['model']['sell_threshold']
        
        # 市场状态
        market_regime = self.calculate_market_regime()
        
        # 动态调整阈值
        dynamic_buy_threshold = buy_threshold.copy()
        dynamic_sell_threshold = sell_threshold.copy()
        
        # 根据市场状态调整
        for regime in market_regime.unique():
            mask = market_regime == regime
            if regime == 'uptrend':
                dynamic_buy_threshold[mask] *= 0.9  # 降低买入阈值
                dynamic_sell_threshold[mask] *= 1.1  # 提高卖出阈值
            elif regime == 'downtrend':
                dynamic_buy_threshold[mask] *= 1.1  # 提高买入阈值
                dynamic_sell_threshold[mask] *= 0.9  # 降低卖出阈值
            elif regime == 'high_volatility':
                dynamic_buy_threshold[mask] *= 1.2  # 大幅提高买入阈值
                dynamic_sell_threshold[mask] *= 0.8  # 大幅降低卖出阈值
        
        # 生成信号
        buy_signals = (prob_up > dynamic_buy_threshold) & (market_regime != 'high_volatility')
        sell_signals = (prob_down > dynamic_sell_threshold) & (market_regime != 'high_volatility')
        
        # 信号强度
        signal_strength = np.abs(prob_up - prob_down)
        
        return buy_signals, sell_signals, signal_strength
    
    def calculate_position_size(self, signal_strength: pd.Series, risk_metrics: Dict) -> pd.Series:
        """动态仓位计算"""
        print("💰 计算动态仓位...")
        
        # 基础仓位
        base_size = self.config['backtest']['fixed_cash_per_trade'] / self.data['close']
        
        # 基于信号强度的调整
        strength_multiplier = signal_strength.clip(0.3, 1.0)
        
        # 基于风险的调整
        volatility = risk_metrics['volatility']
        risk_multiplier = np.clip(0.1 / volatility, 0.5, 2.0)
        
        # 基于市场状态的调整
        market_regime = self.calculate_market_regime()
        regime_multiplier = pd.Series(1.0, index=self.data.index)
        regime_multiplier[market_regime == 'high_volatility'] = 0.5
        regime_multiplier[market_regime == 'uptrend'] = 1.2
        regime_multiplier[market_regime == 'downtrend'] = 0.8
        
        # 综合仓位
        position_size = base_size * strength_multiplier * risk_multiplier * regime_multiplier
        
        return position_size.clip(0)
    
    def run_backtest(self) -> vbt.Portfolio:
        """运行回测"""
        print("🔄 运行综合回测...")
        
        # 生成信号
        buy_signals, sell_signals, signal_strength = self.generate_signals()
        
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
        
        return pf
    
    def analyze_performance(self, pf: vbt.Portfolio) -> Dict:
        """性能分析"""
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
        
        # 胜率分析
        consecutive_wins = 0
        consecutive_losses = 0
        max_consecutive_wins = 0
        max_consecutive_losses = 0
        
        if len(trades) > 0:
            for pnl in trades['PnL']:
                if pnl > 0:
                    consecutive_wins += 1
                    consecutive_losses = 0
                    max_consecutive_wins = max(max_consecutive_wins, consecutive_wins)
                else:
                    consecutive_losses += 1
                    consecutive_wins = 0
                    max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
        
        return {
            'total_return': stats['Total Return [%]'],
            'annualized_return': stats['Annualized Return [%]'],
            'volatility': stats['Annualized Volatility [%]'],
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_trades': len(trades),
            'max_consecutive_wins': max_consecutive_wins,
            'max_consecutive_losses': max_consecutive_losses,
            'avg_win': avg_win,
            'avg_loss': avg_loss
        }
    
    def create_visualizations(self, pf: vbt.Portfolio, buy_signals: pd.Series, 
                           sell_signals: pd.Series, signal_strength: pd.Series):
        """创建可视化图表"""
        print("📊 生成可视化图表...")
        
        # 创建综合图表
        fig = plt.figure(figsize=(20, 24))
        
        # 1. 价格和信号图
        ax1 = plt.subplot(6, 1, 1)
        plt.plot(self.data.index, self.data['close'], label='Close Price', alpha=0.7, linewidth=1)
        
        # 买入信号
        buy_points = self.data[buy_signals]
        plt.scatter(buy_points.index, buy_points['close'], 
                   color='green', marker='^', s=50, label='Buy Signal', alpha=0.8)
        
        # 卖出信号
        sell_points = self.data[sell_signals]
        plt.scatter(sell_points.index, sell_points['close'], 
                   color='red', marker='v', s=50, label='Sell Signal', alpha=0.8)
        
        plt.title('价格走势与交易信号', fontsize=14, fontweight='bold')
        plt.ylabel('价格')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 2. 投资组合价值
        ax2 = plt.subplot(6, 1, 2)
        portfolio_value = pf.value()
        plt.plot(portfolio_value.index, portfolio_value, label='投资组合价值', color='blue', linewidth=2)
        plt.axhline(y=self.config['backtest']['initial_cash'], color='red', 
                   linestyle='--', alpha=0.7, label='初始资金')
        plt.title('投资组合价值变化', fontsize=14, fontweight='bold')
        plt.ylabel('价值')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 3. 回撤分析
        ax3 = plt.subplot(6, 1, 3)
        drawdown = pf.drawdowns.records_readable
        if len(drawdown) > 0:
            plt.fill_between(drawdown['Start'], 0, drawdown['Drawdown'], 
                           alpha=0.3, color='red', label='回撤')
        plt.title('回撤分析', fontsize=14, fontweight='bold')
        plt.ylabel('回撤 %')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 4. 信号强度分布
        ax4 = plt.subplot(6, 1, 4)
        plt.hist(signal_strength, bins=30, alpha=0.7, color='purple', edgecolor='black')
        plt.axvline(x=0.5, color='red', linestyle='--', label='最小信号强度')
        plt.title('信号强度分布', fontsize=14, fontweight='bold')
        plt.xlabel('信号强度')
        plt.ylabel('频次')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 5. 市场状态分析
        ax5 = plt.subplot(6, 1, 5)
        market_regime = self.calculate_market_regime()
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
        
        plt.title('市场状态识别', fontsize=14, fontweight='bold')
        plt.ylabel('价格')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        # 6. 收益分布
        ax6 = plt.subplot(6, 1, 6)
        returns = pf.returns()
        plt.hist(returns, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        plt.axvline(x=0, color='red', linestyle='--', label='零收益线')
        plt.title('收益分布', fontsize=14, fontweight='bold')
        plt.xlabel('收益率')
        plt.ylabel('频次')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('backtest/comprehensive_analysis.png', dpi=150, bbox_inches='tight')
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
        plt.savefig('backtest/equity_curve.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def save_results(self, pf: vbt.Portfolio, performance: Dict, 
                    buy_signals: pd.Series, sell_signals: pd.Series, 
                    signal_strength: pd.Series):
        """保存回测结果"""
        print("💾 保存回测结果...")
        
        # 保存统计报告
        stats_df = pf.stats().to_frame('value')
        stats_df.to_csv('backtest/stats.csv')
        
        # 保存交易记录
        if len(pf.trades.records_readable) > 0:
            trades_df = pf.trades.records_readable
            trades_df.to_csv('backtest/trades.csv', index=False)
        
        # 保存信号数据
        signal_data = pd.DataFrame({
            'timestamp': self.data.index,
            'close': self.data['close'],
            'buy_signal': buy_signals,
            'sell_signal': sell_signals,
            'signal_strength': signal_strength,
            'market_regime': self.calculate_market_regime()
        })
        signal_data.to_csv('backtest/signals.csv', index=False)
        
        # 保存性能报告
        with open('backtest/performance_report.json', 'w', encoding='utf-8') as f:
            json.dump(performance, f, indent=2, ensure_ascii=False)
        
        print("✅ 结果保存完成")
    
    def run_comprehensive_backtest(self):
        """运行综合回测"""
        print("🚀 开始多方考虑的综合回测...")
        print("="*60)
        
        # 1. 加载数据
        self.load_data()
        
        # 2. 加载模型
        self.load_model()
        
        # 3. 运行回测
        pf = self.run_backtest()
        
        # 4. 生成信号
        buy_signals, sell_signals, signal_strength = self.generate_signals()
        
        # 5. 性能分析
        performance = self.analyze_performance(pf)
        
        # 6. 创建可视化
        self.create_visualizations(pf, buy_signals, sell_signals, signal_strength)
        
        # 7. 保存结果
        self.save_results(pf, performance, buy_signals, sell_signals, signal_strength)
        
        # 8. 输出结果
        print("\n" + "="*60)
        print("📊 回测结果汇总")
        print("="*60)
        
        print(f"📈 总收益率: {performance['total_return']:.2f}%")
        print(f"📊 年化收益率: {performance['annualized_return']:.2f}%")
        print(f"📉 最大回撤: {performance['max_drawdown']:.2f}%")
        print(f"⚡ 夏普比率: {performance['sharpe_ratio']:.2f}")
        print(f"🎯 胜率: {performance['win_rate']:.2f}")
        print(f"💰 盈亏比: {performance['profit_factor']:.2f}")
        print(f"📊 总交易次数: {performance['total_trades']}")
        print(f"🔥 最大连续盈利: {performance['max_consecutive_wins']}")
        print(f"❄️ 最大连续亏损: {performance['max_consecutive_losses']}")
        
        print(f"\n📁 生成的文件:")
        print(f"  - backtest/stats.csv: 详细统计")
        print(f"  - backtest/trades.csv: 交易记录")
        print(f"  - backtest/signals.csv: 信号数据")
        print(f"  - backtest/performance_report.json: 性能报告")
        print(f"  - backtest/equity_curve.png: 权益曲线")
        print(f"  - backtest/comprehensive_analysis.png: 综合分析图")
        
        print("\n✅ 综合回测完成！")
        
        return pf, performance

def main():
    """主函数"""
    # 创建回测器
    backtester = ComprehensiveBacktester('data/feat_2025_3_to_2025_6.parquet')
    
    # 运行综合回测
    pf, performance = backtester.run_comprehensive_backtest()
    
    return pf, performance

if __name__ == '__main__':
    main()

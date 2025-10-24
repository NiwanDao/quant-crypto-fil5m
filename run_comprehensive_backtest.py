"""
多方考虑的综合回测系统主运行脚本
整合所有模块，提供完整的回测分析
"""

import pandas as pd
import numpy as np
import vectorbt as vbt
import yaml
import os
import sys
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 导入自定义模块
sys.path.append('/home/shiyi/quant-crypto-fil5m/backtest')
from comprehensive_backtest import ComprehensiveBacktester
from risk_manager import AdvancedRiskManager
from multi_timeframe_analyzer import MultiTimeframeAnalyzer
from market_regime_detector import MarketRegimeDetector
from cost_liquidity_analyzer import CostLiquidityAnalyzer
from performance_evaluator import PerformanceEvaluator

class MasterBacktester:
    """主回测系统"""
    
    def __init__(self, data_path: str, config_path: str = 'conf/config.yml'):
        self.data_path = data_path
        self.config_path = config_path
        self.config = self._load_config()
        
        # 初始化各个模块
        self.backtester = ComprehensiveBacktester(data_path, config_path)
        self.risk_manager = AdvancedRiskManager(self.config)
        self.mtf_analyzer = MultiTimeframeAnalyzer(self.config)
        self.regime_detector = MarketRegimeDetector(self.config)
        self.cost_analyzer = CostLiquidityAnalyzer(self.config)
        self.performance_evaluator = PerformanceEvaluator(self.config)
        
        # 存储结果
        self.results = {}
        
    def _load_config(self) -> dict:
        """加载配置文件"""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def run_comprehensive_analysis(self):
        """运行综合分析"""
        print("🚀 开始多方考虑的综合回测分析...")
        print("=" * 80)
        
        # 1. 数据加载和预处理
        print("\n📊 步骤1: 数据加载和预处理")
        print("-" * 40)
        data = self.backtester.load_data()
        self.results['data'] = data
        
        # 2. 模型加载
        print("\n🤖 步骤2: 模型加载")
        print("-" * 40)
        model = self.backtester.load_model()
        self.results['model'] = model
        
        # 3. 市场状态识别
        print("\n🔍 步骤3: 市场状态识别")
        print("-" * 40)
        market_features = self.regime_detector.calculate_market_features(data)
        market_regimes = self.regime_detector.detect_market_regimes(market_features)
        self.results['market_regimes'] = market_regimes
        self.results['market_features'] = market_features
        
        # 4. 多时间框架分析
        print("\n⏰ 步骤4: 多时间框架分析")
        print("-" * 40)
        mtf_signals = self.mtf_analyzer.generate_multi_timeframe_signals(data)
        mtf_alignment = self.mtf_analyzer.calculate_timeframe_alignment(data)
        self.results['mtf_signals'] = mtf_signals
        self.results['mtf_alignment'] = mtf_alignment
        
        # 5. 风险分析
        print("\n⚠️ 步骤5: 风险分析")
        print("-" * 40)
        risk_metrics = self.risk_manager.calculate_risk_metrics(data, data['returns'])
        self.results['risk_metrics'] = risk_metrics
        
        # 6. 成本和流动性分析
        print("\n💰 步骤6: 成本和流动性分析")
        print("-" * 40)
        liquidity_metrics = self.cost_analyzer.calculate_liquidity_metrics(
            data['volume'], data['close'], data['high'], data['low']
        )
        self.results['liquidity_metrics'] = liquidity_metrics
        
        # 7. 生成综合信号
        print("\n🎯 步骤7: 生成综合信号")
        print("-" * 40)
        buy_signals, sell_signals, signal_strength = self._generate_comprehensive_signals(
            data, market_regimes, mtf_signals, risk_metrics, liquidity_metrics
        )
        self.results['signals'] = {
            'buy_signals': buy_signals,
            'sell_signals': sell_signals,
            'signal_strength': signal_strength
        }
        
        # 8. 运行回测
        print("\n🔄 步骤8: 运行回测")
        print("-" * 40)
        pf = self._run_enhanced_backtest(data, buy_signals, sell_signals, signal_strength)
        self.results['portfolio'] = pf
        
        # 9. 性能评估
        print("\n📈 步骤9: 性能评估")
        print("-" * 40)
        returns = pf.returns()
        performance_metrics = self.performance_evaluator.calculate_comprehensive_metrics(returns)
        self.results['performance_metrics'] = performance_metrics
        
        # 10. 生成报告和可视化
        print("\n📊 步骤10: 生成报告和可视化")
        print("-" * 40)
        self._generate_comprehensive_reports(data, pf, returns)
        
        print("\n✅ 综合分析完成！")
        return self.results
    
    def _generate_comprehensive_signals(self, data: pd.DataFrame, market_regimes: pd.Series,
                                      mtf_signals: dict, risk_metrics: dict, 
                                      liquidity_metrics: dict) -> tuple:
        """生成综合信号"""
        print("🎯 生成综合交易信号...")
        
        # 基础模型信号
        if isinstance(self.results['model'], list):
            # 集成模型
            predictions = []
            for model in self.results['model']:
                pred = model.predict_proba(data[self.backtester.features])[:, 1]
                predictions.append(pred)
            prob_up = np.mean(predictions, axis=0)
        else:
            # 单个模型
            prob_up = self.results['model'].predict_proba(data[self.backtester.features])[:, 1]
        
        prob_down = 1 - prob_up
        
        # 获取阈值
        buy_threshold = self.config['model']['proba_threshold']
        sell_threshold = self.config['model']['sell_threshold']
        
        # 基础信号
        base_buy_signals = prob_up > buy_threshold
        base_sell_signals = prob_down > sell_threshold
        
        # 市场状态过滤
        regime_filter = market_regimes != 'high_volatility'
        
        # 流动性过滤
        liquidity_filter = self.cost_analyzer.generate_liquidity_adjusted_signals(
            base_buy_signals, liquidity_metrics
        )
        
        # 风险过滤
        risk_filter = self.risk_manager.generate_risk_adjusted_signals(
            base_buy_signals, risk_metrics
        )
        
        # 多时间框架确认
        mtf_buy_signals, mtf_sell_signals = self.mtf_analyzer.combine_multi_timeframe_signals(mtf_signals)
        mtf_filter = (mtf_buy_signals > 0.5) | (mtf_sell_signals > 0.5)
        
        # 综合信号
        buy_signals = base_buy_signals & regime_filter & liquidity_filter & risk_filter & mtf_filter
        sell_signals = base_sell_signals & regime_filter & liquidity_filter & risk_filter & mtf_filter
        
        # 信号强度
        signal_strength = np.abs(prob_up - prob_down)
        
        return buy_signals, sell_signals, signal_strength
    
    def _run_enhanced_backtest(self, data: pd.DataFrame, buy_signals: pd.Series,
                             sell_signals: pd.Series, signal_strength: pd.Series) -> vbt.Portfolio:
        """运行增强回测"""
        print("🔄 运行增强回测...")
        
        # 动态仓位计算
        position_size = self.risk_manager.calculate_dynamic_position_sizing(
            buy_signals, self.results['risk_metrics'], 
            self.config['backtest']['initial_cash']
        )
        
        # 交易成本
        fees = self.config['fees_slippage']['taker_fee_bps'] / 1e4
        slippage = self.config['fees_slippage']['base_slippage_bps'] / 1e4
        
        # 运行回测
        pf = vbt.Portfolio.from_signals(
            close=data['close'],
            entries=buy_signals,
            exits=sell_signals,
            fees=fees,
            slippage=slippage,
            init_cash=self.config['backtest']['initial_cash'],
            size=position_size
        )
        
        return pf
    
    def _generate_comprehensive_reports(self, data: pd.DataFrame, pf: vbt.Portfolio, 
                                      returns: pd.Series):
        """生成综合报告"""
        print("📊 生成综合报告和可视化...")
        
        # 创建输出目录
        os.makedirs('backtest', exist_ok=True)
        
        # 1. 基础统计报告
        stats = pf.stats()
        stats_df = stats.to_frame('value')
        stats_df.to_csv('backtest/enhanced_stats.csv')
        
        # 2. 交易记录
        if len(pf.trades.records_readable) > 0:
            trades_df = pf.trades.records_readable
            trades_df.to_csv('backtest/enhanced_trades.csv', index=False)
        
        # 3. 信号数据
        signal_data = pd.DataFrame({
            'timestamp': data.index,
            'close': data['close'],
            'buy_signal': self.results['signals']['buy_signals'],
            'sell_signal': self.results['signals']['sell_signals'],
            'signal_strength': self.results['signals']['signal_strength'],
            'market_regime': self.results['market_regimes'],
            'mtf_alignment': self.results['mtf_alignment']
        })
        signal_data.to_csv('backtest/enhanced_signals.csv', index=False)
        
        # 4. 性能报告
        performance_report = self.performance_evaluator.generate_performance_report(
            self.results['performance_metrics'], self.config
        )
        
        # 5. 详细指标
        self.performance_evaluator.save_metrics_to_json(self.results['performance_metrics'])
        
        # 6. 可视化分析
        self.performance_evaluator.create_comprehensive_visualization(
            data, returns, pf.value(), self.results['signals']
        )
        
        # 7. 市场状态可视化
        self.regime_detector.create_regime_visualization(
            data, self.results['market_regimes'], self.results['market_features']
        )
        
        # 8. 成本分析可视化
        if 'liquidity_metrics' in self.results:
            self.cost_analyzer.create_cost_analysis_visualization(
                {}, self.results['liquidity_metrics']
            )
        
        print("✅ 所有报告和可视化已生成")
    
    def print_summary(self):
        """打印分析摘要"""
        print("\n" + "=" * 80)
        print("📊 综合分析摘要")
        print("=" * 80)
        
        if 'performance_metrics' in self.results:
            metrics = self.results['performance_metrics']
            
            print(f"📈 总收益率: {metrics.get('total_return', 0):.2%}")
            print(f"📊 年化收益率: {metrics.get('annualized_return', 0):.2%}")
            print(f"📉 最大回撤: {metrics.get('max_drawdown', 0):.2%}")
            print(f"⚡ 夏普比率: {metrics.get('sharpe_ratio', 0):.2f}")
            print(f"🎯 胜率: {metrics.get('win_rate', 0):.2%}")
            print(f"💰 盈亏比: {metrics.get('profit_factor', 0):.2f}")
            
            if 'portfolio' in self.results:
                pf = self.results['portfolio']
                print(f"📊 总交易次数: {len(pf.trades.records_readable)}")
                print(f"💵 最终价值: {pf.value().iloc[-1]:.2f}")
        
        print(f"\n📁 生成的文件:")
        print(f"  - backtest/enhanced_stats.csv: 详细统计")
        print(f"  - backtest/enhanced_trades.csv: 交易记录")
        print(f"  - backtest/enhanced_signals.csv: 信号数据")
        print(f"  - backtest/performance_report.txt: 性能报告")
        print(f"  - backtest/detailed_metrics.json: 详细指标")
        print(f"  - backtest/comprehensive_performance_analysis.png: 性能分析图")
        print(f"  - backtest/market_regime_analysis.png: 市场状态分析图")
        print(f"  - backtest/cost_liquidity_analysis.png: 成本流动性分析图")

def main():
    """主函数"""
    print("🚀 启动多方考虑的综合回测系统")
    print("基于 feat_2025_3_to_2025_6.parquet 数据")
    print("=" * 80)
    
    # 创建主回测器
    master_backtester = MasterBacktester('data/feat_2025_3_to_2025_6.parquet')
    
    try:
        # 运行综合分析
        results = master_backtester.run_comprehensive_analysis()
        
        # 打印摘要
        master_backtester.print_summary()
        
        print("\n🎉 综合分析完成！")
        print("请查看 backtest/ 目录下的详细报告和可视化图表")
        
    except Exception as e:
        print(f"❌ 分析过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()

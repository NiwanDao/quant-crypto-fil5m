# backtest_validation.py
import pandas as pd
import numpy as np
from validation.model_validation import ModelValidator
import vectorbt as vbt
from datetime import datetime, timedelta
from typing import Dict, List

class BacktestValidator:
    def __init__(self, model_validator: ModelValidator):
        self.validator = model_validator
    
    def safe_get_stats(self, pf, stat_name, default_value=0.0):
        """安全获取统计指标"""
        try:
            stats = pf.stats()
            # 尝试不同的可能键名
            possible_keys = {
                'win_rate': ['Win Rate [%]', 'Win Rate', 'win_rate', 'WinRate'],
                'total_trades': ['Total Trades', 'Total Trades', 'total_trades', 'Trades']
            }
            
            if stat_name in possible_keys:
                for key in possible_keys[stat_name]:
                    if key in stats:
                        return stats[key]
            
            # 如果找不到，返回默认值
            return default_value
        except Exception as e:
            print(f"⚠️ 获取统计指标 {stat_name} 失败: {e}")
            return default_value
    

    def improved_walk_forward_backtest(self, df: pd.DataFrame, initial_cash: float = 10000, 
                                    lookback_days: int = 30, step_days: int = 5) -> Dict:
        """改进的Walk-Forward回测 - 修复信号问题"""
        print("📈 开始改进的Walk-Forward回测...")
        
        results = []
        current_date = df.index.min() + timedelta(days=lookback_days)
        end_date = df.index.max()
        
        # 获取特征列名
        feature_cols = self.validator.models[0].feature_name_
        
        period_count = 0
        max_periods = 20
        
        while current_date <= end_date and period_count < max_periods:
            period_count += 1
            
            # 测试数据开始日期
            test_start = current_date
            test_end = min(test_start + timedelta(days=step_days), end_date)
            
            # 分割数据
            test_mask = (df.index >= test_start) & (df.index <= test_end)
            
            if test_mask.sum() == 0:
                print(f"周期 {period_count}: 无测试数据")
                break
                
            try:
                # 使用模型预测
                X_test = df[test_mask][feature_cols]
                y_proba = self.validator.ensemble_predict(X_test)
                
                # 调试信息
                print(f"周期 {period_count}: 预测概率范围 [{y_proba.min():.4f}, {y_proba.max():.4f}]")
                
                # 生成交易信号 - 添加调试
                buy_threshold = self.validator.thresholds['buy_threshold']
                sell_threshold = self.validator.thresholds.get('sell_threshold', 0.1)  # 默认值
                
                entries = y_proba > buy_threshold
                exits = y_proba < sell_threshold
                
                print(f"  买入信号: {entries.sum()}, 卖出信号: {exits.sum()}")
                
                
                # 如果仍然没有信号，尝试调整阈值
                if entries.sum() == 0 and exits.sum() == 0:
                    print(f"  ⚠️ 无交易信号，尝试动态调整阈值...")
                    # 动态调整阈值
                    dynamic_buy_threshold = np.percentile(y_proba, 70)  # 前30%作为买入信号
                    dynamic_sell_threshold = np.percentile(y_proba, 30)  # 后30%作为卖出信号
                    
                    entries = y_proba > dynamic_buy_threshold
                    exits = y_proba < dynamic_sell_threshold
                    print(f"  动态阈值 - 买入: {dynamic_buy_threshold:.4f}, 卖出: {dynamic_sell_threshold:.4f}")
                    print(f"  动态信号 - 买入: {entries.sum()}, 卖出: {exits.sum()}")
                
                # 执行回测
                close = df.loc[test_mask, 'close']
                
                # 添加交易费用
                fees = 0.001  # 0.1%
                
                pf = vbt.Portfolio.from_signals(
                    close=close,
                    entries=entries,
                    exits=exits,
                    init_cash=initial_cash,
                    fees=fees,
                    freq='1d'
                )
                
                # 获取统计信息
                total_return = pf.total_return()
                total_trades = len(pf.orders) if hasattr(pf, 'orders') else 0
                
                period_result = {
                    'period_start': test_start,
                    'period_end': test_end,
                    'total_return': total_return,
                    'total_trades': total_trades,
                    'buy_signals': entries.sum(),
                    'sell_signals': exits.sum(),
                    'portfolio_value': pf.value().iloc[-1] if len(pf.value()) > 0 else initial_cash
                }
                results.append(period_result)
                
                print(f"周期 {period_count}: {test_start.date()} 到 {test_end.date()}")
                print(f"  收益: {total_return:.2%}, 交易数: {total_trades}")
                print(f"  买入信号: {entries.sum()}, 卖出信号: {exits.sum()}")
                print("-" * 50)
                
            except Exception as e:
                print(f"❌ 周期 {period_count} 回测失败: {e}")
                results.append({
                    'period_start': test_start,
                    'period_end': test_end,
                    'total_return': 0.0,
                    'total_trades': 0,
                    'buy_signals': 0,
                    'sell_signals': 0,
                    'portfolio_value': initial_cash,
                    'error': str(e)
                })
            
            current_date = test_end + timedelta(days=1)
        
        return self._analyze_backtest_results(results, initial_cash)

    def _analyze_backtest_results(self, results: List, initial_cash: float) -> Dict:
        """分析回测结果"""
        if not results:
            return {'error': '没有回测结果'}
        
        results_df = pd.DataFrame(results)
        
        # 计算成功周期
        winning_periods = (results_df['total_return'] > 0).sum()
        total_periods = len(results_df)
        total_trades = results_df['total_trades'].sum()
        
        summary = {
            'total_periods': total_periods,
            'win_periods': winning_periods,
            'success_rate': winning_periods / total_periods if total_periods > 0 else 0.0,
            'avg_return': results_df['total_return'].mean(),
            'total_trades': total_trades,
            'avg_trades_per_period': total_trades / total_periods if total_periods > 0 else 0,
            'final_portfolio_value': results_df['portfolio_value'].iloc[-1],
            'total_buy_signals': results_df['buy_signals'].sum(),
            'total_sell_signals': results_df['sell_signals'].sum()
        }
        
        print(f"\n📊 回测总结:")
        print(f"  总周期数: {summary['total_periods']}")
        print(f"  成功周期: {summary['win_periods']} ({summary['success_rate']:.2%})")
        print(f"  平均收益: {summary['avg_return']:.2%}")
        print(f"  总交易数: {summary['total_trades']}")
        print(f"  平均每周期交易: {summary['avg_trades_per_period']:.1f}")
        print(f"  总买入信号: {summary['total_buy_signals']}")
        print(f"  总卖出信号: {summary['total_sell_signals']}")
        print(f"  最终价值: ${summary['final_portfolio_value']:.2f}")
        
        return summary

    def walk_forward_backtest(self, df: pd.DataFrame, initial_cash: float = 10000, 
                            lookback_days: int = 60, step_days: int = 7) -> Dict:
        """Walk-Forward 回测验证 - 修复版本"""
        print("📈 开始Walk-Forward回测...")
        
        results = []
        current_date = df.index.min() + timedelta(days=lookback_days)
        end_date = df.index.max()
        
        # 获取特征列名
        feature_cols = self.validator.models[0].feature_name_
        
        period_count = 0
        max_periods = 20  # 限制最大周期数
        
        while current_date <= end_date and period_count < max_periods:
            period_count += 1
            
            # 训练数据截止日期
            train_end = current_date
            # 测试数据开始日期
            test_start = current_date
            test_end = min(test_start + timedelta(days=step_days), end_date)
            
            # 分割数据
            train_mask = df.index <= train_end
            test_mask = (df.index >= test_start) & (df.index <= test_end)
            
            if test_mask.sum() == 0:
                print(f"周期 {period_count}: 无测试数据")
                break
                
            try:
                # 使用模型预测
                X_test = df[test_mask][feature_cols]
                y_proba = self.validator.ensemble_predict(X_test)
                
                # 生成交易信号
                entries = y_proba > self.validator.thresholds['buy_threshold']
                exits = y_proba < self.validator.thresholds['sell_threshold']
                
                # 检查是否有交易信号
                if entries.sum() == 0 and exits.sum() == 0:
                    print(f"周期 {period_count}: {test_start.date()} 到 {test_end.date()} - 无交易信号")
                    current_date = test_end + timedelta(days=1)
                    continue
                
                # 执行回测
                close = df.loc[test_mask, 'close']
                
                # 添加简单的交易费用
                fees = 0.001  # 0.1% 交易费用
                
                pf = vbt.Portfolio.from_signals(
                    close=close,
                    entries=entries,
                    exits=exits,
                    init_cash=initial_cash,
                    fees=fees,
                    freq='1d'  # 设置频率
                )
                
                # 安全地获取统计指标
                total_return = pf.total_return()
                sharpe_ratio = pf.sharpe_ratio() if not np.isnan(pf.sharpe_ratio()) else 0.0
                max_drawdown = pf.max_drawdown()
                total_trades = self.safe_get_stats(pf, 'total_trades', 0)
                win_rate = self.safe_get_stats(pf, 'win_rate', 0.0)
                
                # 手动计算胜率（备用方法）
                if total_trades > 0:
                    trades = pf.trades.records_readable
                    if not trades.empty and 'Return' in trades.columns:
                        winning_trades = len(trades[trades['Return'] > 0])
                        manual_win_rate = winning_trades / total_trades
                    else:
                        manual_win_rate = 0.0
                else:
                    manual_win_rate = 0.0
                
                # 使用手动计算的胜率如果自动获取失败
                if win_rate == 0.0 and manual_win_rate > 0:
                    win_rate = manual_win_rate
                
                # 安全获取最终组合价值
                try:
                    portfolio_values = pf.value()
                    final_portfolio_value = portfolio_values.iloc[-1] if len(portfolio_values) > 0 else initial_cash
                except:
                    final_portfolio_value = initial_cash
                
                period_result = {
                    'period_start': test_start,
                    'period_end': test_end,
                    'total_return': total_return,
                    'sharpe_ratio': sharpe_ratio,
                    'max_drawdown': max_drawdown,
                    'total_trades': total_trades,
                    'win_rate': win_rate,
                    'portfolio_value': final_portfolio_value
                }
                results.append(period_result)
                
                print(f"周期 {period_count}: {test_start.date()} 到 {test_end.date()}")
                print(f"  收益: {total_return:.2%}, 夏普: {sharpe_ratio:.2f}")
                print(f"  最大回撤: {max_drawdown:.2%}, 交易数: {total_trades}")
                print(f"  胜率: {win_rate:.2%}, 最终价值: ${final_portfolio_value:.2f}")
                print("-" * 50)
                
            except Exception as e:
                print(f"❌ 周期 {period_count} 回测失败: {e}")
                # 添加一个空结果继续执行
                results.append({
                    'period_start': test_start,
                    'period_end': test_end,
                    'total_return': 0.0,
                    'sharpe_ratio': 0.0,
                    'max_drawdown': 0.0,
                    'total_trades': 0,
                    'win_rate': 0.0,
                    'portfolio_value': initial_cash,
                    'error': str(e)
                })
            
            current_date = test_end + timedelta(days=1)
        
        if not results:
            print("❌ 没有有效的回测结果")
            return {
                'avg_return': 0.0,
                'std_return': 0.0,
                'avg_sharpe': 0.0,
                'win_periods': 0,
                'total_periods': 0,
                'success_rate': 0.0,
                'error': '没有有效的回测结果'
            }
        
        # 分析回测结果
        results_df = pd.DataFrame(results)
        
        # 计算成功周期（收益 > 0）
        winning_periods = (results_df['total_return'] > 0).sum()
        total_periods = len(results_df)
        
        # 安全地计算最终组合价值
        if not results_df.empty:
            # 使用 .iloc 来按位置索引，避免警告
            final_portfolio_value = results_df['portfolio_value'].iloc[-1]
        else:
            final_portfolio_value = initial_cash
        
        summary = {
            'avg_return': results_df['total_return'].mean(),
            'std_return': results_df['total_return'].std(),
            'avg_sharpe': results_df['sharpe_ratio'].mean(),
            'avg_max_drawdown': results_df['max_drawdown'].mean(),
            'avg_win_rate': results_df['win_rate'].mean(),
            'total_trades': results_df['total_trades'].sum(),
            'win_periods': winning_periods,
            'total_periods': total_periods,
            'success_rate': winning_periods / total_periods if total_periods > 0 else 0.0,
            'final_portfolio_value': final_portfolio_value
        }
        
        print(f"\n📊 Walk-Forward回测总结:")
        print(f"  总周期数: {summary['total_periods']}")
        print(f"  成功周期: {summary['win_periods']} ({summary['success_rate']:.2%})")
        print(f"  平均周期收益: {summary['avg_return']:.2%} ± {summary['std_return']:.2%}")
        print(f"  平均夏普比率: {summary['avg_sharpe']:.2f}")
        print(f"  平均最大回撤: {summary['avg_max_drawdown']:.2%}")
        print(f"  平均胜率: {summary['avg_win_rate']:.2%}")
        print(f"  总交易次数: {summary['total_trades']}")
        print(f"  最终组合价值: ${summary['final_portfolio_value']:.2f}")
        
        return summary

# 完整的验证流程
def run_complete_validation():
    """运行完整的验证流程"""
    
    try:
        # 1. 模型性能验证
        df = pd.read_parquet('data/feat_2024_8_to_2025_3.parquet')
        features = [c for c in df.columns if c not in ['y']]
        X, y = df[features], df['y']
        
        validator = ModelValidator(
            model_path='models/ensemble_models.pkl',
            thresholds_path='models/optimal_thresholds.json',
            config_path='conf/config.yml'
        )
        
        print("🎯 阶段1: 模型性能验证")
        model_results = validator.run_comprehensive_validation(X, y)
        
        print("\n🎯 阶段2: 回测验证")
        backtest_validator = BacktestValidator(validator)
        backtest_results = backtest_validator.improved_walk_forward_backtest(df)
        
        # 综合评估
        print("\n" + "=" * 50)
        print("🎯 最终验证结论")
        print("=" * 50)
        
        auc = model_results['basic_performance']['auc']
        success_rate = backtest_results.get('success_rate', 0.0)
        
        print(f"模型AUC: {auc:.4f}")
        print(f"回测成功率: {success_rate:.2%}")
        
        if auc > 0.65 and success_rate > 0.6:
            print("✅ 模型验证通过 - 建议投入实盘测试")
        elif auc > 0.6 and success_rate > 0.55:
            print("⚠️ 模型表现一般 - 建议进一步优化")
        else:
            print("❌ 模型需要重新训练 - 当前表现不佳")
        
        return {
            'model_validation': model_results,
            'backtest_validation': backtest_results
        }
        
    except Exception as e:
        print(f"❌ 验证流程失败: {e}")
        return {
            'error': str(e)
        }


# 临时调整阈值进行测试
def test_adjusted_thresholds(validator, df):
    """测试调整后的阈值"""
    print("🎯 测试调整阈值...")
    
    # 获取预测概率
    feature_cols = validator.models[0].feature_name_
    X = df[feature_cols]
    y_proba = validator.ensemble_predict(X)
    
    # 建议的阈值范围
    threshold_options = [
        (0.10, 0.05),  # 更宽松
        (0.15, 0.10),  # 适中
        (0.20, 0.15),  # 更严格
    ]
    
    for buy_thresh, sell_thresh in threshold_options:
        buy_signals = (y_proba > buy_thresh).sum()
        sell_signals = (y_proba < sell_thresh).sum()
        
        print(f"买入阈值 {buy_thresh:.2f}, 卖出阈值 {sell_thresh:.2f}:")
        print(f"  买入信号: {buy_signals} ({buy_signals/len(y_proba):.2%})")
        print(f"  卖出信号: {sell_signals} ({sell_signals/len(y_proba):.2%})")



if __name__ == '__main__':
    # df = pd.read_parquet('data/feat_2024_8_to_2025_3.parquet')
    # validator = ModelValidator(
    #     model_path='models/ensemble_models.pkl',
    #     thresholds_path='models/optimal_thresholds.json',
    #     config_path='conf/config.yml'
    # )

    # test_adjusted_thresholds(validator, df);
    results = run_complete_validation()
"""
增强版模型训练完整流程
包含数据预处理、特征工程、模型训练和评估
"""

import os
import sys
import pandas as pd
import numpy as np
import yaml
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

def run_data_preparation():
    """
    运行数据预处理
    """
    print("🚀 步骤1: 数据预处理")
    print("=" * 50)
    
    try:
        from data.prepare_enhanced import prepare_enhanced_dataset
        result = prepare_enhanced_dataset()
        if result is not None:
            print("✅ 数据预处理完成")
            return True
        else:
            print("❌ 数据预处理失败")
            return False
    except Exception as e:
        print(f"❌ 数据预处理出错: {str(e)}")
        return False

def run_model_training():
    """
    运行模型训练
    """
    print("\n🚀 步骤2: 模型训练")
    print("=" * 50)
    
    try:
        from models.train_enhanced import main as train_main
        train_main()
        print("✅ 模型训练完成")
        return True
    except Exception as e:
        print(f"❌ 模型训练出错: {str(e)}")
        return False

def run_backtest_validation():
    """
    运行回测验证
    """
    print("\n🚀 步骤3: 回测验证")
    print("=" * 50)
    
    try:
        # 检查是否有增强版模型
        if os.path.exists('models/lgb_trend_enhanced.pkl'):
            print("✅ 找到增强版模型，开始回测验证...")
            
            # 更新回测脚本使用增强版模型
            from run_backtest_no_vectorbt import SimpleBacktester
            
            # 创建回测器
            backtester = SimpleBacktester('data/feat_enhanced.parquet')
            
            # 运行回测
            portfolio_value, trades, performance = backtester.run_comprehensive_backtest()
            
            print("✅ 回测验证完成")
            return True
        else:
            print("❌ 未找到增强版模型")
            return False
    except Exception as e:
        print(f"❌ 回测验证出错: {str(e)}")
        return False

def compare_performance():
    """
    比较性能提升
    """
    print("\n🚀 步骤4: 性能比较")
    print("=" * 50)
    
    try:
        # 读取原始性能报告
        original_perf = None
        if os.path.exists('backtest/performance_report.json'):
            import json
            with open('backtest/performance_report.json', 'r') as f:
                original_perf = json.load(f)
        
        # 读取增强版性能报告
        enhanced_perf = None
        if os.path.exists('models/performance_metrics_enhanced.json'):
            import json
            with open('models/performance_metrics_enhanced.json', 'r') as f:
                enhanced_perf = json.load(f)
        
        if original_perf and enhanced_perf:
            print("📊 性能比较:")
            print(f"  原始模型:")
            print(f"    总收益率: {original_perf.get('total_return', 0):.2%}")
            print(f"    夏普比率: {original_perf.get('sharpe_ratio', 0):.2f}")
            print(f"    胜率: {original_perf.get('win_rate', 0):.2%}")
            print(f"    最大回撤: {original_perf.get('max_drawdown', 0):.2%}")
            
            print(f"  增强版模型:")
            print(f"    平均AUC: {enhanced_perf.get('mean_auc', 0):.4f}")
            print(f"    平均F1: {enhanced_perf.get('mean_f1', 0):.4f}")
            print(f"    平均精确率: {enhanced_perf.get('mean_precision', 0):.4f}")
            print(f"    平均召回率: {enhanced_perf.get('mean_recall', 0):.4f}")
            
            print("✅ 性能比较完成")
        else:
            print("⚠️ 无法进行性能比较，缺少性能数据")
        
        return True
    except Exception as e:
        print(f"❌ 性能比较出错: {str(e)}")
        return False

def main():
    """
    主函数 - 运行完整的增强版训练流程
    """
    print("🚀 开始增强版模型训练完整流程")
    print("=" * 60)
    print("📋 流程包括:")
    print("  1. 数据预处理和特征工程")
    print("  2. 增强版模型训练")
    print("  3. 回测验证")
    print("  4. 性能比较")
    print("=" * 60)
    
    # 记录开始时间
    start_time = datetime.now()
    
    # 步骤1: 数据预处理
    step1_success = run_data_preparation()
    if not step1_success:
        print("❌ 数据预处理失败，停止流程")
        return
    
    # 步骤2: 模型训练
    step2_success = run_model_training()
    if not step2_success:
        print("❌ 模型训练失败，停止流程")
        return
    
    # 步骤3: 回测验证
    step3_success = run_backtest_validation()
    if not step3_success:
        print("⚠️ 回测验证失败，但继续流程")
    
    # 步骤4: 性能比较
    step4_success = compare_performance()
    
    # 计算总耗时
    end_time = datetime.now()
    total_time = end_time - start_time
    
    # 输出最终结果
    print("\n" + "=" * 60)
    print("📊 增强版训练流程完成")
    print("=" * 60)
    
    print(f"⏱️ 总耗时: {total_time}")
    print(f"📊 流程状态:")
    print(f"  ✅ 数据预处理: {'成功' if step1_success else '失败'}")
    print(f"  ✅ 模型训练: {'成功' if step2_success else '失败'}")
    print(f"  ✅ 回测验证: {'成功' if step3_success else '失败'}")
    print(f"  ✅ 性能比较: {'成功' if step4_success else '失败'}")
    
    if step1_success and step2_success:
        print("\n🎉 增强版模型训练成功完成！")
        print("📁 生成的文件:")
        print("  - data/feat_enhanced.parquet: 增强数据")
        print("  - models/lgb_trend_enhanced.pkl: 增强模型")
        print("  - models/ensemble_models_enhanced.pkl: 集成模型")
        print("  - models/optimal_thresholds_enhanced.json: 最优阈值")
        print("  - models/performance_metrics_enhanced.json: 性能指标")
        print("  - backtest/: 回测结果")
    else:
        print("\n❌ 增强版模型训练失败")
        print("💡 请检查错误信息并重试")

if __name__ == '__main__':
    main()

"""
测试优化后的模型性能
比较原始模型和优化模型的性能差异
"""

import pandas as pd
import numpy as np
import joblib
import json
import yaml
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, precision_recall_curve, f1_score, classification_report
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

from utils.dynamic_thresholds import DynamicThresholdManager
from utils.ensemble_predictor import EnsemblePredictor, RobustSignalGenerator

def load_config():
    """加载配置"""
    with open('conf/config.yml', 'r') as f:
        return yaml.safe_load(f)

def load_data():
    """加载数据"""
    df = pd.read_parquet('data/feat.parquet')
    features = [c for c in df.columns if c not in ['y']]
    X, y = df[features], df['y']
    return X, y, features

def test_original_model():
    """测试原始模型"""
    print("🔍 测试原始模型...")
    
    try:
        # 尝试加载原始模型
        original_model = joblib.load('models/lgb_trend.pkl')
        X, y, features = load_data()
        
        # 时间序列分割
        tscv = TimeSeriesSplit(n_splits=5)
        scores = []
        
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # 使用原始模型预测
            proba = original_model.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(y_val, proba)
            scores.append(auc)
        
        avg_auc = np.mean(scores)
        std_auc = np.std(scores)
        
        print(f"📊 原始模型平均AUC: {avg_auc:.4f} ± {std_auc:.4f}")
        return {
            'model_type': 'original',
            'avg_auc': avg_auc,
            'std_auc': std_auc,
            'scores': scores
        }
        
    except Exception as e:
        print(f"❌ 原始模型测试失败: {e}")
        return None

def test_optimized_model():
    """测试优化模型"""
    print("🚀 测试优化模型...")
    
    try:
        # 加载集成模型
        ensemble_predictor = EnsemblePredictor()
        if not ensemble_predictor.load_models():
            print("❌ 集成模型加载失败")
            return None
        
        X, y, features = load_data()
        
        # 时间序列分割
        tscv = TimeSeriesSplit(n_splits=5)
        scores = []
        uncertainties = []
        confidences = []
        
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # 使用集成模型预测
            prediction_result = ensemble_predictor.predict_ensemble(X_val)
            proba = prediction_result['ensemble_proba']
            uncertainty = prediction_result['uncertainty']
            confidence = prediction_result['confidence']
            
            auc = roc_auc_score(y_val, proba)
            scores.append(auc)
            uncertainties.append(np.mean(uncertainty))
            confidences.append(np.mean(confidence))
        
        avg_auc = np.mean(scores)
        std_auc = np.std(scores)
        avg_uncertainty = np.mean(uncertainties)
        avg_confidence = np.mean(confidences)
        
        print(f"📊 优化模型平均AUC: {avg_auc:.4f} ± {std_auc:.4f}")
        print(f"📊 平均不确定性: {avg_uncertainty:.4f}")
        print(f"📊 平均置信度: {avg_confidence:.4f}")
        
        return {
            'model_type': 'optimized',
            'avg_auc': avg_auc,
            'std_auc': std_auc,
            'scores': scores,
            'avg_uncertainty': avg_uncertainty,
            'avg_confidence': avg_confidence
        }
        
    except Exception as e:
        print(f"❌ 优化模型测试失败: {e}")
        return None

def test_dynamic_thresholds():
    """测试动态阈值"""
    print("🎯 测试动态阈值...")
    
    try:
        # 加载数据
        X, y, features = load_data()
        
        # 初始化动态阈值管理器
        threshold_manager = DynamicThresholdManager()
        
        # 模拟市场数据
        prices = pd.Series(np.random.randn(100).cumsum() + 100)
        volumes = pd.Series(np.random.rand(100) * 1000)
        atr = 0.5
        current_price = prices.iloc[-1]
        
        # 测试动态阈值
        dynamic_thresholds = threshold_manager.get_dynamic_thresholds(
            current_price=current_price,
            atr=atr,
            prices=prices,
            volumes=volumes,
            signal_strength=0.8
        )
        
        print(f"📈 动态买入阈值: {dynamic_thresholds['buy_threshold']:.4f}")
        print(f"📉 动态卖出阈值: {dynamic_thresholds['sell_threshold']:.4f}")
        print(f"🌊 市场波动性: {dynamic_thresholds['market_volatility']}")
        print(f"📊 市场趋势: {dynamic_thresholds['market_trend']}")
        print(f"🔄 市场状态: {dynamic_thresholds['market_regime']}")
        
        return dynamic_thresholds
        
    except Exception as e:
        print(f"❌ 动态阈值测试失败: {e}")
        return None

def test_robust_signals():
    """测试鲁棒信号生成"""
    print("🛡️ 测试鲁棒信号生成...")
    
    try:
        # 加载数据
        X, y, features = load_data()
        
        # 初始化组件
        ensemble_predictor = EnsemblePredictor()
        if not ensemble_predictor.load_models():
            print("❌ 集成模型加载失败")
            return None
        
        robust_generator = RobustSignalGenerator(ensemble_predictor)
        
        # 使用最后100个样本测试
        test_X = X.tail(100)
        test_y = y.tail(100)
        
        signals = []
        confidences = []
        uncertainties = []
        
        for i in range(len(test_X)):
            sample_X = test_X.iloc[i:i+1]
            sample_y = test_y.iloc[i]
            
            # 生成鲁棒信号
            signal_result = robust_generator.generate_robust_signal(
                sample_X,
                base_buy_threshold=0.6,
                base_sell_threshold=0.4,
                min_confidence=0.7
            )
            
            signals.append(signal_result['side'])
            confidences.append(signal_result['strength'])
            
            # 更新性能跟踪
            robust_generator.update_performance(signal_result, sample_y)
        
        # 统计信号分布
        signal_counts = pd.Series(signals).value_counts()
        avg_confidence = np.mean(confidences)
        
        print(f"📊 信号分布: {signal_counts.to_dict()}")
        print(f"📊 平均信号强度: {avg_confidence:.4f}")
        
        # 获取性能统计
        performance = robust_generator.get_performance_stats()
        print(f"📊 性能统计: {performance}")
        
        return {
            'signal_distribution': signal_counts.to_dict(),
            'avg_confidence': avg_confidence,
            'performance': performance
        }
        
    except Exception as e:
        print(f"❌ 鲁棒信号测试失败: {e}")
        return None

def compare_models():
    """比较模型性能"""
    print("\n" + "="*60)
    print("📊 模型性能比较")
    print("="*60)
    
    # 测试原始模型
    original_results = test_original_model()
    
    # 测试优化模型
    optimized_results = test_optimized_model()
    
    if original_results and optimized_results:
        improvement = optimized_results['avg_auc'] - original_results['avg_auc']
        improvement_pct = (improvement / original_results['avg_auc']) * 100
        
        print(f"\n📈 性能提升:")
        print(f"   AUC提升: {improvement:.4f} ({improvement_pct:.2f}%)")
        print(f"   原始模型: {original_results['avg_auc']:.4f} ± {original_results['std_auc']:.4f}")
        print(f"   优化模型: {optimized_results['avg_auc']:.4f} ± {optimized_results['std_auc']:.4f}")
        
        return {
            'original': original_results,
            'optimized': optimized_results,
            'improvement': improvement,
            'improvement_pct': improvement_pct
        }
    else:
        print("❌ 无法完成模型比较")
        return None

def create_performance_report():
    """创建性能报告"""
    print("\n" + "="*60)
    print("📋 生成性能报告")
    print("="*60)
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'model_comparison': compare_models(),
        'dynamic_thresholds': test_dynamic_thresholds(),
        'robust_signals': test_robust_signals()
    }
    
    # 保存报告
    with open('models/performance_report.json', 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print("💾 性能报告已保存到 models/performance_report.json")
    
    return report

def main():
    """主测试函数"""
    print("🚀 开始优化模型性能测试...")
    
    try:
        # 创建性能报告
        report = create_performance_report()
        
        print("\n✅ 测试完成！")
        print("📁 生成的文件:")
        print("  - models/performance_report.json: 性能报告")
        
        # 打印总结
        if report['model_comparison']:
            comparison = report['model_comparison']
            print(f"\n📊 总结:")
            print(f"   AUC提升: {comparison['improvement']:.4f} ({comparison['improvement_pct']:.2f}%)")
            
            if comparison['improvement'] > 0:
                print("   ✅ 优化成功！模型性能有所提升")
            else:
                print("   ⚠️ 优化效果不明显，可能需要调整参数")
        
    except Exception as e:
        print(f"❌ 测试过程中出现错误: {e}")

if __name__ == '__main__':
    main()







# model_validation.py
import pandas as pd
import numpy as np
import joblib
import json
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    precision_recall_curve, f1_score, precision_score, recall_score
)
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, List
import yaml

class ModelValidator:
    def __init__(self, model_path: str, thresholds_path: str, config_path: str):
        self.models = joblib.load(model_path)
        with open(thresholds_path, 'r') as f:
            self.thresholds = json.load(f)
        self.config = self.load_config(config_path)
        
    def load_config(self, config_path: str):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def basic_performance_validation(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """基础性能验证 - 修复版本"""
        print("🧪 基础性能验证...")
        
        # 集成预测
        y_proba = self.ensemble_predict(X)
        
        # 安全地计算指标
        try:
            y_pred_buy = (y_proba > self.thresholds['buy_threshold']).astype(int)
            y_pred_sell = (y_proba < self.thresholds['sell_threshold']).astype(int)
            
            metrics = {
                'auc': roc_auc_score(y, y_proba),
                'f1_buy': f1_score(y, y_pred_buy, zero_division=0),
                'f1_sell': f1_score(1-y, y_pred_sell, zero_division=0),
                'precision_buy': precision_score(y, y_pred_buy, zero_division=0),
                'recall_buy': recall_score(y, y_pred_buy, zero_division=0),
                'precision_sell': precision_score(1-y, y_pred_sell, zero_division=0),
                'recall_sell': recall_score(1-y, y_pred_sell, zero_division=0),
                'buy_signals_ratio': y_pred_buy.mean(),
                'sell_signals_ratio': y_pred_sell.mean()
            }
            
        except Exception as e:
            print(f"❌ 基础性能验证失败: {e}")
            # 返回默认值
            metrics = {
                'auc': 0.5,
                'f1_buy': 0.0,
                'f1_sell': 0.0,
                'precision_buy': 0.0,
                'recall_buy': 0.0,
                'precision_sell': 0.0,
                'recall_sell': 0.0,
                'buy_signals_ratio': 0.0,
                'sell_signals_ratio': 0.0
            }
        
        print("📊 基础性能指标:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")
        
        return metrics
    def time_series_cross_validation(self, X: pd.DataFrame, y: pd.Series, n_splits: int = 5) -> Dict:
        """时间序列交叉验证"""
        print("🔄 时间序列交叉验证...")
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        cv_metrics = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
            
            y_proba = self.ensemble_predict(X_val)
            y_pred = (y_proba > self.thresholds['buy_threshold']).astype(int)
            
            fold_metrics = {
                'fold': fold + 1,
                'auc': roc_auc_score(y_val, y_proba),
                'f1': f1_score(y_val, y_pred),
                'precision': precision_score(y_val, y_pred),
                'recall': recall_score(y_val, y_pred),
                'signals_ratio': y_pred.mean()
            }
            cv_metrics.append(fold_metrics)
            
            print(f"  折叠 {fold + 1}: AUC={fold_metrics['auc']:.4f}, F1={fold_metrics['f1']:.4f}")
        
        # 计算统计信息
        cv_df = pd.DataFrame(cv_metrics)
        summary = {
            'mean_auc': cv_df['auc'].mean(),
            'std_auc': cv_df['auc'].std(),
            'mean_f1': cv_df['f1'].mean(),
            'std_f1': cv_df['f1'].std(),
            'stability_score': cv_df['auc'].mean() / (cv_df['auc'].std() + 1e-8)
        }
        
        print(f"📈 交叉验证总结:")
        print(f"  AUC: {summary['mean_auc']:.4f} ± {summary['std_auc']:.4f}")
        print(f"  F1: {summary['mean_f1']:.4f} ± {summary['std_f1']:.4f}")
        print(f"  稳定性分数: {summary['stability_score']:.4f}")
        
        return summary
    
    def threshold_sensitivity_analysis(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """阈值敏感性分析 - 修复版本"""
        print("🎯 阈值敏感性分析...")
        
        y_proba = self.ensemble_predict(X)
        
        # 调整阈值范围，确保包含实际使用的阈值
        current_threshold = self.thresholds['buy_threshold']
        
        # 动态设置阈值范围，围绕当前阈值
        threshold_min = max(0.1, current_threshold - 0.2)
        threshold_max = min(0.9, current_threshold + 0.2)
        
        thresholds = np.arange(threshold_min, threshold_max, 0.05)
        
        # 确保包含当前阈值
        if current_threshold not in thresholds:
            thresholds = np.sort(np.append(thresholds, current_threshold))
        
        results = []
        
        for threshold in thresholds:
            y_pred = (y_proba > threshold).astype(int)
            
            # 检查是否有足够的样本进行计算
            if len(np.unique(y_pred)) < 2:
                # 如果只有单一预测，跳过或设置默认值
                continue
            
            try:
                result = {
                    'threshold': threshold,
                    'f1': f1_score(y, y_pred, zero_division=0),
                    'precision': precision_score(y, y_pred, zero_division=0),
                    'recall': recall_score(y, y_pred, zero_division=0),
                    'signals_ratio': y_pred.mean(),
                    'accuracy': (y_pred == y).mean()
                }
                results.append(result)
            except Exception as e:
                print(f"⚠️ 阈值 {threshold:.4f} 计算失败: {e}")
                continue
        
        if not results:
            print("❌ 无法进行阈值分析")
            return {
                'current_threshold': current_threshold,
                'optimal_threshold': current_threshold,
                'improvement': 0.0,
                'error': '无法计算阈值敏感性'
            }
        
        results_df = pd.DataFrame(results)
        
        # 找到最优阈值
        if not results_df.empty:
            optimal_idx = results_df['f1'].idxmax()
            optimal_threshold = results_df.loc[optimal_idx, 'threshold']
            optimal_f1 = results_df.loc[optimal_idx, 'f1']
            
            # 安全地查找当前阈值的F1分数
            current_threshold_data = results_df[results_df['threshold'] == current_threshold]
            
            if not current_threshold_data.empty:
                current_f1 = current_threshold_data['f1'].iloc[0]
            else:
                # 如果当前阈值不在结果中，找到最接近的阈值
                closest_idx = (results_df['threshold'] - current_threshold).abs().idxmin()
                closest_threshold = results_df.loc[closest_idx, 'threshold']
                current_f1 = results_df.loc[closest_idx, 'f1']
                print(f"⚠️ 当前阈值 {current_threshold:.4f} 不在测试范围，使用最接近值 {closest_threshold:.4f}")
            
            improvement = optimal_f1 - current_f1
            
            print(f"🔍 当前阈值: {current_threshold:.4f} (F1={current_f1:.4f})")
            print(f"🎯 最优阈值: {optimal_threshold:.4f} (F1={optimal_f1:.4f})")
            print(f"📈 F1改进: {improvement:.4f}")
            
            # 绘制敏感性分析图
            self._plot_threshold_sensitivity(results_df, current_threshold, optimal_threshold)
            
            return {
                'current_threshold': current_threshold,
                'optimal_threshold': optimal_threshold,
                'improvement': improvement,
                'current_f1': current_f1,
                'optimal_f1': optimal_f1
            }
        else:
            print("❌ 无法计算最优阈值")
            return {
                'current_threshold': current_threshold,
                'optimal_threshold': current_threshold,
                'improvement': 0.0,
                'error': '无法计算最优阈值'
            }

    def _plot_threshold_sensitivity(self, results_df: pd.DataFrame, current_threshold: float, optimal_threshold: float):
        """绘制阈值敏感性分析图 - 辅助函数"""
        try:
            plt.figure(figsize=(12, 8))
            
            # 创建子图
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            
            # 第一个子图：性能指标
            ax1.plot(results_df['threshold'], results_df['f1'], label='F1 Score', marker='o', linewidth=2, color='blue')
            ax1.plot(results_df['threshold'], results_df['precision'], label='Precision', marker='s', alpha=0.7, color='green')
            ax1.plot(results_df['threshold'], results_df['recall'], label='Recall', marker='^', alpha=0.7, color='orange')
            
            ax1.axvline(current_threshold, color='red', linestyle='--', 
                    label=f'Current ({current_threshold:.3f})', alpha=0.8, linewidth=2)
            ax1.axvline(optimal_threshold, color='purple', linestyle='--', 
                    label=f'Optimal ({optimal_threshold:.3f})', alpha=0.8, linewidth=2)
            
            ax1.set_xlabel('Threshold')
            ax1.set_ylabel('Score')
            ax1.set_title('Threshold Sensitivity Analysis - Performance Metrics')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 第二个子图：信号比例和准确率
            ax2.plot(results_df['threshold'], results_df['signals_ratio'], 
                    label='Signals Ratio', marker='d', color='brown', linewidth=2)
            ax2.plot(results_df['threshold'], results_df['accuracy'], 
                    label='Accuracy', marker='x', color='gray', linewidth=2)
            
            ax2.axvline(current_threshold, color='red', linestyle='--', alpha=0.8)
            ax2.axvline(optimal_threshold, color='purple', linestyle='--', alpha=0.8)
            
            ax2.set_xlabel('Threshold')
            ax2.set_ylabel('Ratio / Accuracy')
            ax2.set_title('Threshold Sensitivity Analysis - Signals and Accuracy')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # 保存图片
            import os
            os.makedirs('validation', exist_ok=True)
            plt.savefig('validation/threshold_sensitivity.png', dpi=150, bbox_inches='tight')
            plt.close()
            
            print("✅ 阈值敏感性分析图已保存")
            
        except Exception as e:
            print(f"⚠️ 绘制阈值敏感性分析图失败: {e}")
        
    def feature_importance_analysis(self, X: pd.DataFrame) -> pd.DataFrame:
        """特征重要性分析"""
        print("🔍 特征重要性分析...")
        
        # 计算平均特征重要性
        importance_data = []
        for i, model in enumerate(self.models):
            importance = model.feature_importances_
            for j, (feature, imp) in enumerate(zip(X.columns, importance)):
                importance_data.append({
                    'feature': feature,
                    'importance': imp,
                    'model': f'model_{i+1}'
                })
        
        importance_df = pd.DataFrame(importance_data)
        avg_importance = importance_df.groupby('feature')['importance'].mean().sort_values(ascending=False)
        
        # 绘制特征重要性图
        plt.figure(figsize=(12, 8))
        avg_importance.head(20).plot(kind='barh')
        plt.title('Top 20 Feature Importance')
        plt.xlabel('Average Importance')
        plt.tight_layout()
        plt.savefig('validation/feature_importance.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print("🏆 Top 10 重要特征:")
        for i, (feature, importance) in enumerate(avg_importance.head(10).items()):
            print(f"  {i+1:2d}. {feature}: {importance:.4f}")
        
        return avg_importance
    
    def model_consistency_check(self, X: pd.DataFrame) -> Dict:
        """模型一致性检查"""
        print("🔄 模型一致性检查...")
        
        # 计算模型间预测的相关性
        predictions = []
        for model in self.models:
            pred = model.predict_proba(X)[:, 1]
            predictions.append(pred)
        
        predictions_array = np.array(predictions)
        correlation_matrix = np.corrcoef(predictions_array)
        
        # 计算平均相关性（排除对角线）
        mask = ~np.eye(correlation_matrix.shape[0], dtype=bool)
        avg_correlation = correlation_matrix[mask].mean()
        
        # 计算预测标准差
        pred_std = predictions_array.std(axis=0).mean()
        
        consistency_metrics = {
            'avg_model_correlation': avg_correlation,
            'avg_prediction_std': pred_std,
            'consistency_score': 1 - pred_std  # 标准差越小，一致性越好
        }
        
        print(f"📊 模型一致性指标:")
        print(f"  平均模型间相关性: {consistency_metrics['avg_model_correlation']:.4f}")
        print(f"  平均预测标准差: {consistency_metrics['avg_prediction_std']:.4f}")
        print(f"  一致性分数: {consistency_metrics['consistency_score']:.4f}")
        
        return consistency_metrics
    
    def ensemble_predict(self, X: pd.DataFrame) -> np.ndarray:
        """集成模型预测"""
        predictions = []
        for model in self.models:
            pred = model.predict_proba(X)[:, 1]
            predictions.append(pred)
        return np.mean(predictions, axis=0)
    
    def run_comprehensive_validation(self, X: pd.DataFrame, y: pd.Series, output_dir: str = 'validation'):
        """运行全面验证"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        print("🎯 开始全面模型验证...")
        print("=" * 50)
        
        validation_results = {}
        
        # 1. 基础性能验证
        validation_results['basic_performance'] = self.basic_performance_validation(X, y)
        print("-" * 30)
        
        # 2. 时间序列交叉验证
        validation_results['cross_validation'] = self.time_series_cross_validation(X, y)
        print("-" * 30)
        
        # 3. 阈值敏感性分析
        validation_results['threshold_analysis'] = self.threshold_sensitivity_analysis(X, y)
        print("-" * 30)
        
        # 4. 特征重要性分析
        validation_results['feature_importance'] = self.feature_importance_analysis(X)
        print("-" * 30)
        
        # 5. 模型一致性检查
        validation_results['model_consistency'] = self.model_consistency_check(X)
        print("-" * 30)
        
        # 保存验证结果
        self.save_validation_results(validation_results, output_dir)
        
        # 生成验证报告
        self.generate_validation_report(validation_results, output_dir)
        
        return validation_results
    
    def save_validation_results(self, results: Dict, output_dir: str):
        """保存验证结果"""
        # 保存基础指标
        basic_metrics = results['basic_performance']
        pd.DataFrame([basic_metrics]).to_csv(f'{output_dir}/basic_metrics.csv', index=False)
        
        # 保存特征重要性
        results['feature_importance'].to_csv(f'{output_dir}/feature_importance.csv')
        
        print(f"💾 验证结果已保存到 {output_dir}/ 目录")
    
    def generate_validation_report(self, results: Dict, output_dir: str):
        """生成验证报告"""
        report = []
        report.append("=" * 60)
        report.append("             模型验证报告")
        report.append("=" * 60)
        
        # 基础性能
        bp = results['basic_performance']
        report.append("\n📊 基础性能指标:")
        report.append(f"  AUC: {bp['auc']:.4f}")
        report.append(f"  买入F1: {bp['f1_buy']:.4f}")
        report.append(f"  卖出F1: {bp['f1_sell']:.4f}")
        report.append(f"  买入信号比例: {bp['buy_signals_ratio']:.4f}")
        
        # 交叉验证
        cv = results['cross_validation']
        report.append("\n🔄 交叉验证结果:")
        report.append(f"  平均AUC: {cv['mean_auc']:.4f} ± {cv['std_auc']:.4f}")
        report.append(f"  稳定性分数: {cv['stability_score']:.4f}")
        
        # 阈值分析
        ta = results['threshold_analysis']
        report.append("\n🎯 阈值分析:")
        report.append(f"  当前阈值: {ta['current_threshold']:.4f}")
        report.append(f"  建议阈值: {ta['optimal_threshold']:.4f}")
        report.append(f"  F1提升: {ta['improvement']:.4f}")
        
        # 模型一致性
        mc = results['model_consistency']
        report.append("\n🔄 模型一致性:")
        report.append(f"  平均相关性: {mc['avg_model_correlation']:.4f}")
        report.append(f"  一致性分数: {mc['consistency_score']:.4f}")
        
        # 评估结论
        report.append("\n📈 评估结论:")
        if bp['auc'] > 0.7:
            report.append("  ✅ 模型性能优秀")
        elif bp['auc'] > 0.6:
            report.append("  ⚠️ 模型性能良好")
        else:
            report.append("  ❌ 模型性能需要改进")
        
        if cv['stability_score'] > 10:
            report.append("  ✅ 模型稳定性优秀")
        else:
            report.append("  ⚠️ 模型稳定性需要关注")
        
        # 保存报告
        report_text = '\n'.join(report)
        with open(f'{output_dir}/validation_report.txt', 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print(report_text)

# 使用示例
def main():
    # 加载数据
    df = pd.read_parquet('data/feat_2025_3_to_2025_6.parquet')
    features = [c for c in df.columns if c not in ['y']]
    X, y = df[features], df['y']
    
    # 创建验证器
    validator = ModelValidator(
        model_path='models/ensemble_models.pkl',
        thresholds_path='models/optimal_thresholds.json',
        config_path='conf/config.yml'
    )
    
    # 运行全面验证
    results = validator.run_comprehensive_validation(X, y)

if __name__ == '__main__':
    main()
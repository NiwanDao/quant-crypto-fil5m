"""
模型集成预测器
提供模型集成、鲁棒性改进和不确定性量化功能
"""

import numpy as np
import pandas as pd
import joblib
from typing import List, Dict, Tuple, Optional
import json
import os
from datetime import datetime
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

class EnsemblePredictor:
    """集成预测器"""
    
    def __init__(self, models_path: str = 'models/ensemble_models.pkl'):
        self.models_path = models_path
        self.models = None
        self.feature_importance = None
        self.model_weights = None
        self.uncertainty_threshold = 0.1  # 不确定性阈值
        
    def load_models(self) -> bool:
        """加载集成模型"""
        try:
            if os.path.exists(self.models_path):
                self.models = joblib.load(self.models_path)
                print(f"✅ 成功加载 {len(self.models)} 个集成模型")
                return True
            else:
                print(f"❌ 模型文件不存在: {self.models_path}")
                return False
        except Exception as e:
            print(f"❌ 加载模型失败: {e}")
            return False
    
    def calculate_model_weights(self, X_val: pd.DataFrame, y_val: pd.Series) -> np.ndarray:
        """计算模型权重（基于验证集性能）"""
        if not self.models:
            return np.ones(len(self.models)) / len(self.models)
        
        weights = []
        for model in self.models:
            try:
                proba = model.predict_proba(X_val)[:, 1]
                auc = roc_auc_score(y_val, proba)
                weights.append(max(0.1, auc))  # 确保权重为正
            except Exception as e:
                print(f"计算模型权重时出错: {e}")
                weights.append(0.1)
        
        # 归一化权重
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        self.model_weights = weights
        return weights
    
    def predict_ensemble(self, X: pd.DataFrame, use_weights: bool = True) -> Dict:
        """集成预测"""
        if not self.models:
            raise ValueError("模型未加载，请先调用 load_models()")
        
        predictions = []
        individual_predictions = []
        
        for i, model in enumerate(self.models):
            try:
                proba = model.predict_proba(X)[:, 1]
                individual_predictions.append(proba)
                
                if use_weights and self.model_weights is not None:
                    weighted_proba = proba * self.model_weights[i]
                    predictions.append(weighted_proba)
                else:
                    predictions.append(proba)
            except Exception as e:
                print(f"模型 {i} 预测失败: {e}")
                continue
        
        if not predictions:
            raise ValueError("所有模型预测都失败了")
        
        # 计算集成预测
        if use_weights and self.model_weights is not None:
            ensemble_proba = np.sum(predictions, axis=0)
        else:
            ensemble_proba = np.mean(predictions, axis=0)
        
        # 计算不确定性（预测方差）
        individual_array = np.array(individual_predictions)
        uncertainty = np.std(individual_array, axis=0)
        
        # 计算置信度
        confidence = 1.0 - np.clip(uncertainty, 0, 1)
        
        return {
            'ensemble_proba': ensemble_proba,
            'uncertainty': uncertainty,
            'confidence': confidence,
            'individual_predictions': individual_predictions,
            'model_agreement': 1.0 - uncertainty  # 模型一致性
        }
    
    def get_feature_importance(self) -> Dict:
        """获取特征重要性"""
        if not self.models:
            return {}
        
        # 计算每个模型的特征重要性
        all_importances = []
        for model in self.models:
            try:
                importance = model.feature_importances_
                all_importances.append(importance)
            except Exception as e:
                print(f"获取特征重要性失败: {e}")
                continue
        
        if not all_importances:
            return {}
        
        # 计算平均特征重要性
        avg_importance = np.mean(all_importances, axis=0)
        std_importance = np.std(all_importances, axis=0)
        
        self.feature_importance = {
            'mean': avg_importance,
            'std': std_importance,
            'stability': 1.0 - (std_importance / (avg_importance + 1e-8))
        }
        
        return self.feature_importance
    
    def predict_with_uncertainty(self, X: pd.DataFrame, 
                               uncertainty_threshold: float = 0.1) -> Dict:
        """带不确定性的预测"""
        prediction_result = self.predict_ensemble(X)
        
        ensemble_proba = prediction_result['ensemble_proba']
        uncertainty = prediction_result['uncertainty']
        confidence = prediction_result['confidence']
        
        # 根据不确定性调整预测
        high_uncertainty_mask = uncertainty > uncertainty_threshold
        
        # 对高不确定性的预测进行保守处理
        adjusted_proba = ensemble_proba.copy()
        adjusted_proba[high_uncertainty_mask] = 0.5  # 设为中性
        
        return {
            'proba': adjusted_proba,
            'uncertainty': uncertainty,
            'confidence': confidence,
            'high_uncertainty_mask': high_uncertainty_mask,
            'adjusted_proba': adjusted_proba,
            'uncertainty_threshold': uncertainty_threshold
        }
    
    def get_model_diagnostics(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> Dict:
        """获取模型诊断信息"""
        if not self.models:
            return {}
        
        diagnostics = {
            'n_models': len(self.models),
            'model_weights': self.model_weights.tolist() if self.model_weights is not None else None,
            'feature_importance': self.get_feature_importance()
        }
        
        # 如果有真实标签，计算性能指标
        if y is not None:
            prediction_result = self.predict_ensemble(X)
            ensemble_proba = prediction_result['ensemble_proba']
            
            try:
                auc = roc_auc_score(y, ensemble_proba)
                diagnostics['auc'] = auc
            except Exception as e:
                diagnostics['auc_error'] = str(e)
        
        return diagnostics
    
    def save_ensemble_info(self, filepath: str = 'models/ensemble_info.json'):
        """保存集成模型信息"""
        info = {
            'n_models': len(self.models) if self.models else 0,
            'model_weights': self.model_weights.tolist() if self.model_weights is not None else None,
            'feature_importance': self.feature_importance,
            'uncertainty_threshold': self.uncertainty_threshold,
            'saved_at': datetime.now().isoformat()
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(info, f, indent=2)
    
    def load_ensemble_info(self, filepath: str = 'models/ensemble_info.json'):
        """加载集成模型信息"""
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                info = json.load(f)
                self.model_weights = np.array(info.get('model_weights', []))
                self.feature_importance = info.get('feature_importance', {})
                self.uncertainty_threshold = info.get('uncertainty_threshold', 0.1)

class RobustSignalGenerator:
    """鲁棒信号生成器"""
    
    def __init__(self, ensemble_predictor: EnsemblePredictor):
        self.ensemble_predictor = ensemble_predictor
        self.signal_history = []
        self.performance_tracker = {
            'total_signals': 0,
            'correct_signals': 0,
            'win_rate': 0.0,
            'recent_performance': []
        }
    
    def generate_robust_signal(self, X: pd.DataFrame, 
                             base_buy_threshold: float = 0.6,
                             base_sell_threshold: float = 0.4,
                             min_confidence: float = 0.7,
                             uncertainty_threshold: float = 0.1) -> Dict:
        """生成鲁棒交易信号"""
        
        # 获取集成预测
        prediction_result = self.ensemble_predictor.predict_with_uncertainty(
            X, uncertainty_threshold
        )
        
        proba = prediction_result['adjusted_proba']
        uncertainty = prediction_result['uncertainty']
        confidence = prediction_result['confidence']
        high_uncertainty_mask = prediction_result['high_uncertainty_mask']
        
        # 生成信号
        signal = self._generate_signal(
            proba, uncertainty, confidence, high_uncertainty_mask,
            base_buy_threshold, base_sell_threshold, min_confidence
        )
        
        # 记录信号历史
        signal_record = {
            'timestamp': datetime.now().isoformat(),
            'proba': float(proba[0]) if len(proba) > 0 else 0.0,
            'uncertainty': float(uncertainty[0]) if len(uncertainty) > 0 else 0.0,
            'confidence': float(confidence[0]) if len(confidence) > 0 else 0.0,
            'signal': signal['side'],
            'strength': signal['strength'],
            'reason': signal['reason']
        }
        
        self.signal_history.append(signal_record)
        
        # 保持历史记录在合理范围内
        if len(self.signal_history) > 1000:
            self.signal_history = self.signal_history[-500:]
        
        return signal
    
    def _generate_signal(self, proba: np.ndarray, uncertainty: np.ndarray, 
                        confidence: np.ndarray, high_uncertainty_mask: np.ndarray,
                        buy_threshold: float, sell_threshold: float, 
                        min_confidence: float) -> Dict:
        """生成交易信号"""
        
        if len(proba) == 0:
            return {
                'side': 'flat',
                'strength': 0.0,
                'reason': 'no_prediction'
            }
        
        p_up = proba[0]
        p_down = 1.0 - p_up
        conf = confidence[0]
        unc = uncertainty[0]
        
        # 检查置信度
        if conf < min_confidence:
            return {
                'side': 'flat',
                'strength': conf,
                'reason': f'low_confidence_{conf:.3f}'
            }
        
        # 检查不确定性
        if high_uncertainty_mask[0]:
            return {
                'side': 'flat',
                'strength': 1.0 - unc,
                'reason': f'high_uncertainty_{unc:.3f}'
            }
        
        # 生成信号
        if p_up > buy_threshold:
            strength = min(1.0, p_up + conf - 0.5)
            return {
                'side': 'buy',
                'strength': strength,
                'reason': f'strong_buy_signal_{p_up:.3f}'
            }
        elif p_down > sell_threshold:
            strength = min(1.0, p_down + conf - 0.5)
            return {
                'side': 'sell',
                'strength': strength,
                'reason': f'strong_sell_signal_{p_down:.3f}'
            }
        else:
            return {
                'side': 'flat',
                'strength': max(p_up, p_down),
                'reason': f'weak_signal_{p_up:.3f}'
            }
    
    def update_performance(self, signal_result: Dict, actual_outcome: Optional[bool] = None):
        """更新性能跟踪"""
        self.performance_tracker['total_signals'] += 1
        
        if actual_outcome is not None:
            if actual_outcome:
                self.performance_tracker['correct_signals'] += 1
            
            # 更新胜率
            self.performance_tracker['win_rate'] = (
                self.performance_tracker['correct_signals'] / 
                self.performance_tracker['total_signals']
            )
            
            # 记录近期表现
            self.performance_tracker['recent_performance'].append(actual_outcome)
            if len(self.performance_tracker['recent_performance']) > 100:
                self.performance_tracker['recent_performance'] = \
                    self.performance_tracker['recent_performance'][-50:]
    
    def get_performance_stats(self) -> Dict:
        """获取性能统计"""
        return self.performance_tracker.copy()

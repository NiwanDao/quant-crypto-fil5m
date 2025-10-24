"""
增强版模型训练脚本
包含改进的特征工程、平衡的标签生成、高级模型训练策略
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit, StratifiedKFold
from sklearn.metrics import roc_auc_score, precision_recall_curve, f1_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
import joblib
import os
import optuna
from typing import Tuple, Dict, List
import yaml
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 导入增强特征工程
import sys
sys.path.append('..')
from utils.enhanced_features import (
    build_enhanced_features, 
    build_enhanced_labels, 
    select_enhanced_features,
    detect_enhanced_data_leakage
)

CONF_PATH = '../conf/config.yml'
DATA_PATH = '../data/feat_2024_8_to_2025_3.parquet'

def load_conf():
    with open(CONF_PATH, 'r') as f:
        return yaml.safe_load(f)

def optimize_hyperparameters_enhanced(X: pd.DataFrame, y: pd.Series, n_trials: int = 100) -> Dict:
    """
    增强版超参数优化 - 专门针对不平衡数据优化
    """
    print("🔍 开始增强版超参数优化...")
    
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 200, 2000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 200),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 200),
            'min_child_weight': trial.suggest_float('min_child_weight', 0.001, 10.0, log=True),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', 0.5, 5.0),  # 处理不平衡数据
            'random_state': 42,
            'verbose': -1,
            'n_jobs': -1
        }
        
        # 使用时间序列交叉验证
        tscv = TimeSeriesSplit(n_splits=5)
        scores = []
        
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # 处理不平衡数据
            pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
            params['scale_pos_weight'] = pos_weight
            
            mdl = lgb.LGBMClassifier(**params)
            mdl.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                eval_metric='auc',
                callbacks=[lgb.early_stopping(50)]
            )
            
            proba = mdl.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(y_val, proba)
            scores.append(auc)
        
        return np.mean(scores)
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    
    print(f"✅ 最优AUC: {study.best_value:.4f}")
    print(f"📊 最优参数: {study.best_params}")
    
    return study.best_params

def optimize_thresholds_enhanced(y_true: np.ndarray, y_proba: np.ndarray) -> Tuple[float, float]:
    """
    增强版阈值优化 - 考虑不平衡数据
    """
    print("🎯 优化交易阈值（增强版）...")
    
    # 计算类别权重
    pos_weight = len(y_true[y_true == 0]) / len(y_true[y_true == 1])
    
    # 买入阈值: 最大化F1分数
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    best_buy_threshold = thresholds[np.argmax(f1_scores[:len(thresholds)])]
    
    # 卖出阈值: 基于分位数，考虑不平衡
    best_sell_threshold = np.percentile(y_proba, 20)  # 更保守的卖出阈值
    
    # 确保阈值合理性
    best_buy_threshold = max(0.1, min(0.9, best_buy_threshold))
    best_sell_threshold = max(0.1, min(0.9, best_sell_threshold))
    
    print(f"📈 最优买入阈值: {best_buy_threshold:.4f}")
    print(f"📉 最优卖出阈值: {best_sell_threshold:.4f}")
    print(f"⚖️ 类别权重: {pos_weight:.2f}")
    
    return best_buy_threshold, best_sell_threshold

def train_ensemble_models_enhanced(X: pd.DataFrame, y: pd.Series, best_params: Dict, n_models: int = 7) -> List:
    """
    增强版集成模型训练 - 使用更多样化的策略
    """
    print(f"🤖 训练{n_models}个增强集成模型...")
    
    models = []
    tscv = TimeSeriesSplit(n_splits=7)
    
    for seed in range(n_models):
        print(f"训练模型 {seed + 1}/{n_models}")
        
        # 为每个模型使用不同的随机种子和参数
        params = best_params.copy()
        params['random_state'] = seed * 42
        
        # 添加模型多样性
        if seed % 2 == 0:
            params['boosting_type'] = 'gbdt'
        else:
            params['boosting_type'] = 'dart'
        
        # 调整学习率增加多样性
        params['learning_rate'] *= (0.8 + 0.4 * seed / n_models)
        
        best_model, best_auc = None, -1.0
        
        for fold, (tr, va) in enumerate(tscv.split(X), start=1):
            Xtr, Xva = X.iloc[tr], X.iloc[va]
            ytr, yva = y.iloc[tr], y.iloc[va]
            
            # 处理不平衡数据
            pos_weight = len(ytr[ytr == 0]) / len(ytr[ytr == 1])
            params['scale_pos_weight'] = pos_weight
            
            mdl = lgb.LGBMClassifier(**params)
            
            mdl.fit(
                Xtr, ytr,
                eval_set=[(Xva, yva)],
                eval_metric='auc',
                callbacks=[lgb.early_stopping(50)]
            )
            
            proba = mdl.predict_proba(Xva)[:, 1]
            auc = roc_auc_score(yva, proba)
            
            if auc > best_auc:
                best_auc, best_model = auc, mdl
        
        models.append(best_model)
        print(f"模型 {seed + 1} 最佳AUC: {best_auc:.4f}")
    
    return models

def ensemble_predict_enhanced(models: List, X: pd.DataFrame) -> np.ndarray:
    """
    增强版集成预测 - 使用加权平均
    """
    predictions = []
    weights = []
    
    for i, model in enumerate(models):
        pred = model.predict_proba(X)[:, 1]
        predictions.append(pred)
        # 给后面的模型更高权重（假设它们训练得更好）
        weights.append(1.0 + i * 0.1)
    
    # 加权平均
    weights = np.array(weights)
    weights = weights / weights.sum()
    
    weighted_predictions = np.average(predictions, axis=0, weights=weights)
    return weighted_predictions

def validate_model_performance_enhanced(models: List, X: pd.DataFrame, y: pd.Series) -> Dict:
    """
    增强版模型性能验证
    """
    print("📊 进行增强版模型性能验证...")
    
    # 时间序列交叉验证
    tscv = TimeSeriesSplit(n_splits=5)
    results = []
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # 集成预测
        y_proba = ensemble_predict_enhanced(models, X_val)
        y_pred = (y_proba > 0.5).astype(int)
        
        # 计算指标
        auc = roc_auc_score(y_val, y_proba)
        f1 = f1_score(y_val, y_pred)
        precision = precision_recall_curve(y_val, y_proba)[0]
        recall = precision_recall_curve(y_val, y_proba)[1]
        
        results.append({
            'fold': fold,
            'auc': auc,
            'f1': f1,
            'precision': precision[1] if len(precision) > 1 else 0,
            'recall': recall[1] if len(recall) > 1 else 0
        })
    
    # 计算平均性能
    avg_results = {
        'mean_auc': np.mean([r['auc'] for r in results]),
        'std_auc': np.std([r['auc'] for r in results]),
        'mean_f1': np.mean([r['f1'] for r in results]),
        'std_f1': np.std([r['f1'] for r in results]),
        'mean_precision': np.mean([r['precision'] for r in results]),
        'mean_recall': np.mean([r['recall'] for r in results])
    }
    
    print(f"📈 平均AUC: {avg_results['mean_auc']:.4f} ± {avg_results['std_auc']:.4f}")
    print(f"📈 平均F1: {avg_results['mean_f1']:.4f} ± {avg_results['std_f1']:.4f}")
    print(f"📈 平均精确率: {avg_results['mean_precision']:.4f}")
    print(f"📈 平均召回率: {avg_results['mean_recall']:.4f}")
    
    return avg_results

def save_enhanced_results(best_params: Dict, thresholds: Tuple[float, float], 
                        models: List, performance: Dict, conf: Dict):
    """
    保存增强版训练结果
    """
    os.makedirs('../models', exist_ok=True)
    
    # 保存最优参数
    with open('../models/best_params_enhanced.json', 'w') as f:
        json.dump(best_params, f, indent=2)
    
    # 保存最优阈值
    thresholds_dict = {
        'buy_threshold': float(thresholds[0]),
        'sell_threshold': float(thresholds[1]),
        'optimized_at': datetime.now().isoformat()
    }
    with open('../models/optimal_thresholds_enhanced.json', 'w') as f:
        json.dump(thresholds_dict, f, indent=2)
    
    # 保存集成模型
    joblib.dump(models, '../models/ensemble_models_enhanced.pkl')
    
    # 保存主模型
    main_model = models[0]
    main_model.booster_.save_model('../models/lgb_trend_enhanced.txt')
    joblib.dump(main_model, '../models/lgb_trend_enhanced.pkl')
    
    # 保存性能指标
    with open('../models/performance_metrics_enhanced.json', 'w') as f:
        json.dump(performance, f, indent=2)
    
    # 更新配置文件
    conf['model'].update({
        'proba_threshold': float(thresholds[0]),
        'sell_threshold': float(thresholds[1]),
        'use_ensemble': True,
        'n_ensemble_models': len(models),
        'enhanced_training': True,
        'optimized_at': datetime.now().isoformat()
    })
    
    with open(CONF_PATH, 'w') as f:
        yaml.dump(conf, f, default_flow_style=False)
    
    print("💾 增强版训练结果已保存")

def main():
    print("🚀 开始增强版模型训练...")
    print("=" * 60)
    
    # 加载配置
    conf = load_conf()
    
    # 加载数据
    print("📊 加载数据...")
    df = pd.read_parquet(DATA_PATH)
    print(f"📊 原始数据: {df.shape}")
    
    # 构建增强特征
    print("\n🔧 构建增强特征...")
    df_with_features = build_enhanced_features(df)
    print(f"📊 特征构建后: {df_with_features.shape}")
    
    # 构建增强标签
    print("\n🎯 构建增强标签...")
    df_with_labels = build_enhanced_labels(df_with_features, forward_n=4, thr=0.01)
    print(f"📊 标签构建后: {df_with_labels.shape}")
    
    # 数据泄露检测
    print("\n🔍 进行数据泄露检测...")
    if not detect_enhanced_data_leakage(df_with_labels):
        print("❌ 数据泄露检测失败，停止训练")
        return
    
    # 特征选择
    print("\n🎯 进行特征选择...")
    final_df = select_enhanced_features(df_with_labels, top_k=60)
    print(f"📊 最终数据: {final_df.shape}")
    
    # 准备训练数据
    feature_cols = [c for c in final_df.columns if c != 'y']
    X = final_df[feature_cols]
    y = final_df['y']
    
    print(f"📊 特征数: {len(feature_cols)}")
    print(f"📊 样本数: {len(X)}")
    print(f"📊 标签分布: {y.value_counts(normalize=True).to_dict()}")
    
    # 1. 超参数优化
    print("\n🔍 开始超参数优化...")
    best_params = optimize_hyperparameters_enhanced(X, y, n_trials=50)
    
    # 2. 训练集成模型
    print("\n🤖 训练增强集成模型...")
    models = train_ensemble_models_enhanced(X, y, best_params, n_models=7)
    
    # 3. 模型性能验证
    print("\n📊 验证模型性能...")
    performance = validate_model_performance_enhanced(models, X, y)
    
    # 4. 阈值优化
    print("\n🎯 优化交易阈值...")
    tscv = TimeSeriesSplit(n_splits=5)
    train_idx, val_idx = list(tscv.split(X))[-1]
    X_val_for_thresh, y_val_for_thresh = X.iloc[val_idx], y.iloc[val_idx]
    
    y_proba = ensemble_predict_enhanced(models, X_val_for_thresh)
    buy_threshold, sell_threshold = optimize_thresholds_enhanced(y_val_for_thresh.values, y_proba)
    
    # 5. 保存结果
    print("\n💾 保存训练结果...")
    save_enhanced_results(best_params, (buy_threshold, sell_threshold), models, performance, conf)
    
    # 6. 最终评估
    print("\n" + "="*60)
    print("📊 增强版模型训练完成")
    print("="*60)
    
    print(f"🎯 模型性能:")
    print(f"  📈 平均AUC: {performance['mean_auc']:.4f} ± {performance['std_auc']:.4f}")
    print(f"  📈 平均F1: {performance['mean_f1']:.4f} ± {performance['std_f1']:.4f}")
    print(f"  📈 平均精确率: {performance['mean_precision']:.4f}")
    print(f"  📈 平均召回率: {performance['mean_recall']:.4f}")
    
    print(f"\n🎯 最优阈值:")
    print(f"  📈 买入阈值: {buy_threshold:.4f}")
    print(f"  📉 卖出阈值: {sell_threshold:.4f}")
    
    print(f"\n📁 生成的文件:")
    print(f"  - models/lgb_trend_enhanced.pkl: 主模型")
    print(f"  - models/ensemble_models_enhanced.pkl: 集成模型")
    print(f"  - models/optimal_thresholds_enhanced.json: 最优阈值")
    print(f"  - models/best_params_enhanced.json: 最优参数")
    print(f"  - models/performance_metrics_enhanced.json: 性能指标")
    print(f"  - conf/config.yml: 更新的配置")
    
    print("\n✅ 增强版模型训练完成！")

if __name__ == '__main__':
    main()

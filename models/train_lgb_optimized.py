"""
优化的LightGBM训练脚本 - 修复版本
包含早停机制、超参数调优、动态阈值优化和模型集成
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, precision_recall_curve, f1_score
import joblib
import os
import optuna
from typing import Tuple, Dict, List
import yaml
import json
from datetime import datetime

DATA_PATH = 'data/feat_2024_8_to_2025_3.parquet'
MODEL_PATH = 'models/lgb_trend_optimized.txt'
SKLEARN_DUMP = 'models/lgb_trend_optimized.pkl'
ENSEMBLE_MODELS_PATH = 'models/ensemble_models.pkl'
THRESHOLDS_PATH = 'models/optimal_thresholds.json'
CONF_PATH = 'conf/config.yml'

def load_conf():
    with open(CONF_PATH, 'r') as f:
        return yaml.safe_load(f)

def optimize_hyperparameters(X: pd.DataFrame, y: pd.Series, n_trials: int = 50) -> Dict:
    """使用Optuna进行超参数优化 - 修复版本"""
    print("🔍 开始超参数优化...")
    
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 100),
            'subsample': trial.suggest_float('subsample', 0.7, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 2.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 2.0),
            'random_state': 42,
            'verbose': -1,
            'n_jobs': -1
        }
        
        # 使用时间序列交叉验证
        tscv = TimeSeriesSplit(n_splits=3)
        scores = []
        
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # 修复：使用正确的训练方式
            mdl = lgb.LGBMClassifier(**params)
            mdl.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                eval_metric='logloss',
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

def optimize_thresholds(y_true: np.ndarray, y_proba: np.ndarray) -> Tuple[float, float]:
    """优化买入和卖出阈值"""
    print("🎯 优化交易阈值...")
    
    # 优化买入阈值
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    best_buy_threshold = thresholds[np.argmax(f1_scores[:len(thresholds)])]
    
    # 优化卖出阈值
    y_proba_down = 1 - y_proba
    precision, recall, thresholds = precision_recall_curve(1-y_true, y_proba_down)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    best_sell_threshold = thresholds[np.argmax(f1_scores[:len(thresholds)])]
    
    print(f"📈 最优买入阈值: {best_buy_threshold:.4f}")
    print(f"📉 最优卖出阈值: {best_sell_threshold:.4f}")
    
    return best_buy_threshold, best_sell_threshold

def train_ensemble_models(X: pd.DataFrame, y: pd.Series, best_params: Dict, n_models: int = 5) -> List:
    """训练集成模型 - 修复版本"""
    print(f"🤖 训练{n_models}个集成模型...")
    
    models = []
    tscv = TimeSeriesSplit(n_splits=5)
    
    for seed in range(n_models):
        print(f"训练模型 {seed + 1}/{n_models}")
        
        # 为每个模型使用不同的随机种子
        params = best_params.copy()
        params['random_state'] = seed * 42  # 不同的随机种子
        
        best_model, best_auc = None, -1.0
        
        for fold, (tr, va) in enumerate(tscv.split(X), start=1):
            Xtr, Xva = X.iloc[tr], X.iloc[va]
            ytr, yva = y.iloc[tr], y.iloc[va]
            
            mdl = lgb.LGBMClassifier(**params)
            
            # 修复：使用正确的参数名
            mdl.fit(
                Xtr, ytr,
                eval_set=[(Xva, yva)],
                eval_metric='logloss',
            )
            
            proba = mdl.predict_proba(Xva)[:, 1]
            auc = roc_auc_score(yva, proba)
            
            if auc > best_auc:
                best_auc, best_model = auc, mdl
        
        models.append(best_model)
        print(f"模型 {seed + 1} 最佳AUC: {best_auc:.4f}")
    
    return models

def ensemble_predict(models: List, X: pd.DataFrame) -> np.ndarray:
    """集成模型预测"""
    predictions = []
    for model in models:
        pred = model.predict_proba(X)[:, 1]
        predictions.append(pred)
    
    # 返回平均预测结果
    return np.mean(predictions, axis=0)

def save_optimization_results(best_params: Dict, thresholds: Tuple[float, float], 
                            models: List, conf: Dict):
    """保存优化结果"""
    os.makedirs('models', exist_ok=True)
    
    # 保存最优参数
    with open('models/best_params.json', 'w') as f:
        json.dump(best_params, f, indent=2)
    
    # 保存最优阈值
    thresholds_dict = {
        'buy_threshold': float(thresholds[0]),
        'sell_threshold': float(thresholds[1]),
        'optimized_at': datetime.now().isoformat()
    }
    with open(THRESHOLDS_PATH, 'w') as f:
        json.dump(thresholds_dict, f, indent=2)
    
    # 保存集成模型
    joblib.dump(models, ENSEMBLE_MODELS_PATH)
    
    # 保存主模型（第一个模型）
    main_model = models[0]
    main_model.booster_.save_model(MODEL_PATH)
    joblib.dump(main_model, SKLEARN_DUMP)
    
    # 更新配置文件
    conf['model'].update({
        'proba_threshold': float(thresholds[0]),
        'sell_threshold': float(thresholds[1]),
        'use_ensemble': True,
        'n_ensemble_models': len(models),
        'optimized_at': datetime.now().isoformat()
    })
    
    with open(CONF_PATH, 'w') as f:
        yaml.dump(conf, f, default_flow_style=False)
    
    print("💾 优化结果已保存")

def validate_model_stability(models: List, X: pd.DataFrame, y: pd.Series) -> Dict:
    """验证模型稳定性"""
    print("🔍 验证模型稳定性...")
    
    tscv = TimeSeriesSplit(n_splits=3)
    stability_metrics = []
    
    for train_idx, val_idx in tscv.split(X):
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        
        # 单个模型预测
        single_preds = []
        for model in models:
            pred = model.predict_proba(X_val)[:, 1]
            single_preds.append(pred)
        
        # 集成预测
        ensemble_pred = np.mean(single_preds, axis=0)
        
        # 计算指标
        ensemble_auc = roc_auc_score(y_val, ensemble_pred)
        
        # 计算模型间相关性
        corr_matrix = np.corrcoef(single_preds)
        avg_correlation = (np.sum(corr_matrix) - len(models)) / (len(models) * (len(models) - 1))
        
        stability_metrics.append({
            'auc': ensemble_auc,
            'diversity': 1 - avg_correlation,  # 多样性指标
            'std_auc': np.std([roc_auc_score(y_val, pred) for pred in single_preds])
        })
    
    avg_metrics = {
        'mean_auc': np.mean([m['auc'] for m in stability_metrics]),
        'mean_diversity': np.mean([m['diversity'] for m in stability_metrics]),
        'stability_score': np.mean([m['auc'] for m in stability_metrics]) * np.mean([m['diversity'] for m in stability_metrics])
    }
    
    print(f"📊 模型稳定性分析:")
    print(f"  - 平均AUC: {avg_metrics['mean_auc']:.4f}")
    print(f"  - 模型多样性: {avg_metrics['mean_diversity']:.4f}")
    print(f"  - 稳定性分数: {avg_metrics['stability_score']:.4f}")
    
    return avg_metrics

def main():
    print("🚀 开始LightGBM模型优化...")
    
    # 加载配置和数据
    conf = load_conf()
    df = pd.read_parquet(DATA_PATH)
    
    # 数据质量检查
    if df.isnull().sum().sum() > 0:
        print("⚠️ 数据包含空值，进行清理...")
        df = df.dropna()
    
    features = [c for c in df.columns if c not in ['y']]
    X, y = df[features], df['y']
    
    print(f"📊 数据概览: {len(df)} 条记录, {len(features)} 个特征")
    print(f"📈 正样本比例: {y.mean():.4f}")
    
    # 1. 超参数优化
    best_params = optimize_hyperparameters(X, y, n_trials=30)
    
    # 2. 训练集成模型
    models = train_ensemble_models(X, y, best_params, n_models=5)
    
    # 3. 模型稳定性验证
    stability_metrics = validate_model_stability(models, X, y)
    
    # 4. 阈值优化
    # 使用最后一个时间序列分割进行阈值优化
    tscv = TimeSeriesSplit(n_splits=5)
    train_idx, val_idx = list(tscv.split(X))[-1]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
    
    y_proba = ensemble_predict(models, X_val)
    buy_threshold, sell_threshold = optimize_thresholds(y_val.values, y_proba)
    
    # 5. 保存结果
    save_optimization_results(best_params, (buy_threshold, sell_threshold), models, conf)
    
    # 6. 最终评估
    print("\n" + "="*50)
    print("📊 最终模型评估")
    print("="*50)
    
    print(f"🎯 模型稳定性分数: {stability_metrics['stability_score']:.4f}")
    print(f"📈 最优买入阈值: {buy_threshold:.4f}")
    print(f"📉 最优卖出阈值: {sell_threshold:.4f}")
    
    # 计算不同阈值下的F1分数
    y_pred_buy = (y_proba > buy_threshold).astype(int)
    y_pred_sell = (y_proba < sell_threshold).astype(int)
    
    f1_buy = f1_score(y_val, y_pred_buy)
    f1_sell = f1_score(1-y_val, y_pred_sell)
    
    print(f"📈 买入信号F1分数: {f1_buy:.4f}")
    print(f"📉 卖出信号F1分数: {f1_sell:.4f}")
    
    print("\n✅ 模型优化完成！")
    print("📁 生成的文件:")
    print("  - models/lgb_trend_optimized.pkl: 主模型")
    print("  - models/ensemble_models.pkl: 集成模型")
    print("  - models/optimal_thresholds.json: 最优阈值")
    print("  - models/best_params.json: 最优参数")
    print("  - conf/config.yml: 更新的配置")

if __name__ == '__main__':
    main()
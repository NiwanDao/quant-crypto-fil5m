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
import matplotlib.pyplot as plt

DATA_PATH = "data/feat_2024_8_to_2025_3.parquet"
MODEL_PATH = "models/lgb_trend_optimized.txt"
SKLEARN_DUMP = "models/lgb_trend_optimized.pkl"
ENSEMBLE_MODELS_PATH = "models/ensemble_models.pkl"
THRESHOLDS_PATH = "models/optimal_thresholds.json"
CONF_PATH = "conf/config.yml"


def load_conf():
    with open(CONF_PATH, "r") as f:
        return yaml.safe_load(f)


def optimize_hyperparameters(X: pd.DataFrame, y: pd.Series, n_trials: int = 50) -> Dict:
    """使用Optuna进行超参数优化 - 修复版本"""
    print("🔍 开始超参数优化...")

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 20, 100),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 2.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 2.0),
            "random_state": 42,
            "verbose": -1,
            "n_jobs": -1,
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
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                eval_metric="auc",
                callbacks=[lgb.early_stopping(30)],
            )

            proba = mdl.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(y_val, proba)
            scores.append(auc)

        return np.mean(scores)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    print(f"✅ 最优AUC: {study.best_value:.4f}")
    print(f"📊 最优参数: {study.best_params}")

    return study.best_params


def optimize_thresholds(y_true: np.ndarray, y_proba: np.ndarray) -> Tuple[float, float]:
    """改进的买入/卖出阈值优化"""
    print("🎯 优化交易阈值（动态/分位数法）...")

    # 买入阈值: 最大化F1
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    best_buy_threshold = thresholds[np.argmax(f1_scores[: len(thresholds)])]

    # 卖出阈值: 取买入概率的低分位
    best_sell_threshold = np.percentile(y_proba, 30)  # 可调

    print(f"📈 最优买入阈值: {best_buy_threshold:.4f}")
    print(f"📉 最优卖出阈值: {best_sell_threshold:.4f}")

    return best_buy_threshold, best_sell_threshold


def train_ensemble_models(
    X: pd.DataFrame, y: pd.Series, best_params: Dict, n_models: int = 5
) -> List:
    """训练集成模型 - 修复版本"""
    print(f"🤖 训练{n_models}个集成模型...")

    models = []
    tscv = TimeSeriesSplit(n_splits=5)

    for seed in range(n_models):
        print(f"训练模型 {seed + 1}/{n_models}")

        # 为每个模型使用不同的随机种子
        params = best_params.copy()
        params["random_state"] = seed * 42  # 不同的随机种子

        best_model, best_auc = None, -1.0

        for fold, (tr, va) in enumerate(tscv.split(X), start=1):
            Xtr, Xva = X.iloc[tr], X.iloc[va]
            ytr, yva = y.iloc[tr], y.iloc[va]

            mdl = lgb.LGBMClassifier(**params)

            # 修复：使用正确的参数名
            mdl.fit(
                Xtr,
                ytr,
                eval_set=[(Xva, yva)],
                eval_metric="auc",  # 或 'logloss'
                callbacks=[lgb.early_stopping(30)],  # 30轮未提升就停止
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


def save_optimization_results(
    best_params: Dict, thresholds: Tuple[float, float], models: List, conf: Dict
):
    """保存优化结果"""
    os.makedirs("models", exist_ok=True)

    # 保存最优参数
    with open("models/best_params.json", "w") as f:
        json.dump(best_params, f, indent=2)

    # 保存最优阈值
    thresholds_dict = {
        "buy_threshold": float(thresholds[0]),
        "sell_threshold": float(thresholds[1]),
        "optimized_at": datetime.now().isoformat(),
    }
    with open(THRESHOLDS_PATH, "w") as f:
        json.dump(thresholds_dict, f, indent=2)

    # 保存集成模型
    joblib.dump(models, ENSEMBLE_MODELS_PATH)

    # 保存主模型（第一个模型）
    main_model = models[0]
    main_model.booster_.save_model(MODEL_PATH)
    joblib.dump(main_model, SKLEARN_DUMP)

    # 更新配置文件
    conf["model"].update(
        {
            "proba_threshold": float(thresholds[0]),
            "sell_threshold": float(thresholds[1]),
            "use_ensemble": True,
            "n_ensemble_models": len(models),
            "optimized_at": datetime.now().isoformat(),
        }
    )

    with open(CONF_PATH, "w") as f:
        yaml.dump(conf, f, default_flow_style=False)

    print("💾 优化结果已保存")


# 可视化模型退化
# 若多次训练后 stability_gap 持续扩大 → 模型逐渐无法稳定传递信号，需要重训。


# 自动监控阈值体系
# 若 mean_disagreement（分歧率）突然升高 → 市场结构变化或特征不再有效
def validate_model_stability(
    models,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    n_splits: int = 5,
    window_size: int | None = None,
    use_spearman: bool = True,
    buy_threshold: float = 0.6,
    sell_threshold: float = 0.4,
    clip_proba_eps: float = 1e-6,
) -> Dict:
    """
    面向 FIL 15m 交易的模型稳定性评估（不重训）
    - 以时间序列滚动窗口评估：集成与单模型的 AUC/PR-AUC/Brier/KS
    - 多样性：成员模型预测的相关性(秩相关) → Fisher z 平均 → 反相关 1 - r̄
    - 一致性：std_auc（单模型 AUC 的标准差）
    - 分歧率：disagreement（基于买/卖阈值的信号差异度）

    参数
    ----
    models: List[sklearn-like]
        已训练好的 LightGBM 模型列表
    X, y:
        同一时间轴下的特征与标签（索引需按时间升序）
    n_splits: int
        滚动划分的折数（与 window_size 二选一，优先 window_size）
    window_size: int | None
        验证窗口长度（样本数）；若提供，将以固定滑窗评估
    use_spearman: bool
        True 用 Spearman（秩相关），False 用 Pearson
    buy_threshold/sell_threshold: float
        用于计算分歧率（disagreement）
    clip_proba_eps: float
        概率裁剪，避免极端值造成度量不稳定

    返回
    ----
    {
      'mean_auc': float,
      'mean_diversity': float,
      'stability_score': float,           # = mean_auc * mean_diversity * (1 - mean_std_auc)
      'mean_std_auc': float,
      'mean_pr_auc': float,
      'mean_brier': float,
      'mean_ks': float,
      'mean_disagreement': float,
      'by_window': pd.DataFrame           # 各窗口明细
    }
    """
    import numpy as np
    import pandas as pd
    from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
    from scipy.stats import spearmanr, ks_2samp

    assert len(X) == len(y), "X 和 y 长度不一致"
    assert len(models) >= 2, "集成至少需要两个模型"
    assert (X.index == y.index).all(), "X 与 y 的索引（时间轴）需对齐且有序"

    # 严格按时间排序
    X = X.sort_index()
    y = y.sort_index().astype(int)

    n = len(X)
    windows = []

    if window_size is not None:
        # 固定长度滑窗：尽量覆盖全局（不重叠或允许适度重叠可自行改写）
        start_points = np.linspace(0, n - window_size, num=max(1, n_splits), dtype=int)
        for s in start_points:
            e = int(s + window_size)
            windows.append((s, e))
    else:
        # 按折数切成 n_splits 个递增时间窗口
        edges = np.linspace(0, n, num=n_splits + 1, dtype=int)
        for i in range(n_splits):
            s, e = edges[i], edges[i + 1]
            if e - s > 0:
                windows.append((s, e))

    records = []
    for w_idx, (s, e) in enumerate(windows, start=1):
        Xv, yv = X.iloc[s:e], y.iloc[s:e]
        if len(Xv) == 0:
            continue

        # 成员模型概率
        member_probs = []
        member_aucs = []

        for m in models:
            p = m.predict_proba(Xv)[:, 1]
            p = np.clip(p, clip_proba_eps, 1 - clip_proba_eps)
            member_probs.append(p)
            # 单模型 AUC（窗口内）
            try:
                member_aucs.append(roc_auc_score(yv, p))
            except ValueError:
                member_aucs.append(np.nan)

        member_probs = np.array(member_probs)  # shape: [n_models, n_samples]
        ensemble_prob = member_probs.mean(axis=0)

        # ---- 集成性能指标（窗口内）----
        try:
            auc = roc_auc_score(yv, ensemble_prob)
        except ValueError:
            auc = np.nan

        try:
            pr_auc = average_precision_score(yv, ensemble_prob)
        except ValueError:
            pr_auc = np.nan

        try:
            brier = brier_score_loss(yv, ensemble_prob)
        except ValueError:
            brier = np.nan

        # KS 统计（正负样本分布分离度），越大越好
        # 这里直接用 ensemble_prob 在正负样本上的两样本KS
        pos_scores = ensemble_prob[yv == 1]
        neg_scores = ensemble_prob[yv == 0]
        if len(pos_scores) > 0 and len(neg_scores) > 0:
            ks = ks_2samp(pos_scores, neg_scores).statistic
        else:
            ks = np.nan

        # ---- 多样性（成员间相关）----
        # 相关矩阵：Spearman（秩相关）更稳健；否则 Pearson
        n_models = member_probs.shape[0]
        corr_mat = np.zeros((n_models, n_models))
        for i in range(n_models):
            for j in range(n_models):
                if i == j:
                    corr_mat[i, j] = 1.0
                    continue
                if use_spearman:
                    r, _ = spearmanr(member_probs[i], member_probs[j])
                else:
                    # Pearson
                    xi = member_probs[i] - member_probs[i].mean()
                    xj = member_probs[j] - member_probs[j].mean()
                    denom = np.sqrt((xi**2).sum()) * np.sqrt((xj**2).sum())
                    r = float((xi * xj).sum() / denom) if denom > 0 else 0.0
                # 处理可能的 nan
                if np.isnan(r):
                    r = 0.0
                corr_mat[i, j] = np.clip(r, -0.999999, 0.999999)

        # Fisher z 变换后平均，再逆变换求平均相关
        # 仅对上三角非对角做平均
        zs, cnt = 0.0, 0
        for i in range(n_models):
            for j in range(i + 1, n_models):
                r = corr_mat[i, j]
                z = 0.5 * np.log((1 + r) / (1 - r))
                zs += z
                cnt += 1
        if cnt > 0:
            z_mean = zs / cnt
            r_mean = (np.exp(2 * z_mean) - 1) / (np.exp(2 * z_mean) + 1)
        else:
            r_mean = 1.0
        diversity = 1.0 - r_mean  # 反相关作为多样性

        # ---- 一致性（std_auc）----
        member_aucs = np.array(member_aucs, dtype=float)
        std_auc = np.nanstd(member_aucs)

        # ---- 分歧率（disagreement）----
        # 用买/卖阈值将各模型概率离散成 {+1, 0, -1} 信号，统计 pairwise 差异
        # +1: p>buy_threshold, -1: p<sell_threshold, 0: 观望
        def probs_to_signal(p):
            sig = np.zeros_like(p, dtype=int)
            sig[p > buy_threshold] = 1
            sig[p < sell_threshold] = -1
            return sig

        member_sigs = np.array(
            [probs_to_signal(p) for p in member_probs]
        )  # [n_models, n_samples]
        # 两两比较差异率（不计都为0的样本，可选项：此处纳入整体）
        disagreements = []
        for i in range(n_models):
            for j in range(i + 1, n_models):
                diff = (member_sigs[i] != member_sigs[j]).mean()
                disagreements.append(diff)
        disagreement = float(np.mean(disagreements)) if disagreements else 0.0

        records.append(
            {
                "window_idx": w_idx,
                "start": Xv.index[0],
                "end": Xv.index[-1],
                "auc": float(auc) if auc == auc else np.nan,
                "pr_auc": float(pr_auc) if pr_auc == pr_auc else np.nan,
                "brier": float(brier) if brier == brier else np.nan,
                "ks": float(ks) if ks == ks else np.nan,
                "diversity": float(diversity),
                "std_auc": float(std_auc) if std_auc == std_auc else np.nan,
                "disagreement": float(disagreement),
            }
        )

    by_win = pd.DataFrame.from_records(records)
    if by_win.empty:
        return {
            "mean_auc": np.nan,
            "mean_diversity": np.nan,
            "stability_score": np.nan,
            "mean_std_auc": np.nan,
            "mean_pr_auc": np.nan,
            "mean_brier": np.nan,
            "mean_ks": np.nan,
            "mean_disagreement": np.nan,
            "by_window": by_win,
        }

    mean_auc = by_win["auc"].mean()
    mean_div = by_win["diversity"].mean()
    mean_std_auc = by_win["std_auc"].mean()
    mean_pr_auc = by_win["pr_auc"].mean()
    mean_brier = by_win["brier"].mean()
    mean_ks = by_win["ks"].mean()
    mean_disagreement = by_win["disagreement"].mean()

    # 综合稳健度：准确(auc) × 多样(diversity) × 一致性(1-std_auc)（可按需改权重）
    stability_score = (
        (mean_auc if pd.notna(mean_auc) else 0.0)
        * (mean_div if pd.notna(mean_div) else 0.0)
        * (1.0 - (mean_std_auc if pd.notna(mean_std_auc) else 0.0))
    )

    return {
        "mean_auc": float(mean_auc),
        "mean_diversity": float(mean_div),
        "stability_score": float(stability_score),
        "mean_std_auc": float(mean_std_auc),
        "mean_pr_auc": float(mean_pr_auc),
        "mean_brier": float(mean_brier),
        "mean_ks": float(mean_ks),
        "mean_disagreement": float(mean_disagreement),
        "by_window": by_win,
    }


def save_stability_report(
    stability_model: dict,
    stability_signal: dict,
    report_path: str = "models/stability_report.csv",
):
    """
    保存并可视化稳定性损耗报告。
    比较模型层面与信号层面稳定性差异，长期追踪模型退化。
    """

    # 计算稳定性差异
    stability_gap = (
        stability_model["stability_score"] - stability_signal["stability_score"]
    )

    # 记录时间与指标
    record = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "mean_auc_model": stability_model["mean_auc"],
        "mean_div_model": stability_model["mean_diversity"],
        "mean_auc_signal": stability_signal["mean_auc"],
        "mean_div_signal": stability_signal["mean_diversity"],
        "mean_disagreement": stability_signal.get("mean_disagreement", np.nan),
        "std_auc_signal": stability_signal.get("mean_std_auc", np.nan),
        "stability_gap": stability_gap,
        "stability_model_score": stability_model["stability_score"],
        "stability_signal_score": stability_signal["stability_score"],
    }

    # 创建/追加报告文件
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    if os.path.exists(report_path):
        df = pd.read_csv(report_path)
        df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
    else:
        df = pd.DataFrame([record])

    df.to_csv(report_path, index=False)
    print(f"💾 稳定性报告已更新: {report_path}")

    # 可视化趋势（可选）
    if len(df) > 2:
        plt.figure(figsize=(10, 6))
        plt.plot(
            df["timestamp"],
            df["stability_model_score"],
            label="模型层面稳定性",
            marker="o",
        )
        plt.plot(
            df["timestamp"],
            df["stability_signal_score"],
            label="信号层面稳定性",
            marker="x",
        )
        plt.plot(
            df["timestamp"],
            df["stability_gap"],
            label="稳定性损耗",
            linestyle="--",
            color="gray",
        )
        plt.xticks(rotation=45)
        plt.xlabel("时间")
        plt.ylabel("得分 / 稳定性差")
        plt.title("📊 模型稳定性与信号一致性趋势")
        plt.legend()
        plt.tight_layout()
        plt.savefig("models/stability_trend.png")
        plt.close()
        print("📈 生成趋势图: models/stability_trend.png")


# ========= 评估函数（Val/Test通用）=========
def evaluate_split(y_true, proba, buy_th, sell_th):
    from sklearn.metrics import (
        roc_auc_score,
        average_precision_score,
        brier_score_loss,
        f1_score,
        precision_score,
        recall_score,
    )
    import numpy as np

    # 区分 + 校准
    auc = roc_auc_score(y_true, proba)
    pr_auc = average_precision_score(y_true, proba)
    brier = brier_score_loss(y_true, proba)

    # 阈值信号
    long_sig = (proba > buy_th).astype(int)
    short_sig = (proba < sell_th).astype(int)

    # 分类指标（示例：做多信号）
    f1_long = f1_score(y_true, long_sig, zero_division=0)
    prec_long = precision_score(y_true, long_sig, zero_division=0)
    rec_long = recall_score(y_true, long_sig, zero_division=0)

    # Lift@5%：取Top 5% 概率的样本
    k = max(1, int(0.05 * len(proba)))
    topk_idx = np.argsort(proba)[-k:]
    lift_at_5p = (
        float(y_true.iloc[topk_idx].mean() / y_true.mean())
        if y_true.mean() > 0
        else np.nan
    )

    # TODO: 若你有回测函数，可在这里加净收益、Sharpe、MaxDD等
    # pnl = backtest_net_metrics(long_sig, short_sig, ...)

    return {
        "auc": float(auc),
        "pr_auc": float(pr_auc),
        "brier": float(brier),
        "f1_long": float(f1_long),
        "prec_long": float(prec_long),
        "rec_long": float(rec_long),
        "lift_at_5p": float(lift_at_5p),
        # **pnl
    }


def main():
    print("🚀 开始LightGBM模型优化...")

    # 加载配置
    conf = load_conf()

    # 分别加载训练、验证、测试数据
    train_df = pd.read_parquet("data/feat_2024_8_to_2025_3.parquet")
    val_df = pd.read_parquet("data/feat_2025_3_to_2025_6.parquet")
    test_df = pd.read_parquet("data/feat_2025_6_to_now.parquet")

    # 训练集用于训练
    train_features = [c for c in train_df.columns if c not in ["y"]]
    X_train, y_train = train_df[train_features], train_df["y"]

    # 验证集用于超参数优化和早停
    val_features = [c for c in val_df.columns if c not in ["y"]]
    X_val, y_val = val_df[val_features], val_df["y"]

    # 测试集用于最终评估
    test_features = [c for c in test_df.columns if c not in ["y"]]
    X_test, y_test = test_df[test_features], test_df["y"]

    print(f"📊 训练集: {len(train_df)} 条记录")
    print(f"📊 验证集: {len(val_df)} 条记录")
    print(f"📊 测试集: {len(test_df)} 条记录")

    # 数据质量检查
    if train_df.isnull().sum().sum() > 0:
        print("⚠️ 训练数据包含空值，进行清理...")
        train_df = train_df.dropna()
    if val_df.isnull().sum().sum() > 0:
        print("⚠️ 验证数据包含空值，进行清理...")
        val_df = val_df.dropna()
    if test_df.isnull().sum().sum() > 0:
        print("⚠️ 测试数据包含空值，进行清理...")
        test_df = test_df.dropna()

    # 1. 超参数优化
    best_params = optimize_hyperparameters(X_train, y_train, n_trials=30)

    # 2. 训练集成模型
    # 用同一套最优超参数（best_params），基于时间序列交叉验证（TimeSeriesSplit）反复训练多个LightGBM模型（使用不同随机种子），并挑选出每个随机种子的最优子模型，组成一个模型集成。
    models = train_ensemble_models(X_train, y_train, best_params, n_models=5)

    # 3. 稳定性评估 —— 阶段一：模型层面（不依赖阈值）
    print("\n📘 阶段一：模型层面稳定性（结构一致性）")
    stability_model = validate_model_stability(
        models, X_train, y_train, n_splits=5, use_spearman=True
    )

    # 4. 阈值优化
    # 使用最后一个时间序列分割进行阈值优化
    tscv = TimeSeriesSplit(n_splits=5)
    train_idx, val_idx = list(tscv.split(X_train))[-1]
    X_val_for_thresh, y_val_for_thresh = X_train.iloc[val_idx], y_train.iloc[val_idx]

    y_proba = ensemble_predict(models, X_val_for_thresh)
    buy_threshold, sell_threshold = optimize_thresholds(
        y_val_for_thresh.values, y_proba
    )

    # 5. 稳定性评估 —— 阶段二：信号层面（带阈值）
    print("\n📙 阶段二：信号层面稳定性（实盘一致性）")
    stability_signal = validate_model_stability(
        models,
        X_train,
        y_train,
        n_splits=5,  # 时间序列分割数量（或用 window_size 控制窗口长度）
        use_spearman=True,  # 使用 Spearman 秩相关，抗噪更强
        buy_threshold=buy_threshold,  # 使用你刚才优化得到的买入阈值
        sell_threshold=sell_threshold,  # 使用你刚才优化得到的卖出阈值
    )

    # 8. 比较两个阶段
    # 自动生成趋势图：
    # 蓝线：模型层面稳定性（结构稳定）
    # 橙线：信号层面稳定性（实盘执行）
    # 灰虚线：两者差距（稳定性损耗）

    save_stability_report(stability_model, stability_signal)

    # 9 “最终评估”摘要
    print("\n" + "=" * 50)
    print(" 训练阶段最终摘要")
    print("=" * 50)
    print(f" 模型稳定性分数: {stability_signal['stability_score']:.4f}")
    print(f" 最优买入阈值: {buy_threshold:.4f} | 📉 最优卖出阈值: {sell_threshold:.4f}")

    # 10 验证集评估（用于对比）
    print("\n" + "=" * 30)
    print("📙 验证集评估（用于对比）")
    print("=" * 30)
    val_proba = ensemble_predict(models, X_val)
    val_metrics = evaluate_split(y_val, val_proba, buy_threshold, sell_threshold)
    print(" | ".join([f"{k}:{v:.4f}" for k, v in val_metrics.items()]))

    # 11 测试集最终评估（锁参锁阈值）
    print("\n" + "=" * 30)
    print("🎯 测试集最终评估（锁参/锁阈值）")
    print("=" * 30)
    test_proba = ensemble_predict(models, X_test)
    test_metrics = evaluate_split(y_test, test_proba, buy_threshold, sell_threshold)
    print(" | ".join([f"{k}:{v:.4f}" for k, v in test_metrics.items()]))

    # 12 Val vs Test 对比（泛化差）
    print("\n🧭 差异（Test - Val）")
    for k in val_metrics:
        if k in test_metrics:
            try:
                print(f"{k}: {test_metrics[k]-val_metrics[k]:+.4f}")
            except Exception:
                pass

    # 13. 保存优化结果（参数/阈值/集成）
    save_optimization_results(
        best_params, (buy_threshold, sell_threshold), models, conf
    )

    print("\n✅ 模型优化与评估完成！")


if __name__ == "__main__":
    main()

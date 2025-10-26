import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator, StochasticOscillator, ROCIndicator
from ta.trend import EMAIndicator, MACD, ADXIndicator, IchimokuIndicator
from ta.volatility import BollingerBands, AverageTrueRange, DonchianChannel
from ta.volume import OnBalanceVolumeIndicator, AccDistIndexIndicator
from ta.others import DailyReturnIndicator


def build_features_safe(df: pd.DataFrame) -> pd.DataFrame:
    """
    安全版本的特征工程 - 避免数据泄露
    """
    df = df.copy()
    
    # 确保按时间顺序处理
    df = df.sort_index()
    
    # 1. 基础价格特征 - 只使用历史信息
    df['returns'] = df['close'].pct_change()
    df['returns_lag1'] = df['returns'].shift(1)  # 使用滞后值
    df['high_low_ratio'] = df['high'] / df['low']
    df['open_close_ratio'] = df['close'] / df['open']
    df['body_size'] = abs(df['close'] - df['open']) / df['open']
    
    # 2. 趋势特征 - 使用滞后值避免未来信息
    for period in [5, 10, 20]:
        # 使用shift(1)确保只使用历史数据计算EMA
        df[f'ema_{period}'] = df['close'].shift(1).ewm(span=period).mean()
        df[f'price_ema_ratio_{period}'] = df['close'] / df[f'ema_{period}']
    
    # 3. MACD - 使用历史数据计算
    def safe_macd(close_series, fast=12, slow=26, signal=9):
        """安全的MACD计算"""
        ema_fast = close_series.shift(1).ewm(span=fast).mean()
        ema_slow = close_series.shift(1).ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    macd_line, signal_line, histogram = safe_macd(df['close'])
    df['macd'] = macd_line
    df['macd_signal'] = signal_line
    df['macd_histogram'] = histogram
    
    # 4. RSI - 使用历史数据计算
    def safe_rsi(series, window=14):
        """安全的RSI计算"""
        # 使用shift确保没有未来信息
        deltas = series.shift(1).diff()
        gains = deltas.where(deltas > 0, 0)
        losses = -deltas.where(deltas < 0, 0)
        
        avg_gains = gains.rolling(window=window, min_periods=1).mean()
        avg_losses = losses.rolling(window=window, min_periods=1).mean()
        
        rs = avg_gains / (avg_losses + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    for period in [7, 14]:
        df[f'rsi_{period}_safe'] = safe_rsi(df['close'], period)
    
    # 5. 布林带 - 使用历史数据
    def safe_bollinger_bands(close_series, window=20, num_std=2):
        """安全的布林带计算"""
        rolling_mean = close_series.shift(1).rolling(window=window).mean()
        rolling_std = close_series.shift(1).rolling(window=window).std()
        
        upper_band = rolling_mean + (rolling_std * num_std)
        lower_band = rolling_mean - (rolling_std * num_std)
        
        return upper_band, lower_band, rolling_mean
    
    bb_upper, bb_lower, bb_middle = safe_bollinger_bands(df['close'])
    df['bb_upper_safe'] = bb_upper
    df['bb_lower_safe'] = bb_lower
    df['bb_middle_safe'] = bb_middle
    df['bb_position_safe'] = (df['close'] - bb_lower) / (bb_upper - bb_lower)
    
    # 6. 成交量特征 - 使用滞后值
    df['volume_lag1'] = df['volume'].shift(1)
    df['volume_ratio_safe'] = df['volume_lag1'] / df['volume_lag1'].rolling(20).mean()
    
    # 7. 价格位置特征 - 使用历史高低点
    df['resistance_20'] = df['high'].shift(1).rolling(20).max()  # 使用shift(1)
    df['support_20'] = df['low'].shift(1).rolling(20).min()     # 使用shift(1)
    df['breakout_high'] = (df['close'] > df['resistance_20']).astype(int)
    df['breakout_low'] = (df['close'] < df['support_20']).astype(int)
    
    # 清理数据
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.dropna()
    
    print(f"✅ 安全特征构建完成: {len(df.columns)} 个特征")
    print(f"📊 有效样本数: {len(df)}")
    
    return df


def detect_data_leakage(df, target_col='y'):
    """检测数据泄露"""
    print("🔍 检测数据泄露...")
    
    # 检查特征与标签的相关性
    feature_cols = [col for col in df.columns if col != target_col]
    correlations = []
    
    for col in feature_cols:
        corr = df[col].corr(df[target_col])
        correlations.append((col, corr))
    
    # 排序并显示高相关性特征
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)
    
    print("📊 特征与标签相关性Top 10:")
    suspicious_features = []
    for col, corr in correlations[:10]:
        print(f"  {col}: {corr:.4f}")
        if abs(corr) > 0.8:
            suspicious_features.append((col, corr))
    
    if suspicious_features:
        print("❌ 发现可疑的高相关性特征（可能数据泄露）:")
        for col, corr in suspicious_features:
            print(f"  ⚠️ {col}: {corr:.4f}")
        return False
    else:
        print("✅ 未发现明显数据泄露")
        return True

def build_labels_safe(df: pd.DataFrame, forward_n: int = 4, thr: float = 0.008) -> pd.DataFrame:
    """
    修复版标签生成 - 创建更合理的交易信号
    """
    df = df.copy()
    
    # 方法1: 简单收益率标签 (保持兼容性)
    future_return = df['close'].pct_change(forward_n).shift(-forward_n)
    df['y_simple'] = (future_return > thr).astype(int)
    
    # 方法2: 改进的波动率调整标签
    volatility = df['returns'].rolling(forward_n).std()
    adaptive_threshold = thr * (volatility / volatility.median())  # 根据波动率调整阈值
    df['y_vol_adjusted'] = (future_return > adaptive_threshold).astype(int)
    
    # 方法3: 相对强度标签 (推荐)
    future_max = df['close'].shift(-forward_n).rolling(forward_n).max()
    future_min = df['close'].shift(-forward_n).rolling(forward_n).min()
    
    # 买入信号: 未来最高价比当前价上涨超过阈值
    buy_signal = (future_max / df['close'] - 1) > thr
    
    # 卖出信号: 未来最低价比当前价下跌超过阈值  
    sell_signal = (1 - future_min / df['close']) > thr
    
    # 三分类标签
    df['y_tri'] = 0  # 0: 持有
    df.loc[buy_signal & ~sell_signal, 'y_tri'] = 1  # 1: 买入
    df.loc[sell_signal & ~buy_signal, 'y_tri'] = 2  # 2: 卖出
    
    # 使用三分类中的买入信号作为主要标签
    df['y'] = (df['y_tri'] == 1).astype(int)
    
    # 删除未来数据不可用的行
    df = df.dropna()
    
    # 分析标签分布
    print("🎯 标签分布分析:")
    print(f"  总样本数: {len(df)}")
    print(f"  买入信号 (y=1): {(df['y'] == 1).sum()} ({(df['y'] == 1).mean():.2%})")
    print(f"  非买入信号 (y=0): {(df['y'] == 0).sum()} ({(df['y'] == 0).mean():.2%})")
    
    if (df['y'] == 1).mean() > 0.7 or (df['y'] == 1).mean() < 0.3:
        print("⚠️  警告: 标签严重不平衡，建议调整阈值!")
        print(f"💡 建议阈值范围: {thr * 0.5:.4f} - {thr * 2:.4f}")
    
    return df


def build_multi_feature_labels(df: pd.DataFrame, forward_n: int = 4, base_thr: float = 0.008) -> pd.DataFrame:
    """
    基于多个feature的智能标签生成系统
    
    参数:
    - df: 包含特征的DataFrame
    - forward_n: 向前预测的周期数
    - base_thr: 基础阈值
    """
    df = df.copy()
    
    print("🚀 开始基于多特征的智能标签生成...")
    
    # 1. 计算未来收益率
    future_return = df['close'].pct_change(forward_n).shift(-forward_n)
    
    # 2. 特征权重计算 - 基于历史表现
    feature_weights = calculate_feature_weights(df, future_return)
    
    # 3. 动态阈值计算
    dynamic_thresholds = calculate_dynamic_thresholds(df, future_return, base_thr)
    
    # 4. 多特征综合评分
    composite_scores = calculate_composite_scores(df, feature_weights)
    
    # 5. 生成多种标签方法
    labels = generate_ensemble_labels(df, future_return, composite_scores, dynamic_thresholds)
    
    # 6. 标签融合和优化
    final_labels = optimize_labels(df, labels, future_return)
    
    # 7. 添加所有标签到DataFrame
    for label_name, label_values in final_labels.items():
        df[label_name] = label_values
    
    # 删除包含NaN的行
    df = df.dropna()
    
    # 8. 标签质量分析
    analyze_label_quality(df, future_return)
    
    print("✅ 多特征标签生成完成!")
    return df


def calculate_feature_weights(df: pd.DataFrame, future_return: pd.Series) -> dict:
    """
    计算特征权重 - 基于与未来收益率的相关性
    """
    print("📊 计算特征权重...")
    
    # 获取所有特征列（排除价格和标签相关列）
    feature_cols = [col for col in df.columns 
                   if col not in ['open', 'high', 'low', 'close', 'volume', 'returns'] 
                   and not col.startswith('y_')]
    
    weights = {}
    correlations = []
    
    for col in feature_cols:
        if col in df.columns:
            # 计算与未来收益率的相关性
            corr = df[col].corr(future_return)
            if not np.isnan(corr):
                weights[col] = abs(corr)  # 使用绝对值
                correlations.append((col, corr))
    
    # 归一化权重
    total_weight = sum(weights.values())
    if total_weight > 0:
        weights = {k: v/total_weight for k, v in weights.items()}
    
    # 显示Top 10权重特征
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)
    print("🏆 Top 10 特征权重:")
    for i, (col, corr) in enumerate(correlations[:10]):
        weight = weights.get(col, 0)
        print(f"  {i+1:2d}. {col}: 权重={weight:.4f}, 相关性={corr:.4f}")
    
    return weights


def calculate_dynamic_thresholds(df: pd.DataFrame, future_return: pd.Series, base_thr: float) -> dict:
    """
    计算动态阈值 - 根据市场状态调整
    """
    print("🎯 计算动态阈值...")
    
    # 计算市场状态指标
    volatility = df['returns'].rolling(20).std()
    trend_strength = abs(df['returns'].rolling(20).mean())
    volume_ratio = df['volume'] / df['volume'].rolling(20).mean()
    
    # 市场状态分类
    high_vol = volatility > volatility.quantile(0.7)
    strong_trend = trend_strength > trend_strength.quantile(0.7)
    high_volume = volume_ratio > volume_ratio.quantile(0.7)
    
    # 动态阈值计算
    thresholds = {}
    
    # 基础阈值
    thresholds['base'] = base_thr
    
    # 高波动率时降低阈值
    thresholds['vol_adjusted'] = np.where(high_vol, base_thr * 0.7, base_thr)
    
    # 强趋势时提高阈值
    thresholds['trend_adjusted'] = np.where(strong_trend, base_thr * 1.3, base_thr)
    
    # 高成交量时降低阈值
    thresholds['volume_adjusted'] = np.where(high_volume, base_thr * 0.8, base_thr)
    
    # 综合调整
    thresholds['composite'] = (thresholds['vol_adjusted'] + 
                              thresholds['trend_adjusted'] + 
                              thresholds['volume_adjusted']) / 3
    
    print(f"📈 动态阈值范围: {np.min(thresholds['composite']):.4f} - {np.max(thresholds['composite']):.4f}")
    
    return thresholds


def calculate_composite_scores(df: pd.DataFrame, feature_weights: dict) -> pd.Series:
    """
    计算多特征综合评分
    """
    print("🧮 计算综合评分...")
    
    composite_score = pd.Series(0, index=df.index)
    
    for feature, weight in feature_weights.items():
        if feature in df.columns:
            # 标准化特征值
            feature_values = df[feature]
            if feature_values.std() > 0:
                normalized_values = (feature_values - feature_values.mean()) / feature_values.std()
                composite_score += normalized_values * weight
    
    # 归一化到0-1范围
    if composite_score.std() > 0:
        composite_score = (composite_score - composite_score.min()) / (composite_score.max() - composite_score.min())
    
    print(f"📊 综合评分范围: {composite_score.min():.4f} - {composite_score.max():.4f}")
    
    return composite_score


def generate_ensemble_labels(df: pd.DataFrame, future_return: pd.Series, 
                           composite_scores: pd.Series, thresholds: dict) -> dict:
    """
    生成集成标签 - 多种方法组合
    """
    print("🎭 生成集成标签...")
    
    labels = {}
    
    # 方法1: 基于综合评分的标签
    score_threshold = composite_scores.quantile(0.7)  # 前30%作为买入信号
    labels['y_score_based'] = (composite_scores > score_threshold).astype(int)
    
    # 方法2: 动态阈值标签
    labels['y_dynamic'] = (future_return > thresholds['composite']).astype(int)
    
    # 方法3: 多条件组合标签
    # 条件1: 综合评分高
    condition1 = composite_scores > composite_scores.quantile(0.6)
    # 条件2: 未来收益率超过动态阈值
    condition2 = future_return > thresholds['composite']
    # 条件3: 技术指标支持（RSI不在超买区间）
    rsi_cols = [col for col in df.columns if 'rsi' in col.lower()]
    condition3 = True
    if rsi_cols:
        rsi_values = df[rsi_cols[0]]  # 使用第一个RSI
        condition3 = rsi_values < 80  # 不在超买区间
    
    labels['y_multi_condition'] = (condition1 & condition2 & condition3).astype(int)
    
    # 方法4: 相对强度标签
    future_max = df['close'].shift(-4).rolling(4).max()
    future_min = df['close'].shift(-4).rolling(4).min()
    
    buy_signal = (future_max / df['close'] - 1) > thresholds['composite']
    sell_signal = (1 - future_min / df['close']) > thresholds['composite']
    
    labels['y_relative_strength'] = (buy_signal & ~sell_signal).astype(int)
    
    # 方法5: 波动率调整标签
    volatility = df['returns'].rolling(4).std()
    vol_adjusted_threshold = thresholds['composite'] * (volatility / volatility.median())
    labels['y_vol_adjusted'] = (future_return > vol_adjusted_threshold).astype(int)
    
    print(f"📊 生成了 {len(labels)} 种标签方法")
    
    return labels


def optimize_labels(df: pd.DataFrame, labels: dict, future_return: pd.Series) -> dict:
    """
    优化标签质量 - 选择最佳组合
    """
    print("🔧 优化标签质量...")
    
    optimized_labels = {}
    
    # 计算每种标签的质量指标
    label_metrics = {}
    
    for label_name, label_values in labels.items():
        if len(label_values.dropna()) > 0:
            # 计算准确率（简化版）
            accuracy = (label_values == (future_return > 0)).mean()
            
            # 计算标签分布
            positive_ratio = label_values.mean()
            
            # 计算与未来收益率的相关性
            correlation = label_values.corr(future_return)
            
            label_metrics[label_name] = {
                'accuracy': accuracy,
                'positive_ratio': positive_ratio,
                'correlation': correlation,
                'balance_score': 1 - abs(positive_ratio - 0.5) * 2  # 平衡性评分
            }
    
    # 选择最佳标签组合
    best_labels = []
    for label_name, metrics in label_metrics.items():
        # 综合评分 = 准确率 * 相关性 * 平衡性
        composite_score = (metrics['accuracy'] * 
                          abs(metrics['correlation']) * 
                          metrics['balance_score'])
        
        if composite_score > 0.1:  # 最低质量阈值
            best_labels.append((label_name, composite_score))
    
    # 按评分排序
    best_labels.sort(key=lambda x: x[1], reverse=True)
    
    print("🏆 最佳标签方法:")
    for i, (label_name, score) in enumerate(best_labels[:5]):
        metrics = label_metrics[label_name]
        print(f"  {i+1}. {label_name}: 综合评分={score:.4f}")
        print(f"     准确率={metrics['accuracy']:.4f}, 相关性={metrics['correlation']:.4f}")
        print(f"     正样本比例={metrics['positive_ratio']:.4f}")
    
    # 生成最终标签
    if best_labels:
        # 使用最佳标签作为主标签
        best_label_name = best_labels[0][0]
        optimized_labels['y'] = labels[best_label_name]
        
        # 保留所有标签供选择
        for label_name, _ in best_labels:
            optimized_labels[label_name] = labels[label_name]
    
    return optimized_labels


def analyze_label_quality(df: pd.DataFrame, future_return: pd.Series):
    """
    分析标签质量
    """
    print("\n📊 标签质量分析:")
    print("=" * 50)
    
    label_cols = [col for col in df.columns if col.startswith('y')]
    
    for col in label_cols:
        if col in df.columns:
            label_values = df[col]
            if len(label_values.dropna()) > 0:
                # 基本统计
                total_samples = len(label_values)
                positive_samples = (label_values == 1).sum()
                positive_ratio = positive_samples / total_samples
                
                # 与未来收益率的相关性
                correlation = label_values.corr(future_return)
                
                # 预测准确性（简化版）
                accuracy = (label_values == (future_return > 0)).mean()
                
                print(f"\n🎯 {col}:")
                print(f"  总样本数: {total_samples}")
                print(f"  正样本数: {positive_samples} ({positive_ratio:.2%})")
                print(f"  与未来收益率相关性: {correlation:.4f}")
                print(f"  预测准确性: {accuracy:.4f}")
                
                # 质量评估
                if abs(correlation) > 0.1 and 0.3 < positive_ratio < 0.7:
                    print(f"  ✅ 标签质量: 良好")
                elif abs(correlation) > 0.05:
                    print(f"  ⚠️  标签质量: 一般")
                else:
                    print(f"  ❌ 标签质量: 较差")
    
    print("\n" + "=" * 50)

# 特征选择函数 - 删除低质量特征
def select_important_features(df, target_col='y', top_k=30):
    """
    选择最重要的特征
    """
    from sklearn.ensemble import RandomForestClassifier
    
    X = df.drop(columns=[target_col, 'y_tri', 'y_simple', 'y_vol_adjusted'], errors='ignore')
    y = df[target_col]
    
    # 使用随机森林进行特征重要性排序
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X.fillna(0), y)
    
    # 获取特征重要性
    importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # 选择top_k特征
    selected_features = importance_df.head(top_k)['feature'].tolist()
    
    print(f"🎯 选择了 {len(selected_features)} 个最重要特征")
    print("🏆 Top 10 特征:")
    for i, row in importance_df.head(10).iterrows():
        print(f"  {i+1:2d}. {row['feature']}: {row['importance']:.4f}")
    
    return df[selected_features + [target_col]]

# =============================================================================
# 使用示例和说明
# =============================================================================

def demo_multi_feature_labeling():
    """
    演示基于多特征的标签生成系统
    """
    print("🚀 多特征标签生成系统演示")
    print("=" * 60)
    
    # 示例用法
    usage_example = """
    # 1. 生成基础特征
    df_with_features = build_features_safe(raw_data)
    
    # 2. 使用多特征标签生成系统
    df_with_multi_labels = build_multi_feature_labels(
        df_with_features, 
        forward_n=4,        # 预测未来4个周期
        base_thr=0.008      # 基础阈值0.8%
    )
    
    # 3. 特征选择（可选）
    final_df = select_important_features(df_with_multi_labels, top_k=25)
    
    # 4. 查看结果
    print(f"🎉 最终数据集: {final_df.shape}")
    print(f"📊 主标签分布: {final_df['y'].value_counts(normalize=True).to_dict()}")
    
    # 5. 比较不同标签方法
    label_cols = [col for col in final_df.columns if col.startswith('y')]
    print("\\n📈 不同标签方法对比:")
    for col in label_cols:
        if col in final_df.columns:
            ratio = final_df[col].mean()
            print(f"  {col}: {ratio:.2%} 正样本比例")
    """
    
    print("📖 使用示例:")
    print(usage_example)
    
    print("\n🔧 系统特性:")
    print("  ✅ 自动特征权重计算")
    print("  ✅ 动态阈值调整")
    print("  ✅ 多方法标签集成")
    print("  ✅ 标签质量评估")
    print("  ✅ 避免数据泄露")
    
    print("\n📊 支持的标签方法:")
    print("  🎯 y_score_based: 基于综合评分的标签")
    print("  🎯 y_dynamic: 动态阈值标签")
    print("  🎯 y_multi_condition: 多条件组合标签")
    print("  🎯 y_relative_strength: 相对强度标签")
    print("  🎯 y_vol_adjusted: 波动率调整标签")
    
    print("\n💡 参数调优建议:")
    print("  📈 forward_n: 3-7 (预测周期)")
    print("  📈 base_thr: 0.005-0.015 (基础阈值)")
    print("  📈 top_k: 20-40 (特征选择数量)")


def compare_labeling_methods(df: pd.DataFrame, forward_n: int = 4, base_thr: float = 0.008):
    """
    比较不同标签生成方法的效果
    
    参数:
    - df: 包含特征的DataFrame
    - forward_n: 预测周期
    - base_thr: 基础阈值
    """
    print("🔍 标签方法对比分析")
    print("=" * 50)
    
    # 生成不同方法的标签
    methods = {
        '传统方法': build_labels_safe(df, forward_n, base_thr),
        '多特征方法': build_multi_feature_labels(df, forward_n, base_thr)
    }
    
    comparison_results = {}
    
    for method_name, labeled_df in methods.items():
        print(f"\n📊 {method_name} 结果:")
        
        # 基本统计
        total_samples = len(labeled_df)
        positive_samples = (labeled_df['y'] == 1).sum()
        positive_ratio = positive_samples / total_samples
        
        # 计算未来收益率用于评估
        future_return = labeled_df['close'].pct_change(forward_n).shift(-forward_n)
        
        # 相关性分析
        correlation = labeled_df['y'].corr(future_return)
        
        # 准确率（简化版）
        accuracy = (labeled_df['y'] == (future_return > 0)).mean()
        
        comparison_results[method_name] = {
            'total_samples': total_samples,
            'positive_ratio': positive_ratio,
            'correlation': correlation,
            'accuracy': accuracy
        }
        
        print(f"  总样本数: {total_samples}")
        print(f"  正样本比例: {positive_ratio:.2%}")
        print(f"  与未来收益率相关性: {correlation:.4f}")
        print(f"  预测准确率: {accuracy:.4f}")
    
    # 推荐最佳方法
    print("\n🏆 方法推荐:")
    best_method = max(comparison_results.items(), 
                     key=lambda x: abs(x[1]['correlation']) * x[1]['accuracy'])
    
    print(f"  推荐方法: {best_method[0]}")
    print(f"  综合评分: {abs(best_method[1]['correlation']) * best_method[1]['accuracy']:.4f}")
    
    return comparison_results


# 使用建议
if __name__ == "__main__":
    # 演示多特征标签生成系统
    demo_multi_feature_labeling()
    
    print("\n" + "=" * 60)
    print("📝 完整使用流程:")
    print("""
    # 1. 生成特征
    df_with_features = build_features_safe(raw_data)
    
    # 2. 生成多特征标签
    df_with_multi_labels = build_multi_feature_labels(
        df_with_features, 
        forward_n=4,        # 预测未来4个周期
        base_thr=0.008      # 基础阈值0.8%
    )
    
    # 3. 特征选择
    final_df = select_important_features(df_with_multi_labels, top_k=25)
    
    # 4. 数据泄露检测
    detect_data_leakage(final_df, target_col='y')
    
    # 5. 方法对比（可选）
    comparison_results = compare_labeling_methods(df_with_features)
    
    print(f"🎉 最终数据集: {final_df.shape}")
    print(f"📊 标签分布: {final_df['y'].value_counts(normalize=True).to_dict()}")
    """)    
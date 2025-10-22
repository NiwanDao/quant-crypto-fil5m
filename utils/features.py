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

def build_labels_safe(df: pd.DataFrame, forward_n: int = 4, thr: float = 0.01) -> pd.DataFrame:
    """
    安全版本的标签生成 - 确保没有数据泄露
    """
    df = df.copy()
    
    # 确保时间顺序
    df = df.sort_index()
    
    # 计算未来收益率 - 使用shift确保时间正确
    future_return = (df['close'].shift(-forward_n) - df['close']) / df['close']
    
    # 创建二元标签
    df['y'] = (future_return > thr).astype(int)
    
    # 删除无法计算未来收益的行（最后forward_n行）
    df = df.iloc[:-forward_n] if forward_n > 0 else df
    
    # 分析标签分布
    label_dist = df['y'].value_counts(normalize=True)
    print("🎯 安全标签分布:")
    print(f"  买入信号 (y=1): {label_dist.get(1, 0):.2%}")
    print(f"  非买入信号 (y=0): {label_dist.get(0, 0):.2%}")
    
    if label_dist.get(1, 0) > 0.7 or label_dist.get(1, 0) < 0.3:
        print("⚠️ 标签分布不平衡，建议调整阈值")
    
    return df.dropna()

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

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    修复版特征工程 - 专注于高质量、有预测力的特征
    """
    df = df.copy()
    
    # 1. 基础价格特征
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    df['high_low_ratio'] = df['high'] / df['low']
    df['open_close_ratio'] = df['close'] / df['open']
    df['body_size'] = abs(df['close'] - df['open']) / df['open']  # K线实体大小
    
    # 2. 趋势特征 - 减少冗余，提高质量
    # EMA系列
    for period in [5, 10, 20]:
        ema = EMAIndicator(df['close'], window=period)
        df[f'ema_{period}'] = ema.ema_indicator()
        df[f'price_ema_ratio_{period}'] = df['close'] / df[f'ema_{period}']
    
    # MACD - 只保留核心信号
    macd = MACD(df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_histogram'] = macd.macd_diff()  # MACD柱状图
    
    # ADX - 趋势强度
    adx = ADXIndicator(df['high'], df['low'], df['close'], window=14)
    df['adx'] = adx.adx()
    df['adx_pos'] = adx.adx_pos()
    df['adx_neg'] = adx.adx_neg()
    
    # 3. 动量特征 - 精选有效的动量指标
    # RSI多时间框架
    for period in [7, 14]:
        rsi = RSIIndicator(df['close'], window=period)
        df[f'rsi_{period}'] = rsi.rsi()
    
    # 随机指标KDJ - 修复计算
    stoch = StochasticOscillator(high=df['high'], low=df['low'], close=df['close'], window=14, smooth_window=3)
    df['stoch_k'] = stoch.stoch()
    df['stoch_d'] = stoch.stoch_signal()
    df['stoch_j'] = 3 * df['stoch_k'] - 2 * df['stoch_d']
    
    # 价格动量
    for period in [3, 5, 10]:
        roc = ROCIndicator(df['close'], window=period)
        df[f'roc_{period}'] = roc.roc()
    
    # 4. 波动率特征
    # 布林带
    bb = BollingerBands(df['close'], window=20, window_dev=2)
    df['bb_upper'] = bb.bollinger_hband()
    df['bb_lower'] = bb.bollinger_lband()
    df['bb_middle'] = bb.bollinger_mavg()
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    
    # ATR
    atr = AverageTrueRange(df['high'], df['low'], df['close'], window=14)
    df['atr'] = atr.average_true_range()
    df['atr_ratio'] = df['atr'] / df['close']  # 标准化ATR
    
    # 唐奇安通道
    dc = DonchianChannel(high=df['high'], low=df['low'], close=df['close'], window=20)
    df['dc_upper'] = dc.donchian_channel_hband()
    df['dc_lower'] = dc.donchian_channel_lband()
    df['dc_position'] = (df['close'] - df['dc_lower']) / (df['dc_upper'] - df['dc_lower'])
    
    # 5. 成交量特征 - 修复成交量指标
    # OBV
    obv = OnBalanceVolumeIndicator(close=df['close'], volume=df['volume'])
    df['obv'] = obv.on_balance_volume()
    df['obv_ema_10'] = df['obv'].ewm(span=10).mean()
    df['obv_ema_20'] = df['obv'].ewm(span=20).mean()
    df['obv_slope'] = df['obv'].diff(3)  # OBV的3期斜率
    
    # 累积派发线
    adi = AccDistIndexIndicator(high=df['high'], low=df['low'], close=df['close'], volume=df['volume'])
    df['adi'] = adi.acc_dist_index()
    
    # 成交量动量
    df['volume_sma_5'] = df['volume'].rolling(5).mean()
    df['volume_sma_20'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume_sma_5'] / df['volume_sma_20']
    df['volume_zscore'] = (df['volume'] - df['volume'].rolling(20).mean()) / df['volume'].rolling(20).std()
    
    # 6. 价格-成交量关系
    df['price_volume_corr'] = df['close'].rolling(10).corr(df['volume'])  # 价量相关性
    df['close_vwap'] = df['close'] / (df['volume'] * df['close']).rolling(5).sum() / df['volume'].rolling(5).sum()
    
    # 7. 市场结构特征
    # 支撑阻力突破
    df['resistance_20'] = df['high'].rolling(20).max()
    df['support_20'] = df['low'].rolling(20).min()
    df['breakout_high'] = (df['close'] > df['resistance_20'].shift(1)).astype(int)
    df['breakout_low'] = (df['close'] < df['support_20'].shift(1)).astype(int)
    
    # 价格位置特征
    df['day_range'] = (df['high'] - df['low']) / df['open']
    df['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
    
    # 8. 特征衍生 - 创建交互特征
    df['rsi_volume_interaction'] = df['rsi_14'] * df['volume_ratio']  # RSI与成交量的交互
    df['macd_vol_interaction'] = df['macd_histogram'] * df['volume_zscore']  # MACD与成交量Z-score的交互
    
    # 9. 技术形态特征
    # 价格与均线的关系
    df['above_ema_20'] = (df['close'] > df['ema_20']).astype(int)
    df['ema_slope_20'] = df['ema_20'].diff(3) / df['ema_20'].shift(3)  # EMA斜率
    
    # 清理数据
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # 删除缺失值过多的列
    missing_ratio = df.isnull().sum() / len(df)
    cols_to_drop = missing_ratio[missing_ratio > 0.3].index
    df = df.drop(columns=cols_to_drop)
    
    print(f"✅ 特征构建完成: {len(df.columns)} 个特征")
    print(f"📊 有效样本数: {len(df.dropna())}")
    
    return df.dropna()

def build_labels(df: pd.DataFrame, forward_n: int = 4, thr: float = 0.008) -> pd.DataFrame:
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

# 使用建议
# # 1. 生成特征
# df_with_features = build_features(raw_data)

# # 2. 生成标签  
# df_with_labels = build_labels(df_with_features, forward_n=4, thr=0.01)  # 尝试1%阈值

# # 3. 特征选择
# final_df = select_important_features(df_with_labels, top_k=25)

# print(f"🎉 最终数据集: {final_df.shape}")
# print(f"📊 标签分布: {final_df['y'].value_counts(normalize=True).to_dict()}")    
"""
增强版特征工程 - 专门针对加密货币交易优化
包含更多市场微观结构特征、情绪指标和高级技术分析
"""

import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator, StochasticOscillator, ROCIndicator, WilliamsRIndicator
from ta.trend import EMAIndicator, MACD, ADXIndicator, IchimokuIndicator, AroonIndicator
from ta.volatility import BollingerBands, AverageTrueRange, DonchianChannel, KeltnerChannel
from ta.volume import OnBalanceVolumeIndicator, AccDistIndexIndicator, VolumeSMAIndicator
from ta.others import DailyReturnIndicator
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def build_enhanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    增强版特征工程 - 专门针对加密货币交易优化
    包含更多市场微观结构特征、情绪指标和高级技术分析
    """
    df = df.copy()
    
    # 确保按时间顺序处理
    df = df.sort_index()
    
    print("🔧 开始构建增强特征...")
    
    # ==================== 1. 基础价格特征 ====================
    print("📊 构建基础价格特征...")
    
    # 价格变化率
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    df['returns_2'] = df['close'].pct_change(2)
    df['returns_3'] = df['close'].pct_change(3)
    df['returns_5'] = df['close'].pct_change(5)
    
    # 价格位置特征
    df['high_low_ratio'] = df['high'] / df['low']
    df['open_close_ratio'] = df['close'] / df['open']
    df['body_size'] = abs(df['close'] - df['open']) / df['open']
    df['upper_shadow'] = (df['high'] - np.maximum(df['open'], df['close'])) / df['open']
    df['lower_shadow'] = (np.minimum(df['open'], df['close']) - df['low']) / df['open']
    
    # 价格动量
    for period in [1, 2, 3, 5, 10]:
        df[f'price_momentum_{period}'] = df['close'] / df['close'].shift(period) - 1
    
    # ==================== 2. 趋势特征 ====================
    print("📈 构建趋势特征...")
    
    # EMA系列 - 多时间框架
    for period in [5, 10, 20, 50]:
        df[f'ema_{period}'] = df['close'].ewm(span=period).mean()
        df[f'price_ema_ratio_{period}'] = df['close'] / df[f'ema_{period}']
        df[f'ema_slope_{period}'] = df[f'ema_{period}'].diff(3) / df[f'ema_{period}'].shift(3)
    
    # EMA交叉信号
    df['ema_5_10_cross'] = (df['ema_5'] > df['ema_10']).astype(int)
    df['ema_10_20_cross'] = (df['ema_10'] > df['ema_20']).astype(int)
    df['ema_20_50_cross'] = (df['ema_20'] > df['ema_50']).astype(int)
    
    # MACD增强版
    macd = MACD(df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_histogram'] = macd.macd_diff()
    df['macd_histogram_change'] = df['macd_histogram'].diff()
    df['macd_signal_strength'] = abs(df['macd'] - df['macd_signal'])
    
    # ADX趋势强度
    adx = ADXIndicator(df['high'], df['low'], df['close'], window=14)
    df['adx'] = adx.adx()
    df['adx_pos'] = adx.adx_pos()
    df['adx_neg'] = adx.adx_neg()
    df['adx_trend_strength'] = (df['adx'] > 25).astype(int)
    df['adx_direction'] = (df['adx_pos'] > df['adx_neg']).astype(int)
    
    # Aroon指标
    aroon = AroonIndicator(df['high'], df['low'], window=14)
    df['aroon_up'] = aroon.aroon_up()
    df['aroon_down'] = aroon.aroon_down()
    df['aroon_oscillator'] = df['aroon_up'] - df['aroon_down']
    
    # ==================== 3. 动量特征 ====================
    print("⚡ 构建动量特征...")
    
    # RSI多时间框架
    for period in [7, 14, 21]:
        rsi = RSIIndicator(df['close'], window=period)
        df[f'rsi_{period}'] = rsi.rsi()
        df[f'rsi_{period}_overbought'] = (df[f'rsi_{period}'] > 70).astype(int)
        df[f'rsi_{period}_oversold'] = (df[f'rsi_{period}'] < 30).astype(int)
        df[f'rsi_{period}_divergence'] = df[f'rsi_{period}'].diff(3)
    
    # 随机指标KDJ
    stoch = StochasticOscillator(df['high'], df['low'], df['close'], window=14, smooth_window=3)
    df['stoch_k'] = stoch.stoch()
    df['stoch_d'] = stoch.stoch_signal()
    df['stoch_j'] = 3 * df['stoch_k'] - 2 * df['stoch_d']
    df['stoch_overbought'] = (df['stoch_k'] > 80).astype(int)
    df['stoch_oversold'] = (df['stoch_k'] < 20).astype(int)
    
    # Williams %R
    williams_r = WilliamsRIndicator(df['high'], df['low'], df['close'], lbp=14)
    df['williams_r'] = williams_r.williams_r()
    df['williams_r_overbought'] = (df['williams_r'] > -20).astype(int)
    df['williams_r_oversold'] = (df['williams_r'] < -80).astype(int)
    
    # ROC变化率
    for period in [3, 5, 10, 20]:
        roc = ROCIndicator(df['close'], window=period)
        df[f'roc_{period}'] = roc.roc()
    
    # ==================== 4. 波动率特征 ====================
    print("📊 构建波动率特征...")
    
    # 布林带增强版
    bb = BollingerBands(df['close'], window=20, window_dev=2)
    df['bb_upper'] = bb.bollinger_hband()
    df['bb_lower'] = bb.bollinger_lband()
    df['bb_middle'] = bb.bollinger_mavg()
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    df['bb_squeeze'] = (df['bb_width'] < df['bb_width'].rolling(20).mean()).astype(int)
    df['bb_breakout_upper'] = (df['close'] > df['bb_upper']).astype(int)
    df['bb_breakout_lower'] = (df['close'] < df['bb_lower']).astype(int)
    
    # Keltner通道
    kc = KeltnerChannel(df['high'], df['low'], df['close'], window=20)
    df['kc_upper'] = kc.keltner_channel_hband()
    df['kc_lower'] = kc.keltner_channel_lband()
    df['kc_middle'] = kc.keltner_channel_mband()
    df['kc_position'] = (df['close'] - df['kc_lower']) / (df['kc_upper'] - df['kc_lower'])
    
    # ATR波动率
    atr = AverageTrueRange(df['high'], df['low'], df['close'], window=14)
    df['atr'] = atr.average_true_range()
    df['atr_ratio'] = df['atr'] / df['close']
    df['atr_percentile'] = df['atr_ratio'].rolling(50).rank(pct=True)
    
    # 唐奇安通道
    dc = DonchianChannel(df['high'], df['low'], df['close'], window=20)
    df['dc_upper'] = dc.donchian_channel_hband()
    df['dc_lower'] = dc.donchian_channel_lband()
    df['dc_position'] = (df['close'] - df['dc_lower']) / (df['dc_upper'] - df['dc_lower'])
    df['dc_breakout'] = ((df['close'] > df['dc_upper']) | (df['close'] < df['dc_lower'])).astype(int)
    
    # ==================== 5. 成交量特征 ====================
    print("📈 构建成交量特征...")
    
    # OBV增强版
    obv = OnBalanceVolumeIndicator(df['close'], df['volume'])
    df['obv'] = obv.on_balance_volume()
    df['obv_ema_10'] = df['obv'].ewm(span=10).mean()
    df['obv_ema_20'] = df['obv'].ewm(span=20).mean()
    df['obv_slope'] = df['obv'].diff(3)
    df['obv_divergence'] = df['obv'].diff(5) / df['close'].diff(5)
    
    # 累积派发线
    adi = AccDistIndexIndicator(df['high'], df['low'], df['close'], df['volume'])
    df['adi'] = adi.acc_dist_index()
    df['adi_slope'] = df['adi'].diff(3)
    
    # 成交量移动平均
    for period in [5, 10, 20]:
        df[f'volume_sma_{period}'] = df['volume'].rolling(period).mean()
        df[f'volume_ratio_{period}'] = df['volume'] / df[f'volume_sma_{period}']
    
    # 成交量Z-score
    df['volume_zscore'] = (df['volume'] - df['volume'].rolling(20).mean()) / df['volume'].rolling(20).std()
    df['volume_percentile'] = df['volume'].rolling(50).rank(pct=True)
    
    # 价量关系
    df['price_volume_corr'] = df['close'].rolling(10).corr(df['volume'])
    df['volume_price_trend'] = df['volume'] * df['returns']
    
    # ==================== 6. 市场微观结构特征 ====================
    print("🔍 构建市场微观结构特征...")
    
    # 支撑阻力位
    for period in [10, 20, 50]:
        df[f'resistance_{period}'] = df['high'].rolling(period).max()
        df[f'support_{period}'] = df['low'].rolling(period).min()
        df[f'breakout_high_{period}'] = (df['close'] > df[f'resistance_{period}'].shift(1)).astype(int)
        df[f'breakout_low_{period}'] = (df['close'] < df[f'support_{period}'].shift(1)).astype(int)
    
    # 价格位置
    df['day_range'] = (df['high'] - df['low']) / df['open']
    df['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
    df['close_position_20'] = (df['close'] - df['low'].rolling(20).min()) / (df['high'].rolling(20).max() - df['low'].rolling(20).min())
    
    # 价格缺口
    df['gap_up'] = (df['open'] > df['high'].shift(1)).astype(int)
    df['gap_down'] = (df['open'] < df['low'].shift(1)).astype(int)
    df['gap_size'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
    
    # ==================== 7. 高级技术形态特征 ====================
    print("🎯 构建高级技术形态特征...")
    
    # 价格模式识别
    df['higher_high'] = (df['high'] > df['high'].shift(1)).astype(int)
    df['lower_low'] = (df['low'] < df['low'].shift(1)).astype(int)
    df['higher_low'] = (df['low'] > df['low'].shift(1)).astype(int)
    df['lower_high'] = (df['high'] < df['high'].shift(1)).astype(int)
    
    # 连续涨跌
    df['consecutive_up'] = (df['close'] > df['close'].shift(1)).astype(int)
    df['consecutive_down'] = (df['close'] < df['close'].shift(1)).astype(int)
    
    # 计算连续天数
    df['consecutive_up_days'] = df['consecutive_up'].groupby((df['consecutive_up'] != df['consecutive_up'].shift()).cumsum()).cumsum()
    df['consecutive_down_days'] = df['consecutive_down'].groupby((df['consecutive_down'] != df['consecutive_down'].shift()).cumsum()).cumsum()
    
    # ==================== 8. 情绪和资金流特征 ====================
    print("💭 构建情绪和资金流特征...")
    
    # 市场情绪指标
    df['fear_greed'] = (df['rsi_14'] - 50) / 50  # 简化的恐惧贪婪指数
    df['market_sentiment'] = (df['rsi_14'] + df['stoch_k'] + df['williams_r'] + 100) / 3
    
    # 资金流向
    df['money_flow'] = df['close'] * df['volume']
    df['money_flow_sma'] = df['money_flow'].rolling(20).mean()
    df['money_flow_ratio'] = df['money_flow'] / df['money_flow_sma']
    
    # 大单识别（基于成交量异常）
    df['volume_anomaly'] = (df['volume'] > df['volume'].rolling(20).mean() * 2).astype(int)
    df['price_volume_anomaly'] = ((df['volume_anomaly'] == 1) & (abs(df['returns']) > df['returns'].rolling(20).std() * 2)).astype(int)
    
    # ==================== 9. 时间特征 ====================
    print("⏰ 构建时间特征...")
    
    # 时间周期特征
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['day_of_month'] = df.index.day
    df['month'] = df.index.month
    
    # 周期性特征
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    # 交易时段特征
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['is_asia_session'] = ((df['hour'] >= 0) & (df['hour'] < 8)).astype(int)
    df['is_europe_session'] = ((df['hour'] >= 8) & (df['hour'] < 16)).astype(int)
    df['is_america_session'] = ((df['hour'] >= 16) & (df['hour'] < 24)).astype(int)
    
    # ==================== 10. 特征交互和衍生 ====================
    print("🔗 构建特征交互...")
    
    # RSI与成交量的交互
    df['rsi_volume_interaction'] = df['rsi_14'] * df['volume_ratio_20']
    df['rsi_momentum_interaction'] = df['rsi_14'] * df['price_momentum_5']
    
    # MACD与波动率的交互
    df['macd_volatility_interaction'] = df['macd_histogram'] * df['bb_width']
    df['macd_volume_interaction'] = df['macd_histogram'] * df['volume_zscore']
    
    # 趋势与动量的交互
    df['trend_momentum_interaction'] = df['ema_slope_20'] * df['price_momentum_10']
    df['adx_momentum_interaction'] = df['adx'] * df['price_momentum_5']
    
    # 波动率与成交量的交互
    df['volatility_volume_interaction'] = df['atr_ratio'] * df['volume_ratio_20']
    
    # ==================== 11. 统计特征 ====================
    print("📊 构建统计特征...")
    
    # 滚动统计特征
    for period in [5, 10, 20]:
        df[f'returns_skew_{period}'] = df['returns'].rolling(period).skew()
        df[f'returns_kurt_{period}'] = df['returns'].rolling(period).kurt()
        df[f'returns_std_{period}'] = df['returns'].rolling(period).std()
        df[f'returns_mean_{period}'] = df['returns'].rolling(period).mean()
    
    # 价格分位数
    df['close_percentile_20'] = df['close'].rolling(20).rank(pct=True)
    df['close_percentile_50'] = df['close'].rolling(50).rank(pct=True)
    
    # 异常值检测
    df['price_outlier'] = (abs(df['returns']) > df['returns'].rolling(20).std() * 3).astype(int)
    df['volume_outlier'] = (df['volume'] > df['volume'].rolling(20).mean() + 3 * df['volume'].rolling(20).std()).astype(int)
    
    # ==================== 12. 数据清理 ====================
    print("🧹 清理数据...")
    
    # 处理无穷大和NaN值
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # 删除缺失值过多的列
    missing_ratio = df.isnull().sum() / len(df)
    cols_to_drop = missing_ratio[missing_ratio > 0.5].index
    if len(cols_to_drop) > 0:
        print(f"⚠️ 删除缺失值过多的列: {len(cols_to_drop)} 个")
        df = df.drop(columns=cols_to_drop)
    
    # 删除常数列
    constant_cols = df.columns[df.nunique() <= 1]
    if len(constant_cols) > 0:
        print(f"⚠️ 删除常数列: {len(constant_cols)} 个")
        df = df.drop(columns=constant_cols)
    
    # 删除高度相关的列
    corr_matrix = df.select_dtypes(include=[np.number]).corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    high_corr_cols = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]
    if len(high_corr_cols) > 0:
        print(f"⚠️ 删除高相关性列: {len(high_corr_cols)} 个")
        df = df.drop(columns=high_corr_cols)
    
    # 最终清理
    df = df.dropna()
    
    print(f"✅ 增强特征构建完成!")
    print(f"📊 最终特征数: {len(df.columns)}")
    print(f"📊 有效样本数: {len(df)}")
    
    return df

def build_enhanced_labels(df: pd.DataFrame, forward_n: int = 4, thr: float = 0.01) -> pd.DataFrame:
    """
    增强版标签生成 - 创建更平衡和有效的交易信号
    """
    df = df.copy()
    
    print("🎯 构建增强标签...")
    
    # 方法1: 基础收益率标签
    future_return = df['close'].pct_change(forward_n).shift(-forward_n)
    df['y_simple'] = (future_return > thr).astype(int)
    
    # 方法2: 波动率调整标签
    volatility = df['returns'].rolling(forward_n).std()
    adaptive_threshold = thr * (volatility / volatility.median())
    df['y_vol_adjusted'] = (future_return > adaptive_threshold).astype(int)
    
    # 方法3: 相对强度标签（推荐）
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
    
    # 方法4: 多时间框架标签
    for period in [2, 4, 8]:
        future_return_period = df['close'].pct_change(period).shift(-period)
        df[f'y_{period}'] = (future_return_period > thr).astype(int)
    
    # 方法5: 趋势强度标签
    trend_strength = abs(df['ema_20'] - df['ema_50']) / df['ema_50']
    strong_trend = trend_strength > trend_strength.quantile(0.7)
    df['y_trend'] = (df['y_simple'] & strong_trend).astype(int)
    
    # 删除未来数据不可用的行
    df = df.dropna()
    
    # 分析标签分布
    print("🎯 标签分布分析:")
    print(f"  总样本数: {len(df)}")
    print(f"  买入信号 (y=1): {(df['y'] == 1).sum()} ({(df['y'] == 1).mean():.2%})")
    print(f"  非买入信号 (y=0): {(df['y'] == 0).sum()} ({(df['y'] == 0).mean():.2%})")
    
    # 检查标签平衡性
    label_ratio = (df['y'] == 1).mean()
    if label_ratio > 0.7 or label_ratio < 0.3:
        print("⚠️ 标签分布不平衡，建议调整阈值!")
        print(f"💡 当前阈值: {thr}")
        print(f"💡 建议阈值范围: {thr * 0.5:.4f} - {thr * 2:.4f}")
    
    return df

def select_enhanced_features(df: pd.DataFrame, target_col: str = 'y', top_k: int = 50) -> pd.DataFrame:
    """
    增强版特征选择 - 使用多种方法选择最佳特征
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
    from sklearn.linear_model import LassoCV
    import pandas as pd
    
    print(f"🎯 开始特征选择，目标特征数: {top_k}")
    
    # 准备数据
    feature_cols = [col for col in df.columns if col not in ['y', 'y_tri', 'y_simple', 'y_vol_adjusted', 'y_2', 'y_4', 'y_8', 'y_trend']]
    X = df[feature_cols].fillna(0)
    y = df[target_col]
    
    print(f"📊 原始特征数: {len(feature_cols)}")
    
    # 方法1: 随机森林特征重要性
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    rf_importance = pd.DataFrame({
        'feature': feature_cols,
        'rf_importance': rf.feature_importances_
    }).sort_values('rf_importance', ascending=False)
    
    # 方法2: F统计量
    f_selector = SelectKBest(score_func=f_classif, k=min(top_k, len(feature_cols)))
    f_selector.fit(X, y)
    f_scores = pd.DataFrame({
        'feature': feature_cols,
        'f_score': f_selector.scores_
    }).sort_values('f_score', ascending=False)
    
    # 方法3: 互信息
    mi_scores = mutual_info_classif(X, y, random_state=42)
    mi_importance = pd.DataFrame({
        'feature': feature_cols,
        'mi_score': mi_scores
    }).sort_values('mi_score', ascending=False)
    
    # 方法4: Lasso正则化
    lasso = LassoCV(cv=5, random_state=42, max_iter=1000)
    lasso.fit(X, y)
    lasso_importance = pd.DataFrame({
        'feature': feature_cols,
        'lasso_coef': abs(lasso.coef_)
    }).sort_values('lasso_coef', ascending=False)
    
    # 综合评分
    feature_scores = pd.DataFrame({'feature': feature_cols})
    
    # 标准化各方法得分
    feature_scores = feature_scores.merge(rf_importance[['feature', 'rf_importance']], on='feature')
    feature_scores = feature_scores.merge(f_scores[['feature', 'f_score']], on='feature')
    feature_scores = feature_scores.merge(mi_importance[['feature', 'mi_score']], on='feature')
    feature_scores = feature_scores.merge(lasso_importance[['feature', 'lasso_coef']], on='feature')
    
    # 标准化得分
    for col in ['rf_importance', 'f_score', 'mi_score', 'lasso_coef']:
        feature_scores[f'{col}_norm'] = (feature_scores[col] - feature_scores[col].min()) / (feature_scores[col].max() - feature_scores[col].min())
    
    # 综合得分
    feature_scores['combined_score'] = (
        feature_scores['rf_importance_norm'] * 0.3 +
        feature_scores['f_score_norm'] * 0.3 +
        feature_scores['mi_score_norm'] * 0.2 +
        feature_scores['lasso_coef_norm'] * 0.2
    )
    
    # 选择top_k特征
    selected_features = feature_scores.nlargest(top_k, 'combined_score')['feature'].tolist()
    
    print(f"✅ 特征选择完成，选择了 {len(selected_features)} 个特征")
    print("🏆 Top 10 特征:")
    for i, row in feature_scores.nlargest(10, 'combined_score').iterrows():
        print(f"  {i+1:2d}. {row['feature']}: {row['combined_score']:.4f}")
    
    # 返回选择的特征
    return df[selected_features + [target_col]]

def detect_enhanced_data_leakage(df: pd.DataFrame, target_col: str = 'y') -> bool:
    """
    增强版数据泄露检测
    """
    print("🔍 进行增强数据泄露检测...")
    
    feature_cols = [col for col in df.columns if col != target_col]
    suspicious_features = []
    
    # 检查特征与标签的相关性
    correlations = []
    for col in feature_cols:
        if df[col].dtype in ['int64', 'float64']:
            corr = df[col].corr(df[target_col])
            if not np.isnan(corr):
                correlations.append((col, corr))
    
    # 排序并显示高相关性特征
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)
    
    print("📊 特征与标签相关性Top 15:")
    for col, corr in correlations[:15]:
        print(f"  {col}: {corr:.4f}")
        if abs(corr) > 0.8:
            suspicious_features.append((col, corr))
    
    # 检查时间序列特征
    time_suspicious = []
    for col in feature_cols:
        if 'future' in col.lower() or 'ahead' in col.lower():
            time_suspicious.append(col)
    
    if suspicious_features:
        print("❌ 发现可疑的高相关性特征（可能数据泄露）:")
        for col, corr in suspicious_features:
            print(f"  ⚠️ {col}: {corr:.4f}")
        return False
    
    if time_suspicious:
        print("❌ 发现可能包含未来信息的特征:")
        for col in time_suspicious:
            print(f"  ⚠️ {col}")
        return False
    
    print("✅ 未发现明显数据泄露")
    return True

# 使用示例
if __name__ == "__main__":
    # 示例用法
    print("🚀 增强特征工程示例")
    print("=" * 50)
    
    # 这里需要加载实际数据
    # df = pd.read_parquet('data/your_data.parquet')
    # df_with_features = build_enhanced_features(df)
    # df_with_labels = build_enhanced_labels(df_with_features, forward_n=4, thr=0.01)
    # final_df = select_enhanced_features(df_with_labels, top_k=50)
    # 
    # print(f"🎉 最终数据集: {final_df.shape}")
    # print(f"📊 标签分布: {final_df['y'].value_counts(normalize=True).to_dict()}")
    
    print("✅ 增强特征工程模块准备就绪")

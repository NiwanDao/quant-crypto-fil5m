import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import EMAIndicator, MACD
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['ret'] = df['close'].pct_change()
    # Trend/vol features
    df['ema20'] = EMAIndicator(df['close'], 20).ema_indicator()
    df['ema50'] = EMAIndicator(df['close'], 50).ema_indicator()
    macd = MACD(df['close'])
    df['macd'] = macd.macd()
    df['macd_sig'] = macd.macd_signal()
    df['rsi'] = RSIIndicator(df['close'], 14).rsi()
    bb = BollingerBands(df['close'], 20, 2)
    df['bbw'] = (bb.bollinger_hband() - bb.bollinger_lband()) / df['close']
    atr = AverageTrueRange(df['high'], df['low'], df['close'], 14)
    df['atr'] = atr.average_true_range()

    # Lightweight order-book proxies (to be upgraded to real depth later)
    df['expansion'] = (df['high'] - df['low']) / df['close']
    df['vwap'] = (df['close'] * df['volume']).rolling(20).sum() / (df['volume'].rolling(20).sum() + 1e-9)
    df['impact_mom'] = df['close'] - df['vwap']
    df['mom3'] = df['close'].pct_change(3)
    df['mom6'] = df['close'].pct_change(6)
    
    # Volume indicators
    df['volume_sma'] = df['volume'].rolling(20).mean()  # 成交量移动平均
    df['volume_ratio'] = df['volume'] / df['volume_sma']  # 成交量比率
    df['volume_rate'] = df['volume'].pct_change()  # 成交量变化率
    
    # OBV (On-Balance Volume) - 能量潮
    obv = OnBalanceVolumeIndicator(df['close'], df['volume'])
    df['obv'] = obv.on_balance_volume()
    df['obv_ema'] = df['obv'].ewm(span=20).mean()  # OBV的EMA平滑
    
    # Volume-Price Trend (VPT) - 成交量价格趋势
    df['vpt'] = (df['volume'] * df['close'].pct_change()).cumsum()
    
    # Volume Weighted Average Price deviation
    df['vwap_dev'] = (df['close'] - df['vwap']) / df['vwap']  # 价格与VWAP的偏离度
    
    # Volume momentum
    df['vol_mom'] = df['volume'].rolling(5).sum() / df['volume'].rolling(20).sum()  # 5期vs20期成交量动量
    
    # KDJ指标 (随机指标) - 只保留核心指标
    stoch = StochasticOscillator(df['high'], df['low'], df['close'], window=14, smooth_window=3)
    df['kdj_k'] = stoch.stoch()  # K值
    df['kdj_d'] = stoch.stoch_signal()  # D值
    df['kdj_j'] = 3 * df['kdj_k'] - 2 * df['kdj_d']  # J值 = 3K - 2D
    df['kdj_kd_diff'] = df['kdj_k'] - df['kdj_d']  # K-D差值

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    return df

def build_labels(df: pd.DataFrame, forward_n: int = 4, thr: float = 0.001) -> pd.DataFrame:
    df = df.copy()
    fwd = df['close'].pct_change(forward_n).shift(-forward_n)
    df['y'] = (fwd > thr).astype(int)
    df.dropna(inplace=True)
    return df

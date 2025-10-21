import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, MACD
from ta.volatility import BollingerBands, AverageTrueRange

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

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    return df

def build_labels(df: pd.DataFrame, forward_n: int = 4, thr: float = 0.001) -> pd.DataFrame:
    df = df.copy()
    fwd = df['close'].pct_change(forward_n).shift(-forward_n)
    df['y'] = (fwd > thr).astype(int)
    df.dropna(inplace=True)
    return df

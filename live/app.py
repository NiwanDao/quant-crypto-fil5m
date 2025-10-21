import os
import yaml
import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
import ccxt
from datetime import datetime, timezone, timedelta
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv

from utils.features import build_features

load_dotenv()

CONF_PATH = 'conf/config.yml'

def load_conf():
    import yaml
    with open(CONF_PATH, 'r') as f:
        return yaml.safe_load(f)

conf = load_conf()

# Exchange (public endpoints only for MVP)
ex = ccxt.__dict__[conf['exchange']['id']]({
    'enableRateLimit': conf['exchange'].get('enableRateLimit', True),
    'options': conf['exchange'].get('options', {}),
    'apiKey': os.getenv('BINANCE_API_KEY', None),
    'secret': os.getenv('BINANCE_SECRET', None),
})

SYMBOL = conf['symbol']
TIMEFRAME = conf['timeframe']

# Load model & feature list (from training)
mdl = joblib.load('models/lgb_trend.pkl')
# Infer features from training data
_feat_df = pd.read_parquet('data/feat.parquet')
FEATURES = [c for c in _feat_df.columns if c not in ['y']]

app = FastAPI(title='Quant Crypto FIL/USDT 5m — Paper Trading API')

class SignalResponse(BaseModel):
    side: str
    proba_up: float
    price: float
    stop: float | None = None
    trail: float | None = None
    ts: str

def fetch_latest_ohlcv(n=400):
    o = ex.fetch_ohlcv(SYMBOL, timeframe=TIMEFRAME, limit=n)
    df = pd.DataFrame(o, columns=['ts','open','high','low','close','volume'])
    df['ts'] = pd.to_datetime(df['ts'], unit='ms', utc=True)
    df.set_index('ts', inplace=True)
    df.sort_index(inplace=True)
    return df

@app.get('/health')
def health():
    return {'status':'ok','symbol':SYMBOL,'timeframe':TIMEFRAME}

@app.get('/signal', response_model=SignalResponse)
def signal():
    # fetch recent bars and build features
    raw = fetch_latest_ohlcv(600)
    feat = build_features(raw.copy())
    feat = feat.reindex(columns=list(set(FEATURES) & set(feat.columns))).dropna()
    # align features to trained columns (fill missing with 0)
    for c in FEATURES:
        if c not in feat.columns:
            feat[c] = 0.0
    feat = feat[FEATURES]
    # predict
    p_up = float(mdl.predict_proba(feat.tail(1))[:,1][0])
    price = float(raw['close'].iloc[-1])

    side = 'buy' if p_up > conf['model']['proba_threshold'] else 'flat'

    # Dynamic stop/trail with ATR from disk features if available, else proxy
    # For simplicity, reuse last ATR from data/feat.parquet if timestamps align; else use 1% proxy
    try:
        last_feat = pd.read_parquet('data/feat.parquet').iloc[-1]
        atr = float(last_feat.get('atr', price*0.01))
    except Exception:
        atr = price*0.01

    stop = price - conf['risk']['atr_stop_mult'] * atr if side=='buy' else None
    trail = price - conf['risk']['atr_trail_mult'] * atr if side=='buy' else None

    return SignalResponse(
        side=side,
        proba_up=p_up,
        price=price,
        stop=stop,
        trail=trail,
        ts=datetime.now(timezone.utc).isoformat()
    )

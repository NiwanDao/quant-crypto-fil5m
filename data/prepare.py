import os, time, math
import ccxt
import pandas as pd
import yaml
from datetime import datetime, timedelta, timezone
from utils.features import build_features, build_labels

CONF_PATH = 'conf/config.yml'

def load_conf():
    with open(CONF_PATH, 'r') as f:
        return yaml.safe_load(f)

def fetch_ohlcv_all(ex, symbol, timeframe, since_ms, batch_limit=1000):
    all_rows = []
    since = since_ms
    while True:
        o = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=batch_limit)
        if not o:
            break
        all_rows += o
        if len(o) < batch_limit:
            break
        since = o[-1][0] + 1  # next ms
        time.sleep(ex.rateLimit/1000.0 if hasattr(ex, 'rateLimit') else 0.5)
    return all_rows

def main():
    conf = load_conf()
    ex = ccxt.__dict__[conf['exchange']['id']]({
        'enableRateLimit': conf['exchange'].get('enableRateLimit', True),
        'options': conf['exchange'].get('options', {})
    })
    symbol = conf['symbol']
    timeframe = conf['timeframe']
    lookback_days = conf['fetch']['lookback_days']
    batch_limit = conf['fetch']['batch_limit']

    since_dt = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    since_ms = int(since_dt.timestamp() * 1000)

    print(f'Fetching {symbol} {timeframe} since {since_dt.isoformat()}...')
    rows = fetch_ohlcv_all(ex, symbol, timeframe, since_ms, batch_limit=batch_limit)
    df = pd.DataFrame(rows, columns=['ts','open','high','low','close','volume'])
    df['ts'] = pd.to_datetime(df['ts'], unit='ms', utc=True)
    df.set_index('ts', inplace=True)
    df.sort_index(inplace=True)

    # Build features & labels
    df = build_features(df)
    df = build_labels(df, forward_n=conf['features']['forward_n'], thr=conf['features']['label_threshold'])

    os.makedirs('data', exist_ok=True)
    df.to_parquet('data/feat.parquet')
    print('Saved data/feat.parquet with', len(df), 'rows.')

if __name__ == '__main__':
    main()

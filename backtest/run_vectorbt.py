import yaml, pandas as pd, numpy as np
import vectorbt as vbt
import lightgbm as lgb
import joblib

CONF_PATH = 'conf/config.yml'

def load_conf():
    with open(CONF_PATH, 'r') as f:
        return yaml.safe_load(f)

def slippage_rate(base_bps=1, trade_size=1.0, adv=1e6):
    # simplistic slippage (bps to rate)
    return base_bps / 1e4 + 0.001 * abs(trade_size) / adv

def main():
    conf = load_conf()
    df = pd.read_parquet('data/feat.parquet')
    features = [c for c in df.columns if c not in ['y']]
    # load model
    mdl = joblib.load('models/lgb_trend.pkl')
    p_up = mdl.predict_proba(df[features])[:,1]

    # entry/exit
    entries = p_up > conf['model']['proba_threshold']
    exits = ~entries.shift(1).fillna(False)

    close = df['close']

    fees = conf['fees_slippage']['taker_fee_bps']/1e4
    slip = conf['fees_slippage']['base_slippage_bps']/1e4

    size = conf['backtest']['fixed_cash_per_trade'] / close
    size = size.clip(lower=0)

    pf = vbt.Portfolio.from_signals(
        close=close,
        entries=entries,
        exits=exits,
        fees=fees,
        slippage=slip,
        init_cash=conf['backtest']['initial_cash'],
        size=size
    )

    print(pf.stats())
    # Save report
    pf.stats().to_frame('value').to_csv('backtest/stats.csv')
    pf.total_return().vbt.plot().figure.savefig('backtest/equity_curve.png', dpi=150)

if __name__ == '__main__':
    main()

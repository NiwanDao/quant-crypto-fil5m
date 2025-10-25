import os
import yaml
import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
import ccxt
from datetime import datetime, timezone, timedelta
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from functools import lru_cache
from typing import Optional, Dict, List

from utils.features import build_features
from simulated_trading import SimulatedTrader

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

app = FastAPI(title='Quant Crypto FIL/USDT 15m — Auto Trading API')

# 初始化模拟交易器
trader = SimulatedTrader(ex, conf)

class SignalResponse(BaseModel):
    side: str
    proba_up: float
    proba_down: float  # 添加下跌概率
    price: float
    stop: float | None = None
    trail: float | None = None
    take_profit: float | None = None
    signal_strength: float | None = None  # 信号强度 (0-1)
    atr: float | None = None  # 当前ATR值
    market_volatility: str | None = None  # 市场波动性: low/medium/high
    ts: str

class TradingStatus(BaseModel):
    is_trading_enabled: bool
    current_balance: float
    total_pnl: float
    unrealized_pnl: float
    position: Optional[Dict] = None
    active_orders: List[Dict]
    trade_count: int
    current_price: float

class AutoTradeRequest(BaseModel):
    enable: bool
    risk_percent: float = 0.01  # 每次交易风险百分比
    min_signal_strength: float = 0.5  # 最小信号强度

def fetch_latest_ohlcv(n=400):
    o = ex.fetch_ohlcv(SYMBOL, timeframe=TIMEFRAME, limit=n)
    df = pd.DataFrame(o, columns=['ts','open','high','low','close','volume'])
    df['ts'] = pd.to_datetime(df['ts'], unit='ms', utc=True)
    df.set_index('ts', inplace=True)
    df.sort_index(inplace=True)
    return df

@lru_cache(maxsize=1)
def get_cached_atr():
    """缓存ATR值，避免重复读取文件"""
    try:
        last_feat = pd.read_parquet('data/feat.parquet').iloc[-1]
        return float(last_feat.get('atr', 0.01))
    except Exception:
        return 0.01

def calculate_signal_strength(p_up, p_down, features):
    """计算信号强度"""
    # 基于概率差异和特征一致性
    prob_diff = abs(p_up - p_down)
    
    # 基于技术指标一致性
    rsi = features.get('rsi', 50)
    macd = features.get('macd', 0)
    macd_sig = features.get('macd_sig', 0)
    
    # RSI一致性评分
    rsi_score = 0
    if p_up > 0.6:  # 看涨信号
        if rsi < 70:  # RSI未超买
            rsi_score = 0.3
        elif rsi > 80:  # RSI超买
            rsi_score = -0.2
    elif p_down > 0.6:  # 看跌信号
        if rsi > 30:  # RSI未超卖
            rsi_score = 0.3
        elif rsi < 20:  # RSI超卖
            rsi_score = -0.2
    
    # MACD一致性评分
    macd_score = 0
    if p_up > 0.6 and macd > macd_sig:  # MACD金叉
        macd_score = 0.3
    elif p_down > 0.6 and macd < macd_sig:  # MACD死叉
        macd_score = 0.3
    
    # 综合信号强度
    strength = min(1.0, max(0.0, prob_diff + rsi_score + macd_score))
    return strength

def assess_market_volatility(atr, price):
    """评估市场波动性"""
    atr_pct = (atr / price) * 100
    if atr_pct < 1.0:
        return "low"
    elif atr_pct < 2.5:
        return "medium"
    else:
        return "high"

def get_dynamic_risk_params(atr, volatility, signal_strength):
    """根据市场状态动态调整风控参数"""
    base_stop_mult = conf['risk']['atr_stop_mult']
    base_tp_mult = conf['risk'].get('atr_tp_mult', 2.0)
    
    # 根据波动性调整
    if volatility == "high":
        stop_mult = base_stop_mult * 1.2  # 高波动时放宽止损
        tp_mult = base_tp_mult * 1.1
    elif volatility == "low":
        stop_mult = base_stop_mult * 0.8  # 低波动时收紧止损
        tp_mult = base_tp_mult * 0.9
    else:
        stop_mult = base_stop_mult
        tp_mult = base_tp_mult
    
    # 根据信号强度调整
    strength_factor = 0.8 + (signal_strength * 0.4)  # 0.8-1.2
    stop_mult *= strength_factor
    tp_mult *= strength_factor
    
    return stop_mult, tp_mult

def execute_auto_trade(signal_data: dict, risk_percent: float = 0.01, min_strength: float = 0.5) -> dict:
    """执行自动交易"""
    if not trader.is_trading_enabled:
        return {'success': False, 'message': '自动交易未启用'}
    
    # 检查信号强度
    if signal_data['signal_strength'] < min_strength:
        return {'success': False, 'message': f'信号强度不足: {signal_data["signal_strength"]:.3f} < {min_strength}'}
    
    # 检查止损止盈
    triggered_orders = trader.check_stop_loss_take_profit()
    
    # 执行新交易
    if signal_data['side'] == 'buy':
        if trader.position and trader.position.side == 'long':
            return {'success': False, 'message': '已有持仓，跳过买入信号'}
        
        amount = trader.calculate_position_size(signal_data['price'], risk_percent)
        result = trader.place_market_order(
            'buy', 
            amount, 
            stop_loss=signal_data['stop'],
            take_profit=signal_data['take_profit']
        )
        
    elif signal_data['side'] == 'sell':
        if not trader.position or trader.position.side != 'long':
            return {'success': False, 'message': '没有持仓可卖'}
        
        result = trader.place_market_order(
            'sell', 
            trader.position.amount,
            stop_loss=signal_data['stop'],
            take_profit=signal_data['take_profit']
        )
    
    else:  # flat
        return {'success': False, 'message': '观望信号，不执行交易'}
    
    return {
        'success': result['success'],
        'message': result['message'],
        'triggered_orders': triggered_orders
    }

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
    current_features = feat.tail(1).iloc[0].to_dict()

    # 计算下跌概率和信号强度
    p_down = 1.0 - p_up
    signal_strength = calculate_signal_strength(p_up, p_down, current_features)
    
    # 获取ATR和市场波动性
    atr = get_cached_atr()
    if atr == 0.01:  # 如果缓存失败，使用价格估算
        atr = price * 0.01
    volatility = assess_market_volatility(atr, price)
    
    # 改进的信号判断逻辑
    buy_threshold = conf['model']['proba_threshold']
    sell_threshold = conf['model'].get('sell_threshold', 0.40)
    
    # 添加信号强度过滤
    min_strength = conf['model'].get('min_signal_strength', 0.3)
    
    if p_up > buy_threshold and signal_strength > min_strength:
        side = 'buy'
    elif p_down > sell_threshold and signal_strength > min_strength:
        side = 'sell'
    else:
        side = 'flat'

    # 动态风控参数
    if side != 'flat':
        stop_mult, tp_mult = get_dynamic_risk_params(atr, volatility, signal_strength)
        
        if side == 'buy':
            stop = price - stop_mult * atr
            trail = price - conf['risk']['atr_trail_mult'] * atr
            take_profit = price + tp_mult * atr
        elif side == 'sell':
            stop = price + stop_mult * atr
            trail = price + conf['risk']['atr_trail_mult'] * atr
            take_profit = price - tp_mult * atr
    else:
        stop = trail = take_profit = None

    return SignalResponse(
        side=side,
        proba_up=p_up,
        proba_down=p_down,  # 添加下跌概率
        price=price,
        stop=stop,
        trail=trail,
        take_profit=take_profit,
        signal_strength=signal_strength,
        atr=atr,
        market_volatility=volatility,
        ts=datetime.now(timezone.utc).isoformat()
    )

# 交易相关API端点

@app.get('/trading/status', response_model=TradingStatus)
def get_trading_status():
    """获取交易状态"""
    status = trader.get_trading_status()
    return TradingStatus(**status)

@app.post('/trading/enable')
def enable_trading():
    """启用自动交易"""
    result = trader.enable_trading()
    return result

@app.post('/trading/disable')
def disable_trading():
    """禁用自动交易"""
    result = trader.disable_trading()
    return result

@app.post('/trading/reset')
def reset_account():
    """重置账户"""
    result = trader.reset_account()
    return result

@app.post('/trading/auto-trade')
def auto_trade(request: AutoTradeRequest):
    """自动交易接口"""
    # 获取最新信号
    signal_response = signal()
    signal_data = signal_response.dict()
    
    # 执行自动交易
    trade_result = execute_auto_trade(
        signal_data, 
        request.risk_percent, 
        request.min_signal_strength
    )
    
    return {
        'signal': signal_data,
        'trade_result': trade_result,
        'trading_status': trader.get_trading_status()
    }

@app.get('/trading/signal-and-trade')
def signal_and_trade():
    """获取信号并自动交易（如果启用）"""
    # 获取信号
    signal_response = signal()
    signal_data = signal_response.dict()
    
    # 如果自动交易启用，执行交易
    trade_result = None
    if trader.is_trading_enabled:
        trade_result = execute_auto_trade(signal_data)
    
    return {
        'signal': signal_data,
        'trade_result': trade_result,
        'trading_enabled': trader.is_trading_enabled
    }

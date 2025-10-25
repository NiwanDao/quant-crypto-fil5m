"""
优化的实时交易API
使用集成模型、动态阈值和鲁棒性检查
"""

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
import json

from utils.features import build_features
from utils.dynamic_thresholds import DynamicThresholdManager
from utils.ensemble_predictor import EnsemblePredictor, RobustSignalGenerator
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

# 初始化优化组件
ensemble_predictor = EnsemblePredictor()
dynamic_threshold_manager = DynamicThresholdManager()
robust_signal_generator = RobustSignalGenerator(ensemble_predictor)

# 加载模型
if conf['model']['use_ensemble']:
    if not ensemble_predictor.load_models():
        print("⚠️ 集成模型加载失败，回退到单一模型")
        # 回退到单一模型
        try:
            main_model = joblib.load('models/lgb_trend.pkl')
            ensemble_predictor.models = [main_model]
        except Exception as e:
            print(f"❌ 单一模型也加载失败: {e}")
            raise HTTPException(status_code=500, detail="模型加载失败")
else:
    # 使用单一模型
    try:
        main_model = joblib.load('models/lgb_trend.pkl')
        ensemble_predictor.models = [main_model]
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        raise HTTPException(status_code=500, detail="模型加载失败")

# 推断特征列表
_feat_df = pd.read_parquet('data/feat.parquet')
FEATURES = [c for c in _feat_df.columns if c not in ['y']]

app = FastAPI(title='Quant Crypto FIL/USDT 15m — Optimized Auto Trading API')

# 初始化模拟交易器
trader = SimulatedTrader(ex, conf)

class OptimizedSignalResponse(BaseModel):
    side: str
    proba_up: float
    proba_down: float
    price: float
    stop: float | None = None
    trail: float | None = None
    take_profit: float | None = None
    signal_strength: float | None = None
    confidence: float | None = None
    uncertainty: float | None = None
    atr: float | None = None
    market_volatility: str | None = None
    market_trend: str | None = None
    market_regime: str | None = None
    dynamic_thresholds: Dict | None = None
    model_agreement: float | None = None
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
    model_performance: Dict | None = None

class AutoTradeRequest(BaseModel):
    enable: bool
    risk_percent: float = 0.01
    min_signal_strength: float = 0.5
    use_robust_signals: bool = True

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

def get_market_context(df: pd.DataFrame, current_price: float, atr: float) -> Dict:
    """获取市场上下文信息"""
    prices = df['close']
    volumes = df['volume']
    
    # 评估市场状态
    market_volatility = dynamic_threshold_manager.assess_market_volatility(atr, current_price)
    market_trend = dynamic_threshold_manager.assess_market_trend(prices)
    market_regime = dynamic_threshold_manager.assess_market_regime(prices, volumes)
    
    return {
        'volatility': market_volatility,
        'trend': market_trend,
        'regime': market_regime,
        'atr': atr,
        'price': current_price
    }

def get_optimized_signal(df: pd.DataFrame) -> Dict:
    """获取优化的交易信号"""
    
    # 构建特征
    feat = build_features(df.copy())
    feat = feat.reindex(columns=list(set(FEATURES) & set(feat.columns))).dropna()
    
    # 对齐特征到训练列
    for c in FEATURES:
        if c not in feat.columns:
            feat[c] = 0.0
    feat = feat[FEATURES]
    
    if len(feat) == 0:
        raise ValueError("特征构建失败")
    
    # 获取最新特征
    latest_features = feat.tail(1)
    current_price = float(df['close'].iloc[-1])
    atr = get_cached_atr()
    
    # 获取市场上下文
    market_context = get_market_context(df, current_price, atr)
    
    # 使用集成模型预测
    if conf['model']['use_ensemble']:
        prediction_result = ensemble_predictor.predict_ensemble(latest_features)
        proba_up = float(prediction_result['ensemble_proba'][0])
        uncertainty = float(prediction_result['uncertainty'][0])
        confidence = float(prediction_result['confidence'][0])
        model_agreement = float(prediction_result['model_agreement'][0])
    else:
        # 使用单一模型
        model = ensemble_predictor.models[0]
        proba_up = float(model.predict_proba(latest_features)[:,1][0])
        uncertainty = 0.0  # 单一模型无法计算不确定性
        confidence = 1.0
        model_agreement = 1.0
    
    proba_down = 1.0 - proba_up
    
    # 获取动态阈值
    if conf['dynamic_thresholds']['enabled']:
        dynamic_thresholds = dynamic_threshold_manager.get_dynamic_thresholds(
            current_price=current_price,
            atr=atr,
            prices=df['close'],
            volumes=df['volume'],
            signal_strength=confidence,
            recent_performance=robust_signal_generator.get_performance_stats()
        )
        buy_threshold = dynamic_thresholds['buy_threshold']
        sell_threshold = dynamic_thresholds['sell_threshold']
    else:
        # 使用静态阈值
        buy_threshold = conf['model']['proba_threshold']
        sell_threshold = conf['model']['sell_threshold']
        dynamic_thresholds = None
    
    # 生成鲁棒信号
    if conf['robustness']['enabled']:
        signal_result = robust_signal_generator.generate_robust_signal(
            latest_features,
            base_buy_threshold=buy_threshold,
            base_sell_threshold=sell_threshold,
            min_confidence=conf['robustness']['min_signal_confidence'],
            uncertainty_threshold=conf['model']['uncertainty_threshold']
        )
        side = signal_result['side']
        signal_strength = signal_result['strength']
        signal_reason = signal_result['reason']
    else:
        # 简单信号生成
        if proba_up > buy_threshold and confidence > conf['model']['min_confidence']:
            side = 'buy'
            signal_strength = min(1.0, proba_up + confidence - 0.5)
        elif proba_down > sell_threshold and confidence > conf['model']['min_confidence']:
            side = 'sell'
            signal_strength = min(1.0, proba_down + confidence - 0.5)
        else:
            side = 'flat'
            signal_strength = max(proba_up, proba_down)
        signal_reason = f'simple_signal_{side}'
    
    # 计算止损止盈
    if side != 'flat':
        stop_mult = conf['risk']['atr_stop_mult']
        tp_mult = conf['risk'].get('atr_tp_mult', 2.0)
        
        if side == 'buy':
            stop = current_price - stop_mult * atr
            trail = current_price - conf['risk']['atr_trail_mult'] * atr
            take_profit = current_price + tp_mult * atr
        elif side == 'sell':
            stop = current_price + stop_mult * atr
            trail = current_price + conf['risk']['atr_trail_mult'] * atr
            take_profit = current_price - tp_mult * atr
    else:
        stop = trail = take_profit = None
    
    return {
        'side': side,
        'proba_up': proba_up,
        'proba_down': proba_down,
        'price': current_price,
        'stop': stop,
        'trail': trail,
        'take_profit': take_profit,
        'signal_strength': signal_strength,
        'confidence': confidence,
        'uncertainty': uncertainty,
        'atr': atr,
        'market_volatility': market_context['volatility'],
        'market_trend': market_context['trend'],
        'market_regime': market_context['regime'],
        'dynamic_thresholds': dynamic_thresholds,
        'model_agreement': model_agreement,
        'signal_reason': signal_reason,
        'ts': datetime.now(timezone.utc).isoformat()
    }

@app.get('/health')
def health():
    return {
        'status': 'ok',
        'symbol': SYMBOL,
        'timeframe': TIMEFRAME,
        'model_type': 'ensemble' if conf['model']['use_ensemble'] else 'single',
        'dynamic_thresholds': conf['dynamic_thresholds']['enabled'],
        'robustness_enabled': conf['robustness']['enabled']
    }

@app.get('/signal', response_model=OptimizedSignalResponse)
def signal():
    """获取优化的交易信号"""
    try:
        # 获取最新数据
        raw = fetch_latest_ohlcv(600)
        
        # 获取优化信号
        signal_data = get_optimized_signal(raw)
        
        return OptimizedSignalResponse(**signal_data)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"信号生成失败: {str(e)}")

@app.get('/trading/status', response_model=TradingStatus)
def get_trading_status():
    """获取交易状态"""
    status = trader.get_trading_status()
    
    # 添加模型性能信息
    if conf['model']['use_ensemble']:
        model_performance = robust_signal_generator.get_performance_stats()
        status['model_performance'] = model_performance
    
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
    try:
        # 获取最新信号
        signal_response = signal()
        signal_data = signal_response.dict()
        
        # 检查鲁棒性
        if request.use_robust_signals and conf['robustness']['enabled']:
            if signal_data['confidence'] < conf['robustness']['min_signal_confidence']:
                return {
                    'success': False,
                    'message': f'信号置信度不足: {signal_data["confidence"]:.3f}',
                    'signal': signal_data
                }
            
            if signal_data['uncertainty'] > conf['robustness']['max_uncertainty']:
                return {
                    'success': False,
                    'message': f'信号不确定性过高: {signal_data["uncertainty"]:.3f}',
                    'signal': signal_data
                }
        
        # 执行交易逻辑（这里可以添加实际的交易执行代码）
        trade_result = {
            'success': True,
            'message': '信号生成成功，等待执行',
            'signal': signal_data
        }
        
        return trade_result
        
    except Exception as e:
        return {
            'success': False,
            'message': f'自动交易失败: {str(e)}',
            'signal': None
        }

@app.get('/model/diagnostics')
def get_model_diagnostics():
    """获取模型诊断信息"""
    try:
        # 获取最新数据
        raw = fetch_latest_ohlcv(100)
        feat = build_features(raw.copy())
        feat = feat.reindex(columns=list(set(FEATURES) & set(feat.columns))).dropna()
        
        for c in FEATURES:
            if c not in feat.columns:
                feat[c] = 0.0
        feat = feat[FEATURES]
        
        if len(feat) == 0:
            return {'error': '特征构建失败'}
        
        # 获取模型诊断
        diagnostics = ensemble_predictor.get_model_diagnostics(feat)
        
        # 获取阈值统计
        threshold_stats = dynamic_threshold_manager.get_threshold_statistics()
        
        return {
            'model_diagnostics': diagnostics,
            'threshold_statistics': threshold_stats,
            'performance_stats': robust_signal_generator.get_performance_stats()
        }
        
    except Exception as e:
        return {'error': f'诊断信息获取失败: {str(e)}'}

@app.get('/model/feature_importance')
def get_feature_importance():
    """获取特征重要性"""
    try:
        importance = ensemble_predictor.get_feature_importance()
        
        if not importance:
            return {'error': '特征重要性不可用'}
        
        # 创建特征重要性排序
        features = FEATURES
        mean_importance = importance['mean']
        std_importance = importance['std']
        stability = importance['stability']
        
        feature_importance_list = []
        for i, feature in enumerate(features):
            feature_importance_list.append({
                'feature': feature,
                'importance': float(mean_importance[i]),
                'std': float(std_importance[i]),
                'stability': float(stability[i])
            })
        
        # 按重要性排序
        feature_importance_list.sort(key=lambda x: x['importance'], reverse=True)
        
        return {
            'feature_importance': feature_importance_list[:20],  # 返回前20个重要特征
            'total_features': len(features)
        }
        
    except Exception as e:
        return {'error': f'特征重要性获取失败: {str(e)}'}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)







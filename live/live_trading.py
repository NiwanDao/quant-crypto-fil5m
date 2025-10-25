"""
实盘交易系统
支持实时信号生成、交易执行、风险管理和监控
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
import yaml
import json
import os
import time
import logging
import requests
import websocket
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import warnings
from dataclasses import dataclass
from enum import Enum

# 导入WebSocket管理器
from websocket_manager import WebSocketManager
from data_fetcher import DataFetcher
from risk_manager import RiskManager

warnings.filterwarnings('ignore')

class TradingMode(Enum):
    """交易模式"""
    PAPER = "paper"  # 模拟交易
    LIVE = "live"    # 实盘交易

class OrderSide(Enum):
    """订单方向"""
    BUY = "buy"
    SELL = "sell"

class OrderStatus(Enum):
    """订单状态"""
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"

@dataclass
class Order:
    """订单信息"""
    id: str
    symbol: str
    side: OrderSide
    amount: float
    price: float
    status: OrderStatus
    timestamp: datetime
    filled_amount: float = 0.0
    filled_price: float = 0.0

@dataclass
class Position:
    """持仓信息"""
    symbol: str
    amount: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    timestamp: datetime
    side: Optional[str] = None
    entry_time: Optional[datetime] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    # 移动止损相关字段
    trailing_stop: Optional[float] = None
    trailing_stop_activated: bool = False
    highest_price: Optional[float] = None  # 多头持仓的最高价
    lowest_price: Optional[float] = None  # 空头持仓的最低价
    trailing_distance: Optional[float] = None  # 移动止损距离

class LiveTradingSystem:
    """实盘交易系统"""
    
    def __init__(self, config_path: str = 'conf/config.yml', mode: TradingMode = TradingMode.PAPER):
        self.config_path = config_path
        self.mode = mode
        self.config = self._load_config()
        self.model = None
        self.exchange = None
        self.positions = {}  # 实际持仓数据 (Position对象)
        self.price_tracking = {}  # 价格跟踪数据 (字典格式)
        self.orders = {}
        self.balance = {}
        self.last_signal_time = None
        self.is_running = False
        
        # WebSocket相关
        self.data_fetcher = None
        self.websocket_manager = None
        self.enable_websocket = self.config.get('data_fetching', {}).get('enable_websocket', False)
        
        # 混合时间框架配置
        self.signal_timeframe = self.config.get('data_fetching', {}).get('signal_timeframe', '15m')
        self.stop_timeframe = self.config.get('data_fetching', {}).get('stop_timeframe', '1m')
        self.emergency_stop = self.config.get('data_fetching', {}).get('emergency_stop', 'realtime')
        self.stop_check_interval = self.config.get('data_fetching', {}).get('stop_check_interval', 60)
        
        # 混合止损策略配置
        self.use_hybrid_stops = self.config.get('risk_management', {}).get('use_hybrid_stops', True)
        self.emergency_stop_loss = self.config.get('risk_management', {}).get('emergency_stop_loss', 0.05)
        self.stop_loss_1m_atr_mult = self.config.get('risk_management', {}).get('stop_loss_1m_atr_mult', 1.5)
        self.trailing_stop_1m = self.config.get('risk_management', {}).get('trailing_stop_1m', True)
        self.realtime_emergency_stop = self.config.get('risk_management', {}).get('realtime_emergency_stop', True)
        
        # WebSocket专用1分钟止损配置
        self.websocket_only_1m_stops = self.config.get('risk_management', {}).get('websocket_only_1m_stops', True)
        self.websocket_kline_complete_only = self.config.get('risk_management', {}).get('websocket_kline_complete_only', True)
        
        # 时间跟踪
        self.last_signal_time = None
        self.last_stop_check_time = None
        
        # 设置日志
        self._setup_logging()
        
        # 初始化组件
        self._initialize_components()
        
    def _load_config(self):
        """加载配置文件"""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _setup_logging(self):
        """设置日志系统"""
        os.makedirs('logs', exist_ok=True)
        
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'logs/live_trading_{datetime.now().strftime("%Y%m%d")}.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('LiveTrading')
    
    def _initialize_components(self):
        """初始化组件"""
        self.logger.info("🚀 初始化实盘交易系统...")
        
        # 加载模型
        self._load_model()
        
        # 初始化交易所
        self._initialize_exchange()
        
        # 初始化数据获取器
        self._initialize_data_fetcher()
        
        # 初始化WebSocket管理器
        if self.enable_websocket:
            self._initialize_websocket_manager()
        
        # 初始化账户信息
        self._initialize_account()
        
        # 初始化风险管理器
        self._initialize_risk_manager()
        
        # 输出配置信息
        self._log_configuration()
        
        self.logger.info("✅ 系统初始化完成")
    
    def _log_configuration(self):
        """输出配置信息"""
        try:
            self.logger.info("📋 交易系统配置信息:")
            self.logger.info(f"   - 交易模式: {self.mode}")
            self.logger.info(f"   - 交易对: {self.config.get('trading', {}).get('symbol', 'N/A')}")
            self.logger.info(f"   - 信号时间框架: {self.signal_timeframe}")
            self.logger.info(f"   - 止损时间框架: {self.stop_timeframe}")
            self.logger.info(f"   - WebSocket启用: {self.enable_websocket}")
            self.logger.info(f"   - 混合止损策略: {self.use_hybrid_stops}")
            self.logger.info(f"   - WebSocket专用1分钟止损: {self.websocket_only_1m_stops}")
            self.logger.info(f"   - 仅K线完成时触发: {self.websocket_kline_complete_only}")
            self.logger.info(f"   - 1分钟移动止损: {self.trailing_stop_1m}")
            self.logger.info(f"   - 实时紧急止损: {self.realtime_emergency_stop}")
        except Exception as e:
            self.logger.error(f"❌ 输出配置信息失败: {str(e)}")
    
    def _initialize_data_fetcher(self):
        """初始化数据获取器"""
        try:
            self.data_fetcher = DataFetcher(self.exchange, self.config)
            self.logger.info("✅ 数据获取器初始化成功")
        except Exception as e:
            self.logger.error(f"❌ 数据获取器初始化失败: {str(e)}")
    
    def _initialize_risk_manager(self):
        """初始化风险管理器"""
        try:
            # 从配置中获取风险管理参数
            risk_config = self.config.get('risk_management', {})
            
            # 合并默认配置
            default_risk_config = {
                'max_portfolio_risk': 0.02,
                'max_position_risk': 0.005,
                'max_drawdown_limit': 0.15,
                'volatility_limit': 0.3,
                'position_size_method': 'volatility',
                'base_position_size': 0.1,
                'max_position_size': 0.2,
                'stop_loss_atr_mult': 2.0,
                'take_profit_atr_mult': 3.0,
                'trailing_stop': True,
                'min_portfolio_value': 1000,
                'use_hybrid_stops': True,
                'emergency_stop_loss': 0.05,
                'stop_loss_1m_atr_mult': 1.5,
                'trailing_stop_1m': True,
                'realtime_emergency_stop': True,
                'websocket_only_1m_stops': True,
                'websocket_kline_complete_only': True
            }
            
            # 合并配置
            final_risk_config = {**default_risk_config, **risk_config}
            
            self.risk_manager = RiskManager(final_risk_config)
            self.logger.info("✅ 风险管理器初始化成功")
        except Exception as e:
            self.logger.error(f"❌ 风险管理器初始化失败: {str(e)}")
            self.risk_manager = None
    
    def _initialize_websocket_manager(self):
        """初始化WebSocket管理器"""
        try:
            self.websocket_manager = WebSocketManager(self.data_fetcher, self.config)
            
            # 添加数据回调
            self.websocket_manager.add_data_callback(self._handle_realtime_data)
            
            self.logger.info("✅ WebSocket管理器初始化成功")
        except Exception as e:
            self.logger.error(f"❌ WebSocket管理器初始化失败: {str(e)}")
    
    def _handle_realtime_data(self, symbol: str, market_data: Dict):
        """处理实时数据"""
        try:
            # 打印接收到的实时数据
            # print(f"📊 LiveTradingSystem接收到实时数据: {symbol}")
            # print(f"   时间: {market_data['timestamp']}")
            # print(f"   开盘: {market_data['open']}")
            # print(f"   最高: {market_data['high']}")
            # print(f"   最低: {market_data['low']}")
            # print(f"   收盘: {market_data['close']}")
            # print(f"   成交量: {market_data['volume']}")
            # print(f"   K线完成: {market_data['is_closed']}")
            # print("-" * 50)
            
            # 更新最新价格
            if symbol not in self.price_tracking:
                self.price_tracking[symbol] = {}
            
            self.price_tracking[symbol]['latest_price'] = market_data['close']
            self.price_tracking[symbol]['last_update'] = datetime.now()
            
            # 如果是1分钟K线数据，使用WebSocket数据进行1分钟止损检查
            timeframe = market_data.get('timeframe', '')
            if timeframe == '1m' and self.use_hybrid_stops:
                # 检查是否仅在K线完成时触发止损检查
                is_kline_complete = market_data.get('is_closed', False)
                should_check_stops = True
                
                if self.websocket_kline_complete_only:
                    should_check_stops = is_kline_complete
                    if should_check_stops:
                        self.logger.debug("🕐 WebSocket 1分钟K线完成，检查止损")
                    else:
                        self.logger.debug("🕐 WebSocket 1分钟K线进行中，跳过止损检查")
                else:
                    self.logger.debug("🕐 WebSocket 1分钟K线数据，检查止损")
                
                if should_check_stops:
                    current_price = market_data['close']
                    
                    # 使用WebSocket的实时数据检查1分钟止损
                    if symbol in self.positions:
                        # 直接使用WebSocket缓存数据进行1分钟止损检查
                        self.check_1m_stop_loss_with_websocket_optimized(symbol, current_price, market_data)
            
            # 混合止损策略：实时紧急止损检查
            if self.use_hybrid_stops and self.realtime_emergency_stop:
                self._check_realtime_emergency_stop(symbol, market_data['close'])
            
            # 检查是否需要更新止损（只对实际持仓检查）
            if symbol in self.positions and self.positions[symbol].trailing_stop:
                self._update_trailing_stop(symbol, market_data['close'])
            
            # 记录数据
            self.logger.debug(f"📊 实时数据更新: {symbol} - {market_data['close']}")
            
        except Exception as e:
            self.logger.error(f"❌ 处理实时数据失败: {str(e)}")
    
    def _update_trailing_stop(self, symbol: str, current_price: float):
        """更新移动止损"""
        try:
            if symbol not in self.positions:
                return
                
            position = self.positions[symbol]
            if not isinstance(position, Position):
                return
                
            if not position.trailing_stop:
                return
            
            side = position.side
            trailing_distance = position.trailing_distance or 0.02  # 2% 默认距离
            
            if side == 'buy':
                # 买入订单：价格上升时更新止损
                new_stop = current_price * (1 - trailing_distance)
                if new_stop > (position.stop_loss or 0):
                    position.stop_loss = new_stop
                    self.logger.info(f"📈 更新买入止损: {symbol} - {new_stop:.4f}")
            
            elif side == 'sell':
                # 卖出订单：价格下降时更新止损
                new_stop = current_price * (1 + trailing_distance)
                if new_stop < (position.stop_loss or float('inf')):
                    position.stop_loss = new_stop
                    self.logger.info(f"📉 更新卖出止损: {symbol} - {new_stop:.4f}")
            
        except Exception as e:
            self.logger.error(f"❌ 更新移动止损失败: {str(e)}")
    
    def _load_model(self):
        """加载交易模型"""
        try:
            # 尝试加载集成模型
            self.model = joblib.load('models/ensemble_models.pkl')
            self.logger.info("✅ 集成模型加载成功")
        except:
            try:
                # 回退到优化模型
                self.model = joblib.load('models/lgb_trend_optimized.pkl')
                self.logger.info("✅ 优化模型加载成功")
            except:
                # 回退到基础模型
                model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'lgb_trend.pkl')
                self.model = joblib.load(model_path)
                self.logger.info("✅ 基础模型加载成功")
    
    def _initialize_exchange(self):
        """初始化交易所接口"""
        if self.mode == TradingMode.PAPER:
            self.exchange = PaperTradingExchange(self.config)
        else:
            self.exchange = BinanceExchange(self.config)
        
        self.logger.info(f"✅ 交易所接口初始化完成 ({self.mode.value})")
    
    def _initialize_account(self):
        """初始化账户信息"""
        if self.mode == TradingMode.PAPER:
            # 模拟账户
            self.balance = {
                'USDT': self.config['backtest']['initial_cash'],
                'FIL': 0.0
            }
        else:
            # 获取真实账户余额
            self.balance = self.exchange.get_balance()
        
        self.logger.info(f"💰 账户余额: {self.balance}")
    
    def get_market_data(self, symbol: str, timeframe: str = '15m', limit: int = 100) -> pd.DataFrame:
        """获取市场数据"""
        try:
            # 获取K线数据
            klines = self.exchange.get_klines(symbol, timeframe, limit)
            
            # 转换为DataFrame - ccxt.fetch_ohlcv返回6列数据
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume'
            ])
            
            # 数据类型转换
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col])
            
            # 时间索引
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # 计算收益率
            df['returns'] = df['close'].pct_change()
            
            return df
            
        except Exception as e:
            self.logger.error(f"❌ 获取市场数据失败: {str(e)}")
            return pd.DataFrame()
    
    def calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算技术特征"""
        try:
            # 确保有returns列
            if 'returns' not in df.columns:
                df['returns'] = df['close'].pct_change()
            
            # 基础特征
            df['returns_lag1'] = df['returns'].shift(1)
            df['high_low_ratio'] = df['high'] / df['low']
            df['open_close_ratio'] = df['open'] / df['close']
            df['body_size'] = abs(df['close'] - df['open'])
            
            # EMA特征
            df['ema_5'] = df['close'].ewm(span=5).mean()
            df['ema_10'] = df['close'].ewm(span=10).mean()
            df['ema_20'] = df['close'].ewm(span=20).mean()
            
            # 价格与EMA的比率
            df['price_ema_ratio_5'] = df['close'] / df['ema_5']
            df['price_ema_ratio_10'] = df['close'] / df['ema_10']
            df['price_ema_ratio_20'] = df['close'] / df['ema_20']
            
            # MACD
            df['macd'] = df['ema_12'] - df['ema_26'] if 'ema_12' in df.columns and 'ema_26' in df.columns else df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            df['macd_histogram'] = df['macd'] - df['macd_signal']
            
            # RSI (安全版本)
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=7).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=7).mean()
            rs = gain / loss
            df['rsi_7_safe'] = 100 - (100 / (1 + rs))
            
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi_14_safe'] = 100 - (100 / (1 + rs))
            
            # 布林带 (安全版本)
            df['bb_middle_safe'] = df['close'].rolling(20).mean()
            bb_std = df['close'].rolling(20).std()
            df['bb_upper_safe'] = df['bb_middle_safe'] + (bb_std * 2)
            df['bb_lower_safe'] = df['bb_middle_safe'] - (bb_std * 2)
            df['bb_position_safe'] = (df['close'] - df['bb_lower_safe']) / (df['bb_upper_safe'] - df['bb_lower_safe'])
            
            # 成交量特征
            df['volume_lag1'] = df['volume'].shift(1)
            df['volume_ratio_safe'] = df['volume'] / df['volume'].rolling(20).mean()
            
            # 支撑阻力
            df['resistance_20'] = df['high'].rolling(20).max()
            df['support_20'] = df['low'].rolling(20).min()
            
            # 突破特征
            df['breakout_high'] = (df['close'] > df['high'].shift(1)).astype(int)
            df['breakout_low'] = (df['close'] < df['low'].shift(1)).astype(int)
            
            return df
            
        except Exception as e:
            self.logger.error(f"❌ 特征计算失败: {str(e)}")
            return df
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """计算ATR指标"""
        try:
            high = df['high']
            low = df['low']
            close = df['close']
            
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(period).mean()
            
            return atr
        except Exception as e:
            self.logger.error(f"❌ ATR计算失败: {str(e)}")
            return pd.Series(0.01, index=df.index)
    
    def generate_signal(self, df: pd.DataFrame) -> Tuple[bool, bool, float]:
        """生成交易信号"""
        try:
            if len(df) < 50:  # 需要足够的历史数据
                return False, False, 0.0
            
            # 获取最新数据
            latest_data = df.iloc[-1:].copy()
            
            # 模型实际使用的特征列表
            model_features = [
                'open', 'high', 'low', 'close', 'volume', 'returns', 'returns_lag1', 
                'high_low_ratio', 'open_close_ratio', 'body_size', 'ema_5', 
                'price_ema_ratio_5', 'ema_10', 'price_ema_ratio_10', 'ema_20', 
                'price_ema_ratio_20', 'macd', 'macd_signal', 'macd_histogram', 
                'rsi_7_safe', 'rsi_14_safe', 'bb_upper_safe', 'bb_lower_safe', 
                'bb_middle_safe', 'bb_position_safe', 'volume_lag1', 
                'volume_ratio_safe', 'resistance_20', 'support_20', 'breakout_high', 
                'breakout_low'
            ]
            
            # 确保所有特征都存在
            for feature in model_features:
                if feature not in latest_data.columns:
                    latest_data[feature] = 0.0
            
            # 只使用模型需要的特征，并确保没有NaN值
            feature_cols = model_features
            latest_data[feature_cols] = latest_data[feature_cols].fillna(0.0)
            
            # 模型预测
            if isinstance(self.model, list):
                # 集成模型
                predictions = []
                for model in self.model:
                    pred = model.predict_proba(latest_data[feature_cols])[:, 1]
                    predictions.append(pred)
                prob_up = np.mean(predictions, axis=0)[0]
            else:
                # 单个模型
                prob_up = self.model.predict_proba(latest_data[feature_cols])[:, 1][0]
            
            prob_down = 1 - prob_up
            
            # 获取阈值
            buy_threshold = self.config['model']['proba_threshold']
            sell_threshold = self.config['model'].get('sell_threshold', 0.8)
            min_signal_strength = self.config['model'].get('min_signal_strength', 0.3)
            
            # 信号强度
            signal_strength = abs(prob_up - prob_down)
            
            # 生成信号 - 确保买入和卖出信号不会同时触发
            buy_signal = prob_up > buy_threshold and signal_strength > min_signal_strength
            # sell_signal = prob_down > sell_threshold and signal_strength > min_signal_strength
            sell_signal = False
            buy_signal = True
            
            # 如果同时有买入和卖出信号，选择概率更高的那个
            if buy_signal and sell_signal:
                if prob_up > prob_down:
                    sell_signal = False  # 优先买入
                    self.logger.info(f"⚠️ 同时检测到买入和卖出信号，优先选择买入信号")
                else:
                    buy_signal = False  # 优先卖出
                    self.logger.info(f"⚠️ 同时检测到买入和卖出信号，优先选择卖出信号")
            
            self.logger.info(f"📊 信号分析: prob_up={prob_up:.3f}, prob_down={prob_down:.3f}, strength={signal_strength:.3f}")
            self.logger.info(f"🎯 信号结果: buy={buy_signal}, sell={sell_signal}")
            
            return buy_signal, sell_signal, signal_strength
            
        except Exception as e:
            self.logger.error(f"❌ 信号生成失败: {str(e)}")
            return False, False, 0.0
    
    def calculate_position_size(self, signal_strength: float, current_price: float) -> float:
        """计算仓位大小"""
        try:
            # 基础仓位
            base_cash = self.config['position_sizing']['fixed_cash_per_trade']
            
            # 基于信号强度的调整 - 限制在合理范围内
            strength_multiplier = np.clip(signal_strength, 0.1, 1.0)
            
            # 基于可用资金的调整
            usdt_balance = self.balance.get('USDT', 0)
            
            # 处理Balance对象或直接数值
            if hasattr(usdt_balance, 'free'):
                # 如果是Balance对象，使用free字段
                available_cash = usdt_balance.free
            else:
                # 如果是直接数值
                available_cash = float(usdt_balance)
            
            # 安全检查：确保可用资金大于0
            if available_cash <= 0:
                self.logger.warning(f"⚠️ 可用资金为0或负数: {available_cash}")
                return 0.0
            
            # 限制最大仓位为可用资金的20%，但不超过基础仓位
            max_cash = min(base_cash, available_cash * 0.2)
            
            # 计算仓位
            position_cash = base_cash * strength_multiplier
            position_cash = min(position_cash, max_cash)
            
            # 确保仓位金额不为0
            if position_cash <= 0:
                self.logger.warning(f"⚠️ 计算出的仓位金额为0: position_cash={position_cash}")
                return 0.0
            
            # 确保当前价格大于0
            if current_price <= 0:
                self.logger.error(f"❌ 当前价格无效: {current_price}")
                return 0.0
            
            position_size = position_cash / current_price
            
            # 安全检查：确保仓位大小合理
            if position_size <= 0:
                self.logger.warning(f"⚠️ 计算出的仓位大小为0或负数: {position_size}")
                return 0.0
            
            # 检查仓位是否超过可用资金
            required_cash = position_size * current_price
            if required_cash > available_cash:
                self.logger.warning(f"⚠️ 仓位过大，调整仓位: 需要{required_cash:.2f} USDT, 可用{available_cash:.2f} USDT")
                # 调整仓位到可用资金的90%
                position_size = (available_cash * 0.9) / current_price
            
            self.logger.info(f"💰 仓位计算: 基础={base_cash}, 强度倍数={strength_multiplier:.3f}, 可用资金={available_cash:.2f}, 仓位金额={position_cash:.2f}, 最终仓位={position_size:.6f}")
            
            return position_size
            
        except Exception as e:
            self.logger.error(f"❌ 仓位计算失败: {str(e)}")
            return 0.0
    
    def execute_trade(self, symbol: str, side: OrderSide, amount: float, price: float) -> Optional[Order]:
        """执行交易"""
        try:
            # 安全检查：验证输入参数
            if amount <= 0:
                self.logger.error(f"❌ 交易数量无效: {amount}")
                return None
            
            if price <= 0:
                self.logger.error(f"❌ 交易价格无效: {price}")
                return None
            
            # 安全检查：防止异常大的交易
            max_amount = 10000  # 最大交易数量
            if amount > max_amount:
                self.logger.error(f"❌ 交易数量过大: {amount} > {max_amount}")
                return None
            
            max_price = 1000  # 最大价格（USDT）
            if price > max_price:
                self.logger.error(f"❌ 交易价格异常: {price} > {max_price}")
                return None
            
            # 计算交易总价值
            total_value = amount * price
            max_value = 50000  # 最大单笔交易价值
            if total_value > max_value:
                self.logger.error(f"❌ 交易价值过大: {total_value:.2f} > {max_value}")
                return None
            
            self.logger.info(f"🔍 交易安全检查通过: 数量={amount:.6f}, 价格={price:.4f}, 总价值={total_value:.2f}")
            
            # 余额验证
            balance_valid, reject_reason = self._validate_balance(side, amount, price)
            if not balance_valid:
                # 创建拒绝的订单对象用于记录
                order_id = f"{side.value}_{int(time.time() * 1000)}"
                order = Order(
                    id=order_id,
                    symbol=symbol,
                    side=side,
                    amount=amount,
                    price=price,
                    status=OrderStatus.REJECTED,
                    timestamp=datetime.now()
                )
                order.reject_reason = reject_reason
                return order
            
            # 创建订单
            order_id = f"{side.value}_{int(time.time() * 1000)}"
            order = Order(
                id=order_id,
                symbol=symbol,
                side=side,
                amount=amount,
                price=price,
                status=OrderStatus.PENDING,
                timestamp=datetime.now()
            )
            
            self.logger.info(f"📝 创建订单: {order_id} - {side.value} {amount:.6f} {symbol} @ {price:.4f}")
            
            # 提交订单前添加调试信息
            self.logger.info(f"🔍 提交订单前状态: 订单ID={order_id}, 状态={order.status}")
            self.logger.info(f"🔍 当前余额详情: {self.balance}")
            self.logger.info(f"🔍 订单参数: 方向={side.value}, 数量={amount:.6f}, 价格={price:.4f}, 总价值={amount * price:.2f} USDT")
            self.logger.info(f"🔍 交易环境: 模式={self.mode.value}, 交易所={type(self.exchange).__name__}")
            
            # 提交订单
            if self.mode == TradingMode.PAPER:
                # 模拟交易
                self.logger.info("📝 使用模拟交易模式")
                success = self.exchange.place_order(order)
            else:
                # 实盘交易
                self.logger.info("📝 使用实盘交易模式")
                success = self.exchange.place_order(order)
            
            # 添加订单提交后的状态检查
            self.logger.info(f"🔍 订单提交后状态: 成功={success}, 订单状态={order.status}")
            self.logger.info(f"🔍 success类型: {type(success)}, 值: {repr(success)}")
            self.logger.info(f"🔍 即将进入判断: success={success}, bool(success)={bool(success)}")
            
            if success:
                self.logger.info(f"🎯 进入成功分支: success={success}")
                self.orders[order_id] = order
                self.logger.info(f"✅ 订单提交成功: {order_id}")
                return order
            else:
                self.logger.info(f"🎯 进入失败分支: success={success}")
                # 安全地获取状态值
                status_value = order.status.value if hasattr(order.status, 'value') else str(order.status)
                
                # 获取详细的拒绝原因
                reject_reason = self._get_reject_reason(order)
                
                # 添加更详细的诊断信息
                # 处理Balance对象，提取数值
                usdt_balance = self.balance.get('USDT', 0)
                fil_balance = self.balance.get('FIL', 0)
                
                # 如果余额是Balance对象，提取可用余额
                if hasattr(usdt_balance, 'free'):
                    usdt_value = usdt_balance.free
                elif hasattr(usdt_balance, 'total'):
                    usdt_value = usdt_balance.total
                else:
                    usdt_value = float(usdt_balance)
                
                if hasattr(fil_balance, 'free'):
                    fil_value = fil_balance.free
                elif hasattr(fil_balance, 'total'):
                    fil_value = fil_balance.total
                else:
                    fil_value = float(fil_balance)
                
                self.logger.error(f"❌ 订单提交失败: {order_id} - 状态: {status_value}")
                self.logger.error(f"❌ 拒绝原因: {reject_reason}")
                self.logger.error(f"🔍 订单详情: 方向={order.side.value}, 数量={order.amount:.6f}, 价格={order.price:.4f}, 总价值={order.amount * order.price:.2f} USDT")
                self.logger.error(f"🔍 当前余额: USDT={usdt_value:.2f}, FIL={fil_value:.6f}")
                self.logger.error(f"🔍 交易模式: {self.mode.value}")
                self.logger.error(f"🔍 交易所类型: {type(self.exchange).__name__}")
                
                # 控制台输出
                print(f"❌ 订单提交失败: {order_id} - 状态: {status_value}")
                print(f"❌ 拒绝原因: {reject_reason}")
                print(f"🔍 订单详情: 方向={order.side.value}, 数量={order.amount:.6f}, 价格={order.price:.4f}, 总价值={order.amount * order.price:.2f} USDT")
                print(f"🔍 当前余额: USDT={usdt_value:.2f}, FIL={fil_value:.6f}")
                print(f"🔍 交易模式: {self.mode.value}")
                print(f"🔍 交易所类型: {type(self.exchange).__name__}")
                
                return None
                
        except Exception as e:
            self.logger.error(f"❌ 交易执行失败: {str(e)}")
            import traceback
            self.logger.error(f"详细错误: {traceback.format_exc()}")
            return None
    
    def _get_reject_reason(self, order: Order) -> str:
        """获取订单拒绝的详细原因"""
        try:
            # 首先检查订单对象是否已经存储了拒绝原因
            if hasattr(order, 'reject_reason') and order.reject_reason:
                return order.reject_reason
            
            # 如果没有存储的拒绝原因，则重新分析
            if order.side == OrderSide.BUY:
                cost = order.amount * order.price
                available_usdt = self.balance.get('USDT', 0)
                
                # 处理Balance对象或直接数值
                if hasattr(available_usdt, 'free'):
                    available_usdt = available_usdt.free
                else:
                    available_usdt = float(available_usdt)
                
                # 计算余额差异
                balance_diff = available_usdt - cost
                balance_ratio = (available_usdt / cost * 100) if cost > 0 else 0
                
                if cost > available_usdt:
                    return f"余额不足: 需要 {cost:.2f} USDT, 可用 {available_usdt:.2f} USDT, 差额: {balance_diff:.2f} USDT, 余额比例: {balance_ratio:.1f}%"
                else:
                    # 添加更多诊断信息
                    return f"买入订单被拒绝 - 成本: {cost:.2f} USDT, 可用: {available_usdt:.2f} USDT, 余额充足: {balance_diff:.2f} USDT, 订单状态: {order.status}, 可能原因: 交易所限制/网络问题/API错误"
                    
            elif order.side == OrderSide.SELL:
                available_fil = self.balance.get('FIL', 0)
                
                # 处理Balance对象或直接数值
                if hasattr(available_fil, 'free'):
                    available_fil = available_fil.free
                else:
                    available_fil = float(available_fil)
                
                # 计算余额差异
                balance_diff = available_fil - order.amount
                balance_ratio = (available_fil / order.amount * 100) if order.amount > 0 else 0
                
                if order.amount > available_fil:
                    return f"余额不足: 需要 {order.amount:.6f} FIL, 可用 {available_fil:.6f} FIL, 差额: {balance_diff:.6f} FIL, 余额比例: {balance_ratio:.1f}%"
                else:
                    # 添加更多诊断信息
                    return f"卖出订单被拒绝 - 数量: {order.amount:.6f} FIL, 可用: {available_fil:.6f} FIL, 余额充足: {balance_diff:.6f} FIL, 订单状态: {order.status}, 可能原因: 交易所限制/网络问题/API错误"
            else:
                return f"无效的订单方向: {order.side}, 支持的方向: BUY, SELL"
                
        except Exception as e:
            import traceback
            return f"获取拒绝原因时出错: {str(e)}, 详细错误: {traceback.format_exc()}"

    def _debug_balance(self, side: OrderSide, amount: float, price: float):
        """调试余额信息"""
        self.logger.info(f"🔍 余额调试信息:")
        self.logger.info(f"  - 订单方向: {side.value}")
        self.logger.info(f"  - 订单数量: {amount:.6f}")
        self.logger.info(f"  - 订单价格: {price:.4f}")
        self.logger.info(f"  - 订单总价值: {amount * price:.2f} USDT")
        self.logger.info(f"  - 当前余额: {self.balance}")
        self.logger.info(f"  - 交易模式: {self.mode.value}")
        self.logger.info(f"  - 交易所类型: {type(self.exchange).__name__}")
        
        if side == OrderSide.BUY:
            cost = amount * price
            available_usdt = self.balance.get('USDT', 0)
            
            # 处理余额对象
            if hasattr(available_usdt, 'free'):
                actual_usdt = available_usdt.free
                self.logger.info(f"  - USDT余额对象: 可用={available_usdt.free}, 冻结={getattr(available_usdt, 'used', 'N/A')}")
            else:
                actual_usdt = float(available_usdt)
                self.logger.info(f"  - USDT直接数值: {actual_usdt}")
            
            # 计算余额充足性
            balance_diff = actual_usdt - cost
            balance_ratio = (actual_usdt / cost * 100) if cost > 0 else 0
            
            self.logger.info(f"  - 买入成本: {cost:.2f} USDT")
            self.logger.info(f"  - 可用USDT: {actual_usdt:.2f} USDT")
            self.logger.info(f"  - 余额差异: {balance_diff:.2f} USDT")
            self.logger.info(f"  - 余额比例: {balance_ratio:.1f}%")
            self.logger.info(f"  - 余额充足: {'是' if balance_diff >= 0 else '否'}")
                
        elif side == OrderSide.SELL:
            available_fil = self.balance.get('FIL', 0)
            
            # 处理余额对象
            if hasattr(available_fil, 'free'):
                actual_fil = available_fil.free
                self.logger.info(f"  - FIL余额对象: 可用={available_fil.free}, 冻结={getattr(available_fil, 'used', 'N/A')}")
            else:
                actual_fil = float(available_fil)
                self.logger.info(f"  - FIL直接数值: {actual_fil}")
            
            # 计算余额充足性
            balance_diff = actual_fil - amount
            balance_ratio = (actual_fil / amount * 100) if amount > 0 else 0
            
            self.logger.info(f"  - 卖出数量: {amount:.6f} FIL")
            self.logger.info(f"  - 可用FIL: {actual_fil:.6f} FIL")
            self.logger.info(f"  - 余额差异: {balance_diff:.6f} FIL")
            self.logger.info(f"  - 余额比例: {balance_ratio:.1f}%")
            self.logger.info(f"  - 余额充足: {'是' if balance_diff >= 0 else '否'}")

    def _validate_balance(self, side: OrderSide, amount: float, price: float) -> Tuple[bool, str]:
        """验证余额是否足够"""
        try:
            # 添加调试信息
            self._debug_balance(side, amount, price)
            
            if side == OrderSide.BUY:
                cost = amount * price
                available_usdt = self.balance.get('USDT', 0)
                
                # 处理Balance对象或直接数值
                if hasattr(available_usdt, 'free'):
                    available_usdt = available_usdt.free
                else:
                    available_usdt = float(available_usdt)
                
                if cost > available_usdt:
                    reject_reason = f"余额不足: 需要 {cost:.2f} USDT, 可用 {available_usdt:.2f} USDT"
                    self.logger.error(f"❌ {reject_reason}")
                    return False, reject_reason
                    
            elif side == OrderSide.SELL:
                available_fil = self.balance.get('FIL', 0)
                
                # 处理Balance对象或直接数值
                if hasattr(available_fil, 'free'):
                    available_fil = available_fil.free
                else:
                    available_fil = float(available_fil)
                
                if amount > available_fil:
                    reject_reason = f"余额不足: 需要 {amount:.6f} FIL, 可用 {available_fil:.6f} FIL"
                    self.logger.error(f"❌ {reject_reason}")
                    return False, reject_reason
            
            return True, ""
            
        except Exception as e:
            self.logger.error(f"❌ 余额验证失败: {str(e)}")
            return False, f"余额验证异常: {str(e)}"
    
    def update_positions(self):
        """更新持仓信息"""
        try:
            if self.mode == TradingMode.PAPER:
                # 模拟持仓更新
                self.exchange.update_positions()
            else:
                # 获取真实持仓
                positions = self.exchange.get_positions()
                # 确保positions是Position对象格式
                self.positions = {}
                for symbol, pos_data in positions.items():
                    if isinstance(pos_data, Position):
                        self.positions[symbol] = pos_data
                    elif isinstance(pos_data, dict):
                        # 将字典转换为Position对象
                        self.positions[symbol] = Position(
                            symbol=pos_data.get('symbol', symbol),
                            amount=pos_data.get('amount', 0),
                            entry_price=pos_data.get('entry_price', 0),
                            current_price=pos_data.get('current_price', 0),
                            unrealized_pnl=pos_data.get('unrealized_pnl', 0),
                            timestamp=datetime.now(),
                            side=pos_data.get('side'),
                            entry_time=pos_data.get('entry_time'),
                            stop_loss=pos_data.get('stop_loss'),
                            take_profit=pos_data.get('take_profit'),
                            trailing_stop=pos_data.get('trailing_stop'),
                            trailing_stop_activated=pos_data.get('trailing_stop_activated', False),
                            highest_price=pos_data.get('highest_price'),
                            lowest_price=pos_data.get('lowest_price'),
                            trailing_distance=pos_data.get('trailing_distance')
                        )
            
            # 更新账户余额
            self.balance = self.exchange.get_balance()
            
            # 检查移动止损
            self.check_trailing_stops()
            
        except Exception as e:
            self.logger.error(f"❌ 持仓更新失败: {str(e)}")
    
    def check_trailing_stops(self):
        """检查移动止损"""
        try:
            if not self.positions:
                return
            
            for symbol, position in self.positions.items():
                # 检查position是否为Position对象
                if not isinstance(position, Position):
                    # 如果position是字典，跳过移动止损检查
                    self.logger.debug(f"⚠️ 跳过移动止损检查: {symbol} - position不是Position对象")
                    continue
                
                # 获取当前价格
                current_price = self.exchange.get_current_price(symbol)
                if current_price == 0:
                    continue
                
                # 计算ATR
                df = self.get_market_data(symbol, '15m', 20)
                if df.empty:
                    continue
                
                atr_series = self._calculate_atr(df)
                atr_value = atr_series.iloc[-1] if len(atr_series) > 0 else 0.01
                
                # 更新移动止损
                if hasattr(self, 'risk_manager') and self.risk_manager:
                    # 将Position对象转换为字典格式
                    position_dict = {
                        'symbol': position.symbol,
                        'amount': position.amount,
                        'entry_price': position.entry_price,
                        'current_price': current_price,
                        'side': position.side,
                        'trailing_stop': position.trailing_stop,
                        'trailing_stop_activated': position.trailing_stop_activated,
                        'highest_price': position.highest_price,
                        'lowest_price': position.lowest_price,
                        'trailing_distance': position.trailing_distance
                    }
                    
                    new_trailing_stop, triggered = self.risk_manager.update_trailing_stop(
                        position_dict, current_price, atr_value
                    )
                    
                    # 更新Position对象
                    position.trailing_stop = new_trailing_stop
                    position.trailing_stop_activated = position_dict['trailing_stop_activated']
                    position.highest_price = position_dict['highest_price']
                    position.lowest_price = position_dict['lowest_price']
                    position.trailing_distance = position_dict['trailing_distance']
                    
                    # 保存持仓状态
                    if hasattr(self, 'save_position_state'):
                        self.save_position_state(position)
                    
                    # 如果触发移动止损，执行平仓
                    if triggered:
                        self.logger.warning(f"🚨 移动止损触发，平仓 {symbol}")
                        self.execute_trade(symbol, OrderSide.SELL, position.amount, current_price)
                        
        except Exception as e:
            self.logger.error(f"❌ 移动止损检查失败: {str(e)}")
    
    def risk_management(self) -> bool:
        """风险管理检查"""
        try:
            # 检查最大回撤
            current_value = self.get_portfolio_value()
            initial_value = self.config['backtest']['initial_cash']
            
            if current_value < initial_value * 0.8:  # 回撤超过20%
                self.logger.warning("⚠️ 回撤超过20%，停止交易")
                return False
            
            # 检查单笔交易风险
            risk_config = self.config.get('risk_management', {})
            max_risk_per_trade = risk_config.get('max_position_risk', 0.01)
            # 这里可以添加更详细的风险检查
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 风险管理检查失败: {str(e)}")
            return False
    
    def get_portfolio_value(self) -> float:
        """获取投资组合总价值"""
        try:
            if self.mode == TradingMode.PAPER:
                return self.exchange.get_portfolio_value()
            else:
                # 计算真实投资组合价值
                total_value = 0
                for symbol, position in self.positions.items():
                    if not isinstance(position, Position):
                        continue
                    current_price = self.exchange.get_current_price(symbol)
                    total_value += position.amount * current_price
                
                # 处理USDT余额
                usdt_balance = self.balance.get('USDT', 0)
                if hasattr(usdt_balance, 'free'):
                    # 如果是Balance对象，使用free字段
                    total_value += usdt_balance.free
                else:
                    # 如果是直接数值
                    total_value += float(usdt_balance)
                
                return total_value
        except Exception as e:
            self.logger.error(f"❌ 获取投资组合价值失败: {str(e)}")
            return 0.0
    
    def run_trading_cycle(self):
        """运行一个交易周期"""
        try:
            symbol = self.config['trading']['symbol']
            timeframe = self.config['trading']['timeframe']
            
            # 获取市场数据
            df = self.get_market_data(symbol, timeframe, 100)
            if df.empty:
                self.logger.warning("⚠️ 无法获取市场数据")
                return
            
            # 添加数据验证 - 检查原始数据
            self.logger.info(f"🔍 原始数据验证: close列后5个值={df['close'].tail(5).tolist()}")
            self.logger.info(f"🔍 DataFrame列名: {df.columns.tolist()}")
            
            # 计算特征
            df = self.calculate_features(df)
            
            # 添加数据验证
            self.logger.info(f"🔍 特征计算后数据验证: close列后5个值={df['close'].tail(5).tolist()}")
            
            # 生成信号
            buy_signal, sell_signal, signal_strength = self.generate_signal(df)
            
            # 更新持仓
            self.update_positions()
            
            # 风险管理检查
            if not self.risk_management():
                return
            
            current_price = df['close'].iloc[-1]
            
            # 添加价格验证和调试
            self.logger.info(f"🔍 价格调试: 原始价格={current_price}, 类型={type(current_price)}")
            
            # 确保价格是正确的数值
            current_price = float(current_price)
            self.logger.info(f"🔍 转换后价格: {current_price}")
            
            # 执行交易逻辑
            if buy_signal and not self.positions.get(symbol):
                # 买入信号且无持仓
                position_size = self.calculate_position_size(signal_strength, current_price)
                if position_size > 0:
                    order = self.execute_trade(symbol, OrderSide.BUY, position_size, current_price)
                    if order and order.status == OrderStatus.FILLED:
                        self.logger.info(f"🟢 买入信号执行: {position_size:.6f} {symbol} @ {current_price:.4f}")
                        
                        # 初始化移动止损
                        if hasattr(self, 'risk_manager') and self.risk_manager:
                            atr_series = self._calculate_atr(df)
                            atr_value = atr_series.iloc[-1] if len(atr_series) > 0 else 0.01
                            position_dict = {
                                'symbol': symbol,
                                'amount': position_size,
                                'entry_price': current_price,
                                'current_price': current_price,
                                'side': 'buy'
                            }
                            self.risk_manager.initialize_trailing_stop(
                                position_dict, current_price, current_price, atr_value, 'buy'
                            )
                            
                            # 更新持仓信息
                            if symbol in self.positions:
                                position = self.positions[symbol]
                                position.trailing_stop = position_dict['trailing_stop']
                                position.trailing_stop_activated = position_dict['trailing_stop_activated']
                                position.highest_price = position_dict['highest_price']
                                position.lowest_price = position_dict['lowest_price']
                                position.trailing_distance = position_dict['trailing_distance']
                else:
                    self.logger.warning(f"⚠️ 买入信号但仓位大小为0，跳过交易")
            
            elif sell_signal and self.positions.get(symbol):
                # 卖出信号且有持仓
                position = self.positions[symbol]
                if not isinstance(position, Position):
                    self.logger.warning(f"⚠️ 持仓数据格式错误: {symbol}")
                    return
                order = self.execute_trade(symbol, OrderSide.SELL, position.amount, current_price)
                if order:
                    self.logger.info(f"🔴 卖出信号执行: {position.amount:.6f} {symbol} @ {current_price:.4f}")
                else:
                    self.logger.warning(f"⚠️ 卖出信号但交易执行失败")
            
            elif sell_signal and not self.positions.get(symbol):
                # 卖出信号但无持仓 - 这是正常情况，记录但不执行
                self.logger.info(f"📊 检测到卖出信号但当前无持仓，跳过交易")
            
            elif buy_signal and self.positions.get(symbol):
                # 买入信号但已有持仓 - 记录但不执行
                self.logger.info(f"📊 检测到买入信号但已有持仓，跳过交易")
            
            # 记录状态
            portfolio_value = self.get_portfolio_value()
            self.logger.info(f"💰 投资组合价值: ${portfolio_value:.2f}")
            
        except Exception as e:
            self.logger.error(f"❌ 交易周期执行失败: {str(e)}")
    
    def start_trading(self):
        """开始交易"""
        self.logger.info("🚀 开始实盘交易...")
        self.is_running = True
        
        # 启动WebSocket连接
        if self.enable_websocket and self.websocket_manager:
            symbol = self.config.get('symbol', 'FIL/USDT')
            self.websocket_manager.start(symbol)
            self.logger.info(f"✅ WebSocket连接已启动: {symbol}")
        
        while self.is_running:
            try:
                # 使用混合时间框架策略
                if self.use_hybrid_stops:
                    self.run_hybrid_trading_cycle()
                else:
                    self.run_trading_cycle()
                
                # 混合策略的等待时间：更频繁的检查
                if self.use_hybrid_stops:
                    # 每60秒检查一次（1分钟止损检查间隔）
                    sleep_time = self.stop_check_interval
                    self.logger.info(f"⏰ 混合策略运行中，{self.stop_check_interval}秒后再次检查...")
                else:
                    # 原有的15分钟等待逻辑
                    current_time = datetime.now()
                    current_minute = current_time.minute
                    
                    # 计算到下一个关键时间点的分钟数
                    if current_minute < 15:
                        next_minute = 15
                    elif current_minute < 30:
                        next_minute = 30
                    elif current_minute < 45:
                        next_minute = 45
                    else:
                        next_minute = 60  # 下一个小时的0分
                    
                    # 计算需要等待的秒数
                    minutes_to_wait = next_minute - current_minute
                    if minutes_to_wait <= 0:
                        minutes_to_wait = 15  # 如果计算错误，默认15分钟
                    
                    sleep_time = minutes_to_wait * 60
                    self.logger.info(f"⏰ 当前时间: {current_time.strftime('%H:%M')}, 等待 {minutes_to_wait} 分钟到 {next_minute:02d} 分进行下一轮分析...")
                
                time.sleep(sleep_time)
                
            except KeyboardInterrupt:
                self.logger.info("⏹️ 收到停止信号")
                break
            except Exception as e:
                self.logger.error(f"❌ 交易循环错误: {str(e)}")
                time.sleep(60)  # 错误后等待1分钟
        
        self.logger.info("🏁 交易系统已停止")
    
    def stop_trading(self):
        """停止交易"""
        self.is_running = False
        
        # 停止WebSocket连接
        if self.enable_websocket and self.websocket_manager:
            self.websocket_manager.stop_all()
            self.logger.info("✅ WebSocket连接已停止")
        
        self.logger.info("⏹️ 交易系统停止中...")
    
    def get_websocket_status(self) -> Dict:
        """获取WebSocket状态"""
        if not self.enable_websocket or not self.websocket_manager:
            return {'enabled': False, 'message': 'WebSocket未启用'}
        
        try:
            status = self.websocket_manager.get_connection_status()
            statistics = self.websocket_manager.get_statistics()
            
            return {
                'enabled': True,
                'connections': status,
                'statistics': statistics
            }
        except Exception as e:
            return {
                'enabled': True,
                'error': str(e)
            }
    
    def get_latest_websocket_data(self, symbol: str) -> Optional[Dict]:
        """获取最新WebSocket数据"""
        if not self.enable_websocket or not self.websocket_manager:
            return None
        
        try:
            return self.websocket_manager.get_latest_data(symbol)
        except Exception as e:
            self.logger.error(f"❌ 获取WebSocket数据失败: {str(e)}")
            return None
    
    def is_15m_signal_time(self) -> bool:
        """检查是否是15分钟信号时间"""
        try:
            current_time = datetime.now()
            current_minute = current_time.minute
            
            # 15分钟信号时间：0, 15, 30, 45分钟
            signal_minutes = [0, 15, 30, 45]
            
            if current_minute in signal_minutes:
                # 检查是否已经处理过这个时间点的信号
                if self.last_signal_time:
                    time_diff = (current_time - self.last_signal_time).total_seconds()
                    if time_diff < 60:  # 1分钟内不重复处理
                        return False
                
                self.last_signal_time = current_time
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"❌ 检查15分钟信号时间失败: {str(e)}")
            return False
    
    def is_1m_stop_check_time(self) -> bool:
        """检查是否是1分钟止损检查时间"""
        try:
            current_time = datetime.now()
            
            # 检查间隔
            if self.last_stop_check_time:
                time_diff = (current_time - self.last_stop_check_time).total_seconds()
                if time_diff < self.stop_check_interval:
                    return False
            
            self.last_stop_check_time = current_time
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 检查1分钟止损时间失败: {str(e)}")
            return False
    
    def _check_realtime_emergency_stop(self, symbol: str, current_price: float):
        """检查实时紧急止损"""
        try:
            if symbol not in self.positions:
                return
            
            position = self.positions[symbol]
            if not isinstance(position, Position):
                return
                
            if not position.entry_price:
                return
            
            # 计算紧急止损价格
            entry_price = position.entry_price
            emergency_stop_price = entry_price * (1 - self.emergency_stop_loss)
            
            # 检查是否触发紧急止损
            if current_price <= emergency_stop_price:
                self.logger.warning(f"🚨 实时紧急止损触发: {symbol} - 价格: {current_price:.4f}, 止损价: {emergency_stop_price:.4f}")
                self._execute_emergency_stop(symbol, current_price)
            
        except Exception as e:
            self.logger.error(f"❌ 实时紧急止损检查失败: {str(e)}")
    
    def _execute_emergency_stop(self, symbol: str, current_price: float):
        """执行紧急止损"""
        try:
            if symbol not in self.positions:
                return
                
            position = self.positions[symbol]
            if not isinstance(position, Position):
                return
                
            amount = position.amount
            
            if amount > 0:
                # 执行紧急止损
                order = self.execute_trade(symbol, OrderSide.SELL, amount, current_price)
                if order:
                    self.logger.warning(f"🚨 紧急止损执行: {symbol} - {amount:.6f} @ {current_price:.4f}")
                    # 清空持仓
                    if symbol in self.positions:
                        del self.positions[symbol]
                else:
                    self.logger.error(f"❌ 紧急止损执行失败: {symbol}")
            
        except Exception as e:
            self.logger.error(f"❌ 执行紧急止损失败: {str(e)}")
    
    def check_1m_stop_loss_with_websocket(self, symbol: str, current_price: float, df_1m: pd.DataFrame, websocket_data: Dict):
        """使用WebSocket 1分钟数据检查止损（已优化，推荐使用check_1m_stop_loss_with_websocket_optimized）"""
        try:
            # 重定向到优化版本
            self.logger.info("🔄 重定向到WebSocket优化版本")
            self.check_1m_stop_loss_with_websocket_optimized(symbol, current_price, websocket_data)
            
        except Exception as e:
            self.logger.error(f"❌ WebSocket 1分钟止损检查失败: {str(e)}")
    
    def check_1m_stop_loss_with_websocket_optimized(self, symbol: str, current_price: float, websocket_data: Dict):
        """使用WebSocket数据检查1分钟止损（完全基于WebSocket，无REST API依赖）"""
        try:
            if symbol not in self.positions:
                return
            
            position = self.positions[symbol]
            if not isinstance(position, Position):
                return
                
            if not position.entry_price:
                return
            
            # 使用WebSocket缓存数据计算ATR
            if hasattr(self, 'data_fetcher') and self.data_fetcher:
                atr_1m = self.data_fetcher.calculate_atr_from_websocket(symbol, 14)
            else:
                # 备用方案：使用价格估算ATR
                atr_1m = current_price * 0.01
                self.logger.warning(f"⚠️ 无法获取WebSocket ATR，使用价格估算: {atr_1m:.6f}")
            
            if atr_1m <= 0:
                self.logger.warning(f"⚠️ ATR值无效: {atr_1m}")
                return
            
            # 计算1分钟止损价格
            entry_price = position.entry_price
            stop_distance = atr_1m * self.stop_loss_1m_atr_mult
            stop_loss_price = entry_price - stop_distance
            
            # 使用WebSocket的实时价格检查是否触发止损
            self.logger.debug(f"📊 WebSocket优化1分钟止损检查: {symbol}")
            self.logger.debug(f"   当前价: {current_price:.4f}")
            self.logger.debug(f"   入场价: {entry_price:.4f}")
            self.logger.debug(f"   ATR: {atr_1m:.6f}")
            self.logger.debug(f"   止损距离: {stop_distance:.6f}")
            self.logger.debug(f"   止损价: {stop_loss_price:.4f}")
            
            if current_price <= stop_loss_price:
                self.logger.warning(f"🚨 WebSocket优化1分钟止损触发: {symbol}")
                self.logger.warning(f"   价格: {current_price:.4f} <= 止损价: {stop_loss_price:.4f}")
                self._execute_1m_stop_loss(symbol, current_price)
            
            # 更新1分钟移动止损
            if self.trailing_stop_1m and position.trailing_stop_activated:
                self._update_1m_trailing_stop_optimized(position, current_price, atr_1m)
            
        except Exception as e:
            self.logger.error(f"❌ WebSocket优化1分钟止损检查失败: {str(e)}")
    
    def check_1m_stop_loss(self, symbol: str, current_price: float):
        """使用1分钟数据检查止损（使用REST API）"""
        try:
            if symbol not in self.positions:
                return
            
            position = self.positions[symbol]
            if not position.get('entry_price'):
                return
            
            # 获取1分钟数据计算ATR
            df_1m = self.get_market_data(symbol, '1m', 20)
            if df_1m.empty:
                return
            
            atr_1m = self._calculate_atr(df_1m)
            if atr_1m.empty or atr_1m.iloc[-1] == 0:
                return
            
            # 计算1分钟止损价格
            entry_price = position['entry_price']
            stop_distance = atr_1m.iloc[-1] * self.stop_loss_1m_atr_mult
            stop_loss_price = entry_price - stop_distance
            
            # 检查是否触发止损
            if current_price <= stop_loss_price:
                self.logger.warning(f"🚨 1分钟止损触发: {symbol} - 价格: {current_price:.4f}, 止损价: {stop_loss_price:.4f}")
                self._execute_1m_stop_loss(symbol, current_price)
            
            # 更新1分钟移动止损
            if self.trailing_stop_1m and position.get('trailing_stop_activated'):
                self._update_1m_trailing_stop(position, current_price, atr_1m.iloc[-1])
            
        except Exception as e:
            self.logger.error(f"❌ 1分钟止损检查失败: {str(e)}")
    
    def _execute_1m_stop_loss(self, symbol: str, current_price: float):
        """执行1分钟止损"""
        try:
            position = self.positions[symbol]
            amount = position.get('amount', 0)
            
            if amount > 0:
                # 执行1分钟止损
                order = self.execute_trade(symbol, OrderSide.SELL, amount, current_price)
                if order:
                    self.logger.warning(f"🚨 1分钟止损执行: {symbol} - {amount:.6f} @ {current_price:.4f}")
                    # 清空持仓
                    if symbol in self.positions:
                        del self.positions[symbol]
                else:
                    self.logger.error(f"❌ 1分钟止损执行失败: {symbol}")
            
        except Exception as e:
            self.logger.error(f"❌ 执行1分钟止损失败: {str(e)}")
    
    def _update_1m_trailing_stop(self, position: Dict, current_price: float, atr_1m: float):
        """更新1分钟移动止损"""
        try:
            if not position.get('trailing_stop_activated'):
                return
            
            entry_price = position['entry_price']
            side = position.get('side', 'buy')
            
            # 计算1分钟移动止损距离
            trailing_distance = atr_1m * self.stop_loss_1m_atr_mult
            
            if side == 'buy':
                # 多头移动止损
                new_stop = current_price - trailing_distance
                if new_stop > position.get('stop_loss', 0):
                    position['stop_loss'] = new_stop
                    self.logger.info(f"📈 更新1分钟移动止损: {new_stop:.4f}")
            
        except Exception as e:
            self.logger.error(f"❌ 更新1分钟移动止损失败: {str(e)}")
    
    def _update_1m_trailing_stop_optimized(self, position: Position, current_price: float, atr_1m: float):
        """更新1分钟移动止损（基于WebSocket数据优化）"""
        try:
            if not position.trailing_stop_activated:
                return
            
            entry_price = position.entry_price
            side = position.side
            
            # 计算1分钟移动止损距离
            trailing_distance = atr_1m * self.stop_loss_1m_atr_mult
            
            if side == 'buy':
                # 多头移动止损
                new_stop = current_price - trailing_distance
                if new_stop > (position.stop_loss or 0):
                    position.stop_loss = new_stop
                    self.logger.info(f"📈 WebSocket优化更新1分钟移动止损: {new_stop:.4f}")
                    
                    # 更新最高价记录
                    if not position.highest_price or current_price > position.highest_price:
                        position.highest_price = current_price
                        self.logger.debug(f"📊 更新最高价: {current_price:.4f}")
            
            elif side == 'sell':
                # 空头移动止损
                new_stop = current_price + trailing_distance
                if new_stop < (position.stop_loss or float('inf')):
                    position.stop_loss = new_stop
                    self.logger.info(f"📉 WebSocket优化更新1分钟移动止损: {new_stop:.4f}")
                    
                    # 更新最低价记录
                    if not position.lowest_price or current_price < position.lowest_price:
                        position.lowest_price = current_price
                        self.logger.debug(f"📊 更新最低价: {current_price:.4f}")
            
            # 保存持仓状态
            if hasattr(self, 'save_position_state'):
                self.save_position_state(position)
            
        except Exception as e:
            self.logger.error(f"❌ WebSocket优化更新1分钟移动止损失败: {str(e)}")
    
    def run_hybrid_trading_cycle(self):
        """运行混合时间框架交易周期"""
        try:
            symbol = self.config['trading']['symbol']
            
            # 1. 15分钟信号生成
            if self.is_15m_signal_time():
                self.logger.info("🕐 15分钟信号时间，生成交易信号")
                df_15m = self.get_market_data(symbol, self.signal_timeframe, 100)
                if not df_15m.empty:
                    # 计算特征
                    df_15m = self.calculate_features(df_15m)
                    
                    # 生成信号
                    buy_signal, sell_signal, signal_strength = self.generate_signal(df_15m)
                    
                    current_price = df_15m['close'].iloc[-1]
                    
                    # 执行交易逻辑
                    if buy_signal and not self.positions.get(symbol):
                        position_size = self.calculate_position_size(signal_strength, current_price)
                        if position_size > 0:
                            order = self.execute_trade(symbol, OrderSide.BUY, position_size, current_price)
                            if order and order.status == OrderStatus.FILLED:
                                self.logger.info(f"🟢 15分钟买入信号执行: {position_size:.6f} {symbol} @ {current_price:.4f}")
                                
                                # 初始化移动止损
                                if hasattr(self, 'risk_manager') and self.risk_manager:
                                    atr_series = self._calculate_atr(df_15m)
                                    atr_value = atr_series.iloc[-1] if len(atr_series) > 0 else 0.01
                                    position_dict = {
                                        'symbol': symbol,
                                        'amount': position_size,
                                        'entry_price': current_price,
                                        'current_price': current_price,
                                        'side': 'buy'
                                    }
                                    self.risk_manager.initialize_trailing_stop(
                                        position_dict, current_price, current_price, atr_value, 'buy'
                                    )
                    
                    elif sell_signal and self.positions.get(symbol):
                        position = self.positions[symbol]
                        order = self.execute_trade(symbol, OrderSide.SELL, position.amount, current_price)
                        if order:
                            self.logger.info(f"🔴 15分钟卖出信号执行: {position.amount:.6f} {symbol} @ {current_price:.4f}")
                    else:
                        self.logger.debug(f"🔍 15分钟信号时间，没有持仓，不执行交易")
            # 2. 1分钟止损检查（根据配置选择WebSocket或REST API方式）
            if self.is_1m_stop_check_time() and self.use_hybrid_stops:
                # 检查是否仅使用WebSocket进行1分钟止损检查
                if self.websocket_only_1m_stops:
                    self.logger.debug("🕐 1分钟止损检查时间（WebSocket专用模式，跳过REST API）")
                    # WebSocket方式的1分钟止损检查在_handle_realtime_data中处理
                else:
                    self.logger.debug("🕐 1分钟止损检查时间（REST API方式）")
                    df_1m = self.get_market_data(symbol, self.stop_timeframe, 20)
                    if not df_1m.empty:
                        current_price = df_1m['close'].iloc[-1]
                        self.check_1m_stop_loss(symbol, current_price)
            
            # 3. 更新持仓
            self.update_positions()
            
            # 4. 风险管理检查
            if not self.risk_management():
                return
            
            # 5. 记录状态
            portfolio_value = self.get_portfolio_value()
            self.logger.info(f"💰 投资组合价值: ${portfolio_value:.2f}")
            
        except Exception as e:
            self.logger.error(f"❌ 混合交易周期执行失败: {str(e)}")
    
    def get_balance(self) -> Dict[str, float]:
        """获取账户余额"""
        return self.balance.copy()
    
    def get_trading_costs_summary(self) -> Dict[str, float]:
        """获取交易成本统计"""
        total_commission = 0.0
        total_slippage = 0.0
        total_trades = 0
        
        # 统计所有已成交订单的交易成本
        # 处理字典格式的订单（LiveTradingSystem）或列表格式的订单（PaperTradingExchange）
        orders_to_check = self.orders.values() if isinstance(self.orders, dict) else self.orders
        
        for order in orders_to_check:
            if order.status == OrderStatus.FILLED and hasattr(order, 'trading_costs'):
                total_commission += order.trading_costs.get('commission', 0)
                total_slippage += order.trading_costs.get('slippage', 0)
                total_trades += 1
        
        return {
            'total_commission': total_commission,
            'total_slippage': total_slippage,
            'total_trading_costs': total_commission + total_slippage,
            'total_trades': total_trades,
            'avg_commission_per_trade': total_commission / max(total_trades, 1),
            'avg_slippage_per_trade': total_slippage / max(total_trades, 1)
        }
    
    def display_balance_with_costs(self):
        """显示余额和交易成本统计"""
        balance = self.get_balance()
        costs = self.get_trading_costs_summary()
        
        print(f"\n💰 账户余额:")
        print(f"   USDT: {balance['USDT']:.2f}")
        print(f"   FIL:  {balance['FIL']:.6f}")
        
        if costs['total_trades'] > 0:
            print(f"\n📊 交易成本统计:")
            print(f"   总交易次数: {costs['total_trades']}")
            print(f"   总手续费: {costs['total_commission']:.2f} USDT")
            print(f"   总滑点成本: {costs['total_slippage']:.2f} USDT")
            print(f"   总交易成本: {costs['total_trading_costs']:.2f} USDT")
            print(f"   平均手续费/交易: {costs['avg_commission_per_trade']:.2f} USDT")
            print(f"   平均滑点/交易: {costs['avg_slippage_per_trade']:.2f} USDT")
            
            # 计算成本占比
            try:
                current_price = self.get_current_price()
                total_balance = balance['USDT'] + balance['FIL'] * current_price
                if total_balance > 0:
                    cost_ratio = costs['total_trading_costs'] / total_balance * 100
                    print(f"   交易成本占比: {cost_ratio:.2f}%")
            except:
                pass


class PaperTradingExchange:
    """模拟交易交易所"""
    
    def __init__(self, config):
        self.config = config
        self.positions = {}
        self.balance = {
            'USDT': config['backtest']['initial_cash'],
            'FIL': 0.0
        }
        self.orders = {}
        self.current_price = 0.0
    
    def get_klines(self, symbol: str, timeframe: str, limit: int) -> List:
        """获取K线数据（模拟） - 使用真实市场数据"""
        # 使用ccxt获取真实的市场数据
        import ccxt
        
        try:
            # 初始化交易所（使用binance获取真实数据）
            exchange = ccxt.binance({
                'enableRateLimit': True,
            })
            
            # 获取真实的K线数据
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            
            # 更新当前价格
            if ohlcv:
                self.current_price = float(ohlcv[-1][4])  # close price
            
            return ohlcv
            
        except Exception as e:
            print(f"❌ 获取真实市场数据失败: {str(e)}")
            # 如果获取失败，返回空列表
            return []
    
    def get_balance(self) -> Dict[str, float]:
        """获取账户余额"""
        return self.balance.copy()
    
    def get_current_price(self) -> float:
        """获取当前价格（模拟）"""
        # 返回一个模拟的当前价格
        return 5.0  # 假设FIL价格为5 USDT
    
    def get_trading_costs_summary(self) -> Dict[str, float]:
        """获取交易成本统计"""
        total_commission = 0.0
        total_slippage = 0.0
        total_trades = 0
        
        # 统计所有已成交订单的交易成本
        # 处理字典格式的订单（LiveTradingSystem）或列表格式的订单（PaperTradingExchange）
        orders_to_check = self.orders.values() if isinstance(self.orders, dict) else self.orders
        
        for order in orders_to_check:
            if order.status == OrderStatus.FILLED and hasattr(order, 'trading_costs'):
                total_commission += order.trading_costs.get('commission', 0)
                total_slippage += order.trading_costs.get('slippage', 0)
                total_trades += 1
        
        return {
            'total_commission': total_commission,
            'total_slippage': total_slippage,
            'total_trading_costs': total_commission + total_slippage,
            'total_trades': total_trades,
            'avg_commission_per_trade': total_commission / max(total_trades, 1),
            'avg_slippage_per_trade': total_slippage / max(total_trades, 1)
        }
    
    def display_balance_with_costs(self):
        """显示余额和交易成本统计"""
        balance = self.get_balance()
        costs = self.get_trading_costs_summary()
        
        print(f"\n💰 账户余额:")
        print(f"   USDT: {balance['USDT']:.2f}")
        print(f"   FIL:  {balance['FIL']:.6f}")
        
        if costs['total_trades'] > 0:
            print(f"\n📊 交易成本统计:")
            print(f"   总交易次数: {costs['total_trades']}")
            print(f"   总手续费: {costs['total_commission']:.2f} USDT")
            print(f"   总滑点成本: {costs['total_slippage']:.2f} USDT")
            print(f"   总交易成本: {costs['total_trading_costs']:.2f} USDT")
            print(f"   平均手续费/交易: {costs['avg_commission_per_trade']:.2f} USDT")
            print(f"   平均滑点/交易: {costs['avg_slippage_per_trade']:.2f} USDT")
            
            # 计算成本占比
            total_balance = balance['USDT'] + balance['FIL'] * self.get_current_price()
            if total_balance > 0:
                cost_ratio = costs['total_trading_costs'] / total_balance * 100
                print(f"   交易成本占比: {cost_ratio:.2f}%")
    
    def calculate_realistic_slippage(self, order: Order, current_price: float) -> float:
        """计算更真实的滑点，考虑市场波动性和订单大小"""
        try:
            # 基础滑点率
            base_slippage = self.config.get('slippage', 0.0001)
            
            # 获取市场波动性数据
            market_data = self.get_market_data()
            if market_data:
                # 计算最近的价格波动性
                prices = [float(bar['close']) for bar in market_data[-20:]]  # 最近20根K线
                if len(prices) > 1:
                    # 计算价格标准差作为波动性指标
                    price_std = np.std(prices)
                    volatility_factor = min(price_std / current_price, 0.01)  # 限制最大波动性影响
                else:
                    volatility_factor = 0.001
            else:
                volatility_factor = 0.001
            
            # 订单大小影响（大订单滑点更大）
            order_value = order.amount * current_price
            # 假设10,000 USDT为基准，大订单滑点增加
            size_factor = min(order_value / 10000, 2.0)  # 最大2倍滑点
            
            # 计算最终滑点
            final_slippage = base_slippage + volatility_factor * size_factor
            
            # 限制滑点范围（0.01% - 0.5%）
            final_slippage = max(0.0001, min(final_slippage, 0.005))
            
            return final_slippage
            
        except Exception as e:
            print(f"⚠️ 滑点计算异常，使用默认值: {str(e)}")
            return self.config.get('slippage', 0.0001)
    
    def simulate_partial_fill(self, order: Order, current_price: float) -> float:
        """模拟部分成交情况（大订单可能不会完全成交）"""
        try:
            # 计算订单相对市场深度的影响
            order_value = order.amount * current_price
            
            # 模拟市场深度：假设市场能承受的最大单笔交易
            max_single_trade = 50000  # 50,000 USDT
            
            if order_value <= max_single_trade:
                # 小订单完全成交
                return 1.0
            else:
                # 大订单部分成交，成交率随订单大小递减
                fill_ratio = max(0.3, min(0.9, max_single_trade / order_value))
                return fill_ratio
                
        except Exception as e:
            print(f"⚠️ 部分成交模拟异常: {str(e)}")
            return 1.0  # 默认完全成交
    
    def place_order(self, order: Order) -> bool:
        """下单（模拟）- 考虑滑点和手续费"""
        try:
            # 获取交易成本参数
            commission_rate = self.config.get('commission', 0.001)  # 默认0.1%手续费
            
            # 获取当前价格用于滑点计算
            current_price = self.get_current_price()
            
            # 计算动态滑点
            slippage_rate = self.calculate_realistic_slippage(order, current_price)
            
            # 模拟部分成交
            fill_ratio = self.simulate_partial_fill(order, current_price)
            actual_amount = order.amount * fill_ratio
          
            if order.side == OrderSide.BUY:
                # 计算滑点影响的实际成交价格（买入时价格可能上涨）
                actual_price = order.price * (1 + slippage_rate)
                
                # 计算总成本：实际成交数量 * 实际价格 + 手续费
                gross_cost = actual_amount * actual_price
                commission_fee = gross_cost * commission_rate
                total_cost = gross_cost + commission_fee
                order_value = order.amount * order.price  # 原始订单价值
                
                available_usdt = self.balance['USDT']
                
                # 处理Balance对象或直接数值
                if hasattr(available_usdt, 'free'):
                    available_usdt_value = available_usdt.free
                else:
                    available_usdt_value = float(available_usdt)
                
                print(f"🔍 买入订单检查:")
                print(f"   原始价格: {order.price:.4f} USDT")
                print(f"   实际价格: {actual_price:.4f} USDT (滑点: +{slippage_rate*100:.3f}%)")
                print(f"   订单价值: {order_value:.2f} USDT")
                print(f"   成交比例: {fill_ratio*100:.1f}% (实际数量: {actual_amount:.6f} FIL)")
                print(f"   总成本: {total_cost:.2f} USDT (含手续费: {commission_fee:.2f} USDT)")
                print(f"   可用余额: {available_usdt_value:.2f} USDT")
                
                # 添加日志记录
                self.logger.info(f"🔍 买入订单检查: 原始价格={order.price:.4f}, 实际价格={actual_price:.4f}, 滑点={slippage_rate*100:.3f}%, "
                           f"订单价值={order_value:.2f}, 成交比例={fill_ratio*100:.1f}%, 总成本={total_cost:.2f}, 可用余额={available_usdt_value:.2f}")
                
                if total_cost <= available_usdt_value:
                    # 扣除总成本
                    self.balance['USDT'] -= total_cost
                    # 增加FIL数量（使用实际成交数量）
                    self.balance['FIL'] += actual_amount
                    
                    # 更新订单状态
                    if fill_ratio >= 1.0:
                        order.status = OrderStatus.FILLED
                    else:
                        order.status = OrderStatus.PARTIALLY_FILLED
                    order.filled_amount = actual_amount
                    order.filled_price = actual_price
                    
                    # 记录交易成本
                    if not hasattr(order, 'trading_costs'):
                        order.trading_costs = {}
                    order.trading_costs = {
                        'commission': commission_fee,
                        'slippage': gross_cost - (actual_amount * order.price),
                        'total_cost': total_cost,
                        'fill_ratio': fill_ratio
                    }
                    
                    if fill_ratio >= 1.0:
                        print(f"✅ 买入订单完全成交: {actual_amount:.6f} FIL @ {actual_price:.4f} USDT")
                    else:
                        print(f"⚠️ 买入订单部分成交: {actual_amount:.6f}/{order.amount:.6f} FIL @ {actual_price:.4f} USDT")
                    print(f"   手续费: {commission_fee:.2f} USDT, 滑点成本: {order.trading_costs['slippage']:.2f} USDT")
                    return True
                else:
                    reject_reason = f"余额不足: 需要 {total_cost:.2f} USDT, 可用 {available_usdt_value:.2f} USDT"
                    print(f"❌ 买入订单失败: {reject_reason}")
                    order.status = OrderStatus.REJECTED
                    if not hasattr(order, 'reject_reason'):
                        order.reject_reason = reject_reason
                    return False
                    
            else:  # SELL
                # 计算滑点影响的实际成交价格（卖出时价格可能下跌）
                actual_price = order.price * (1 - slippage_rate)
                
                # 计算总收入：实际成交数量 * 实际价格
                gross_proceeds = actual_amount * actual_price
                commission_fee = gross_proceeds * commission_rate
                net_proceeds = gross_proceeds - commission_fee
                
                available_fil = self.balance['FIL']
                
                # 处理Balance对象或直接数值
                if hasattr(available_fil, 'free'):
                    available_fil_value = available_fil.free
                else:
                    available_fil_value = float(available_fil)
                
                print(f"🔍 卖出订单检查:")
                print(f"   原始价格: {order.price:.4f} USDT")
                print(f"   实际价格: {actual_price:.4f} USDT (滑点: -{slippage_rate*100:.3f}%)")
                print(f"   成交比例: {fill_ratio*100:.1f}% (实际数量: {actual_amount:.6f} FIL)")
                print(f"   总收入: {gross_proceeds:.2f} USDT")
                print(f"   手续费: {commission_fee:.2f} USDT")
                print(f"   净收入: {net_proceeds:.2f} USDT")
                print(f"   可用FIL: {available_fil_value:.6f} FIL")
                
                # 添加日志记录
                self.logger.info(f"🔍 卖出订单检查: 原始价格={order.price:.4f}, 实际价格={actual_price:.4f}, 滑点={slippage_rate*100:.3f}%, "
                           f"成交比例={fill_ratio*100:.1f}%, 总收入={gross_proceeds:.2f}, 手续费={commission_fee:.2f}, "
                           f"净收入={net_proceeds:.2f}, 可用FIL={available_fil_value:.6f}")
                
                if actual_amount <= available_fil_value:
                    # 减少FIL数量（使用实际成交数量）
                    self.balance['FIL'] -= actual_amount
                    # 增加USDT净收入
                    self.balance['USDT'] += net_proceeds
                    
                    # 更新订单状态
                    if fill_ratio >= 1.0:
                        order.status = OrderStatus.FILLED
                    else:
                        order.status = OrderStatus.PARTIALLY_FILLED
                    order.filled_amount = actual_amount
                    order.filled_price = actual_price
                    
                    # 记录交易成本
                    if not hasattr(order, 'trading_costs'):
                        order.trading_costs = {}
                    order.trading_costs = {
                        'commission': commission_fee,
                        'slippage': (actual_amount * order.price) - gross_proceeds,
                        'net_proceeds': net_proceeds,
                        'fill_ratio': fill_ratio
                    }
                    
                    if fill_ratio >= 1.0:
                        print(f"✅ 卖出订单完全成交: {actual_amount:.6f} FIL @ {actual_price:.4f} USDT")
                    else:
                        print(f"⚠️ 卖出订单部分成交: {actual_amount:.6f}/{order.amount:.6f} FIL @ {actual_price:.4f} USDT")
                    print(f"   手续费: {commission_fee:.2f} USDT, 滑点损失: {order.trading_costs['slippage']:.2f} USDT")
                    return True
                else:
                    reject_reason = f"余额不足: 需要 {order.amount:.6f} FIL, 可用 {available_fil_value:.6f} FIL"
                    print(f"❌ 卖出订单失败: {reject_reason}")
                    order.status = OrderStatus.REJECTED
                    if not hasattr(order, 'reject_reason'):
                        order.reject_reason = reject_reason
                    return False
            
        except Exception as e:
            reject_reason = f"订单处理异常: {str(e)}"
            print(f"❌ {reject_reason}")
            order.status = OrderStatus.REJECTED
            # 将拒绝原因存储到订单对象中
            if not hasattr(order, 'reject_reason'):
                order.reject_reason = reject_reason
            return False
    
    def get_positions(self) -> Dict[str, Position]:
        """获取持仓"""
        positions = {}
        if self.balance['FIL'] > 0:
            # 从trading_state.json加载持仓信息
            position_data = self._load_position_from_state()
            
            positions['FIL/USDT'] = Position(
                symbol='FIL/USDT',
                amount=self.balance['FIL'],
                entry_price=position_data.get('entry_price', self.current_price),
                current_price=self.current_price,
                unrealized_pnl=0.0,
                timestamp=datetime.now(),
                side='buy',
                entry_time=position_data.get('entry_time', datetime.now()),
                stop_loss=position_data.get('stop_loss'),
                take_profit=position_data.get('take_profit'),
                # 移动止损字段
                trailing_stop=position_data.get('trailing_stop'),
                trailing_stop_activated=position_data.get('trailing_stop_activated', False),
                highest_price=position_data.get('highest_price'),
                lowest_price=position_data.get('lowest_price'),
                trailing_distance=position_data.get('trailing_distance')
            )
        return positions
    
    def _load_position_from_state(self) -> Dict:
        """从trading_state.json加载持仓状态"""
        try:
            state_file = 'live/trading_state.json'
            if os.path.exists(state_file):
                with open(state_file, 'r') as f:
                    state = json.load(f)
                    return state.get('position', {})
            return {}
        except Exception as e:
            print(f"加载持仓状态失败: {e}")
            return {}
    
    def update_positions(self):
        """更新持仓"""
        # 模拟持仓更新 - 从trading_state.json加载持仓状态
        try:
            state_file = 'live/trading_state.json'
            if os.path.exists(state_file):
                with open(state_file, 'r') as f:
                    state = json.load(f)
                    if state.get('position'):
                        # 如果有持仓，更新到positions字典
                        position = state['position']
                        symbol = position['symbol']
                        # 处理entry_time，确保它是datetime对象
                        entry_time = position.get('entry_time')
                        if isinstance(entry_time, str):
                            entry_time = datetime.fromisoformat(entry_time)
                        elif entry_time is None:
                            entry_time = datetime.now()
                        
                        self.positions[symbol] = Position(
                            symbol=position['symbol'],
                            amount=position['amount'],
                            entry_price=position['entry_price'],
                            current_price=position.get('current_price', position['entry_price']),
                            unrealized_pnl=position.get('unrealized_pnl', 0.0),
                            timestamp=datetime.now(),
                            side=position.get('side'),
                            entry_time=entry_time,
                            stop_loss=position.get('stop_loss'),
                            take_profit=position.get('take_profit')
                        )
                    else:
                        # 如果没有持仓，清空positions
                        self.positions = {}
        except Exception as e:
            print(f"更新持仓状态失败: {e}")
            self.positions = {}
    
    def save_position_state(self, position: Position):
        """保存持仓状态到文件"""
        try:
            state_file = 'live/trading_state.json'
            os.makedirs(os.path.dirname(state_file), exist_ok=True)
            
            # 准备状态数据
            state = {
                'position': {
                    'symbol': position.symbol,
                    'amount': position.amount,
                    'entry_price': position.entry_price,
                    'current_price': position.current_price,
                    'side': position.side,
                    'entry_time': position.entry_time.isoformat() if position.entry_time else None,
                    'stop_loss': position.stop_loss,
                    'take_profit': position.take_profit,
                    # 移动止损字段
                    'trailing_stop': position.trailing_stop,
                    'trailing_stop_activated': position.trailing_stop_activated,
                    'highest_price': position.highest_price,
                    'lowest_price': position.lowest_price,
                    'trailing_distance': position.trailing_distance
                },
                'last_updated': datetime.now().isoformat()
            }
            
            with open(state_file, 'w') as f:
                json.dump(state, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            print(f"保存持仓状态失败: {e}")
    
    def get_portfolio_value(self) -> float:
        """获取投资组合价值"""
        # 处理Balance对象或直接数值
        usdt_balance = self.balance['USDT']
        fil_balance = self.balance['FIL']
        
        if hasattr(usdt_balance, 'free'):
            usdt_value = usdt_balance.free
        else:
            usdt_value = float(usdt_balance)
            
        if hasattr(fil_balance, 'free'):
            fil_value = fil_balance.free
        else:
            fil_value = float(fil_balance)
        
        return usdt_value + fil_value * self.current_price


class BinanceExchange:
    """币安交易所接口"""
    
    def __init__(self, config):
        self.config = config
        self.base_url = "https://api.binance.com"
        self.api_key = os.getenv('BINANCE_API_KEY')
        self.secret_key = os.getenv('BINANCE_SECRET_KEY')
        
        if not self.api_key or not self.secret_key:
            raise ValueError("请设置 BINANCE_API_KEY 和 BINANCE_SECRET_KEY 环境变量")
    
    def get_klines(self, symbol: str, timeframe: str, limit: int) -> List:
        """获取K线数据"""
        try:
            url = f"{self.base_url}/api/v3/klines"
            params = {
                'symbol': symbol.replace('/', ''),
                'interval': timeframe,
                'limit': limit
            }
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            return response.json()
            
        except Exception as e:
            raise Exception(f"获取K线数据失败: {str(e)}")
    
    def get_balance(self) -> Dict[str, float]:
        """获取账户余额"""
        try:
            # 这里需要实现币安API的签名认证
            # 为了安全，建议使用专门的交易库如ccxt
            pass
        except Exception as e:
            raise Exception(f"获取余额失败: {str(e)}")
    
    def place_order(self, order: Order) -> bool:
        """下单（实盘交易）"""
        try:
            # 使用ccxt库调用币安API下单
            if not hasattr(self, 'exchange') or self.exchange is None:
                raise Exception("交易所连接未初始化")
            
            # 构建订单参数
            symbol = order.symbol
            side = order.side.value if hasattr(order.side, 'value') else order.side
            amount = order.amount
            price = order.price
            order_type = 'market'  # 默认市价单
            
            # 调用币安API下单
            result = self.exchange.create_order(
                symbol=symbol,
                type=order_type,
                side=side,
                amount=amount,
                price=price
            )
            
            # 更新订单状态
            if result and 'id' in result:
                order.id = result['id']
                order.status = OrderStatus.PENDING
                order.filled_amount = result.get('filled', 0.0)
                order.filled_price = result.get('price', 0.0)
                
                self.logger.info(f"✅ 实盘订单提交成功: {order.id} - {side} {amount:.6f} {symbol} @ {price:.4f}")
                return True
            else:
                order.status = OrderStatus.REJECTED
                order.reject_reason = "API返回无效结果"
                self.logger.error(f"❌ 实盘订单提交失败: API返回无效结果")
                return False
                
        except Exception as e:
            reject_reason = f"实盘下单失败: {str(e)}"
            self.logger.error(f"❌ {reject_reason}")
            order.status = OrderStatus.REJECTED
            if not hasattr(order, 'reject_reason'):
                order.reject_reason = reject_reason
            return False


def main():
    """主函数"""
    print("🚀 启动实盘交易系统")
    print("=" * 50)
    
    # 选择交易模式
    mode_input = input("选择交易模式 (1: 模拟交易, 2: 实盘交易): ").strip()
    mode = TradingMode.PAPER if mode_input == "1" else TradingMode.LIVE
    
    if mode == TradingMode.LIVE:
        print("⚠️ 警告: 实盘交易模式将使用真实资金!")
        confirm = input("确认继续? (yes/no): ").strip().lower()
        if confirm != 'yes':
            print("❌ 已取消")
            return
    
    try:
        # 创建交易系统
        trading_system = LiveTradingSystem(mode=mode)
        
        # 开始交易
        trading_system.start_trading()
        
    except KeyboardInterrupt:
        print("\n⏹️ 用户中断")
    except Exception as e:
        print(f"❌ 系统错误: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()

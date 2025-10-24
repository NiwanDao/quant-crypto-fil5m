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

class LiveTradingSystem:
    """实盘交易系统"""
    
    def __init__(self, config_path: str = 'conf/config.yml', mode: TradingMode = TradingMode.PAPER):
        self.config_path = config_path
        self.mode = mode
        self.config = self._load_config()
        self.model = None
        self.exchange = None
        self.positions = {}
        self.orders = {}
        self.balance = {}
        self.last_signal_time = None
        self.is_running = False
        
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
            level=logging.INFO,
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
        
        # 初始化账户信息
        self._initialize_account()
        
        self.logger.info("✅ 系统初始化完成")
    
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
                self.model = joblib.load('models/lgb_trend.pkl')
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
            
            # 生成信号
            buy_signal = prob_up > buy_threshold and signal_strength > min_signal_strength
            sell_signal = prob_down > sell_threshold and signal_strength > min_signal_strength
            
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
            
            # 基于信号强度的调整
            strength_multiplier = np.clip(signal_strength, 0.3, 1.0)
            
            # 基于可用资金的调整
            usdt_balance = self.balance.get('USDT', 0)
            
            # 处理Balance对象或直接数值
            if hasattr(usdt_balance, 'free'):
                # 如果是Balance对象，使用free字段
                available_cash = usdt_balance.free
            else:
                # 如果是直接数值
                available_cash = float(usdt_balance)
            
            max_cash = min(base_cash, available_cash * 0.1)  # 最多使用10%的资金
            
            # 计算仓位
            position_cash = base_cash * strength_multiplier
            position_cash = min(position_cash, max_cash)
            
            position_size = position_cash / current_price
            
            self.logger.info(f"💰 仓位计算: 基础={base_cash}, 强度倍数={strength_multiplier:.3f}, 可用资金={available_cash:.2f}, 最终仓位={position_size:.6f}")
            
            return position_size
            
        except Exception as e:
            self.logger.error(f"❌ 仓位计算失败: {str(e)}")
            return 0.0
    
    def execute_trade(self, symbol: str, side: OrderSide, amount: float, price: float) -> Optional[Order]:
        """执行交易"""
        try:
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
            self.logger.info(f"🔍 提交订单前状态: 订单ID={order_id}, 状态={order.status}, 余额={self.balance}")
            
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
            
            if success:
                self.orders[order_id] = order
                self.logger.info(f"✅ 订单提交成功: {order_id}")
                return order
            else:
                # 安全地获取状态值
                status_value = order.status.value if hasattr(order.status, 'value') else str(order.status)
                
                # 获取详细的拒绝原因
                reject_reason = self._get_reject_reason(order)
                self.logger.error(f"❌ 订单提交失败: {order_id} - 状态: {status_value}")
                self.logger.error(f"❌ 拒绝原因: {reject_reason}")
                print(f"❌ 订单提交失败: {order_id} - 状态: {status_value}")
                print(f"❌ 拒绝原因: {reject_reason}")
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
                
                if cost > available_usdt:
                    return f"余额不足: 需要 {cost:.2f} USDT, 可用 {available_usdt:.2f} USDT"
                else:
                    # 添加更多诊断信息
                    return f"买入订单被拒绝 - 成本: {cost:.2f} USDT, 可用: {available_usdt:.2f} USDT, 订单状态: {order.status}"
                    
            elif order.side == OrderSide.SELL:
                available_fil = self.balance.get('FIL', 0)
                
                # 处理Balance对象或直接数值
                if hasattr(available_fil, 'free'):
                    available_fil = available_fil.free
                else:
                    available_fil = float(available_fil)
                
                if order.amount > available_fil:
                    return f"余额不足: 需要 {order.amount:.6f} FIL, 可用 {available_fil:.6f} FIL"
                else:
                    # 添加更多诊断信息
                    return f"卖出订单被拒绝 - 数量: {order.amount:.6f} FIL, 可用: {available_fil:.6f} FIL, 订单状态: {order.status}"
            else:
                return "无效的订单方向"
                
        except Exception as e:
            return f"获取拒绝原因时出错: {str(e)}"

    def _debug_balance(self, side: OrderSide, amount: float, price: float):
        """调试余额信息"""
        self.logger.info(f"🔍 余额调试信息:")
        self.logger.info(f"  - 订单方向: {side}")
        self.logger.info(f"  - 订单数量: {amount}")
        self.logger.info(f"  - 订单价格: {price}")
        self.logger.info(f"  - 当前余额: {self.balance}")
        
        if side == OrderSide.BUY:
            cost = amount * price
            available_usdt = self.balance.get('USDT', 0)
            
            self.logger.info(f"  - 买入成本: {cost:.2f} USDT")
            self.logger.info(f"  - USDT余额类型: {type(available_usdt)}")
            self.logger.info(f"  - USDT余额值: {available_usdt}")
            
            if hasattr(available_usdt, 'free'):
                self.logger.info(f"  - USDT可用余额: {available_usdt.free}")
            else:
                self.logger.info(f"  - USDT直接数值: {float(available_usdt)}")
                
        elif side == OrderSide.SELL:
            available_fil = self.balance.get('FIL', 0)
            
            self.logger.info(f"  - 卖出数量: {amount:.6f} FIL")
            self.logger.info(f"  - FIL余额类型: {type(available_fil)}")
            self.logger.info(f"  - FIL余额值: {available_fil}")
            
            if hasattr(available_fil, 'free'):
                self.logger.info(f"  - FIL可用余额: {available_fil.free}")
            else:
                self.logger.info(f"  - FIL直接数值: {float(available_fil)}")

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
                self.positions = positions
            
            # 更新账户余额
            self.balance = self.exchange.get_balance()
            
        except Exception as e:
            self.logger.error(f"❌ 持仓更新失败: {str(e)}")
    
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
                    if order:
                        self.logger.info(f"🟢 买入信号执行: {position_size:.6f} {symbol} @ {current_price:.4f}")
                else:
                    self.logger.warning(f"⚠️ 买入信号但仓位大小为0，跳过交易")
            
            elif sell_signal and self.positions.get(symbol):
                # 卖出信号且有持仓
                position = self.positions[symbol]
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
        
        while self.is_running:
            try:
                self.run_trading_cycle()
                
                # 等待下一个周期 - 根据分钟数(0,15,30,45)动态计算
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
        self.logger.info("⏹️ 交易系统停止中...")


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
    
    def place_order(self, order: Order) -> bool:
        """下单（模拟）"""
        try:
            if order.side == OrderSide.BUY:
                cost = order.amount * order.price
                available_usdt = self.balance['USDT']
                
                print(f"🔍 买入订单检查: 成本={cost:.2f} USDT, 可用余额={available_usdt:.2f} USDT")
                
                if cost <= available_usdt:
                    self.balance['USDT'] -= cost
                    self.balance['FIL'] += order.amount
                    order.status = OrderStatus.FILLED
                    order.filled_amount = order.amount
                    order.filled_price = order.price
                    print(f"✅ 买入订单成功: {order.amount:.6f} FIL @ {order.price:.4f} USDT")
                    return True
                else:
                    reject_reason = f"余额不足: 需要 {cost:.2f} USDT, 可用 {available_usdt:.2f} USDT"
                    print(f"❌ 买入订单失败: {reject_reason}")
                    order.status = OrderStatus.REJECTED
                    # 将拒绝原因存储到订单对象中
                    if not hasattr(order, 'reject_reason'):
                        order.reject_reason = reject_reason
                    return False
            else:  # SELL
                available_fil = self.balance['FIL']
                
                print(f"🔍 卖出订单检查: 数量={order.amount:.6f} FIL, 可用余额={available_fil:.6f} FIL")
                
                if order.amount <= available_fil:
                    proceeds = order.amount * order.price
                    self.balance['FIL'] -= order.amount
                    self.balance['USDT'] += proceeds
                    order.status = OrderStatus.FILLED
                    order.filled_amount = order.amount
                    order.filled_price = order.price
                    print(f"✅ 卖出订单成功: {order.amount:.6f} FIL @ {order.price:.4f} USDT")
                    return True
                else:
                    reject_reason = f"余额不足: 需要 {order.amount:.6f} FIL, 可用 {available_fil:.6f} FIL"
                    print(f"❌ 卖出订单失败: {reject_reason}")
                    order.status = OrderStatus.REJECTED
                    # 将拒绝原因存储到订单对象中
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
            positions['FIL/USDT'] = Position(
                symbol='FIL/USDT',
                amount=self.balance['FIL'],
                entry_price=0.0,  # 简化处理
                current_price=self.current_price,
                unrealized_pnl=0.0,
                timestamp=datetime.now(),
                side='buy',
                entry_time=datetime.now(),
                stop_loss=None,
                take_profit=None
            )
        return positions
    
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
        """下单"""
        try:
            # 这里需要实现币安API的订单提交
            # 为了安全，建议使用专门的交易库如ccxt
            pass
        except Exception as e:
            raise Exception(f"下单失败: {str(e)}")


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

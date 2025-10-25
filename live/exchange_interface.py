"""
交易所接口模块
支持多个主流交易所的统一接口
"""

import ccxt
import pandas as pd
import numpy as np
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

class ExchangeType(Enum):
    """交易所类型"""
    BINANCE = "binance"
    OKX = "okx"
    BYBIT = "bybit"
    GATEIO = "gateio"

@dataclass
class MarketData:
    """市场数据结构"""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

@dataclass
class OrderInfo:
    """订单信息"""
    id: str
    symbol: str
    side: str  # 'buy' or 'sell'
    amount: float
    price: float
    status: str
    filled: float
    remaining: float
    timestamp: datetime

@dataclass
class Balance:
    """余额信息"""
    free: float
    used: float
    total: float

class ExchangeInterface:
    """统一交易所接口"""
    
    def __init__(self, exchange_type: ExchangeType, config: Dict):
        self.exchange_type = exchange_type
        self.config = config
        self.exchange = None
        self.logger = logging.getLogger('ExchangeInterface')
        
        self._initialize_exchange()
    
    def _initialize_exchange(self):
        """初始化交易所连接"""
        try:
            if self.exchange_type == ExchangeType.BINANCE:
                self.exchange = ccxt.binance({
                    'apiKey': self.config.get('api_key'),
                    'secret': self.config.get('secret_key'),
                    'sandbox': self.config.get('sandbox', False),
                    'rateLimit': 1200,
                    'enableRateLimit': True,
                    'options': {
                        'defaultType': 'spot',  # 只使用现货
                        'loadMarketsOnDemand': True,  # 按需加载市场
                    }
                })
            elif self.exchange_type == ExchangeType.OKX:
                self.exchange = ccxt.okx({
                    'apiKey': self.config.get('api_key'),
                    'secret': self.config.get('secret_key'),
                    'password': self.config.get('passphrase'),
                    'sandbox': self.config.get('sandbox', False),
                    'rateLimit': 100,
                    'enableRateLimit': True,
                })
            elif self.exchange_type == ExchangeType.BYBIT:
                self.exchange = ccxt.bybit({
                    'apiKey': self.config.get('api_key'),
                    'secret': self.config.get('secret_key'),
                    'sandbox': self.config.get('sandbox', False),
                    'rateLimit': 120,
                    'enableRateLimit': True,
                })
            elif self.exchange_type == ExchangeType.GATEIO:
                self.exchange = ccxt.gateio({
                    'apiKey': self.config.get('api_key'),
                    'secret': self.config.get('secret_key'),
                    'sandbox': self.config.get('sandbox', False),
                    'rateLimit': 1000,
                    'enableRateLimit': True,
                })
            else:
                raise ValueError(f"不支持的交易所类型: {self.exchange_type}")
            
            # 测试连接 - 只测试现货市场
            try:
                # 获取服务器时间测试连接
                server_time = self.exchange.fetch_time()
                self.logger.info(f"✅ {self.exchange_type.value} 交易所连接成功，服务器时间: {server_time}")
            except Exception as e:
                self.logger.warning(f"⚠️ 服务器时间获取失败，但继续初始化: {str(e)}")
                # 尝试获取特定交易对的市场信息
                try:
                    test_symbol = 'FIL/USDT'
                    ticker = self.exchange.fetch_ticker(test_symbol)
                    self.logger.info(f"✅ {self.exchange_type.value} 交易所连接成功，测试交易对: {test_symbol}")
                except Exception as e2:
                    self.logger.error(f"❌ 交易所连接测试失败: {str(e2)}")
                    raise
            
        except Exception as e:
            self.logger.error(f"❌ 交易所连接失败: {str(e)}")
            raise
    
    def get_market_data(self, symbol: str, timeframe: str = '15m', limit: int = 100) -> pd.DataFrame:
        """获取市场数据"""
        try:
            # 标准化交易对格式
            symbol = self._normalize_symbol(symbol)
            
            # 获取K线数据
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            
            # 转换为DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
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
    
    def get_current_price(self, symbol: str) -> float:
        """获取当前价格"""
        try:
            symbol = self._normalize_symbol(symbol)
            ticker = self.exchange.fetch_ticker(symbol)
            return float(ticker['last'])
        except Exception as e:
            self.logger.error(f"❌ 获取当前价格失败: {str(e)}")
            return 0.0
    
    def get_balance(self) -> Dict[str, Balance]:
        """获取账户余额"""
        try:
            balance = self.exchange.fetch_balance()
            
            # 转换为标准格式
            result = {}
            for currency, info in balance.items():
                if isinstance(info, dict) and 'free' in info:
                    result[currency] = Balance(
                        free=float(info['free']),
                        used=float(info['used']),
                        total=float(info['total'])
                    )
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ 获取余额失败: {str(e)}")
            return {}
    
    def get_positions(self) -> Dict[str, Dict]:
        """获取持仓信息"""
        try:
            if hasattr(self.exchange, 'fetch_positions'):
                positions = self.exchange.fetch_positions()
                return {pos['symbol']: pos for pos in positions if pos['contracts'] > 0}
            else:
                # 现货交易，从余额计算持仓
                balance = self.get_balance()
                positions = {}
                for currency, bal in balance.items():
                    if bal.total > 0:
                        positions[f"{currency}/USDT"] = {
                            'amount': bal.total,
                            'side': 'long',
                            'unrealizedPnl': 0.0
                        }
                return positions
                
        except Exception as e:
            self.logger.error(f"❌ 获取持仓失败: {str(e)}")
            return {}
    
    def place_market_order(self, symbol: str, side: str, amount: float) -> Optional[OrderInfo]:
        """下市价单"""
        try:
            symbol = self._normalize_symbol(symbol)
            
            order = self.exchange.create_market_order(symbol, side, amount)
            
            return OrderInfo(
                id=order['id'],
                symbol=symbol,
                side=side,
                amount=amount,
                price=0.0,  # 市价单
                status=order['status'],
                filled=float(order.get('filled', 0)),
                remaining=float(order.get('remaining', amount)),
                timestamp=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"❌ 下市价单失败: {str(e)}")
            return None
    
    def place_limit_order(self, symbol: str, side: str, amount: float, price: float) -> Optional[OrderInfo]:
        """下限价单"""
        try:
            symbol = self._normalize_symbol(symbol)
            
            order = self.exchange.create_limit_order(symbol, side, amount, price)
            
            return OrderInfo(
                id=order['id'],
                symbol=symbol,
                side=side,
                amount=amount,
                price=price,
                status=order['status'],
                filled=float(order.get('filled', 0)),
                remaining=float(order.get('remaining', amount)),
                timestamp=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"❌ 下限价单失败: {str(e)}")
            return None
    
    def cancel_order(self, order_id: str, symbol: str) -> bool:
        """取消订单"""
        try:
            symbol = self._normalize_symbol(symbol)
            result = self.exchange.cancel_order(order_id, symbol)
            return result.get('status') == 'canceled'
        except Exception as e:
            self.logger.error(f"❌ 取消订单失败: {str(e)}")
            return False
    
    def get_order_status(self, order_id: str, symbol: str) -> Optional[OrderInfo]:
        """获取订单状态"""
        try:
            symbol = self._normalize_symbol(symbol)
            order = self.exchange.fetch_order(order_id, symbol)
            
            return OrderInfo(
                id=order['id'],
                symbol=symbol,
                side=order['side'],
                amount=float(order['amount']),
                price=float(order['price']),
                status=order['status'],
                filled=float(order.get('filled', 0)),
                remaining=float(order.get('remaining', 0)),
                timestamp=datetime.fromtimestamp(order['timestamp'] / 1000)
            )
            
        except Exception as e:
            self.logger.error(f"❌ 获取订单状态失败: {str(e)}")
            return None
    
    def get_trading_fees(self, symbol: str) -> Dict[str, float]:
        """获取交易手续费"""
        try:
            symbol = self._normalize_symbol(symbol)
            fees = self.exchange.fetch_trading_fees(symbol)
            return {
                'maker': float(fees.get('maker', 0)),
                'taker': float(fees.get('taker', 0))
            }
        except Exception as e:
            self.logger.error(f"❌ 获取交易手续费失败: {str(e)}")
            return {'maker': 0.001, 'taker': 0.001}  # 默认手续费
    
    def _normalize_symbol(self, symbol: str) -> str:
        """标准化交易对格式"""
        # 移除斜杠，转换为交易所要求的格式
        if '/' in symbol:
            base, quote = symbol.split('/')
            return f"{base}{quote}"
        return symbol
    
    def test_connection(self) -> bool:
        """测试连接"""
        try:
            self.exchange.fetch_balance()
            return True
        except Exception as e:
            self.logger.error(f"❌ 连接测试失败: {str(e)}")
            return False


class PaperTradingExchange:
    """模拟交易交易所"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.positions = {}
        # 从配置中获取初始资金，如果exchange配置中没有，则使用默认值
        initial_cash = config.get('initial_cash', 10000)
        self.balance = {
            'USDT': initial_cash,
            'FIL': 0.0
        }
        self.orders = {}
        self.current_price = 0.0
        self.logger = logging.getLogger('PaperTrading')
    
    def get_market_data(self, symbol: str, timeframe: str = '15m', limit: int = 100) -> pd.DataFrame:
        """获取模拟市场数据 - 使用真实市场数据"""
        # 使用ccxt获取真实的市场数据
        import ccxt
        
        try:
            # 初始化交易所（使用binance获取真实数据）
            exchange = ccxt.binance({
                'enableRateLimit': True,
            })
            
            # 获取真实的K线数据
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            
            # 转换为DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # 数据类型转换
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col])
            
            # 时间索引
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # 计算收益率
            df['returns'] = df['close'].pct_change()
            
            # 更新当前价格
            self.current_price = float(df['close'].iloc[-1])
            
            return df
            
        except Exception as e:
            print(f"❌ 获取真实市场数据失败: {str(e)}")
            # 如果获取失败，返回空DataFrame
            return pd.DataFrame()
    
    def get_current_price(self, symbol: str) -> float:
        """获取当前价格"""
        return self.current_price
    
    def get_klines(self, symbol: str, timeframe: str = '15m', limit: int = 100):
        """获取K线数据（兼容性方法） - 返回OHLCV列表"""
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
    
    def update_positions(self):
        """更新持仓信息（模拟交易不需要实际更新）"""
        pass
    
    def get_portfolio_value(self) -> float:
        """获取投资组合价值"""
        return self.balance['USDT'] + self.balance['FIL'] * self.current_price
    
    def get_balance(self) -> Dict[str, Balance]:
        """获取账户余额"""
        from dataclasses import dataclass
        
        @dataclass
        class Balance:
            free: float
            used: float
            total: float
        
        return {
            'USDT': Balance(
                free=self.balance['USDT'],
                used=0.0,
                total=self.balance['USDT']
            ),
            'FIL': Balance(
                free=self.balance['FIL'],
                used=0.0,
                total=self.balance['FIL']
            )
        }
    
    def get_positions(self) -> Dict[str, Dict]:
        """获取持仓信息"""
        positions = {}
        if self.balance['FIL'] > 0:
            positions['FIL/USDT'] = {
                'amount': self.balance['FIL'],
                'side': 'long',
                'unrealizedPnl': 0.0
            }
        return positions
    
    def place_market_order(self, symbol: str, side: str, amount: float) -> Optional[OrderInfo]:
        """下市价单（模拟）"""
        try:
            order_id = f"{side}_{int(time.time() * 1000)}"
            current_price = self.get_current_price(symbol)
            
            if side == 'buy':
                cost = amount * current_price
                if cost <= self.balance['USDT']:
                    self.balance['USDT'] -= cost
                    self.balance['FIL'] += amount
                    
                    order = OrderInfo(
                        id=order_id,
                        symbol=symbol,
                        side=side,
                        amount=amount,
                        price=current_price,
                        status='closed',
                        filled=amount,
                        remaining=0.0,
                        timestamp=datetime.now()
                    )
                    
                    self.logger.info(f"✅ 模拟买入: {amount:.6f} {symbol} @ {current_price:.4f}")
                    return order
            else:  # sell
                if amount <= self.balance['FIL']:
                    proceeds = amount * current_price
                    self.balance['FIL'] -= amount
                    self.balance['USDT'] += proceeds
                    
                    order = OrderInfo(
                        id=order_id,
                        symbol=symbol,
                        side=side,
                        amount=amount,
                        price=current_price,
                        status='closed',
                        filled=amount,
                        remaining=0.0,
                        timestamp=datetime.now()
                    )
                    
                    self.logger.info(f"✅ 模拟卖出: {amount:.6f} {symbol} @ {current_price:.4f}")
                    return order
            
            self.logger.error(f"❌ 模拟交易失败: 余额不足")
            return None
            
        except Exception as e:
            self.logger.error(f"❌ 模拟交易失败: {str(e)}")
            return None
    
    def place_limit_order(self, symbol: str, side: str, amount: float, price: float) -> Optional[OrderInfo]:
        """下限价单（模拟）"""
        # 模拟交易中，限价单立即成交
        return self.place_market_order(symbol, side, amount)
    
    def cancel_order(self, order_id: str, symbol: str) -> bool:
        """取消订单（模拟）"""
        return True
    
    def get_order_status(self, order_id: str, symbol: str) -> Optional[OrderInfo]:
        """获取订单状态（模拟）"""
        return None
    
    def get_trading_fees(self, symbol: str) -> Dict[str, float]:
        """获取交易手续费（模拟）"""
        return {'maker': 0.001, 'taker': 0.001}
    
    def test_connection(self) -> bool:
        """测试连接（模拟）"""
        return True
    
    def place_order(self, order) -> bool:
        """下单（模拟）"""
        try:
            # 导入枚举类型（如果需要的话）
            from live.live_trading import OrderStatus
            
            # 获取订单方向（处理枚举类型或字符串类型）
            side = order.side
            if hasattr(side, 'value'):
                side_value = side.value
            else:
                side_value = side
            
            if side_value == 'buy':
                cost = order.amount * order.price
                if cost <= self.balance['USDT']:
                    self.balance['USDT'] -= cost
                    self.balance['FIL'] += order.amount
                    # 处理枚举类型的status
                    if hasattr(order.status, 'value'):
                        order.status = OrderStatus.FILLED
                    else:
                        order.status = 'filled'
                    order.filled_amount = order.amount
                    order.filled_price = order.price
                    self.logger.info(f"✅ 买入订单成功: {order.amount:.6f} FIL @ {order.price:.4f}")
                    return True
                else:
                    self.logger.warning(f"❌ 余额不足: 需要 {cost:.2f} USDT, 可用 {self.balance['USDT']:.2f} USDT")
            else:  # SELL
                if order.amount <= self.balance['FIL']:
                    proceeds = order.amount * order.price
                    self.balance['FIL'] -= order.amount
                    self.balance['USDT'] += proceeds
                    # 处理枚举类型的status
                    if hasattr(order.status, 'value'):
                        order.status = OrderStatus.FILLED
                    else:
                        order.status = 'filled'
                    order.filled_amount = order.amount
                    order.filled_price = order.price
                    self.logger.info(f"✅ 卖出订单成功: {order.amount:.6f} FIL @ {order.price:.4f}")
                    return True
                else:
                    self.logger.warning(f"❌ FIL余额不足: 需要 {order.amount:.6f} FIL, 可用 {self.balance['FIL']:.6f} FIL")
            
            # 处理枚举类型的status
            if hasattr(order.status, 'value'):
                order.status = OrderStatus.REJECTED
            else:
                order.status = 'rejected'
            return False
            
        except Exception as e:
            self.logger.error(f"place_order 异常: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            if hasattr(order.status, 'value'):
                from live.live_trading import OrderStatus
                order.status = OrderStatus.REJECTED
            else:
                order.status = 'rejected'
            return False


def create_exchange(exchange_type: str, config: Dict, paper_trading: bool = False):
    """创建交易所实例"""
    if paper_trading:
        # 对于模拟交易，需要传入完整的配置，包括initial_cash
        return PaperTradingExchange(config)
    
    exchange_enum = ExchangeType(exchange_type.lower())
    return ExchangeInterface(exchange_enum, config)


# 使用示例
if __name__ == '__main__':
    # 配置示例
    config = {
        'api_key': 'your_api_key',
        'secret_key': 'your_secret_key',
        'sandbox': True,  # 使用测试网
        'initial_cash': 10000
    }
    
    # 创建交易所实例
    exchange = create_exchange('binance', config, paper_trading=True)
    
    # 测试功能
    print("测试连接:", exchange.test_connection())
    print("获取余额:", exchange.get_balance())
    print("获取持仓:", exchange.get_positions())

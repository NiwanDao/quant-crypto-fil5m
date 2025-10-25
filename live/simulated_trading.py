"""
模拟实盘交易模块
使用币安测试网进行安全的模拟交易
"""

import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
import json
import os
from dataclasses import dataclass, asdict

@dataclass
class Position:
    """持仓信息"""
    symbol: str
    side: str  # 'long' or 'short'
    amount: float
    entry_price: float
    entry_time: str
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    unrealized_pnl: float = 0.0

@dataclass
class Order:
    """订单信息"""
    order_id: str
    symbol: str
    side: str  # 'buy' or 'sell'
    amount: float
    price: float
    status: str  # 'open', 'filled', 'cancelled'
    timestamp: str
    order_type: str = 'market'  # 'market' or 'limit'

class SimulatedTrader:
    """模拟交易器"""
    
    def __init__(self, exchange: ccxt.Exchange, config: dict):
        self.exchange = exchange
        self.config = config
        self.symbol = config['symbol']
        self.initial_balance = config['backtest']['initial_cash']
        self.current_balance = self.initial_balance
        self.position: Optional[Position] = None
        self.orders: List[Order] = []
        self.trade_history: List[Dict] = []
        self.is_trading_enabled = False
        
        # 加载交易状态
        self.load_trading_state()
    
    def load_trading_state(self):
        """加载交易状态"""
        state_file = 'live/trading_state.json'
        if os.path.exists(state_file):
            try:
                with open(state_file, 'r') as f:
                    state = json.load(f)
                    self.current_balance = state.get('balance', self.initial_balance)
                    self.is_trading_enabled = state.get('enabled', False)
                    if state.get('position'):
                        self.position = Position(**state['position'])
            except Exception as e:
                print(f"加载交易状态失败: {e}")
    
    def save_trading_state(self):
        """保存交易状态"""
        state_file = 'live/trading_state.json'
        state = {
            'balance': self.current_balance,
            'enabled': self.is_trading_enabled,
            'position': asdict(self.position) if self.position else None,
            'last_update': datetime.now(timezone.utc).isoformat()
        }
        try:
            with open(state_file, 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            print(f"保存交易状态失败: {e}")
    
    def get_account_balance(self) -> Dict:
        """获取账户余额"""
        try:
            balance = self.exchange.fetch_balance()
            return {
                'total': balance['total'],
                'free': balance['free'],
                'used': balance['used']
            }
        except Exception as e:
            print(f"获取余额失败: {e}")
            return {'total': {}, 'free': {}, 'used': {}}
    
    def get_current_price(self) -> float:
        """获取当前价格"""
        try:
            ticker = self.exchange.fetch_ticker(self.symbol)
            return float(ticker['last'])
        except Exception as e:
            print(f"获取价格失败: {e}")
            return 0.0
    
    def calculate_position_size(self, price: float, risk_percent: float = 0.01) -> float:
        """计算仓位大小"""
        risk_amount = self.current_balance * risk_percent
        position_size = risk_amount / price
        return round(position_size, 6)  # 保留6位小数
    
    def place_market_order(self, side: str, amount: float, stop_loss: Optional[float] = None, 
                          take_profit: Optional[float] = None) -> Dict:
        """下市价单"""
        try:
            current_price = self.get_current_price()
            if current_price == 0:
                return {'success': False, 'message': '无法获取当前价格'}
            
            # 在测试网模式下，我们模拟订单执行
            order_id = f"sim_{int(datetime.now().timestamp() * 1000)}"
            
            # 创建订单记录
            order = Order(
                order_id=order_id,
                symbol=self.symbol,
                side=side,
                amount=amount,
                price=current_price,
                status='filled',
                timestamp=datetime.now(timezone.utc).isoformat(),
                order_type='market'
            )
            
            # 更新持仓
            if side == 'buy':
                if self.position and self.position.side == 'long':
                    # 加仓
                    total_amount = self.position.amount + amount
                    avg_price = (self.position.amount * self.position.entry_price + amount * current_price) / total_amount
                    self.position.amount = total_amount
                    self.position.entry_price = avg_price
                else:
                    # 开多仓
                    self.position = Position(
                        symbol=self.symbol,
                        side='long',
                        amount=amount,
                        entry_price=current_price,
                        entry_time=datetime.now(timezone.utc).isoformat(),
                        stop_loss=stop_loss,
                        take_profit=take_profit
                    )
                
                # 更新余额
                cost = amount * current_price
                self.current_balance -= cost
                
            elif side == 'sell':
                if self.position and self.position.side == 'long':
                    # 平仓
                    pnl = (current_price - self.position.entry_price) * amount
                    self.current_balance += amount * current_price + pnl
                    
                    # 记录交易历史
                    self.trade_history.append({
                        'symbol': self.symbol,
                        'side': 'sell',
                        'amount': amount,
                        'entry_price': self.position.entry_price,
                        'exit_price': current_price,
                        'pnl': pnl,
                        'timestamp': datetime.now(timezone.utc).isoformat()
                    })
                    
                    # 更新持仓
                    remaining_amount = self.position.amount - amount
                    if remaining_amount <= 0.001:  # 基本平完
                        self.position = None
                    else:
                        self.position.amount = remaining_amount
                else:
                    return {'success': False, 'message': '没有持仓可卖'}
            
            self.orders.append(order)
            self.save_trading_state()
            
            return {
                'success': True,
                'order_id': order_id,
                'price': current_price,
                'amount': amount,
                'message': f'{side}订单执行成功'
            }
            
        except Exception as e:
            return {'success': False, 'message': f'下单失败: {str(e)}'}
    
    def check_stop_loss_take_profit(self) -> List[Dict]:
        """检查止损止盈"""
        triggered_orders = []
        
        if not self.position:
            return triggered_orders
        
        current_price = self.get_current_price()
        if current_price == 0:
            return triggered_orders
        
        # 检查止损
        if self.position.stop_loss:
            if self.position.side == 'long' and current_price <= self.position.stop_loss:
                result = self.place_market_order('sell', self.position.amount)
                if result['success']:
                    triggered_orders.append({
                        'type': 'stop_loss',
                        'price': current_price,
                        'message': f'止损触发，平仓价格: {current_price}'
                    })
        
        # 检查止盈
        if self.position.take_profit:
            if self.position.side == 'long' and current_price >= self.position.take_profit:
                result = self.place_market_order('sell', self.position.amount)
                if result['success']:
                    triggered_orders.append({
                        'type': 'take_profit',
                        'price': current_price,
                        'message': f'止盈触发，平仓价格: {current_price}'
                    })
        
        return triggered_orders
    
    def get_trading_status(self) -> Dict:
        """获取交易状态"""
        current_price = self.get_current_price()
        
        # 计算未实现盈亏
        unrealized_pnl = 0.0
        if self.position:
            if self.position.side == 'long':
                unrealized_pnl = (current_price - self.position.entry_price) * self.position.amount
        
        # 计算总盈亏
        total_pnl = self.current_balance - self.initial_balance
        for trade in self.trade_history:
            total_pnl += trade['pnl']
        
        return {
            'is_trading_enabled': self.is_trading_enabled,
            'current_balance': self.current_balance,
            'initial_balance': self.initial_balance,
            'total_pnl': total_pnl,
            'unrealized_pnl': unrealized_pnl,
            'position': asdict(self.position) if self.position else None,
            'active_orders': [asdict(order) for order in self.orders if order.status == 'open'],
            'trade_count': len(self.trade_history),
            'current_price': current_price
        }
    
    def enable_trading(self):
        """启用交易"""
        self.is_trading_enabled = True
        self.save_trading_state()
        return {'success': True, 'message': '交易已启用'}
    
    def disable_trading(self):
        """禁用交易"""
        self.is_trading_enabled = False
        self.save_trading_state()
        return {'success': True, 'message': '交易已禁用'}
    
    def reset_account(self):
        """重置账户"""
        self.current_balance = self.initial_balance
        self.position = None
        self.orders = []
        self.trade_history = []
        self.is_trading_enabled = False
        self.save_trading_state()
        return {'success': True, 'message': '账户已重置'}

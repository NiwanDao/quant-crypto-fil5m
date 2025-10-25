"""
WebSocket管理器
负责管理实时数据流和自动重连
"""

import time
import threading
import logging
from datetime import datetime
from typing import Dict, Callable, Optional
from dataclasses import dataclass
import pandas as pd

@dataclass
class WebSocketConfig:
    """WebSocket配置"""
    auto_reconnect: bool = True
    reconnect_interval: int = 5
    max_reconnect_attempts: int = 10
    heartbeat_interval: int = 30
    connection_timeout: int = 10

class WebSocketManager:
    """WebSocket管理器"""
    
    def __init__(self, data_fetcher, config: Dict):
        self.data_fetcher = data_fetcher
        self.config = config
        self.logger = logging.getLogger('WebSocketManager')
        
        # WebSocket配置
        ws_config = config.get('data_fetching', {}).get('websocket_config', {})
        self.ws_config = WebSocketConfig(**ws_config)
        
        # 连接状态
        self.connections = {}
        self.reconnect_attempts = {}
        self.is_running = False
        
        # 数据缓存
        self.latest_data = {}
        self.data_callbacks = []
        
        # 监控线程
        self.monitor_thread = None
        
    def start(self, symbol: str, callback: Optional[Callable] = None):
        """启动WebSocket连接"""
        try:
            if symbol in self.connections:
                self.logger.warning(f"⚠️ WebSocket连接已存在: {symbol}")
                return
            
            # 创建数据处理回调
            def data_callback(market_data):
                self._handle_market_data(symbol, market_data)
                if callback:
                    callback(market_data)
            
            # 启动WebSocket连接
            self.data_fetcher.start_websocket_stream(symbol, data_callback)
            
            # 记录连接
            self.connections[symbol] = {
                'start_time': datetime.now(),
                'callback': callback,
                'reconnect_attempts': 0
            }
            
            # 重置重连计数
            self.reconnect_attempts[symbol] = 0
            
            # 启动监控线程
            if not self.is_running:
                self._start_monitor()
            
            self.logger.info(f"✅ WebSocket管理器已启动: {symbol}")
            
        except Exception as e:
            self.logger.error(f"❌ 启动WebSocket管理器失败: {str(e)}")
    
    def stop(self, symbol: str):
        """停止WebSocket连接"""
        try:
            if symbol in self.connections:
                # 停止WebSocket连接
                self.data_fetcher.stop_websocket_stream(symbol)
                
                # 删除连接记录
                del self.connections[symbol]
                
                # 删除重连计数
                if symbol in self.reconnect_attempts:
                    del self.reconnect_attempts[symbol]
                
                self.logger.info(f"✅ WebSocket管理器已停止: {symbol}")
            else:
                self.logger.warning(f"⚠️ 未找到WebSocket连接: {symbol}")
                
        except Exception as e:
            self.logger.error(f"❌ 停止WebSocket管理器失败: {str(e)}")
    
    def stop_all(self):
        """停止所有WebSocket连接"""
        try:
            symbols = list(self.connections.keys())
            for symbol in symbols:
                self.stop(symbol)
            
            # 停止监控线程
            self.is_running = False
            if self.monitor_thread and self.monitor_thread.is_alive():
                self.monitor_thread.join(timeout=5)
            
            self.logger.info("✅ 所有WebSocket连接已停止")
            
        except Exception as e:
            self.logger.error(f"❌ 停止所有WebSocket连接失败: {str(e)}")
    
    def _handle_market_data(self, symbol: str, market_data: Dict):
        """处理市场数据"""
        try:
            # 更新最新数据
            self.latest_data[symbol] = market_data
            
            # 调用注册的回调函数
            for callback in self.data_callbacks:
                try:
                    callback(symbol, market_data)
                except Exception as e:
                    self.logger.error(f"❌ 数据回调函数执行失败: {str(e)}")
            
            # 记录数据接收
            self.logger.debug(f"📊 接收到数据: {symbol} - {market_data['close']}")
            
        except Exception as e:
            self.logger.error(f"❌ 处理市场数据失败: {str(e)}")
    
    def _start_monitor(self):
        """启动监控线程"""
        try:
            self.is_running = True
            self.monitor_thread = threading.Thread(target=self._monitor_connections, daemon=True)
            self.monitor_thread.start()
            
            self.logger.info("✅ WebSocket监控线程已启动")
            
        except Exception as e:
            self.logger.error(f"❌ 启动监控线程失败: {str(e)}")
    
    def _monitor_connections(self):
        """监控WebSocket连接状态"""
        while self.is_running:
            try:
                # 检查连接状态
                ws_status = self.data_fetcher.get_websocket_status()
                
                for symbol in list(self.connections.keys()):
                    if symbol in ws_status:
                        status = ws_status[symbol]
                        
                        # 检查连接是否断开
                        if not status['connected'] or not status['thread_alive']:
                            self.logger.warning(f"⚠️ WebSocket连接断开: {symbol}")
                            
                            # 自动重连
                            if self.ws_config.auto_reconnect:
                                self._attempt_reconnect(symbol)
                            else:
                                self.logger.error(f"❌ WebSocket连接断开且未启用自动重连: {symbol}")
                                self.stop(symbol)
                    else:
                        # 连接不存在，尝试重连
                        if self.ws_config.auto_reconnect:
                            self._attempt_reconnect(symbol)
                
                # 等待下次检查
                time.sleep(self.ws_config.heartbeat_interval)
                
            except Exception as e:
                self.logger.error(f"❌ 监控连接状态失败: {str(e)}")
                time.sleep(5)
    
    def _attempt_reconnect(self, symbol: str):
        """尝试重连WebSocket"""
        try:
            if symbol not in self.reconnect_attempts:
                self.reconnect_attempts[symbol] = 0
            
            self.reconnect_attempts[symbol] += 1
            
            # 检查重连次数
            if self.reconnect_attempts[symbol] > self.ws_config.max_reconnect_attempts:
                self.logger.error(f"❌ 达到最大重连次数: {symbol}")
                self.stop(symbol)
                return
            
            self.logger.info(f"🔄 尝试重连WebSocket: {symbol} (第{self.reconnect_attempts[symbol]}次)")
            
            # 等待重连间隔
            time.sleep(self.ws_config.reconnect_interval)
            
            # 重新启动连接
            if symbol in self.connections:
                callback = self.connections[symbol]['callback']
                self.start(symbol, callback)
            
        except Exception as e:
            self.logger.error(f"❌ 重连WebSocket失败: {str(e)}")
    
    def add_data_callback(self, callback: Callable):
        """添加数据回调函数"""
        self.data_callbacks.append(callback)
        self.logger.info(f"✅ 已添加数据回调函数: {callback.__name__}")
    
    def remove_data_callback(self, callback: Callable):
        """移除数据回调函数"""
        if callback in self.data_callbacks:
            self.data_callbacks.remove(callback)
            self.logger.info(f"✅ 已移除数据回调函数: {callback.__name__}")
    
    def get_latest_data(self, symbol: str) -> Optional[Dict]:
        """获取最新数据"""
        return self.latest_data.get(symbol)
    
    def get_connection_status(self) -> Dict:
        """获取连接状态"""
        status = {}
        
        for symbol, connection in self.connections.items():
            ws_status = self.data_fetcher.get_websocket_status().get(symbol, {})
            
            status[symbol] = {
                'connected': ws_status.get('connected', False),
                'thread_alive': ws_status.get('thread_alive', False),
                'start_time': connection['start_time'].isoformat(),
                'reconnect_attempts': self.reconnect_attempts.get(symbol, 0),
                'latest_data': self.latest_data.get(symbol) is not None
            }
        
        return status
    
    def get_statistics(self) -> Dict:
        """获取统计信息"""
        stats = {
            'total_connections': len(self.connections),
            'active_connections': 0,
            'total_reconnect_attempts': sum(self.reconnect_attempts.values()),
            'data_callbacks': len(self.data_callbacks),
            'latest_data_count': len(self.latest_data)
        }
        
        # 计算活跃连接数
        ws_status = self.data_fetcher.get_websocket_status()
        for symbol in self.connections:
            if symbol in ws_status and ws_status[symbol]['connected']:
                stats['active_connections'] += 1
        
        return stats

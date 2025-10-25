#!/usr/bin/env python3
"""
WebSocket功能测试脚本
用于验证WebSocket连接和数据接收功能
"""

import sys
import os
import time
import logging
from datetime import datetime
from typing import Dict

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_fetcher import DataFetcher
from websocket_manager import WebSocketManager
from exchange_interface import ExchangeInterface

def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )

def test_websocket_data_callback(symbol: str, market_data: Dict):
    """测试WebSocket数据回调函数"""
    print(f"📊 接收到数据: {symbol}")
    print(f"   时间: {market_data['timestamp']}")
    print(f"   价格: {market_data['close']}")
    print(f"   成交量: {market_data['volume']}")
    print(f"   K线完成: {market_data['is_closed']}")
    print("-" * 50)

def test_binance_websocket():
    """测试币安WebSocket连接"""
    print("🔍 测试币安WebSocket连接...")
    
    # 创建配置
    config = {
        'exchange': {
            'id': 'binance',
            'enableRateLimit': True
        },
        'data_fetching': {
            'enable_websocket': True,
            'websocket_config': {
                'auto_reconnect': True,
                'reconnect_interval': 5,
                'max_reconnect_attempts': 5,
                'heartbeat_interval': 30,
                'connection_timeout': 10
            }
        }
    }
    
    # 创建交易所接口
    exchange = ExchangeInterface(config)
    
    # 创建数据获取器
    data_fetcher = DataFetcher(exchange, config)
    
    # 创建WebSocket管理器
    ws_manager = WebSocketManager(data_fetcher, config)
    
    # 添加数据回调
    ws_manager.add_data_callback(test_websocket_data_callback)
    
    try:
        # 启动WebSocket连接
        symbol = "FILUSDT"
        print(f"🚀 启动WebSocket连接: {symbol}")
        ws_manager.start(symbol)
        
        # 运行测试
        print("⏳ 运行测试30秒...")
        time.sleep(30)
        
        # 获取状态
        status = ws_manager.get_connection_status()
        statistics = ws_manager.get_statistics()
        
        print("\n📈 WebSocket状态:")
        for symbol, info in status.items():
            print(f"  {symbol}:")
            print(f"    连接状态: {info['connected']}")
            print(f"    线程状态: {info['thread_alive']}")
            print(f"    重连次数: {info['reconnect_attempts']}")
            print(f"    最新数据: {info['latest_data']}")
        
        print(f"\n📊 统计信息:")
        print(f"  总连接数: {statistics['total_connections']}")
        print(f"  活跃连接: {statistics['active_connections']}")
        print(f"  总重连次数: {statistics['total_reconnect_attempts']}")
        print(f"  数据回调数: {statistics['data_callbacks']}")
        
        # 获取最新数据
        latest_data = ws_manager.get_latest_data(symbol)
        if latest_data:
            print(f"\n📊 最新数据:")
            print(f"  价格: {latest_data['close']}")
            print(f"  时间: {latest_data['timestamp']}")
        
    except KeyboardInterrupt:
        print("\n⏹️ 用户中断测试")
    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")
    finally:
        # 停止WebSocket连接
        print("🛑 停止WebSocket连接...")
        ws_manager.stop_all()
        print("✅ 测试完成")

def test_websocket_reconnect():
    """测试WebSocket重连功能"""
    print("🔍 测试WebSocket重连功能...")
    
    config = {
        'exchange': {
            'id': 'binance',
            'enableRateLimit': True
        },
        'data_fetching': {
            'enable_websocket': True,
            'websocket_config': {
                'auto_reconnect': True,
                'reconnect_interval': 3,
                'max_reconnect_attempts': 3,
                'heartbeat_interval': 10,
                'connection_timeout': 5
            }
        }
    }
    
    exchange = ExchangeInterface(config)
    data_fetcher = DataFetcher(exchange, config)
    ws_manager = WebSocketManager(data_fetcher, config)
    
    def reconnect_callback(symbol: str, market_data: Dict):
        print(f"🔄 重连测试 - 接收到数据: {symbol} - {market_data['close']}")
    
    ws_manager.add_data_callback(reconnect_callback)
    
    try:
        symbol = "FILUSDT"
        ws_manager.start(symbol)
        
        print("⏳ 运行重连测试60秒...")
        time.sleep(60)
        
        # 检查重连统计
        status = ws_manager.get_connection_status()
        statistics = ws_manager.get_statistics()
        
        print(f"\n📈 重连测试结果:")
        print(f"  总重连次数: {statistics['total_reconnect_attempts']}")
        print(f"  活跃连接: {statistics['active_connections']}")
        
    except KeyboardInterrupt:
        print("\n⏹️ 用户中断测试")
    except Exception as e:
        print(f"❌ 重连测试失败: {str(e)}")
    finally:
        ws_manager.stop_all()
        print("✅ 重连测试完成")

def main():
    """主函数"""
    setup_logging()
    
    print("🚀 WebSocket功能测试开始")
    print("=" * 50)
    
    try:
        # 测试基本WebSocket功能
        test_binance_websocket()
        
        print("\n" + "=" * 50)
        
        # 测试重连功能
        test_websocket_reconnect()
        
    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")
    
    print("\n🏁 所有测试完成")

if __name__ == "__main__":
    main()

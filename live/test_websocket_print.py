#!/usr/bin/env python3
"""
测试WebSocket实时数据打印功能
验证 _start_binance_websocket 方法中的数据打印
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
from exchange_interface import ExchangeInterface, ExchangeType

def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )

def simple_data_callback(symbol: str, market_data: Dict):
    """简单的数据回调函数"""
    print(f"🔄 回调函数接收到数据: {symbol} - 价格: {market_data['close']}")

def test_websocket_data_printing():
    """测试WebSocket数据打印功能"""
    print("🔍 测试WebSocket实时数据打印功能...")
    print("=" * 60)
    print("注意：现在 _start_binance_websocket 方法会直接打印接收到的实时数据")
    print("=" * 60)
    
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
    exchange = ExchangeInterface(ExchangeType.BINANCE, config)
    
    # 创建数据获取器
    data_fetcher = DataFetcher(exchange, config)
    
    # 创建WebSocket管理器
    ws_manager = WebSocketManager(data_fetcher, config)
    
    # 添加数据回调
    ws_manager.add_data_callback(simple_data_callback)
    
    try:
        # 启动WebSocket连接
        symbol = "FILUSDT"
        print(f"🚀 启动WebSocket连接: {symbol}")
        print("📊 现在应该能看到实时数据打印了...")
        print("-" * 60)
        
        ws_manager.start(symbol)
        
        # 运行测试
        print("⏳ 运行测试60秒，观察实时数据打印...")
        time.sleep(60)
        
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

def main():
    """主函数"""
    setup_logging()
    
    print("🚀 WebSocket实时数据打印测试开始")
    print("=" * 60)
    print("这个测试会验证 _start_binance_websocket 方法中的数据打印功能")
    print("你应该能看到类似以下的输出：")
    print("📊 接收到实时数据: FILUSDT")
    print("   时间: 2024-01-01 12:00:00")
    print("   开盘: 5.123")
    print("   最高: 5.145")
    print("   最低: 5.120")
    print("   收盘: 5.135")
    print("   成交量: 12345.67")
    print("   K线完成: True")
    print("-" * 50)
    print("=" * 60)
    
    try:
        test_websocket_data_printing()
    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")
    
    print("\n🏁 测试完成")

if __name__ == "__main__":
    main()

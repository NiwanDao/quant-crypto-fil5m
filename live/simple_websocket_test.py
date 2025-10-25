#!/usr/bin/env python3
"""
简单的WebSocket数据打印测试
直接测试data_fetcher中的WebSocket功能
"""

import sys
import os
import time
import logging
from datetime import datetime

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_fetcher import DataFetcher
from exchange_interface import ExchangeInterface, ExchangeType

def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.DEBUG,  # 改为DEBUG级别以获取更多信息
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )

def simple_callback(market_data):
    """简单的回调函数"""
    print(f"🔄 回调函数接收到数据: {market_data['symbol']} - 价格: {market_data['close']}")

def test_simple_websocket():
    """测试简单的WebSocket数据打印"""
    print("🔍 测试简单WebSocket数据打印...")
    print("=" * 60)
    
    # 创建配置
    config = {
        'exchange': {
            'id': 'binance',
            'enableRateLimit': True
        },
        'data_fetching': {
            'enable_websocket': True
        }
    }
    
    # 创建交易所接口
    exchange = ExchangeInterface(ExchangeType.BINANCE, config)
    
    # 创建数据获取器
    data_fetcher = DataFetcher(exchange, config)
    
    try:
        # 直接启动WebSocket连接
        symbol = "FILUSDT"
        print(f"🚀 直接启动WebSocket连接: {symbol}")
        print("📊 现在应该能看到实时数据打印了...")
        print("-" * 60)
        
        # 直接调用start_websocket_stream
        data_fetcher.start_websocket_stream(symbol, simple_callback)
        
        # 运行测试
        print("⏳ 运行测试30秒，观察实时数据打印...")
        
        # 每5秒检查一次状态
        for i in range(6):
            time.sleep(5)
            status = data_fetcher.get_websocket_status()
            print(f"\n📈 WebSocket状态 (第{i+1}次检查):")
            for symbol, info in status.items():
                print(f"  {symbol}:")
                print(f"    连接状态: {info['connected']}")
                print(f"    线程状态: {info['thread_alive']}")
                print(f"    回调函数: {info.get('callback', 'None')}")
            
            if not status:
                print("⚠️ 没有活跃的WebSocket连接")
                break
        
    except KeyboardInterrupt:
        print("\n⏹️ 用户中断测试")
    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")
    finally:
        # 停止WebSocket连接
        print("🛑 停止WebSocket连接...")
        data_fetcher.stop_websocket_stream(symbol)
        print("✅ 测试完成")

def main():
    """主函数"""
    setup_logging()
    
    print("🚀 简单WebSocket数据打印测试开始")
    print("=" * 60)
    print("这个测试会直接调用data_fetcher的WebSocket功能")
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
        test_simple_websocket()
    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")
    
    print("\n🏁 测试完成")

if __name__ == "__main__":
    main()

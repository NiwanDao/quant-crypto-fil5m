#!/usr/bin/env python3
"""
测试LiveTradingSystem的WebSocket数据打印功能
"""

import sys
import os
import time
import logging
from datetime import datetime

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from live_trading import LiveTradingSystem, TradingMode

def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )

def test_live_trading_websocket():
    """测试LiveTradingSystem的WebSocket数据打印"""
    print("🔍 测试LiveTradingSystem的WebSocket数据打印功能...")
    print("=" * 60)
    print("注意：现在LiveTradingSystem会打印接收到的实时数据")
    print("=" * 60)
    
    try:
        # 创建LiveTradingSystem实例
        trading_system = LiveTradingSystem(
            config_path='/home/shiyi/quant-crypto-fil5m/live/live_config.yml',
            mode=TradingMode.PAPER
        )
        
        print("🚀 启动LiveTradingSystem...")
        print("📊 现在应该能看到实时数据打印了...")
        print("-" * 60)
        
        # 启动交易系统
        trading_system.start_trading()
        
        # 运行测试
        print("⏳ 运行测试60秒，观察实时数据打印...")
        time.sleep(60)
        
        # 获取WebSocket状态
        ws_status = trading_system.get_websocket_status()
        print(f"\n📈 WebSocket状态:")
        print(f"  启用状态: {ws_status.get('enabled', False)}")
        if ws_status.get('enabled'):
            print(f"  连接状态: {ws_status.get('connections', {})}")
            print(f"  统计信息: {ws_status.get('statistics', {})}")
        
        # 获取最新数据
        latest_data = trading_system.get_latest_websocket_data('FIL/USDT')
        if latest_data:
            print(f"\n📊 最新数据:")
            print(f"  价格: {latest_data['close']}")
            print(f"  时间: {latest_data['timestamp']}")
        
    except KeyboardInterrupt:
        print("\n⏹️ 用户中断测试")
    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")
    finally:
        # 停止交易系统
        print("🛑 停止LiveTradingSystem...")
        trading_system.stop()
        print("✅ 测试完成")

def main():
    """主函数"""
    setup_logging()
    
    print("🚀 LiveTradingSystem WebSocket数据打印测试开始")
    print("=" * 60)
    print("这个测试会验证LiveTradingSystem中的WebSocket数据打印功能")
    print("你应该能看到类似以下的输出：")
    print("📊 LiveTradingSystem接收到实时数据: FIL/USDT")
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
        test_live_trading_websocket()
    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")
    
    print("\n🏁 测试完成")

if __name__ == "__main__":
    main()

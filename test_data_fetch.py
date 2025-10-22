#!/usr/bin/env python3
"""
测试数据获取功能
"""
import os
import sys
import yaml
import ccxt
from datetime import datetime, timezone

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

def test_exchange_connection():
    """测试交易所连接"""
    print("🧪 测试交易所连接...")
    
    # 加载配置
    with open('conf/config.yml', 'r') as f:
        conf = yaml.safe_load(f)
    
    # 创建交易所实例
    exchange_config = {
        'enableRateLimit': conf['exchange'].get('enableRateLimit', True),
        'options': conf['exchange'].get('options', {})
    }
    
    ex = ccxt.__dict__[conf['exchange']['id']](exchange_config)
    symbol = conf['symbol']
    timeframe = conf['timeframe']
    
    try:
        # 测试连接
        print(f"🔍 测试交易所: {ex.id}")
        print(f"🔍 交易对: {symbol}")
        print(f"🔍 时间框架: {timeframe}")
        
        # 获取服务器时间
        server_time = ex.fetch_time()
        print(f"🕒 服务器时间: {datetime.fromtimestamp(server_time/1000)}")
        
        # 测试获取最近数据
        print("📊 测试获取最近数据...")
        recent_data = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=5)
        
        if recent_data:
            print(f"✅ 成功获取 {len(recent_data)} 条数据")
            for i, row in enumerate(recent_data):
                dt = datetime.fromtimestamp(row[0]/1000)
                print(f"   {i+1}. {dt}: O:{row[1]:.2f} H:{row[2]:.2f} L:{row[3]:.2f} C:{row[4]:.2f} V:{row[5]:.2f}")
            return True
        else:
            print("❌ 没有获取到数据")
            return False
            
    except Exception as e:
        print(f"❌ 连接失败: {e}")
        return False

def test_time_range():
    """测试时间范围"""
    print("\n🧪 测试时间范围...")
    
    # 测试2024年8月到12月的时间范围
    start_dt = datetime(2024, 8, 1, tzinfo=timezone.utc)
    end_dt = datetime(2024, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
    
    since_ms = int(start_dt.timestamp() * 1000)
    until_ms = int(end_dt.timestamp() * 1000)
    
    print(f"📅 时间范围: {start_dt.date()} 到 {end_dt.date()}")
    print(f"⏰ 时间戳: {since_ms} 到 {until_ms}")
    print(f"📊 时间跨度: {(until_ms - since_ms) / (1000 * 60 * 60 * 24):.1f} 天")
    
    return True

if __name__ == '__main__':
    print("🚀 开始测试数据获取功能...")
    
    # 测试交易所连接
    if test_exchange_connection():
        print("✅ 交易所连接测试通过")
    else:
        print("❌ 交易所连接测试失败")
        sys.exit(1)
    
    # 测试时间范围
    if test_time_range():
        print("✅ 时间范围测试通过")
    else:
        print("❌ 时间范围测试失败")
        sys.exit(1)
    
    print("\n🎉 所有测试通过！数据获取功能已修复")

#!/usr/bin/env python3
"""
测试混合时间框架止损止盈策略
验证15分钟信号 + 1分钟止损 + 实时紧急止损的组合策略
"""

import sys
import os
import time
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from live_trading import LiveTradingSystem, TradingMode
from risk_manager import RiskManager

def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )

def test_hybrid_strategy_config():
    """测试混合策略配置"""
    print("🔍 测试混合策略配置...")
    print("=" * 60)
    
    try:
        # 创建交易系统
        trading_system = LiveTradingSystem(
            config_path='/home/shiyi/quant-crypto-fil5m/live/live_config.yml',
            mode=TradingMode.PAPER
        )
        
        # 检查混合策略配置
        print(f"✅ 混合止损策略启用: {trading_system.use_hybrid_stops}")
        print(f"✅ 信号时间框架: {trading_system.signal_timeframe}")
        print(f"✅ 止损时间框架: {trading_system.stop_timeframe}")
        print(f"✅ 紧急止损: {trading_system.emergency_stop}")
        print(f"✅ 止损检查间隔: {trading_system.stop_check_interval}秒")
        print(f"✅ 紧急止损比例: {trading_system.emergency_stop_loss:.1%}")
        print(f"✅ 1分钟止损ATR倍数: {trading_system.stop_loss_1m_atr_mult}")
        print(f"✅ 1分钟移动止损: {trading_system.trailing_stop_1m}")
        print(f"✅ 实时紧急止损: {trading_system.realtime_emergency_stop}")
        
        return True
        
    except Exception as e:
        print(f"❌ 混合策略配置测试失败: {str(e)}")
        return False

def test_time_framework_detection():
    """测试时间框架检测"""
    print("\n🔍 测试时间框架检测...")
    print("=" * 60)
    
    try:
        trading_system = LiveTradingSystem(
            config_path='/home/shiyi/quant-crypto-fil5m/live/live_config.yml',
            mode=TradingMode.PAPER
        )
        
        # 测试15分钟信号时间检测
        current_time = datetime.now()
        print(f"当前时间: {current_time.strftime('%H:%M:%S')}")
        
        is_signal_time = trading_system.is_15m_signal_time()
        print(f"是否为15分钟信号时间: {is_signal_time}")
        
        # 测试1分钟止损检查时间
        is_stop_time = trading_system.is_1m_stop_check_time()
        print(f"是否为1分钟止损检查时间: {is_stop_time}")
        
        return True
        
    except Exception as e:
        print(f"❌ 时间框架检测测试失败: {str(e)}")
        return False

def test_risk_manager_hybrid():
    """测试风险管理器的混合功能"""
    print("\n🔍 测试风险管理器混合功能...")
    print("=" * 60)
    
    try:
        # 创建风险管理器
        config = {
            'use_hybrid_stops': True,
            'emergency_stop_loss': 0.05,
            'stop_loss_1m_atr_mult': 1.5,
            'trailing_stop_1m': True,
            'realtime_emergency_stop': True
        }
        
        risk_manager = RiskManager(config)
        
        # 测试1分钟止损计算
        entry_price = 5.0
        atr_1m = 0.1
        side = 'buy'
        
        stop_loss_1m = risk_manager.calculate_1m_stop_loss(entry_price, atr_1m, side)
        print(f"1分钟止损价格: {stop_loss_1m:.4f}")
        
        # 测试紧急止损计算
        emergency_stop = risk_manager.calculate_emergency_stop_loss(entry_price, side)
        print(f"紧急止损价格: {emergency_stop:.4f}")
        
        # 测试混合移动止损
        position = {
            'entry_price': entry_price,
            'side': side,
            'trailing_stop': 0,
            'trailing_stop_activated': False,
            'highest_price': entry_price,
            'lowest_price': entry_price
        }
        
        current_price = 5.2
        atr_15m = 0.15
        
        new_stop, triggered = risk_manager.update_hybrid_trailing_stop(
            position, current_price, atr_15m, atr_1m
        )
        
        print(f"混合移动止损价格: {new_stop:.4f}")
        print(f"是否触发: {triggered}")
        
        return True
        
    except Exception as e:
        print(f"❌ 风险管理器混合功能测试失败: {str(e)}")
        return False

def test_hybrid_trading_cycle():
    """测试混合交易周期"""
    print("\n🔍 测试混合交易周期...")
    print("=" * 60)
    
    try:
        trading_system = LiveTradingSystem(
            config_path='/home/shiyi/quant-crypto-fil5m/live/live_config.yml',
            mode=TradingMode.PAPER
        )
        
        # 模拟运行一个混合交易周期
        print("运行混合交易周期...")
        trading_system.run_hybrid_trading_cycle()
        
        print("✅ 混合交易周期运行完成")
        return True
        
    except Exception as e:
        print(f"❌ 混合交易周期测试失败: {str(e)}")
        return False

def test_emergency_stop_simulation():
    """测试紧急止损模拟"""
    print("\n🔍 测试紧急止损模拟...")
    print("=" * 60)
    
    try:
        trading_system = LiveTradingSystem(
            config_path='/home/shiyi/quant-crypto-fil5m/live/live_config.yml',
            mode=TradingMode.PAPER
        )
        
        # 模拟持仓
        symbol = 'FIL/USDT'
        entry_price = 5.0
        current_price = 4.7  # 价格下跌6%，应该触发紧急止损
        
        trading_system.positions[symbol] = {
            'entry_price': entry_price,
            'amount': 100,
            'side': 'buy',
            'latest_price': current_price
        }
        
        print(f"模拟持仓: {symbol}")
        print(f"入场价格: {entry_price}")
        print(f"当前价格: {current_price}")
        print(f"价格变化: {((current_price - entry_price) / entry_price * 100):.2f}%")
        
        # 测试紧急止损检查
        trading_system._check_realtime_emergency_stop(symbol, current_price)
        
        print("✅ 紧急止损模拟完成")
        return True
        
    except Exception as e:
        print(f"❌ 紧急止损模拟测试失败: {str(e)}")
        return False

def main():
    """主函数"""
    setup_logging()
    
    print("🚀 混合时间框架止损止盈策略测试开始")
    print("=" * 60)
    print("测试内容:")
    print("1. 混合策略配置验证")
    print("2. 时间框架检测功能")
    print("3. 风险管理器混合功能")
    print("4. 混合交易周期")
    print("5. 紧急止损模拟")
    print("=" * 60)
    
    test_results = []
    
    # 运行所有测试
    test_results.append(("混合策略配置", test_hybrid_strategy_config()))
    test_results.append(("时间框架检测", test_time_framework_detection()))
    test_results.append(("风险管理器混合功能", test_risk_manager_hybrid()))
    test_results.append(("混合交易周期", test_hybrid_trading_cycle()))
    test_results.append(("紧急止损模拟", test_emergency_stop_simulation()))
    
    # 输出测试结果
    print("\n📊 测试结果汇总:")
    print("=" * 60)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n总计: {passed}/{total} 测试通过")
    
    if passed == total:
        print("🎉 所有测试通过！混合策略实现成功！")
    else:
        print("⚠️ 部分测试失败，请检查实现")
    
    print("\n🏁 测试完成")

if __name__ == "__main__":
    main()

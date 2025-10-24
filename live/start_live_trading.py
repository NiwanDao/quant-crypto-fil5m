#!/usr/bin/env python3
"""
实盘交易启动脚本
支持模拟交易和实盘交易模式
"""

import os
import sys
import yaml
import logging
import argparse
from datetime import datetime
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from live.live_trading import LiveTradingSystem, TradingMode
from live.exchange_interface import create_exchange
from live.data_fetcher import DataFetcher
from live.risk_manager import RiskManager
from live.monitoring import TradingMonitor

def setup_logging(config):
    """设置日志系统"""
    log_config = config.get('logging', {})
    log_level = getattr(logging, log_config.get('level', 'INFO'))
    
    # 创建日志目录
    os.makedirs('logs', exist_ok=True)
    
    # 配置日志格式
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # 文件日志
    file_handler = logging.FileHandler(
        f'logs/live_trading_{datetime.now().strftime("%Y%m%d")}.log'
    )
    file_handler.setFormatter(logging.Formatter(log_format))
    
    # 控制台日志
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(log_format))
    
    # 配置根日志器
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.addHandler(file_handler)
    
    if log_config.get('console_output', True):
        root_logger.addHandler(console_handler)
    
    return root_logger

def load_config(config_path: str):
    """加载配置文件"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"❌ 配置文件加载失败: {str(e)}")
        sys.exit(1)

def validate_config(config):
    """验证配置"""
    required_sections = ['trading', 'exchange', 'model', 'risk_management']
    
    for section in required_sections:
        if section not in config:
            print(f"❌ 配置文件缺少必要部分: {section}")
            sys.exit(1)
    
    # 检查交易模式
    trading_mode = config['trading']['mode']
    if trading_mode not in ['paper', 'live']:
        print(f"❌ 无效的交易模式: {trading_mode}")
        sys.exit(1)
    
    # 检查交易所配置
    if trading_mode == 'live':
        exchange_config = config['exchange']
        if not exchange_config.get('api_key') and not os.getenv('BINANCE_API_KEY'):
            print("⚠️ 实盘交易需要设置API密钥")
            print("请设置环境变量 BINANCE_API_KEY 和 BINANCE_SECRET_KEY")
            confirm = input("确认继续? (yes/no): ").strip().lower()
            if confirm != 'yes':
                sys.exit(1)
    
    return True

def setup_environment():
    """设置环境"""
    # 设置环境变量
    os.environ.setdefault('PYTHONPATH', str(project_root))
    
    # 创建必要的目录
    directories = ['logs', 'logs/monitoring', 'logs/trading', 'data/live']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def create_trading_components(config):
    """创建交易组件"""
    try:
        # 创建交易所接口
        exchange_config = config['exchange']
        paper_trading = config['trading']['mode'] == 'paper'
        
        exchange = create_exchange(
            exchange_type=exchange_config['type'],
            config=exchange_config,
            paper_trading=paper_trading
        )
        
        # 创建数据获取器
        data_fetcher = DataFetcher(exchange, config.get('data_fetching', {}))
        
        # 创建风险管理器
        risk_manager = RiskManager(config.get('risk_management', {}))
        
        # 创建监控器
        monitor = TradingMonitor(config.get('monitoring', {}))
        
        return exchange, data_fetcher, risk_manager, monitor
        
    except Exception as e:
        print(f"❌ 组件创建失败: {str(e)}")
        sys.exit(1)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='实盘交易系统启动脚本')
    parser.add_argument('--config', '-c', default='live/live_config.yml', 
                       help='配置文件路径')
    parser.add_argument('--mode', '-m', choices=['paper', 'live'], 
                       help='交易模式 (覆盖配置文件)')
    parser.add_argument('--symbol', '-s', help='交易对 (覆盖配置文件)')
    parser.add_argument('--dry-run', action='store_true', 
                       help='干运行模式，不执行实际交易')
    parser.add_argument('--verbose', '-v', action='store_true', 
                       help='详细输出')
    
    args = parser.parse_args()
    
    print("🚀 启动实盘交易系统")
    print("=" * 50)
    
    # 设置环境
    setup_environment()
    
    # 加载配置
    config = load_config(args.config)
    
    # 覆盖配置
    if args.mode:
        config['trading']['mode'] = args.mode
    if args.symbol:
        config['trading']['symbol'] = args.symbol
    
    # 验证配置
    validate_config(config)
    
    # 设置日志
    logger = setup_logging(config)
    
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # 显示配置信息
    print(f"📊 交易模式: {config['trading']['mode']}")
    print(f"💰 交易对: {config['trading']['symbol']}")
    print(f"⏰ 时间周期: {config['trading']['timeframe']}")
    print(f"🏦 交易所: {config['exchange']['type']}")
    
    if config['trading']['mode'] == 'live':
        print("⚠️ 警告: 实盘交易模式将使用真实资金!")
        if not args.dry_run:
            confirm = input("确认继续? (yes/no): ").strip().lower()
            if confirm != 'yes':
                print("❌ 已取消")
                sys.exit(0)
    
    try:
        # 创建交易组件
        exchange, data_fetcher, risk_manager, monitor = create_trading_components(config)
        
        # 创建交易系统
        trading_system = LiveTradingSystem(
            config_path=args.config,
            mode=TradingMode.PAPER if config['trading']['mode'] == 'paper' else TradingMode.LIVE
        )
        
        # 注入组件
        trading_system.exchange = exchange
        trading_system.data_fetcher = data_fetcher
        trading_system.risk_manager = risk_manager
        trading_system.monitor = monitor
        
        # 启动监控
        if config.get('monitoring', {}).get('enabled', True):
            monitor.start_monitoring()
            print("✅ 监控系统已启动")
        
        # 干运行模式
        if args.dry_run:
            print("🔍 干运行模式 - 不执行实际交易")
            print("系统将运行一个周期后退出")
            
            # 运行一个交易周期
            trading_system.run_trading_cycle()
            print("✅ 干运行完成")
            return
        
        # 开始交易
        print("🎯 开始交易...")
        trading_system.start_trading()
        
    except KeyboardInterrupt:
        print("\n⏹️ 用户中断")
        logger.info("用户中断交易系统")
    except Exception as e:
        print(f"❌ 系统错误: {str(e)}")
        logger.error(f"系统错误: {str(e)}", exc_info=True)
        sys.exit(1)
    finally:
        # 清理资源
        try:
            if 'monitor' in locals():
                monitor.stop_monitoring()
            print("🏁 交易系统已停止")
        except Exception as e:
            print(f"⚠️ 清理资源时出错: {str(e)}")

if __name__ == '__main__':
    main()

import os, time, math
import ccxt
import pandas as pd
import yaml
from datetime import datetime, timedelta, timezone
import sys

# 添加路径以便导入utils
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from utils.features import build_features_safe, build_labels_safe,detect_data_leakage

CONF_PATH = 'conf/config.yml'

def load_conf():
    with open(CONF_PATH, 'r') as f:
        return yaml.safe_load(f)

def fetch_ohlcv_all(ex, symbol, timeframe, since_ms, until_ms, batch_limit=1000):
    """获取指定时间范围内的OHLCV数据（改进版本）"""
    all_rows = []
    since = since_ms
    
    print(f"📡 开始获取数据: {symbol} {timeframe}")
    print(f"⏰ 时间范围: {datetime.fromtimestamp(since_ms/1000)} 到 {datetime.fromtimestamp(until_ms/1000)}")
    
    batch_count = 0
    max_batches = 50  # 防止无限循环
    consecutive_errors = 0
    max_consecutive_errors = 3
    
    while since < until_ms and batch_count < max_batches:
        batch_count += 1
        
        try:
            print(f"🔄 获取批次 {batch_count}, 时间戳: {since} ({datetime.fromtimestamp(since/1000)})")
            
            # 使用fetch_ohlcv获取数据
            ohlcv_data = ex.fetch_ohlcv(
                symbol, 
                timeframe=timeframe, 
                since=since, 
                limit=batch_limit
            )
            
            if not ohlcv_data:
                print("❌ 没有获取到数据，可能已到达数据末尾")
                break
            
            # 验证数据质量
            if not validate_ohlcv_data(ohlcv_data):
                print("❌ 数据质量检查失败")
                consecutive_errors += 1
                if consecutive_errors >= max_consecutive_errors:
                    print("❌ 连续错误过多，停止获取")
                    break
                continue
            
            consecutive_errors = 0  # 重置错误计数
            print(f"📊 获取到 {len(ohlcv_data)} 条K线数据")
            
            # 添加到总数据
            all_rows.extend(ohlcv_data)
            
            # 更新下一个起始时间（使用最后一根K线的时间戳 + 1个时间间隔）
            last_timestamp = ohlcv_data[-1][0]
            
            # 计算时间间隔（毫秒）
            timeframe_ms = {
                '15m': 15 * 60 * 1000,
                '1h': 60 * 60 * 1000,
                '4h': 4 * 60 * 60 * 1000,
                '1d': 24 * 60 * 60 * 1000
            }.get(timeframe, 15 * 60 * 1000)  # 默认15分钟
            
            since = last_timestamp + timeframe_ms
            
            # 如果最后一根K线的时间已经超过结束时间，就停止
            if last_timestamp >= until_ms:
                print("✅ 已达到结束时间，停止获取")
                break
                
            # 速率限制
            time.sleep(ex.rateLimit / 1000.0 if hasattr(ex, 'rateLimit') else 0.5)
            
        except Exception as e:
            print(f"❌ 获取数据时出错: {e}")
            consecutive_errors += 1
            if consecutive_errors >= max_consecutive_errors:
                print("❌ 连续错误过多，停止获取")
                break
            time.sleep(2)  # 错误后等待更长时间
    
    print(f"✅ 数据获取完成，总共 {len(all_rows)} 条记录")
    return all_rows

def validate_ohlcv_data(ohlcv_data):
    """验证OHLCV数据质量"""
    if not ohlcv_data or len(ohlcv_data) == 0:
        return False
    
    for row in ohlcv_data:
        if len(row) != 6:
            print(f"❌ 数据格式错误: 期望6列，实际{len(row)}列")
            return False
        
        timestamp, open_price, high, low, close, volume = row
        
        # 检查时间戳
        if timestamp <= 0:
            print(f"❌ 无效时间戳: {timestamp}")
            return False
        
        # 检查价格数据
        if not all(isinstance(x, (int, float)) and x > 0 for x in [open_price, high, low, close]):
            print(f"❌ 无效价格数据: O:{open_price} H:{high} L:{low} C:{close}")
            return False
        
        # 检查OHLC逻辑
        if not (low <= open_price <= high and low <= close <= high):
            print(f"❌ OHLC逻辑错误: O:{open_price} H:{high} L:{low} C:{close}")
            return False
        
        # 检查成交量
        if not isinstance(volume, (int, float)) or volume < 0:
            print(f"❌ 无效成交量: {volume}")
            return False
    
    return True

def validate_final_data(df):
    """验证最终数据的质量"""
    print("🔍 检查数据完整性...")
    
    # 检查基本数据
    if df.empty:
        print("❌ 数据为空")
        return False
    
    # 检查必要列
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"❌ 缺少必要列: {missing_columns}")
        return False
    
    # 检查数据范围
    print(f"📊 数据统计:")
    print(f"   - 总记录数: {len(df)}")
    print(f"   - 时间范围: {df.index.min()} 到 {df.index.max()}")
    print(f"   - 价格范围: {df['low'].min():.2f} - {df['high'].max():.2f}")
    
    # 检查缺失值
    missing_data = df[required_columns].isnull().sum()
    if missing_data.any():
        print(f"❌ 发现缺失值: {missing_data[missing_data > 0].to_dict()}")
        return False
    
    # 检查价格逻辑
    invalid_ohlc = df[(df['low'] > df['high']) | 
                     (df['open'] < df['low']) | (df['open'] > df['high']) |
                     (df['close'] < df['low']) | (df['close'] > df['high'])]
    
    if not invalid_ohlc.empty:
        print(f"❌ 发现 {len(invalid_ohlc)} 条OHLC逻辑错误的数据")
        return False
    
    # 检查负值
    negative_values = df[(df[required_columns] <= 0).any(axis=1)]
    if not negative_values.empty:
        print(f"❌ 发现 {len(negative_values)} 条包含负值或零的数据")
        return False
    
    print("✅ 数据质量检查通过")
    return True

def debug_exchange_info(ex, symbol):
    """调试交易所和信息"""
    print(f"🔍 交易所: {ex.id}")
    print(f"🔍 交易对: {symbol}")
    
    try:
        # 检查市场信息
        markets = ex.load_markets()
        if symbol in markets:
            print(f"✅ 交易对 {symbol} 存在")
        else:
            print(f"❌ 交易对 {symbol} 不存在")
            print(f"可用交易对示例: {list(markets.keys())[:5]}")
            
        # 检查服务器时间
        server_time = ex.fetch_time()
        print(f"🕒 服务器时间: {datetime.fromtimestamp(server_time/1000)}")
        
    except Exception as e:
        print(f"❌ 交易所信息检查失败: {e}")

def test_fetch_recent_data(ex, symbol, timeframe):
    """测试获取最近的数据"""
    print("🧪 测试获取最近数据...")
    try:
        # 获取最近的数据
        recent_data = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=5)
        if recent_data:
            print(f"✅ 测试成功！获取到 {len(recent_data)} 条最近数据")
            for i, row in enumerate(recent_data[-3:]):  # 显示最后3条
                dt = datetime.fromtimestamp(row[0]/1000)
                print(f"   {i+1}. {dt}: O:{row[1]:.2f} H:{row[2]:.2f} L:{row[3]:.2f} C:{row[4]:.2f} V:{row[5]:.2f}")
            return True
        else:
            print("❌ 测试失败：没有获取到数据")
            return False
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

def main():
    conf = load_conf()
    
    # 创建交易所实例
    exchange_config = {
        'enableRateLimit': conf['exchange'].get('enableRateLimit', True),
        'options': conf['exchange'].get('options', {})
    }
    
    # 添加代理配置（如果需要）
    if 'proxy' in conf['exchange']:
        exchange_config['proxies'] = conf['exchange']['proxy']
    
    ex = ccxt.__dict__[conf['exchange']['id']](exchange_config)
    
    symbol = conf['symbol']
    timeframe = conf['timeframe']
    batch_limit = conf['fetch'].get('batch_limit', 500)

    # 调试信息
    debug_exchange_info(ex, symbol)
    
    # 测试获取数据
    if not test_fetch_recent_data(ex, symbol, timeframe):
        print("❌ 无法获取数据，请检查配置和网络连接")
        return

    # 定义时间段：2025-3到2025-6（修正时间范围）
    start_dt = datetime(2025, 3, 1, tzinfo=timezone.utc)
    end_dt = datetime(2025, 6, 30, 23, 59, 59, tzinfo=timezone.utc)  # 2025年3月到6月
    
    since_ms = int(start_dt.timestamp() * 1000)
    until_ms = int(end_dt.timestamp() * 1000)

    print(f'\n🎯 开始获取历史数据: {symbol} {timeframe}')
    print(f'📅 时间范围: {start_dt.date()} 到 {end_dt.date()}')
    print(f'⏰ 时间戳: {since_ms} 到 {until_ms}')

    rows = fetch_ohlcv_all(ex, symbol, timeframe, since_ms, until_ms, batch_limit=batch_limit)
    
    if not rows:
        print("❌ 没有获取到任何数据")
        return
        
    # 处理数据
    df = pd.DataFrame(rows, columns=['ts','open','high','low','close','volume'])
    df['ts'] = pd.to_datetime(df['ts'], unit='ms', utc=True)
    df.set_index('ts', inplace=True)
    df.sort_index(inplace=True)
    
    print(f"📈 数据统计:")
    print(f"   - 总记录数: {len(df)}")
    print(f"   - 时间范围: {df.index.min()} 到 {df.index.max()}")
    print(f"   - 价格范围: {df['low'].min():.2f} - {df['high'].max():.2f}")
    print(f"   - 最新收盘价: {df['close'].iloc[-1]:.2f}")

    # 构建特征和标签
    print("\n🔧 构建特征和标签...")
    df = build_features_safe(df)
    
    df = build_labels_safe(df, forward_n=conf['features']['forward_n'], thr=conf['features']['label_threshold'])
    
    if not detect_data_leakage(df):
        print("❌ 数据泄露检测失败，停止处理")
        return None

    # 最终数据质量检查
    print("\n🔍 进行最终数据质量检查...")
    if not validate_final_data(df):
        print("❌ 最终数据质量检查失败")
        return
    
    # 保存数据
    os.makedirs('data', exist_ok=True)
    output_file = 'data/feat_2025_3_to_2025_6.parquet'
    df.to_parquet(output_file)
    
    print(f'\n✅ 数据保存成功: {output_file}')
    print(f'📊 最终数据形状: {df.shape}')
    print(f'📅 数据时间范围: {df.index.min()} 到 {df.index.max()}')

if __name__ == '__main__':
    main()
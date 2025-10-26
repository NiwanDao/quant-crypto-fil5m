import os
import time
import ccxt
import pandas as pd
import yaml
from datetime import datetime, timezone
import argparse
import sys

# 添加路径以便导入utils
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from utils.features import build_features_safe, build_multi_feature_labels, detect_data_leakage

CONF_PATH = "conf/config.yml"


def load_conf():
    with open(CONF_PATH, "r") as f:
        return yaml.safe_load(f)


def fetch_ohlcv_all(ex, symbol, timeframe, since_ms, until_ms, batch_limit=1000):
    """获取指定时间范围内的OHLCV数据（改进版本）"""
    all_rows = []
    since = since_ms

    print(f"📡 开始获取数据: {symbol} {timeframe}")
    print(
        f"⏰ 时间范围: {datetime.fromtimestamp(since_ms/1000)} 到 {datetime.fromtimestamp(until_ms/1000)}"
    )

    batch_count = 0
    max_batches = 50  # 防止无限循环
    consecutive_errors = 0
    max_consecutive_errors = 3

    while since < until_ms and batch_count < max_batches:
        batch_count += 1

        try:
            print(
                f"🔄 获取批次 {batch_count}, 时间戳: {since} ({datetime.fromtimestamp(since/1000)})"
            )

            # 使用fetch_ohlcv获取数据
            ohlcv_data = ex.fetch_ohlcv(
                symbol, timeframe=timeframe, since=since, limit=batch_limit
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
                "15m": 15 * 60 * 1000,
                "1h": 60 * 60 * 1000,
                "4h": 4 * 60 * 60 * 1000,
                "1d": 24 * 60 * 60 * 1000,
            }.get(
                timeframe, 15 * 60 * 1000
            )  # 默认15分钟

            since = last_timestamp + timeframe_ms

            # 如果最后一根K线的时间已经超过结束时间，就停止
            if last_timestamp >= until_ms:
                print("✅ 已达到结束时间，停止获取")
                break

            # 速率限制
            time.sleep(ex.rateLimit / 1000.0 if hasattr(ex, "rateLimit") else 0.5)

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
        if not all(
            isinstance(x, (int, float)) and x > 0
            for x in [open_price, high, low, close]
        ):
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
    required_columns = ["open", "high", "low", "close", "volume"]
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
    invalid_ohlc = df[
        (df["low"] > df["high"])
        | (df["open"] < df["low"])
        | (df["open"] > df["high"])
        | (df["close"] < df["low"])
        | (df["close"] > df["high"])
    ]

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
                dt = datetime.fromtimestamp(row[0] / 1000)
                print(
                    f"   {i+1}. {dt}: O:{row[1]:.2f} H:{row[2]:.2f} L:{row[3]:.2f} C:{row[4]:.2f} V:{row[5]:.2f}"
                )
            return True
        else:
            print("❌ 测试失败：没有获取到数据")
            return False
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False


def parse_date_arg(date_str, is_end=False):
    """解析日期参数，支持 'YYYY-MM-DD'、'YYYY-M-D' 和 'now'（仅限结束时间）"""
    if date_str.lower() == "now":
        if not is_end:
            raise ValueError("'now' 仅可用于 --to")
        return datetime.now(timezone.utc)

    # 使用 pandas 解析并设为 UTC 起始/结束
    dt = pd.to_datetime(date_str, utc=True)
    dt = dt.to_pydatetime()
    if is_end:
        # 设为当天 23:59:59（与原脚本一致）
        return dt.replace(
            hour=23, minute=59, second=59, microsecond=0, tzinfo=timezone.utc
        )
    return dt.replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=timezone.utc)


def default_output_path(start_dt: datetime, end_dt: datetime | None):
    def ym(dt: datetime):
        return f"{dt.year}_{dt.month}"

    start_part = ym(start_dt)
    end_part = "now" if end_dt is None else ym(end_dt)
    return os.path.join("data", f"feat_{start_part}_to_{end_part}.parquet")


def ensure_parent_dir(path: str):
    parent = os.path.dirname(path)
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)


def process_range(
    ex,
    conf,
    start_dt: datetime,
    end_dt: datetime | None,
    output_file: str | None = None,
):
    since_ms = int(start_dt.timestamp() * 1000)
    until_ms = int((end_dt or datetime.now(timezone.utc)).timestamp() * 1000)

    symbol = conf["symbol"]
    timeframe = conf["timeframe"]
    batch_limit = conf["fetch"].get("batch_limit", 500)

    print(f"\n🎯 开始获取历史数据: {symbol} {timeframe}")
    print(
        f"📅 时间范围: {start_dt.date()} 到 {(end_dt.date() if end_dt else datetime.now(timezone.utc).date())}"
    )
    print(f"⏰ 时间戳: {since_ms} 到 {until_ms}")

    rows = fetch_ohlcv_all(
        ex, symbol, timeframe, since_ms, until_ms, batch_limit=batch_limit
    )

    if not rows:
        print("❌ 没有获取到任何数据")
        return

    # 处理数据
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df.set_index("ts", inplace=True)
    df.sort_index(inplace=True)

    print(f"📈 数据统计:")
    print(f"   - 总记录数: {len(df)}")
    print(f"   - 时间范围: {df.index.min()} 到 {df.index.max()}")
    print(f"   - 价格范围: {df['low'].min():.2f} - {df['high'].max():.2f}")
    print(f"   - 最新收盘价: {df['close'].iloc[-1]:.2f}")

    # 构建特征和标签
    print("\n🔧 构建特征和标签...")
    df = build_features_safe(df)

    df = build_multi_feature_labels(
        df,
        forward_n=conf["features"]["forward_n"],
        base_thr=conf["features"]["label_threshold"],
    )

    # 数据泄露检测
    if not detect_data_leakage(df):
        print("❌ 数据泄露检测失败，停止处理")
        return None

    # 最终数据质量检查
    print("\n🔍 进行最终数据质量检查...")
    if not validate_final_data(df):
        print("❌ 最终数据质量检查失败")
        return

    # 保存数据
    final_output = output_file or default_output_path(start_dt, end_dt)
    ensure_parent_dir(final_output)
    df.to_parquet(final_output)

    print(f"\n✅ 数据保存成功: {final_output}")
    print(f"📊 最终数据形状: {df.shape}")
    print(f"📅 数据时间范围: {df.index.min()} 到 {df.index.max()}")


def main():
    parser = argparse.ArgumentParser(description="按时间范围生成特征数据集")
    parser.add_argument(
        "--from", dest="start_date", required=False, help="开始日期，如 2024-08-01"
    )
    parser.add_argument(
        "--to", dest="end_date", required=False, help="结束日期，如 2025-03-31 或 now"
    )
    parser.add_argument(
        "--out", dest="output_path", default=None, help="输出文件路径（.parquet）"
    )
    parser.add_argument(
        "--run-all",
        action="store_true",
        help="一次性生成三个数据集：2024-08~2025-03、2025-03~2025-06、2025-06~now",
    )
    args = parser.parse_args()

    conf = load_conf()

    # 创建交易所实例
    exchange_config = {
        "enableRateLimit": conf["exchange"].get("enableRateLimit", True),
        "options": conf["exchange"].get("options", {}),
    }

    # 添加代理配置（如果需要）
    if "proxy" in conf["exchange"]:
        exchange_config["proxies"] = conf["exchange"]["proxy"]

    ex = ccxt.__dict__[conf["exchange"]["id"]](exchange_config)

    symbol = conf["symbol"]
    timeframe = conf["timeframe"]
    batch_limit = conf["fetch"].get("batch_limit", 500)

    # 调试信息
    debug_exchange_info(ex, symbol)

    # 测试获取数据
    if not test_fetch_recent_data(ex, symbol, timeframe):
        print("❌ 无法获取数据，请检查配置和网络连接")
        return

    # --run-all 模式：一次性跑完三段数据
    if args.run_all:
        ranges = [
            (
                datetime(2024, 8, 1, tzinfo=timezone.utc),
                datetime(2025, 3, 31, 23, 59, 59, tzinfo=timezone.utc),
                "data/feat_2024_8_to_2025_3.parquet",
            ),
            (
                datetime(2025, 3, 1, tzinfo=timezone.utc),
                datetime(2025, 6, 30, 23, 59, 59, tzinfo=timezone.utc),
                "data/feat_2025_3_to_2025_6.parquet",
            ),
            (
                datetime(2025, 6, 1, tzinfo=timezone.utc),
                None,
                "data/feat_2025_6_to_now.parquet",
            ),
        ]

        for idx, (s_dt, e_dt, out_path) in enumerate(ranges, 1):
            print(f"\n===== 开始第 {idx} 段 =====")
            process_range(ex, conf, s_dt, e_dt, out_path)
        print("\n🎉 全部数据集生成完成！")
        return

    # 单段运行模式（需要提供 --from 与 --to）
    if not args.start_date or not args.end_date:
        print("❌ 请提供 --from 与 --to，或使用 --run-all")
        return

    start_dt = parse_date_arg(args.start_date, is_end=False)
    end_dt = (
        None
        if args.end_date.lower() == "now"
        else parse_date_arg(args.end_date, is_end=True)
    )

    if end_dt is not None and start_dt > end_dt:
        raise ValueError("开始日期不能晚于结束日期")

    process_range(ex, conf, start_dt, end_dt, args.output_path)


if __name__ == "__main__":
    main()

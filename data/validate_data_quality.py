#!/usr/bin/env python3
"""
数据质量验证脚本 - 检查各时间段数据的完整性和质量
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, timezone

def validate_data_file(file_path, period_name):
    """验证单个数据文件的质量"""
    print(f"\n{'='*50}")
    print(f"验证 {period_name} 数据: {file_path}")
    print(f"{'='*50}")
    
    if not os.path.exists(file_path):
        print(f"❌ 文件不存在: {file_path}")
        return False
    
    try:
        # 读取数据
        df = pd.read_parquet(file_path)
        
        print(f"📊 数据概览:")
        print(f"   - 总行数: {len(df):,}")
        print(f"   - 时间范围: {df.index.min()} 到 {df.index.max()}")
        print(f"   - 列数: {len(df.columns)}")
        
        # 检查时间连续性
        time_diff = df.index.to_series().diff()
        expected_interval = pd.Timedelta(minutes=15)  # 15分钟K线
        gaps = time_diff[time_diff > expected_interval * 1.5]  # 允许1.5倍间隔的容差
        
        if len(gaps) > 0:
            print(f"⚠️  发现 {len(gaps)} 个时间间隔异常")
            print(f"   最大间隔: {gaps.max()}")
        else:
            print("✅ 时间连续性正常")
        
        # 检查缺失值
        missing_data = df.isnull().sum()
        if missing_data.sum() > 0:
            print(f"⚠️  发现缺失值:")
            for col, count in missing_data[missing_data > 0].items():
                print(f"   - {col}: {count} 个缺失值")
        else:
            print("✅ 无缺失值")
        
        # 检查价格数据合理性
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            if col in df.columns:
                if (df[col] <= 0).any():
                    print(f"⚠️  {col} 列存在非正值")
                else:
                    print(f"✅ {col} 列数据合理")
        
        # 检查OHLC逻辑
        if all(col in df.columns for col in price_cols):
            invalid_ohlc = (df['high'] < df['low']) | (df['high'] < df['open']) | (df['high'] < df['close']) | (df['low'] > df['open']) | (df['low'] > df['close'])
            if invalid_ohlc.any():
                print(f"⚠️  发现 {invalid_ohlc.sum()} 个OHLC逻辑错误")
            else:
                print("✅ OHLC逻辑正确")
        
        # 检查特征列
        feature_cols = [col for col in df.columns if col not in ['open', 'high', 'low', 'close', 'volume']]
        if feature_cols:
            print(f"✅ 包含 {len(feature_cols)} 个特征列")
        
        # 检查标签列
        if 'label' in df.columns:
            label_dist = df['label'].value_counts()
            print(f"📈 标签分布:")
            for label, count in label_dist.items():
                print(f"   - {label}: {count} ({count/len(df)*100:.1f}%)")
        
        return True
        
    except Exception as e:
        print(f"❌ 读取文件失败: {e}")
        return False

def main():
    print("开始验证各时间段数据质量...")
    print(f"验证时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 定义要验证的文件
    data_files = [
        ('data/feat_2024_8_to_2025_3.parquet', '2024-8 到 2025-3'),
        ('data/feat_2025_3_to_2025_6.parquet', '2025-3 到 2025-6'),
        ('data/feat_2025_6_to_now.parquet', '2025-6 至今')
    ]
    
    success_count = 0
    total_count = len(data_files)
    
    for file_path, period_name in data_files:
        if validate_data_file(file_path, period_name):
            success_count += 1
    
    print(f"\n{'='*60}")
    print(f"数据质量验证总结")
    print(f"{'='*60}")
    print(f"验证通过: {success_count}/{total_count}")
    print(f"完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if success_count == total_count:
        print("🎉 所有数据质量验证通过！")
    else:
        print("⚠️  部分数据质量验证失败，请检查数据文件")

if __name__ == '__main__':
    main()

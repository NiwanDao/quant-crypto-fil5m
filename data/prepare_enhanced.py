"""
增强版数据预处理脚本
包含数据清洗、特征工程、标签生成和平衡处理
"""

import os
import sys
import pandas as pd
import numpy as np
import yaml
from datetime import datetime, timedelta, timezone
import warnings
warnings.filterwarnings('ignore')

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from utils.enhanced_features import (
    build_enhanced_features, 
    build_enhanced_labels, 
    select_enhanced_features,
    detect_enhanced_data_leakage
)

CONF_PATH = '../conf/config.yml'

def load_conf():
    with open(CONF_PATH, 'r') as f:
        return yaml.safe_load(f)

def clean_and_validate_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    数据清洗和验证
    """
    print("🧹 开始数据清洗...")
    
    original_len = len(df)
    
    # 1. 删除重复数据
    df = df.drop_duplicates()
    print(f"📊 删除重复数据: {original_len - len(df)} 条")
    
    # 2. 删除异常价格数据
    df = df[df['close'] > 0]
    df = df[df['high'] >= df['low']]
    df = df[df['high'] >= df['open']]
    df = df[df['high'] >= df['close']]
    df = df[df['low'] <= df['open']]
    df = df[df['low'] <= df['close']]
    
    # 3. 删除异常成交量数据
    df = df[df['volume'] >= 0]
    
    # 4. 删除价格跳跃过大的数据（可能是数据错误）
    price_change = df['close'].pct_change().abs()
    df = df[price_change < 0.5]  # 单次价格变化不超过50%
    
    # 5. 删除缺失值过多的行
    df = df.dropna(thresh=len(df.columns) * 0.8)
    
    print(f"📊 数据清洗完成: {original_len} -> {len(df)} 条记录")
    print(f"📊 数据质量: {len(df)/original_len:.2%}")
    
    return df

def balance_dataset(df: pd.DataFrame, target_col: str = 'y') -> pd.DataFrame:
    """
    数据集平衡处理
    """
    print("⚖️ 开始数据集平衡处理...")
    
    # 分析标签分布
    label_counts = df[target_col].value_counts()
    print(f"📊 原始标签分布:")
    for label, count in label_counts.items():
        print(f"  标签 {label}: {count} ({count/len(df):.2%})")
    
    # 如果数据不平衡，进行平衡处理
    if len(label_counts) == 2:
        pos_count = label_counts.get(1, 0)
        neg_count = label_counts.get(0, 0)
        
        if pos_count > 0 and neg_count > 0:
            # 计算平衡比例
            min_count = min(pos_count, neg_count)
            max_count = max(pos_count, neg_count)
            imbalance_ratio = max_count / min_count
            
            print(f"📊 不平衡比例: {imbalance_ratio:.2f}")
            
            if imbalance_ratio > 2.0:  # 如果严重不平衡
                print("⚠️ 数据严重不平衡，进行平衡处理...")
                
                # 方法1: 下采样多数类
                if pos_count > neg_count:
                    # 正样本多，下采样正样本
                    pos_samples = df[df[target_col] == 1].sample(n=min_count, random_state=42)
                    neg_samples = df[df[target_col] == 0]
                else:
                    # 负样本多，下采样负样本
                    neg_samples = df[df[target_col] == 0].sample(n=min_count, random_state=42)
                    pos_samples = df[df[target_col] == 1]
                
                df_balanced = pd.concat([pos_samples, neg_samples]).sample(frac=1, random_state=42)
                
                print(f"📊 平衡后数据: {len(df_balanced)} 条记录")
                print(f"📊 平衡后分布:")
                balanced_counts = df_balanced[target_col].value_counts()
                for label, count in balanced_counts.items():
                    print(f"  标签 {label}: {count} ({count/len(df_balanced):.2%})")
                
                return df_balanced
    
    print("✅ 数据已平衡或无需平衡处理")
    return df

def create_time_series_splits(df: pd.DataFrame, n_splits: int = 5) -> List[Tuple[int, int]]:
    """
    创建时间序列分割
    """
    print(f"📅 创建时间序列分割: {n_splits} 个分割")
    
    total_len = len(df)
    split_size = total_len // n_splits
    
    splits = []
    for i in range(n_splits):
        start_idx = i * split_size
        end_idx = (i + 1) * split_size if i < n_splits - 1 else total_len
        splits.append((start_idx, end_idx))
        print(f"  分割 {i+1}: {start_idx} - {end_idx} ({end_idx-start_idx} 条记录)")
    
    return splits

def prepare_enhanced_dataset():
    """
    准备增强版数据集
    """
    print("🚀 开始准备增强版数据集...")
    print("=" * 60)
    
    # 加载配置
    conf = load_conf()
    
    # 检查数据文件
    data_files = [
        '../data/feat_2024_8_to_2025_3.parquet',
        '../data/feat_2025_3_to_2025_6.parquet'
    ]
    
    available_files = []
    for file_path in data_files:
        if os.path.exists(file_path):
            available_files.append(file_path)
            print(f"✅ 找到数据文件: {file_path}")
        else:
            print(f"❌ 数据文件不存在: {file_path}")
    
    if not available_files:
        print("❌ 没有找到可用的数据文件")
        return None
    
    # 合并所有可用数据
    print(f"\n📊 合并 {len(available_files)} 个数据文件...")
    all_data = []
    
    for file_path in available_files:
        df = pd.read_parquet(file_path)
        print(f"📊 加载 {file_path}: {df.shape}")
        all_data.append(df)
    
    # 合并数据
    combined_df = pd.concat(all_data, ignore_index=False)
    combined_df = combined_df.sort_index()  # 按时间排序
    print(f"📊 合并后数据: {combined_df.shape}")
    
    # 数据清洗
    print(f"\n🧹 数据清洗...")
    cleaned_df = clean_and_validate_data(combined_df)
    
    # 构建增强特征
    print(f"\n🔧 构建增强特征...")
    df_with_features = build_enhanced_features(cleaned_df)
    print(f"📊 特征构建后: {df_with_features.shape}")
    
    # 构建增强标签
    print(f"\n🎯 构建增强标签...")
    df_with_labels = build_enhanced_labels(df_with_features, forward_n=4, thr=0.01)
    print(f"📊 标签构建后: {df_with_labels.shape}")
    
    # 数据泄露检测
    print(f"\n🔍 进行数据泄露检测...")
    if not detect_enhanced_data_leakage(df_with_labels):
        print("❌ 数据泄露检测失败，停止处理")
        return None
    
    # 数据集平衡
    print(f"\n⚖️ 数据集平衡处理...")
    balanced_df = balance_dataset(df_with_labels)
    
    # 特征选择
    print(f"\n🎯 特征选择...")
    final_df = select_enhanced_features(balanced_df, top_k=60)
    print(f"📊 最终数据: {final_df.shape}")
    
    # 创建时间序列分割
    print(f"\n📅 创建时间序列分割...")
    splits = create_time_series_splits(final_df, n_splits=5)
    
    # 保存处理后的数据
    output_dir = '../data'
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存完整数据
    output_file = os.path.join(output_dir, 'feat_enhanced.parquet')
    final_df.to_parquet(output_file)
    print(f"💾 保存增强数据: {output_file}")
    
    # 保存时间序列分割
    splits_file = os.path.join(output_dir, 'time_series_splits.json')
    import json
    with open(splits_file, 'w') as f:
        json.dump(splits, f, indent=2)
    print(f"💾 保存时间序列分割: {splits_file}")
    
    # 保存数据统计
    stats = {
        'total_samples': len(final_df),
        'total_features': len(final_df.columns) - 1,  # 减去标签列
        'label_distribution': final_df['y'].value_counts(normalize=True).to_dict(),
        'time_range': {
            'start': str(final_df.index.min()),
            'end': str(final_df.index.max())
        },
        'created_at': datetime.now().isoformat()
    }
    
    stats_file = os.path.join(output_dir, 'enhanced_data_stats.json')
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"💾 保存数据统计: {stats_file}")
    
    # 输出最终统计
    print(f"\n" + "="*60)
    print("📊 增强版数据集准备完成")
    print("="*60)
    
    print(f"📊 最终数据统计:")
    print(f"  📈 总样本数: {len(final_df)}")
    print(f"  📈 总特征数: {len(final_df.columns) - 1}")
    print(f"  📈 时间范围: {final_df.index.min()} 到 {final_df.index.max()}")
    print(f"  📈 标签分布: {final_df['y'].value_counts(normalize=True).to_dict()}")
    
    print(f"\n📁 生成的文件:")
    print(f"  - {output_file}: 增强数据")
    print(f"  - {splits_file}: 时间序列分割")
    print(f"  - {stats_file}: 数据统计")
    
    return final_df

def main():
    """
    主函数
    """
    try:
        result = prepare_enhanced_dataset()
        if result is not None:
            print("\n✅ 增强版数据集准备完成！")
            print("🎯 现在可以运行增强版模型训练了")
        else:
            print("\n❌ 数据集准备失败")
    except Exception as e:
        print(f"\n❌ 处理过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()

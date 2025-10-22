def main_safe():
    """安全的数据准备流程"""
    print("🚀 开始安全数据准备...")
    
    # 1. 加载原始数据
    # [你的数据获取代码...]
    
    # 2. 使用安全特征工程
    df_safe = build_features_safe(raw_data)
    
    # 3. 使用安全标签生成
    df_final = build_labels_safe(df_safe, forward_n=4, thr=0.01)
    
    # 4. 检测数据泄露
    if not detect_data_leakage(df_final):
        print("❌ 数据泄露检测失败，停止处理")
        return None
    
    # 5. 特征选择
    final_df = select_important_features(df_final, top_k=25)
    
    print(f"🎉 安全数据准备完成!")
    print(f"📊 最终数据形状: {final_df.shape}")
    print(f"🎯 标签分布: {final_df['y'].value_counts(normalize=True).to_dict()}")
    
    return final_df

# 运行安全版本
if __name__ == '__main__':
    safe_data = main_safe()
    if safe_data is not None:
        safe_data.to_parquet('data/feat_2024_8_to_2025_3_safe.parquet')
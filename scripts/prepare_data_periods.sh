#!/bin/bash

# 数据准备脚本 - 为3个不同时间段准备数据
# 使用方法: ./scripts/prepare_data_periods.sh
export PYTHONPATH=/home/shiyi/quant-crypto-fil5m 


set -e  # 遇到错误时退出

echo "=========================================="
echo "开始准备3个时间段的数据"
echo "=========================================="
echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"

# 切换到项目根目录
cd /home/shiyi/quant-crypto-fil5m

echo ""
echo "1. 准备 2024-8 到 2025-3 数据..."
python data/prepare_2024_8_to_2025_3.py

echo ""
echo "2. 准备 2025-3 到 2025-6 数据..."
python data/prepare_2025_3_to_2025_6.py

echo ""
echo "3. 准备 2025-6 至今数据..."
python data/prepare_2025_6_to_now.py

echo ""
echo "4. 验证数据质量..."
python data/validate_data_quality.py

echo ""
echo "=========================================="
echo "数据准备完成"
echo "完成时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="

# 显示生成的文件
echo ""
echo "生成的数据文件:"
ls -la data/feat_*.parquet 2>/dev/null || echo "未找到数据文件"

echo ""
echo "数据文件大小:"
du -h data/feat_*.parquet 2>/dev/null || echo "未找到数据文件"

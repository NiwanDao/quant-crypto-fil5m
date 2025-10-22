# 数据准备阶段说明

本项目已为3个不同时间段准备了独立的数据准备脚本，用于获取和处理FIL/USDT的15分钟K线数据。

## 时间段划分

1. **2024-8 到 2025-3** - 历史数据训练集
2. **2025-3 到 2025-6** - 中期验证集  
3. **2025-6 至今** - 最新数据测试集

## 文件结构

```
data/
├── prepare_2024_8_to_2025_3.py      # 2024-8到2025-3数据准备脚本
├── prepare_2025_3_to_2025_6.py      # 2025-3到2025-6数据准备脚本
├── prepare_2025_6_to_now.py         # 2025-6至今数据准备脚本
├── prepare_all_periods.py           # 主数据准备脚本
├── validate_data_quality.py         # 数据质量验证脚本
└── feat_*.parquet                   # 生成的数据文件

scripts/
└── prepare_data_periods.sh          # 便捷执行脚本
```

## 使用方法

### 方法1: 使用便捷脚本（推荐）
```bash
./scripts/prepare_data_periods.sh
```

### 方法2: 单独运行各时间段
```bash
# 准备2024-8到2025-3数据
python data/prepare_2024_8_to_2025_3.py

# 准备2025-3到2025-6数据  
python data/prepare_2025_3_to_2025_6.py

# 准备2025-6至今数据
python data/prepare_2025_6_to_now.py
```

### 方法3: 使用主脚本
```bash
python data/prepare_all_periods.py
```

## 数据验证

运行数据质量验证：
```bash
python data/validate_data_quality.py
```

## 生成的数据文件

- `data/feat_2024_8_to_2025_3.parquet` - 2024-8到2025-3期间数据
- `data/feat_2025_3_to_2025_6.parquet` - 2025-3到2025-6期间数据  
- `data/feat_2025_6_to_now.parquet` - 2025-6至今数据

## 数据特征

每个数据文件包含：
- **基础OHLCV数据**: open, high, low, close, volume
- **技术指标特征**: 通过`utils.features.build_features()`生成
- **标签数据**: 通过`utils.features.build_labels()`生成
- **时间索引**: UTC时区的15分钟K线时间戳

## 注意事项

1. **网络连接**: 需要稳定的网络连接来获取交易所数据
2. **API限制**: 脚本已包含速率限制，避免触发API限制
3. **数据完整性**: 运行后请使用验证脚本检查数据质量
4. **存储空间**: 确保有足够的磁盘空间存储数据文件

## 故障排除

如果遇到问题：
1. 检查网络连接
2. 确认配置文件`conf/config.yml`设置正确
3. 查看错误日志信息
4. 运行数据验证脚本检查数据质量

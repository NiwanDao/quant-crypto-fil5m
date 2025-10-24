# 多方考虑的综合回测系统

基于 `feat_2025_3_to_2025_6.parquet` 数据特征的多方考虑回测代码，包含风险管理、市场状态识别、多时间框架分析、交易成本考虑等综合功能。

## 🚀 快速开始

### 1. 简化版回测（推荐）

```bash
cd /home/shiyi/quant-crypto-fil5m
python run_simple_backtest.py
```

### 2. 完整版回测

```bash
cd /home/shiyi/quant-crypto-fil5m
python run_comprehensive_backtest.py
```

## 📊 系统特性

### 核心功能

1. **多方考虑的回测框架**
   - 市场状态识别（趋势/震荡/高波动/低波动）
   - 动态阈值调整
   - 风险调整信号生成

2. **高级风险管理**
   - 动态止损计算
   - 凯利公式仓位管理
   - VaR和CVaR风险指标
   - 流动性风险评估

3. **多时间框架分析**
   - 15分钟、1小时、4小时、1天多周期分析
   - 时间框架对齐度计算
   - 多周期信号融合

4. **市场状态识别**
   - 基于机器学习的市场状态分类
   - 状态转移矩阵分析
   - 适应性策略切换

5. **交易成本考虑**
   - 动态交易费用计算
   - 滑点分析
   - 流动性成本评估
   - 市场冲击分析

6. **综合性能评估**
   - 多维度性能指标
   - 风险调整收益分析
   - 稳定性指标
   - 可视化分析

## 📁 文件结构

```
backtest/
├── comprehensive_backtest.py      # 主回测框架
├── risk_manager.py               # 风险管理模块
├── multi_timeframe_analyzer.py  # 多时间框架分析
├── market_regime_detector.py    # 市场状态识别
├── cost_liquidity_analyzer.py   # 成本流动性分析
├── performance_evaluator.py     # 性能评估
└── run_comprehensive_backtest.py # 完整版运行脚本

run_simple_backtest.py           # 简化版运行脚本
```

## 🎯 主要模块说明

### 1. ComprehensiveBacktester
- **功能**: 主回测框架
- **特性**: 
  - 数据加载和预处理
  - 模型加载和预测
  - 信号生成和回测执行
  - 结果保存和可视化

### 2. AdvancedRiskManager
- **功能**: 高级风险管理
- **特性**:
  - 动态止损计算
  - 凯利公式仓位管理
  - 风险指标计算
  - 风险调整信号

### 3. MultiTimeframeAnalyzer
- **功能**: 多时间框架分析
- **特性**:
  - 多周期数据重采样
  - 趋势强度计算
  - 支撑阻力识别
  - 多周期信号融合

### 4. MarketRegimeDetector
- **功能**: 市场状态识别
- **特性**:
  - 市场特征计算
  - 状态分类（K-means聚类）
  - 状态转移分析
  - 适应性策略

### 5. CostLiquidityAnalyzer
- **功能**: 交易成本和流动性分析
- **特性**:
  - 动态交易费用
  - 滑点计算
  - 流动性指标
  - 成本归因分析

### 6. PerformanceEvaluator
- **功能**: 综合性能评估
- **特性**:
  - 多维度性能指标
  - 风险调整收益
  - 稳定性分析
  - 可视化报告

## 📈 输出结果

### 数据文件
- `simple_stats.csv`: 详细统计指标
- `simple_trades.csv`: 交易记录
- `simple_signals.csv`: 信号数据
- `simple_performance_report.json`: 性能报告

### 可视化图表
- `simple_equity_curve.png`: 权益曲线
- `simple_comprehensive_analysis.png`: 综合分析图

## 🔧 配置说明

系统使用 `conf/config.yml` 配置文件，主要参数：

```yaml
backtest:
  fixed_cash_per_trade: 1000      # 每笔交易固定金额
  initial_cash: 10000             # 初始资金
  size_mode: fixed_cash           # 仓位模式

model:
  proba_threshold: 0.197657989565356  # 买入阈值
  sell_threshold: 0.11137939722286525 # 卖出阈值
  use_ensemble: true              # 使用集成模型

fees_slippage:
  taker_fee_bps: 6               # 交易费用（基点）
  base_slippage_bps: 1           # 基础滑点（基点）

risk:
  max_risk_per_trade: 0.005      # 每笔交易最大风险
  atr_stop_mult: 1.5             # ATR止损倍数
  dynamic_risk: true             # 动态风险管理
```

## 📊 性能指标

### 收益指标
- 总收益率
- 年化收益率
- 累积收益

### 风险指标
- 年化波动率
- 最大回撤
- VaR (95%, 99%)
- CVaR (95%, 99%)

### 风险调整收益
- 夏普比率
- 索提诺比率
- 卡玛比率
- 信息比率

### 交易指标
- 胜率
- 盈亏比
- 平均盈利/亏损
- 最大连续盈利/亏损

### 稳定性指标
- 夏普比率稳定性
- 收益稳定性
- 回撤稳定性

## 🎨 可视化分析

### 1. 价格走势与交易信号
- 价格曲线
- 买入/卖出信号标记
- 市场状态颜色编码

### 2. 投资组合价值变化
- 权益曲线
- 初始资金基准线
- 回撤分析

### 3. 市场状态分析
- 状态分布饼图
- 波动率分布
- 趋势强度分析
- 状态转移矩阵

### 4. 风险收益分析
- 收益分布直方图
- 风险收益散点图
- 滚动夏普比率
- 分位数分析

## ⚠️ 注意事项

1. **数据要求**: 确保 `data/feat_2025_3_to_2025_6.parquet` 文件存在
2. **模型要求**: 确保训练好的模型文件存在
3. **依赖库**: 确保安装了所有必要的Python库
4. **内存使用**: 大数据集可能需要较多内存

## 🚀 运行建议

1. **首次运行**: 建议使用简化版 `run_simple_backtest.py`
2. **完整分析**: 数据量较大时使用完整版 `run_comprehensive_backtest.py`
3. **参数调整**: 根据实际需求调整配置文件中的参数
4. **结果分析**: 重点关注风险调整收益指标和稳定性指标

## 📞 技术支持

如有问题，请检查：
1. 数据文件是否存在
2. 模型文件是否完整
3. 依赖库是否安装
4. 配置文件是否正确

## 🔄 更新日志

- **v1.0**: 初始版本，包含基础回测功能
- **v1.1**: 添加风险管理模块
- **v1.2**: 添加多时间框架分析
- **v1.3**: 添加市场状态识别
- **v1.4**: 添加成本流动性分析
- **v1.5**: 添加综合性能评估
- **v1.6**: 优化可视化效果

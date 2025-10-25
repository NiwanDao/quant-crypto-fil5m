# 实盘交易系统

基于机器学习的FIL加密货币实盘交易系统，支持模拟交易和实盘交易模式。

## 🚀 功能特性

### 核心功能
- **智能信号生成**: 基于机器学习模型的实时交易信号
- **多交易所支持**: 支持Binance、OKX、Bybit、Gate.io等主流交易所
- **风险管理**: 多层次风险管理策略，包括止损、止盈、仓位控制
- **实时监控**: 系统健康度、性能指标、风险指标实时监控
- **智能告警**: 邮件、微信、钉钉多渠道告警通知

### 交易模式
- **模拟交易**: 使用虚拟资金进行策略验证
- **实盘交易**: 连接真实交易所进行实际交易

### 风险管理
- **动态仓位管理**: 基于波动率、凯利公式等多种仓位计算方法
- **止损止盈**: 基于ATR的动态止损止盈策略
- **风险控制**: 最大回撤、波动率、单笔风险等多维度风险控制
- **跟踪止损**: 自动调整止损位，锁定利润

## 📁 项目结构

```
live/
├── live_trading.py          # 核心交易系统
├── exchange_interface.py    # 交易所接口
├── data_fetcher.py          # 数据获取和特征计算
├── risk_manager.py          # 风险管理
├── monitoring.py            # 监控和日志系统
├── start_live_trading.py    # 启动脚本
├── live_config.yml          # 配置文件
├── requirements_live.txt    # 依赖包
└── README_LIVE.md          # 说明文档
```

## 🛠️ 安装配置

### 1. 安装依赖

```bash
# 安装Python依赖
pip install -r live/requirements_live.txt

# 安装TA-Lib（技术指标库）
# Ubuntu/Debian
sudo apt-get install libta-lib-dev
pip install TA-Lib

# macOS
brew install ta-lib
pip install TA-Lib

# Windows
# 下载预编译的whl文件
pip install TA_Lib-0.4.25-cp39-cp39-win_amd64.whl
```

### 2. 配置环境变量

```bash
# 交易所API密钥（实盘交易需要）
export BINANCE_API_KEY="your_api_key"
export BINANCE_SECRET_KEY="your_secret_key"

# 邮件通知（可选）
export EMAIL_USERNAME="your_email@gmail.com"
export EMAIL_PASSWORD="your_app_password"
export EMAIL_FROM="your_email@gmail.com"
export EMAIL_TO="recipient@gmail.com"

# 微信通知（可选）
export WECHAT_WEBHOOK_URL="https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=your_key"

# 钉钉通知（可选）
export DINGTALK_WEBHOOK_URL="https://oapi.dingtalk.com/robot/send?access_token=your_token"
```

### 3. 配置文件设置

编辑 `live/live_config.yml` 文件，根据你的需求调整配置：

```yaml
# 交易模式
trading:
  mode: "paper"  # paper: 模拟交易, live: 实盘交易
  symbol: "FIL/USDT"
  timeframe: "15m"

# 风险管理
risk_management:
  max_portfolio_risk: 0.02  # 最大投资组合风险2%
  max_position_risk: 0.005  # 最大单笔风险0.5%
  max_drawdown_limit: 0.15  # 最大回撤限制15%
```

## 🚀 使用方法

### 1. 模拟交易（推荐先使用）

```bash
# 启动模拟交易
python live/start_live_trading.py --mode paper

# 干运行模式（运行一个周期后退出）
python live/start_live_trading.py --mode paper --dry-run

# 详细输出
python live/start_live_trading.py --mode paper --verbose
```

### 2. 实盘交易

```bash
# 启动实盘交易（需要设置API密钥）
python live/start_live_trading.py --mode live

# 指定交易对
python live/start_live_trading.py --mode live --symbol "BTC/USDT"
```

### 3. 自定义配置

```bash
# 使用自定义配置文件
python live/start_live_trading.py --config my_config.yml
```

## 📊 监控和日志

### 日志文件
- `logs/live_trading_YYYYMMDD.log`: 主交易日志
- `logs/monitoring/monitor_YYYYMMDD.log`: 监控日志
- `logs/monitoring/performance_history.csv`: 性能历史
- `logs/monitoring/alerts_history.json`: 告警历史

### 监控指标
- 投资组合价值变化
- 回撤分析
- 风险评分
- 交易频率
- 系统资源使用情况

### 告警通知
系统支持多种告警方式：
- 邮件通知
- 企业微信机器人
- 钉钉机器人

## ⚠️ 风险提示

### 重要警告
1. **实盘交易风险**: 实盘交易涉及真实资金，存在亏损风险
2. **策略验证**: 建议先在模拟环境中充分验证策略
3. **资金管理**: 不要投入超过承受能力的资金
4. **监控重要**: 定期检查系统运行状态和交易结果

### 安全建议
1. **API密钥安全**: 妥善保管交易所API密钥，设置IP白名单
2. **权限控制**: 使用只读权限的API密钥进行测试
3. **资金限制**: 设置合理的交易限额
4. **定期备份**: 定期备份配置和交易数据

## 🔧 故障排除

### 常见问题

1. **TA-Lib安装失败**
   ```bash
   # Ubuntu/Debian
   sudo apt-get update
   sudo apt-get install libta-lib-dev
   pip install TA-Lib
   ```

2. **API连接失败**
   - 检查网络连接
   - 验证API密钥是否正确
   - 确认交易所服务状态

3. **模型加载失败**
   - 确保模型文件存在
   - 检查模型文件路径
   - 验证模型文件完整性

4. **内存不足**
   - 减少历史数据量
   - 调整缓存大小
   - 增加系统内存

### 日志分析

```bash
# 查看最新日志
tail -f logs/live_trading_$(date +%Y%m%d).log

# 查看错误日志
grep "ERROR" logs/live_trading_*.log

# 查看交易记录
grep "交易" logs/live_trading_*.log
```

## 📈 性能优化

### 系统优化
1. **内存管理**: 定期清理历史数据
2. **网络优化**: 使用稳定的网络连接
3. **资源监控**: 监控CPU、内存、磁盘使用情况
4. **数据缓存**: 合理设置数据缓存大小

### 策略优化
1. **参数调优**: 根据市场情况调整策略参数
2. **风险管理**: 根据回测结果调整风险参数
3. **信号过滤**: 优化信号生成逻辑
4. **仓位管理**: 动态调整仓位大小

## 🤝 贡献指南

欢迎提交Issue和Pull Request来改进系统：

1. Fork项目
2. 创建特性分支
3. 提交更改
4. 推送到分支
5. 创建Pull Request

## 📄 许可证

本项目采用MIT许可证，详见LICENSE文件。

## 📞 支持

如有问题，请通过以下方式联系：

- 提交Issue
- 发送邮件
- 微信群讨论

---

**免责声明**: 本系统仅供学习和研究使用，不构成投资建议。使用本系统进行实盘交易的风险由用户自行承担。

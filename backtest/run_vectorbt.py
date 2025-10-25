import yaml
import pandas as pd
import numpy as np
import vectorbt as vbt
import lightgbm as lgb
import joblib
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

CONF_PATH = 'conf/config.yml'

def load_conf():
    with open(CONF_PATH, 'r') as f:
        return yaml.safe_load(f)

def calculate_signal_strength(p_up, p_down, features_df):
    """计算信号强度"""
    strengths = []
    for i in range(len(features_df)):
        row = features_df.iloc[i]
        prob_diff = abs(p_up[i] - p_down[i])
        
        # RSI一致性评分
        rsi = row.get('rsi', 50)
        rsi_score = 0
        if p_up[i] > 0.6:
            if rsi < 70:
                rsi_score = 0.3
            elif rsi > 80:
                rsi_score = -0.2
        elif p_down[i] > 0.6:
            if rsi > 30:
                rsi_score = 0.3
            elif rsi < 20:
                rsi_score = -0.2
        
        # MACD一致性评分
        macd = row.get('macd', 0)
        macd_sig = row.get('macd_sig', 0)
        macd_score = 0
        if p_up[i] > 0.6 and macd > macd_sig:
            macd_score = 0.3
        elif p_down[i] > 0.6 and macd < macd_sig:
            macd_score = 0.3
        
        strength = min(1.0, max(0.0, prob_diff + rsi_score + macd_score))
        strengths.append(strength)
    
    return np.array(strengths)

def analyze_trades(pf):
    """分析交易详情"""
    trades = pf.trades.records_readable
    if len(trades) == 0:
        return {}
    
    # 基本统计
    total_trades = len(trades)
    winning_trades = len(trades[trades['PnL'] > 0])
    losing_trades = len(trades[trades['PnL'] < 0])
    win_rate = winning_trades / total_trades if total_trades > 0 else 0
    
    # 盈亏统计
    total_pnl = trades['PnL'].sum()
    avg_win = trades[trades['PnL'] > 0]['PnL'].mean() if winning_trades > 0 else 0
    avg_loss = trades[trades['PnL'] < 0]['PnL'].mean() if losing_trades > 0 else 0
    profit_factor = abs(avg_win * winning_trades / (avg_loss * losing_trades)) if losing_trades > 0 and avg_loss != 0 else float('inf')
    
    # 最大连续盈亏
    pnl_series = trades['PnL'].values
    max_consecutive_wins = 0
    max_consecutive_losses = 0
    current_wins = 0
    current_losses = 0
    
    for pnl in pnl_series:
        if pnl > 0:
            current_wins += 1
            current_losses = 0
            max_consecutive_wins = max(max_consecutive_wins, current_wins)
        else:
            current_losses += 1
            current_wins = 0
            max_consecutive_losses = max(max_consecutive_losses, current_losses)
    
    return {
        'total_trades': total_trades,
        'winning_trades': winning_trades,
        'losing_trades': losing_trades,
        'win_rate': win_rate,
        'total_pnl': total_pnl,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor,
        'max_consecutive_wins': max_consecutive_wins,
        'max_consecutive_losses': max_consecutive_losses,
        'best_trade': trades['PnL'].max(),
        'worst_trade': trades['PnL'].min()
    }

def create_comprehensive_plots(pf, df, entries, exits, signal_strength, initial_cash):
    """创建综合图表"""
    fig, axes = plt.subplots(4, 1, figsize=(15, 20))
    
    # 1. 价格和信号
    ax1 = axes[0]
    ax1.plot(df.index, df['close'], label='Close Price', alpha=0.7)
    
    # 标记买入信号
    buy_signals = df[entries]
    ax1.scatter(buy_signals.index, buy_signals['close'], 
               color='green', marker='^', s=100, label='Buy Signal', alpha=0.8)
    
    # 标记卖出信号
    sell_signals = df[exits]
    ax1.scatter(sell_signals.index, sell_signals['close'], 
               color='red', marker='v', s=100, label='Sell Signal', alpha=0.8)
    
    ax1.set_title('Price and Trading Signals')
    ax1.set_ylabel('Price')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 投资组合价值
    ax2 = axes[1]
    portfolio_value = pf.value()
    ax2.plot(portfolio_value.index, portfolio_value, label='Portfolio Value', color='blue')
    ax2.axhline(y=initial_cash, color='red', linestyle='--', alpha=0.7, label='Initial Capital')
    ax2.set_title('Portfolio Value Over Time')
    ax2.set_ylabel('Portfolio Value')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 回撤
    ax3 = axes[2]
    drawdown = pf.drawdowns.records_readable
    if len(drawdown) > 0:
        # 检查列名并绘制回撤
        if 'Start' in drawdown.columns and 'Drawdown' in drawdown.columns:
            ax3.fill_between(drawdown['Start'], 0, drawdown['Drawdown'], 
                            alpha=0.3, color='red', label='Drawdown')
        else:
            # 如果没有回撤数据，绘制空图
            ax3.text(0.5, 0.5, 'No Drawdown Data', ha='center', va='center', transform=ax3.transAxes)
    else:
        ax3.text(0.5, 0.5, 'No Drawdown Data', ha='center', va='center', transform=ax3.transAxes)
    
    ax3.set_title('Drawdown Analysis')
    ax3.set_ylabel('Drawdown %')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 信号强度分布
    ax4 = axes[3]
    ax4.hist(signal_strength, bins=30, alpha=0.7, color='purple', edgecolor='black')
    ax4.axvline(x=0.5, color='red', linestyle='--', label='Min Signal Strength')
    ax4.set_title('Signal Strength Distribution')
    ax4.set_xlabel('Signal Strength')
    ax4.set_ylabel('Frequency')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('backtest/comprehensive_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()

def main():
    print("🚀 开始综合回测分析...")
    
    # 加载配置和数据
    conf = load_conf()
    df = pd.read_parquet('data/feat.parquet')
    features = [c for c in df.columns if c not in ['y']]
    
    print(f"📊 数据概览: {len(df)} 条记录, {len(features)} 个特征")
    
    # 加载模型并预测
    mdl = joblib.load('models/lgb_trend.pkl')
    p_up = mdl.predict_proba(df[features])[:,1]
    p_down = 1.0 - p_up
    
    # 计算信号强度
    signal_strength = calculate_signal_strength(p_up, p_down, df[features])
    
    # 改进的信号生成逻辑
    buy_threshold = conf['model']['proba_threshold']
    sell_threshold = conf['model'].get('sell_threshold', 0.40)
    min_strength = conf['model'].get('min_signal_strength', 0.3)
    
    # 生成交易信号
    entries = (p_up > buy_threshold) & (signal_strength > min_strength)
    sell_signals = (p_down > sell_threshold) & (signal_strength > min_strength)
    
    # 转换为pandas Series
    entries_series = pd.Series(entries, index=df.index)
    exits_series = pd.Series(sell_signals, index=df.index)
    
    # 确保退出信号在买入信号之后
    exits = exits_series & entries_series.shift(1).fillna(False)
    
    print(f"📈 买入信号: {entries.sum()} 个")
    print(f"📉 卖出信号: {exits.sum()} 个")
    print(f"💪 平均信号强度: {signal_strength.mean():.3f}")
    
    # 设置交易参数
    close = df['close']
    fees = conf['fees_slippage']['taker_fee_bps']/1e4
    slip = conf['fees_slippage']['base_slippage_bps']/1e4
    
    # 动态仓位大小（基于信号强度）
    base_size = conf['backtest']['fixed_cash_per_trade'] / close
    size = base_size * signal_strength  # 信号强度越高，仓位越大
    size = size.clip(lower=0)
    
    # 运行回测
    print("🔄 运行回测...")
    pf = vbt.Portfolio.from_signals(
        close=close,
        entries=entries,
        exits=exits,
        fees=fees,
        slippage=slip,
        init_cash=conf['backtest']['initial_cash'],
        size=size
    )
    
    # 基本统计
    print("\n" + "="*50)
    print("📊 回测结果")
    print("="*50)
    stats = pf.stats()
    print(stats)
    
    # 详细交易分析
    print("\n" + "="*50)
    print("📈 交易详情分析")
    print("="*50)
    trade_analysis = analyze_trades(pf)
    for key, value in trade_analysis.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    
    # 保存结果
    print("\n💾 保存结果...")
    
    # 保存统计报告
    stats_df = stats.to_frame('value')
    stats_df.to_csv('backtest/stats.csv')
    
    # 保存交易记录
    if len(pf.trades.records_readable) > 0:
        trades_df = pf.trades.records_readable
        trades_df.to_csv('backtest/trades.csv', index=False)
    
    # 保存信号数据
    signal_data = pd.DataFrame({
        'timestamp': df.index,
        'close': df['close'],
        'p_up': p_up,
        'p_down': p_down,
        'signal_strength': signal_strength,
        'buy_signal': entries,
        'sell_signal': exits
    })
    signal_data.to_csv('backtest/signals.csv', index=False)
    
    # 创建图表
    print("📊 生成图表...")
    create_comprehensive_plots(pf, df, entries, exits, signal_strength, conf['backtest']['initial_cash'])
    
    # 创建简单的权益曲线图
    plt.figure(figsize=(12, 6))
    portfolio_value = pf.value()
    plt.plot(portfolio_value.index, portfolio_value, label='Portfolio Value', linewidth=2)
    plt.axhline(y=conf['backtest']['initial_cash'], color='red', linestyle='--', alpha=0.7, label='Initial Capital')
    plt.title('Portfolio Value Over Time')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('backtest/equity_curve.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("\n✅ 回测完成！")
    print("📁 生成的文件:")
    print("  - backtest/stats.csv: 基本统计")
    print("  - backtest/trades.csv: 交易记录")
    print("  - backtest/signals.csv: 信号数据")
    print("  - backtest/equity_curve.png: 权益曲线")
    print("  - backtest/comprehensive_analysis.png: 综合分析图")

if __name__ == '__main__':
    main()

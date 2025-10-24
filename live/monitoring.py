"""
监控和日志系统
实时监控交易状态、性能指标和系统健康度
"""

import pandas as pd
import numpy as np
import logging
import json
import os
import time
import threading
import smtplib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, asdict
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests
import matplotlib.pyplot as plt
import seaborn as sns
from enum import Enum

class AlertLevel(Enum):
    """告警级别"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class Alert:
    """告警信息"""
    timestamp: datetime
    level: AlertLevel
    category: str
    message: str
    details: Dict
    resolved: bool = False

@dataclass
class PerformanceMetrics:
    """性能指标"""
    timestamp: datetime
    portfolio_value: float
    total_return: float
    daily_return: float
    max_drawdown: float
    current_drawdown: float
    sharpe_ratio: float
    win_rate: float
    profit_factor: float
    total_trades: int
    active_positions: int
    risk_score: float

class TradingMonitor:
    """交易监控器"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger('TradingMonitor')
        
        # 监控数据
        self.performance_history = []
        self.alerts = []
        self.system_metrics = {}
        
        # 监控配置
        self.alert_thresholds = config.get('alert_thresholds', {})
        self.notification_config = config.get('notifications', {})
        
        # 监控状态
        self.is_monitoring = False
        self.monitor_thread = None
        
        # 设置日志
        self._setup_logging()
        
    def _setup_logging(self):
        """设置监控日志"""
        os.makedirs('logs/monitoring', exist_ok=True)
        
        # 监控专用日志
        monitor_logger = logging.getLogger('TradingMonitor')
        monitor_handler = logging.FileHandler(
            f'logs/monitoring/monitor_{datetime.now().strftime("%Y%m%d")}.log'
        )
        monitor_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        monitor_handler.setFormatter(monitor_formatter)
        monitor_logger.addHandler(monitor_handler)
        monitor_logger.setLevel(logging.INFO)
    
    def start_monitoring(self, update_interval: int = 60):
        """开始监控"""
        try:
            self.is_monitoring = True
            self.monitor_thread = threading.Thread(
                target=self._monitoring_loop,
                args=(update_interval,),
                daemon=True
            )
            self.monitor_thread.start()
            
            self.logger.info("✅ 监控系统已启动")
            
        except Exception as e:
            self.logger.error(f"❌ 监控启动失败: {str(e)}")
    
    def stop_monitoring(self):
        """停止监控"""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        
        self.logger.info("⏹️ 监控系统已停止")
    
    def _monitoring_loop(self, update_interval: int):
        """监控循环"""
        while self.is_monitoring:
            try:
                # 检查系统健康度
                self._check_system_health()
                
                # 检查性能指标
                self._check_performance_metrics()
                
                # 检查风险指标
                self._check_risk_metrics()
                
                # 检查网络连接
                self._check_network_connectivity()
                
                # 清理过期数据
                self._cleanup_old_data()
                
                time.sleep(update_interval)
                
            except Exception as e:
                self.logger.error(f"❌ 监控循环错误: {str(e)}")
                time.sleep(60)  # 错误后等待1分钟
    
    def update_performance_metrics(self, metrics: PerformanceMetrics):
        """更新性能指标"""
        try:
            self.performance_history.append(metrics)
            
            # 保持最近1000条记录
            if len(self.performance_history) > 1000:
                self.performance_history = self.performance_history[-1000:]
            
            # 检查性能告警
            self._check_performance_alerts(metrics)
            
        except Exception as e:
            self.logger.error(f"❌ 性能指标更新失败: {str(e)}")
    
    def _check_system_health(self):
        """检查系统健康度"""
        try:
            # 检查内存使用
            import psutil
            memory_usage = psutil.virtual_memory().percent
            cpu_usage = psutil.cpu_percent()
            
            # 检查磁盘空间
            disk_usage = psutil.disk_usage('/').percent
            
            # 更新系统指标
            self.system_metrics.update({
                'memory_usage': memory_usage,
                'cpu_usage': cpu_usage,
                'disk_usage': disk_usage,
                'timestamp': datetime.now()
            })
            
            # 检查系统告警
            if memory_usage > 90:
                self._create_alert(
                    AlertLevel.WARNING,
                    'system',
                    f'内存使用率过高: {memory_usage:.1f}%',
                    {'memory_usage': memory_usage}
                )
            
            if cpu_usage > 90:
                self._create_alert(
                    AlertLevel.WARNING,
                    'system',
                    f'CPU使用率过高: {cpu_usage:.1f}%',
                    {'cpu_usage': cpu_usage}
                )
            
            if disk_usage > 90:
                self._create_alert(
                    AlertLevel.WARNING,
                    'system',
                    f'磁盘使用率过高: {disk_usage:.1f}%',
                    {'disk_usage': disk_usage}
                )
                
        except Exception as e:
            self.logger.error(f"❌ 系统健康检查失败: {str(e)}")
    
    def _check_performance_metrics(self):
        """检查性能指标"""
        try:
            if not self.performance_history:
                return
            
            latest_metrics = self.performance_history[-1]
            
            # 检查回撤告警
            max_drawdown_limit = self.alert_thresholds.get('max_drawdown', 0.1)
            if latest_metrics.current_drawdown < -max_drawdown_limit:
                self._create_alert(
                    AlertLevel.WARNING,
                    'performance',
                    f'回撤超限: {latest_metrics.current_drawdown:.2%}',
                    {'current_drawdown': latest_metrics.current_drawdown}
                )
            
            # 检查收益率告警
            daily_return_limit = self.alert_thresholds.get('daily_return', -0.05)
            if latest_metrics.daily_return < daily_return_limit:
                self._create_alert(
                    AlertLevel.ERROR,
                    'performance',
                    f'日收益率过低: {latest_metrics.daily_return:.2%}',
                    {'daily_return': latest_metrics.daily_return}
                )
            
            # 检查夏普比率告警
            sharpe_limit = self.alert_thresholds.get('sharpe_ratio', 0.5)
            if latest_metrics.sharpe_ratio < sharpe_limit:
                self._create_alert(
                    AlertLevel.WARNING,
                    'performance',
                    f'夏普比率过低: {latest_metrics.sharpe_ratio:.2f}',
                    {'sharpe_ratio': latest_metrics.sharpe_ratio}
                )
                
        except Exception as e:
            self.logger.error(f"❌ 性能指标检查失败: {str(e)}")
    
    def _check_risk_metrics(self):
        """检查风险指标"""
        try:
            if not self.performance_history:
                return
            
            latest_metrics = self.performance_history[-1]
            
            # 检查风险评分告警
            risk_score_limit = self.alert_thresholds.get('risk_score', 0.8)
            if latest_metrics.risk_score > risk_score_limit:
                self._create_alert(
                    AlertLevel.CRITICAL,
                    'risk',
                    f'风险评分过高: {latest_metrics.risk_score:.2f}',
                    {'risk_score': latest_metrics.risk_score}
                )
            
            # 检查持仓数量告警
            position_limit = self.alert_thresholds.get('max_positions', 10)
            if latest_metrics.active_positions > position_limit:
                self._create_alert(
                    AlertLevel.WARNING,
                    'risk',
                    f'持仓数量过多: {latest_metrics.active_positions}',
                    {'active_positions': latest_metrics.active_positions}
                )
                
        except Exception as e:
            self.logger.error(f"❌ 风险指标检查失败: {str(e)}")
    
    def _check_network_connectivity(self):
        """检查网络连接"""
        try:
            # 检查网络连接
            test_urls = [
                'https://api.binance.com/api/v3/ping',
                'https://api.bybit.com/v5/market/time'
            ]
            
            for url in test_urls:
                success = False
                for attempt in range(3):  # 重试3次
                    try:
                        response = requests.get(url, timeout=10)
                        if response.status_code == 200:
                            success = True
                            break
                        else:
                            self.logger.warning(f"⚠️ {url} 返回状态码: {response.status_code}")
                    except requests.RequestException as e:
                        self.logger.warning(f"⚠️ {url} 连接失败 (尝试 {attempt + 1}/3): {str(e)}")
                        if attempt < 2:  # 不是最后一次尝试
                            time.sleep(2)  # 等待2秒后重试
                
                if not success:
                    self._create_alert(
                        AlertLevel.ERROR,
                        'network',
                        f'网络连接失败: {url}',
                        {'url': url, 'attempts': 3}
                    )
                    
        except Exception as e:
            self.logger.error(f"❌ 网络连接检查失败: {str(e)}")
    
    def _check_performance_alerts(self, metrics: PerformanceMetrics):
        """检查性能告警"""
        try:
            # 检查连续亏损
            if len(self.performance_history) >= 5:
                recent_returns = [m.daily_return for m in self.performance_history[-5:]]
                if all(r < 0 for r in recent_returns):
                    self._create_alert(
                        AlertLevel.WARNING,
                        'performance',
                        '连续5日亏损',
                        {'recent_returns': recent_returns}
                    )
            
            # 检查交易频率
            if metrics.total_trades > 100:  # 假设日交易次数过多
                self._create_alert(
                    AlertLevel.INFO,
                    'performance',
                    f'交易频率较高: {metrics.total_trades}',
                    {'total_trades': metrics.total_trades}
                )
                
        except Exception as e:
            self.logger.error(f"❌ 性能告警检查失败: {str(e)}")
    
    def _create_alert(self, level: AlertLevel, category: str, message: str, details: Dict):
        """创建告警"""
        try:
            alert = Alert(
                timestamp=datetime.now(),
                level=level,
                category=category,
                message=message,
                details=details
            )
            
            self.alerts.append(alert)
            
            # 记录日志
            self.logger.warning(f"🚨 {level.value.upper()}: {message}")
            
            # 发送通知
            self._send_notification(alert)
            
            # 保持最近1000条告警
            if len(self.alerts) > 1000:
                self.alerts = self.alerts[-1000:]
                
        except Exception as e:
            self.logger.error(f"❌ 告警创建失败: {str(e)}")
    
    def _send_notification(self, alert: Alert):
        """发送通知"""
        try:
            # 邮件通知
            if self.notification_config.get('email', {}).get('enabled', False):
                self._send_email_notification(alert)
            
            # 微信通知
            if self.notification_config.get('wechat', {}).get('enabled', False):
                self._send_wechat_notification(alert)
            
            # 钉钉通知
            if self.notification_config.get('dingtalk', {}).get('enabled', False):
                self._send_dingtalk_notification(alert)
                
        except Exception as e:
            self.logger.error(f"❌ 通知发送失败: {str(e)}")
    
    def _send_email_notification(self, alert: Alert):
        """发送邮件通知"""
        try:
            email_config = self.notification_config.get('email', {})
            
            if not email_config.get('enabled', False):
                return
            
            # 创建邮件
            msg = MIMEMultipart()
            msg['From'] = email_config['from_email']
            msg['To'] = email_config['to_email']
            msg['Subject'] = f"交易系统告警 - {alert.level.value.upper()}"
            
            # 邮件内容
            body = f"""
告警时间: {alert.timestamp}
告警级别: {alert.level.value.upper()}
告警类别: {alert.category}
告警信息: {alert.message}
详细信息: {json.dumps(alert.details, indent=2, ensure_ascii=False)}
            """
            
            msg.attach(MIMEText(body, 'plain', 'utf-8'))
            
            # 发送邮件
            server = smtplib.SMTP(email_config['smtp_server'], email_config['smtp_port'])
            server.starttls()
            server.login(email_config['username'], email_config['password'])
            server.send_message(msg)
            server.quit()
            
            self.logger.info("✅ 邮件通知已发送")
            
        except Exception as e:
            self.logger.error(f"❌ 邮件通知发送失败: {str(e)}")
    
    def _send_wechat_notification(self, alert: Alert):
        """发送微信通知"""
        try:
            wechat_config = self.notification_config.get('wechat', {})
            
            if not wechat_config.get('enabled', False):
                return
            
            # 使用企业微信机器人
            webhook_url = wechat_config.get('webhook_url')
            if not webhook_url:
                return
            
            message = {
                "msgtype": "text",
                "text": {
                    "content": f"交易系统告警\n级别: {alert.level.value.upper()}\n信息: {alert.message}\n时间: {alert.timestamp}"
                }
            }
            
            response = requests.post(webhook_url, json=message)
            if response.status_code == 200:
                self.logger.info("✅ 微信通知已发送")
            else:
                self.logger.error(f"❌ 微信通知发送失败: {response.status_code}")
                
        except Exception as e:
            self.logger.error(f"❌ 微信通知发送失败: {str(e)}")
    
    def _send_dingtalk_notification(self, alert: Alert):
        """发送钉钉通知"""
        try:
            dingtalk_config = self.notification_config.get('dingtalk', {})
            
            if not dingtalk_config.get('enabled', False):
                return
            
            webhook_url = dingtalk_config.get('webhook_url')
            if not webhook_url:
                return
            
            message = {
                "msgtype": "text",
                "text": {
                    "content": f"交易系统告警\n级别: {alert.level.value.upper()}\n信息: {alert.message}\n时间: {alert.timestamp}"
                }
            }
            
            response = requests.post(webhook_url, json=message)
            if response.status_code == 200:
                self.logger.info("✅ 钉钉通知已发送")
            else:
                self.logger.error(f"❌ 钉钉通知发送失败: {response.status_code}")
                
        except Exception as e:
            self.logger.error(f"❌ 钉钉通知发送失败: {str(e)}")
    
    def _cleanup_old_data(self):
        """清理过期数据"""
        try:
            # 清理30天前的告警
            cutoff_date = datetime.now() - timedelta(days=30)
            self.alerts = [alert for alert in self.alerts if alert.timestamp > cutoff_date]
            
            # 清理90天前的性能历史
            cutoff_date = datetime.now() - timedelta(days=90)
            self.performance_history = [
                metrics for metrics in self.performance_history 
                if metrics.timestamp > cutoff_date
            ]
            
        except Exception as e:
            self.logger.error(f"❌ 数据清理失败: {str(e)}")
    
    def get_performance_summary(self) -> Dict:
        """获取性能摘要"""
        try:
            if not self.performance_history:
                return {}
            
            latest = self.performance_history[-1]
            
            # 计算统计信息
            returns = [m.daily_return for m in self.performance_history if m.daily_return is not None]
            
            summary = {
                'current_portfolio_value': latest.portfolio_value,
                'total_return': latest.total_return,
                'current_drawdown': latest.current_drawdown,
                'max_drawdown': latest.max_drawdown,
                'sharpe_ratio': latest.sharpe_ratio,
                'win_rate': latest.win_rate,
                'profit_factor': latest.profit_factor,
                'total_trades': latest.total_trades,
                'active_positions': latest.active_positions,
                'risk_score': latest.risk_score,
                'avg_daily_return': np.mean(returns) if returns else 0,
                'volatility': np.std(returns) if returns else 0,
                'best_day': max(returns) if returns else 0,
                'worst_day': min(returns) if returns else 0
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"❌ 性能摘要生成失败: {str(e)}")
            return {}
    
    def get_alerts_summary(self, hours: int = 24) -> Dict:
        """获取告警摘要"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            recent_alerts = [alert for alert in self.alerts if alert.timestamp > cutoff_time]
            
            # 按级别统计
            level_counts = {}
            category_counts = {}
            
            for alert in recent_alerts:
                level_counts[alert.level.value] = level_counts.get(alert.level.value, 0) + 1
                category_counts[alert.category] = category_counts.get(alert.category, 0) + 1
            
            summary = {
                'total_alerts': len(recent_alerts),
                'level_counts': level_counts,
                'category_counts': category_counts,
                'recent_alerts': [
                    {
                        'timestamp': alert.timestamp.isoformat(),
                        'level': alert.level.value,
                        'category': alert.category,
                        'message': alert.message
                    }
                    for alert in recent_alerts[-10:]  # 最近10条
                ]
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"❌ 告警摘要生成失败: {str(e)}")
            return {}
    
    def create_performance_charts(self, output_dir: str = 'logs/monitoring/charts'):
        """创建性能图表"""
        try:
            if not self.performance_history:
                return
            
            os.makedirs(output_dir, exist_ok=True)
            
            # 准备数据
            df = pd.DataFrame([asdict(m) for m in self.performance_history])
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            
            # 创建图表
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # 投资组合价值曲线
            axes[0, 0].plot(df.index, df['portfolio_value'])
            axes[0, 0].set_title('投资组合价值')
            axes[0, 0].set_ylabel('价值')
            axes[0, 0].grid(True)
            
            # 回撤曲线
            axes[0, 1].fill_between(df.index, df['current_drawdown'], 0, alpha=0.3, color='red')
            axes[0, 1].set_title('回撤曲线')
            axes[0, 1].set_ylabel('回撤 %')
            axes[0, 1].grid(True)
            
            # 日收益率分布
            axes[1, 0].hist(df['daily_return'].dropna(), bins=30, alpha=0.7)
            axes[1, 0].set_title('日收益率分布')
            axes[1, 0].set_xlabel('收益率')
            axes[1, 0].set_ylabel('频次')
            axes[1, 0].grid(True)
            
            # 风险评分
            axes[1, 1].plot(df.index, df['risk_score'])
            axes[1, 1].set_title('风险评分')
            axes[1, 1].set_ylabel('风险评分')
            axes[1, 1].grid(True)
            
            plt.tight_layout()
            plt.savefig(f'{output_dir}/performance_charts.png', dpi=150, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"✅ 性能图表已保存: {output_dir}/performance_charts.png")
            
        except Exception as e:
            self.logger.error(f"❌ 性能图表创建失败: {str(e)}")
    
    def save_monitoring_data(self, output_dir: str = 'logs/monitoring'):
        """保存监控数据"""
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            # 保存性能历史
            if self.performance_history:
                performance_df = pd.DataFrame([asdict(m) for m in self.performance_history])
                performance_df.to_csv(f'{output_dir}/performance_history.csv', index=False)
            
            # 保存告警历史
            if self.alerts:
                alerts_data = []
                for alert in self.alerts:
                    alert_dict = asdict(alert)
                    alert_dict['timestamp'] = alert_dict['timestamp'].isoformat()
                    alerts_data.append(alert_dict)
                
                with open(f'{output_dir}/alerts_history.json', 'w', encoding='utf-8') as f:
                    json.dump(alerts_data, f, indent=2, ensure_ascii=False)
            
            # 保存系统指标
            if self.system_metrics:
                with open(f'{output_dir}/system_metrics.json', 'w', encoding='utf-8') as f:
                    json.dump(self.system_metrics, f, indent=2, ensure_ascii=False, default=str)
            
            self.logger.info(f"✅ 监控数据已保存: {output_dir}")
            
        except Exception as e:
            self.logger.error(f"❌ 监控数据保存失败: {str(e)}")


# 使用示例
if __name__ == '__main__':
    # 配置示例
    config = {
        'alert_thresholds': {
            'max_drawdown': 0.1,
            'daily_return': -0.05,
            'sharpe_ratio': 0.5,
            'risk_score': 0.8,
            'max_positions': 10
        },
        'notifications': {
            'email': {
                'enabled': False,
                'smtp_server': 'smtp.gmail.com',
                'smtp_port': 587,
                'username': 'your_email@gmail.com',
                'password': 'your_password',
                'from_email': 'your_email@gmail.com',
                'to_email': 'recipient@gmail.com'
            },
            'wechat': {
                'enabled': False,
                'webhook_url': 'https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=your_key'
            },
            'dingtalk': {
                'enabled': False,
                'webhook_url': 'https://oapi.dingtalk.com/robot/send?access_token=your_token'
            }
        }
    }
    
    # 创建监控器
    monitor = TradingMonitor(config)
    
    # 启动监控
    monitor.start_monitoring()
    
    print("监控系统已启动")

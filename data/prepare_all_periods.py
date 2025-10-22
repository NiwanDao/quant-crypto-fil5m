#!/usr/bin/env python3
"""
数据准备主脚本 - 为3个不同时间段准备数据
1. 2024-8 到 2025-3
2. 2025-3 到 2025-6  
3. 2025-6 至今
"""

import subprocess
import sys
import os
from datetime import datetime

def run_script(script_name, period_name):
    """运行指定的数据准备脚本"""
    print(f"\n{'='*60}")
    print(f"开始准备 {period_name} 数据")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=True, text=True, check=True)
        print(f"✅ {period_name} 数据准备完成")
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {period_name} 数据准备失败")
        print(f"错误信息: {e.stderr}")
        return False

def main():
    print("开始准备3个时间段的数据...")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 切换到项目根目录
    os.chdir('/home/shiyi/quant-crypto-fil5m')
    
    # 定义要运行的脚本和对应的时期
    scripts = [
        ('data/prepare_2024_8_to_2025_3.py', '2024-8 到 2025-3'),
        ('data/prepare_2025_3_to_2025_6.py', '2025-3 到 2025-6'),
        ('data/prepare_2025_6_to_now.py', '2025-6 至今')
    ]
    
    success_count = 0
    total_count = len(scripts)
    
    for script, period in scripts:
        if run_script(script, period):
            success_count += 1
        else:
            print(f"跳过 {period} 的后续处理")
    
    print(f"\n{'='*60}")
    print(f"数据准备完成总结")
    print(f"{'='*60}")
    print(f"成功: {success_count}/{total_count}")
    print(f"完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if success_count == total_count:
        print("🎉 所有时间段数据准备完成！")
    else:
        print("⚠️  部分时间段数据准备失败，请检查错误信息")

if __name__ == '__main__':
    main()

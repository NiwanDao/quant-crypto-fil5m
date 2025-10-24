"""
运行LightGBM模型优化的主脚本
执行所有优化步骤：训练、测试、部署
"""

import os
import sys
import subprocess
import time
from datetime import datetime

def run_command(command, description):
    """运行命令并显示结果"""
    print(f"\n{'='*60}")
    print(f"🚀 {description}")
    print(f"{'='*60}")
    print(f"执行命令: {command}")
    
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ {description} 成功完成")
            if result.stdout:
                print("输出:")
                print(result.stdout)
        else:
            print(f"❌ {description} 失败")
            print("错误信息:")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ 执行命令时出错: {e}")
        return False
    
    return True

def check_requirements():
    """检查依赖项"""
    print("🔍 检查依赖项...")
    
    required_packages = [
        'optuna',
        'lightgbm',
        'scikit-learn',
        'pandas',
        'numpy',
        'matplotlib',
        'seaborn'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ 缺少依赖包: {missing_packages}")
        print("请运行: pip install " + " ".join(missing_packages))
        return False
    
    print("✅ 所有依赖项已安装")
    return True

def main():
    """主函数"""
    print("🚀 LightGBM模型优化流程")
    print("="*60)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 检查依赖项
    if not check_requirements():
        print("❌ 依赖项检查失败，退出")
        return
    
    # 步骤1: 数据准备
    print("\n📊 步骤1: 数据准备")
    if not run_command("python data/prepare.py", "数据获取和特征构建"):
        print("❌ 数据准备失败，请检查数据源")
        return
    
    # 步骤2: 运行优化的模型训练
    print("\n🤖 步骤2: 优化模型训练")
    if not run_command("python models/train_lgb_optimized.py", "LightGBM模型优化训练"):
        print("❌ 模型训练失败")
        return
    
    # 步骤3: 测试优化后的模型
    print("\n🧪 步骤3: 模型性能测试")
    if not run_command("python test_optimized_model.py", "优化模型性能测试"):
        print("⚠️ 模型测试失败，但继续执行")
    
    # 步骤4: 运行优化后的回测
    print("\n📈 步骤4: 优化回测")
    if not run_command("python backtest/run_vectorbt.py", "优化回测"):
        print("⚠️ 回测失败，但继续执行")
    
    # 步骤5: 启动优化的API服务
    print("\n🌐 步骤5: 启动优化API服务")
    print("API服务将在后台启动...")
    print("访问地址: http://localhost:8000")
    print("API文档: http://localhost:8000/docs")
    
    # 启动API服务（在后台）
    try:
        api_process = subprocess.Popen([
            "python", "-m", "uvicorn", 
            "live.app_optimized:app", 
            "--host", "0.0.0.0", 
            "--port", "8000",
            "--reload"
        ])
        
        print("✅ API服务已启动")
        print("按 Ctrl+C 停止服务")
        
        # 等待用户中断
        try:
            api_process.wait()
        except KeyboardInterrupt:
            print("\n🛑 停止API服务...")
            api_process.terminate()
            api_process.wait()
            print("✅ API服务已停止")
            
    except Exception as e:
        print(f"❌ 启动API服务失败: {e}")
    
    print(f"\n完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("🎉 LightGBM模型优化完成！")

if __name__ == '__main__':
    main()


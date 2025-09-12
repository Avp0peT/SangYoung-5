#!/usr/bin/env python3
"""
人体解析项目主入口脚本
支持训练模型、推理测试和启动Web服务
"""

import argparse
import os
import sys
from train import main as train_main
from inference import HumanParsingInference, main as inference_main
from web_app import init_model, app
import subprocess

def setup_environment():
    """设置环境并安装依赖"""
    print("正在设置环境...")
    
    # 检查Python版本
    if sys.version_info < (3, 7):
        print("需要Python 3.7或更高版本")
        return False
    
    # 安装依赖
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("依赖安装完成")
        return True
    except subprocess.CalledProcessError:
        print("依赖安装失败，请手动运行: pip install -r requirements.txt")
        return False

def train_model():
    """训练模型"""
    print("开始训练模型...")
    train_main()
    print("训练完成！模型保存在 checkpoints/ 目录")

def test_inference():
    """测试推理"""
    print("测试推理功能...")
    inference_main()

def start_web_service():
    """启动Web服务"""
    print("启动Web服务...")
    init_model()
    app.run(debug=True, host='0.0.0.0', port=5000)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='人体解析项目')
    parser.add_argument('--setup', action='store_true', help='安装依赖并设置环境')
    parser.add_argument('--train', action='store_true', help='训练模型')
    parser.add_argument('--test', action='store_true', help='测试推理')
    parser.add_argument('--web', action='store_true', help='启动Web服务')
    parser.add_argument('--all', action='store_true', help='执行完整流程：安装依赖、训练、测试')
    
    args = parser.parse_args()
    
    if not any(vars(args).values()):
        parser.print_help()
        return
    
    # 设置环境
    if args.setup or args.all:
        if not setup_environment():
            return
    
    # 训练模型
    if args.train or args.all:
        train_model()
    
    # 测试推理
    if args.test or args.all:
        test_inference()
    
    # 启动Web服务
    if args.web:
        start_web_service()

if __name__ == '__main__':
    main()

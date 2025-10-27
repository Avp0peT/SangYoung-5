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


def process_images_cli(input_path, model_path=None, output_base=None, no_visualize=False, device=None):
    """Process a single image or directory of images (wrapped for CLI usage).

    This re-uses HumanParsingInference to avoid an extra script.
    """
    project_root = os.path.dirname(os.path.abspath(__file__))

    # Find default model if not provided
    def find_default_model(project_root):
        candidates = [
            os.path.join(project_root, 'checkpoints', 'final_model.pth'),
            os.path.join(project_root, 'checkpoints', 'best_model.pth')
        ]
        for c in candidates:
            if os.path.exists(c):
                return c
        ckpt_dir = os.path.join(project_root, 'checkpoints')
        if os.path.isdir(ckpt_dir):
            files = [os.path.join(ckpt_dir, f) for f in os.listdir(ckpt_dir) if f.endswith('.pth')]
            if files:
                files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
                return files[0]
        return None

    if model_path is None:
        model_path = find_default_model(project_root)
    if model_path is None:
        print('No model checkpoint found. Please provide --model or put a checkpoint in checkpoints/')
        return

    if device is None:
        device = 'cuda' if __import__('torch').cuda.is_available() else 'cpu'

    print(f"Using model: {model_path}")
    print(f"Device: {device}")

    inference = HumanParsingInference(model_path=model_path, device=device)

    # helper utils
    def list_images_in_dir(directory, exts=None):
        if exts is None:
            exts = ['.jpg', '.jpeg', '.png', '.bmp']
        images = []
        for root, _, files in os.walk(directory):
            for f in files:
                if os.path.splitext(f.lower())[1] in exts:
                    images.append(os.path.join(root, f))
        return images

    def make_run_dir(base='static/results'):
        from datetime import datetime
        import uuid
        os.makedirs(base, exist_ok=True)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_id = uuid.uuid4().hex[:8]
        run_dir = os.path.join(base, f"{ts}_{run_id}")
        os.makedirs(run_dir, exist_ok=True)
        return run_dir

    base_output = output_base or os.path.join(project_root, 'static', 'results')
    run_dir = make_run_dir(base_output)

    to_process = []
    if os.path.isfile(input_path):
        to_process = [input_path]
    elif os.path.isdir(input_path):
        images = list_images_in_dir(input_path)
        images.sort()
        to_process = images
    else:
        print('Input path is not a file or directory')
        return

    if not to_process:
        print('No images found to process')
        return

    for img_path in to_process:
        name = os.path.splitext(os.path.basename(img_path))[0]
        out_dir = os.path.join(run_dir, name)
        try:
            os.makedirs(out_dir, exist_ok=True)
            inference.process_image(img_path, output_dir=out_dir, visualize=(not no_visualize))
            print(f"Processed: {img_path} -> {out_dir}")
        except Exception as e:
            print(f"Failed: {img_path}: {e}")

    print(f"All outputs saved under: {run_dir}")

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
    parser.add_argument('--process', action='store_true', help='Process a single image or a folder of images')
    parser.add_argument('--input', '-i', help='Path to image file or folder to process (required with --process)')
    parser.add_argument('--model', '-m', help='Path to model checkpoint (.pth). If omitted, script will try to find one in checkpoints/')
    parser.add_argument('--output', '-o', help='Base output directory (default: static/results/)')
    parser.add_argument('--no-visualize', action='store_true', help='Do not save visualization images (only masks and parts)')
    parser.add_argument('--device', help='Torch device to use (e.g., cpu or cuda). By default auto-detected')
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
    
    # 处理图片（单张或目录）
    if args.process:
        if not args.input:
            print('请使用 --input 指定要处理的图片或目录')
            return
        process_images_cli(input_path=args.input, model_path=args.model, output_base=args.output, no_visualize=args.no_visualize, device=args.device)
    
    # 启动Web服务
    if args.web:
        start_web_service()

if __name__ == '__main__':
    main()

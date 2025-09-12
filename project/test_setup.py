#!/usr/bin/env python3
"""
项目设置测试脚本
验证环境配置和基本功能
"""

import os
import sys
import subprocess

def test_imports():
    """测试所有必要的导入"""
    print("测试模块导入...")
    
    try:
        import torch
        import torchvision
        import numpy as np
        import cv2
        from PIL import Image
        import flask
        import matplotlib
        import segmentation_models_pytorch as smp
        
        print("✓ 所有依赖包导入成功")
        print(f"  PyTorch版本: {torch.__version__}")
        print(f"  TorchVision版本: {torchvision.__version__}")
        print(f"  OpenCV版本: {cv2.__version__}")
        print(f"  SMP版本: {smp.__version__}")
        
        # 测试CUDA是否可用
        if torch.cuda.is_available():
            print(f"✓ CUDA可用: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠ CUDA不可用，将使用CPU")
            
        return True
        
    except ImportError as e:
        print(f"✗ 导入失败: {e}")
        return False

def test_dataset():
    """测试数据集加载"""
    print("\n测试数据集加载...")
    
    try:
        from dataset import ATRDataset
        
        # 检查数据集路径
        image_dir = '../humanparsing/JPEGImages'
        mask_dir = '../humanparsing/SegmentationClassAug'
        
        if not os.path.exists(image_dir):
            print(f"✗ 图像目录不存在: {image_dir}")
            return False
            
        if not os.path.exists(mask_dir):
            print(f"✗ 掩码目录不存在: {mask_dir}")
            return False
        
        # 尝试创建数据集实例
        dataset = ATRDataset(image_dir, mask_dir)
        print(f"✓ 数据集加载成功")
        print(f"  图像数量: {len(dataset)}")
        print(f"  类别数量: {len(dataset.classes)}")
        print(f"  类别列表: {dataset.classes}")
        
        return True
        
    except Exception as e:
        print(f"✗ 数据集测试失败: {e}")
        return False

def test_model_creation():
    """测试模型创建"""
    print("\n测试模型创建...")
    
    try:
        import segmentation_models_pytorch as smp
        
        # 创建模型
        model = smp.Unet(
            encoder_name='resnet34',
            encoder_weights=None,
            in_channels=3,
            classes=18,
        )
        
        print("✓ 模型创建成功")
        print(f"  参数数量: {sum(p.numel() for p in model.parameters()):,}")
        
        return True
        
    except Exception as e:
        print(f"✗ 模型创建失败: {e}")
        return False

def test_file_structure():
    """测试文件结构"""
    print("\n测试文件结构...")
    
    required_files = [
        'dataset.py',
        'train.py', 
        'inference.py',
        'web_app.py',
        'main.py',
        'requirements.txt',
        'templates/index.html'
    ]
    
    all_exists = True
    for file in required_files:
        if os.path.exists(file):
            print(f"✓ {file} 存在")
        else:
            print(f"✗ {file} 不存在")
            all_exists = False
    
    return all_exists

def main():
    """主测试函数"""
    print("=" * 50)
    print("人体解析项目 - 环境测试")
    print("=" * 50)
    
    tests = [
        test_file_structure,
        test_imports,
        test_dataset,
        test_model_creation
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"✗ 测试执行失败: {e}")
            results.append(False)
    
    print("\n" + "=" * 50)
    print("测试结果汇总:")
    print("=" * 50)
    
    passed = sum(results)
    total = len(results)
    
    print(f"通过测试: {passed}/{total}")
    
    if passed == total:
        print("🎉 所有测试通过！项目环境配置成功")
        print("\n下一步操作:")
        print("1. 训练模型: python main.py --train")
        print("2. 测试推理: python main.py --test") 
        print("3. 启动Web服务: python main.py --web")
    else:
        print("⚠ 部分测试失败，请检查环境配置")
        print("\n建议操作:")
        print("1. 安装依赖: pip install -r requirements.txt")
        print("2. 检查数据集路径是否正确")
        print("3. 确保所有必要文件存在")
    
    return all(results)

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)

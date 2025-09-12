#!/usr/bin/env python3
"""
测试训练好的模型对指定图片的效果
"""

import os
import sys
from inference import HumanParsingInference

def test_trained_model():
    """测试训练好的模型"""
    # 测试图片路径
    test_image_path = 'C:/Users/13944/Desktop/img/slj.jpg'
    
    print(f"测试图片: {test_image_path}")
    
    # 检查图片是否存在
    if not os.path.exists(test_image_path):
        print(f"错误: 图片不存在: {test_image_path}")
        return False
    
    # 创建推理器
    print("初始化推理器...")
    inference = HumanParsingInference()
    
    # 检查是否有训练好的模型
    model_path = 'checkpoints/best_model.pth'
    if os.path.exists(model_path):
        print("加载训练好的模型...")
        inference.load_model(model_path)
        print("✅ 模型加载成功")
    else:
        print("❌ 错误: 没有找到训练好的模型")
        print("请先运行训练: python main.py --train")
        return False
    
    # 处理图片
    print("处理图片中...")
    try:
        result = inference.process_image(
            test_image_path, 
            output_dir='trained_test_results',
            visualize=True
        )
        
        print("✅ 图片处理成功!")
        print(f"原始图像: {result['original_image']}")
        print(f"分割掩码: {result['prediction_mask']}")
        print(f"可视化结果: {result['visualization']}")
        print(f"提取的身体部位: {len(result['extracted_parts'])} 个")
        
        for part_name, part_path in result['extracted_parts'].items():
            print(f"  - {part_name}: {part_path}")
            
        return True
        
    except Exception as e:
        print(f"❌ 处理图片时出错: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    print("=" * 50)
    print("训练模型测试")
    print("=" * 50)
    
    success = test_trained_model()
    
    print("\n" + "=" * 50)
    if success:
        print("✅ 测试完成！结果保存在 trained_test_results/ 目录")
    else:
        print("❌ 测试失败")
    
    sys.exit(0 if success else 1)

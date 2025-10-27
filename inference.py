import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
import os
import matplotlib.pyplot as plt
from dataset import ATRDataset
import segmentation_models_pytorch as smp

class HumanParsingInference:
    """人体解析推理类"""
    
    def __init__(self, model_path=None, num_classes=18, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.num_classes = num_classes
        self.model = None
        self.class_names = [
            'background', 'hat', 'hair', 'sunglasses', 'upper-clothes',
            'skirt', 'pants', 'dress', 'belt', 'left-shoe', 'right-shoe',
            'face', 'left-leg', 'right-leg', 'left-arm', 'right-arm',
            'bag', 'scarf'
        ]
        
        # 创建模型
        self.create_model()
        
        # 加载模型（如果提供了模型路径）
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def create_model(self):
        """创建U-Net分割模型"""
        self.model = smp.Unet(
            encoder_name='resnet34',
            encoder_weights=None,  # 不加载预训练权重
            in_channels=3,
            classes=self.num_classes,
        )
        self.model = self.model.to(self.device)
        self.model.eval()
    
    def load_model(self, model_path):
        """加载预训练模型"""
        checkpoint = torch.load(model_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        print(f"模型已从 {model_path} 加载")
    
    def preprocess_image(self, image, image_size=(512, 512)):
        """预处理图像"""
        if isinstance(image, str):
            # 从文件路径读取图像
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif isinstance(image, np.ndarray):
            # 已经是numpy数组
            if len(image.shape) == 3 and image.shape[2] == 3:
                pass  # 已经是RGB
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 保存原始尺寸
        original_size = image.shape[:2]
        
        # 调整大小
        image_resized = cv2.resize(image, image_size)
        
        # 归一化并转换为Tensor
        image_tensor = torch.from_numpy(image_resized).float() / 255.0
        image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
        image_tensor = image_tensor.to(self.device)
        
        return image_tensor, original_size, image_resized
    
    def predict(self, image, image_size=(512, 512)):
        """预测图像的分割结果"""
        # 预处理图像
        image_tensor, original_size, image_resized = self.preprocess_image(image, image_size)
        
        # 推理
        with torch.no_grad():
            output = self.model(image_tensor)
            pred_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
        
        # 将预测掩码调整回原始尺寸
        pred_mask_original = cv2.resize(pred_mask.astype(np.uint8), 
                                      (original_size[1], original_size[0]), 
                                      interpolation=cv2.INTER_NEAREST)
        
        return pred_mask_original, pred_mask, image_resized
    
    def visualize_result(self, image, pred_mask, save_path=None, alpha=0.6):
        """可视化分割结果"""
        if isinstance(image, str):
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 创建彩色分割图
        colored_mask = np.zeros_like(image)
        for class_idx in range(self.num_classes):
            mask = pred_mask == class_idx
            color = self.get_class_color(class_idx)
            colored_mask[mask] = color
        
        # 叠加原图和分割图
        overlayed = cv2.addWeighted(image, 1 - alpha, colored_mask, alpha, 0)
        
        # 创建对比图
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 原始图像
        axes[0].imshow(image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # 分割掩码
        axes[1].imshow(colored_mask)
        axes[1].set_title('Segmentation Mask')
        axes[1].axis('off')
        
        # 叠加结果
        axes[2].imshow(overlayed)
        axes[2].set_title('Overlayed Result')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()
        else:
            plt.show()
        
        return overlayed, colored_mask
    
    def get_class_color(self, class_idx):
        """获取类别颜色"""
        # 为每个类别生成不同的颜色
        hue = class_idx * 180 // self.num_classes
        hsv_color = np.uint8([[[hue, 255, 255]]])
        rgb_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2RGB)[0][0]
        return rgb_color
    
    def extract_body_parts(self, image, pred_mask, output_dir='output_parts'):
        """提取并保存各个身体部位"""
        os.makedirs(output_dir, exist_ok=True)
        
        if isinstance(image, str):
            original_image = cv2.imread(image)
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        else:
            original_image = image.copy()
        
        extracted_parts = {}
        
        for class_idx, class_name in enumerate(self.class_names):
            if class_idx == 0:  # 跳过背景
                continue
                
            # 创建该部位的掩码
            part_mask = (pred_mask == class_idx).astype(np.uint8) * 255
            
            if np.any(part_mask > 0):  # 如果该部位存在
                # 提取该部位
                part_image = original_image.copy()
                part_image[part_mask == 0] = 0  # 将非该部位的区域设为黑色
                
                # 保存部位图像
                part_filename = f"{class_name}.png"
                part_path = os.path.join(output_dir, part_filename)
                cv2.imwrite(part_path, cv2.cvtColor(part_image, cv2.COLOR_RGB2BGR))
                
                extracted_parts[class_name] = part_path
                print(f"已提取并保存: {class_name}")
        
        return extracted_parts
    
    def process_image(self, image_path, output_dir='results', visualize=True):
        """处理单张图像"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 获取文件名
        filename = os.path.basename(image_path)
        name_without_ext = os.path.splitext(filename)[0]
        
        # 预测
        pred_mask, _, image_resized = self.predict(image_path)
        
        # 提取身体部位
        parts_dir = os.path.join(output_dir, f'{name_without_ext}_parts')
        extracted_parts = self.extract_body_parts(image_path, pred_mask, parts_dir)
        
        # 可视化结果
        if visualize:
            vis_path = os.path.join(output_dir, f'{name_without_ext}_result.png')
            self.visualize_result(image_path, pred_mask, vis_path)
        
        # 保存原始预测掩码
        mask_path = os.path.join(output_dir, f'{name_without_ext}_mask.png')
        cv2.imwrite(mask_path, pred_mask)
        
        print(f"处理完成: {filename}")
        print(f"结果保存在: {output_dir}")
        
        return {
            'original_image': image_path,
            'prediction_mask': mask_path,
            'visualization': vis_path if visualize else None,
            'extracted_parts': extracted_parts
        }

def main():
    """主推理函数"""
    # 创建推理器
    inference = HumanParsingInference()
    
    # 测试图像路径
    test_image_path = r'C:\Users\13944\Desktop\img\slj.jpg'
    
    if os.path.exists(test_image_path):
        # 处理测试图像
        result = inference.process_image(test_image_path, output_dir='test_results')
        print("测试完成!")
    else:
        print(f"测试图像不存在: {test_image_path}")
        print("请先训练模型或提供正确的图像路径")

if __name__ == '__main__':
    main()

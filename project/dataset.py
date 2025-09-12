import os
import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class ATRDataset(Dataset):
    """ATR人体解析数据集类"""
    
    def __init__(self, image_dir, mask_dir, transform=None, image_size=(512, 512)):
        """
        初始化数据集
        
        Args:
            image_dir: 图像目录路径
            mask_dir: 掩码目录路径
            transform: 数据增强变换
            image_size: 图像尺寸
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_size = image_size
        
        # 获取所有图像文件名
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
        
        # ATR数据集类别定义 (18个身体部位 + 背景)
        self.classes = [
            'background', 'hat', 'hair', 'sunglasses', 'upper-clothes',
            'skirt', 'pants', 'dress', 'belt', 'left-shoe', 'right-shoe',
            'face', 'left-leg', 'right-leg', 'left-arm', 'right-arm',
            'bag', 'scarf'
        ]
        
        # 类别到索引的映射
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # 获取图像文件名
        img_name = self.image_files[idx]
        
        # 构建图像和掩码路径
        img_path = os.path.join(self.image_dir, img_name)
        mask_name = img_name.replace('.jpg', '.png')
        mask_path = os.path.join(self.mask_dir, mask_name)
        
        # 读取图像
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.image_size)
        
        # 读取掩码
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            mask = cv2.resize(mask, self.image_size, interpolation=cv2.INTER_NEAREST)
        else:
            # 如果掩码不存在，创建全零掩码
            mask = np.zeros(self.image_size, dtype=np.uint8)
        
        # 转换为Tensor
        image = transforms.ToTensor()(image)
        mask = torch.from_numpy(mask).long()
        
        if self.transform:
            image = self.transform(image)
        
        return image, mask
    
    def get_class_names(self):
        """获取类别名称列表"""
        return self.classes
    
    def get_class_colors(self):
        """获取每个类别的颜色映射"""
        # 为每个类别生成不同的颜色
        colors = []
        for i in range(len(self.classes)):
            # 生成不同的颜色
            hue = i * 180 // len(self.classes)
            color = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2RGB)[0][0]
            colors.append(tuple(color))
        return colors

def create_data_loaders(image_dir, mask_dir, batch_size=8, train_ratio=0.8, image_size=(512, 512)):
    """
    创建训练和验证数据加载器
    
    Args:
        image_dir: 图像目录路径
        mask_dir: 掩码目录路径
        batch_size: 批次大小
        train_ratio: 训练集比例
        image_size: 图像尺寸
    
    Returns:
        train_loader, val_loader, dataset
    """
    # 数据增强
    train_transform = transforms.Compose([
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 创建完整数据集
    full_dataset = ATRDataset(image_dir, mask_dir, transform=None, image_size=image_size)
    
    # 划分训练集和验证集
    dataset_size = len(full_dataset)
    train_size = int(train_ratio * dataset_size)
    val_size = dataset_size - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    # 为训练集和验证集分别设置不同的变换
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, val_loader, full_dataset

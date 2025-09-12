import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from dataset import ATRDataset, create_data_loaders
import os
import time
from tqdm import tqdm
import matplotlib.pyplot as plt

class HumanParsingTrainer:
    """人体解析模型训练器"""
    
    def __init__(self, num_classes=18, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.num_classes = num_classes
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        
        # 创建模型
        self.create_model()
        
    def create_model(self):
        """创建U-Net分割模型"""
        self.model = smp.Unet(
            encoder_name='resnet34',        # 使用ResNet34作为编码器
            encoder_weights='imagenet',     # 使用ImageNet预训练权重
            in_channels=3,                  # 输入通道数
            classes=self.num_classes,       # 输出类别数
        )
        self.model = self.model.to(self.device)
        
        # 定义损失函数（交叉熵损失）
        self.criterion = nn.CrossEntropyLoss()
        
        # 定义优化器
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4, weight_decay=1e-5)
        
        # 定义学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
    
    def train_epoch(self, train_loader):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        correct_pixels = 0
        total_pixels = 0
        
        progress_bar = tqdm(train_loader, desc='Training')
        
        for images, masks in progress_bar:
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(images)
            
            # 计算损失
            loss = self.criterion(outputs, masks)
            
            # 反向传播
            loss.backward()
            self.optimizer.step()
            
            # 统计信息
            total_loss += loss.item()
            
            # 计算准确率
            _, predicted = torch.max(outputs, 1)
            correct_pixels += (predicted == masks).sum().item()
            total_pixels += masks.numel()
            
            # 更新进度条
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{correct_pixels/total_pixels:.4f}'
            })
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct_pixels / total_pixels
        
        return avg_loss, accuracy
    
    def validate(self, val_loader):
        """验证模型"""
        self.model.eval()
        total_loss = 0
        correct_pixels = 0
        total_pixels = 0
        
        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc='Validation')
            
            for images, masks in progress_bar:
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                # 前向传播
                outputs = self.model(images)
                
                # 计算损失
                loss = self.criterion(outputs, masks)
                total_loss += loss.item()
                
                # 计算准确率
                _, predicted = torch.max(outputs, 1)
                correct_pixels += (predicted == masks).sum().item()
                total_pixels += masks.numel()
                
                # 更新进度条
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{correct_pixels/total_pixels:.4f}'
                })
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct_pixels / total_pixels
        
        return avg_loss, accuracy
    
    def train(self, train_loader, val_loader, num_epochs=50, save_dir='checkpoints'):
        """训练模型"""
        os.makedirs(save_dir, exist_ok=True)
        
        train_losses = []
        val_losses = []
        train_accs = []
        val_accs = []
        
        best_val_loss = float('inf')
        
        print(f"开始训练，使用设备: {self.device}")
        print(f"训练集大小: {len(train_loader.dataset)}")
        print(f"验证集大小: {len(val_loader.dataset)}")
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print("-" * 50)
            
            # 训练
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # 验证
            val_loss, val_acc = self.validate(val_loader)
            
            # 更新学习率
            self.scheduler.step(val_loss)
            
            # 保存统计信息
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accs.append(train_acc)
            val_accs.append(val_acc)
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'train_acc': train_acc,
                    'val_acc': val_acc,
                }, os.path.join(save_dir, 'best_model.pth'))
                print("保存最佳模型")
            
            # 每10个epoch保存一次检查点
            if (epoch + 1) % 10 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'train_acc': train_acc,
                    'val_acc': val_acc,
                }, os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth'))
        
        # 保存最终模型
        torch.save({
            'epoch': num_epochs,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, os.path.join(save_dir, 'final_model.pth'))
        
        # 绘制训练曲线
        self.plot_training_curves(train_losses, val_losses, train_accs, val_accs, save_dir)
        
        return train_losses, val_losses, train_accs, val_accs
    
    def plot_training_curves(self, train_losses, val_losses, train_accs, val_accs, save_dir):
        """绘制训练曲线"""
        plt.figure(figsize=(12, 5))
        
        # 损失曲线
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Val Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # 准确率曲线
        plt.subplot(1, 2, 2)
        plt.plot(train_accs, label='Train Accuracy')
        plt.plot(val_accs, label='Val Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'training_curves.png'))
        plt.close()
    
    def load_model(self, model_path):
        """加载预训练模型"""
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"加载模型从 {model_path}")

def main():
    """主训练函数"""
    # 数据路径
    image_dir = '../humanparsing/JPEGImages'
    mask_dir = '../humanparsing/SegmentationClassAug'
    
    # 创建数据加载器
    train_loader, val_loader, dataset = create_data_loaders(
        image_dir, mask_dir, batch_size=8, image_size=(512, 512)
    )
    
    # 创建训练器
    trainer = HumanParsingTrainer(num_classes=len(dataset.classes))
    
    # 开始训练
    trainer.train(train_loader, val_loader, num_epochs=50, save_dir='checkpoints')

if __name__ == '__main__':
    main()

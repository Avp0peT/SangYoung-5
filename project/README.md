# 人体解析项目 (Human Parsing Project)

基于ATR数据集训练的人体部位分割模型，提供图像中人体部位的自动识别、分割和提取功能，并包含完整的Web服务接口。

## 功能特性

- 🎯 **精准分割**: 识别和分割18个人体部位
- 🌐 **Web界面**: 友好的图形用户界面
- 🔌 **REST API**: 提供编程接口
- 📊 **可视化**: 实时显示分割结果
- 💾 **导出功能**: 单独保存每个身体部位
- 🚀 **高性能**: 基于PyTorch和U-Net架构

## 支持的身体部位

1. 背景 (background)
2. 帽子 (hat)
3. 头发 (hair)
4. 太阳镜 (sunglasses)
5. 上衣 (upper-clothes)
6. 裙子 (skirt)
7. 裤子 (pants)
8. 连衣裙 (dress)
9. 腰带 (belt)
10. 左鞋 (left-shoe)
11. 右鞋 (right-shoe)
12. 脸部 (face)
13. 左腿 (left-leg)
14. 右腿 (right-leg)
15. 左臂 (left-arm)
16. 右臂 (right-arm)
17. 包 (bag)
18. 围巾 (scarf)

## 项目结构

```
project/
├── dataset.py          # 数据集处理模块
├── train.py           # 模型训练模块
├── inference.py       # 推理预测模块
├── web_app.py         # Web服务接口
├── main.py           # 主入口脚本
├── requirements.txt   # 依赖包列表
├── templates/
│   └── index.html    # Web界面模板
└── README.md         # 项目说明
```

## 快速开始

### 1. 安装依赖

```bash
cd project
pip install -r requirements.txt
```

或者使用主脚本自动安装：

```bash
python main.py --setup
```

### 2. 训练模型

```bash
python main.py --train
```

或者直接运行训练脚本：

```bash
python train.py
```

### 3. 测试推理

```bash
python main.py --test
```

或者直接运行推理脚本：

```bash
python inference.py
```

### 4. 启动Web服务

```bash
python main.py --web
```

或者直接运行Web应用：

```bash
python web_app.py
```

访问 http://localhost:5000 使用Web界面

## 使用方法

### 通过Web界面

1. 启动Web服务
2. 打开浏览器访问 http://localhost:5000
3. 上传包含人物的图像
4. 查看分割结果和提取的身体部位

### 通过API接口

```python
import requests

# 发送图像进行预测
url = "http://localhost:5000/api/predict"
files = {'image': open('path/to/image.jpg', 'rb')}
response = requests.post(url, files=files)

if response.json()['success']:
    result = response.json()
    print("预测成功!")
    print("可用的身体部位:", list(result['parts'].keys()))
else:
    print("预测失败:", response.json()['error'])
```

### 健康检查

```bash
curl http://localhost:5000/health
```

## 数据集

本项目使用ATR (Atrous Spatial Pyramid Pooling) 人体解析数据集，包含：
- JPEGImages/: 原始图像
- SegmentationClassAug/: 分割标注掩码

## 模型架构

- **主干网络**: ResNet34
- **分割头**: U-Net decoder
- **预训练权重**: ImageNet
- **输入尺寸**: 512x512
- **输出类别**: 18个身体部位 + 背景

## 性能指标

- 训练轮次: 50 epochs
- 批次大小: 8
- 学习率: 1e-4
- 优化器: Adam
- 损失函数: CrossEntropyLoss

## 文件说明

- `dataset.py`: 数据加载和预处理
- `train.py`: 模型训练和验证
- `inference.py`: 图像推理和结果可视化
- `web_app.py`: Flask Web服务
- `main.py`: 统一入口脚本

## 开发说明

### 添加新的身体部位

1. 在 `dataset.py` 的 `classes` 列表中添加新类别
2. 重新训练模型
3. 更新前端显示逻辑

### 自定义模型架构

修改 `train.py` 中的 `create_model()` 方法：

```python
self.model = smp.Unet(
    encoder_name='resnet50',      # 更换编码器
    encoder_weights='imagenet',
    in_channels=3,
    classes=self.num_classes,
)
```

### 调整训练参数

修改 `train.py` 中的训练参数：

```python
def train(self, train_loader, val_loader, num_epochs=100, save_dir='checkpoints'):
    # 调整训练轮次和其他参数
```

## 常见问题

### Q: 训练时出现内存不足错误
A: 减小批次大小或图像尺寸

### Q: Web服务无法启动
A: 检查端口5000是否被占用，或更换端口

### Q: 预测结果不准确
A: 确保训练数据充足，或调整模型参数

### Q: 依赖安装失败
A: 手动安装特定版本的包，或使用conda环境

## 技术支持

如有问题，请检查：
1. 依赖包是否安装完整
2. 数据集路径是否正确
3. 模型文件是否存在

## 许可证

本项目仅供学习和研究使用。

## 更新日志

### v1.0.0
- 初始版本发布
- 支持18个人体部位分割
- 提供Web界面和API接口
- 完整的训练和推理流程

---

**注意**: 首次使用时请先训练模型以获得最佳效果。未训练的模型将使用随机权重，预测结果可能不准确。

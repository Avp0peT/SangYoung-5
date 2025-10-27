# 人体解析项目

该仓库实现了人体解析（人体部位分割）流水线，基于 U-Net（使用 `segmentation_models_pytorch`）对图像中的18个人体部位进行分割、可视化和提取。

## 需求

Python 3.7+，以及 `requirements.txt` 中列出的依赖。

安装依赖：

```powershell
python -m pip install -r requirements.txt
```

## 快速使用 — 单张图片

处理单张图片并保存可视化、掩码与提取的部位：

```powershell
python main.py --process --input C:\path\to\image.jpg
```

可选：指定模型检查点：

```powershell
python main.py --process --input C:\path\to\image.jpg --model project\checkpoints\final_model.pth
```

禁用可视化（仅保存掩码和部位）：

```powershell
python main.py --process --input C:\path\to\image.jpg --no-visualize
```

## 快速使用 — 文件夹（批量）

对文件夹内所有图片递归处理：

```powershell
python main.py --process --input C:\path\to\image_folder
```

处理结果保存在 `static/results/<timestamp>_<id>/<image_name>/`。

## 目录结构（概要）

```
project/
├── dataset.py
├── train.py
├── inference.py
├── web_app.py
├── main.py
├── checkpoints/
└── static/results/
```

## 说明

- `inference.py` 中包含 `HumanParsingInference`，提供预处理、推理、可视化与部位提取功能。
- `main.py` 提供统一入口：训练、测试、Web 服务与图片处理。

如需我将更详细的中文 README 内容（例如项目总结、测试结果、使用示例）合并到此文件，请告知，我会补充完整。

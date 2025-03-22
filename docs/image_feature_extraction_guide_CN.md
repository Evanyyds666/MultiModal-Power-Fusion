# 图像特征提取与处理快速验证指南

本文档提供了使用Chinese-CLIP和YOLOv8快速验证图像特征提取流程的步骤指南。

## 1. 系统要求

- Python 3.8+
- CUDA 11.4+ (GPU加速推荐)
- 最小8GB RAM (推荐16GB+)
- 最小10GB硬盘空间(用于模型和数据)

## 2. 环境配置

### 2.1 安装依赖

```bash
# 安装项目基本依赖
pip install -r requirements.txt

# 安装CN-CLIP
pip install cn_clip
pip install ftfy regex tqdm

# 安装YOLOv8
pip install ultralytics
```

### 2.2 下载预训练模型

```bash
# 创建模型目录
mkdir -p models

# 下载YOLOv8预训练模型
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt -O models/yolov8x.pt

# Chinese-CLIP模型会在首次使用时自动下载
```

## 3. 数据集准备

### 3.1 数据集目录结构

创建以下目录结构：

```
data/
├── images/
│   ├── transformer/          # 变压器设备图像
│   │   ├── img001.jpg
│   │   ├── img002.jpg
│   │   └── ...
│   ├── circuit_breaker/      # 断路器设备图像
│   │   ├── img001.jpg
│   │   ├── img002.jpg
│   │   └── ...
│   └── other_devices/        # 其他设备图像
│       ├── img001.jpg
│       ├── img002.jpg
│       └── ...
└── output/                   # 输出目录
```

### 3.2 数据集要求

- 每个设备类别建议至少20-30张图像
- 图像格式支持: JPG, PNG, JPEG, BMP, WEBP
- 建议分辨率不低于640x640
- 图像文件名格式建议使用英文和数字，避免特殊字符

### 3.3 收集示例数据(可选)

如果没有准备好数据，可以使用以下命令从互联网收集一些示例图像：

```bash
mkdir -p data/images/{transformer,circuit_breaker,insulator}

# 可以使用第三方工具如 google-images-download 或手动下载
```

## 4. 特征提取流程

### 4.1 直接使用多模态流水线

最快捷的方式是直接使用多模态流水线，仅使用图像特征：

```bash
python run_multimodal_pipeline.py \
  --images data/images \
  --output results/baseline_test \
  --yolo-model models/yolov8x.pt \
  --clip-model ViT-B-16 \
  --fusion concat \
  --visualize
```

### 4.2 分步提取特征

如果需要单独执行每个步骤：

#### 4.2.1 提取CN-CLIP特征

```bash
mkdir -p results/clip_features

python Feature-extraction/clip_feature_extraction.py \
  --images data/images \
  --output results/clip_features \
  --model ViT-B-16
```

#### 4.2.2 提取YOLOv8特征

```bash
mkdir -p results/yolo_features

python Feature-extraction/yolo_feature_extraction.py \
  --model models/yolov8x.pt \
  --images data/images \
  --output results/yolo_features \
  --conf 0.25
```

#### 4.2.3 手动融合特征

如果需要手动融合特征，可以使用以下命令：

```bash
python evaluation/fusion_features.py \
  --clip results/clip_features \
  --yolo results/yolo_features \
  --output results/fused_features \
  --method concat \
  --visualize
```

## 5. 使用GAN生成故障图像(可选)

如果需要使用GAN模块生成故障图像：

### 5.1 训练GAN模型

```bash
python gan_fault_generation.py train \
  --normal data/images/transformer \
  --output models/gan/transformer \
  --epochs 50 \
  --batch-size 4
```

### 5.2 生成故障图像

```bash
python gan_fault_generation.py generate \
  --model models/gan/transformer/generator_final.pth \
  --input data/images/transformer \
  --output data/faults/transformer \
  --variations 3
```

### 5.3 可视化对比正常图像和故障图像

```bash
python gan_fault_generation.py visualize \
  --normal data/images/transformer \
  --gen-fault data/faults/transformer \
  --output results/gan_visualization/transformer_comparison.png
```

### 5.4 使用GAN图像进行特征提取

```bash
python run_multimodal_pipeline.py \
  --images data/images/transformer \
  --output results/with_gan_test \
  --yolo-model models/yolov8x.pt \
  --clip-model ViT-B-16 \
  --use-gan \
  --gan-model models/gan/transformer/generator_final.pth \
  --fusion concat \
  --visualize
```

## 6. 结果验证

### 6.1 查看提取的特征

```bash
# 检查CLIP特征
ls -la results/baseline_test/clip_features
# 查看特征维度
python -c "import numpy as np; print(np.load('results/baseline_test/clip_features/transformer_img001.npy').shape)"

# 检查YOLO特征
ls -la results/baseline_test/yolo_features

# 检查融合特征
ls -la results/baseline_test/fused_features
```

### 6.2 查看可视化结果

```bash
# 查看特征可视化结果
ls -la results/baseline_test/*.png
```

### 6.3 特征文件命名规则

提取的特征文件会以原始图像文件名为基础，以`.npy`为扩展名保存。例如:

- 原始图像: `transformer/img001.jpg`
- CLIP特征: `clip_features/transformer_img001.npy`
- YOLO特征: `yolo_features/transformer_img001.npy`
- 融合特征: `fused_features/transformer_img001_fused.npy`

## 7. 常见问题排解

### 7.1 内存不足

- 降低批处理大小: 修改 `--batch-size` 参数为较小值
- 处理较小的图像: 使用 `--image-size` 参数降低处理分辨率

### 7.2 CUDA相关错误

- 检查CUDA版本是否匹配: `nvcc --version` 和 `pip list | grep torch`
- 尝试在CPU模式下运行: `CUDA_VISIBLE_DEVICES='' python run_multimodal_pipeline.py ...`

### 7.3 预训练模型下载失败

- 如果自动下载失败，可以手动下载模型并放置在正确的目录中
- 对于Chinese-CLIP模型，可参考[官方仓库](https://github.com/OFA-Sys/Chinese-CLIP)手动下载

## 8. 后续步骤

### 8.1 分析特征效果

- 检查t-SNE和PCA可视化结果，观察不同类别的分离情况
- 使用提取的特征训练简单的分类器验证效果

### 8.2 调整参数优化性能

- 调整YOLO的置信度阈值 (`--conf`)
- 尝试不同的特征融合方法 (`--fusion`)
- 尝试不同的中文CLIP模型变体 (`--clip-model`)

## 9. 参考文献

1. Chinese-CLIP: [https://github.com/OFA-Sys/Chinese-CLIP](https://github.com/OFA-Sys/Chinese-CLIP)
2. YOLOv8: [https://github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)
3. 生成对抗网络(GAN): [https://arxiv.org/abs/1406.2661](https://arxiv.org/abs/1406.2661)

## 10. 维护与支持

如有问题或建议，请在项目仓库提交Issue或联系开发团队。

---

祝您使用愉快! 
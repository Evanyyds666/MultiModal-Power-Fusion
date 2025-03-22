# 图像特征验证指南

本指南帮助您快速验证多模态融合系统中的图像特征提取和处理流程，重点关注Chinese-CLIP和YOLOv8模块。

## 目录

- [环境要求](#环境要求)
- [数据准备](#数据准备)
- [快速开始](#快速开始)
- [特征提取](#特征提取)
- [特征融合](#特征融合)
- [模型评估](#模型评估)
- [常见问题](#常见问题)

## 环境要求

在开始验证之前，确保您的系统满足以下要求：

- Python 3.8+
- PyTorch 1.10+
- CUDA 11.3+（如果使用GPU）
- 依赖包：cn-clip, ultralytics, numpy, matplotlib, scikit-learn, tqdm

您可以使用以下命令安装所需依赖：

```bash
pip install torch torchvision torchaudio
pip install cn_clip ultralytics numpy matplotlib scikit-learn tqdm pandas
```

下载预训练模型：

```bash
# 下载Chinese-CLIP模型
mkdir -p models/cn-clip
# 下载地址可参考: https://github.com/OFA-Sys/Chinese-CLIP

# 下载YOLOv8模型
pip install ultralytics
python -c "from ultralytics import YOLO; YOLO('yolov8x.pt')"
```

## 数据准备

1. 创建数据目录结构：

```bash
mkdir -p data/images/normal/transformer data/images/normal/circuit_breaker
```

2. 将变压器和断路器的正常图像放入相应目录：

```bash
# 放置变压器图像
cp your_transformer_images/* data/images/normal/transformer/

# 放置断路器图像
cp your_circuit_breaker_images/* data/images/normal/circuit_breaker/
```

3. (可选) 如果有故障图像，可以放入特定目录：

```bash
mkdir -p data/images/fault/transformer data/images/fault/circuit_breaker
cp your_fault_transformer_images/* data/images/fault/transformer/
cp your_fault_circuit_breaker_images/* data/images/fault/circuit_breaker/
```

## 快速开始

以下是快速验证系统的步骤：

1. **准备数据**
   按照[数据准备](#数据准备)部分的说明设置数据目录。

2. **运行多模态管道**：

```bash
python run_multimodal_pipeline.py \
    --image_path data/images/normal \
    --output_dir results/baseline \
    --clip_model_name ViT-B-16 \
    --extract_clip \
    --extract_yolo \
    --visualize
```

3. **检查结果**：

```bash
ls -la results/baseline/
```

## 特征提取

### CLIP特征提取

中文CLIP特征提取支持单个图像或图像目录：

```bash
python Feature-extraction/clip_feature_extraction.py \
    --model_name ViT-B-16 \
    --image_path data/images/normal \
    --output_dir results/clip_features \
    --batch_size 32
```

### YOLO特征提取

YOLO特征提取用于获取对象检测特征：

```bash
python Feature-extraction/yolo_feature_extraction.py \
    --model_path models/yolov8x.pt \
    --image_path data/images/normal \
    --output_dir results/yolo_features \
    --conf 0.25 \
    --batch_size 16
```

### 生成GAN故障图像（可选）

如果需要生成故障图像进行测试：

```bash
# 训练GAN模型
python gan_fault_generation.py train \
    --normal_images data/images/normal/transformer \
    --output_dir models/gan/transformer \
    --epochs 100 \
    --batch_size 16

# 生成故障图像
python gan_fault_generation.py generate \
    --normal_images data/images/normal/transformer \
    --output_dir data/images/generated/transformer \
    --model_path models/gan/transformer \
    --num_images 20 \
    --severity_levels 3

# 可视化比较
python gan_fault_generation.py visualize-comparison \
    --normal_images data/images/normal/transformer \
    --generated_images data/images/generated/transformer \
    --output_dir results/visualization
```

## 特征融合

为了融合不同类型的特征，使用融合脚本：

```bash
python evaluation/fusion_features.py \
    --clip results/clip_features \
    --yolo results/yolo_features \
    --output results/fused_features \
    --method concat \
    --visualize
```

融合方法选项：
- `concat`: 特征连接（默认）
- `average`: 特征平均
- `weighted`: 加权特征融合

## 模型评估

评估融合特征的性能：

```bash
python evaluation/evaluate_model.py \
    --feature_dir results/fused_features \
    --model svm \
    --test_size 0.2 \
    --output_dir results/evaluation \
    --visualize
```

模型选项：
- `svm`: 支持向量机（默认）
- `rf`: 随机森林
- `mlp`: 多层感知器

评估结果将显示在控制台并保存到指定的输出目录。

## 常见问题

### 1. 特征提取失败或报错

请检查：
- 图像路径是否正确
- 图像格式是否支持（jpg, png, bmp等）
- 模型文件是否已正确下载

### 2. 内存不足

对于大量或高分辨率图像：
- 减小批处理大小（`--batch_size`）
- 处理前调整图像大小
- 使用多次运行处理不同子集

### 3. 特征融合时出现维度不匹配

确保所有特征输入使用相同的数据集和命名约定。如果使用不同来源的特征，可能需要手动对齐文件名。

### 4. 可视化结果不清晰

如果t-SNE或PCA可视化效果不佳：
- 尝试调整特征选择
- 增加训练迭代次数
- 对特征进行归一化处理

---

若有任何问题，请参阅相应模块的详细文档或提交Issue。 
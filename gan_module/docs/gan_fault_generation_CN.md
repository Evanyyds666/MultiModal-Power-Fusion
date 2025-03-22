# GAN故障图像生成模块使用指南

## 概述

GAN故障图像生成模块是MultiModal-Power-Fusion项目的重要组成部分，旨在解决变电站设备故障数据稀缺的问题。通过基于生成对抗网络(GAN)的方法，该模块可以从正常设备图像生成逼真的故障图像，以增强数据集并提高故障检测模型的性能。

## 功能特点

- **无需真实故障图像**：只需提供正常设备图像即可训练模型
- **可调节故障严重程度**：支持生成不同严重程度的故障图像
- **批量处理**：支持批量生成大量故障图像用于数据集构建
- **多设备支持**：适用于变压器、断路器等多种变电站设备
- **可视化工具**：提供直观的故障图像对比和可视化功能

## 工作原理

GAN故障图像生成模块基于条件生成对抗网络的架构，包含以下核心组件：

1. **生成器(Generator)**：学习将正常设备图像转换为故障图像
2. **判别器(Discriminator)**：学习区分真实故障图像和生成的故障图像
3. **损失函数**：包括对抗损失、内容保持损失等，指导模型生成逼真的故障图像

该模块通过非监督学习方式进行训练，不需要成对的正常-故障图像数据，只需要正常设备图像即可。

## 安装和依赖

本模块已包含在主项目中，只需确保安装了以下依赖：

- PyTorch >= 1.8.0
- torchvision >= 0.9.0
- numpy
- matplotlib
- pillow
- tqdm

## 基本使用流程

### 1. 准备数据

将正常设备图像放入专用目录，例如：
```
data/
  └── images/
      └── normal/
          ├── transformer/  # 变压器正常图像
          └── circuit_breaker/  # 断路器正常图像
```

### 2. 训练GAN模型

使用`train`命令训练对应设备类型的GAN模型：

```bash
python gan_fault_generation.py train \
  --normal data/images/normal/transformer \
  --output models/gan/transformer \
  --epochs 200 \
  --batch-size 8 \
  --lr 0.0002
```

### 3. 生成故障图像

使用训练好的模型生成故障图像：

```bash
python gan_fault_generation.py generate \
  --model models/gan/transformer/generator_final.pth \
  --input data/images/normal/transformer \
  --output data/images/generated_faults/transformer \
  --variations 5 \
  --severity 0.7
```

### 4. 可视化比较

比较正常图像和生成的故障图像：

```bash
python gan_fault_generation.py visualize \
  --normal data/images/normal/transformer \
  --gen-fault data/images/generated_faults/transformer \
  --output results/comparisons/transformer_comparison.png \
  --samples 3
```

### 5. 创建图像蒙太奇

创建故障图像蒙太奇，便于整体查看：

```bash
python gan_fault_generation.py montage \
  --input data/images/generated_faults/transformer \
  --output results/montages/transformer_montage.png \
  --rows 4 \
  --cols 6
```

### 6. 批量生成数据集

批量生成故障图像数据集：

```bash
python gan_fault_generation.py batch-generate \
  --model models/gan/transformer/generator_final.pth \
  --input data/images/normal/transformer \
  --output datasets/fault_dataset \
  --variations 10 \
  --min-severity 0.3 \
  --max-severity 0.9
```

## 高级用法

### 调整故障严重程度

通过`--severity`参数调整生成故障的严重程度（范围0-1）：

```bash
python gan_fault_generation.py generate \
  --model models/gan/transformer/generator_final.pth \
  --input data/images/normal/transformer/image1.jpg \
  --output results/severity_test \
  --variations 1 \
  --severity 0.3 0.5 0.7 0.9
```

### 添加随机噪声

通过`--add-noise`参数为生成的故障图像添加随机噪声，增加多样性：

```bash
python gan_fault_generation.py generate \
  --model models/gan/transformer/generator_final.pth \
  --input data/images/normal/transformer \
  --output data/images/generated_faults/transformer_noisy \
  --variations 3 \
  --add-noise
```

### 可视化不同严重程度的效果

比较不同严重程度下的故障图像效果：

```bash
python gan_fault_generation.py visualize-severity \
  --model models/gan/transformer/generator_final.pth \
  --input data/images/normal/transformer/image1.jpg \
  --output results/severity_comparison.png \
  --levels 0.2 0.4 0.6 0.8 1.0
```

## 参数详解

### 训练参数

| 参数 | 描述 | 默认值 |
|------|------|--------|
| `--normal` | 正常图像目录路径 | 必填 |
| `--fault` | 真实故障图像目录路径（可选） | None |
| `--output` | 模型输出目录 | 必填 |
| `--epochs` | 训练轮数 | 100 |
| `--batch-size` | 批次大小 | 4 |
| `--lr` | 学习率 | 0.0002 |
| `--image-size` | 图像大小 | 256 |
| `--content-weight` | 内容损失权重 | 10.0 |
| `--adv-weight` | 对抗损失权重 | 1.0 |

### 生成参数

| 参数 | 描述 | 默认值 |
|------|------|--------|
| `--model` | 生成器模型路径 | 必填 |
| `--input` | 输入图像或目录 | 必填 |
| `--output` | 输出目录 | 必填 |
| `--variations` | 每张输入图像生成的变体数量 | 3 |
| `--severity` | 故障严重程度（0-1） | 0.7 |
| `--image-size` | 图像大小 | 256 |
| `--add-noise` | 是否添加随机噪声 | False |

### 可视化参数

| 参数 | 描述 | 默认值 |
|------|------|--------|
| `--normal` | 正常图像目录 | 必填 |
| `--gen-fault` | 生成的故障图像目录 | 必填 |
| `--real-fault` | 真实故障图像目录（可选） | None |
| `--output` | 输出图像路径 | 必填 |
| `--samples` | 要可视化的样本数量 | 3 |
| `--image-size` | 图像大小 | 256 |

## 常见问题

### Q: 如何判断生成的故障图像质量？

**A**: 可以通过以下方式评估生成图像质量：
1. 使用`visualize`命令对比正常和生成的故障图像
2. 咨询领域专家评估生成图像的真实性
3. 在下游任务(如故障检测)中测试生成图像的有效性

### Q: 需要多少正常图像才能训练出好的模型？

**A**: 建议至少提供20-50张同类设备的正常图像。图像数量越多，生成质量通常越好。图像应该覆盖不同角度和光照条件。

### Q: 训练时间需要多久？

**A**: 训练时间取决于数据量、图像大小和硬件配置。在中端GPU上，通常需要1-3小时完成100轮训练。如果没有GPU，训练时间会显著增加。

### Q: 如何提高生成图像的质量？

**A**: 可以尝试以下方法：
1. 增加训练数据量和多样性
2. 延长训练时间(增加epochs)
3. 调整模型超参数，如学习率和损失权重
4. 尝试不同的图像预处理方法

## 高级定制

### 自定义生成器和判别器

如果默认的GAN架构不满足需求，可以通过修改`gan_module/models.py`文件自定义网络结构。

### 自定义损失函数

可以在`gan_module/train.py`中修改损失函数的计算方式和权重，以适应特定的故障类型。

### 集成到自动化流水线

以下是将GAN故障生成集成到自动化流水线的示例脚本：

```python
import os
from gan_module.train import train_gan
from gan_module.generate import generate_fault_images

# 1. 训练模型
train_gan(
    normal_dir="data/images/normal/transformer",
    output_dir="models/gan/transformer",
    epochs=150,
    batch_size=8
)

# 2. 生成故障图像
generate_fault_images(
    model_path="models/gan/transformer/generator_final.pth",
    input_dir="data/images/normal/transformer",
    output_dir="data/images/generated_faults/transformer",
    variations=5,
    severity=0.7
)

# 3. 用生成的故障图像训练故障检测模型
# ... 接入故障检测模型训练代码
```

## 快速入门

我们提供了快速入门脚本，帮助新用户快速上手：

```bash
# 设置示例数据结构
python gan_quickstart.py setup

# 训练GAN模型
python gan_quickstart.py train

# 生成故障图像
python gan_quickstart.py generate

# 运行完整流程
python gan_quickstart.py all
```

详细的快速入门指南请参考 [快速入门](quickstart.md)。

## 参考资料

- [Pix2Pix: Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/abs/1611.07004)
- [CycleGAN: Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593)
- [UNIT: Unsupervised Image-to-Image Translation Networks](https://arxiv.org/abs/1703.00848)

## 维护者

[王德欣] ([2754702166@qq.com])

---

如有问题或建议，请提交Issue或Pull Request到项目仓库。 
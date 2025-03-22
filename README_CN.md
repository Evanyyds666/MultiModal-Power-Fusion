# MultiModal-Power-Fusion

## 项目简介

MultiModal-Power-Fusion 是一个针对变电站设备的多模态数据融合项目，旨在通过融合图像、声音和文本等多模态数据，提高设备类型识别和故障检测的准确性和可靠性。本项目包含一系列特征提取模块、数据融合算法以及基于GAN的故障图像生成工具。

## 特性

- 多模态融合：结合图像、声音和文本数据进行更全面的设备分析
- 模块化设计：各个特征提取和融合组件高度模块化，方便扩展
- GAN故障生成：基于正常设备图像自动生成各类故障图像，解决数据不平衡问题
- 跨模态学习：利用不同模态之间的互补信息提高识别和检测性能

## 安装

### 环境要求

- Python 3.8+
- CUDA 11.4+ (GPU加速推荐)
- 依赖库 (详见 requirements.txt)

### 安装步骤

1. 克隆项目仓库
   ```bash
   git clone https://github.com/your-username/MultiModal-Power-Fusion.git
   cd MultiModal-Power-Fusion
   ```

2. 安装依赖
   ```bash
   pip install -r requirements.txt
   ```

3. 准备数据目录
   ```bash
   mkdir -p data/{images,audio,text}
   ```

## 模块说明

### 图像特征提取

图像特征提取模块支持使用多种预训练模型对设备图像进行特征提取：

- CN-CLIP：利用中文CLIP模型提取图像特征，支持与文本描述进行匹配
- YOLO：用于设备目标检测和区域定位
- ResNet/DenseNet：用于提取深层图像特征

详细用法请参考 [图像特征提取文档](docs/image_feature_extraction.md)

### 声音特征提取

声音特征提取模块支持处理设备运行时录制的声音：

- 支持提取MFCC、梅尔谱、声谱图等声音特征
- 提供降噪和声音分段功能
- 支持使用预训练的音频模型进行特征提取

详细用法请参考 [声音特征提取文档](docs/audio_feature_extraction.md)

### 文本特征提取

文本特征提取模块处理设备说明书、维修记录等文本信息：

- 支持中文设备描述文本处理
- 利用预训练语言模型提取文本特征
- 支持专业词汇和领域知识的文本嵌入

详细用法请参考 [文本特征提取文档](docs/text_feature_extraction.md)

### 多模态融合

多模态融合模块将各种模态的特征进行融合，提供多种融合策略：

- 特征级融合：将不同模态特征直接连接或加权融合
- 决策级融合：对各模态的决策结果进行投票或加权
- 交叉注意力融合：利用注意力机制进行模态间信息交互

详细用法请参考 [多模态融合文档](docs/multimodal_fusion.md)

### GAN故障图像生成

GAN故障图像生成模块可以基于正常设备图像生成各类故障图像：

- 支持多种设备类型的故障生成
- 可调节故障严重程度
- 提供批量生成和可视化工具

#### GAN模块使用流程

1. **准备数据**：将正常设备图像放入指定目录
2. **训练模型**：使用正常图像训练GAN模型
3. **生成故障**：利用训练好的模型生成不同严重程度的故障图像
4. **可视化**：对比正常图像和生成的故障图像

#### 快速入门示例

```bash
# 创建示例数据结构
python gan_quickstart.py setup

# 训练GAN模型
python gan_quickstart.py train

# 生成故障图像
python gan_quickstart.py generate

# 运行完整流程
python gan_quickstart.py all
```

详细用法请参考 [GAN故障生成文档](docs/gan_fault_generation.md)

## 使用示例

### 基本使用流程

1. **数据准备**：
   ```bash
   # 将设备图像放入图像目录
   cp /path/to/your/images/* data/images/
   
   # 将设备声音文件放入声音目录
   cp /path/to/your/audio/* data/audio/
   
   # 将设备文本描述放入文本目录
   cp /path/to/your/text/* data/text/
   ```

2. **特征提取**：
   ```bash
   # 提取图像特征
   python extract_image_features.py --input data/images --output features/image_features
   
   # 提取声音特征
   python extract_audio_features.py --input data/audio --output features/audio_features
   
   # 提取文本特征
   python extract_text_features.py --input data/text --output features/text_features
   ```

3. **融合与预测**：
   ```bash
   # 融合多模态特征并进行预测
   python fusion_prediction.py --image-features features/image_features \
                              --audio-features features/audio_features \
                              --text-features features/text_features \
                              --output results/fusion_results.json
   ```

4. **生成故障图像**：
   ```bash
   # 训练GAN模型
   python gan_fault_generation.py train --normal data/images/normal \
                                      --output models/gan_model \
                                      --epochs 100
   
   # 生成故障图像
   python gan_fault_generation.py generate --model models/gan_model/generator_final.pth \
                                          --input data/images/normal \
                                          --output data/images/generated_faults \
                                          --variations 5
   ```

### 高级用法

详细的高级用法和参数设置请参考 [高级用法文档](docs/advanced_usage.md)

## 数据集构建指南

### 图像数据集

- **数量要求**：每种设备类型建议至少收集20-50张图像
- **拍摄要求**：不同角度、不同光照条件下的设备图像
- **标注要求**：设备类型、故障类型（如有）

### 声音数据集

- **数量要求**：每种设备类型建议至少收集10-20段声音
- **录制要求**：设备正常运行和故障状态下的声音
- **标注要求**：设备类型、运行状态、环境噪声水平

### 文本数据集

- **内容要求**：设备说明书、维修记录、故障描述等
- **格式要求**：结构化文本，包含设备信息和故障信息
- **标注要求**：关键信息提取和分类

详细的数据集构建指南请参考 [数据集构建文档](docs/dataset_construction.md)

## 贡献指南

我们欢迎各种形式的贡献，包括但不限于：

- 提交bug和功能请求
- 改进代码和文档
- 添加新的特征提取方法
- 优化融合算法
- 提供测试数据和验证结果

详细的贡献指南请参考 [贡献指南文档](docs/contributing.md)

## 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

## 联系方式

- 项目维护者：[王德欣]
- 电子邮件：[2754702166@qq.com]
- 项目网站：[项目网站URL] 
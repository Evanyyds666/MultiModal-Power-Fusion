# 多模态特征提取和融合项目

这个项目结合了Chinese-CLIP、YOLOv8和Wav2Vec2模型来提取和融合图像、文本和音频的多模态特征，专注于电网业务环境下变电站一次设备的类型识别和故障检测。

## 功能特点

- 使用Chinese-CLIP提取图像的语义特征和文本特征
- 使用YOLOv8提取图像的目标检测特征
- 使用Wav2Vec2或传统方法提取音频特征
- 支持多种特征融合方法（拼接、平均、加权）
- 提供特征可视化工具
- 灵活支持单模态或多模态特征处理

## 安装

```bash
# 克隆项目
git clone https://github.com/Evanyyds666/MultiModal-Power-Fusion.git
cd MultiModal-Power-Fusion

# 安装依赖
pip install -r requirements.txt
```

## 下载预训练模型

项目使用的模型会在需要时自动下载，但您也可以手动预先下载：

```bash
# 下载YOLOv8模型
mkdir -p models
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt -O models/yolov8x.pt

# 其他模型（如Chinese-CLIP和Wav2Vec2）会在运行时自动下载
```

## 数据集准备指南

### 变电站设备多模态数据集构建

为了快速构建高质量的baseline，建议按以下规模准备数据：

#### 图像数据
- **数量**：每类设备至少30-50张图片（共约10-15类设备）
- **标注**：对于设备类型识别，只需分类标签；对于故障检测，需框定故障位置
- **格式**：JPG或PNG，分辨率建议≥1024x768
- **采集建议**：不同角度、不同光照条件、不同天气下的设备照片

#### 音频数据
- **数量**：每类设备/故障约20-30段音频（每段3-10秒）
- **格式**：WAV或MP3，采样率≥16kHz
- **采集建议**：在不同环境噪声条件下录制设备正常运行和故障状态声音

#### 文本数据
- **数量**：每类设备约15-20条描述文本
- **内容**：包括设备描述、典型故障描述、诊断报告等
- **格式**：UTF-8编码的TXT文件

### 组织结构
```
dataset/
├── images/
│   ├── transformer/
│   ├── circuit_breaker/
│   └── ...
├── audio/
│   ├── normal/
│   ├── fault_type1/
│   └── ...
└── text/
    ├── descriptions/
    └── fault_reports/
```

## 使用方法

### 完整多模态流水线

运行完整的多模态特征处理流水线：

```bash
python run_multimodal_pipeline.py --images examples/images --audio examples/audio --texts examples/text --output results
```

参数说明：
- `--images`: 输入图像或图像目录（可选）
- `--audio`: 输入音频或音频目录（可选）
- `--texts`: 输入文本或文本目录（可选）
- `--output`: 输出目录（默认：multimodal_results）
- `--clip-model`: CLIP模型名称（默认：ViT-B-16）
- `--yolo-model`: YOLOv8模型路径（默认：models/yolov8x.pt）
- `--audio-model`: 音频模型名称（默认：facebook/wav2vec2-base-960h）
- `--fusion`: 特征融合方法，可选"concat"、"average"、"weighted"（默认：concat）
- `--visualize`: 是否可视化融合特征
- `--traditional-audio`: 是否使用传统方法提取音频特征（MFCC、色度、梅尔频谱）

### 单独组件

#### CLIP图像特征提取

```bash
python clip_feature_extraction.py --images examples/images --output clip_features
```

#### YOLO图像特征提取（通过pipeline脚本）

```bash
python run_multimodal_pipeline.py --images examples/images --output yolo_features --yolo-model models/yolov8x.pt
```

#### 文本特征提取

```bash
python text_feature_extraction.py --texts examples/text --output text_features
```

#### 音频特征提取

```bash
# 使用深度学习模型提取
python audio_feature_extraction.py --audio examples/audio --output audio_features

# 使用传统方法提取
python audio_feature_extraction.py --audio examples/audio --output audio_features --traditional
```

#### 特征融合

```bash
python run_multimodal_pipeline.py --images examples/images --audio examples/audio --fusion weighted --output fused_features
```

#### 特征可视化

```bash
python evaluation/visualize_features.py --features results/fused_features --output visualization.png
```

## 知识图谱构建指南

### 基于多模态特征的知识图谱

1. **实体定义**：基于设备类型、部件、故障类型等定义实体
2. **关系定义**：定义实体间关系（如"部分-整体"、"故障-原因"等）
3. **特征关联**：将提取的多模态特征与知识图谱实体关联
4. **跨模态对齐**：使用融合特征进行跨模态实体对齐

```bash
# 示例：使用融合特征进行实体对齐
python knowledge_graph/entity_alignment.py --features results/fused_features --output kg_alignment
```

## 项目结构

```
MultiModal-Power-Fusion/
├── clip_feature_extraction.py      # CLIP特征提取脚本
├── audio_feature_extraction.py     # 音频特征提取脚本
├── text_feature_extraction.py      # 文本特征提取脚本
├── run_multimodal_pipeline.py      # 多模态流水线
├── evaluation/
│   ├── feature_fusion.py           # 特征融合工具
│   └── visualize_features.py       # 特征可视化工具
├── models/
│   └── yolov8x.pt                  # YOLOv8模型
├── examples/
│   ├── images/                     # 示例图像
│   ├── audio/                      # 示例音频
│   └── text/                       # 示例文本
├── utils/
│   ├── data_preparation.py         # 数据准备工具
│   └── model_utils.py              # 模型工具函数
└── knowledge_graph/                # 知识图谱构建工具
    ├── entity_extraction.py        # 实体提取
    └── entity_alignment.py         # 实体对齐
```

## 团队协作指南

### 环境设置

每个团队成员应按照以下步骤设置开发环境：

1. 克隆代码库并安装依赖
```bash
git clone https://github.com/Evanyyds666/MultiModal-Power-Fusion.git
cd MultiModal-Power-Fusion
pip install -r requirements.txt
```

2. 下载预训练模型（可选，也可在运行时自动下载）
```bash
mkdir -p models
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt -O models/yolov8x.pt
```

3. 验证安装
```bash
# 测试基本功能
python run_multimodal_pipeline.py --images examples/images --output test_results
```

### 开发职责分工

建议的团队分工：

1. **图像特征工程师**：负责改进CLIP和YOLO特征提取，优化图像处理流程
2. **音频/文本特征工程师**：负责音频特征和文本特征的提取与优化
3. **融合与知识图谱工程师**：负责特征融合策略和知识图谱构建

### 代码贡献流程

1. 创建功能分支
2. 提交代码并添加测试
3. 提交合并请求并进行代码审查
4. 合并到主分支

## 引用

如果您使用了本项目的代码，请引用以下项目：

- [Chinese-CLIP](https://github.com/OFA-Sys/Chinese-CLIP)
- [YOLOv8](https://github.com/ultralytics/ultralytics)
- [Wav2Vec2](https://github.com/pytorch/fairseq/tree/main/examples/wav2vec)

## 许可证

MIT 
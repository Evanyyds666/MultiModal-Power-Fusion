#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 故障图像生成模块

import os
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import numpy as np
from .models import UNetGenerator
from .data_utils import set_seed

def generate_fault_images(model_path, input_dir, output_dir, num_variations=3, image_size=256, 
                          severity=None, add_noise=True):
    """
    使用训练好的生成器生成故障图像
    
    参数:
        model_path: 生成器模型路径
        input_dir: 输入正常图像目录
        output_dir: 输出故障图像目录
        num_variations: 每张图生成的变体数量
        image_size: 图像大小
        severity: 故障严重程度 (0-1.0，None表示随机)
        add_noise: 是否添加随机噪声产生变化
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 设置随机种子
    set_seed()
    
    # 加载模型
    generator = UNetGenerator().to(device)
    generator.load_state_dict(torch.load(model_path, map_location=device))
    generator.eval()
    
    # 图像转换
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取输入图像
    image_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print(f"警告: 在 {input_dir} 中没有找到图像文件")
        return
    
    print(f"找到 {len(image_files)} 张图像。正在生成故障变体...")
    
    for img_path in tqdm(image_files, desc="生成故障图像"):
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        
        # 加载图像
        try:
            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img).unsqueeze(0).to(device)
            
            # 生成多个变体
            for i in range(num_variations):
                with torch.no_grad():
                    # 添加随机噪声以获得变体
                    noise = torch.zeros_like(img_tensor)
                    if add_noise:
                        noise_level = 0.05 if severity is None else max(0.01, severity * 0.1)
                        noise = torch.randn_like(img_tensor) * noise_level
                    
                    # 生成故障图像
                    fake_fault = generator(img_tensor + noise)
                    
                    # 如果指定了严重程度，使用原图和生成图像的线性插值
                    if severity is not None:
                        fake_fault = img_tensor * (1 - severity) + fake_fault * severity
                    
                    # 转回图像
                    fake_fault = fake_fault.squeeze(0).cpu().detach()
                    fake_fault = (fake_fault * 0.5) + 0.5
                    fake_fault = fake_fault.permute(1, 2, 0).numpy()
                    
                    # 保存图像
                    plt.figure(figsize=(10, 10))
                    plt.imshow(fake_fault)
                    plt.axis('off')
                    
                    output_path = os.path.join(output_dir, f"{img_name}_fault_var{i+1}.png")
                    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
                    plt.close()
        except Exception as e:
            print(f"处理图像 {img_path} 时出错: {e}")
    
    print(f"故障图像生成完成，已保存到 {output_dir}")

def generate_fault_dataset(model_path, normal_dir, output_base_dir, variations_per_image=3, severity_levels=None):
    """
    生成完整的故障数据集，包括不同类型的设备
    
    参数:
        model_path: 生成器模型路径（可以是目录或特定模型）
        normal_dir: 正常设备图像根目录（包含子目录）
        output_base_dir: 输出根目录
        variations_per_image: 每张图像的变体数量
        severity_levels: 严重程度级别列表，如 [0.3, 0.7, 1.0]
    """
    # 设置随机种子
    set_seed()
    
    if severity_levels is None:
        severity_levels = [None]  # 使用模型默认
    
    # 检查输入是目录还是文件
    if os.path.isdir(model_path):
        # 查找目录中的所有.pth文件
        model_files = [os.path.join(model_path, f) for f in os.listdir(model_path) 
                     if f.endswith('.pth') and f.startswith('generator')]
        if not model_files:
            print(f"在 {model_path} 中未找到生成器模型文件")
            return
        model_file = model_files[0]  # 使用第一个找到的模型文件
    else:
        model_file = model_path
    
    print(f"使用模型: {model_file}")
    
    # 检查normal_dir是否包含子目录（不同类别）
    if not os.path.isdir(normal_dir):
        print(f"错误: {normal_dir} 不是有效目录")
        return
    
    subdirs = [d for d in os.listdir(normal_dir) if os.path.isdir(os.path.join(normal_dir, d))]
    
    if subdirs:
        # 如果有子目录，为每个子目录生成故障图像
        print(f"找到 {len(subdirs)} 个设备类别: {', '.join(subdirs)}")
        for subdir in subdirs:
            input_dir = os.path.join(normal_dir, subdir)
            
            for severity in severity_levels:
                severity_str = f"_sev{severity}" if severity is not None else ""
                output_dir = os.path.join(output_base_dir, f"{subdir}{severity_str}")
                
                print(f"为 {subdir} 生成故障图像，严重程度: {severity if severity is not None else '默认'}")
                generate_fault_images(
                    model_file, 
                    input_dir, 
                    output_dir, 
                    num_variations=variations_per_image,
                    severity=severity
                )
    else:
        # 如果没有子目录，直接生成
        for severity in severity_levels:
            severity_str = f"_sev{severity}" if severity is not None else ""
            output_dir = os.path.join(output_base_dir, f"fault{severity_str}")
            
            print(f"生成故障图像，严重程度: {severity if severity is not None else '默认'}")
            generate_fault_images(
                model_file, 
                normal_dir, 
                output_dir, 
                num_variations=variations_per_image,
                severity=severity
            )
    
    print(f"故障数据集生成完成，已保存到 {output_base_dir}") 
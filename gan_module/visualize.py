#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 故障图像可视化模块

import os
import random
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from .data_utils import set_seed

def visualize_comparisons(normal_dir, gen_fault_dir, real_fault_dir=None, output_path="comparisons.png", 
                         num_samples=5, image_size=256):
    """
    生成对比可视化：正常设备 vs 生成的故障 vs 真实故障(如有)
    
    参数:
        normal_dir: 正常设备图像目录
        gen_fault_dir: 生成的故障图像目录
        real_fault_dir: 真实故障图像目录（可选）
        output_path: 输出图像路径
        num_samples: 采样数量
        image_size: 图像大小
    """
    # 设置随机种子
    set_seed()
    
    # 获取图像
    normal_images = [os.path.join(normal_dir, f) for f in os.listdir(normal_dir) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not normal_images:
        print(f"警告: 在 {normal_dir} 中没有找到图像文件")
        return
    
    # 随机采样
    if len(normal_images) > num_samples:
        normal_images = random.sample(normal_images, num_samples)
    
    # 创建可视化
    cols = 3 if real_fault_dir else 2
    fig, axs = plt.subplots(len(normal_images), cols, figsize=(cols*5, len(normal_images)*5))
    
    # 处理单行情况
    if len(normal_images) == 1:
        axs = [axs]
    
    for i, img_path in enumerate(normal_images):
        img_name = os.path.basename(img_path)
        img_base_name = os.path.splitext(img_name)[0]
        
        # 加载正常图像
        try:
            normal_img = Image.open(img_path).convert('RGB')
            normal_img = normal_img.resize((image_size, image_size), Image.LANCZOS)
            
            # 显示正常图像
            if len(normal_images) == 1 and cols == 1:
                axs[0].imshow(normal_img)
                axs[0].set_title('正常设备')
                axs[0].axis('off')
            else:
                axs[i][0].imshow(normal_img)
                axs[i][0].set_title('正常设备')
                axs[i][0].axis('off')
            
            # 查找生成的故障图像（假设命名格式匹配）
            gen_fault_path = None
            for f in os.listdir(gen_fault_dir):
                if f.startswith(img_base_name) and f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    gen_fault_path = os.path.join(gen_fault_dir, f)
                    break
            
            # 显示生成的故障图像
            if gen_fault_path:
                gen_fault_img = Image.open(gen_fault_path).convert('RGB')
                gen_fault_img = gen_fault_img.resize((image_size, image_size), Image.LANCZOS)
                
                if len(normal_images) == 1 and cols == 1:
                    axs[1].imshow(gen_fault_img)
                    axs[1].set_title('生成的故障')
                    axs[1].axis('off')
                else:
                    axs[i][1].imshow(gen_fault_img)
                    axs[i][1].set_title('生成的故障')
                    axs[i][1].axis('off')
            else:
                print(f"警告: 未找到对应的生成故障图像: {img_base_name}")
            
            # 查找真实故障图像（如果有）
            if real_fault_dir:
                real_fault_path = None
                for f in os.listdir(real_fault_dir):
                    if f.startswith(img_base_name) or f == img_name:
                        real_fault_path = os.path.join(real_fault_dir, f)
                        break
                
                # 显示真实故障图像
                if real_fault_path:
                    real_fault_img = Image.open(real_fault_path).convert('RGB')
                    real_fault_img = real_fault_img.resize((image_size, image_size), Image.LANCZOS)
                    
                    if len(normal_images) == 1:
                        axs[2].imshow(real_fault_img)
                        axs[2].set_title('真实故障')
                        axs[2].axis('off')
                    else:
                        axs[i][2].imshow(real_fault_img)
                        axs[i][2].set_title('真实故障')
                        axs[i][2].axis('off')
        except Exception as e:
            print(f"处理图像时出错: {e}")
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"对比可视化已保存到 {output_path}")

def visualize_severity_comparison(generator, input_image_path, output_path="severity_comparison.png", 
                                 severity_levels=None, image_size=256):
    """
    可视化不同严重程度的故障生成效果
    
    参数:
        generator: 加载好的生成器模型
        input_image_path: 输入正常图像路径
        output_path: 输出图像路径
        severity_levels: 严重程度级别列表，如 [0.1, 0.3, 0.5, 0.7, 1.0]
        image_size: 图像大小
    """
    import torch
    from torchvision import transforms
    
    if severity_levels is None:
        severity_levels = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    
    device = generator.parameters().__next__().device
    
    # 图像转换
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # 加载原始图像
    try:
        img = Image.open(input_image_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device)
        
        fig, axs = plt.subplots(1, len(severity_levels), figsize=(len(severity_levels)*4, 4))
        
        for i, severity in enumerate(severity_levels):
            with torch.no_grad():
                if severity == 0.0:
                    # 显示原始图像
                    img_to_show = img_tensor.clone()
                else:
                    # 生成故障图像
                    fake_fault = generator(img_tensor)
                    
                    # 线性插值
                    img_to_show = img_tensor * (1 - severity) + fake_fault * severity
                
                # 转回图像格式
                img_to_show = img_to_show.squeeze(0).cpu().detach()
                img_to_show = (img_to_show * 0.5) + 0.5
                img_to_show = img_to_show.permute(1, 2, 0).numpy()
                
                # 显示图像
                axs[i].imshow(img_to_show)
                axs[i].set_title(f'严重程度: {severity:.1f}')
                axs[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        print(f"严重程度对比已保存到 {output_path}")
    except Exception as e:
        print(f"可视化时出错: {e}")

def create_montage(input_dir, output_path, grid_size=(4, 4), image_size=256, random_select=True):
    """
    创建图像蒙太奇，显示多张图像的网格
    
    参数:
        input_dir: 输入图像目录
        output_path: 输出图像路径
        grid_size: 网格大小 (行, 列)
        image_size: 单张图像大小
        random_select: 是否随机选择图像
    """
    # 获取所有图像
    images = [os.path.join(input_dir, f) for f in os.listdir(input_dir) 
             if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not images:
        print(f"警告: 在 {input_dir} 中没有找到图像文件")
        return
    
    # 计算需要的图像数量
    num_images = grid_size[0] * grid_size[1]
    
    # 确保有足够的图像
    if len(images) < num_images:
        print(f"警告: 只找到 {len(images)} 张图像，少于请求的 {num_images} 张")
        # 重复图像以满足需求
        while len(images) < num_images:
            images += images[:num_images - len(images)]
    
    # 随机选择图像
    if random_select:
        selected_images = random.sample(images, num_images)
    else:
        selected_images = images[:num_images]
    
    # 创建画布
    fig, axs = plt.subplots(grid_size[0], grid_size[1], figsize=(grid_size[1]*4, grid_size[0]*4))
    
    # 填充图像
    for idx, img_path in enumerate(selected_images):
        row = idx // grid_size[1]
        col = idx % grid_size[1]
        
        try:
            img = Image.open(img_path).convert('RGB')
            img = img.resize((image_size, image_size), Image.LANCZOS)
            
            if grid_size[0] == 1 and grid_size[1] == 1:
                axs.imshow(img)
                axs.axis('off')
            elif grid_size[0] == 1:
                axs[col].imshow(img)
                axs[col].axis('off')
            elif grid_size[1] == 1:
                axs[row].imshow(img)
                axs[row].axis('off')
            else:
                axs[row, col].imshow(img)
                axs[row, col].axis('off')
        except Exception as e:
            print(f"处理图像 {img_path} 时出错: {e}")
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"图像蒙太奇已保存到 {output_path}") 
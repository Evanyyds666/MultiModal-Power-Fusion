#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# CLIP特征提取模块

import os
import argparse
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path

def extract_features(model, preprocess, images_path, output_dir, batch_size=32):
    """提取图像的CLIP特征
    
    Args:
        model: 加载的CLIP模型
        preprocess: CLIP预处理函数
        images_path: 图像或图像目录的路径
        output_dir: 输出特征的目录
        batch_size: 批处理大小
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 处理输入路径
    if os.path.isfile(images_path):
        image_files = [images_path]
    else:
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
        image_files = []
        
        # 递归遍历目录
        for root, _, files in os.walk(images_path):
            for file in files:
                if file.lower().endswith(image_extensions):
                    image_files.append(os.path.join(root, file))
    
    if not image_files:
        print(f"警告: 在 {images_path} 中没有找到图像文件")
        return
    
    # 分批处理图像
    for i in tqdm(range(0, len(image_files), batch_size), desc="提取CLIP特征"):
        batch_files = image_files[i:i+batch_size]
        
        # 加载和预处理图像
        batch_images = []
        valid_indices = []
        valid_files = []
        
        for idx, img_path in enumerate(batch_files):
            try:
                img = Image.open(img_path).convert("RGB")
                processed_img = preprocess(img)
                batch_images.append(processed_img)
                valid_indices.append(idx)
                valid_files.append(img_path)
            except Exception as e:
                print(f"处理图像 {img_path} 失败: {e}")
        
        if not batch_images:
            continue
        
        # 将图像堆叠成一个批次
        batch_tensor = torch.stack(batch_images).to(device)
        
        # 提取特征
        with torch.no_grad():
            image_features = model.encode_image(batch_tensor)
            # 归一化特征
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        # 保存特征
        for idx, img_path in enumerate(valid_files):
            # 计算相对路径，用于保持目录结构
            if os.path.isdir(images_path):
                rel_path = os.path.relpath(img_path, images_path)
                # 将目录分隔符替换为下划线
                rel_path = rel_path.replace(os.path.sep, "_")
            else:
                rel_path = os.path.basename(img_path)
            
            # 生成输出文件名
            feature_name = os.path.splitext(rel_path)[0] + ".npy"
            feature_path = os.path.join(output_dir, feature_name)
            
            # 保存为numpy格式
            np.save(feature_path, image_features[idx].cpu().numpy())
    
    print(f"特征提取完成，已保存到 {output_dir}")

def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(description="提取图像的CLIP特征")
    parser.add_argument("--model", type=str, default="ViT-B-16", 
                        help="CLIP模型名称，可选: ViT-B-16, ViT-L-14, RN50")
    parser.add_argument("--images", type=str, required=True, 
                        help="图像文件或包含图像的目录路径")
    parser.add_argument("--output", type=str, default="clip_features", 
                        help="保存特征的输出目录")
    parser.add_argument("--batch-size", type=int, default=32, 
                        help="处理图像的批次大小")
    
    args = parser.parse_args()
    
    try:
        from cn_clip.clip import load_from_name, available_models
        print(f"可用的CLIP模型: {', '.join(available_models())}")
    except ImportError:
        print("错误: 需要安装cn_clip包。请执行: pip install cn_clip")
        return
    
    print(f"加载CLIP模型: {args.model}")
    model, preprocess = load_from_name(args.model)
    
    print(f"处理图像: {args.images}")
    extract_features(model, preprocess, args.images, args.output, args.batch_size)

if __name__ == "__main__":
    main() 
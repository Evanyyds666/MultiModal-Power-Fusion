#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm

def normalize_feature(feature):
    """归一化特征向量"""
    norm = np.linalg.norm(feature)
    if norm > 0:
        return feature / norm
    return feature

def fuse_features(clip_dir, yolo_dir, output_dir, fusion_method='concat', clip_weight=0.5):
    """融合CLIP和YOLO特征
    
    Args:
        clip_dir: CLIP特征目录
        yolo_dir: YOLO特征目录
        output_dir: 融合特征输出目录
        fusion_method: 融合方法，可选 'concat', 'average', 'weighted'
        clip_weight: CLIP特征权重 (0-1)，仅当fusion_method='weighted'时使用
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有CLIP特征文件
    clip_files = [f for f in os.listdir(clip_dir) if f.endswith('.npy')]
    
    # 遍历并融合特征
    for clip_file in tqdm(clip_files, desc="融合特征"):
        base_name = clip_file
        
        # 获取对应的YOLO特征文件
        yolo_file = os.path.join(yolo_dir, base_name)
        clip_file_path = os.path.join(clip_dir, clip_file)
        
        # 检查YOLO特征是否存在
        if not os.path.exists(yolo_file):
            print(f"警告: 找不到YOLO特征 {yolo_file}，跳过")
            continue
        
        # 加载特征
        clip_feature = np.load(clip_file_path)
        yolo_feature = np.load(yolo_file)
        
        # 归一化特征
        clip_feature = normalize_feature(clip_feature)
        yolo_feature = normalize_feature(yolo_feature)
        
        # 融合特征
        if fusion_method == 'concat':
            # 拼接特征
            fused_feature = np.concatenate([clip_feature, yolo_feature])
            # 重新归一化
            fused_feature = normalize_feature(fused_feature)
        
        elif fusion_method == 'average':
            # 平均融合
            fused_feature = (clip_feature + yolo_feature) / 2
            # 重新归一化
            fused_feature = normalize_feature(fused_feature)
            
        elif fusion_method == 'weighted':
            # 加权融合
            yolo_weight = 1.0 - clip_weight
            fused_feature = clip_weight * clip_feature + yolo_weight * yolo_feature
            # 重新归一化
            fused_feature = normalize_feature(fused_feature)
            
        else:
            raise ValueError(f"不支持的融合方法: {fusion_method}")
        
        # 保存融合特征
        output_path = os.path.join(output_dir, base_name)
        np.save(output_path, fused_feature)
    
    print(f"特征融合完成，已保存到 {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="融合CLIP和YOLO特征")
    parser.add_argument("--clip-features", type=str, required=True,
                        help="CLIP特征目录")
    parser.add_argument("--yolo-features", type=str, required=True,
                        help="YOLO特征目录")
    parser.add_argument("--output", type=str, default="fused_features",
                        help="融合特征输出目录")
    parser.add_argument("--method", type=str, default="concat",
                        choices=["concat", "average", "weighted"],
                        help="特征融合方法")
    parser.add_argument("--clip-weight", type=float, default=0.5,
                        help="CLIP特征权重 (0-1)，仅在weighted方法中使用")
    
    args = parser.parse_args()
    
    fuse_features(
        args.clip_features, 
        args.yolo_features, 
        args.output, 
        args.method, 
        args.clip_weight
    )

if __name__ == "__main__":
    main() 
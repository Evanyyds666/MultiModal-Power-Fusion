#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# YOLO特征提取模块

import os
import torch
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm
from pathlib import Path

def extract_yolo_features(model, images_path, output_dir, conf_threshold=0.25):
    """提取图像的YOLO特征
    
    Args:
        model: 加载的YOLO模型
        images_path: 图像或图像目录的路径
        output_dir: 输出特征的目录
        conf_threshold: 置信度阈值
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 处理输入路径
    if os.path.isfile(images_path):
        image_files = [images_path]
    else:
        image_files = [os.path.join(images_path, f) for f in os.listdir(images_path) 
                      if f.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'webp'))]
    
    # 处理每张图像
    for img_path in tqdm(image_files, desc="提取YOLO特征"):
        try:
            # 使用YOLO进行目标检测
            results = model(img_path, conf=conf_threshold)
            
            # 提取特征
            features = []
            
            # 如果没有检测到任何目标，保存一个空特征向量
            if len(results[0].boxes) == 0:
                feature_vector = np.zeros(20)  # 使用默认大小
            else:
                # 提取每个检测框的信息
                for box in results[0].boxes:
                    # 获取边界框坐标
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    # 获取置信度
                    conf = box.conf[0].cpu().numpy()
                    
                    # 获取类别
                    cls = box.cls[0].cpu().numpy()
                    
                    # 计算宽度和高度
                    width = x2 - x1
                    height = y2 - y1
                    
                    # 计算中心点
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    
                    # 提取特征向量：[中心x, 中心y, 宽度, 高度, 置信度, 类别]
                    box_feature = np.array([center_x, center_y, width, height, conf, cls])
                    features.append(box_feature)
                
                # 将所有特征向量合并为一个特征向量
                # 排序，使最高置信度的框在前面
                sorted_features = sorted(features, key=lambda x: x[4], reverse=True)
                
                # 限制最大检测框数量
                max_boxes = 10
                if len(sorted_features) > max_boxes:
                    sorted_features = sorted_features[:max_boxes]
                
                # 将特征列表转换为NumPy数组
                feature_vector = np.concatenate(sorted_features)
            
            # 保存特征
            img_name = os.path.basename(img_path)
            feature_name = os.path.splitext(img_name)[0] + ".npy"
            feature_path = os.path.join(output_dir, feature_name)
            
            np.save(feature_path, feature_vector)
            
        except Exception as e:
            print(f"处理图像 {img_path} 时出错: {e}")
    
    print(f"YOLO特征提取完成，已保存到 {output_dir}")

def main():
    """命令行入口"""
    import argparse
    from ultralytics import YOLO
    
    parser = argparse.ArgumentParser(description="提取图像的YOLO特征")
    parser.add_argument("--model", type=str, default="yolov8n.pt", 
                        help="YOLO模型路径")
    parser.add_argument("--images", type=str, required=True, 
                        help="图像文件或包含图像的目录路径")
    parser.add_argument("--output", type=str, default="yolo_features", 
                        help="保存特征的输出目录")
    parser.add_argument("--conf", type=float, default=0.25, 
                        help="置信度阈值")
    
    args = parser.parse_args()
    
    print(f"加载YOLO模型: {args.model}")
    model = YOLO(args.model)
    
    print(f"处理图像: {args.images}")
    extract_yolo_features(model, args.images, args.output, args.conf)

if __name__ == "__main__":
    main() 
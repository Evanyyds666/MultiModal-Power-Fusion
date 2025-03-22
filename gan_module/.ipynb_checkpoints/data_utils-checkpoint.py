#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 数据处理工具

import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

def set_seed(seed=42):
    """设置随机种子以确保结果可复现
    
    Args:
        seed: 随机种子值
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    print(f"已设置随机种子为: {seed}")

class PowerEquipmentDataset(Dataset):
    """电力设备图像数据集"""
    
    def __init__(self, normal_dir, fault_dir=None, transform=None, paired=True):
        """
        参数:
            normal_dir (str): 正常设备图像目录
            fault_dir (str): 故障设备图像目录（如果有）
            transform: 图像变换
            paired (bool): 是否为配对数据集
        """
        self.transform = transform
        self.paired = paired
        
        # 获取所有正常图像文件
        self.normal_images = [os.path.join(normal_dir, f) for f in os.listdir(normal_dir) 
                            if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # 如果有故障图像且是配对数据集
        if fault_dir and paired:
            self.fault_images = [os.path.join(fault_dir, os.path.basename(f)) 
                              for f in self.normal_images 
                              if os.path.exists(os.path.join(fault_dir, os.path.basename(f)))]
            self.normal_images = [f for f in self.normal_images 
                               if os.path.exists(os.path.join(fault_dir, os.path.basename(f)))]
            print(f"找到 {len(self.fault_images)} 对配对图像。")
        # 如果有故障图像但不是配对数据集
        elif fault_dir:
            self.fault_images = [os.path.join(fault_dir, f) for f in os.listdir(fault_dir) 
                              if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            print(f"找到 {len(self.normal_images)} 张正常图像和 {len(self.fault_images)} 张故障图像。")
        else:
            self.fault_images = []
            print(f"找到 {len(self.normal_images)} 张正常图像。将使用自监督模式训练。")
            
    def __len__(self):
        return len(self.normal_images)
    
    def __getitem__(self, idx):
        normal_img_path = self.normal_images[idx]
        normal_image = Image.open(normal_img_path).convert('RGB')
        
        if self.fault_images and idx < len(self.fault_images):
            fault_img_path = self.fault_images[idx]
            fault_image = Image.open(fault_img_path).convert('RGB')
        else:
            # 如果没有对应的故障图像，创建一个模拟的故障图像
            fault_image = normal_image.copy()
            
        if self.transform:
            normal_image = self.transform(normal_image)
            fault_image = self.transform(fault_image)
            
        return {'normal': normal_image, 'fault': fault_image, 
                'normal_path': normal_img_path, 
                'fault_path': self.fault_images[idx] if self.fault_images and idx < len(self.fault_images) else ""} 
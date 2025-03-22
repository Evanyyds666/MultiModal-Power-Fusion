#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# GAN故障图像生成主入口脚本

import os
import argparse
import torch
from gan_module.data_utils import set_seed
from gan_module.models import UNetGenerator, PatchDiscriminator
from gan_module.train import train_gan
from gan_module.generate import generate_fault_images, generate_fault_dataset
from gan_module.visualize import visualize_comparisons, create_montage

def main():
    parser = argparse.ArgumentParser(description="变电站设备故障图像生成")
    subparsers = parser.add_subparsers(dest="command", help='操作命令')
    
    # 训练命令
    train_parser = subparsers.add_parser('train', help='训练GAN模型')
    train_parser.add_argument("--normal", type=str, required=True, 
                             help="正常设备图像目录")
    train_parser.add_argument("--fault", type=str, default=None, 
                             help="故障设备图像目录（如有）")
    train_parser.add_argument("--output", type=str, default="gan_models", 
                             help="输出目录（模型和样本）")
    train_parser.add_argument("--epochs", type=int, default=100, 
                             help="训练轮数")
    train_parser.add_argument("--batch-size", type=int, default=4, 
                             help="批次大小")
    train_parser.add_argument("--image-size", type=int, default=256, 
                             help="图像大小")
    train_parser.add_argument("--save-interval", type=int, default=10, 
                             help="保存检查点的间隔轮数")
    
    # 生成命令
    generate_parser = subparsers.add_parser('generate', help='生成故障图像')
    generate_parser.add_argument("--model", type=str, required=True, 
                                help="生成器模型路径")
    generate_parser.add_argument("--input", type=str, required=True, 
                                help="输入正常图像目录")
    generate_parser.add_argument("--output", type=str, default="generated_faults", 
                                help="输出故障图像目录")
    generate_parser.add_argument("--variations", type=int, default=3, 
                                help="每张图生成的变体数量")
    generate_parser.add_argument("--image-size", type=int, default=256, 
                                help="图像大小")
    generate_parser.add_argument("--severity", type=float, default=None, 
                                help="故障严重程度 (0-1.0)")
    
    # 批量生成命令
    batch_generate_parser = subparsers.add_parser('batch-generate', help='批量生成故障图像数据集')
    batch_generate_parser.add_argument("--model", type=str, required=True, 
                                     help="生成器模型路径")
    batch_generate_parser.add_argument("--normal-dir", type=str, required=True, 
                                     help="正常设备图像根目录（包含子目录）")
    batch_generate_parser.add_argument("--output", type=str, default="generated_dataset", 
                                     help="输出根目录")
    batch_generate_parser.add_argument("--variations", type=int, default=3, 
                                     help="每张图像的变体数量")
    batch_generate_parser.add_argument("--severity-levels", type=float, nargs="+", default=None, 
                                     help="严重程度级别列表，如 0.3 0.7 1.0")
    
    # 可视化命令
    vis_parser = subparsers.add_parser('visualize', help='生成对比可视化')
    vis_parser.add_argument("--normal", type=str, required=True, 
                          help="正常设备图像目录")
    vis_parser.add_argument("--gen-fault", type=str, required=True, 
                          help="生成的故障图像目录")
    vis_parser.add_argument("--real-fault", type=str, default=None, 
                          help="真实故障图像目录（可选）")
    vis_parser.add_argument("--output", type=str, default="comparisons.png", 
                          help="输出图像路径")
    vis_parser.add_argument("--samples", type=int, default=5, 
                          help="采样数量")
    
    # 蒙太奇命令
    montage_parser = subparsers.add_parser('montage', help='创建图像蒙太奇')
    montage_parser.add_argument("--input", type=str, required=True, 
                              help="输入图像目录")
    montage_parser.add_argument("--output", type=str, default="montage.png", 
                              help="输出图像路径")
    montage_parser.add_argument("--rows", type=int, default=4, 
                              help="网格行数")
    montage_parser.add_argument("--cols", type=int, default=4, 
                              help="网格列数")
    montage_parser.add_argument("--random", action="store_true", 
                              help="随机选择图像")
    
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed()
    
    # 执行对应命令
    if args.command == 'train':
        # 创建输出目录
        os.makedirs(args.output, exist_ok=True)
        
        # 训练GAN
        train_gan(
            args.normal, 
            args.fault, 
            args.output, 
            epochs=args.epochs, 
            batch_size=args.batch_size, 
            image_size=args.image_size,
            save_interval=args.save_interval
        )
        
    elif args.command == 'generate':
        # 创建输出目录
        os.makedirs(args.output, exist_ok=True)
        
        # 生成故障图像
        generate_fault_images(
            args.model,
            args.input,
            args.output,
            num_variations=args.variations,
            image_size=args.image_size,
            severity=args.severity
        )
        
    elif args.command == 'batch-generate':
        # 批量生成故障图像数据集
        generate_fault_dataset(
            args.model,
            args.normal_dir,
            args.output,
            variations_per_image=args.variations,
            severity_levels=args.severity_levels
        )
        
    elif args.command == 'visualize':
        # 创建可视化
        visualize_comparisons(
            args.normal,
            args.gen_fault,
            args.real_fault,
            args.output,
            args.samples
        )
        
    elif args.command == 'montage':
        # 创建图像蒙太奇
        create_montage(
            args.input,
            args.output,
            grid_size=(args.rows, args.cols),
            random_select=args.random
        )
        
    else:
        parser.print_help()
    
if __name__ == "__main__":
    main() 
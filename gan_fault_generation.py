#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# GAN故障图像生成主程序

import os
import sys
import argparse
import torch
from pathlib import Path

def check_module(module_name):
    """检查模块是否可用"""
    try:
        __import__(module_name)
        return True
    except ImportError:
        return False

def main():
    parser = argparse.ArgumentParser(description="GAN故障图像生成工具")
    subparsers = parser.add_subparsers(dest="command", help="子命令")
    
    # 训练命令
    train_parser = subparsers.add_parser("train", help="训练GAN模型")
    train_parser.add_argument("--normal", type=str, required=True, help="正常设备图像目录路径")
    train_parser.add_argument("--fault", type=str, default=None, help="真实故障图像目录路径（可选）")
    train_parser.add_argument("--output", type=str, required=True, help="模型输出目录")
    train_parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    train_parser.add_argument("--batch-size", type=int, default=4, help="批次大小")
    train_parser.add_argument("--lr", type=float, default=0.0002, help="学习率")
    train_parser.add_argument("--image-size", type=int, default=256, help="图像大小")
    train_parser.add_argument("--content-weight", type=float, default=10.0, help="内容损失权重")
    train_parser.add_argument("--adv-weight", type=float, default=1.0, help="对抗损失权重")
    
    # 生成命令
    generate_parser = subparsers.add_parser("generate", help="生成故障图像")
    generate_parser.add_argument("--model", type=str, required=True, help="生成器模型路径")
    generate_parser.add_argument("--input", type=str, required=True, help="输入正常图像或目录")
    generate_parser.add_argument("--output", type=str, required=True, help="输出故障图像目录")
    generate_parser.add_argument("--variations", type=int, default=3, help="每张图像生成的变体数量")
    generate_parser.add_argument("--severity", type=float, default=0.7, help="故障严重程度(0-1)")
    generate_parser.add_argument("--image-size", type=int, default=256, help="图像大小")
    generate_parser.add_argument("--add-noise", action="store_true", help="是否添加随机噪声")
    
    # 批量生成命令
    batch_generate_parser = subparsers.add_parser("batch-generate", help="批量生成故障图像数据集")
    batch_generate_parser.add_argument("--model", type=str, required=True, help="生成器模型路径")
    batch_generate_parser.add_argument("--input", type=str, required=True, help="输入正常图像目录")
    batch_generate_parser.add_argument("--output", type=str, required=True, help="输出故障图像目录")
    batch_generate_parser.add_argument("--variations", type=int, default=3, help="每张图像生成的变体数量")
    batch_generate_parser.add_argument("--min-severity", type=float, default=0.3, help="最小故障严重程度")
    batch_generate_parser.add_argument("--max-severity", type=float, default=0.9, help="最大故障严重程度")
    
    # 可视化比较命令
    visualize_parser = subparsers.add_parser("visualize", help="可视化对比正常图像和故障图像")
    visualize_parser.add_argument("--normal", type=str, required=True, help="正常设备图像目录")
    visualize_parser.add_argument("--gen-fault", type=str, required=True, help="生成的故障图像目录")
    visualize_parser.add_argument("--real-fault", type=str, default=None, help="真实故障图像目录(可选)")
    visualize_parser.add_argument("--output", type=str, required=True, help="输出图像路径")
    visualize_parser.add_argument("--samples", type=int, default=3, help="采样数量")
    visualize_parser.add_argument("--image-size", type=int, default=256, help="图像大小")
    
    # 蒙太奇可视化命令
    montage_parser = subparsers.add_parser("montage", help="创建图像蒙太奇")
    montage_parser.add_argument("--input", type=str, required=True, help="输入图像目录")
    montage_parser.add_argument("--output", type=str, required=True, help="输出图像路径")
    montage_parser.add_argument("--rows", type=int, default=4, help="行数")
    montage_parser.add_argument("--cols", type=int, default=4, help="列数")
    montage_parser.add_argument("--image-size", type=int, default=256, help="图像大小")
    montage_parser.add_argument("--random", action="store_true", help="是否随机选择图像")
    
    # 严重程度可视化命令
    severity_parser = subparsers.add_parser("visualize-severity", help="可视化不同严重程度的效果")
    severity_parser.add_argument("--model", type=str, required=True, help="生成器模型路径")
    severity_parser.add_argument("--input", type=str, required=True, help="输入正常图像路径")
    severity_parser.add_argument("--output", type=str, required=True, help="输出图像路径")
    severity_parser.add_argument("--levels", type=float, nargs="+", default=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0], help="严重程度级别")
    
    args = parser.parse_args()
    
    # 检查gan_module是否可用
    if not check_module("gan_module"):
        print("错误: gan_module 不可用，请确保已正确安装GAN模块")
        sys.exit(1)
    
    from gan_module import set_seed, train_gan, generate_fault_images, generate_fault_dataset
    from gan_module import visualize_comparisons, create_montage, visualize_severity_comparison
    from gan_module.models import UNetGenerator
    
    # 设置随机种子
    set_seed()
    
    # 执行对应命令
    if args.command == "train":
        print(f"使用 {args.normal} 中的正常图像训练GAN模型...")
        train_gan(
            args.normal,
            args.fault,
            args.output,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            lambda_l1=args.content_weight,
            image_size=args.image_size
        )
    
    elif args.command == "generate":
        print(f"正在生成故障图像...")
        generate_fault_images(
            args.model,
            args.input,
            args.output,
            num_variations=args.variations,
            image_size=args.image_size,
            severity=args.severity,
            add_noise=args.add_noise
        )
    
    elif args.command == "batch-generate":
        print(f"正在批量生成故障图像数据集...")
        severity_levels = None
        if args.min_severity is not None and args.max_severity is not None:
            # 创建3个不同严重程度
            severity_levels = [
                args.min_severity,
                (args.min_severity + args.max_severity) / 2,
                args.max_severity
            ]
        
        generate_fault_dataset(
            args.model,
            args.input,
            args.output,
            variations_per_image=args.variations,
            severity_levels=severity_levels
        )
    
    elif args.command == "visualize":
        print(f"创建可视化比较...")
        visualize_comparisons(
            args.normal,
            args.gen_fault,
            args.real_fault,
            args.output,
            args.samples,
            args.image_size
        )
    
    elif args.command == "montage":
        print(f"创建图像蒙太奇...")
        create_montage(
            args.input,
            args.output,
            (args.rows, args.cols),
            args.image_size,
            args.random
        )
    
    elif args.command == "visualize-severity":
        print(f"可视化不同严重程度效果...")
        # 加载生成器模型
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        generator = UNetGenerator().to(device)
        generator.load_state_dict(torch.load(args.model, map_location=device))
        generator.eval()
        
        visualize_severity_comparison(
            generator,
            args.input,
            args.output,
            args.levels
        )
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 
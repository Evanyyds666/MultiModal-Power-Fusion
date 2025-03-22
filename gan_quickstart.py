#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# GAN快速入门示例脚本

import os
import sys
import argparse
from pathlib import Path

def create_example_structure():
    """创建示例目录结构"""
    # 创建示例目录
    os.makedirs("examples/gan_data/normal/transformer", exist_ok=True)
    os.makedirs("examples/gan_data/normal/circuit_breaker", exist_ok=True)
    
    print("✓ 创建示例目录结构完成")
    print("  请在以下目录放置正常设备图像：")
    print("  - examples/gan_data/normal/transformer/")
    print("  - examples/gan_data/normal/circuit_breaker/")
    
    # 检查examples/images是否有图像，如果有就复制一些到示例目录
    if os.path.exists("examples/images"):
        import shutil
        import random
        
        image_files = [f for f in os.listdir("examples/images") 
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if image_files:
            # 随机选择一些图像
            sample_count = min(5, len(image_files))
            sampled_images = random.sample(image_files, sample_count)
            
            # 复制到示例目录
            for i, img in enumerate(sampled_images):
                if i % 2 == 0:
                    dest_dir = "examples/gan_data/normal/transformer"
                else:
                    dest_dir = "examples/gan_data/normal/circuit_breaker"
                
                shutil.copy(
                    os.path.join("examples/images", img),
                    os.path.join(dest_dir, img)
                )
            
            print(f"✓ 已复制 {sample_count} 张示例图像到目录")
        else:
            print("! examples/images 中没有找到图像")
    else:
        print("! examples/images 目录不存在")
    
    print("\n现在，您可以：")
    print("1. 手动添加更多图像到示例目录")
    print("2. 或者直接使用现有的示例图像继续教程")

def run_training_example():
    """运行训练示例"""
    # 检查是否有足够的图像
    transformer_dir = "examples/gan_data/normal/transformer"
    circuit_dir = "examples/gan_data/normal/circuit_breaker"
    
    transformer_imgs = [f for f in os.listdir(transformer_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    circuit_imgs = [f for f in os.listdir(circuit_dir) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if len(transformer_imgs) < 3 or len(circuit_imgs) < 3:
        print("! 警告：示例目录中的图像太少。模型训练可能效果不佳。")
        print(f"  - {transformer_dir}: {len(transformer_imgs)} 张图像")
        print(f"  - {circuit_dir}: {len(circuit_imgs)} 张图像")
        print("  建议每个类别至少使用10-20张图像获得更好效果。")
        
        response = input("是否继续训练示例？(y/n): ")
        if response.lower() != 'y':
            print("退出示例训练。请添加更多图像后再试。")
            return
    
    # 创建输出目录
    os.makedirs("examples/gan_output", exist_ok=True)
    
    # 训练变压器模型
    print("\n[1/2] 训练变压器故障生成模型...")
    transformer_cmd = f"python gan_fault_generation.py train --normal {transformer_dir} --output examples/gan_output/transformer_model --epochs 30 --batch-size 2"
    print(f"执行命令: {transformer_cmd}")
    os.system(transformer_cmd)
    
    # 训练断路器模型
    print("\n[2/2] 训练断路器故障生成模型...")
    circuit_cmd = f"python gan_fault_generation.py train --normal {circuit_dir} --output examples/gan_output/circuit_model --epochs 30 --batch-size 2"
    print(f"执行命令: {circuit_cmd}")
    os.system(circuit_cmd)
    
    print("\n✓ 训练示例完成！")

def run_generation_example():
    """运行生成示例"""
    # 检查模型是否存在
    transformer_model = "examples/gan_output/transformer_model/generator_final.pth"
    circuit_model = "examples/gan_output/circuit_model/generator_final.pth"
    
    if not os.path.exists(transformer_model) or not os.path.exists(circuit_model):
        print("! 错误：模型文件不存在。请先运行训练示例。")
        return
    
    # 创建输出目录
    os.makedirs("examples/gan_output/generated_faults/transformer", exist_ok=True)
    os.makedirs("examples/gan_output/generated_faults/circuit_breaker", exist_ok=True)
    
    # 生成变压器故障图像
    print("\n[1/2] 生成变压器故障图像...")
    transformer_cmd = f"python gan_fault_generation.py generate --model {transformer_model} --input examples/gan_data/normal/transformer --output examples/gan_output/generated_faults/transformer --variations 3"
    print(f"执行命令: {transformer_cmd}")
    os.system(transformer_cmd)
    
    # 生成断路器故障图像
    print("\n[2/2] 生成断路器故障图像...")
    circuit_cmd = f"python gan_fault_generation.py generate --model {circuit_model} --input examples/gan_data/normal/circuit_breaker --output examples/gan_output/generated_faults/circuit_breaker --variations 3"
    print(f"执行命令: {circuit_cmd}")
    os.system(circuit_cmd)
    
    # 生成可视化
    print("\n创建可视化比较...")
    vis_cmd = f"python gan_fault_generation.py visualize --normal examples/gan_data/normal/transformer --gen-fault examples/gan_output/generated_faults/transformer --output examples/gan_output/transformer_comparison.png --samples 3"
    print(f"执行命令: {vis_cmd}")
    os.system(vis_cmd)
    
    vis_cmd = f"python gan_fault_generation.py visualize --normal examples/gan_data/normal/circuit_breaker --gen-fault examples/gan_output/generated_faults/circuit_breaker --output examples/gan_output/circuit_comparison.png --samples 3"
    print(f"执行命令: {vis_cmd}")
    os.system(vis_cmd)
    
    # 创建蒙太奇
    print("\n创建故障图像蒙太奇...")
    montage_cmd = f"python gan_fault_generation.py montage --input examples/gan_output/generated_faults/transformer --output examples/gan_output/transformer_montage.png --rows 2 --cols 3 --random"
    print(f"执行命令: {montage_cmd}")
    os.system(montage_cmd)
    
    print("\n✓ 生成示例完成！")
    print("  您可以在以下位置查看结果：")
    print("  - 生成的故障图像: examples/gan_output/generated_faults/")
    print("  - 比较可视化: examples/gan_output/transformer_comparison.png")
    print("  - 图像蒙太奇: examples/gan_output/transformer_montage.png")

def run_pipeline_example():
    """运行整合到多模态流水线的示例"""
    # 检查生成的故障图像是否存在
    transformer_fault_dir = "examples/gan_output/generated_faults/transformer"
    circuit_fault_dir = "examples/gan_output/generated_faults/circuit_breaker"
    
    if not os.path.exists(transformer_fault_dir) or not os.path.exists(circuit_fault_dir):
        print("! 错误：生成的故障图像不存在。请先运行生成示例。")
        return
    
    # 创建输出目录
    os.makedirs("examples/multimodal_results", exist_ok=True)
    
    # 检查pipeline脚本是否存在
    if not os.path.exists("run_multimodal_pipeline.py"):
        print("! 错误：run_multimodal_pipeline.py 不存在。")
        print("  请先确保主项目的多模态流水线脚本已创建。")
        return
    
    # 运行多模态特征提取和融合，包括生成的故障图像
    print("\n运行多模态特征提取和融合流水线...")
    pipeline_cmd = f"python run_multimodal_pipeline.py --images examples/gan_data/normal --output examples/multimodal_results"
    print(f"执行命令: {pipeline_cmd}")
    os.system(pipeline_cmd)
    
    print("\n✓ 多模态流水线示例完成！")
    print("  您可以在以下位置查看结果：")
    print("  - 多模态特征: examples/multimodal_results/")

def main():
    parser = argparse.ArgumentParser(description="GAN故障图像生成快速入门")
    parser.add_argument("step", choices=["setup", "train", "generate", "pipeline", "all"],
                       help="要运行的示例步骤")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print(" GAN故障图像生成快速入门")
    print("=" * 60)
    
    # 检查gan_module是否存在
    if not os.path.exists("gan_module") or not os.path.exists("gan_fault_generation.py"):
        print("! 错误：gan_module 目录或 gan_fault_generation.py 不存在。")
        print("  请确保已正确安装GAN模块。")
        sys.exit(1)
    
    if args.step == "setup" or args.step == "all":
        print("\n[步骤1] 创建示例数据结构")
        print("-" * 60)
        create_example_structure()
    
    if args.step == "train" or args.step == "all":
        print("\n[步骤2] 训练GAN模型")
        print("-" * 60)
        run_training_example()
    
    if args.step == "generate" or args.step == "all":
        print("\n[步骤3] 生成故障图像")
        print("-" * 60)
        run_generation_example()
    
    if args.step == "pipeline" or args.step == "all":
        print("\n[步骤4] 整合到多模态流水线")
        print("-" * 60)
        run_pipeline_example()
    
    if args.step == "all":
        print("\n恭喜！您已完成GAN故障图像生成的所有步骤。")
        print("现在您可以将这些技术应用到您自己的数据集中。")
    
    print("\n提示：")
    print("- 要获得更好的效果，请为每个设备类别收集20-50张图像")
    print("- 真实故障图像可以提高模型质量，但不是必需的")
    print("- 尝试不同的严重程度参数生成多样化的故障图像")

if __name__ == "__main__":
    main() 
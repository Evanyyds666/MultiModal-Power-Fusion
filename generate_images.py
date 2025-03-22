#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

def generate_power_equipment_images(output_dir, num_images=5, categories=None, prompts=None):
    """生成电力设备相关的图像
    
    Args:
        output_dir: 输出图像的目录
        num_images: 每个类别生成的图像数量
        categories: 设备类别名称列表
        prompts: 自定义提示词列表
    """
    # 默认的电力设备类别
    default_categories = [
        "transformer", "circuit_breaker", "capacitor", "insulator", "generator",
        "power_line", "switch_gear", "reactor", "voltage_regulator", "busbar"
    ]
    categories = categories or default_categories
    
    # 中英文提示词模板
    prompt_templates = {
        "zh": [
            "高清照片，{equipment}，变电站设备，工业摄影，精细细节，真实感",
            "专业工业摄影，{equipment}，变电站内部，工程设计图，高清晰度，真实照片",
            "详细的技术摄影，{equipment}，电气设备，白天，变电站场景，真实感",
            "工业设备{equipment}，电网基础设施，特写镜头，高清图像，阳光照射",
            "专业的电力系统摄影，{equipment}，工业环境，详细纹理，真实场景"
        ],
        "en": [
            "high resolution photo of {equipment}, electrical substation equipment, industrial photography, fine details, realistic",
            "professional industrial photography of {equipment}, inside power substation, engineering design, high definition, real photo",
            "detailed technical photography of {equipment}, electrical equipment, daylight, substation setting, realistic",
            "industrial equipment {equipment}, power grid infrastructure, close-up shot, high-definition image, sunlight",
            "professional power system photography of {equipment}, industrial environment, detailed textures, real scene"
        ]
    }
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载模型
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda")
    
    if prompts:
        # 使用自定义提示词
        for i, prompt in enumerate(prompts):
            for j in range(num_images):
                image = pipe(prompt).images[0]
                
                # 保存图像
                output_path = os.path.join(output_dir, f"custom_prompt_{i}_image_{j}.png")
                image.save(output_path)
                print(f"生成图像 {output_path}")
    else:
        # 使用默认模板和类别
        for category in categories:
            # 为每个类别创建子目录
            category_dir = os.path.join(output_dir, category)
            os.makedirs(category_dir, exist_ok=True)
            
            for j in range(num_images):
                # 随机选择语言和提示词模板
                language = "en" if j % 2 == 0 else "zh"
                template_idx = j % len(prompt_templates[language])
                
                # 构建提示词
                template = prompt_templates[language][template_idx]
                prompt = template.format(equipment=category.replace("_", " "))
                
                print(f"生成 {category} 图像 {j+1}/{num_images}，使用提示词: {prompt}")
                
                try:
                    # 生成图像
                    image = pipe(prompt).images[0]
                    
                    # 保存图像
                    output_path = os.path.join(category_dir, f"{category}_{j}.png")
                    image.save(output_path)
                    print(f"已保存到 {output_path}")
                except Exception as e:
                    print(f"生成图像失败: {e}")
    
    print(f"图像生成完成，共生成 {len(categories) * num_images} 张图像")

def main():
    parser = argparse.ArgumentParser(description="生成电力设备相关的图像")
    parser.add_argument("--output", type=str, default="generated_images", 
                       help="保存生成图像的目录")
    parser.add_argument("--num", type=int, default=5, 
                       help="每个类别生成的图像数量")
    parser.add_argument("--categories", type=str, nargs="+", 
                       help="要生成的设备类别列表（如果不指定，使用默认类别）")
    parser.add_argument("--prompts", type=str, nargs="+", 
                       help="自定义提示词列表（如果指定，将忽略categories参数）")
    
    args = parser.parse_args()
    generate_power_equipment_images(args.output, args.num, args.categories, args.prompts)

if __name__ == "__main__":
    main() 
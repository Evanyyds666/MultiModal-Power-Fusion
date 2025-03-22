#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 文本特征提取模块

import os
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path

def extract_text_features(text_path, output_dir, model_name='bert-base-chinese'):
    """提取文本特征
    
    Args:
        text_path: 文本文件或目录路径
        output_dir: 输出特征的目录
        model_name: 预训练模型名称
    """
    try:
        from transformers import AutoTokenizer, AutoModel
    except ImportError:
        print("错误: 需要安装transformers库才能提取文本特征")
        return
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载预训练模型和分词器
    print(f"加载模型: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    # 移动模型到适当的设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    
    # 处理输入路径
    if os.path.isfile(text_path):
        text_files = [text_path]
    else:
        text_files = [os.path.join(text_path, f) for f in os.listdir(text_path) 
                     if f.lower().endswith(('txt', 'md', 'json', 'csv'))]
    
    if not text_files:
        print(f"警告: 在 {text_path} 中没有找到文本文件")
        return
    
    # 处理每个文本文件
    for text_file in tqdm(text_files, desc="提取文本特征"):
        try:
            # 读取文本文件
            with open(text_file, 'r', encoding='utf-8') as f:
                text = f.read().strip()
            
            # 处理过长的文本
            max_length = tokenizer.model_max_length
            if len(text) > max_length * 4:  # 如果文本非常长，仅使用开头和结尾部分
                first_part = text[:max_length]
                last_part = text[-max_length:]
                text = first_part + last_part
            
            # 对文本进行分词
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # 提取特征
            with torch.no_grad():
                outputs = model(**inputs)
                # 使用[CLS]标记对应的隐藏状态作为文本特征
                feature_vector = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
            
            # 保存特征
            text_name = os.path.basename(text_file)
            feature_name = os.path.splitext(text_name)[0] + ".npy"
            feature_path = os.path.join(output_dir, feature_name)
            
            np.save(feature_path, feature_vector)
            
        except Exception as e:
            print(f"处理文本 {text_file} 时出错: {e}")
    
    print(f"文本特征提取完成，已保存到 {output_dir}")

def extract_keyword_features(text_path, output_dir, keywords=None, language='chinese'):
    """提取关键词特征
    
    Args:
        text_path: 文本文件或目录路径
        output_dir: 输出特征的目录
        keywords: 关键词列表，如果为None则自动提取
        language: 文本语言，'chinese'或'english'
    """
    try:
        import jieba
        import jieba.analyse
    except ImportError:
        print("错误: 需要安装jieba库才能提取关键词特征")
        return
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 如果没有提供关键词，使用预设的电力设备相关词汇
    if keywords is None:
        if language == 'chinese':
            keywords = [
                '变压器', '断路器', '绝缘子', '避雷器', '电容器', '电抗器', '互感器',
                '母线', '开关', '线路', '电缆', '接地装置', '隔离开关', '配电箱',
                '故障', '过载', '短路', '漏电', '过热', '老化', '腐蚀', '裂纹',
                '松动', '振动', '噪声', '放电', '绝缘破坏', '温度升高', '油位异常',
                '泄漏', '变形', '锈蚀', '污秽', '损坏', '异响', '冒烟', '异味'
            ]
        else:  # english
            keywords = [
                'transformer', 'circuit breaker', 'insulator', 'arrester', 'capacitor', 'reactor', 'current transformer',
                'bus', 'switch', 'line', 'cable', 'grounding', 'disconnect switch', 'distribution box',
                'fault', 'overload', 'short circuit', 'leakage', 'overheat', 'aging', 'corrosion', 'crack',
                'loose', 'vibration', 'noise', 'discharge', 'insulation damage', 'temperature rise', 'oil level',
                'leak', 'deformation', 'rust', 'contamination', 'damage', 'sound', 'smoke', 'odor'
            ]
    
    # 处理输入路径
    if os.path.isfile(text_path):
        text_files = [text_path]
    else:
        text_files = [os.path.join(text_path, f) for f in os.listdir(text_path) 
                     if f.lower().endswith(('txt', 'md', 'json', 'csv'))]
    
    if not text_files:
        print(f"警告: 在 {text_path} 中没有找到文本文件")
        return
    
    # 处理每个文本文件
    for text_file in tqdm(text_files, desc="提取关键词特征"):
        try:
            # 读取文本文件
            with open(text_file, 'r', encoding='utf-8') as f:
                text = f.read().strip()
            
            # 分词
            if language == 'chinese':
                words = list(jieba.cut(text))
            else:  # english
                words = text.lower().split()
            
            # 构建关键词特征向量
            feature_vector = np.zeros(len(keywords))
            
            for i, keyword in enumerate(keywords):
                # 计算关键词在文本中的出现次数
                if language == 'chinese':
                    count = sum(1 for word in words if keyword in word)
                else:  # english
                    count = sum(1 for word in words if keyword.lower() in word.lower())
                
                feature_vector[i] = count
            
            # 归一化特征向量
            norm = np.linalg.norm(feature_vector)
            if norm > 0:
                feature_vector = feature_vector / norm
            
            # 保存特征
            text_name = os.path.basename(text_file)
            feature_name = os.path.splitext(text_name)[0] + "_keywords.npy"
            feature_path = os.path.join(output_dir, feature_name)
            
            np.save(feature_path, feature_vector)
            
            # 如果没有提供关键词，也自动提取文本中的关键词
            if keywords is None:
                if language == 'chinese':
                    extracted_keywords = jieba.analyse.extract_tags(text, topK=20)
                else:  # english
                    extracted_keywords = text.lower().split()[:20]  # 简化处理，实际应使用更好的方法
                
                keyword_str = ', '.join(extracted_keywords)
                with open(os.path.join(output_dir, f"{os.path.splitext(text_name)[0]}_extracted_keywords.txt"), 'w', encoding='utf-8') as f:
                    f.write(keyword_str)
            
        except Exception as e:
            print(f"处理文本 {text_file} 时出错: {e}")
    
    print(f"关键词特征提取完成，已保存到 {output_dir}")

def main():
    """命令行入口"""
    import argparse
    
    parser = argparse.ArgumentParser(description="提取文本特征")
    parser.add_argument("--input", type=str, required=True, 
                        help="文本文件或目录路径")
    parser.add_argument("--output", type=str, default="text_features", 
                        help="输出特征目录")
    parser.add_argument("--method", type=str, choices=["bert", "keywords", "both"], default="bert",
                        help="特征提取方法")
    parser.add_argument("--model", type=str, default="bert-base-chinese",
                        help="BERT模型名称或路径")
    parser.add_argument("--language", type=str, choices=["chinese", "english"], default="chinese",
                        help="文本语言")
    
    args = parser.parse_args()
    
    if args.method in ["bert", "both"]:
        print(f"使用 {args.model} 提取文本特征...")
        extract_text_features(args.input, args.output, args.model)
    
    if args.method in ["keywords", "both"]:
        print("提取关键词特征...")
        extract_keyword_features(args.input, args.output, language=args.language)

if __name__ == "__main__":
    main()
